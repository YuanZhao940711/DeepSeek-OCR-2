from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import inspect

import sys
from pathlib import Path

def add_repo_to_syspath(repo_path: str) -> None:
    """
    Robustly locate deepseek_ocr.py under repo_path and add its parent dir to sys.path.
    This avoids guessing folder names like DeepSeek-OCR-vllm.
    """
    p = Path(repo_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"--deepseek_repo not found: {p}")

    # 1) direct candidates: repo root itself
    candidates = [p]

    # 2) try to find deepseek_ocr.py anywhere under repo
    found = list(p.rglob("deepseek_ocr2.py"))
    for f in found:
        candidates.append(f.parent)

    # 3) also add any directory named 'process' parent (common in this repo)
    found_process = list(p.rglob("process"))
    for d in found_process:
        if d.is_dir():
            candidates.append(d.parent)

    # add unique candidates
    for c in candidates:
        c = c.resolve()
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


# --------- 清洗：去掉 ref/det 坐标块，保留正文 ---------
REF_DET_PATTERN = re.compile(
    r'(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)', re.DOTALL
)

def deepseek_to_ocr_text(deepseek_raw: str) -> str:
    s = re.sub(REF_DET_PATTERN, "", deepseek_raw or "")
    s = s.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

# --------- 配置 ---------
@dataclass
class DeepSeekOCRConfig:
    model_path: str                          # e.g. /data/diaoliang/vvw/models/DeepSeek_OCR
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    crop_mode: bool = True                   # CROP_MODE
    max_model_len: int = 8192
    block_size: int = 256
    gpu_memory_utilization: float = 0.75
    tensor_parallel_size: int = 1
    temperature: float = 0.0
    max_tokens: int = 8192


# --------- DeepSeek-OCR (vLLM) 可复用封装：engine 只创建一次 ---------
class DeepSeekOCRVLLM:
    def __init__(self, cfg: DeepSeekOCRConfig):
        self.cfg = cfg
        self.engine = None

        # 不在代码里写死 CUDA_VISIBLE_DEVICES；请在 shell 里控制
        os.environ.setdefault("VLLM_USE_V1", "0")

        # deepseek-ocr2 repo 里的依赖
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.model_executor.models.registry import ModelRegistry

        from deepseek_ocr2 import DeepseekOCR2ForCausalLM
        print("[DeepseekOCR2ForCausalLM loaded from]", inspect.getfile(DeepseekOCR2ForCausalLM))
        from process.image_process import DeepseekOCR2Processor
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

        # 注册模型架构
        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

        self.AsyncLLMEngine = AsyncLLMEngine
        self.AsyncEngineArgs = AsyncEngineArgs
        self.SamplingParams = SamplingParams
        self.DeepseekOCRProcessor = DeepseekOCR2Processor
        self.NoRepeatNGramLogitsProcessor = NoRepeatNGramLogitsProcessor

    async def ensure_engine(self):
        if self.engine is None:
            engine_args = self.AsyncEngineArgs(
                model=self.cfg.model_path,
                hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
                # torch_dtype=torch.bfloat16,
                dtype="bfloat16",
                block_size=self.cfg.block_size,
                max_model_len=self.cfg.max_model_len,
                enforce_eager=False,
                trust_remote_code=True,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            )
            self.engine = self.AsyncLLMEngine.from_engine_args(engine_args)

    async def ocr_raw(self, image_path: str) -> str:
        """返回 DeepSeek-OCR 的原始输出（可能包含 ref/det 坐标块）"""
        await self.ensure_engine()

        from PIL import Image, ImageOps
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB") 

        processor = self.DeepseekOCRProcessor()
        image_features = None
        if "<image>" in self.cfg.prompt: 
            image_features = processor.tokenize_with_images(
                images=[img], bos=True, eos=True, cropping=self.cfg.crop_mode
            )

        logits_processors = [
            self.NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            )
        ]

        # sampling_params 是什么
        sampling_params = self.SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        request: Dict[str, Any] = {"prompt": self.cfg.prompt}
        if image_features is not None:
            request["multi_modal_data"] = {"image": image_features}

        request_id = f"dsocr-{time.time_ns()}"
        final_text = ""
        printed_length = 0

        async for request_out in self.engine.generate(request, sampling_params, request_id):
            if request_out.outputs: 
                full_text = request_out.outputs[0].text
                new_text = full_text[printed_length:]
                print(new_text, end='', flush=True)
                printed_length = len(full_text)
                final_text = full_text
        return final_text

    async def ocr_text(self, image_path: str) -> str:
        """返回清洗后的 OCR 文本"""
        raw = await self.ocr_raw(image_path)
        return raw
        # return deepseek_to_ocr_text(raw)


# --------- CLI / Runner ---------
async def run_single(ds: DeepSeekOCRVLLM, image_path: str, save_path: Optional[str]):
    text = await ds.ocr_text(image_path)
    print(text)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[SAVED] OCR text -> {save_path}")

async def run_jsonl(ds: DeepSeekOCRVLLM, input_jsonl: str, output_jsonl: str):
    """
    输入：你的 test.jsonl（每行包含 images: [path]）
    输出：ocr_dump.jsonl（每行包含 image_path, ocr_text）
    """
    ensure_dir(os.path.dirname(output_jsonl))
    n = 0
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            n += 1
            sample = json.loads(line)
            # 兼容 images 是 list[str] 或 list[dict]
            img0 = sample["images"][0]
            image_path = img0["path"] if isinstance(img0, dict) else img0

            ocr_text = await ds.ocr_text(image_path)
            row = {"image_path": image_path, "ocr_text": ocr_text}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"[PROGRESS] {n} processed...")
    print(f"[DONE] wrote -> {output_jsonl} (lines={n})")

def build_args():
    parser = argparse.ArgumentParser("DeepSeek-OCR2 vLLM runner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown.")
    parser.add_argument("--crop_mode", type=int, default=1, help="1=True, 0=False")
    parser.add_argument("--gpu_mem_util", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--deepseek_repo", type=str, default="/data/diaoliang/zhaoyuan/models/DeepSeek-OCR", help="path to DeepSeek-OCR repo (source code)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("single", help="OCR one image")
    s1.add_argument("--image", type=str, required=True)
    s1.add_argument("--save_txt", type=str, default="", help="optional save path")

    s2 = sub.add_parser("jsonl", help="OCR all images in a jsonl")
    s2.add_argument("--input_jsonl", type=str, required=True)
    s2.add_argument("--output_jsonl", type=str, required=True)

    return parser.parse_args()


async def main_async():
    args = build_args()
    if args.deepseek_repo:
        add_repo_to_syspath(args.deepseek_repo)
    
    print("[sys.path head]", sys.path[:5])

    cfg = DeepSeekOCRConfig(
        model_path=args.model_path,
        prompt=args.prompt,
        crop_mode=bool(args.crop_mode),
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    ds = DeepSeekOCRVLLM(cfg)

    if args.cmd == "single":
        await run_single(ds, args.image, args.save_txt or None)
    elif args.cmd == "jsonl":
        await run_jsonl(ds, args.input_jsonl, args.output_jsonl)
    else:
        raise ValueError(args.cmd)

def main():
    # IMPORTANT: 控制 GPU 请在 shell 里做，例如：
    # CUDA_VISIBLE_DEVICES=1 python deepseek_ocr_runner.py ... single ...
    asyncio.run(main_async())

if __name__ == "__main__":
    main()