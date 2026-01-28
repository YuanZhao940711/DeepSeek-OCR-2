CUDA_VISIBLE_DEVICES=1 \
python deepseek_ocr2_runner.py \
    --deepseek_repo /data/diaoliang/zhaoyuan/models/DeepSeek-OCR/DeepSeek-OCR-master \
    --model_path /data/diaoliang/vvw/models/DeepSeek_OCR \
    --prompt "<image>\n<|grounding|>Convert the document to markdown." \
    --crop_mode 1 \
    jsonl \
    --input_jsonl /data/diaoliang/vvw/projects/ms-swift-3.9.0/scripts/a_li_yun_medical/a_li_yun_medical_test.jsonl \
    --output_jsonl result/deepseek_ocr_dump_test.jsonl
