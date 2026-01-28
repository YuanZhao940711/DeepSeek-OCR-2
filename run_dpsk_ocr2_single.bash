CUDA_VISIBLE_DEVICES=1 \
python dpsk_ocr2_runner.py \
    --deepseek_repo /data/diaoliang/zhaoyuan/models/DeepSeek-OCR/DeepSeek-OCR-master \
    --model_path /data/diaoliang/vvw/models/DeepSeek_OCR \
    --prompt "<image>\n<|grounding|>Convert the document to markdown." \
    --crop_mode 1 \
    single \
    --image /data/diaoliang/vvw/projects/ms-swift-3.9.0/scripts/a_li_yun_medical/a_li_yun_medical_dataset/test_imgs/473e34d1ac_8.jpg \
    --save_txt /data/diaoliang/zhaoyuan/tmp/45f04ade2e_10.ocr.txt
    # --image /data/diaoliang/vvw/projects/ms-swift-3.9.0/scripts/a_li_yun_medical/a_li_yun_medical_dataset/test_imgs/45f04ade2e_10.jpg \
    