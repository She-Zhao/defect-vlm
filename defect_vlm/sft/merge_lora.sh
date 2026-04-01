export CUDA_VISIBLE_DEVICES=2,3
swift export \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --adapters /data/ZS/defect-vlm/output/weights/v4-20260329-234354_qwen3_4b_LM_defect_only/checkpoint-1200 \
    --output_dir /data/ZS/defect-vlm/output/merged_model/v4_qwen3_4b_LM_defect_only \
    --merge_lora true