export CUDA_VISIBLE_DEVICES=1
swift export \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --adapters /data/ZS/defect-vlm/output/weights/v3-20260312-165603_qwen3_4b_LM_PRO_VIT/checkpoint-3261-best \
    --output_dir /data/ZS/defect-vlm/output/merged_model/v3_qwen3_4b_LM_PRO_VIT \
    --merge_lora true