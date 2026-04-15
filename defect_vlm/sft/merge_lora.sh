export CUDA_VISIBLE_DEVICES=0
swift export \
    --model /data/ZS/defect-vlm/output/merged_model/v8-stage2-LM \
    --adapters /data/ZS/defect-vlm/output/weights/v11-stage3-LM-PRO-VIT-26k/checkpoint-816-best \
    --output_dir /data/ZS/defect-vlm/output/merged_model/v11-stage3-LM-PRO-VIT-26k \
    --merge_lora true