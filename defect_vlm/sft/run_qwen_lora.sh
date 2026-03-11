#!/bin/bash

# ==============================================================================
# 1. 环境变量配置 (Environment Variables)
# ==============================================================================
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export IMAGE_MAX_TOKEN_NUM=1024
export NPROC_PER_NODE=3
export CUDA_VISIBLE_DEVICES=0,1,2

# ==============================================================================
# 2. 数据集路径统一定义 (Dataset Definitions)
# ==============================================================================
DATA_BASE="/data/ZS/defect_dataset/7_swift_dataset"

# 使用 Bash 数组优雅地管理多文件输入，增删极其方便
TRAIN_FILES=(
    "${DATA_BASE}/train/stripe_phase012_gt_negative.jsonl"
    "${DATA_BASE}/train/stripe_phase012_gt_positive.jsonl"
    "${DATA_BASE}/train/stripe_phase012_gt_rectification.jsonl"
    "${DATA_BASE}/train/stripe_phase123_gt_negative.jsonl"
    "${DATA_BASE}/train/stripe_phase123_gt_positive.jsonl"
    "${DATA_BASE}/train/stripe_phase123_gt_rectification.jsonl"
)

VAL_FILES=(
    "${DATA_BASE}/val/stripe_phase012_gt_negative.jsonl"
    "${DATA_BASE}/val/stripe_phase012_gt_positive.jsonl"
    "${DATA_BASE}/val/stripe_phase012_gt_rectification.jsonl"
    "${DATA_BASE}/val/stripe_phase123_gt_negative.jsonl"
    "${DATA_BASE}/val/stripe_phase123_gt_positive.jsonl"
    "${DATA_BASE}/val/stripe_phase123_gt_rectification.jsonl"
)

# ==============================================================================
# 3. 启动 Swift SFT 微调
# ==============================================================================
echo "🚀 开始启动 Qwen3-VL-4B LoRA 微调..."

swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset "${TRAIN_FILES[@]}" \
    --val_dataset "${VAL_FILES[@]}" \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --load_from_cache_file true \
    \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 64 \
    --lora_alpha 128 \
    --freeze_vit false \
    --freeze_aligner false \
    \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_free true \
    --packing true \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --output_dir /data/ZS/defect-vlm/output/weights