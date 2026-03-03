#!/bin/bash
# 调用不同厂商的api

# --- 配置区域 ---
# 建议使用绝对路径，避免找不到文件
PROJECT_ROOT="/data/ZS/defect-vlm"
SCRIPT_PATH="$PROJECT_ROOT/defect_vlm/pe/call_api.py"
# INPUT_FILE="/data/ZS/defect_dataset/4_api_request/teacher/test/012_pos400_neg300_rect300.jsonl"
# OUTPUT_DIR="/data/ZS/defect_dataset/5_api_response/teacher/test"

INPUT_DIR="/data/ZS/defect_dataset/4_api_request/teacher/retry"
OUTPUT_DIR="/data/ZS/defect_dataset/5_api_response/teacher/retry"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "🚀 开始批量打标任务"
echo "📅 开始时间: $(date)"
echo "📂 输入文件: $INPUT_FILE"
echo "========================================"

# --- 任务列表 ---
echo "👉 Running stripe_phase012_gt_negative..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase012_gt_negative.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase012_gt_negative.jsonl" \
    --concurrency 1

echo "👉 Running stripe_phase012_gt_positive..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase012_gt_positive.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase012_gt_positive.jsonl" \
    --concurrency 1

echo "👉 Running stripe_phase012_gt_rectification..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase012_gt_rectification.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase012_gt_rectification.jsonl" \
    --concurrency 1

echo "👉 Running stripe_phase123_gt_negative..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase123_gt_negative.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase123_gt_negative.jsonl" \
    --concurrency 1

echo "👉 Running stripe_phase123_gt_positive..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase123_gt_positive.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase123_gt_positive.jsonl" \
    --concurrency 1

echo "👉 Running stripe_phase123_gt_rectification..."
python "$SCRIPT_PATH" \
    --provider qwen \
    --model  qwen3-vl-235b-a22b-instruct\
    --input_file "$INPUT_DIR/stripe_phase123_gt_rectification.jsonl" \
    --output_file "$OUTPUT_DIR/stripe_phase123_gt_rectification.jsonl" \
    --concurrency 1

# echo "👉 Running qwen3.5-plus..."
# python "$SCRIPT_PATH" \
#     --provider qwen \
#     --model qwen3.5-plus \
#     --input_file "$INPUT_FILE" \
#     --output_file "$OUTPUT_DIR/qwen3.5-plus.jsonl" \
#     --concurrency 1

echo "========================================"
echo "✅ 所有任务已完成"
echo "📅 结束时间: $(date)"
echo "========================================"