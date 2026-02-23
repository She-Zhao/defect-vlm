#!/bin/bash
# 调用不同厂商的api

# --- 配置区域 ---
# 建议使用绝对路径，避免找不到文件
PROJECT_ROOT="/data/ZS/defect-vlm"
SCRIPT_PATH="$PROJECT_ROOT/defect_vlm/pe/call_api.py"
INPUT_FILE="/data/ZS/defect_dataset/4_api_request/teacher/test/012_pos400_neg300_rect300.jsonl"
OUTPUT_DIR="/data/ZS/defect_dataset/5_api_response/teacher/test"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 激活环境 (根据你的实际情况修改，如果是 conda)
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate defect-vlm

echo "========================================"
echo "🚀 开始批量打标任务"
echo "📅 开始时间: $(date)"
echo "📂 输入文件: $INPUT_FILE"
echo "========================================"

# --- 任务列表 ---


echo "👉 Running gpt-5.1..."
python "$SCRIPT_PATH" \
    --provider openai \
    --model gpt-5.1 \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/gpt-5.1.jsonl" \
    --concurrency 2

echo "👉 Running gpt-5.2..."
python "$SCRIPT_PATH" \
    --provider openai \
    --model gpt-5.2 \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/gpt-5.2.jsonl" \
    --concurrency 2

echo "========================================"
echo "✅ 所有任务已完成"
echo "📅 结束时间: $(date)"
echo "========================================"