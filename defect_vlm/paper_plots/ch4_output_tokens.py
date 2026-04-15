"""
统计输出token的平均长度
"""
import json
from transformers import AutoTokenizer

# 1. 加载你的 4B 模型的 Tokenizer (替换为你的模型路径)
tokenizer = AutoTokenizer.from_pretrained("/data/ZS/model/models/Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)

# 2. 读取你的标注数据集 (替换为你的数据文件路径)
data_path = "/data/ZS/defect_dataset/7_swift_dataset/course/stage3_task_alignment_26k_defect_only.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

total_tokens = 0
num_samples = len(dataset)

# 3. 遍历数据，只计算模型输出部分（假定你的输出字段叫 'output' 或 'assistant'）
for item in dataset:
    # 假设你的目标输出文本在 item["output"] 中
    target_text = item["messages"][1]['content']
    # 计算 token 数量
    tokens = tokenizer.encode(target_text)
    total_tokens += len(tokens)

# 4. 计算并打印平均值
avg_tokens = total_tokens / num_samples
print(f"该阶段数据集的平均输出 Token 长度为: {avg_tokens:.2f}")
