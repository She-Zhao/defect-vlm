"""
对6_sft_dataset里面的数据进行清洗：
    统计所有的数据数量
    去除重复的id
    删除fail_reason字段
"""
import json
from pathlib import Path

def clean_jsonl_data(input_filepath, output_filepath):
    """
    清洗 JSONL 文件：去除重复 id，删除 meta_info 中的 fail_reason 字段。
    """
    seen_ids = set()  # 用于存储已经处理过的 id，用于去重
    processed_count = 0
    duplicate_count = 0
    fail_reason_count = 0
    # 使用 utf-8 编码打开输入和输出文件
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 跳过空行
            if not line.strip():
                continue
            
            # 解析当前行的 JSON 数据
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"解析错误，跳过无效的 JSON 行: {line[:50]}...")
                continue
            
            # 获取数据的 id
            item_id = data.get("id")
            
            # 1. 检查并去除重复的 id
            if item_id in seen_ids:
                duplicate_count += 1
                continue  # 如果已经存在，直接跳过这一行
            
            seen_ids.add(item_id)
            
            # 2. 删除 meta_info 里面的 fail_reason 字段
            # 我们需要先确认 meta_info 存在，且其中包含 fail_reason
            # import pdb; pdb.set_trace()
            if "meta_info" in data and "fail_reason" in data["meta_info"]:
                del data["meta_info"]["fail_reason"]
                fail_reason_count += 1
                
            # 3. 将处理后的数据转回 JSON 字符串并写入新文件
            # ensure_ascii=False 确保中文字符或特殊符号不会被转义成 Unicode 编码
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1
            
    print("=== 处理完成 ===")
    print(f"成功保存的数据条数: {processed_count}")
    print(f"清理的重复数据条数: {duplicate_count}")
    print(f"清理包含 `fail_reason` 数据条数: {fail_reason_count}")

# ========== 使用方法 ==========
if __name__ == "__main__":
    input_dir = Path('/data/ZS/defect_dataset/6_sft_dataset/teacher/val')
    output_dir = Path('/data/ZS/defect_dataset/6_sft_dataset/teacher/val1')

    input_files = sorted(list(input_dir.glob('*.jsonl')))
    
    for input_file in input_files:
        input_path = str(input_file)
        output_path = str(output_dir / input_file.name)
        clean_jsonl_data(input_path, output_path)
    
    
    # 请将这里的路径替换为你实际的文件路径
    # input_file = "/data/ZS/defect_dataset/6_sft_dataset/teacher/train/stripe_phase123_gt_negative.jsonl"   # 你的原始文件
    # output_file = "/data/ZS/defect_dataset/6_sft_dataset/teacher/train/stripe_phase123_gt_negative1.jsonl"   # 清洗后保存的新文件
    
    # clean_jsonl_data(input_file, output_file)
            
