"""
6_sft_dataset下面的jsonl数据转化为ms-swift微调所需要的message格式, 并保存到7_swift_dataset下面
"""

import json
import os
from pathlib import Path

def convert_to_swift_format(input_path, output_path):
    print(f"开始转换数据格式...\n输入文件: {input_path}")
    
    swift_dataset = []
    success_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 提取原始字段
            images = data.get("image", [])
            conversation = data.get("conversation", [])
            
            # 确保对话格式完整 (一问一答)
            if len(conversation) != 2:
                print(f'数据格式错误，conversation内包括 {len(conversation)} 个数据')
                continue
                
            user_text = conversation[0]['value']
            assistant_text = conversation[1]['value']
            
            # 构造 ms-swift 标准格式
            swift_data = {
                "images": images,
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text}
                ]
            }
            swift_dataset.append(swift_data)
            success_count += 1

    # 写入新文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in swift_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"转换完成！成功处理了 {success_count} 条数据。")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    input_dir = Path('/data/ZS/defect_dataset/6_sft_dataset/teacher/train')
    output_dir = Path('/data/ZS/defect_dataset/7_swift_dataset/teacher/train')
    all_jsonl = list(input_dir.glob('*.jsonl'))
    
    for input_file in all_jsonl:
        input_jsonl = str(input_file)
        output_jsonl = str(output_dir / input_file.name)
        # import pdb; pdb.set_trace()
        convert_to_swift_format(input_jsonl, output_jsonl)