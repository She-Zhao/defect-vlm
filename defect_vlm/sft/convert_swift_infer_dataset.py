"""
4_api_request/test/val下的jsonl数据转化为ms-swift推理需要的message格式, 并保存到7_swift_dataset下面
暂时不用这个，因为验证集也需要有回复。预计后面这个脚本会废弃。
"""
import json
import os
from pathlib import Path

def convert_to_swift_infer_format(input_path, output_path):
    print(f"🚀 开始转换推理数据集...\n输入文件: {input_path}")
    
    swift_dataset = []
    success_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. 提取原始数据字段
            sample_id = data.get("id", "")
            images = data.get("image", [])
            conversation = data.get("conversation", [])
            meta_info = data.get("meta_info", {})
            
            # 2. 提取用户指令 (测试集只需输入 prompt)
            user_content = ""
            for turn in conversation:
                if turn.get("from") == "human":
                    user_content = turn.get("value", "")
                    break
            
            # 如果没有找到用户指令，跳过异常数据
            if not user_content:
                print(f"⚠️ 警告: 样本 {sample_id} 未找到 human prompt，已跳过。")
                continue
                
            # 3. 组装为 ms-swift 标准格式
            # 注意：完整保留了 meta_info，这对后续计算 mAP 和准确率至关重要
            swift_data = {
                "id": sample_id,
                "images": images,
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "meta_info": meta_info
            }
            swift_dataset.append(swift_data)
            success_count += 1

    # 4. 写入新的 JSONL 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in swift_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 转换完成！成功处理了 {success_count} 条测试数据。")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    input_dir = Path('/data/ZS/defect_dataset/4_api_request/student/val')
    output_dir = Path('/data/ZS/defect_dataset/7_swift_dataset/student/val')
    all_jsonl = list(input_dir.glob('*.jsonl'))
    
    for input_file in all_jsonl:
        input_jsonl = str(input_file)
        output_jsonl = str(output_dir / input_file.name)
        # import pdb; pdb.set_trace()
        convert_to_swift_infer_format(input_jsonl, output_jsonl)
