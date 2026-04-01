"""
将`/data/ZS/defect_dataset/7_swift_dataset/val`下面的数据转化为像
`/data/ZS/defect_dataset/7_swift_dataset/test/012_pos400_neg300_rect300.jsonl`一样格式的数据
这样可以直接跑验证
"""
import os
import json
import re
import glob

def process_and_merge_val_data(input_dir, output_path):
    # 查找目录下所有的 jsonl 文件
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"⚠️ 在 {input_dir} 目录下没有找到任何 .jsonl 文件。")
        return

    processed_count = 0
    
    # 打开输出文件（覆盖写模式）
    with open(output_path, 'w', encoding='utf-8') as fout:
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            print(f"正在处理文件: {file_name} ...")
            
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if not line.strip():
                        continue
                        
                    data = json.loads(line)
                    
                    # 1. 构造 ID (从 image 路径中提取)
                    # 示例: /.../stripe_phase012_gt_rectification/.../global_1000942.png
                    img_path = data["images"][0]
                    unique_id = "unknown_id"
                    match = re.search(r'(stripe_phase\d+_gt_[a-z]+).*?global_(\d+)\.png', img_path)
                    if match:
                        folder_type = match.group(1)
                        img_num = match.group(2)
                        # 转换名称缩写，对齐你的示例
                        short_type = (folder_type
                                      .replace("stripe_phase", "sp")
                                      .replace("negative", "neg")
                                      .replace("positive", "pos")
                                      .replace("rectification", "rect"))
                        unique_id = f"{short_type}_{img_num}"
                    
                    # 2. 提取 Label
                    label_value = "unknown"
                    assistant_msg = next((m for m in data.get("messages", []) if m["role"] == "assistant"), None)
                    if assistant_msg:
                        try:
                            # 解析 assistant content 里的 JSON
                            ans_dict = json.loads(assistant_msg["content"])
                            label_value = ans_dict.get("defect", "unknown")
                        except json.JSONDecodeError:
                            print(f"⚠️ 警告: 无法解析 {unique_id} 的 Assistant 答案 JSON。")
                            
                    # 3. 过滤 Messages (仅保留 user)
                    user_msg = next((m for m in data.get("messages", []) if m["role"] == "user"), None)
                    messages = [user_msg] if user_msg else []
                    
                    # 4. 构建新的字典结构
                    new_data = {
                        "id": unique_id,
                        "images": data["images"],
                        "messages": messages,
                        "meta_info": {
                            "label": label_value
                        }
                    }
                    
                    # 写入到合并文件
                    fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    processed_count += 1

    print(f"\n🎉 处理完毕！共合并转换了 {processed_count} 条数据。")
    print(f"📁 输出文件保存在: {output_path}")

if __name__ == "__main__":
    # 输入目录 (包含多个 jsonl 文件)
    input_dir = "/data/ZS/defect_dataset/7_swift_dataset/val"
    # 输出合并后的文件路径
    output_path = "/data/ZS/defect_dataset/7_swift_dataset/test/val_merged.jsonl"
    
    process_and_merge_val_data(input_dir, output_path)

