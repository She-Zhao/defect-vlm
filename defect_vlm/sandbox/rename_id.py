"""
将 "id": 1000233 重命名为 "sp012_gt_neg_1000142"
"""
import json

def rename_ids(input_file, output_file):
    # 用于统计各类样本的数量
    counts = {"pos": 0, "neg": 0, "rect": 0, "unknown": 0}

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                original_id = data.get("id")
                
                # 安全获取第一张图片的路径
                images = data.get("image", [])
                if not images:
                    print(f"⚠️ 警告: 行 {line_num} 缺少 image 字段或列表为空，跳过该行。")
                    continue
                
                first_image_path = images[0]
                
                # 根据路径中的关键字确定中间的类别名称
                if "positive" in first_image_path:
                    type_str = "pos"
                elif "negative" in first_image_path:
                    type_str = "neg"
                elif "rectification" in first_image_path:
                    type_str = "rect"
                else:
                    print(f"⚠️ 警告: ID {original_id} 的图片路径未包含目标关键字: {first_image_path}")
                
                counts[type_str if type_str in counts else "unknown"] += 1
                
                # 拼接新的 ID
                new_id = f"sp012_gt_{type_str}_{original_id}"
                data["id"] = new_id
                
                # 重新序列化并写入，确保中文和原有格式不被破坏
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"❌ 错误: 行 {line_num} JSON 解析失败，已跳过。")
                continue

    print("-" * 30)
    print("✅ 处理完成！统计信息如下：")
    print(f"Positive (pos) 样本数: {counts['pos']}")
    print(f"Negative (neg) 样本数: {counts['neg']}")
    print(f"Rectification (rect) 样本数: {counts['rect']}")
    if counts['unknown'] > 0:
        print(f"⚠️ 未匹配关键字 (unk) 样本数: {counts['unknown']}")

if __name__ == "__main__":
    # 请在这里替换为你的实际文件路径
    input_path = "/data/ZS/defect_dataset/backup/5_api_response/qwen3-vl-235b-a22b-thinking.jsonl"
    output_path = "/data/ZS/defect_dataset/backup/5_api_response/qwen3-vl-235b-a22b-thinking1.jsonl"
    
    rename_ids(input_path, output_path)
