"""
保留cot，去掉step1 step2
"""
import json
import os
import re

def process_dataset_cot_only(input_dir, output_dir, file_names):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        if not os.path.exists(input_path):
            print(f"⚠️ 找不到文件: {input_path}")
            continue

        processed_lines = 0
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                for message in data.get("messages", []):
                    # 1. 处理 User 的提示词
                    if message["role"] == "user":
                        content = message["content"]
                        
                        # 步骤A: 删除整个 # Analysis Steps 段落 (一直匹配到 # Output Format 之前)
                        content = re.sub(r'# Analysis Steps[\s\S]*?(?=# Output Format)', '', content)
                        
                        # 步骤B: 替换 # Output Format 下方的 JSON 示例
                        # 使用正则匹配旧的 JSON 块并替换为新的
                        old_json_pattern = r'\{\s*"step1":[\s\S]*?"defect":.*?\n\}'
                        new_json_format = '{\n    "cot": "Output the thought process in chain-of-thought format",\n    "defect": "Defect Category Name" (If there is no defect, please output "background")\n}'
                        content = re.sub(old_json_pattern, new_json_format, content)
                        
                        # 清理可能多余的空行
                        content = content.replace('\n\n\n', '\n\n')
                        message["content"] = content
                        
                    # 2. 处理 Assistant 的回答
                    elif message["role"] == "assistant":
                        try:
                            # 解析原来的 JSON 回答
                            ans_dict = json.loads(message["content"])
                            
                            # 提取 step3 (作为 cot) 和 defect
                            # 如果原数据由于某种原因没有 step3，则给个默认空字符串防止报错
                            cot_content = ans_dict.get("step3", "")
                            defect_content = ans_dict.get("defect", "background")
                            
                            # 构建新的精简版字典
                            new_ans_dict = {
                                "cot": cot_content,
                                "defect": defect_content
                            }
                            
                            # 重新转回 JSON 字符串
                            message["content"] = json.dumps(new_ans_dict, ensure_ascii=False, indent=4)
                        except json.JSONDecodeError:
                            print(f"⚠️ 解析 Assistant 回答失败: {message['content']}")

                # 写入新文件
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
        print(f"✅ 完成处理: {file_name} -> 共转换 {processed_lines} 条数据")

if __name__ == "__main__":
    # 原始数据目录
    input_dir = "/data/ZS/defect_dataset/7_swift_dataset/test"
    # 消融实验（仅保留CoT和defect）数据存放目录
    output_dir = "/data/ZS/defect_dataset/7_swift_dataset/ablation/cot_defect/test"

    # file_names = [
    #     "stripe_phase012_gt_negative.jsonl",
    #     "stripe_phase012_gt_rectification.jsonl",
    #     "stripe_phase123_gt_positive.jsonl",
    #     "stripe_phase012_gt_positive.jsonl",
    #     "stripe_phase123_gt_negative.jsonl",
    #     "stripe_phase123_gt_rectification.jsonl"
    # ]

    file_names = ['val_merged.jsonl']

    process_dataset_cot_only(
        input_dir = input_dir,
        output_dir = output_dir,
        file_names = file_names
    )
    print("\n🎉 '仅保留 CoT' 消融实验数据集已生成完毕！保存在 train_ablation_cot_only 目录下。")
