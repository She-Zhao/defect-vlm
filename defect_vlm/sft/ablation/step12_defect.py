"""
仅保留step1和step2，验证cot的作用
"""
import json
import os
import re

def process_dataset_no_cot(input_dir, output_dir, file_names):
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
                        
                        # 正则删除 Analysis Steps 中的 Step 3
                        # 兼容可能存在的空格或不间断空格 (\u00a0)
                        content = re.sub(r'3\.[ \t\u00a0]+\*\*Step 3.*?\n', '', content)
                        
                        # 正则删除 Output Format 中的 step3 示例行
                        content = re.sub(r'[ \t]*"step3":\s*".*?",\n', '', content)
                        
                        message["content"] = content
                        
                    # 2. 处理 Assistant 的回答
                    elif message["role"] == "assistant":
                        try:
                            # 解析 JSON
                            ans_dict = json.loads(message["content"])
                            
                            # 删除 step3 键值对
                            if "step3" in ans_dict:
                                del ans_dict["step3"]
                                
                            # 重新转为 JSON 字符串
                            message["content"] = json.dumps(ans_dict, ensure_ascii=False, indent=4)
                        except json.JSONDecodeError:
                            print(f"⚠️ 解析 Assistant 回答失败: {message['content']}")

                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
        print(f"✅ 完成处理: {file_name} -> 共转换 {processed_lines} 条数据")

if __name__ == "__main__":
    # 原始数据目录
    input_dir = "/data/ZS/defect_dataset/7_swift_dataset_general/test"
    # 消融实验（无CoT）数据存放目录
    output_dir = "/data/ZS/defect_dataset/7_swift_dataset_general/ablation/step12_defect"

    # file_names = [
    #     "stripe_phase012_gt_negative.jsonl",
    #     "stripe_phase012_gt_rectification.jsonl",
    #     "stripe_phase123_gt_positive.jsonl",
    #     "stripe_phase012_gt_positive.jsonl",
    #     "stripe_phase123_gt_negative.jsonl",
    #     "stripe_phase123_gt_rectification.jsonl"
    # ]

    file_names = ['val_merged.jsonl']

    process_dataset_no_cot(
        input_dir = input_dir,
        output_dir = output_dir,
        file_names = file_names
    )
    print("\n🎉 无 CoT 消融实验数据集已生成完毕！保存在 train_ablation_no_cot 目录下。")
