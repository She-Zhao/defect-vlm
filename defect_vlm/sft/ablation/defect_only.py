"""
生成只有defect_name的微调数据
输入：/data/ZS/defect_dataset/7_swift_dataset/train里面的原始训练数据
输出：
"""
import json
import os
import re

def process_dataset(input_dir, output_dir, file_names):
    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)

    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        if not os.path.exists(input_path):
            print(f"⚠️ 找不到文件，请检查路径: {input_path}")
            continue

        processed_lines = 0
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip():
                    continue
                
                # 解析单行 jsonl
                data = json.loads(line)
                
                for message in data.get("messages", []):
                    # 1. 处理 User 的提示词
                    if message["role"] == "user":
                        content = message["content"]
                        
                        # 删除 # Analysis Steps 到 # Output Format 之间的所有内容
                        content = re.sub(r'# Analysis Steps[\s\S]*?(?=# Output Format)', '', content)
                        
                        # 删除 # Output Format 示例中的 step1, step2, step3 行 (匹配中英文内容)
                        # 这里使用正则表达式匹配包含 "step1": "..." 的整行内容并删除
                        content = re.sub(r'[ \t]*"step[123]":\s*".*?",\n', '', content)
                        
                        message["content"] = content
                        
                    # 2. 处理 Assistant 的回答
                    elif message["role"] == "assistant":
                        try:
                            # 尝试解析 assistant 的回答为 JSON 字典
                            ans_dict = json.loads(message["content"])
                            # 提取 defect 字段
                            defect_value = ans_dict.get("defect", "None")
                            
                            # 重新构建只包含 defect 的 JSON 字符串
                            # 加上缩进和换行，使其符合原数据的格式
                            new_ans = json.dumps({"defect": defect_value}, ensure_ascii=False, indent=4)
                            # 调整格式，让它看起来像手打的 json 字符串（可选项）
                            message["content"] = new_ans
                        except json.JSONDecodeError:
                            # 如果原本的回答不是标准 JSON，打印提示
                            print(f"⚠️ 解析 Assistant 回答失败，保持原样: {message['content']}")

                # 将处理后的字典重新转回 jsonl 行写入新文件
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
        print(f"✅ 完成处理: {file_name} -> 共转换 {processed_lines} 条数据")

if __name__ == "__main__":

    # 原始数据目录和文件名
    input_dir = "/data/ZS/defect_dataset/7_swift_dataset/test"
    # 建议将消融实验的数据存放在新目录，避免覆盖原始数据
    output_dir = "/data/ZS/defect_dataset/7_swift_dataset/ablation/defect_only/test"
    
    # input_dir内待处理的文件
    # file_names = [
    #     "stripe_phase012_gt_negative.jsonl",
    #     "stripe_phase012_gt_rectification.jsonl",
    #     "stripe_phase123_gt_positive.jsonl",
    #     "stripe_phase012_gt_positive.jsonl",
    #     "stripe_phase123_gt_negative.jsonl",
    #     "stripe_phase123_gt_rectification.jsonl"
    # ]

    file_names = ['val_merged.jsonl']
    
    process_dataset(
        input_dir = input_dir,
        output_dir = output_dir,
        file_names = file_names
    )
    print(f"\n🎉 所有文件处理完毕！消融实验数据集已保存在 {output_dir} 目录下。")

