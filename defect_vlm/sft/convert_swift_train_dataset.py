"""
6_sft_dataset里得到的教师模型的回答和4_api_reponse学生模型的提示词转化为ms-swift微调所需要的message格式, 
并包含强力的防止知识泄漏（上帝视角幻觉）的正则清洗与统计功能。
"""

import json
import os
import re
from pathlib import Path

# ==========================================
# 定义防泄漏清洗规则库 (字典存储，方便扩展和统计)
# ==========================================
LEAKAGE_RULES = {
    "GT_Defect_Label": {
        "pattern": re.compile(r'ground\s*truth\s*defect\s*label', re.IGNORECASE),
        "repl": "actual defect",
        "count": 0
    },
    "GT_Label": {
        "pattern": re.compile(r'ground\s*truth\s*label', re.IGNORECASE),
        "repl": "actual defect",
        "count": 0
    },
    "Ground_Truth": {
        # 兜底匹配单独的 ground truth
        "pattern": re.compile(r'ground\s*truth', re.IGNORECASE),
        "repl": "actual situation",
        "count": 0
    },
    "True_Label": {
        # 防止模型自己发挥出 true label 这种词汇
        "pattern": re.compile(r'true\s*label', re.IGNORECASE),
        "repl": "actual defect",
        "count": 0
    }
}

def convert_to_swift_format(stu_path, tea_path, output_path):
    print(f"开始转换数据格式...\n输入学生: {Path(stu_path).name} | 教师: {Path(tea_path).name}")
    
    id2prompt = {}            # stu_id: stu_prompt
    with open(stu_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if not line.strip(): continue
                
                data = json.loads(line)
                stu_id = data['id']
                stu_prompt = data['conversation'][0]['value']
                
                if stu_id not in id2prompt:
                    id2prompt[stu_id] = stu_prompt
                else:
                    print(f"⚠️ id: {stu_id} 存在重复！已跳过")
                    continue
                    
            except Exception as e:
                print(f"处理 {stu_id} 时发生错误: {e}")
                continue
    
    swift_dataset = []
    success_count = 0
    
    with open(tea_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                
                # 提取原始字段
                tea_id = data['id']
                images = data['image']
                conversation = data['conversation']
                    
                assistant_text = conversation[1]['value']
                
                # ==========================================
                # 核心清洗逻辑：遍历规则库进行防泄漏替换并统计
                # ==========================================
                for rule_name, rule_info in LEAKAGE_RULES.items():
                    # re.subn 返回 (new_string, number_of_subs_made)
                    assistant_text, sub_num = rule_info["pattern"].subn(rule_info["repl"], assistant_text)
                    rule_info["count"] += sub_num  # 累加替换次数
                
                if tea_id in id2prompt:
                    user_text = id2prompt[tea_id]
                else:
                    raise ValueError(f"id: {tea_id} 在学生组文件中未找到！")
                
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
            
            except Exception as e:
                print(f"处理id: {tea_id} 时发生错误: {e}")
                continue

    # 写入新文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in swift_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 成功处理了 {success_count} 条数据。\n")

if __name__ == "__main__":
    tea_dir = Path('/data/ZS/defect_dataset/6_sft_dataset/teacher/test')
    stu_dir = Path('/data/ZS/defect_dataset/4_api_request/student/test')
    output_dir = Path('/data/ZS/defect_dataset/7_swift_dataset/test')
    
    tea_jsonls = sorted(list(tea_dir.glob('*.jsonl')))
    stu_jsonls = sorted(list(stu_dir.glob('*.jsonl')))
    
    for stu_file, tea_file in zip(stu_jsonls, tea_jsonls):
        if stu_file.name != tea_file.name:
             raise ValueError(f"文件名称不对齐！学生: {stu_file.name}, 教师: {tea_file.name}")
             
        stu_jsonl = str(stu_file)
        tea_jsonl = str(tea_file)
        output_jsonl = str(output_dir / tea_file.name)
        
        convert_to_swift_format(stu_jsonl, tea_jsonl, output_jsonl)
        
    # ==========================================
    # 全局运行结束，打印清洗报告
    # ==========================================
    print("="*50)
    print("🛡️ 数据防泄漏(幻觉)清洗报告 🛡️")
    print("="*50)
    total_replacements = 0
    for rule_name, rule_info in LEAKAGE_RULES.items():
        print(f"规则 [{rule_name}] 触发拦截/替换次数: {rule_info['count']} 次")
        total_replacements += rule_info['count']
    print("-" * 50)
    print(f"总计修正潜在泄漏点: {total_replacements} 处")
    print("="*50)