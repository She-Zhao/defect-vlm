"""
构造数据，且没有任何cot数据
"""
import os
import json
import re
import random
from tqdm import tqdm

def process_assistant_minimalist(content):
    """
    极简改造：剥离 step1~3，只保留 defect 字段
    """
    try:
        clean_text = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        new_data = {
            "defect": data.get("defect", "background")
        }
        return json.dumps(new_data, ensure_ascii=False, indent=4)
    except Exception as e:
        return None

def process_user_minimalist(content):
    """
    极简改造：对 User 提示词开刀，删除分析步骤，修改 JSON 输出模板
    注意：我们保留了缺陷定义(Defect Definitions)，因为直接分类依然需要特征映射标准
    """
    # 1. 彻底删除“分析步骤” (Analysis Steps)
    analysis_pattern = r"# Analysis Steps[\s\S]*?(?=# Output Format)"
    content = re.sub(analysis_pattern, "", content)
    
    # 2. 修改输出格式约束 (Output Format)，只要求输出 defect
    out_format_pattern = r"{\n\s+\"step1\":[\s\S]*?\"defect\":[\s\S]*?\n}"
    out_format_repl = (
        "{\n"
        "    \"defect\": \"Defect Category Name\" (If there is no defect, please output \"background\")\n"
        "}"
    )
    content = re.sub(out_format_pattern, out_format_repl, content)
    
    return content

def build_stage3_data(input_dir, output_path, categories, random_seed):
    random.seed(random_seed)
    
    final_combined_data = []
    
    print(f"🔍 正在扫描目录: {input_dir}")
    print(f"🎯 计划构建: 100% 纯极简分类数据 (已彻底移除 CoT 干扰)")
    print("-" * 50)

    for category_name, file_list in categories.items():
        category_data = []
        for filename in file_list:
            file_path = os.path.join(input_dir, filename)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        category_data.append(json.loads(line))
        
        # 打乱该类别内的数据
        random.shuffle(category_data)
        
        # 100% 抽取并制作 极简(Minimalist) 数据
        valid_count = 0
        for item in category_data:
            valid = True
            for msg in item["messages"]:
                if msg["role"] == "user":
                    msg["content"] = process_user_minimalist(msg["content"])
                elif msg["role"] == "assistant":
                    new_content = process_assistant_minimalist(msg["content"])
                    if new_content:
                        msg["content"] = new_content
                    else:
                        valid = False
            if valid:
                final_combined_data.append(item)
                valid_count += 1
                
        print(f"✅ [{category_name}] 贡献 -> 100% 纯极简数据: {valid_count}条")

    print("-" * 50)
    print("🔀 正在对数据集进行全局随机打乱 (Shuffle)...")
    random.shuffle(final_combined_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(final_combined_data, desc="💾 写入进度"):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n🎉 大功告成！Stage 3 纯极简数据集构建完毕。")
    print(f"📊 最终写入总量: {len(final_combined_data)} 条数据")
    print(f"📁 保存路径: {output_path}")

if __name__ == "__main__":
    # ================= 🎛️ 核心参数配置区 🎛️ =================
    INPUT_DIR = "/data/ZS/defect_dataset/7_swift_dataset/course/stage3_task_alignment_26k_defect_only"
    
    CATEGORIES_MAP = {
        "Positive": ["stripe_phase012_gt_positive.jsonl", "stripe_phase123_gt_positive.jsonl"],
        "Negative": ["stripe_phase012_gt_negative.jsonl", "stripe_phase123_gt_negative.jsonl"],
        "Rectification": ["stripe_phase012_gt_rectification.jsonl", "stripe_phase123_gt_rectification.jsonl"]
    }
    
    OUTPUT_FILE = "/data/ZS/defect_dataset/7_swift_dataset_general/course/stage3_task_alignment_6k4_val_defect_only.jsonl"
    RANDOM_SEED = 42
    # ========================================================
    
    build_stage3_data(INPUT_DIR, OUTPUT_FILE, CATEGORIES_MAP, RANDOM_SEED)