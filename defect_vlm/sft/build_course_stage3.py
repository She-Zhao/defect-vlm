"""
对general数据集构建课程学习阶段3所需要格式的数据
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

def build_stage3_data(input_dir, output_path, categories, total_minimalist, total_cot, random_seed):
    random.seed(random_seed)
    
    # 计算每个类别应该分配的配额 (均分)
    cat_names = list(categories.keys())
    num_cats = len(cat_names)
    
    min_per_cat = total_minimalist // num_cats
    cot_per_cat = total_cot // num_cats
    
    final_combined_data = []
    
    print(f"🔍 正在扫描目录: {input_dir}")
    print(f"🎯 计划构建: 极简分类(80%)={total_minimalist}条, 经验重放CoT(20%)={total_cot}条")
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
        
        total_loaded = len(category_data)
        required_total = min_per_cat + cot_per_cat
        
        if total_loaded < required_total:
            print(f"⚠️ 警告: [{category_name}] 数据量({total_loaded})不足分配要求({required_total})")
            # 【修复点】动态计算目标比例，而不是硬编码 0.8
            if required_total > 0:
                target_ratio = min_per_cat / required_total
            else:
                target_ratio = 1.0 # 防止除零异常
                
            actual_min = int(total_loaded * target_ratio)
            actual_cot = total_loaded - actual_min
        else:
            actual_min = min_per_cat
            actual_cot = cot_per_cat
            
        # 1. 抽取并制作 极简(Minimalist) 数据
        minimalist_samples = category_data[:actual_min]
        for item in minimalist_samples:
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
                
        # 2. 抽取并保留 原汁原味(CoT) 数据
        cot_samples = category_data[actual_min : actual_min + actual_cot]
        final_combined_data.extend(cot_samples)
        
        print(f"✅ [{category_name}] 贡献 -> 极简数据: {len(minimalist_samples)}条 | CoT数据: {len(cot_samples)}条")

    print("-" * 50)
    print("🔀 正在对 80/20 混合数据集进行全局随机打乱 (Shuffle)...")
    random.shuffle(final_combined_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(final_combined_data, desc="💾 写入进度"):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n🎉 大功告成！Stage 3 终极对齐数据集构建完毕。")
    print(f"📊 最终写入总量: {len(final_combined_data)} 条数据")
    print(f"📁 保存路径: {output_path}")

if __name__ == "__main__":
    # ================= 🎛️ 核心参数配置区 🎛️ =================
    INPUT_DIR = "/data/ZS/defect_dataset/7_swift_dataset_general/val"
    
    # CATEGORIES_MAP = {
    #     "Positive": ["stripe_phase012_gt_positive.jsonl", "stripe_phase123_gt_positive.jsonl"],
    #     "Negative": ["stripe_phase012_gt_negative.jsonl", "stripe_phase123_gt_negative.jsonl"],
    #     "Rectification": ["stripe_phase012_gt_rectification.jsonl", "stripe_phase123_gt_rectification.jsonl"]
    # }

    CATEGORIES_MAP = {
        "Positive": ["sp012_gt_positive.jsonl", "sp123_gt_positive.jsonl"],
        "Negative": ["sp012_gt_negative.jsonl", "sp123_gt_negative.jsonl"],
        "Rectification": ["sp012_gt_rectification.jsonl", "sp123_gt_rectification.jsonl"]
    }

    # 分配策略：总计 20,000 条 (16000 + 4000)
    TOTAL_MINIMALIST = 60000  # 80%
    TOTAL_COT = 0          # 20%
    
    OUTPUT_FILE = "/data/ZS/defect_dataset/7_swift_dataset_general/course/stage3_task_alignment_6k4_val.jsonl"
    RANDOM_SEED = 42
    # ========================================================
    
    build_stage3_data(INPUT_DIR, OUTPUT_FILE, CATEGORIES_MAP, TOTAL_MINIMALIST, TOTAL_COT, RANDOM_SEED)
