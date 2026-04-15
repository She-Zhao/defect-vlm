import os
import json
import random
from tqdm import tqdm

def build_stage2_data(input_dir, output_path, categories, samples_per_category, random_seed):
    """
    构建 Stage 2 数据集的主控流：严格执行 1:1:1 均衡采样
    """
    random.seed(random_seed)
    
    final_combined_data = []
    
    print(f"🔍 正在扫描目录: {input_dir}")
    print(f"🎯 目标各类别采样数: {samples_per_category} 条")
    print("-" * 50)

    # 1. 遍历三个类别，分别进行加载和抽样
    for category_name, file_list in categories.items():
        category_data = []
        
        # 读取该类别下的所有文件
        for filename in file_list:
            file_path = os.path.join(input_dir, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ 警告: 找不到文件 {filename}")
                continue
                
            print(f"📖 正在加载 [{category_name}] 数据: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        category_data.append(json.loads(line))
        
        total_loaded = len(category_data)
        
        # 执行均衡抽样
        if total_loaded >= samples_per_category:
            sampled = random.sample(category_data, samples_per_category)
            print(f"✅ [{category_name}] 数据池共 {total_loaded} 条，成功抽取 {samples_per_category} 条。")
        else:
            sampled = category_data
            print(f"⚠️ 警告: [{category_name}] 数据池仅有 {total_loaded} 条，不足期望值 {samples_per_category}，将使用全部数据。")
            
        final_combined_data.extend(sampled)
        print("-" * 50)

    # 2. 全局打乱 (非常重要！防止模型连续吃到同一类别的样本导致梯度震荡)
    print("🔀 正在对合并后的全局数据集进行随机打乱 (Shuffle)...")
    random.shuffle(final_combined_data)

    # 3. 写入最终文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("💾 正在写入最终文件...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(final_combined_data, desc="写入进度"):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n🎉 大功告成！Stage 2 逻辑注入数据集构建完毕。")
    print(f"📊 最终有效写入总量: {len(final_combined_data)} 条数据")
    print(f"📁 保存路径: {output_path}")


if __name__ == "__main__":
    # ================= 🎛️ 核心参数配置区 🎛️ =================
    
    # 1. 原始数据集目录
    INPUT_DIR = "/data/ZS/defect_dataset/7_swift_dataset/val"
    
    # 2. 类别文件映射字典 (将不同后缀的文件归类)
    CATEGORIES_MAP = {
        "Positive (同意先验)": [
            "stripe_phase012_gt_positive.jsonl",
            "stripe_phase123_gt_positive.jsonl"
        ],
        "Negative (驳斥先验)": [
            "stripe_phase012_gt_negative.jsonl",
            "stripe_phase123_gt_negative.jsonl"
        ],
        "Rectification (纠正先验)": [
            "stripe_phase012_gt_rectification.jsonl",
            "stripe_phase123_gt_rectification.jsonl"
        ]
    }
    
    # 3. 期望的每类采样数量 (按照 1:1:1 比例，每类抽取 2000，总计 6000)
    SAMPLES_PER_CATEGORY = 400
    
    # 4. 输出文件路径
    OUTPUT_FILE = "/data/ZS/defect_dataset/7_swift_dataset/course/stage2_logic_injection_1k2_val.jsonl"
    
    # 5. 随机种子 (保证抽样和打乱的可复现性)
    RANDOM_SEED = 42
    
    # ========================================================
    
    build_stage2_data(
        input_dir=INPUT_DIR,
        output_path=OUTPUT_FILE,
        categories=CATEGORIES_MAP,
        samples_per_category=SAMPLES_PER_CATEGORY,
        random_seed=RANDOM_SEED
    )