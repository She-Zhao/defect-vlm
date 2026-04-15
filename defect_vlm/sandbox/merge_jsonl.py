"""
合并多个 JSONL 文件并进行全局打乱 (Merge and Shuffle JSONL)
适用于组装最终的 SFT 微调数据或 Eval 评测数据。
"""
import json
import random
from pathlib import Path
from tqdm import tqdm

def merge_and_shuffle_jsonl(input_files: list, output_file: Path, random_seed: int = 42):
    """
    读取多个 jsonl 文件，合并数据，全局打乱，并写入新文件。
    """
    # 设置随机种子以保证打乱结果可复现
    random.seed(random_seed)
    all_data = []

    print("-" * 50)
    print("📥 开始读取文件...")
    
    # 1. 遍历读取所有输入文件
    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ 警告: 文件不存在，已跳过 -> {path}")
            continue

        print(f"📖 正在加载: {path.name}")
        file_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
                    file_count += 1
        print(f"   └─ 成功读取 {file_count} 条数据")

    total_loaded = len(all_data)
    print(f"\n✅ 所有文件读取完毕！共加载 {total_loaded} 条数据。")

    if total_loaded == 0:
        print("❌ 没有加载到任何数据，脚本中止。")
        return

    # 2. 全局打乱数据
    print("\n🔀 正在进行全局随机打乱 (Global Shuffle)...")
    random.shuffle(all_data)

    # 3. 写入输出文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 正在写入最终文件: {output_file.name}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(all_data, desc="写入进度"):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n🎉 大功告成！合并并打乱后的数据已保存至:\n📂 {output_file.resolve()}")
    print("-" * 50)

if __name__ == "__main__":
    # ================= 🎛️ 配置区 =================
    
    # 你可以手动指定需要合并的文件列表 (推荐，更精准)
    BASE_DIR = Path('/data/ZS/defect_dataset/7_swift_dataset/ablation/defect_only/val')
    
    # INPUT_FILES = [
    #     BASE_DIR / 'sp012_gt_negative.jsonl',
    #     BASE_DIR / 'sp012_gt_rectification.jsonl',
    #     BASE_DIR / 'sp123_gt_positive.jsonl',
    #     BASE_DIR / 'sp012_gt_positive.jsonl',
    #     BASE_DIR / 'sp123_gt_negative.jsonl',
    #     BASE_DIR / 'sp123_gt_rectification.jsonl'
    # ]
    
    INPUT_FILES = list(BASE_DIR.glob('*.jsonl'))
    
    # 输出合并后的文件路径
    OUTPUT_FILE = Path('/data/ZS/defect_dataset/7_swift_dataset/course/stage3_task_alignment_6k4_val_defect_only.jsonl')
    
    # 随机种子
    RANDOM_SEED = 42
    # ==============================================

    merge_and_shuffle_jsonl(
        input_files=INPUT_FILES, 
        output_file=OUTPUT_FILE, 
        random_seed=RANDOM_SEED
    )