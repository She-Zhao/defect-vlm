"""
根据/0_defect_dataset_raw/paint_stripe/labels/train_val.json里面的标注，
从/0_defect_dataset_raw/paint_stripe/images里面复制出来没有标注的图像
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_unlabeled_images(json_path: str, source_img_dir: str, target_img_dir: str):
    """
    读取 JSON 文件中的标注图像名称，遍历源目录，
    将不在 JSON 中的图像按照原目录结构复制到目标目录。
    """
    print(f"📖 正在加载标注 JSON 文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 提取所有已标注的图像文件名，存入 set 以实现 O(1) 极速查找
    labeled_filenames = set()
    for img_info in data.get('images', []):
        labeled_filenames.add(img_info['file_name'])

    print(f"✅ JSON 中共找到 {len(labeled_filenames)} 张已标注的图像。")

    source_path = Path(source_img_dir)
    target_path = Path(target_img_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"❌ 源图像目录不存在: {source_img_dir}")

    # 2. 扫描源目录下的所有文件
    print(f"🔍 正在扫描源目录及其子目录: {source_img_dir}")
    all_files = list(source_path.rglob('*'))
    # 过滤掉文件夹，只保留实际文件 (并且可以加个后缀过滤，防止拷了隐藏文件)
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
    
    print(f"📦 源目录中共找到 {len(image_files)} 个图像文件。")

    # 3. 开始比对和复制
    copied_count = 0
    skipped_count = 0
    
    # 创建目标总文件夹
    target_path.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(image_files, desc="复制未标注图像"):
        filename = img_file.name

        if filename not in labeled_filenames:
            # 发现未标注数据！
            # 计算它相对于 source_img_dir 的相对路径 (例如: 16col0/0000_0002.png)
            relative_path = img_file.relative_to(source_path)
            
            # 拼接到目标文件夹下 (例如: target_img_dir/16col0/0000_0002.png)
            dest_file = target_path / relative_path

            # 如果目标子文件夹 (如 16col0) 不存在，则自动创建
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # 使用 copy2 可以保留文件的原始创建时间等 metadata
            shutil.copy2(img_file, dest_file)
            copied_count += 1
        else:
            # 已经在 json 里了，跳过
            skipped_count += 1

    # 4. 打印最终数据报表
    print("\n" + "="*40)
    print("📊 数据分离任务完成报告")
    print("="*40)
    print(f"🔹 源目录图像总数:   {len(image_files)}")
    print(f"🔹 已标注 (跳过):     {skipped_count}")
    print(f"🔹 未标注 (已复制):   {copied_count}")
    print(f"📁 未标注数据存放至: {target_img_dir}")
    print("="*40)

if __name__ == "__main__":
    # 配置路径
    JSON_PATH = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json"
    SOURCE_IMG_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images"
    
    # 建议将抽出来的未标注数据单独放在一个同级的文件夹里
    TARGET_IMG_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images_unlabeled"

    extract_unlabeled_images(JSON_PATH, SOURCE_IMG_DIR, TARGET_IMG_DIR)
