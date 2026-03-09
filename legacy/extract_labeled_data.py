"""
根据/0_defect_dataset_raw/paint_stripe/labels/train_val.json里面的标注，
从/0_defect_dataset_raw/paint_stripe/images里面复制出来没有标注的图像
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_labeled_images(json_path: str, source_img_dir: str, target_img_dir: str):
    """
    读取 JSON 文件中的标注图像名称，遍历源目录，
    将【存在于】 JSON 中的图像按照原目录结构复制到目标目录。
    """
    print(f"📖 正在加载标注 JSON 文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 提取所有已标注的图像文件名，存入 set 
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
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
    
    print(f"📦 源目录中共找到 {len(image_files)} 个图像文件。")

    # 3. 开始比对和复制
    copied_count = 0
    skipped_count = 0
    
    target_path.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(image_files, desc="复制已标注图像"):
        filename = img_file.name

        # 核心逻辑：条件反转，如果在 JSON 里，就复制
        if filename in labeled_filenames:
            relative_path = img_file.relative_to(source_path)
            dest_file = target_path / relative_path

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest_file)
            copied_count += 1
        else:
            # 不在 json 里，跳过
            skipped_count += 1

    # 4. 打印最终数据报表
    print("\n" + "="*40)
    print("📊 已标注数据提取任务完成报告")
    print("="*40)
    print(f"🔹 源目录图像总数:   {len(image_files)}")
    print(f"🔹 未标注 (跳过):     {skipped_count}")
    print(f"🔹 已标注 (已复制):   {copied_count}")
    print(f"📁 已标注数据存放至: {target_img_dir}")
    print("="*40)
    
    # 终极安全校验
    if copied_count != len(labeled_filenames):
        print(f"⚠️ 警告: JSON里有 {len(labeled_filenames)} 张图，但只在文件夹里找到了 {copied_count} 张！请检查是否有图片丢失！")

if __name__ == "__main__":
    # 配置路径
    JSON_PATH = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json"
    SOURCE_IMG_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images"
    
    # 存放已标注数据的文件夹
    TARGET_IMG_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images_labeled"

    extract_labeled_images(JSON_PATH, SOURCE_IMG_DIR, TARGET_IMG_DIR)