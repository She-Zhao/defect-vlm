"""
根据/0_defect_dataset_raw/paint_stripe/labels/train_val.json里面的标注,
从/0_defect_dataset_raw/paint_stripe/images里面将标注图像和没有标注的图像分开
标注图像：/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images
未标注图像：/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images_unlabeled
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset_by_json(json_path: str, source_img_dir: str, labeled_dir: str, unlabeled_dir: str):
    """
    读取 JSON 文件，扫描源图像目录，在一次遍历中将图像精准分拣：
    - 存在于 JSON 中的 -> 复制到 labeled_dir
    - 不在 JSON 中的 -> 复制到 unlabeled_dir
    均保持原有的子目录结构。
    """
    print(f"📖 正在加载标注 JSON 文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 提取所有已标注的图像文件名，存入 set 以实现 O(1) 极速查找
    labeled_filenames = set()
    for img_info in data.get('images', []):
        labeled_filenames.add(img_info['file_name'])

    print(f"✅ JSON 中共找到 {len(labeled_filenames)} 张已标注图像的记录。")

    source_path = Path(source_img_dir)
    target_labeled = Path(labeled_dir)
    target_unlabeled = Path(unlabeled_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"❌ 源图像目录不存在: {source_img_dir}")

    # 2. 扫描源目录下的所有文件
    print(f"🔍 正在扫描源目录及其子目录: {source_img_dir}")
    all_files = list(source_path.rglob('*'))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
    
    total_images = len(image_files)
    print(f"📦 源目录中共找到 {total_images} 个图像文件。")

    # 3. 开始分拣和复制
    copied_labeled = 0
    copied_unlabeled = 0
    
    # 提前创建两个目标根目录
    target_labeled.mkdir(parents=True, exist_ok=True)
    target_unlabeled.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(image_files, desc="🔄 正在分拣数据集"):
        filename = img_file.name
        
        # 计算相对路径 (例如: 16col0/0000_0002.png)
        relative_path = img_file.relative_to(source_path)

        if filename in labeled_filenames:
            # 属于已标注数据
            dest_file = target_labeled / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest_file)
            copied_labeled += 1
        else:
            # 属于未标注数据
            dest_file = target_unlabeled / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest_file)
            copied_unlabeled += 1

    # 4. 打印最终数据报表与安全校验
    print("\n" + "="*50)
    print("📊 数据集二合一分拣任务完成报告")
    print("="*50)
    print(f"🔹 源目录图像总数:       {total_images}")
    print(f"✅ 已分拣到 Labeled:    {copied_labeled} 张 -> {labeled_dir}")
    print(f"⬜ 已分拣到 Unlabeled:  {copied_unlabeled} 张 -> {unlabeled_dir}")
    print("="*50)
    
    # 终极安全校验
    if (copied_labeled + copied_unlabeled) != total_images:
        print("⚠️ 严重警告: 分拣后的文件总数与源文件总数不一致！请检查磁盘空间或读写权限。")
    else:
        print("🎉 完美！分拣后的总数完全对齐，两份数据互为完美的互补集！")
        
    if copied_labeled != len(labeled_filenames):
        print(f"⚠️ 警告: JSON里有 {len(labeled_filenames)} 张图，但在源文件夹里只找到了 {copied_labeled} 张匹配的图像！请核对是否缺失原图。")


if __name__ == "__main__":
    # 配置路径
    JSON_PATH = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json"
    SOURCE_IMG_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images"
    
    # 存放分拣后数据的两个互补文件夹
    LABELED_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images"
    UNLABELED_DIR = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/images_unlabeled"

    split_dataset_by_json(JSON_PATH, SOURCE_IMG_DIR, LABELED_DIR, UNLABELED_DIR)