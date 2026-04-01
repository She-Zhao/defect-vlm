"""
提取无标签数据并迁移至飞轮数据文件夹
输入：
/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json  所有已标注图像的文件名集合。
/data/ZS/defect_dataset/1_paint_rgb     原始图像文件夹
输出：
/data/ZS/flywheel_dataset/1_paint_rgb       分割后的没有标注的数据空间
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_unlabeled_images(
    source_rgb_dir: str, 
    labeled_json_path: str, 
    target_rgb_dir: str, 
    target_projects: list, 
    light_sources: list
):
    """
    Args:
        source_rgb_dir: 原始图片根目录 (e.g., /data/ZS/defect_dataset/1_paint_rgb)
        labeled_json_path: 包含所有已标注数据的 JSON 文件路径
        target_rgb_dir: 目标飞轮数据集的根目录 (e.g., /data/ZS/flywheel_dataset/1_paint_rgb)
        target_projects: 需要处理的项目列表 (跳过不需要的，比如 ap_xxx)
        light_sources: 光源子文件夹列表
    """
    source_rgb_dir = Path(source_rgb_dir)
    target_rgb_dir = Path(target_rgb_dir)
    labeled_json_path = Path(labeled_json_path)

    # 1. 解析 JSON 获取已标注文件名集合 (使用 Set 保证 O(1) 的极速查找)
    print(f"📖 正在解析已标注数据列表: {labeled_json_path}")
    with open(labeled_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 提取所有的 file_name
    labeled_filenames = {img['file_name'] for img in json_data['images']}
    print(f"✅ 成功提取了 {len(labeled_filenames)} 个已标注的唯一文件名。")

    # 2. 遍历指定的项目和光源目录
    total_copied = 0

    for project in target_projects:
        print(f"\n🚀 开始处理项目: {project}")
        
        for light in light_sources:
            src_light_dir = source_rgb_dir / project / "images" / light
            dst_light_dir = target_rgb_dir / project / "images" / light
            
            if not src_light_dir.exists():
                print(f"⚠️ 警告: 源目录不存在，跳过 -> {src_light_dir}")
                continue
                
            # 获取源目录下所有的 .png 文件名
            all_files = [f.name for f in src_light_dir.iterdir() if f.is_file() and f.suffix == '.png']
            
            # 过滤出无标签文件 (在 all_files 中，但不在 labeled_filenames 中)
            unlabeled_files = [f for f in all_files if f not in labeled_filenames]
            
            if not unlabeled_files:
                print(f"  - [{light}] 没有找到无标签数据。")
                continue
                
            # 确保目标文件夹存在 (只创建 images 及其子文件夹，不创建 labels)
            dst_light_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  - [{light}] 找到 {len(unlabeled_files)} 张无标签图片，正在复制...")
            
            # 使用 tqdm 显示复制进度条
            for filename in tqdm(unlabeled_files, desc=f"Copying {light}", leave=False):
                src_file = src_light_dir / filename
                dst_file = dst_light_dir / filename
                
                # copy2 可以保留文件的元数据 (如创建时间等)
                shutil.copy2(src_file, dst_file)
                total_copied += 1

    print("\n" + "="*60)
    print(f"🎉 提取完成！共将 {total_copied} 张无标签图片复制到了: {target_rgb_dir}")
    print("="*60)


if __name__ == "__main__":
    # ================= 配置参数 =================
    
    # 原始数据集 RGB 根目录
    source_rgb_dir = "/data/ZS/defect_dataset/1_paint_rgb"
    
    # 包含所有 GT 信息的 JSON (用作排除清单)
    labeled_json_path = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json"
    
    # 飞轮工作空间的新 RGB 根目录
    target_rgb_dir = "/data/ZS/flywheel_dataset/1_paint_rgb"
    
    # 只需要处理的条纹光数据集 (显式过滤掉 ap_phase012 和 ap_phase123)
    target_projects = [
        # "stripe_phase012", 
        "stripe_phase123"
    ]
    
    # 光源目录结构
    light_sources = [
        "16col", 
        "16row", 
        "32col", 
        "32row"
    ]
    
    # ================= 执行主函数 =================
    extract_unlabeled_images(
        source_rgb_dir=source_rgb_dir,
        labeled_json_path=labeled_json_path,
        target_rgb_dir=target_rgb_dir,
        target_projects=target_projects,
        light_sources=light_sources
    )