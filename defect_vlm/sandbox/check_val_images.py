"""
判断 /data/ZS/v11_input/datasets/col3/train_1/images 中的图像是否都在 
/data/ZS/defect_dataset/1_paint_rgb/stripe_phase012/images/16col 中出现过
"""
import os
from pathlib import Path

def check_images_exist(dir1, dir2):
    """
    判断dir1中的图像文件是否都在dir2中存在
    """
    # 获取两个目录中的文件名
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    
    # 找出只在dir1中存在的文件
    only_in_dir1 = files_dir1 - files_dir2
    
    # 找出两个目录都有的文件
    common_files = files_dir1 & files_dir2
    
    print(f"目录1: {dir1}")
    print(f"目录2: {dir2}")
    print("-" * 50)
    print(f"目录1中的文件数量: {len(files_dir1)}")
    print(f"目录2中的文件数量: {len(files_dir2)}")
    print(f"共同存在的文件数量: {len(common_files)}")
    
    if not only_in_dir1:
        print("\n✅ 目录1中的所有图像都在目录2中出现过！")
        return True
    else:
        print(f"\n❌ 目录1中有 {len(only_in_dir1)} 个文件不在目录2中：")
        # 列出不在目录2中的文件（最多显示10个）
        for i, filename in enumerate(sorted(only_in_dir1)[:10]):
            print(f"  {i+1}. {filename}")
        if len(only_in_dir1) > 10:
            print(f"  ... 还有 {len(only_in_dir1)-10} 个文件未显示")
        return False

# 设置目录路径
dir1 = "/data/ZS/v11_input/datasets/col3/train_1/images"
dir2 = "/data/ZS/defect_dataset/1_paint_rgb/stripe_phase012/images/32col"

# 执行检查
check_images_exist(dir1, dir2)
