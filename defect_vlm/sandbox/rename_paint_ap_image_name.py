"""
将绝对相位图数据集的图像重命名
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def rename(old_name: str) -> str:
    """
    转换文件名格式：
    旧：73_2741_1748_2132_300_300_0_2432_2048
    新：00_0073_002741_1748_2132_300_300
    """
    parts = old_name.split('_')
    part1, part2, part3, part4, part5, part6, _, _, _ = parts
    part3, part4 = part3.zfill(4), part4.zfill(4)
    
    # 转换第一部分：补零到4位
    new_part1 = part1.zfill(4)  # 73 -> 0073
    # 转换第二部分：补零到6位  
    new_part2 = part2.zfill(6)  # 2741 -> 002741
    
    new_name = f"00_{new_part1}_{new_part2}_{part3}_{part4}_{part5}_{part6}.png"
    return new_name


root_dir = Path('/data/ZS/defect_dataset/0_defect_dataset_raw/paint_ap/images/ap')
image_names = [f.name for f in root_dir.glob('*.png')]

for image_name in tqdm(image_names, desc='Processing'):
    old_image_path = str(root_dir / image_name)
    new_image_name = rename(image_name)
    new_image_path = str(root_dir / new_image_name)
    
    # import pdb; pdb.set_trace()
    shutil.move(old_image_path, new_image_path)
