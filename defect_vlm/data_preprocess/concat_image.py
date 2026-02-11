"""
将paint_stripe中的灰度图像图像沿通道方向进行拼接，得到伪BGR图像
"""
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import pdb
import argparse

def concat_image(image_dir: str, phase_names: list[str], save_dir: str) -> None:
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    ch0_name, ch1_name, ch2_name = phase_names[0], phase_names[1], phase_names[2]
    ch0_dir = image_dir / ch0_name
    ch0_paths = list(ch0_dir.glob('*.png'))
    
    save_dir.mkdir(parents=True, exist_ok=True)
    for ch0_path in tqdm(ch0_paths, desc='Processing: '):
        try:
            ch0_image = cv2.imread(str(ch0_path), cv2.IMREAD_GRAYSCALE)
            ch1_path = image_dir / ch1_name / ch0_path.name
            ch2_path = image_dir / ch2_name / ch0_path.name
            
            ch1_image = cv2.imread(str(ch1_path), cv2.IMREAD_GRAYSCALE)
            ch2_image = cv2.imread(str(ch2_path), cv2.IMREAD_GRAYSCALE)
            
            rgb_image = np.stack([ch0_image, ch1_image, ch2_image], axis=2)
            
            save_path = str(save_dir / ch0_path.name)
            cv2.imwrite(save_path, rgb_image)       # 注意这里保存的顺序是BGR
        except Exception as e:
            print(f"处理 {ch0_path.name} 时报错： {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', type=str, default='/data/ZS/defect_dataset/0_defect_dataset_raw/paint_ap/images',
        help = '原始图像所在的根目录'
    )
    
    parser.add_argument('--phase_names', default=['sin1', 'sin2', 'sin3'],
        type=str, nargs='+', help='三个通道相位图的名称'
    )
    
    parser.add_argument('--save_dir', default='/data/ZS/defect_dataset/1_paint_rgb/ap_phase123/images',
        type=str, help='拼接后的图像所在的文件夹'
    )
                     
    args = parser.parse_args()
    
    concat_image(args.image_dir, args.phase_names, args.save_dir)
        
        