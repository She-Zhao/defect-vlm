"""
生成 Type ③ 数据：纠错样本 (Rectification Samples) - 修正版
修正点：
1. 逻辑翻转：图像 -> 缺陷 -> 随机生成一次假标签 -> 遍历4个光源。
2. 保证同一缺陷在4个光源下的 prior_label 一致。
3. 保证4个光源文件都存在才保存，避免拼接时缺图。
"""
import json
import cv2
import numpy as np
import os
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import gc

# 设置随机种子
random.seed(42)
np.random.seed(42)

# === 全局配置 ===
REQUIRED_LIGHTS = ['16col', '16row', '32col', '32row']

# ================= 复用辅助函数 (保持不变) =================
def get_dynamic_context_ratio(bbox_size: float) -> float:
    sizes = [20, 100]       
    ratios = [2.0, 0.5]      
    return float(np.interp(bbox_size, sizes, ratios))

def get_region_proposal(image: np.ndarray, bbox: list) -> np.ndarray:
    """根据bbox截取图像局部"""
    if image is None: return None
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    
    h_context_ratio = get_dynamic_context_ratio(h)
    w_context_ratio = get_dynamic_context_ratio(w)
        
    cx, cy = x + w/2, y + h/2
    new_w = w * (1 + w_context_ratio)
    new_h = h * (1 + h_context_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
    
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

# ================= 主逻辑 =================

def main(
    data_root: Path, 
    dataset_name: str, 
    split: str
    ):
    # 路径配置
    raw_dir = data_root / '1_paint_rgb' / dataset_name
    json_path = raw_dir / 'labels' / f'{split}.json'
    image_root_dir = raw_dir / 'images'

    # 输出路径
    output_base_dir = data_root / '2_paint_bbox' / f'{dataset_name}_gt_rectification'
    save_img_dir = output_base_dir / 'images' / split
    save_label_dir = output_base_dir / 'labels'
    
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 获取所有缺陷类别名称
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    all_defect_names = [name for name in categories.values() if name.lower() != 'background']
    
    if len(all_defect_names) < 2:
        print("警告: 缺陷类别不足 2 个，无法构造类别混淆样本！")
        return

    # 建立索引
    img_id_to_info = {img['id']: img for img in data['images']}
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"开始生成纠错样本 (Rectification)...")

    metadata_list = []
    # 如果希望 ID 全局独立，可以设大一点，或者从0开始
    global_id_counter = 0 
    cnt = 0

    # 遍历每一张物理图像
    for image_id, anns in tqdm(img_id_to_anns.items(), desc=f"Processing {split}"):
        if not anns: continue # 如果是无缺陷图，跳过

        image_info = img_id_to_info.get(image_id)
        if not image_info: continue
            
        file_name = image_info['file_name']
        file_stem = Path(file_name).stem

        # --- 第一步：预加载该图的4个光源图像 ---
        light_images = {}
        load_success = True
        for light in REQUIRED_LIGHTS:
            path = image_root_dir / light / file_name
            if not path.exists():
                load_success = False
                break
            img = cv2.imread(str(path))
            if img is None:
                load_success = False
                break
            light_images[light] = img
        
        if not load_success:
            continue # 如果缺光源，整张图放弃

        # --- 第二步：遍历该图中的每一个缺陷 ---
        for idx, ann in enumerate(anns):
            gt_bbox = ann['bbox']
            true_cat_id = ann['category_id']
            true_label = categories.get(true_cat_id, 'unknown')

            # 确定 Prior Label (一个缺陷只随机一次，保证4个光源一致)
            possible_fakes = [c for c in all_defect_names if c != true_label]
            if not possible_fakes: continue
            fake_label = random.choice(possible_fakes)

            # --- 第三步：分别裁剪4个光源 ---
            temp_crops = {}
            crop_success = True
            
            for light in REQUIRED_LIGHTS:
                crop = get_region_proposal(light_images[light], gt_bbox)
                if crop is None or crop.size == 0:
                    crop_success = False
                    break
                temp_crops[light] = crop
            
            if not crop_success: continue

            # --- 第四步：保存 ---
            for light in REQUIRED_LIGHTS:
                # 命名: 原名_光源_rect_真_as_假_ID.png
                save_name = f"{file_stem}_{light}_rect_{true_label}_as_{fake_label}_{global_id_counter}.png"
                save_path = save_img_dir / save_name
                
                cv2.imwrite(str(save_path), temp_crops[light])

                try:
                    rel_original_path = (image_root_dir / light / file_name).relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError:
                    continue

                metadata = {
                    "id": global_id_counter,
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "bbox": gt_bbox,
                    "label": true_label,        # 真相
                    "prior_label": fake_label,  # 误导标签 (4光源一致)
                    "light_source": light,
                    "sample_type": "rectification"
                }
                metadata_list.append(metadata)
                
                # ID 自增 (每个小图一个ID)
                global_id_counter += 1

        # 内存释放
        del light_images
        cnt += 1
        if cnt % 20 == 0: gc.collect()

    # 保存 JSON
    out_json_path = save_label_dir / f'{split}.json'
    print(f"纠错样本元数据已保存至: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    PROJECT_NAME = 'stripe_phase123' 

    main(DATA_ROOT, PROJECT_NAME, 'train')
    main(DATA_ROOT, PROJECT_NAME, 'val')
