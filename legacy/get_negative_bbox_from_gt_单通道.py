"""
根据原始图像和coco标签, 从标注的数据集中获取region proposal的负样本, 并将region proposal图像存储起来
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

# 设置随机种子，保证负样本生成的可复现性
random.seed(42)
np.random.seed(42)

def compute_iou(box1, box2):
    """
    计算两个矩形框的IoU (Intersection over Union)
    box格式: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = w1 * h1
    b2_area = w2 * h2

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0: return 0.0
        
    return inter_area / union_area

def generate_random_bbox(image_shape, gt_bboxes, ref_sizes, max_trials=50):
    """
    尝试生成一个合法的、不与GT重叠的随机BBox
    """
    h_img, w_img = image_shape
    
    for _ in range(max_trials):
        # 1. 随机确定尺寸 (从参考尺寸池中采样，并加扰动)
        if not ref_sizes:
            base_w, base_h = 100, 100
        else:
            base_w, base_h = random.choice(ref_sizes)
            
        scale = random.uniform(0.8, 1.2)
        w = int(base_w * scale)
        h = int(base_h * scale)
        
        w = max(10, min(w, w_img - 1))
        h = max(10, min(h, h_img - 1))

        # 2. 随机确定位置
        if w_img - w <= 0 or h_img - h <= 0:
            continue
            
        x = random.randint(0, w_img - w)
        y = random.randint(0, h_img - h)
        
        candidate_box = [x, y, w, h]

        # 3. 检查与所有GT的IoU
        has_overlap = False
        for gt_box in gt_bboxes:
            # 严格模式：IoU > 0.0 即视为重叠，确保背景纯净
            if compute_iou(candidate_box, gt_box) > 0.0:
                has_overlap = True
                break
        
        if not has_overlap:
            return candidate_box

    return None

def get_region_proposal(image: np.ndarray, bbox: list, context_ratio: float = 0.4) -> np.ndarray:
    """根据bbox截取图像局部"""
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2

    new_w = w * (1 + context_ratio)
    new_h = h * (1 + context_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
      
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

def main(
    data_root: Path, 
    dataset_name: str, 
    split: str, 
    light_sources: list, 
    samples_per_image: int = 2,
    ):
    """
    Args:
        data_root: 数据集根目录锚点 (/data/ZS/defect_dataset)
        dataset_name: 数据集名称 (paint_stripe)
        split: 数据划分 (train 或 val)
        samples_per_image: 每张原图生成多少个负样本
    """
    # ================= 路径配置 =================
    # 输入路径
    raw_dir = data_root / '0_defect_dataset_raw' / dataset_name
    json_path = raw_dir / 'labels' / f'{split}.json'
    image_root_dir = raw_dir / 'images'

    # 输出路径 (注意这里是 _gt_negative)
    output_base_dir = data_root / '1_defect_dataset_bbox' / f'{dataset_name}_gt_negative'
    # 严格遵循你的要求: .../images/train
    save_img_dir = output_base_dir / 'images' / split
    save_label_dir = output_base_dir / 'labels'
    
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)
    
    # ================= 加载数据与预统计 =================
    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_id_to_info = {img['id']: img for img in data['images']}
    
    # 统计所有缺陷尺寸，用于生成逼真的负样本框
    all_defect_sizes = []
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)
        w, h = ann['bbox'][2], ann['bbox'][3]
        all_defect_sizes.append((w, h))

    # 随机抽取一部分尺寸作为参考池，避免计算过慢
    if len(all_defect_sizes) > 2000:
        ref_size_pool = random.sample(all_defect_sizes, 2000)
    else:
        ref_size_pool = all_defect_sizes

    print(f"开始处理 {len(img_id_to_anns)} 张图像，目标生成负样本...")

    # ================= 处理循环 =================
    metadata_list = []
    # 使用独立的ID计数器，或者接着正样本的ID继续（如果分开存文件，从0开始没问题）
    global_id_counter = 0 
    cnt = 0

    for image_id, anns in tqdm(img_id_to_anns.items(), desc=f"Processing Neg {split}"):
        image_info = img_id_to_info.get(image_id)
        if not image_info: continue
            
        file_name = image_info['file_name']
        file_stem = Path(file_name).stem
        
        # 获取该图所有GT框
        current_gt_bboxes = [ann['bbox'] for ann in anns]

        for light in light_sources:
            original_img_path = image_root_dir / light / file_name
            
            if not original_img_path.exists(): continue
            
            raw_image = cv2.imread(str(original_img_path), cv2.IMREAD_GRAYSCALE)
            if raw_image is None: continue
            
            img_h, img_w = raw_image.shape[:2]

            # 生成 N 个负样本
            for i in range(samples_per_image):
                # 随机生成不重叠的 bbox
                rand_bbox = generate_random_bbox(
                    image_shape=(img_h, img_w),
                    gt_bboxes=current_gt_bboxes,
                    ref_sizes=ref_size_pool,
                    max_trials=50
                )

                if rand_bbox is None:
                    continue

                # 裁剪
                crop = get_region_proposal(raw_image, rand_bbox, context_ratio=0.4)
                if crop is None or crop.size == 0: continue
                
                # 构造文件名: 原文件名_光源_background_序号.png
                save_name = f"{file_stem}_{light}_background_{i}.png"
                save_path = save_img_dir / save_name
                
                cv2.imwrite(str(save_path), crop)

                # ================= 记录元数据 =================
                try:
                    rel_original_path = original_img_path.relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError:
                    continue

                metadata = {
                    "id": global_id_counter,
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "bbox": rand_bbox,       # 这里的bbox是随机生成的背景框
                    "label": "background",   # 统一标签
                    "light_source": light
                }
                metadata_list.append(metadata)
                global_id_counter += 1

            del raw_image
        
        cnt += 1
        if cnt % 50 == 0:
            gc.collect()

    # ================= 保存JSON =================
    out_json_path = save_label_dir / f'{split}.json'
    print(f"正在保存负样本元数据到: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    
    # 配置参数
    dataset_name = 'paint_ap'
    # light_sources = ['16col0','16col1','16col2', '16row0', '16row1', '16row2', 
    #                  '32col0', '32col1', '32col2', '32row0', '32row1', '32row2']
    light_sources = ['sin0','sin1', 'sin2', 'sin3'] 

    # 分别处理 train 和 val
    main(DATA_ROOT, dataset_name, 'train', light_sources, samples_per_image=8)
    main(DATA_ROOT, dataset_name, 'val', light_sources, samples_per_image=8)