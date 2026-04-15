"""
生成 Type ③ 数据：纠错样本 (Rectification Samples) - 终极增强版
机制 1: 引入 Bbox Jitter，模拟 YOLO 预测框定位误差。
机制 2: 引入 Confusion Map，生成易混淆的 Hard Negative 先验提示。
机制 3: 保证同一个缺陷的 4 个光源图像使用完全相同的抖动框和相同的错误标签。
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

# 设置随机种子，保证实验可复现
random.seed(42)
np.random.seed(42)

# ================= 辅助函数 =================

def get_hard_prior_hint(true_label: str, all_labels: list, confusion_map: dict, hard_ratio: float = 0.8) -> str:
    """基于混淆字典生成错误先验标签"""
    other_labels = [label for label in all_labels if label != true_label]
    
    if random.random() < hard_ratio and true_label in confusion_map:
        hard_candidates = [label for label in confusion_map[true_label] if label != true_label]
        if hard_candidates:
            return random.choice(hard_candidates)
            
    return random.choice(other_labels)

def compute_iou(box1, box2):
    """计算 IoU"""
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area

def generate_jittered_bbox(gt_bbox, image_w, image_h, target_iou_min, target_iou_max, max_attempts=100):
    """生成抖动框"""
    x, y, w, h = gt_bbox
    for _ in range(max_attempts):
        scale_w = random.uniform(0.7, 1.4)
        scale_h = random.uniform(0.7, 1.4)
        new_w, new_h = w * scale_w, h * scale_h
        
        dx = random.uniform(-0.3, 0.3) * w
        dy = random.uniform(-0.3, 0.3) * h
        
        cx, cy = x + w / 2, y + h / 2
        new_cx, new_cy = cx + dx, cy + dy
        new_x, new_y = new_cx - new_w / 2, new_cy - new_h / 2
        
        new_x = max(0, min(new_x, image_w - 1))
        new_y = max(0, min(new_y, image_h - 1))
        new_w = max(1, min(new_w, image_w - new_x))
        new_h = max(1, min(new_h, image_h - new_y))
        
        jittered_box = [new_x, new_y, new_w, new_h]
        iou = compute_iou(gt_bbox, jittered_box)
        
        if target_iou_min <= iou <= target_iou_max:
            return jittered_box, iou
    return gt_bbox, 1.0

def get_dynamic_context_ratio(bbox_size: float) -> float:
    return float(np.interp(bbox_size, [20, 100], [2.0, 0.5]))

def get_region_proposal(image: np.ndarray, bbox: list) -> np.ndarray:
    if image is None: return None
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    
    h_ratio = get_dynamic_context_ratio(h)
    w_ratio = get_dynamic_context_ratio(w)
        
    cx, cy = x + w/2, y + h/2
    new_w, new_h = w * (1 + w_ratio), h * (1 + h_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
    
    if x_max <= x_min or y_max <= y_min: return None
    return image[y_min:y_max, x_min:x_max].copy()

# ================= 核心处理逻辑 =================

def process_dataset(json_path: Path, image_root_dir: Path, save_img_dir: Path, save_label_dir: Path, split: str, light_sources: list, data_root: Path):
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 解析所有可能的缺陷类别
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    all_defect_names = [name for name in categories.values() if name.lower() != 'background']
    if len(all_defect_names) < 2:
        print("⚠️ 警告: 缺陷类别不足 2 个，无法构造类别混淆样本！")
        return

    # 2. 建立索引
    img_id_to_info = {img['id']: img for img in data['images']}
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"开始生成纠错样本 (包含 Jitter 和 Hard Negative 挖掘)...")

    metadata_list = []
    global_id_counter = 0 
    cnt = 0

    for image_id, anns in tqdm(img_id_to_anns.items(), desc=f"Processing {split}"):
        if not anns: continue 
        
        image_info = img_id_to_info.get(image_id)
        if not image_info: continue
            
        file_name = image_info['file_name']
        file_stem = Path(file_name).stem
        image_w, image_h = image_info['width'], image_info['height']

        # --- 第一步：预加载该图的4个光源图像 ---
        light_images = {}
        load_success = True
        for light in light_sources:
            path = image_root_dir / light / file_name
            if not path.exists():
                load_success = False; break
            img = cv2.imread(str(path))
            if img is None:
                load_success = False; break
            light_images[light] = img
        
        if not load_success: continue 

        # --- 第二步：遍历该图中的每一个缺陷 ---
        for idx, ann in enumerate(anns):
            gt_bbox = ann['bbox']
            true_cat_id = ann['category_id']
            true_label = categories.get(true_cat_id, 'unknown')

            # 机制 2: 生成 Hard Negative 错误标签 (针对当前缺陷只生成一次)
            fake_label = get_hard_prior_hint(true_label, all_defect_names, CONFUSION_MAP, hard_ratio=0.8)

            # 机制 1: 生成 Bbox Jitter (针对当前缺陷只抖动一次)
            if random.random() < 0.5:
                target_iou_min, target_iou_max = 0.5, 0.7 
            else:
                target_iou_min, target_iou_max = 0.75, 0.9
                
            jittered_bbox, actual_iou = generate_jittered_bbox(gt_bbox, image_w, image_h, target_iou_min, target_iou_max)

            # --- 第三步：分别裁剪 4 个光源 (强制使用同一个 jittered_bbox) ---
            temp_crops = {}
            crop_success = True
            for light in light_sources:
                crop = get_region_proposal(light_images[light], jittered_bbox)
                if crop is None or crop.size == 0:
                    crop_success = False; break
                temp_crops[light] = crop
            
            if not crop_success: continue

            # --- 第四步：保存 ---
            for light in light_sources:
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
                    "gt_bbox": gt_bbox,
                    "yolo_jitter_bbox": jittered_bbox,
                    "jitter_iou": round(actual_iou, 4),
                    "label": true_label,        # 真相
                    "prior_label": fake_label,  # 误导标签 (4光源一致，且具备迷惑性)
                    "light_source": light,
                    "sample_type": "rectification"
                }
                metadata_list.append(metadata)
                global_id_counter += 1

        del light_images
        cnt += 1
        if cnt % 20 == 0: gc.collect()

    out_json_path = save_label_dir / f'{split}.json'
    print(f"纠错样本元数据已保存至: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    
    # ================= 混淆字典 (Confusion Map) =================
    # 键: 真实缺陷, 值: 最容易被误判成的缺陷列表
    CONFUSION_MAP = {
        "breakage": ["inclusion"],  
        "inclusion": ["breakage"],  
        "crater": ["inclusion", "bulge"], 
        "bulge": ["crater"],           
        "run": ["bulge"],          
        "scratch": ["breakage"]        
    }

    # ================= 自由路径配置区 =================
    # 待处理的光源类型
    LIGHT_SOURCES = ['16col', '16row', '32col', '32row'] 
    
    # 输入的 JSON 标签路径 (检查 train 还是 val)
    INPUT_JSON_PATH = Path('/data/ZS/defect_dataset/1_paint_rgb/stripe_phase123/labels/val.json')
    
    # 输出的基准目录
    OUTPUT_BASE_DIR = Path('/data/ZS/defect_dataset/2_paint_bbox_general/sp123_gt_rectification')

    # 数据集根目录 (用于计算存入 JSON 的相对路径，维持你的文件关联)
    DATA_ROOT = Path('/data/ZS/defect_dataset')




    # 自动推导其余路径
    split_name = INPUT_JSON_PATH.stem  
    image_root_dir = INPUT_JSON_PATH.parent.parent / 'images' 
    save_img_dir = OUTPUT_BASE_DIR / 'images' / split_name
    save_label_dir = OUTPUT_BASE_DIR / 'labels'
    
    # 执行处理
    process_dataset(
        json_path=INPUT_JSON_PATH,
        image_root_dir=image_root_dir,
        save_img_dir=save_img_dir,
        save_label_dir=save_label_dir,
        split=split_name,
        light_sources=LIGHT_SOURCES,
        data_root=DATA_ROOT
    )