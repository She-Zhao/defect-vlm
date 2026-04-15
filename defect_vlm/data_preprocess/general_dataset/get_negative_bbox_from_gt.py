"""
生成 Type ② 数据：多光源对齐的负样本 (Negative Samples Generation)
引入 Hard Negative Mining 机制：
- Type A: 纯背景 (IoU = 0)
- Type B: 轻度截断的边缘误检 (IoU = 0.1 ~ 0.3)
- Type C: 中度截断的难例误检 (IoU = 0.3 ~ 0.5)
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

random.seed(42)
np.random.seed(42)

# 缺陷类别映射 (模拟 YOLO 乱猜的 Prior Label)
DEFECT_MAP = ('breakage', 'inclusion', 'crater', 'bulge', 'run', 'scratch')

def compute_iou(box1, box2):
    """计算IoU: box=[x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return 0.0 if union_area <= 0 else inter_area / union_area

def generate_pure_bg_bbox(image_w, image_h, gt_bboxes, ref_sizes, max_trials=50):
    """生成纯背景框 (与任何 GT 的 IoU = 0)"""
    for _ in range(max_trials):
        base_w, base_h = random.choice(ref_sizes) if ref_sizes else (100, 100)
        scale = random.uniform(0.8, 1.2)
        w = max(10, min(int(base_w * scale), image_w - 1))
        h = max(10, min(int(base_h * scale), image_h - 1))

        if image_w - w <= 0 or image_h - h <= 0: continue
            
        x = random.randint(0, image_w - w)
        y = random.randint(0, image_h - h)
        candidate_box = [x, y, w, h]

        has_overlap = any(compute_iou(candidate_box, gt) > 0.0 for gt in gt_bboxes)
        if not has_overlap:
            return candidate_box, 0.0
    return None, 0.0

def generate_hard_negative_bbox(target_gt, all_gts, image_w, image_h, target_iou_min, target_iou_max, max_attempts=150):
    """通过大尺度抖动 GT，生成指定 IoU 范围的负样本框"""
    x, y, w, h = target_gt
    
    for _ in range(max_attempts):
        # 负样本往往是被放得很大或者偏离很远，因此缩放和平移的容差给得更大
        scale_w = random.uniform(0.5, 2.0)
        scale_h = random.uniform(0.5, 2.0)
        new_w, new_h = w * scale_w, h * scale_h
        
        dx = random.uniform(-0.8, 0.8) * w
        dy = random.uniform(-0.8, 0.8) * h
        
        cx, cy = x + w / 2, y + h / 2
        new_cx, new_cy = cx + dx, cy + dy
        new_x, new_y = new_cx - new_w / 2, new_cy - new_h / 2
        
        # 边界截断
        new_x = max(0, min(new_x, image_w - 1))
        new_y = max(0, min(new_y, image_h - 1))
        new_w = max(1, min(new_w, image_w - new_x))
        new_h = max(1, min(new_h, image_h - new_y))
        
        candidate_box = [new_x, new_y, new_w, new_h]
        iou_with_target = compute_iou(target_gt, candidate_box)
        
        # 1. 必须落在我们期望的负样本 IoU 区间内
        if target_iou_min <= iou_with_target <= target_iou_max:
            # 2. 安全防碰撞校验：保证这个框不仅对 target_gt 是负样本，对图上**其他所有缺陷**也必须是负样本 (<0.5)
            max_iou_all = max([compute_iou(candidate_box, gt) for gt in all_gts])
            if max_iou_all < 0.5:
                return candidate_box, iou_with_target
                
    return None, 0.0

def get_region_proposal(image: np.ndarray, bbox: list, fixed_context_ratio: float = None) -> np.ndarray:
    if image is None: return None
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    
    if fixed_context_ratio is not None:
        h_ratio = w_ratio = fixed_context_ratio
    else:
        h_ratio = w_ratio = 0.5 
        
    cx, cy = x + w/2, y + h/2
    new_w, new_h = w * (1 + w_ratio), h * (1 + h_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
    
    if x_max <= x_min or y_max <= y_min: return None
    return image[y_min:y_max, x_min:x_max].copy()

def process_negative_dataset(
    json_path: Path, 
    image_root_dir: Path, 
    save_img_dir: Path, 
    save_label_dir: Path, 
    split: str, 
    light_sources: list, 
    data_root: Path,
    samples_per_image: int,
    iou_ratios: dict
):
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_id_to_info = {img['id']: img for img in data['images']}
    img_id_to_anns = defaultdict(list)
    all_defect_sizes = []
    
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)
        w, h = ann['bbox'][2], ann['bbox'][3]
        all_defect_sizes.append((w, h))

    ref_size_pool = random.sample(all_defect_sizes, min(2000, len(all_defect_sizes)))

    print(f"开始生成负样本 (引入难例 IoU 比例控制: {iou_ratios})...")

    metadata_list = []
    global_id_counter = 0
    cnt = 0

    for image_id, image_info in tqdm(img_id_to_info.items(), desc=f"Generating Neg {split}"):
        file_name = image_info['file_name'] 
        file_stem = Path(file_name).stem
        image_w, image_h = image_info['width'], image_info['height']
        
        current_gt_bboxes = [ann['bbox'] for ann in img_id_to_anns[image_id]]

        # --- 1. 预加载 4 个光源 ---
        light_images = {}
        load_success = True
        for light in light_sources:
            img_path = image_root_dir / light / file_name
            if not img_path.exists():
                load_success = False; break
            img = cv2.imread(str(img_path))
            if img is None:
                load_success = False; break
            light_images[light] = img

        if not load_success: continue

        # --- 2. 生成 N 组负样本 ---
        for _ in range(samples_per_image):
            # 决定当前生成的负样本类别
            rand_val = random.random()
            
            # 如果这张图本身就没有缺陷，只能生成纯背景
            if len(current_gt_bboxes) == 0:
                target_type = "pure_bg"
            elif rand_val < iou_ratios["iou_0"]:
                target_type = "pure_bg"
            elif rand_val < iou_ratios["iou_0"] + iou_ratios["iou_0.1_0.3"]:
                target_type = "hard_0.1_0.3"
            else:
                target_type = "hard_0.3_0.5"

            rand_bbox, actual_iou = None, 0.0

            # 核心策略：依据 target_type 去生成对应的框
            if target_type == "pure_bg":
                rand_bbox, actual_iou = generate_pure_bg_bbox(image_w, image_h, current_gt_bboxes, ref_size_pool)
            else:
                target_gt = random.choice(current_gt_bboxes)
                if target_type == "hard_0.1_0.3":
                    rand_bbox, actual_iou = generate_hard_negative_bbox(target_gt, current_gt_bboxes, image_w, image_h, 0.1, 0.3)
                else: # hard_0.3_0.5
                    rand_bbox, actual_iou = generate_hard_negative_bbox(target_gt, current_gt_bboxes, image_w, image_h, 0.3, 0.5)
                
                # 如果受限于空间位置死活摇不出 hard negative，降级为纯背景
                if rand_bbox is None:
                    target_type = "pure_bg_fallback"
                    rand_bbox, actual_iou = generate_pure_bg_bbox(image_w, image_h, current_gt_bboxes, ref_size_pool)

            if rand_bbox is None: continue 

            # 随机一个假标签
            prior_label = random.choice(DEFECT_MAP)

            # --- 3. 裁剪 4 个光源 (强制共享 Bbox) ---
            temp_crops = {}
            crop_success = True
            for light in light_sources:
                crop = get_region_proposal(light_images[light], rand_bbox, fixed_context_ratio=0.4)
                if crop is None or crop.size == 0:
                    crop_success = False; break
                temp_crops[light] = crop
            
            if not crop_success: continue

            # --- 4. 保存 ---
            for light in light_sources:
                save_name = f"{file_stem}_{light}_neg_{global_id_counter}.png"
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
                    "bbox": rand_bbox,        
                    "label": "background",    # 负样本的真相永远是 background
                    "prior_label": prior_label, 
                    "neg_type": target_type,  # 记录是纯背景还是难例
                    "overlap_iou": round(actual_iou, 4), # 记录真实的覆盖率
                    "light_source": light,
                    "sample_type": "negative" 
                }
                metadata_list.append(metadata)
                global_id_counter += 1 

        del light_images
        cnt += 1
        if cnt % 20 == 0: gc.collect()

    out_json_path = save_label_dir / f'{split}.json'
    print(f"正在保存带有难例信息的负样本元数据到: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    # ================= 自由配置区 =================
    # 待处理的光源类型
    LIGHT_SOURCES = ['16col', '16row', '32col', '32row'] 
    
    # 【输入】指定是处理 train 还是 val 的 JSON
    INPUT_JSON_PATH = Path('/data/ZS/defect_dataset/1_paint_rgb/stripe_phase012/labels/val.json')
    
    # 【输出】保存负样本的基准目录
    OUTPUT_BASE_DIR = Path('/data/ZS/defect_dataset/2_paint_bbox_general/sp012_gt_negative')

    # 数据集根目录
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    
    # 【核心参数】：控制生成的负样本数量和难例比例
    SAMPLES_PER_IMAGE = 2 
    
    # 负样本比例配置：三者加起来必须等于 1.0
    # 比如：50% 是纯背景，25% 稍微切到边缘，25% 大面积切到边缘但不足一半
    IOU_RATIOS = {
        "iou_0": 0.6,          
        "iou_0.1_0.3": 0.3,    
        "iou_0.3_0.5": 0.1     
    }
    # ==============================================

    # 自动推导路径
    split_name = INPUT_JSON_PATH.stem  
    image_root_dir = INPUT_JSON_PATH.parent.parent / 'images' 
    save_img_dir = OUTPUT_BASE_DIR / 'images' / split_name
    save_label_dir = OUTPUT_BASE_DIR / 'labels'
    
    # 执行处理
    process_negative_dataset(
        json_path=INPUT_JSON_PATH,
        image_root_dir=image_root_dir,
        save_img_dir=save_img_dir,
        save_label_dir=save_label_dir,
        split=split_name,
        light_sources=LIGHT_SOURCES,
        data_root=DATA_ROOT,
        samples_per_image=SAMPLES_PER_IMAGE,
        iou_ratios=IOU_RATIOS
    )