"""
根据原始图像和coco标签, 从标注的数据集中获取region proposal的正例
引入 Bbox Jitter (抖动) 机制，模拟 YOLO 预测框的不确定性，增强 VLM 抗干扰能力
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

def compute_iou(box1, box2):
    """计算两个 [x, y, w, h] 格式 bounding box 的 IoU"""
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
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def generate_jittered_bbox(gt_bbox, image_w, image_h, target_iou_min, target_iou_max, max_attempts=100):
    """通过随机平移和缩放生成模拟 YOLO 的抖动框"""
    x, y, w, h = gt_bbox
    
    for _ in range(max_attempts):
        scale_w = random.uniform(0.7, 1.4)
        scale_h = random.uniform(0.7, 1.4)
        new_w = w * scale_w
        new_h = h * scale_h
        
        dx = random.uniform(-0.3, 0.3) * w
        dy = random.uniform(-0.3, 0.3) * h
        
        cx, cy = x + w / 2, y + h / 2
        new_cx, new_cy = cx + dx, cy + dy
        new_x = new_cx - new_w / 2
        new_y = new_cy - new_h / 2
        
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
    sizes = [20, 100]       
    ratios = [2, 0.5]       
    return float(np.interp(bbox_size, sizes, ratios))

def get_region_proposal(image: np.ndarray, bbox: list, bbox_format='coco', fixed_context_ratio: float = None) -> np.ndarray:
    image_h, image_w = image.shape[:2]
    
    if bbox_format == 'coco':
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
    elif bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
    else:
        raise ValueError(f"不支持的格式: {bbox_format}")
    
    if fixed_context_ratio is not None:
        h_context_ratio = w_context_ratio = fixed_context_ratio
    else:
        h_context_ratio = get_dynamic_context_ratio(h)
        w_context_ratio = get_dynamic_context_ratio(w)

    new_w = w * (1 + w_context_ratio)
    new_h = h * (1 + h_context_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
      
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

def process_dataset(json_path: Path, image_root_dir: Path, save_img_dir: Path, save_label_dir: Path, split: str, light_sources: list, data_root: Path):
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_id_to_info = {img['id']: img for img in data['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"开始处理 {len(img_id_to_anns)} 张含有标注的图像 (修复版: 保证同一缺陷4光源抖动一致)...")

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

        # --- 核心修复 1：提前预加载该图像的 4 个光源 ---
        light_images = {}
        load_success = True
        for light in light_sources:
            original_img_path = image_root_dir / light / file_name
            if not original_img_path.exists():
                load_success = False; break
            raw_image = cv2.imread(str(original_img_path))
            if raw_image is None:
                load_success = False; break
            light_images[light] = raw_image
            
        if not load_success: continue

        # --- 核心修复 2：以缺陷为主循环，一次抖动，应用四次 ---
        for i, ann in enumerate(anns):
            gt_bbox = ann['bbox']
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id, 'unknown')
            
            # 【生成抖动框】(针对当前缺陷只执行一次)
            if random.random() < 0.5:
                target_iou_min, target_iou_max = 0.5, 0.7 
            else:
                target_iou_min, target_iou_max = 0.75, 0.9
            
            jittered_bbox, actual_iou = generate_jittered_bbox(
                gt_bbox, image_w, image_h, target_iou_min, target_iou_max
            )
            
            # 【分别裁剪并保存 4 个光源】(强制使用同一个 jittered_bbox)
            temp_crops = {}
            crop_success = True
            for light in light_sources:
                crop = get_region_proposal(light_images[light], jittered_bbox)
                if crop is None or crop.size == 0:
                    crop_success = False; break
                temp_crops[light] = crop
                
            if not crop_success: continue
            
            # 写入图片与元数据
            for light in light_sources:
                save_name = f"{file_stem}_{light}_{cat_name}_{i}.png"
                save_path = save_img_dir / save_name
                cv2.imwrite(str(save_path), temp_crops[light])

                try:
                    rel_original_path = (image_root_dir / light / file_name).relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError as e:
                    continue

                metadata = {
                    "id": global_id_counter,
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "gt_bbox": gt_bbox,
                    "yolo_jitter_bbox": jittered_bbox, 
                    "jitter_iou": round(actual_iou, 4), 
                    "label": cat_name,
                    "light_source": light
                }
                metadata_list.append(metadata)
                global_id_counter += 1

        del light_images
        cnt += 1
        if cnt % 50 == 0: gc.collect()

    out_json_path = save_label_dir / f'{split}.json'
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    # 数据集根目录 (用于计算存入 JSON 的相对路径，维持你的文件关联)
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    LIGHT_SOURCES = ['16col', '16row', '32col', '32row']        # 待处理的光源类型
    
    # 输入的 JSON 标签路径 (例如 train.json 或 val.json)
    INPUT_JSON_PATH = Path('/data/ZS/defect_dataset/1_paint_rgb/stripe_phase123/labels/val.json')
    
    # 输出的基准目录
    OUTPUT_BASE_DIR = Path('/data/ZS/defect_dataset/2_paint_bbox_general/sp123_gt_positive')


    # 自动推导其余路径 (极简优雅)
    split_name = INPUT_JSON_PATH.stem  # 自动提取出 'val' 或 'train'
    image_root_dir = INPUT_JSON_PATH.parent.parent / 'images' # 自动向上一级推导出 images 目录
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