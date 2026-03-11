"""
根据原始的伪rgb图像和截取的bbox图像，拼接得到2*2的图像（来自于yolo的预测）
输入：crop_yolo_preds_bbox.py输出的json文件（里面记载了原图像+bbox图像）
输出：拼接后的图像 + json文件（记录了元信息）
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# === 全局配置 ===
LIGHT_ORDER = ["16col", "16row", "32col", "32row"]
LIGHT_ORDER_MAP = {k: i for i, k in enumerate(LIGHT_ORDER)}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_bbox_on_image(image: np.ndarray, bbox: list, expand=2) -> np.ndarray:
    if image is None: return None
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    
    x_min = max(0, int(x - expand))
    y_min = max(0, int(y - expand))
    x_max = min(w_img, int(x + w + expand))
    y_max = min(h_img, int(y + h + expand))

    raw_image = image.copy()
    cv2.rectangle(raw_image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return raw_image

def letter_resize_bbox(image: np.ndarray, target_size=300) -> np.ndarray:
    if image is None: return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_h = int(scale * h)
    new_w = int(scale * w)
    
    resize_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resize_image
    
    return canvas

def composite_2x2_images(images: list, target_size=600) -> np.ndarray:
    cell_size = target_size // 2
    processed_images = []
    
    for img in images:
        if img is None:
            img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        if img.shape[0] != cell_size or img.shape[1] != cell_size:
            img = letter_resize_bbox(img, target_size=cell_size)
        processed_images.append(img)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:cell_size, :cell_size] = processed_images[0]      # TL: 16col
    canvas[:cell_size, cell_size:] = processed_images[1]      # TR: 16row
    canvas[cell_size:, :cell_size] = processed_images[2]      # BL: 32col
    canvas[cell_size:, cell_size:] = processed_images[3]      # BR: 32row

    return canvas

def get_group_key(item: dict) -> str:
    path = item['original_image_path']
    light = item['light_source']
    bbox_str = "_".join(map(str, item['bbox']))
    path_key = path.replace(f"/{light}/", "/LIGHT/") 
    return f"{path_key}@{bbox_str}"

def process_composite_inference(input_json: str, data_root: str, output_img_dir: str, output_json_path: str, start_id: int = 1000001):
    print(f"📖 正在加载元数据: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 确保输出目录存在
    ensure_dir(output_img_dir)
    ensure_dir(os.path.dirname(output_json_path))

    # 按照图像和 bbox 分组
    groups = defaultdict(list)
    for item in raw_data:
        key = get_group_key(item)
        groups[key].append(item)

    processed_data_list = []
    current_id = start_id
    
    for key, items in tqdm(groups.items(), desc="拼接图像"):
        if len(items) != 4:
            continue
        
        try:
            items.sort(key=lambda x: LIGHT_ORDER_MAP[x['light_source']])
        except KeyError:
            continue

        base_item = items[0]
        bbox = base_item['bbox']
        prior_label = base_item.get('prior_label', 'unknown')
        confidence = base_item.get('confidence', 0.0)
        model_source = base_item.get('model_source', 'unknown')

        original_imgs_draw = []
        crop_imgs_resize = []
        orig_paths_record = []
        crop_paths_record = []
        
        valid_group = True
        
        for it in items:
            orig_abs_path = os.path.join(data_root, it['original_image_path'])
            crop_abs_path = os.path.join(data_root, it['crop_image_path'])
            
            orig_img = cv2.imread(orig_abs_path)
            crop_img = cv2.imread(crop_abs_path)
            
            if orig_img is None or crop_img is None:
                valid_group = False
                break
                
            orig_with_box = draw_bbox_on_image(orig_img, bbox)
            crop_resized = letter_resize_bbox(crop_img, target_size=300) 
            
            original_imgs_draw.append(orig_with_box)
            crop_imgs_resize.append(crop_resized)
            
            orig_paths_record.append(it['original_image_path'])
            crop_paths_record.append(it['crop_image_path'])
            
        if not valid_group:
            continue
            
        final_global_img = composite_2x2_images(original_imgs_draw, target_size=600)
        final_local_img = composite_2x2_images(crop_imgs_resize, target_size=600)
        
        global_filename = f"global_{current_id}.png"
        local_filename = f"local_{current_id}.png"
        
        global_save_path = os.path.join(output_img_dir, global_filename)
        local_save_path = os.path.join(output_img_dir, local_filename)
        
        cv2.imwrite(global_save_path, final_global_img)
        cv2.imwrite(local_save_path, final_local_img)
        
        # 自动计算相对路径 (基于 DATA_ROOT)，再也不需要手工拼接 split 或 project 字符串
        try:
            rel_global_path = Path(global_save_path).relative_to(Path(data_root)).as_posix()
            rel_local_path = Path(local_save_path).relative_to(Path(data_root)).as_posix()
        except ValueError:
            rel_global_path = global_save_path
            rel_local_path = local_save_path
        
        out_item = {
            "id": current_id,
            "composite_global_path": rel_global_path,
            "composite_local_path": rel_local_path,
            "bbox": bbox,
            "sample_type": "inference",      
            "label": "unknown",              
            "prior_label": prior_label,      
            "confidence": confidence,        
            "model_source": model_source,    
            "light_source_order": LIGHT_ORDER,
            "original_image_paths": [p.replace("\\", "/") for p in orig_paths_record],
            "original_crop_paths": [p.replace("\\", "/") for p in crop_paths_record]
        }
        
        processed_data_list.append(out_item)
        current_id += 1

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data_list, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 拼接完成！成功生成 {len(processed_data_list)} 组 2x2 图像。")
    print(f"📄 JSON 保存至: {output_json_path}")


if __name__ == "__main__":
    DATA_ROOT = "/data/ZS/defect_dataset"
    
    # 上一步抠图脚本生成的 JSON 文件路径
    INPUT_JSON = "/data/ZS/defect_dataset/10_yolo_preds_bbox/stripe_phase012/labels/val_0p1.json"
    
    # 本次拼接图像的具体存放文件夹
    OUTPUT_IMAGE_DIR = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/images/val_0p1"
    
    # 本次生成的终极 JSON 文件的具体路径
    OUTPUT_JSON_PATH = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p1.json"

    process_composite_inference(
        input_json=INPUT_JSON,
        data_root=DATA_ROOT,
        output_img_dir=OUTPUT_IMAGE_DIR,
        output_json_path=OUTPUT_JSON_PATH,
        start_id=1000001
    )
