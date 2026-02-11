"""
拼接2*2的全局图像和2*2的局部图像，并保存相应的json文件
输出位置：/data/ZS/defect_dataset/3_composite_images
"""

import os
import random
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Union, List, Tuple, Dict, Optional

# === 全局配置 ===
# 定义光源在 2x2 网格中的顺序
# 对应: Top-Left, Top-Right, Bottom-Left, Bottom-Right
random.seed(42)
LIGHT_ORDER = ["16col", "16row", "32col", "32row"]
LIGHT_ORDER_MAP = {k: i for i, k in enumerate(LIGHT_ORDER)}

def ensure_dir(path: str):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def draw_bbox_on_image(image: np.ndarray, bbox: Union[list, tuple], expand=2) -> np.ndarray:
    """
    在图像上绘制bbox红框
    bbox: [x, y, w, h]
    """
    if image is None: return None
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    
    x_min = max(0, int(x - expand))
    y_min = max(0, int(y - expand))
    x_max = min(w_img, int(x + w + expand))
    y_max = min(h_img, int(y + h + expand))

    pt1 = (x_min, y_min)
    pt2 = (x_max, y_max)
    
    # 复制图像避免修改原图
    raw_image = image.copy()
    # BGR 格式，红色为 (0, 0, 255)
    image_with_bbox = cv2.rectangle(raw_image, pt1, pt2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return image_with_bbox

def letter_resize_bbox(image: np.ndarray, target_size=300) -> np.ndarray:
    """保持比例缩放并填充黑边"""
    if image is None: return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_h = int(scale * h)
    new_w = int(scale * w)
    
    resize_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 计算居中位置
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resize_image
    
    return canvas

def composite_2x2_images(images: List[np.ndarray], target_size=600) -> np.ndarray:
    """
    将4张图拼成 2x2 网格
    """
    cell_size = target_size // 2
    processed_images = []
    
    for img in images:
        # 如果子图尺寸不是 cell_size，则进行 resize
        if img is None:
            img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        
        if img.shape[0] != cell_size or img.shape[1] != cell_size:
            img = letter_resize_bbox(img, target_size=cell_size)
        processed_images.append(img)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # TL, TR, BL, BR
    canvas[:cell_size, :cell_size] = processed_images[0]
    canvas[:cell_size, cell_size:] = processed_images[1]
    canvas[cell_size:, :cell_size] = processed_images[2]
    canvas[cell_size:, cell_size:] = processed_images[3]

    return canvas

def get_group_key(item: dict) -> str:
    """
    生成分组唯一键。
    逻辑：同一缺陷在不同光源下，除了路径中的光源名称不同，其他路径后缀和bbox应该是相同的。
    我们去除 path 中的光源字符串作为 Key。
    """
    path = item['original_image_path']
    light = item['light_source']
    bbox_str = "_".join(map(str, item['bbox']))
    
    # 将路径中的 light_source 替换为占位符，例如 '16col' -> 'LIGHT'
    # 假设路径结构相对固定，这里做一个简单的替换
    path_key = path.replace(f"/{light}/", "/LIGHT/") 
    
    # Key = 抽象路径 + bbox
    return f"{path_key}@{bbox_str}"

def process_dataset(
    json_path: str, 
    dataset_root: str, 
    output_root: str, 
    sample_type: str, 
    split: str,
    start_id: int = 1000001
):
    """
    主处理函数
    Args:
        json_path: 输入的 label JSON 路径
        dataset_root: 数据集根目录 (用于拼接相对路径)
        output_root: 输出根目录 (通常是 dataset_root/3_composite_images)
        sample_type: 'positive', 'negative', 'rectification'
        split: 'train' or 'val'
        start_id: 起始 ID
    """
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 1. 自动推断项目名称
    try:
        path_parts = Path(json_path).parts
        project_name = "unknown_project"
        
        # 优先查找包含 _gt_ 的目录名
        for part in path_parts:
            if "_gt_" in part:
                project_name = part  # <--- 修改点：直接使用完整目录名，不要 split
                break
        
        # 保底逻辑：如果没找到 _gt_，直接取 labels 文件夹的上一级目录名
        if project_name == "unknown_project" and "labels" in path_parts:
             idx = path_parts.index("labels")
             project_name = path_parts[idx - 1]
             
        print(f"Detected Project Name: {project_name}")
    except Exception:
        project_name = "default_project"

    # 2. 构建输出目录结构
    # e.g., output_root/stripe_phase012/images/train
    base_out_dir = os.path.join(output_root, project_name)
    img_out_dir = os.path.join(base_out_dir, "images", split)
    lbl_out_dir = os.path.join(base_out_dir, "labels")
    
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    # 3. 分组 (Grouping)
    groups = defaultdict(list)
    for item in raw_data:
        key = get_group_key(item)
        groups[key].append(item)

    processed_data_list = []
    current_id = start_id
    
    # 进度条
    pbar = tqdm(groups.items(), desc=f"Processing {sample_type} - {split}")

    for key, items in pbar:
        # === 校验 ===
        if len(items) != 4:
            # 这种情况可能是数据缺失，跳过
            # print(f"Warning: Group {key} has {len(items)} items, skipping.")
            continue
        
        # === 排序 ===
        # 按照 LIGHT_ORDER_MAP 对 items 进行排序
        # 如果某个光源缺失或重复，这里可能会报错，需要数据本身完整
        try:
            items.sort(key=lambda x: LIGHT_ORDER_MAP[x['light_source']])
        except KeyError as e:
            print(f"Error: Unknown light source {e} in group {key}")
            continue

        # 检查排序后的光源是否完全匹配标准顺序
        current_lights = [x['light_source'] for x in items]
        if current_lights != LIGHT_ORDER:
            # 极其罕见的情况
            print(f"Warning: Light order mismatch {current_lights}, skipping.")
            continue

        # === 提取元数据 ===
        # 既然是同一组，大部分元数据取第一个即可
        base_item = items[0]
        gt_label = base_item['label']
        bbox = base_item['bbox']
        
        # === 核心逻辑：Prior Label 策略 ===
        prior_label = None
        if sample_type == 'positive':
            prior_label = gt_label
        elif sample_type == 'rectification' or sample_type == 'negative':
            # 必须从 items 里拿，如果有这个字段的话
            prior_label = base_item['prior_label']

        # === 图像处理 ===
        original_imgs_draw = []
        crop_imgs_resize = []
        
        orig_paths_record = []
        crop_paths_record = []
        
        valid_group = True
        
        for it in items:
            # 拼接完整绝对路径读取图片
            orig_abs_path = os.path.join(dataset_root, it['original_image_path'])
            crop_abs_path = os.path.join(dataset_root, it['crop_image_path'])
            
            orig_img = cv2.imread(orig_abs_path)
            crop_img = cv2.imread(crop_abs_path)
            
            if orig_img is None or crop_img is None:
                print(f"Error reading image: {orig_abs_path} or {crop_abs_path}")
                valid_group = False
                break
                
            # 全局图：画红框
            orig_with_box = draw_bbox_on_image(orig_img, it['bbox'])
            # 局部图：缩放 (注意：crop图本来就是切好的，不需要再切，只需要resize)
            crop_resized = letter_resize_bbox(crop_img, target_size=300) # 300x300 -> 拼成600
            
            original_imgs_draw.append(orig_with_box)
            crop_imgs_resize.append(crop_resized)
            
            # 记录相对路径用于JSON
            orig_paths_record.append(it['original_image_path'])
            crop_paths_record.append(it['crop_image_path'])
            
        if not valid_group:
            continue
            
        # === 拼接 ===
        # 目标尺寸 600x600 (由4个300x300组成)
        final_global_img = composite_2x2_images(original_imgs_draw, target_size=600)
        final_local_img = composite_2x2_images(crop_imgs_resize, target_size=600)
        
        # === 保存图像 ===
        global_filename = f"global_{current_id}.png"
        local_filename = f"local_{current_id}.png"
        
        global_save_path = os.path.join(img_out_dir, global_filename)
        local_save_path = os.path.join(img_out_dir, local_filename)
        
        cv2.imwrite(global_save_path, final_global_img)
        cv2.imwrite(local_save_path, final_local_img)
        
        # === 构建 JSON Item ===
        # 构造保存到 JSON 里的相对路径
        # 格式: 3_composite_images/stripe_phase012/images/train/global_xxxx.png
        rel_global_path = os.path.join("3_composite_images", project_name, "images", split, global_filename)
        rel_local_path = os.path.join("3_composite_images", project_name, "images", split, local_filename)
        
        out_item = {
            "id": current_id,
            "composite_global_path": rel_global_path,
            "composite_local_path": rel_local_path,
            "bbox": bbox,
            "sample_type": sample_type,
            "label": gt_label,
            "prior_label": prior_label,
            "light_source_order": LIGHT_ORDER,
            "original_image_paths": orig_paths_record,
            "original_crop_paths": crop_paths_record
        }
        
        processed_data_list.append(out_item)
        current_id += 1

    # 4. 保存 JSON
    out_json_path = os.path.join(lbl_out_dir, f"{split}.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data_list, f, ensure_ascii=False, indent=2)
        
    print(f"\nCompleted! Saved {len(processed_data_list)} items.")
    print(f"JSON saved to: {out_json_path}")
    print(f"Images saved to: {img_out_dir}")

if __name__ == "__main__":
    # === 参数配置区 ===
    
    # 1. 你的数据集根目录 (绝对路径，用于拼接读取图片)
    # 脚本会认为 original_image_path 是相对于这个目录的
    DATASET_ROOT = "/data/ZS/defect_dataset"
    
    # 2. 输出保存的根目录
    # 最终会生成到 OUTPUT_ROOT/stripe_phase012/...
    OUTPUT_ROOT = os.path.join(DATASET_ROOT, "3_composite_images")
    
    # 3. 输入参数   唯一需要修改的地方
    # 示例1：处理 Positive Train
    # input_json = "/data/ZS/defect_dataset/2_paint_bbox/stripe_phase123_gt_negative/labels/val.json"
    # run_sample_type = "positive"
    # run_split = "val"
    
    # 示例2：处理 Rectification Train
    # input_json = "/data/ZS/defect_dataset/2_paint_bbox/stripe_phase012_gt_rectification/labels/train.json"
    # run_sample_type = "rectification"
    # run_split = "train"
    
    # 示例3：处理 Negative Val
    input_json = "/data/ZS/defect_dataset/2_paint_bbox/stripe_phase012_gt_rectification/labels/val.json"
    run_sample_type = "negative"
    run_split = "val"

    # 执行
    if os.path.exists(input_json):
        process_dataset(
            json_path=input_json,
            dataset_root=DATASET_ROOT,
            output_root=OUTPUT_ROOT,
            sample_type=run_sample_type,
            split=run_split,
            start_id=1000001
        )
    else:
        print(f"Input file not found: {input_json}")