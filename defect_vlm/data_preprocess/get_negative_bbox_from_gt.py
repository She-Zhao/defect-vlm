"""
生成多光源对齐的负样本 (Negative Samples Generation)
逻辑：先确定一个无缺陷的随机BBox，然后从4个光源的对应位置裁剪图像。
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
# 缺陷类别映射 (用于生成 prior_label)
DEFECT_MAP = ('breakage', 'inclusion', 'crater', 'bulge', 'run', 'scratch')

# 必须存在的4个光源，顺序不敏感，但必须齐备
REQUIRED_LIGHTS = ['16col', '16row', '32col', '32row']

def compute_iou(box1, box2):
    """计算IoU: box=[x, y, w, h]"""
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
    """生成一个不与任何GT重叠的随机BBox"""
    h_img, w_img = image_shape
    
    for _ in range(max_trials):
        # 1. 随机尺寸
        if not ref_sizes:
            base_w, base_h = 100, 100
        else:
            base_w, base_h = random.choice(ref_sizes)
            
        scale = random.uniform(0.8, 1.2)
        w = int(base_w * scale)
        h = int(base_h * scale)
        
        w = max(10, min(w, w_img - 1))
        h = max(10, min(h, h_img - 1))

        # 2. 随机位置
        if w_img - w <= 0 or h_img - h <= 0:
            continue
            
        x = random.randint(0, w_img - w)
        y = random.randint(0, h_img - h)
        
        candidate_box = [x, y, w, h]

        # 3. IoU 检查 (必须是纯背景)
        has_overlap = False
        for gt_box in gt_bboxes:
            if compute_iou(candidate_box, gt_box) > 0.0:
                has_overlap = True
                break
        
        if not has_overlap:
            return candidate_box

    return None

def get_dynamic_context_ratio(bbox_size: float) -> float:
    """根据尺寸动态计算扩充比例"""
    sizes = [20, 100]
    ratios = [2.0, 0.5]
    return float(np.interp(bbox_size, sizes, ratios))

def get_region_proposal(image: np.ndarray, bbox: list, fixed_context_ratio: float = None) -> np.ndarray:
    """裁剪图像"""
    if image is None: return None
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    
    if fixed_context_ratio is not None:
        h_context_ratio = w_context_ratio = fixed_context_ratio
    else:
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

def main(
    data_root: Path, 
    dataset_name: str, 
    split: str, 
    samples_per_image: int = 2
):
    """
    Args:
        data_root: 数据集根目录
        dataset_name: 项目名称 (e.g., stripe_phase012)
        split: train 或 val
        samples_per_image: 每张原图生成几组负样本
    """
    # ================= 路径配置 =================
    raw_dir = data_root / '1_paint_rgb' / dataset_name
    json_path = raw_dir / 'labels' / f'{split}.json'
    image_root_dir = raw_dir / 'images'

    # 输出路径: 2_paint_bbox/stripe_phase012_gt_negative
    output_base_dir = data_root / '2_paint_bbox' / f'{dataset_name}_gt_negative'
    save_img_dir = output_base_dir / 'images' / split
    save_label_dir = output_base_dir / 'labels'
    
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)
    
    # ================= 加载数据 =================
    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立索引：Image ID -> Image Info
    img_id_to_info = {img['id']: img for img in data['images']}
    
    # 建立索引：Image ID -> GT Annotations (用于避让)
    img_id_to_anns = defaultdict(list)
    all_defect_sizes = []
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)
        w, h = ann['bbox'][2], ann['bbox'][3]
        all_defect_sizes.append((w, h))

    # 尺寸参考池
    if len(all_defect_sizes) > 2000:
        ref_size_pool = random.sample(all_defect_sizes, 2000)
    else:
        ref_size_pool = all_defect_sizes

    print(f"开始处理 {len(img_id_to_info)} 张图像，目标生成负样本...")

    # ================= 核心处理循环 =================
    metadata_list = []

    global_id_counter = 0
    
    processed_count = 0

    # 遍历每一张物理图像（通过 ID）
    for image_id, image_info in tqdm(img_id_to_info.items(), desc=f"Generating Neg {split}"):
        
        file_name = image_info['file_name'] # e.g., 0000_00019_....png
        file_stem = Path(file_name).stem
        
        # 获取该图所有的真实缺陷框（用于避让）
        # 注意：即使某张图没有缺陷，img_id_to_anns[image_id] 也会返回空列表，逻辑依然成立
        current_gt_bboxes = [ann['bbox'] for ann in img_id_to_anns[image_id]]

        # 1. 预先读取该 Image ID 对应的 4 张光源图
        #    为了效率，一次性读入内存，处理完释放
        light_images = {} # { '16col': img_array, ... }
        load_success = True
        
        first_img_shape = None # 用于获取宽高

        for light in REQUIRED_LIGHTS:
            # 假设文件结构：1_paint_rgb/stripe_phase012/images/16col/xxx.png
            img_path = image_root_dir / light / file_name
            if not img_path.exists():
                load_success = False
                break
            
            img = cv2.imread(str(img_path))
            if img is None:
                load_success = False
                break
                
            light_images[light] = img
            if first_img_shape is None:
                first_img_shape = img.shape[:2] # h, w

        if not load_success or first_img_shape is None:
            # 如果任何一个光源缺失，整张图跳过，无法做多光源负样本
            continue
            
        img_h, img_w = first_img_shape

        # 2. 生成 N 组负样本
        for _ in range(samples_per_image):
            # A. 随机生成 1 个位置 (BBox)
            rand_bbox = generate_random_bbox(
                image_shape=(img_h, img_w),
                gt_bboxes=current_gt_bboxes,
                ref_sizes=ref_size_pool,
                max_trials=50
            )

            if rand_bbox is None:
                continue # 找不到合适的位置，跳过这次尝试

            # B. 随机分配一个“假”标签 (Prior Label)
            # 模拟：检出器错误地认为这里是 scratch，但实际上是 background
            prior_label = random.choice(DEFECT_MAP)

            # C. 对 4 个光源分别裁剪 (必须全部成功)
            # 我们先暂存，全部成功后再写入磁盘
            temp_crops = {} # { '16col': crop_img, ... }
            crop_success = True
            
            for light in REQUIRED_LIGHTS:
                crop = get_region_proposal(light_images[light], rand_bbox, fixed_context_ratio=0.4)
                if crop is None or crop.size == 0:
                    crop_success = False
                    break
                temp_crops[light] = crop
            
            if not crop_success:
                continue

            # D. 全部成功，保存文件并记录元数据
            # 这一组 4 个样本共享同一个 bbox，但 light_source 不同
            for light in REQUIRED_LIGHTS:
                # 构造文件名: 保持与正样本一致的风格，或加后缀
                # 示例: 00019_..._16col_background_0.png
                
                # 命名格式: {原文件名}_{光源}_neg_{全局ID}.png
                save_name = f"{file_stem}_{light}_neg_{global_id_counter}.png"
                save_path = save_img_dir / save_name
                
                cv2.imwrite(str(save_path), temp_crops[light])

                # 记录 Metadata
                try:
                    rel_original_path = (image_root_dir / light / file_name).relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError:
                    continue

                metadata = {
                    "id": global_id_counter,  
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "bbox": rand_bbox,        # 关键：4个光源的 BBox 必须完全一致
                    "label": "background",    # 真实标签
                    "prior_label": prior_label, # 模拟误检标签 (4个光源保持一致)
                    "light_source": light,
                    "sample_type": "negative" # 显式标记
                }
                metadata_list.append(metadata)
                global_id_counter += 1 

        # 内存清理
        del light_images
        processed_count += 1
        if processed_count % 20 == 0:
            gc.collect()

    # ================= 保存JSON =================
    out_json_path = save_label_dir / f'{split}.json'
    print(f"正在保存负样本元数据到: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    
    # 你的项目名称
    PROJECT_NAME = 'stripe_phase012' # 注意修改这里
    
    # 分别处理 train 和 val
    # samples_per_image=2 表示每张大图生成 2 组 (2x4=8张) 负样本
    main(DATA_ROOT, PROJECT_NAME, 'train', samples_per_image=2)
    main(DATA_ROOT, PROJECT_NAME, 'val', samples_per_image=2)