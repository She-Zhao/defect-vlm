"""
生成多光源对齐的【微小负样本 (Micro Negative Samples / Hard Negatives)】
逻辑：
1. 在纯背景处强制生成 8~20 像素的极小 BBox
2. 强行赋予 breakage, inclusion, scratch 这三种最易混淆的先验标签
3. 训练 VLM 识别微小的正常反光噪点，降低假阳性 (False Positive)
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

# === 核心修改 1：仅针对最容易误报的小目标缺陷生成先验标签 ===
HARD_DEFECT_MAP = ('breakage', 'inclusion', 'scratch')

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

# === 核心修改 2：强制生成极小的 BBox ===
def generate_micro_random_bbox(image_shape, gt_bboxes, max_trials=50):
    """生成一个极小的、且不与任何GT重叠的随机BBox"""
    h_img, w_img = image_shape
    
    for _ in range(max_trials):
        # 尺寸严格限制在 15 到 35 像素之间 (微小目标)
        w = random.randint(8, 20)
        h = random.randint(8, 20)
        
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

def get_region_proposal(image: np.ndarray, bbox: list, fixed_context_ratio: float = None) -> np.ndarray:
    """裁剪图像"""
    if image is None: return None
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    
    # === 核心修改 3：微小目标需要更大的 context_ratio 才能看到条纹背景 ===
    if fixed_context_ratio is not None:
        h_context_ratio = w_context_ratio = fixed_context_ratio
    else:
        h_context_ratio = w_context_ratio = 1.0 # 默认放大1倍
        
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

# === 核心修改 4：路径全量作为参数输入 ===
def main(
    data_root: str,
    json_path: str,
    image_root_dir: str,
    save_img_dir: str,
    save_label_dir: str,
    split: str, 
    samples_per_image: int = 2
):
    """
    Args:
        data_root: 数据集根目录 (用于计算存入 json 中的相对路径)
        json_path: 真实的 GT json 文件绝对路径
        image_root_dir: 原始图像所在的目录绝对路径 (如 1_paint_rgb/stripe_phase012/images)
        save_img_dir: 生成的负样本图片保存绝对路径
        save_label_dir: 生成的负样本 json 标签保存绝对路径
        split: train 或 val
        samples_per_image: 每张原图生成几组微小负样本
    """
    data_root = Path(data_root)
    json_path = Path(json_path)
    image_root_dir = Path(image_root_dir)
    save_img_dir = Path(save_img_dir)
    save_label_dir = Path(save_label_dir)
    
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)
    
    # ================= 加载数据 =================
    print(f"📖 正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立索引：Image ID -> Image Info
    img_id_to_info = {img['id']: img for img in data['images']}
    
    # 建立索引：Image ID -> GT Annotations (用于避让)
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"🚀 开始处理 {len(img_id_to_info)} 张图像，目标生成【微小负样本(Hard Negatives)】...")

    # ================= 核心处理循环 =================
    metadata_list = []
    global_id_counter = 0
    processed_count = 0

    for image_id, image_info in tqdm(img_id_to_info.items(), desc=f"Generating Micro Neg {split}"):
        
        file_name = image_info['file_name'] 
        file_stem = Path(file_name).stem
        current_gt_bboxes = [ann['bbox'] for ann in img_id_to_anns[image_id]]

        # 1. 预先读取 4 张光源图
        light_images = {}
        load_success = True
        first_img_shape = None

        for light in REQUIRED_LIGHTS:
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
                first_img_shape = img.shape[:2]

        if not load_success or first_img_shape is None:
            continue
            
        img_h, img_w = first_img_shape

        # 2. 生成 N 组负样本
        for _ in range(samples_per_image):
            # A. 随机生成 1 个【微小】位置
            rand_bbox = generate_micro_random_bbox(
                image_shape=(img_h, img_w),
                gt_bboxes=current_gt_bboxes,
                max_trials=50
            )

            if rand_bbox is None:
                continue 

            # B. 限定在最易混淆的 3 种缺陷中分配假先验
            prior_label = random.choice(HARD_DEFECT_MAP)

            # C. 裁剪图像 (为微小目标提供足够的背景参考，ratio=1.0)
            temp_crops = {} 
            crop_success = True
            
            for light in REQUIRED_LIGHTS:
                crop = get_region_proposal(light_images[light], rand_bbox, fixed_context_ratio=1.0)
                if crop is None or crop.size == 0:
                    crop_success = False
                    break
                temp_crops[light] = crop
            
            if not crop_success:
                continue

            # D. 保存文件并记录元数据
            for light in REQUIRED_LIGHTS:
                # 命名加上 micro_neg 标识
                save_name = f"{file_stem}_{light}_micro_neg_{global_id_counter}.png"
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
                    "label": "background",    
                    "prior_label": prior_label,
                    "light_source": light,
                    "sample_type": "micro_negative" # 显式标记为微小负样本
                }
                metadata_list.append(metadata)
                global_id_counter += 1 

        del light_images
        processed_count += 1
        if processed_count % 20 == 0:
            gc.collect()

    # ================= 保存JSON =================
    out_json_path = save_label_dir / f'{split}_micro.json'
    print(f"✅ 保存微小负样本元数据到: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    # 配置绝对路径
    DATA_ROOT = "/data/ZS/defect_dataset"
    PROJECT_NAME = "stripe_phase012"
    
    # ======= Train 集合配置 =======
    train_json_path = f"{DATA_ROOT}/1_paint_rgb/{PROJECT_NAME}/labels/train.json"
    train_image_dir = f"{DATA_ROOT}/1_paint_rgb/{PROJECT_NAME}/images"
    train_save_img_dir = f"{DATA_ROOT}/2_paint_bbox/{PROJECT_NAME}_micro_negative/images/train"
    train_save_label_dir = f"{DATA_ROOT}/2_paint_bbox/{PROJECT_NAME}_micro_negative/labels"
    
    main(
        data_root=DATA_ROOT,
        json_path=train_json_path,
        image_root_dir=train_image_dir,
        save_img_dir=train_save_img_dir,
        save_label_dir=train_save_label_dir,
        split='train',
        samples_per_image=2  # 根据需要调整生成数量
    )
    
    # # ======= Val 集合配置 =======
    # val_json_path = f"{DATA_ROOT}/1_paint_rgb/{PROJECT_NAME}/labels/val.json"
    # val_image_dir = f"{DATA_ROOT}/1_paint_rgb/{PROJECT_NAME}/images"
    # val_save_img_dir = f"{DATA_ROOT}/2_paint_bbox/{PROJECT_NAME}_micro_negative/images/val"
    # val_save_label_dir = f"{DATA_ROOT}/2_paint_bbox/{PROJECT_NAME}_micro_negative/labels"

    # main(
    #     data_root=DATA_ROOT,
    #     json_path=val_json_path,
    #     image_root_dir=val_image_dir,
    #     save_img_dir=val_save_img_dir,
    #     save_label_dir=val_save_label_dir,
    #     split='val',
    #     samples_per_image=2
    # )