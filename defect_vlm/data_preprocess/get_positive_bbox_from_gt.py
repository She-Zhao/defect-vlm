"""
根据原始图像和coco标签, 从标注的数据集中获取region proposal的正例, 并将region proposal图像存储起来
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

def get_dynamic_context_ratio(bbox_size: float) -> float:
    """
    根据bbox的尺寸动态计算context_ratio
    策略：
    - 极小目标 (<=20px): 需要大环境，ratio = 2.0 (扩大一倍)
    - 中大目标 (>=100px): 只需要少量环境，ratio = 0.5
    - 中间尺寸: 线性过渡
    """
    # 定义映射关系：[尺寸阈值] -> [对应的ratio]
    # 注意：xp必须是递增的
    sizes = [20, 100]       # 尺寸节点：20像素，100像素
    ratios = [2, 0.5]       # 对应比例：2.0倍， 0.5倍
    
    # 使用 numpy 的线性插值，它可以自动处理边界情况
    # 如果 size < 50，返回 1.2；如果 size > 100，返回 0.6
    return float(np.interp(bbox_size, sizes, ratios))

def get_region_proposal(
    image: np.ndarray, 
    bbox: list, 
    bbox_format='coco', 
    fixed_context_ratio: float = None
) -> np.ndarray:
    """
    根据bbox截取图像局部
    """
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
    
    # 2. 确定 Context Ratio
    if fixed_context_ratio is not None:
        h_context_ratio = w_context_ratio = fixed_context_ratio
    else:
        h_context_ratio = get_dynamic_context_ratio(h)
        w_context_ratio = get_dynamic_context_ratio(w)

    # 计算扩充
    new_w = w * (1 + w_context_ratio)
    new_h = h * (1 + h_context_ratio)
    
    # 转换为坐标 (加 round 减少精度误差)
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
      
    # 确保切片有效
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

def main(data_root: Path, dataset_name: str, split: str, light_sources: list):
    """
    Args:
        data_root: 数据集根目录锚点 (/data/ZS/defect_dataset)
        dataset_name: 数据集名称 (paint_stripe)
        split: 数据划分 (train 或 val)
        light_sources: 光源名称列表
    """
    # ================= 路径配置 =================
    # 输入路径 (0_defect_dataset_raw)
    raw_dir = data_root / '1_paint_rgb' / dataset_name
    json_path = raw_dir / 'labels' / f'{split}.json'
    image_root_dir = raw_dir / 'images'

    # 输出路径 (1_defect_dataset_bbox)
    output_base_dir = data_root / '2_paint_bbox' / f'{dataset_name}_gt_positive'
    save_img_dir = output_base_dir / 'images' / split
    save_label_dir = output_base_dir / 'labels'
    
    # 创建输出目录
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)

    # ================= 加载数据 =================
    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_id_to_info = {img['id']: img for img in data['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"开始处理 {len(img_id_to_anns)} 张含有标注的图像...")

    # ================= 处理循环 =================
    metadata_list = []
    global_id_counter = 0 # 简单的全局ID计数器
    cnt = 0

    for image_id, anns in tqdm(img_id_to_anns.items(), desc=f"Processing {split}"):
        image_info = img_id_to_info.get(image_id)
        if not image_info:
            continue
            
        file_name = image_info['file_name'] 
        file_stem = Path(file_name).stem    

        for light in light_sources:
            # 构造原始图片路径
            original_img_path = image_root_dir / light / file_name
            
            if not original_img_path.exists():
                continue
            
            # 读取图像
            raw_image = cv2.imread(str(original_img_path))
            # import pdb; pdb.set_trace()
            if raw_image is None:
                continue

            for i, ann in enumerate(anns):
                bbox = ann['bbox']
                cat_id = ann['category_id']
                cat_name = cat_id_to_name.get(cat_id, 'unknown')
                
                # 裁剪
                crop = get_region_proposal(raw_image, bbox)
                
                if crop is None or crop.size == 0:
                    continue
                
                # 构造保存文件名: 原文件名_光源_类别_序号.png
                # 例如: part1_16col0_scratch_0.png
                save_name = f"{file_stem}_{light}_{cat_name}_{i}.png"
                save_path = save_img_dir / save_name
                
                # 保存图片
                cv2.imwrite(str(save_path), crop)

                # ================= 记录元数据 =================
                # 计算相对路径
                try:
                    rel_original_path = original_img_path.relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError as e:
                    print(f"路径错误: {e}")
                    continue

                metadata = {
                    "id": global_id_counter,
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "bbox": bbox,
                    "label": cat_name,
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
    print(f"正在保存元数据到: {out_json_path}")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    # 配置锚点路径
    DATA_ROOT = Path('/data/ZS/defect_dataset')
    
    # 修改参数
    dataset_name = 'ap_phase123'
    # light_sources = ['16col0','16col1','16col2', '16row0', '16row1', '16row2', 
    #                  '32col0', '32col1', '32col2', '32row0', '32row1', '32row2']
    light_sources = ['16col','16row', '32col', '32row'] 

    # 分别处理 train 和 val
    main(DATA_ROOT, dataset_name, 'train', light_sources)
    main(DATA_ROOT, dataset_name, 'val', light_sources)
