"""
根据原始图像和coco标签, 获取region proposal, 并将region proposal图像存储起来
"""
import json
import cv2
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import gc

def get_region_proposal(image: np.ndarray, bbox: list, bbox_format='coco', context_ratio: float = 0.4) -> np.ndarray:
    """
    根据bbox截取图像局部 (复用你之前的逻辑，增加了边界检查的鲁棒性)
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

    # 计算扩充
    new_w = w * (1 + context_ratio)
    new_h = h * (1 + context_ratio)
    
    # 转换为坐标 (加 round 减少精度误差)
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
      
    # 确保切片有效
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

def main(json_path: Path, image_root_dir: Path, save_root_dir: Path):
    # 1. 定义光源列表 (根据你的实际文件夹结构)
    light_sources = ['16col0','16col1','16col2', '16row0', '16row1', '16row2', 
                     '32col0', '32col1', '32col2', '32row0', '32row1', '32row2']
    
    print(f"正在加载标签文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 构建索引字典
    # images 字典: id -> file_name
    img_id_to_info = {img['id']: img for img in data['images']}
    # categories 字典: id -> name (用于创建子文件夹)
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # annotations 字典: image_id -> [ann1, ann2, ...]
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    print(f"开始处理 {len(img_id_to_anns)} 张含有标注的图像...")

    cnt = 0
    # 3. 遍历每一张有标注的图
    # 使用 tqdm 显示进度条
    import pdb; pdb.set_trace()
    for image_id, anns in tqdm(img_id_to_anns.items(), desc="Processing"):
        image_info = img_id_to_info.get(image_id)
        if not image_info:
            continue
            
        file_name = image_info['file_name'] # 例如 "part1.bmp"
        file_stem = Path(file_name).stem    # 例如 "part1"

        # 4. 遍历所有光源 (Data Augmentation / Multi-view)
        for light in light_sources:
            # 构建图片路径: root / 光源文件夹 / 图片名
            # 注意：需确认你的目录结构是否真的是这样
            image_path = image_root_dir / light / file_name
            
            if not image_path.exists():
                # 可以在这里记录日志，防止刷屏
                continue
                
            # 读取灰度图 (IMREAD_GRAYSCALE) 或者彩色图，根据需求
            # 注意：cv2.imread 不支持 Path 对象，必须 str()
            raw_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if raw_image is None:
                print(f"警告: 无法读取图片 {image_path}")
                continue

            # 5. 遍历该图上的所有 bbox 并裁剪
            for i, ann in enumerate(anns):
                bbox = ann['bbox']
                cat_id = ann['category_id']
                cat_name = cat_id_to_name.get(cat_id, 'unknown')
                
                # 裁剪
                crop = get_region_proposal(raw_image, bbox, context_ratio=0.2)
                
                if crop is None or crop.size == 0:
                    continue
                
                # 6. 构建保存路径
                # 结构: save_dir / 类别名 / ...
                # 这样方便后续 VLM 按类别加载数据
                category_dir = save_root_dir / cat_name
                category_dir.mkdir(parents=True, exist_ok=True)
                
                # 文件名命名规则: 原文件名_光源_bbox序号.png
                # 避免文件名冲突
                save_name = f"{file_stem}_{light}_{i}.png"
                save_path = category_dir / save_name
                
                cv2.imwrite(str(save_path), crop)

            del raw_image
            
        cnt += 1
        if cnt%50 == 0:
            gc.collect()
            
if __name__ == "__main__":
    # 配置路径
    json_file = Path('/data/ZS/defect_datasets_raw/paint_stripe/paint_stripe_val.json')  # 你的json路径
    img_root = Path('/data/ZS/defect_datasets_raw/paint_stripe/images')         # 你的图片根目录
    output_root = Path('/data/ZS/defect_datasets_vlm/paint_stripe/region_proposal_val') # 结果保存路径
    
    # 运行
    main(json_file, img_root, output_root)