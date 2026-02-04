"""
将 LabelMe (JSON) 格式数据集转换为 COCO 格式
"""
"""
将 LabelMe (JSON) 格式数据集转换为 COCO 格式 (无映射版)
Author: Gemini & User
"""
import json
import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from PIL import Image

def get_bbox_from_points(points: List[List[float]]) -> List[float]:
    """从 LabelMe 的点列表中计算 COCO bbox [xmin, ymin, w, h]"""
    pts = np.array(points)
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])
    width = xmax - xmin
    height = ymax - ymin
    return [float(xmin), float(ymin), float(width), float(height)]

def format_json_str(coco_dict: Dict[str, Any]) -> str:
    """格式化 JSON 字符串，压缩 bbox 和 categories 为单行"""
    json_str = json.dumps(coco_dict, indent=2, ensure_ascii=False)
    
    # 压缩 bbox
    json_str = re.sub(
        r'("bbox": \[)([^\]]+)(\])', 
        lambda m: m.group(1) + ", ".join(x.strip() for x in m.group(2).split(',')) + m.group(3), 
        json_str
    )
    
    # 压缩 categories
    json_str = re.sub(
        r'\{\s*"id":\s*(\d+),\s*"name":\s*"([^"]+)"\s*\}',
        r'{"id": \1, "name": "\2"}',
        json_str
    )
    return json_str

def labelme2coco(
    json_dir: str | Path,
    save_path: str | Path,
    categories: List[Dict[str, Any]],
    dataset_source: str = "custom_labelme"
) -> None:
    
    json_dir = Path(json_dir)
    save_path = Path(save_path)
    
    # 构建 COCO category name -> id 的映射
    # 核心逻辑：直接使用 categories 里的 name 作为键
    cat_name_to_id = {cat['name']: cat['id'] for cat in categories}
    
    info = {
        "description": dataset_source,
        "total_images": 0,
        "total_defects": 0,
        "num_of_defects": {name: 0 for name in cat_name_to_id.keys()}
    }
    
    images = []
    annotations = []
    
    json_files = list(json_dir.glob('*.json'))
    info['total_images'] = len(json_files)
    
    image_id = 1
    annotation_id = 1
    
    print(f"🚀 开始转换 {len(json_files)} 个 LabelMe JSON 文件...")
    
    for json_file in tqdm(json_files, desc="Converting"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # --- 1. 处理图像信息 ---
            img_h = data.get('imageHeight')
            img_w = data.get('imageWidth')
            file_name = os.path.basename(data.get('imagePath', ''))
            
            # 尝试补救缺失的宽高
            if not img_h or not img_w:
                img_path = json_dir / file_name
                if img_path.exists():
                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                else:
                    img_w, img_h = 0, 0

            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": img_w,
                "height": img_h,
                "dataset_source": dataset_source
            })
            
            # --- 2. 处理标注信息 (Shapes) ---
            for shape in data.get('shapes', []):
                label_name = shape.get('label')
                points = shape.get('points')
                
                # 【直接匹配】检查 label 是否在我们的 categories 列表里
                if label_name not in cat_name_to_id:
                    # 如果 json 里有 "不导电"，但 categories 里没有，这里就会跳过
                    # print(f"⚠️ 跳过未定义类别: {label_name}") 
                    continue
                
                cat_id = cat_name_to_id[label_name]
                bbox = get_bbox_from_points(points)
                w, h = bbox[2], bbox[3]
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": int(w * h),
                    "iscrowd": 0
                })
                
                info['num_of_defects'][label_name] += 1
                annotation_id += 1
                
            image_id += 1
            
        except Exception as e:
            print(f"❌ 处理文件 {json_file.name} 出错: {e}")
            continue
        
    info['total_defects'] = sum(info['num_of_defects'].values())
    # --- 保存结果 ---
    coco_format = {
        "info": info,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    
    print(f"💾 正在保存到 {save_path} ...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(format_json_str(coco_format))
        
    print("✅ 转换完成！")

if __name__ == "__main__":
    json_dir = r'D:\Project\Defect_Dataset\AL\决赛_train_1011\all_labels'
    save_path = 'data/al_final.json'

    # 1. 定义 COCO 目标类别 (必须与你模型训练时的 ID 一致)
    TARGET_CATEGORIES = [
        {"id": 0, "name": "不导电"},
        {"id": 1, "name": "擦花"},
        {"id": 2, "name": "角位漏底"},
        {"id": 3, "name": "桔皮"},
        {"id": 4, "name": "漏底"},
        {"id": 5, "name": "喷流"},
        {"id": 6, "name": "漆泡"},
        {"id": 7, "name": "起坑"},
        {"id": 8, "name": "杂色"},        
        {"id": 9, "name": "脏点"}      
    ]
    
    labelme2coco(
        json_dir = json_dir,
        save_path = save_path,
        categories = TARGET_CATEGORIES,
        dataset_source = "al_final"
    )
