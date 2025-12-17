"""
将yolo格式数据集转化为coco格式数据集

author: zhaoshe
"""
import re
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from PIL import Image

def bbox_yolo2coco(x, y, w, h, img_w, img_h):
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    x_abs = round((x-w/2)*img_w)
    y_abs = round((y-h/2)*img_h)
    w_abs = round(img_w * w)
    h_abs = round(img_h * h)
    return max(x_abs, 0), max(y_abs, 0), w_abs, h_abs

def yolo2coco(
    labels_dir: str, 
    images_dir: str, 
    save_path: str,
    dataset_source: str,
    categories: Dict,
    img_suffix='.png'
) -> None:
    """将yolo格式转换为标注coco格式

    Args:
        labels_dir (str): yolo中txt标注文件所在的文件
        images_dir (str): 对应图像所在的文件夹
        save_path (str): 转换后的coco文件保存路径
        dataset_source (str): 数据集描述
        categories (Dict): 缺陷id和名称的对应关系字典 id: name
        img_suffix (str, optional): 图像的后缀. Defaults to '.png'.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    defects_map = {str(item['id']): item['name'] for item in categories}   
    info = {
        "description": "Car paint dataset, on stripe light",
        "total_images": 0,
        "total_defects": 0,
        "num_of_defects": {}
    } 
    images = []
    annotations = []

    label_files = [f for f in labels_dir.glob('*.txt') if f.name!='classes.txt']
    info['total_images'] = len(label_files)
    
    image_id = 1
    label_id = 1
    for label in tqdm(label_files, desc='Processing'):
        try: 
            # 构建images字段里面的item
            image_name = Path(label).stem + img_suffix
            image_path = images_dir / image_name
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                
            image_item = {
                'id': image_id,
                'file_name': image_name,
                'width': img_w,
                'height': img_h,
                'dataset_source': dataset_source
            }
            images.append(image_item)
            
            # 构建annotations字段里面的item
            label_path = labels_dir / label
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                c, x, y, w, h = line.strip().split(' ')
                x_abs, y_abs, w_abs, h_abs = bbox_yolo2coco(x, y, w, h, img_w, img_h)
                annotation = {
                    'id': label_id,
                    'image_id': image_id,
                    'category_id': int(c),
                    'bbox': [x_abs, y_abs, w_abs, h_abs],
                    'area': int(w_abs * h_abs),
                    'iscrowd': 0
                }
                annotations.append(annotation)
                info['num_of_defects'][defects_map[c]] = info['num_of_defects'].get(defects_map[c], 0) + 1
                label_id += 1
            image_id += 1
            
        except Exception as e:
            print(f"处理文件 {label} 时发生错误: {e}")
    
    info['total_defects'] = sum(info['num_of_defects'].values())
    coco_format =  {
        'info': info,
        'categories': categories,
        'images': images,
        'annotations': annotations  
    }
    
    # 对categories和info进行扁平化保存
    # 1. 先生成标准的多行 JSON 字符串
    json_str = json.dumps(coco_format, indent=2, ensure_ascii=False)

    # 2.【bbox处理】将 bbox 列表压缩为一行
    json_str = re.sub(
        r'("bbox": \[)([^\]]+)(\])', 
        lambda m: m.group(1) + ", ".join(x.strip() for x in m.group(2).split(',')) + m.group(3), 
        json_str
    )

    # 3.【categories处理】将 categories 中的字典压缩为一行
    # 匹配模式：{ 任意空白 "id": 数字, 任意空白 "name": "字符串" 任意空白 }
    # 注意：这里假设 id 在 name 前面，这符合你代码中定义的顺序
    json_str = re.sub(
        r'\{\s*"id":\s*(\d+),\s*"name":\s*"([^"]+)"\s*\}',
        r'{"id": \1, "name": "\2"}',
        json_str
    )

    # 4. 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json_str) 
        
if __name__ == "__main__":
    labels_dir = r'D:\Project\Defect_Dataset\PaintDatasets\02\train_val\labels'
    images_dir = r'D:\Project\Defect_Dataset\PaintDatasets\02\train_val\images'
    save_path = r'D:\Github\defect-vlm\data\paint_ap.json'
    dataset_source = 'paint_ap'
    categories = [
        {"id": 0, "name": "breakage"},
        {"id": 1, "name": "inclusion"},
        {"id": 2, "name": "scratch"},
        {"id": 3, "name": "crater"},
        {"id": 4, "name": "run"},
        {"id": 5, "name": "bulge"},
        {"id": 6, "name": "condensate"},
    ]     
    img_suffix = '.png'

    coco = yolo2coco(
        labels_dir=labels_dir,
        images_dir=images_dir,
        save_path=save_path,
        dataset_source=dataset_source,
        categories=categories,
        img_suffix=img_suffix
    )
