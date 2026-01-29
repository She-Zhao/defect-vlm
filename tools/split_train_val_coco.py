"""
将coco格式的数据集划分为训练集和测试集
"""
import json
import random
import re
import copy
from collections import Counter
from pathlib import Path
random.seed(42)

def get_defect_counts(anns, cat_map):
        """辅助函数：统计标注列表中的缺陷数量"""
        # 1. 提取所有标注的 category_id
        cat_ids = [ann['category_id'] for ann in anns]
        # 2. 使用 Counter 统计频次
        counts = Counter(cat_ids)
        # 3. 将 id 转换为 name，并构建结果字典 (初始化所有类别为0，确保格式统一)
        result = {name: 0 for name in cat_map.values()}
        for cid, count in counts.items():
            if cid in cat_map:
                result[cat_map[cid]] = count
        return result

def format_json(json_path):
    """返回格式化的json字符串"""
    # 1. 先生成标准的多行 JSON 字符串
    json_str = json.dumps(json_path, indent=2, ensure_ascii=False)

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

    return json_str

def split_train_val_coco(file_path, train_path, val_path, ratio = 0.2):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    info, categories = data.get('info', {}), data.get('categories', [])
    images, annotations = data.get('images', []), data.get('annotations', [])
    if not images or not annotations:
        print('没有有效的图像！')
        return

    defect_map = {category['id']: category['name'] for category in categories}
    random.shuffle(images)
    total_images = len(images)

    train_images = images[int(ratio*total_images):]
    val_images = images[:int(ratio*total_images)]

    train_image_id = set([image['id'] for image in train_images])
    val_image_id = set([image['id'] for image in val_images])
    train_annotations = [
        annotation for annotation in annotations if annotation['image_id'] in train_image_id
    ]
    val_annotations = [
        annotation for annotation in annotations if annotation['image_id'] in val_image_id
    ]

    train_info = copy.deepcopy(info)
    train_info['num_of_images'] = len(train_images)
    train_info['num_of_defects'] = get_defect_counts(train_annotations, defect_map) 

    val_info = copy.deepcopy(info)
    val_info['num_of_images'] = len(val_images)
    val_info['num_of_defects'] = get_defect_counts(val_annotations, defect_map)

    train = {
        'info': train_info,
        'categories': categories,
        'images': train_images,
        'annotations': train_annotations
    }

    val = {
        'info': val_info,
        'categories': categories,
        'images': val_images,
        'annotations': val_annotations
    }
    
    train_str = format_json(train)
    val_str = format_json(val)

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_str)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_str)   

if __name__ == "__main__":
    file_name = 'paint_stripe.json'
    stem = Path(file_name).stem
    root_dir = Path('data')
    file_path = root_dir / 'raw_label' /file_name
    train_path = root_dir / 'train_label' / f"{stem}_train.json"
    val_path = root_dir / 'val_label' / f"{stem}_val.json"
    print(file_path)
    print(train_path)
    print(val_path)
    
    split_train_val_coco(
        file_path=file_path,
        train_path=train_path,
        val_path=val_path
    )
