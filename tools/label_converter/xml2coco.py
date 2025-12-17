"""
将 XML (Pascal VOC) 格式数据集转换为 COCO 格式
"""
import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

def parse_xml_bbox(bndbox_node: ET.Element) -> Tuple[int, int, int, int]:
    """解析 XML 中的 bndbox 节点并转换为 COCO bbox [x, y, w, h]"""
    xmin = float(bndbox_node.find('xmin').text)
    ymin = float(bndbox_node.find('ymin').text)
    xmax = float(bndbox_node.find('xmax').text)
    ymax = float(bndbox_node.find('ymax').text)

    # 计算宽高
    width = xmax - xmin
    height = ymax - ymin

    # 防御性转换：确保是整数且不小于0
    return int(xmin), int(ymin), max(0, int(width)), max(0, int(height))

def format_json_str(coco_dict: Dict[str, Any]) -> str:
    """将字典转换为格式化好的 JSON 字符串 (bbox 和 categories 压缩为单行)"""
    # 1. 生成基础缩进字符串
    json_str = json.dumps(coco_dict, indent=2, ensure_ascii=False)

    # 2. 压缩 bbox: "bbox": [1, 2, 3, 4]
    json_str = re.sub(
        r'("bbox": \[)([^\]]+)(\])',
        lambda m: m.group(1) + ", ".join(x.strip() for x in m.group(2).split(',')) + m.group(3),
        json_str
    )

    # 3. 压缩 categories: {"id": 1, "name": "foo"}
    json_str = re.sub(
        r'\{\s*"id":\s*(\d+),\s*"name":\s*"([^"]+)"\s*\}',
        r'{"id": \1, "name": "\2"}',
        json_str
    )

    return json_str

def xml2coco(
    xml_dir: str | Path,
    save_path: str | Path,
    categories: List[Dict[str, Any]],
    dataset_source: str = "custom_dataset"
) -> None:

    xml_dir = Path(xml_dir)
    save_path = Path(save_path)

    # 建立 name -> id 的映射表
    # 例如: {'crazing': 0, 'inclusion': 1, ...}
    name_to_id = {cat['name']: cat['id'] for cat in categories}

    # 初始化 COCO 结构
    info = {
        "description": dataset_source,
        "total_images": 0,
        "total_defects": 0,
        "num_of_defects": {name: 0 for name in name_to_id.keys()}
    }

    images = []
    annotations = []

    # 获取所有 xml 文件
    xml_files = list(xml_dir.glob('*.xml'))
    info['total_images'] = len(xml_files)

    image_id = 1
    annotation_id = 1

    print(f"🚀 开始处理 {len(xml_files)} 个 XML 文件...")

    for xml_file in tqdm(xml_files, desc="Converting"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # --- 1. 解析图像信息 ---
            filename = root.find('filename').text

            # 尝试获取图像尺寸 (XML中通常包含 size 节点)
            size_node = root.find('size')
            if size_node is not None:
                width = int(size_node.find('width').text)
                height = int(size_node.find('height').text)
            else:
                # 如果 XML 没写尺寸，这里可以留空或者尝试去读图片文件
                print(f"⚠️ Warning: {xml_file.name} 没有 size 信息，默认设为 0")
                width, height = 0, 0

            image_item = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "dataset_source": dataset_source
            }
            images.append(image_item)

            # --- 2. 解析标注信息 (Object) ---
            for obj in root.findall('object'):
                obj_name = obj.find('name').text

                # 检查这个类别是否在我们定义的 categories 列表里
                if obj_name not in name_to_id:
                    print(f"⚠️ 跳过未知类别: {obj_name} (在文件 {xml_file.name} 中)")
                    continue

                cat_id = name_to_id[obj_name]
                bndbox = obj.find('bndbox')

                x, y, w, h = parse_xml_bbox(bndbox)

                ann_item = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": int(w * h),
                    "iscrowd": 0
                }
                annotations.append(ann_item)

                # 更新统计信息
                info['num_of_defects'][obj_name] += 1
                annotation_id += 1

            image_id += 1

        except Exception as e:
            print(f"❌ 解析错误 {xml_file.name}: {e}")
            continue
    
    info['total_defects'] = sum(info["num_of_defects"].values())
    # 构建最终字典
    coco_format = {
        "info": info,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    # 写入文件
    print(f"💾 正在保存到 {save_path} ...")
    json_output = format_json_str(coco_format)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json_output)

    print("✅ 转换完成！")

if __name__ == "__main__":
    xml_dir = r'D:\Project\Defect_Dataset\FXP\labels'
    save_path = 'data/FXP.json'

    # ⚠️【重要】请在这里定义你的类别映射
    # XML 中的 <name> 必须与这里的 "name" 一致，否则会被跳过
    # al-primary
    MY_CATEGORIES = [
        {"id": 0, "name": "scratch"},
        {"id": 1, "name": "smudge"},
        {"id": 2, "name": "pinhole"},
        {"id": 3, "name": "loose thread"},
        {"id": 4, "name": "crater"},
        {"id": 5, "name": "exposed stitch"},
        {"id": 6, "name": "brand"},
        {"id": 7, "name": "wrinkle"},
        {"id": 8, "name": "damaged surface"},
        {"id": 9, "name": "irregular surface"},
        {"id": 10, "name": "skipping stitch"},
        {"id": 11, "name": "instruction symbol"},
        {"id": 12, "name": "quality mark"},
        {"id": 13, "name": "foaming"},
        {"id": 14, "name": "leather defect"},        
    ]

    xml2coco(
        xml_dir=xml_dir,
        save_path=save_path,
        categories=MY_CATEGORIES,
        dataset_source="FXP"         # 修改描述
    )

