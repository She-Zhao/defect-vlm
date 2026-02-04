"""
将多边形转化为xml格式的检测bbox, 针对kaggle-steel数据集
"""
import os
import csv
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob

# 1. 生成 Pascal VOC 格式的 XML 文件
def create_voc_xml(filename, img_shape, boxes, labels, defect_mapping, output_folder):
    annotation = ET.Element('annotation')
    
    # 文件名
    ET.SubElement(annotation, 'filename').text = filename
    
    # 图片尺寸信息
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    ET.SubElement(size, 'depth').text = str(img_shape[2])
    
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(annotation, 'object')
        # 根据缺陷类别ID进行缺陷名称的映射
        ET.SubElement(obj, 'name').text = defect_mapping[label]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(box[0])
        ET.SubElement(bndbox, 'ymin').text = str(box[1])
        ET.SubElement(bndbox, 'xmax').text = str(box[2])
        ET.SubElement(bndbox, 'ymax').text = str(box[3])
    
    # 保存XML文件
    tree = ET.ElementTree(annotation)
    xml_filename = os.path.join(output_folder, filename.replace('.jpg', '.xml'))
    tree.write(xml_filename, encoding='utf-8')

# 2. 解析"列位置-长度"格式，返回最小外接矩形框
def rle_to_boxes(rle_str, img_height=256, img_width=1600):
    rle = list(map(int, rle_str.split()))
    
    # 创建一个全黑的掩码图像
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # 遍历RLE格式，填充掩码图像
    for i in range(0, len(rle), 2):
        if i + 1 >= len(rle):
            # 如果长度不完整，跳过该部分
            print(f"Incomplete RLE data at index {i}, skipping this entry.")
            continue
        
        start = rle[i]
        length = rle[i+1]
        
        # 计算列号和行号
        col = start // img_height
        row_start = start % img_height
        row_end = row_start + length
        
        # 将掩码的对应区域设为白色（前景）
        mask[row_start:row_end, col] = 255
    
    # 查找前景区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    # 遍历所有找到的轮廓，计算最小外接矩形框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # 找到外接矩形
        boxes.append([x, y, x + w, y + h])  # [xmin, ymin, xmax, ymax]
    
    return boxes

# 3. 主函数 - 处理CSV文件并生成XML标签
def process_csv(csv_path, img_folder, output_folder, defect_mapping):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取CSV文件
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            img_name = row[0]
            
            # 检查第二列是否为空或无效
            if not row[1].strip():  # 如果是空字符串
                print(f"Warning: Missing defect class for image {img_name}, skipping this entry.")
                continue

            try:
                defect_class = int(row[1])
            except ValueError as e:
                print(f"Error: Invalid defect class '{row[1]}' for image {img_name}, skipping this entry.")
                continue
            rle = row[2]
            
            # 对应的图像路径
            img_path = os.path.join(img_folder, img_name)
            
            # 检查图像是否存在
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            # 假设图像是1600x256大小
            img_shape = (256, 1600, 3)  # 高度, 宽度, 通道数
            
            # 转换RLE为矩形框
            boxes = rle_to_boxes(rle, img_height=256, img_width=1600)
            
            # 生成XML文件
            create_voc_xml(img_name, img_shape, boxes, [defect_class] * len(boxes), defect_mapping, output_folder)
            print(f"Processed {img_name}")

# 4. 参数设置与调用
csv_path = r'D:\Project\Defect_Dataset\kaggle-steel\train.csv'   # CSV文件路径
img_folder = r'D:\Project\Defect_Dataset\kaggle-steel\train_images' # 图像文件夹路径
output_folder = r'D:\Project\Defect_Dataset\kaggle-steel\xml'     # XML输出文件夹路径

# 缺陷类别映射，可根据实际需要进行修改
defect_mapping = {
    1: 'class_1',  # 例如，1代表class_1
    2: 'class_2',  # 例如，2代表class_2
    3: 'class_3',  # 例如，3代表class_3
    4: 'class_4'   # 例如，4代表class_4
}

# 执行处理函数
process_csv(csv_path, img_folder, output_folder, defect_mapping)
