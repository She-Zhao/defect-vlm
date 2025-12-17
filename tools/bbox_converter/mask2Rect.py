"""
将mask转化为xml格式的检测文件, 针对磁瓦数据集
"""
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob

# 1. 生成 Pascal VOC 格式的 XML 文件
def create_voc_xml(filename, img_shape, boxes, defect_name, output_folder):
    annotation = ET.Element('annotation')
    
    # 文件名
    ET.SubElement(annotation, 'filename').text = filename
    
    # 图片尺寸信息
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    ET.SubElement(size, 'depth').text = str(img_shape[2])
    
    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = defect_name
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
    xml_filename = os.path.join(output_folder, filename.replace('.png', '.xml'))
    tree.write(xml_filename, encoding='utf-8')

# 2. 查找每个掩码的白色区域并计算最小外接矩形
def extract_bounding_boxes(mask):
    # 找到所有白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    
    return boxes

# 3. 主函数 - 处理每个PNG掩码文件，生成XML
def process_masks(mask_folder, defect_name, output_folder):
    # 获取所有PNG文件
    mask_files = glob(os.path.join(mask_folder, "*.png"))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for mask_file in mask_files:
        # 读取掩码图像
        # mask = cv2.imdecode(np.fromfile(mask_file,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否成功加载
        if mask is None:
            print(f"Error loading mask: {mask_file}")
            continue
        
        # 提取矩形框
        boxes = extract_bounding_boxes(mask)
        
        # 对应的JPEG文件名
        image_filename = mask_file.replace('.png', '.jpg')
        img = cv2.imread(image_filename)
        # img = cv2.imdecode(np.fromfile(image_filename,dtype=np.uint8),-1)
        
        if img is None:
            print(f"Error loading image: {image_filename}")
            continue
        
        # 创建XML文件
        create_voc_xml(os.path.basename(mask_file), img.shape, boxes, defect_name, output_folder)
        print(f"Processed {mask_file}")

# 4. 参数设置与调用
mask_folder = r'D:\Project\Defect_Dataset\MT\MT_Free'  # 掩码文件夹路径
output_folder = r'D:\Project\Defect_Dataset\MT\Free_xml'     # XML输出文件夹路径
defect_name = 'Free'              # 自定义的缺陷类型名称

process_masks(mask_folder, defect_name, output_folder)
