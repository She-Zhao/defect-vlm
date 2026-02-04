import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import re

# 自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def parse_xml_file(xml_file):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 标签统计
    label_count = defaultdict(int)
    
    # 遍历object标签，获取每个标签的name
    for obj in root.findall('object'):
        label = obj.find('name').text
        label_count[label] += 1
    
    return label_count

def process_folder(folder_path):
    # 用于存储每个文件的标签统计
    result = {}
    total_label_count = defaultdict(int)  # 用于统计所有文件中的标签总数
    
    # 遍历文件夹内所有的XML文件
    xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    
    # 对文件名进行自然排序
    xml_files.sort(key=natural_sort_key)
    
    # 解析并统计每个文件的标签
    for filename in xml_files:
        xml_file_path = os.path.join(folder_path, filename)
        label_count = parse_xml_file(xml_file_path)
        result[filename] = label_count
        
        # 累计总的缺陷种类和数量
        for label, count in label_count.items():
            total_label_count[label] += count
    
    return result, total_label_count

def save_results_to_excel(result, total_label_count, output_file):
    # 用于存储最终的DataFrame格式数据
    data = []
    
    for filename, label_count in result.items():
        # 每一行数据的格式：[xml文件名, 第一种缺陷名称, 第一种缺陷数量, 第二种缺陷名称, 第二种缺陷数量, ...]
        row = [filename]
        for label, count in label_count.items():
            row.append(label)
            row.append(count)
        data.append(row)
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(data)

    # 自动补齐列名
    max_cols = df.shape[1]  # 获取最大的列数
    col_names = ["Filename"]
    for i in range(1, max_cols // 2 + 1):
        col_names.extend([f"Defect_{i}_Name", f"Defect_{i}_Count"])
    df.columns = col_names
    
    # 创建总缺陷统计信息的DataFrame
    total_data = [["Defect", "Total Count"]]
    for label, count in total_label_count.items():
        total_data.append([label, count])
    total_df = pd.DataFrame(total_data)

    # 创建Excel writer对象，保存多个表格
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 写入总缺陷统计信息
        total_df.to_excel(writer, index=False, header=False, startrow=0)
        
        # 写入每个文件的缺陷详情
        df.to_excel(writer, index=False, startrow=len(total_data) + 2)

# 示例：指定文件夹路径和输出Excel文件
folder_path = r'D:\Project\Defect_Dataset\1_High\kaggle_steel\xml'  # 替换为你的XML文件夹路径
output_file = r'D:\Project\Defect_Dataset\1_High\kaggle_steel\xml\kaggle钢铁缺陷统计.xlsx'  # 输出Excel文件名称

# 处理文件夹中的XML文件
result, total_label_count = process_folder(folder_path)

# 保存结果到Excel
save_results_to_excel(result, total_label_count, output_file)

print(f"Results have been saved to {output_file}")
