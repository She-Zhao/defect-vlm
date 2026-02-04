import json
import re
from tqdm import tqdm

def rename(old_name: str) -> str:
    """
    转换文件名格式：
    旧：73_2741_1748_2132_300_300_0_2432_2048.png
    新：00_0073_002741_1748_2132_300_300.png
    """
    # 假设 old_name 包含后缀，先分离后缀（虽然你的示例逻辑里是直接split，这里保留你原本的处理逻辑）
    # 如果 old_name 是纯文件名不带路径
    parts = old_name.split('_')
    
    # 注意：split后最后一部分通常带有 .png，需要小心处理
    # 如果你的原始逻辑已经经过验证，这里保持不变
    # 下面基于你提供的逻辑进行微调，确保 .png 后缀处理正确
    if old_name.endswith('.png'):
        suffix = '.png'
        stem = old_name[:-4]
        parts = stem.split('_')
    else:
        suffix = ''
        parts = old_name.split('_')

    try:
        part1, part2, part3, part4, part5, part6, _, _, _ = parts
        
        # 转换逻辑
        new_part1 = part1.zfill(4)  # 73 -> 0073
        new_part2 = part2.zfill(6)  # 2741 -> 002741
        part3 = part3.zfill(4)
        part4 = part4.zfill(4)
        
        new_name = f"00_{new_part1}_{new_part2}_{part3}_{part4}_{part5}_{part6}{suffix}"
        return new_name
    except ValueError:
        # 如果文件名格式不符合预期（比如 split 出来的数量不对），返回原名或报错
        print(f"Warning: format error for {old_name}")
        return old_name

def save_coco_json_formatted(data, save_path):
    """
    保存 COCO 格式 JSON，并对 bbox 和 categories 进行压缩格式化
    """
    print("正在生成 JSON 字符串...")
    # 1. 先生成标准的多行 JSON 字符串
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    print("正在应用正则格式化...")
    # 2.【bbox处理】将 bbox 列表压缩为一行
    # 匹配模式： "bbox": [ 换行 数字, 换行 数字 ... ]
    json_str = re.sub(
        r'("bbox": \[)([^\]]+)(\])', 
        lambda m: m.group(1) + ", ".join(x.strip() for x in m.group(2).split(',')) + m.group(3), 
        json_str
    )

    # 3.【categories处理】将 categories 中的字典压缩为一行
    # 匹配模式：{ "id": 数字, "name": "字符串" } 及其之间的空白
    json_str = re.sub(
        r'\{\s*"id":\s*(\d+),\s*"name":\s*"([^"]+)"\s*\}',
        r'{"id": \1, "name": "\2"}',
        json_str
    )

    # 4. 写入文件
    print(f"正在写入文件: {save_path}")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("保存完成。")

# ================= 主逻辑 =================

json_path = '/data/ZS/defect_dataset/0_defect_dataset_raw/paint_ap/labels/val.json'
save_path = '/data/ZS/defect_dataset/0_defect_dataset_raw/paint_ap/labels/val1.json'

# 读取数据
print(f"正在读取: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理文件名
images = data['images']
# 使用 tqdm 显示进度
for image in tqdm(images, desc="Renaming files"):
    original_name = image['file_name']
    new_name = rename(original_name)
    image['file_name'] = new_name
    # print(f"转换前：{original_name} -> 转换后：{new_name}") # 如果数据量大，建议注释掉 print

# 使用带有格式化功能的保存函数
save_coco_json_formatted(data, save_path)
