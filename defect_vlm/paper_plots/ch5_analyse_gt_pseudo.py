"""
统计目录下所有 .txt 文件中各类别缺陷的数量
labels_dir: 包含 YOLO 标注文件的目录
class_names: dict, {class_id: class_name}
"""

import os
from collections import defaultdict

def count_yolo_labels(labels_dir, class_names):
    """
    统计目录下所有 .txt 文件中各类别缺陷的数量
    labels_dir: 包含 YOLO 标注文件的目录
    class_names: dict, {class_id: class_name}
    """
    counts = defaultdict(int)
    
    # 遍历所有 txt 文件
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
    
    # 输出统计结果
    print(f"{'Class ID':<8} {'Class Name':<12} {'Count':>8}")
    print("-" * 30)
    total = 0
    for cls_id in sorted(class_names.keys()):
        cnt = counts.get(cls_id, 0)
        total += cnt
        print(f"{cls_id:<8} {class_names[cls_id]:<12} {cnt:>8}")
    print("-" * 30)
    print(f"{'Total':<8} {'':<12} {total:>8}")
    return counts

if __name__ == "__main__":
    # LABELS_DIR = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter0/col3/train/labels"
    LABELS_DIR = "/data/ZS/v11_input/datasets/col3/train/labels"
    CLASS_NAMES = {
        0: "breakage",
        1: "inclusion",
        2: "scratch",
        3: "crater",
        4: "run",
        5: "bulge"
    }
    count_yolo_labels(LABELS_DIR, CLASS_NAMES)