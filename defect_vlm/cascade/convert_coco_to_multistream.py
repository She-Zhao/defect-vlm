"""
将0_defect_dataset_raw/paint_stripe里面的 images 和 label（coco的json格式），
划分成为第一章多流主干网络需要的输入数据格式（yolo需要的txt格式）
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # 1. 定义源路径和目标路径
    source_root = Path("/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe")
    target_root = Path("/data/ZS/v11_input/datasets/row3")
    
    label_yolo_dir = source_root / "labels" / "yolo"
    train_json_path = source_root / "labels" / "train.json"
    val_json_path = source_root / "labels" / "val.json"

    # 2. 定义源文件夹到目标后缀的映射关系
    # 按照你的要求，共 6 个模态输入
    stream_mapping = {
        "16row0": "",    # 对应 train / val
        "16row1": "_1",  # 对应 train_1 / val_1
        "16row2": "_2",  # 对应 train_2 / val_2
        "32row0": "_3",  # 对应 train_3 / val_3
        "32row1": "_4",  # 对应 train_4 / val_4
        "32row2": "_5"   # 对应 train_5 / val_5
    }

    # 3. 解析 JSON 获取文件名列表
    def get_filenames_from_json(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 提取所有的 file_name
        return [item['file_name'] for item in data['images']]

    print("正在解析 JSON 文件...")
    splits = {
        "train": get_filenames_from_json(train_json_path),
        "val": get_filenames_from_json(val_json_path)
    }
    print(f"解析完成！找到 Train 图片 {len(splits['train'])} 张，Val 图片 {len(splits['val'])} 张。")

    # 4. 创建目标文件夹结构
    print("正在创建目标文件夹结构...")
    for split_name in splits.keys():
        for _, suffix in stream_mapping.items():
            target_img_dir = target_root / f"{split_name}{suffix}" / "images"
            target_lbl_dir = target_root / f"{split_name}{suffix}" / "labels"
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 5. 执行复制逻辑
    missing_images = 0
    missing_labels = 0

    for split_name, file_list in splits.items():
        # 遍历 JSON 中的每一个文件名 (例如: 0000_...png)
        for filename in tqdm(file_list, desc=f"复制 {split_name} 数据"):
            # 获取对应的 txt 标签文件名
            label_filename = os.path.splitext(filename)[0] + ".txt"
            source_lbl_path = label_yolo_dir / label_filename

            # 遍历 6 个视角
            for source_dir, suffix in stream_mapping.items():
                source_img_path = source_root / "images" / source_dir / filename
                
                target_img_path = target_root / f"{split_name}{suffix}" / "images" / filename
                target_lbl_path = target_root / f"{split_name}{suffix}" / "labels" / label_filename

                # 复制图片
                if source_img_path.exists():
                    shutil.copy2(source_img_path, target_img_path)
                else:
                    missing_images += 1
                    # print(f"\n[警告] 图片丢失: {source_img_path}") # 取消注释可查看具体丢失的图片

                # 复制标签（你的要求是均采用同一个 label）
                if source_lbl_path.exists():
                    shutil.copy2(source_lbl_path, target_lbl_path)
                else:
                    missing_labels += 1
                    # print(f"\n[警告] 标签丢失: {source_lbl_path}")

    print("-" * 50)
    print("✅ 数据集转换与复制完成！")
    if missing_images > 0 or missing_labels > 0:
        print(f"⚠️ 注意：过程中有 {missing_images} 次图片未找到，{missing_labels} 次标签未找到。")
        print("这通常是因为原始文件夹中缺失了对应的模态图片或标注。")

if __name__ == "__main__":
    main()
