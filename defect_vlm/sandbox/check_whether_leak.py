"""
检查是否有验证集泄露到了训练集当中
"""

import json
import os
from pathlib import Path
from typing import List, Set

def get_all_jsonl_files(folder_path: str) -> List[str]:
    """
    递归获取指定文件夹下的所有 .jsonl 文件
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ 警告: 找不到文件夹 {folder_path}")
        return []
    
    # 使用 rglob 递归查找所有 jsonl 文件
    files = [str(p) for p in Path(folder_path).rglob("*.jsonl")]
    return files

def extract_image_paths(jsonl_files: List[str]) -> Set[str]:
    """
    遍历 JSONL 文件列表，提取所有图像的绝对路径
    """
    full_paths = set()
    
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # 提取 images 列表中的所有绝对路径并加入集合
                    for img_path in data.get('images', []):
                        full_paths.add(img_path)
                except Exception as e:
                    print(f"解析错误在 {file_path} 第 {line_num} 行: {e}")
                    
    return full_paths

def main(dir_1: str, dir_2: str):
    print(f"📂 正在读取训练集目录: {dir_1}")
    files_1 = get_all_jsonl_files(dir_1)
    print(f"找到 {len(files_1)} 个 JSONL 文件，正在扫描图像路径...")
    train_paths = extract_image_paths(files_1)
    print(f"✅ 训练集共提取到 {len(train_paths)} 张独立图像路径。\n")

    print(f"📂 正在读取比对目录: {dir_2}")
    files_2 = get_all_jsonl_files(dir_2)
    print(f"找到 {len(files_2)} 个 JSONL 文件，正在扫描图像路径...")
    test_paths = extract_image_paths(files_2)
    print(f"✅ 比对集共提取到 {len(test_paths)} 张独立图像路径。\n")

    print("="*50)
    print("🚨 正在进行数据泄露严格比对 (绝对路径级别)...")
    
    # 1. 核心逻辑：计算集合的交集
    leaked_paths = train_paths.intersection(test_paths)

    # 2. 输出诊断报告
    if len(leaked_paths) == 0:
        print("\n🎉 恭喜！未检测到任何数据泄露！")
        print("这两个文件夹中的图像绝对路径完全没有交集。你的模型指标是真实的！")
    else:
        print(f"\n⚠️ 致命警报！检测到数据泄露！")
        print(f"❌ 发现 {len(leaked_paths)} 张相同的图像同时存在于两端！")
        print("示例重合的绝对路径 (前 5 个):")
        for p in list(leaked_paths)[:5]:
            print(f"  - {p}")
    print("="*50)

if __name__ == "__main__":
    # 1. 配置你的文件夹路径 (只要指定到最外层文件夹即可)
    # 比如训练集文件夹、测试集文件夹或验证集文件夹
    train_dir = "/data/ZS/defect_dataset/7_swift_dataset/train"
    # test_dir = "/data/ZS/defect_dataset/7_swift_dataset/student/test"
    val_dir = "/data/ZS/defect_dataset/7_swift_dataset/val" # 如果你想查验证集，替换掉 test_dir 即可
    
    main(train_dir, val_dir)