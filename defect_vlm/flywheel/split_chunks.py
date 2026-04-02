"""
无标签数据集分块脚本 (Chunk Splitter)
用于 VLM 数据飞轮的渐进式摄入。保证随机性与绝对可复现性。
"""
import random
import json
from pathlib import Path

def split_into_chunks(img_dir, output_dir, num_chunks=3, seed=42):
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    
    # 自动创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 正在扫描图像目录: {img_dir}")
    
    # 1. 获取所有图像文件名并排序（极其重要：保证跨平台的绝对可复现性）
    image_files = sorted([f.name for f in img_dir.glob('*.png')])
    
    total_imgs = len(image_files)
    print(f"📂 共找到 {total_imgs} 张无标注图像。")

    if total_imgs == 0:
        print("⚠️ 未找到任何图像，请检查路径是否正确。")
        return

    # 2. 设置随机种子并全局打乱
    random.seed(seed)
    random.shuffle(image_files)

    # 3. 计算每个 chunk 的基础大小并切分
    chunk_size = total_imgs // num_chunks
    chunks = {}

    for i in range(num_chunks):
        start_idx = i * chunk_size
        # 最后一个 chunk 包揽所有剩余的图像（处理不能整除的情况）
        if i == num_chunks - 1:
            end_idx = total_imgs
        else:
            end_idx = start_idx + chunk_size
            
        chunk_name = f"chunk{i+1}"
        chunks[chunk_name] = image_files[start_idx:end_idx]

    # 4. 方案 A：保存为 JSON 文件 (推荐后续 Python 脚本读取)
    json_path = output_dir / "chunk_splits.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)

    # 5. 方案 B：保存为 TXT 文件 (推荐人工核对和 Bash 脚本读取)
    for chunk_name, files in chunks.items():
        txt_path = output_dir / f"{chunk_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(files))

    print("\n" + "="*50)
    print(f"✅ 成功将数据均匀切分为 {num_chunks} 组！(Seed: {seed})")
    print("="*50)
    for chunk_name, files in chunks.items():
        print(f"   ├── {chunk_name}: {len(files)} 张图像")
    print("="*50)
    print(f"💾 JSON 汇总文件保存至: {json_path}")
    print(f"📄 TXT  清单文件保存至: {output_dir}/")


if __name__ == '__main__':
    # 配置输入和输出路径
    IMG_DIR = "/data/ZS/flywheel_dataset/0_multi_input/sp012/col3/val/images/"
    OUTPUT_DIR = "/data/ZS/flywheel_dataset/chunks/"
    
    # 执行切分
    split_into_chunks(IMG_DIR, OUTPUT_DIR)
