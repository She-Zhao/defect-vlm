import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签，如果没有 SimHei 可改为 DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def analyze_bbox(json_path: str, save_dir: str):
    json_path = Path(json_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载标注文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'annotations' not in data:
        print("错误: JSON文件中未找到 'annotations' 字段")
        return

    widths = []
    heights = []
    aspect_ratios = [] # 宽高比 = w / h
    areas = []

    print("正在提取 BBox 信息...")
    for ann in tqdm(data['annotations']):
        # COCO 格式: [x, y, width, height]
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            w, h = bbox[2], bbox[3]
            # 过滤掉异常数据
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                areas.append(w * h)

    if len(widths) == 0:
        print("未找到有效的 BBox 数据。")
        return

    # 转换为 DataFrame 方便绘图
    df = pd.DataFrame({
        'Width': widths,
        'Height': heights,
        'Aspect Ratio (W/H)': aspect_ratios,
        'Area': areas
    })

    # --- 1. 打印统计信息 ---
    print("\n" + "="*40)
    print(" BBox 统计分析结果")
    print("="*40)
    print(f"{'指标':<10} {'Width':<10} {'Height':<10} {'Ratio':<10}")
    print("-" * 40)
    print(f"{'Min':<10} {np.min(widths):<10.2f} {np.min(heights):<10.2f} {np.min(aspect_ratios):<10.2f}")
    print(f"{'Max':<10} {np.max(widths):<10.2f} {np.max(heights):<10.2f} {np.max(aspect_ratios):<10.2f}")
    print(f"{'Mean':<10} {np.mean(widths):<10.2f} {np.mean(heights):<10.2f} {np.mean(aspect_ratios):<10.2f}")
    print(f"{'Median':<10} {np.median(widths):<10.2f} {np.median(heights):<10.2f} {np.median(aspect_ratios):<10.2f}")
    print("="*40)

    # --- 2. 绘制 Width 和 Height 的联合分布图 (Joint Plot) ---
    print("正在绘制 宽 vs 高 联合分布图...")
    g = sns.jointplot(x="Width", y="Height", data=df, kind="scatter", s=10, alpha=0.6, height=8)
    g.fig.suptitle("BBox Width vs Height Distribution", y=1.02)
    
    # 画出 1:1, 1:2, 2:1 的辅助线
    max_val = max(np.max(widths), np.max(heights))
    g.ax_joint.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='1:1')
    g.ax_joint.legend()
    
    save_path_1 = save_dir / "bbox_width_height_joint.png"
    plt.savefig(save_path_1)
    plt.close()

    # --- 3. 绘制单独的分布直方图 (Dist Plot) ---
    print("正在绘制分布直方图...")
    plt.figure(figsize=(15, 5))

    # Width 分布
    plt.subplot(1, 3, 1)
    sns.histplot(df['Width'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Width')
    plt.xlabel('Width (px)')

    # Height 分布
    plt.subplot(1, 3, 2)
    sns.histplot(df['Height'], bins=50, kde=True, color='orange')
    plt.title('Distribution of Height')
    plt.xlabel('Height (px)')

    # Aspect Ratio 分布 (使用对数坐标可能更好，这里先用普通坐标)
    plt.subplot(1, 3, 3)
    sns.histplot(df['Aspect Ratio (W/H)'], bins=50, kde=True, color='green')
    plt.title('Distribution of Aspect Ratio (W/H)')
    plt.xlabel('Ratio')
    # 如果 Ratio 差异很大（如长尾），可以限制x轴范围方便观察
    # plt.xlim(0, 10) 

    plt.tight_layout()
    save_path_2 = save_dir / "bbox_distributions.png"
    plt.savefig(save_path_2)
    plt.close()
    
    # --- 4. 绘制面积分布 ---
    print("正在绘制面积分布图...")
    plt.figure(figsize=(8, 6))
    # 面积通常跨度很大，建议看对数分布或者过滤掉极端值
    sns.histplot(df['Area'], bins=50, kde=True, color='purple')
    plt.title('Distribution of BBox Area')
    plt.xlabel('Area (px^2)')
    
    save_path_3 = save_dir / "bbox_area.png"
    plt.savefig(save_path_3)
    plt.close()

    print(f"\n分析完成！图片已保存至: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 COCO 格式标注的 BBox 分布")
    parser.add_argument('--json_path', type=str, required=True, help='COCO 格式的 json 文件路径')
    parser.add_argument('--save_dir', type=str, default='./analysis_result', help='结果保存目录')
    
    args = parser.parse_args()
    
    analyze_bbox(args.json_path, args.save_dir)