"""
分析处理前数据集图像的分辨率，绘制散点图、气泡图
"""
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
import matplotlib
import math

def draw_scatter(root_dir: str, save_path: str) -> None:
    """绘制气泡图

    Args:
        root_dir (str): 各个数据集所在的根目录
        save_dir (str): 最终保存图像的路径
    """
    root_dir = Path(root_dir)
    dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    image_suffix = {'.jpg', '.png', '.bmp'}
    image_resolution = defaultdict(list)

    for d in dirs:
        image_dir = d / 'images'
        if image_dir.exists():
            print(f"找到目录：{image_dir}")
            image_paths = [img for img in image_dir.iterdir() if img.suffix.lower() in image_suffix]
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as img:
                        image_resolution[d].append(img.size)
                except Exception as e:
                    print(f"无法读取图像 {image_path}: {e}")
                
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)

    colors = matplotlib.colormaps['tab10']
    for i, (dir, img_res) in enumerate(image_resolution.items()):
        x, y = zip(*img_res)
        ax.scatter(x, y, marker='o', color=colors(i), alpha=0.5, edgecolors='white', label=dir.name)


    ax.set_title('resolution distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('image-width')
    ax.set_ylabel('image-height')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.axvline(x=640, color='red', linestyle='-', alpha=0.4)
    ax.axhline(y=640, color='red', linestyle='-', alpha=0.4)

    plt.savefig(save_path)


def draw_bubble(root_dir: str, save_path: str) -> None:
    """绘制气泡图

    Args:
        root_dir (str): 各个数据集所在的根目录
        save_dir (str): 最终保存图像的路径
    """
    root_dir = Path(root_dir)
    dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    image_suffix = {'.jpg', '.png', '.bmp'}
    image_resolution = defaultdict(Counter)

    for d in dirs:
        image_dir = d / 'images'
        if image_dir.exists():
            print(f"找到目录：{image_dir}")
            image_paths = [img for img in image_dir.iterdir() if img.suffix.lower() in image_suffix]
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as img:
                        image_resolution[d][img.size] += 1      # dir: Counter `(Width, Height): count`
                except Exception as e:
                    print(f"无法读取图像 {image_path}: {e}")
                
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)

    colors = matplotlib.colormaps['tab10']
    for i, (dir, counter) in enumerate(image_resolution.items()):
        points = counter.keys()
        counts = counter.values()           # 某个分辨率的图像数量越多，counts越大，对应的气泡越大
        sizes = [50*math.log(cnt) for cnt in counts]
        x, y = zip(*points)
        ax.scatter(
            x, y,
            marker='o',
            color=colors(i),
            s=sizes,
            alpha=0.5,
            edgecolors='white',
            label=dir.name
        )

    ax.set_title('resolution distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('image-width')
    ax.set_ylabel('image-height')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.axvline(x=640, color='red', linestyle='-', alpha=0.4)
    ax.axhline(y=640, color='red', linestyle='-', alpha=0.4)

    plt.savefig(save_path)


if __name__ == "__main__":
    root_dir = Path(r'D:\Project\Defect_Dataset')
    save_path1 = 'output/figures/原始图像分辨率分布散点图.jpg'
    save_path2 = 'output/figures/原始图像分辨率分布气泡图.jpg'
    # draw_scatter(root_dir=root_dir, save_path=save_path1)
    draw_bubble(root_dir=root_dir, save_path=save_path2)
