import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

def smooth_curve(scalars, weight=0.9):
    last = scalars.iloc[0] if not pd.isna(scalars.iloc[0]) else 0
    smoothed = []
    for point in scalars:
        if pd.isna(point):
            smoothed.append(point)
            continue
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_individual_yolo_metrics(csv_path, output_dir, times_font_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 Times New Roman 字体
    if not os.path.exists(times_font_path):
        raise FileNotFoundError(f"找不到英文字体文件: {times_font_path}")
    
    # 字体大小统一增大
    TITLE_SIZE = 16
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 14
    LEGEND_SIZE = 12
    
    font_en = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)

    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    epochs = df['epoch']

    metrics_to_plot = [
        ('time', 'time', 'time'),
        ('train/box_loss', 'train_box_loss', 'train/box_loss'),
        ('train/cls_loss', 'train_cls_loss', 'train/cls_loss'),
        ('train/dfl_loss', 'train_dfl_loss', 'train/dfl_loss'),
        ('metrics/precision(B)', 'metrics_precision', 'metrics/precision(B)'),
        ('metrics/mAP50-95(B)', 'metrics_mAP50_95', 'metrics/mAP50-95(B)'),
        ('val/box_loss', 'val_box_loss', 'val/box_loss'),
        ('val/cls_loss', 'val_cls_loss', 'val/cls_loss'),
        ('metrics/recall(B)', 'metrics_recall', 'metrics/recall(B)'),
        ('metrics/mAP50(B)', 'metrics_mAP50', 'metrics/mAP50(B)')
    ]

    ACADEMIC_BLUE = '#1f77b4'
    ACADEMIC_ORANGE = '#ff7f0e'

    for col_name, save_name, title in metrics_to_plot:
        if col_name not in df.columns:
            print(f"⚠️ 警告: 找不到列 '{col_name}'，跳过绘制。")
            continue

        y_data = df[col_name]

        # 调整图幅：宽度3英寸，高度4英寸（细长比例 3:4）
        fig, ax = plt.subplots(figsize=(3, 4), dpi=300)

        ax.plot(epochs, y_data, marker='.', markersize=5, linestyle='-',
                linewidth=1.5, color=ACADEMIC_BLUE, alpha=0.8, label='results')

        if col_name != 'time':
            y_smooth = smooth_curve(y_data, weight=0.85)
            ax.plot(epochs, y_smooth, linestyle='--', linewidth=2,
                    color=ACADEMIC_ORANGE, label='smooth')

        # 设置标题和坐标轴标签，使用增大后的字体
        ax.set_title(title, fontproperties=fm.FontProperties(fname=times_font_path, size=TITLE_SIZE))
        ax.set_xlabel('Epoch', fontproperties=font_en)
        ax.set_ylabel(title, fontproperties=font_en)

        # 设置刻度标签字体并增大字号
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(fm.FontProperties(fname=times_font_path, size=TICK_SIZE))
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fm.FontProperties(fname=times_font_path, size=TICK_SIZE))

        # 美化：去掉上、右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 图例（仅对第一个指标添加，避免冗余）
        if col_name == 'train/box_loss':
            legend = ax.legend(frameon=True, fontsize=LEGEND_SIZE, prop=fm.FontProperties(fname=times_font_path, size=LEGEND_SIZE))

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{save_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_path}")

if __name__ == "__main__":
    CSV_PATH = '/data/ZS/v11_input/runs/实验结果/exp2_row3_part_max_cbam/results.csv'
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/yolo训练过程'
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    plot_individual_yolo_metrics(CSV_PATH, OUTPUT_DIR, TIMES_FONT_PATH)