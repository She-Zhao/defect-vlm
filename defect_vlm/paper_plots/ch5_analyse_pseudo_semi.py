"""
分析静态半监督数据集的分布和 GT 之间的数据分布
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

def parse_yolo_labels(file_list, base_dir, is_pseudo=False, img_size=300):
    """解析 YOLO 格式标签文件，计算面积和尺度分类"""
    data = []
    for f in file_list:
        if not f.endswith('.txt'):
            continue
        file_path = os.path.join(base_dir, f)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h * img_size * img_size
                    weight = float(parts[5]) if is_pseudo and len(parts) >= 6 else 1.0
                    data.append({
                        'class_id': class_id, 
                        'w': w, 
                        'h': h, 
                        'area': area, 
                        'weight': weight
                    })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['scale'] = pd.cut(
            df['area'], 
            bins=[0, 32**2, 96**2, float('inf')], 
            labels=['小目标', '中目标', '大目标']
        )
    return df

def analyze_and_plot_distributions(gt_dir, mixed_dir, output_dir, times_font_path, zh_font_path, img_size=300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 字体设置
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 14
    LEGEND_SIZE = 12

    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_zh = fm.FontProperties(fname=zh_font_path, size=AXIS_LABEL_SIZE)
    font_zh_tick = fm.FontProperties(fname=zh_font_path, size=TICK_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    # 2. 区分 GT 和 伪标签文件
    gt_files = set([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
    mixed_files = set([f for f in os.listdir(mixed_dir) if f.endswith('.txt')])
    pseudo_files = mixed_files - gt_files

    print(f"总计找到 GT 图像文件数: {len(gt_files)}")
    print(f"总计找到 伪标签图像文件数: {len(pseudo_files)}")

    # 3. 解析数据
    gt_df = parse_yolo_labels(gt_files, gt_dir, is_pseudo=False, img_size=img_size)
    pseudo_df = parse_yolo_labels(pseudo_files, mixed_dir, is_pseudo=True, img_size=img_size)

    class_map = {0: '破损', 1: '漆渣', 2: '划痕', 3: '凹陷', 4: '流挂', 5: '凸起'}
    gt_df['class_name'] = gt_df['class_id'].map(class_map)
    pseudo_df['class_name'] = pseudo_df['class_id'].map(class_map)

    # 4. 生成统计数据 TXT
    stats_path = os.path.join(output_dir, 'statistics_for_tables.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("========== 表 1: 类别分布统计 (%) ==========\n")
        f.write(f"{'类别':<10} | {'GT 占比 (%)':<15} | {'伪标签 占比 (%)':<15}\n")
        f.write("-" * 50 + "\n")
        
        gt_class_pct = gt_df['class_name'].value_counts(normalize=True) * 100
        ps_class_pct = pseudo_df['class_name'].value_counts(normalize=True) * 100
        
        for cls_name in class_map.values():
            gt_val = gt_class_pct.get(cls_name, 0)
            ps_val = ps_class_pct.get(cls_name, 0)
            f.write(f"{cls_name:<10} | {gt_val:<15.2f} | {ps_val:<15.2f}\n")
            
        f.write("\n========== 表 2: 尺度(目标大小)分布统计 (%) ==========\n")
        f.write(f"{'尺度':<10} | {'GT 占比 (%)':<15} | {'伪标签 占比 (%)':<15}\n")
        f.write("-" * 50 + "\n")
        
        gt_scale_pct = gt_df['scale'].value_counts(normalize=True) * 100
        ps_scale_pct = pseudo_df['scale'].value_counts(normalize=True) * 100
        
        for scale in ['小目标', '中目标', '大目标']:
            gt_val = gt_scale_pct.get(scale, 0)
            ps_val = ps_scale_pct.get(scale, 0)
            f.write(f"{scale:<10} | {gt_val:<15.2f} | {ps_val:<15.2f}\n")

    # 5. 绘图通用设置 (已移除 Title 逻辑)
    ACADEMIC_BLUE = '#1f77b4'
    ACADEMIC_ORANGE = '#ff7f0e'
    
    def apply_style(ax, xlabel, ylabel, x_is_zh=True):
        ax.set_xlabel(xlabel, fontproperties=font_zh)
        ax.set_ylabel(ylabel, fontproperties=font_zh)
        
        # Y轴固定为数字，强制用 Times New Roman
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en_tick)
            
        # X轴根据内容决定
        if not x_is_zh:
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_en_tick)
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # =========================================================
    # 图 1: 类别分布对比柱状图 (调整为窄高比例 4:5)
    # =========================================================
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    x = np.arange(len(class_map))
    width = 0.35
    
    classes = list(class_map.values())
    gt_vals = [gt_class_pct.get(c, 0) for c in classes]
    ps_vals = [ps_class_pct.get(c, 0) for c in classes]
    
    ax.bar(x - width/2, gt_vals, width, label='真实标签', color=ACADEMIC_BLUE, alpha=0.8)
    ax.bar(x + width/2, ps_vals, width, label='生成伪标签', color=ACADEMIC_ORANGE, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontproperties=font_zh_tick)
    
    apply_style(ax, '缺陷类别', '实例占比 (%)', x_is_zh=True)
    
    # 调整图例位置，避免遮挡窄图中的柱子
    ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_class_distribution.png'), bbox_inches='tight')
    plt.close()

    # =========================================================
    # 图 2: 尺度分布对比柱状图 (调整为窄高比例 4:5)
    # =========================================================
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    scales = ['小目标', '中目标', '大目标']
    x_scale = np.arange(len(scales))
    
    gt_scale_vals = [gt_scale_pct.get(s, 0) for s in scales]
    ps_scale_vals = [ps_scale_pct.get(s, 0) for s in scales]
    
    ax.bar(x_scale - width/2, gt_scale_vals, width, label='真实标签', color=ACADEMIC_BLUE, alpha=0.8)
    ax.bar(x_scale + width/2, ps_scale_vals, width, label='生成伪标签', color=ACADEMIC_ORANGE, alpha=0.8)

    ax.set_xticks(x_scale)
    ax.set_xticklabels(scales, fontproperties=font_zh_tick)
    
    apply_style(ax, '缺陷尺度', '实例占比 (%)', x_is_zh=True)
    
    ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_scale_distribution.png'), bbox_inches='tight')
    plt.close()

    # =========================================================
    # 图 3: 边界框面积核密度估计曲线 (KDE) (调整为窄高比例 4:5)
    # =========================================================
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    gt_area_log = np.log1p(gt_df['area'])
    ps_area_log = np.log1p(pseudo_df['area'])
    
    sns.kdeplot(gt_area_log, fill=True, color=ACADEMIC_BLUE, label='真实标签', ax=ax, alpha=0.3, linewidth=2)
    sns.kdeplot(ps_area_log, fill=True, color=ACADEMIC_ORANGE, label='生成伪标签', ax=ax, alpha=0.3, linewidth=2)

    apply_style(ax, '边界框对数面积', '概率密度', x_is_zh=False)
    
    ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_area_kde_density.png'), bbox_inches='tight')
    plt.close()

    print("✅ 所有图像已成功绘制并保存！(已调整为窄高比例且移除图标题)")

if __name__ == "__main__":
    GT_DIR = '/data/ZS/v11_input/datasets/col3/train/labels'
    MIXED_DIR = '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter0/col3/train/labels'
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/静态伪标签'
    
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

    analyze_and_plot_distributions(
        gt_dir=GT_DIR,
        mixed_dir=MIXED_DIR,
        output_dir=OUTPUT_DIR,
        times_font_path=TIMES_FONT_PATH,
        zh_font_path=ZH_FONT_PATH, 
        img_size=300
    )