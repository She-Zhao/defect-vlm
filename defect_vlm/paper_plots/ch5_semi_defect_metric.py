"""
计算半监督训练相较于GT训练各个缺陷指标的提升情况
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def parse_metrics_txt(filepath):
    """解析 Ultralytics 导出的详细 metrics txt 文件"""
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        # 过滤掉空行、表头、分割线以及 'all' 总体指标
        if len(parts) >= 5 and parts[0] not in ['Class', 'all', '-------------------------------------------------------------------']:
            class_name = parts[0]
            # 取最后两个浮点数为 mAP50 和 mAP50-95
            map50 = float(parts[-2])
            map95 = float(parts[-1])
            metrics[class_name] = {'map50': map50, 'map95': map95}
            
    return metrics

def plot_class_comparison(
    baseline_txt, ours_txt, output_dir, network_name, 
    times_font_path, zh_font_path
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 解析数据
    base_data = parse_metrics_txt(baseline_txt)
    ours_data = parse_metrics_txt(ours_txt)

    # 类别映射与排序 (保证 X 轴顺序一致)
    class_map = {
        'breakage': '破损', 'inclusion': '漆渣', 
        'scratch': '划痕', 'crater': '凹陷', 
        'run': '流挂', 'bulge': '凸起'
    }
    
    classes_en = list(class_map.keys())
    classes_zh = list(class_map.values())
    
    # 提取数值 (如果某个类找不到，默认填 0)
    base_m50 = [base_data.get(c, {}).get('map50', 0) for c in classes_en]
    base_m95 = [base_data.get(c, {}).get('map95', 0) for c in classes_en]
    ours_m50 = [ours_data.get(c, {}).get('map50', 0) for c in classes_en]
    ours_m95 = [ours_data.get(c, {}).get('map95', 0) for c in classes_en]

    # 2. 字体与样式精细化设置
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 13
    LEGEND_SIZE = 12
    ANNOTATE_SIZE = 10  # 柱子上方的数据标签字号

    # 英文与数字专属字体 (Times New Roman)
    font_en_label = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)
    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_en_anno = fm.FontProperties(fname=times_font_path, size=ANNOTATE_SIZE)
    
    # 中文专属字体 (宋体)
    font_zh_tick = fm.FontProperties(fname=zh_font_path, size=TICK_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    # 【核心魔法】启用 STIX 引擎处理数学/英文混排，视觉等效于 Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 配色：Baseline 蓝色，Ours 红色
    COLOR_BASE = '#4DBBD5'  
    COLOR_OURS = '#E64B35'  

    x = np.arange(len(classes_zh))
    width = 0.35

    # ==========================================
    # 3. 内部绘图函数 (绘制单张独立图像)
    # ==========================================
    def draw_and_save_bar_chart(data_base, data_ours, ylabel, save_filename):
        # 创建单张图，比例适中偏宽，适合论文排版
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

        # 绘制柱状图 (注意 label 中通过 $\mathrm{...}$ 强制渲染为新罗马字体)
        rects1 = ax.bar(x - width/2, data_base, width, 
                        label=r'Baseline', 
                        color=COLOR_BASE, alpha=0.9, edgecolor='white')
        rects2 = ax.bar(x + width/2, data_ours, width, 
                        label=r'Ours (静态半监督)', 
                        color=COLOR_OURS, alpha=0.9, edgecolor='white')

        # 在柱子上方标注精确数值
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 垂直偏移3个点
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontproperties=font_en_anno, color='#333333')

        autolabel(rects1)
        autolabel(rects2)

        # X 轴：中文类别
        ax.set_xticks(x)
        ax.set_xticklabels(classes_zh, fontproperties=font_zh_tick)
        
        # Y 轴：英文字母标签
        ax.set_ylabel(ylabel, fontproperties=font_en_label)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en_tick)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 扩大 Y 轴上限，留出空间放图例和数据标签
        max_val = max(max(data_base), max(data_ours))
        ax.set_ylim(0, max_val * 1.2)

        # 添加图例：置于正上方中心
        ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.12), ncol=2)

        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已独立保存图像: {save_filename}")

    # ==========================================
    # 4. 生成两张独立图片
    # ==========================================
    draw_and_save_bar_chart(
        base_m50, ours_m50, 
        ylabel='mAP@50', 
        save_filename=f'{network_name}_class_improvement_map50.png'
    )
    
    draw_and_save_bar_chart(
        base_m95, ours_m95, 
        ylabel='mAP@50:95', 
        save_filename=f'{network_name}_class_improvement_map50_95.png'
    )


if __name__ == "__main__":
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/静态伪标签'
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

    TXT_H_BASELINE = '/data/ZS/defect-vlm/output/yolo_decect/gt_row3/detailed_metrics.txt'
    TXT_H_OURS = '/data/ZS/defect-vlm/output/yolo_decect/iter0_row3/detailed_metrics.txt'
    
    TXT_V_BASELINE = '/data/ZS/defect-vlm/output/yolo_decect/gt_col3/detailed_metrics.txt'
    TXT_V_OURS = '/data/ZS/defect-vlm/output/yolo_decect/iter0_col3/detailed_metrics.txt'

    # 1. 绘制横向光 (YOLO_h) 的两张独立图像
    plot_class_comparison(
        baseline_txt=TXT_V_BASELINE, 
        ours_txt=TXT_V_OURS, 
        output_dir=OUTPUT_DIR, 
        network_name='YOLO_v', 
        times_font_path=TIMES_FONT_PATH, 
        zh_font_path=ZH_FONT_PATH
    )

    # 2. 绘制纵向光 (YOLO_v) 各类别提升对比 (确保文件路径存在后再解除注释)
    # plot_class_comparison(
    #     TXT_V_BASELINE, TXT_V_OURS, 
    #     output_dir=OUTPUT_DIR, save_filename='plot_class_improvement_v.png',
    #     network_name='YOLO_v', times_font_path=TIMES_FONT_PATH, zh_font_path=ZH_FONT_PATH
    # )
