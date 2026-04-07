"""
统计数据飞轮三次迭代过程中的 map@50 和 map@50:95 的变化情况
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_iteration_metrics(
    output_dir, times_font_path, zh_font_path,
    yolo_h_50, yolo_v_50, static_h_50, static_v_50,
    yolo_h_95, yolo_v_95, static_h_95, static_v_95
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # 1. 字体与全局样式设置
    # ==========================================
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 13
    LEGEND_SIZE = 12
    ANNOTATE_SIZE = 11

    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_en_label = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)
    font_en_anno = fm.FontProperties(fname=times_font_path, size=ANNOTATE_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    # 【核心魔法】使用 STIX 字体引擎渲染数学公式模式下的英文字母
    # STIX 字体在学术界等价于 Times New Roman，完美解决图例中英混排问题
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 顶会高级配色：玫瑰粉 (YOLO_h) & 克莱因蓝 (YOLO_v)
    COLOR_H = '#E64B35'  
    COLOR_V = '#4DBBD5'  
    
    x_labels = ['Baseline\n(GT)', 'Iter 1', 'Iter 2', 'Iter 3']
    x = np.arange(len(x_labels))

    # ==========================================
    # 内部绘图函数 (绘制并保存单张独立图像)
    # ==========================================
    def draw_and_save_figure(data_h, data_v, static_h, static_v, ylabel, save_filename):
        # 创建单张图像，图幅比例适中偏方，适合两张图并在 LaTeX 一排
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

        # 画静态半监督的天花板 (虚线)
        ax.axhline(y=static_h, color=COLOR_H, linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axhline(y=static_v, color=COLOR_V, linestyle='--', alpha=0.4, linewidth=1.5)
        
        # 在虚线旁边标注文字 (靠左侧)
        ax.text(0.1, static_h + 0.001, f'Static Semi ({static_h:.3f})', 
                color=COLOR_H, alpha=0.7, fontproperties=font_en_anno, va='bottom')
        ax.text(0.1, static_v - 0.001, f'Static Semi ({static_v:.3f})', 
                color=COLOR_V, alpha=0.7, fontproperties=font_en_anno, va='top')

        # 画折线图 
        # 注意 label 的写法：$\mathrm{...}$ 强制启用新罗马风格，后面的中文使用中文字体
        ax.plot(x, data_h, marker='o', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                linewidth=2.5, color=COLOR_H, label=r'$\mathrm{YOLO}_h$ (横向光)')
        ax.plot(x, data_v, marker='s', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                linewidth=2.5, color=COLOR_V, label=r'$\mathrm{YOLO}_v$ (纵向光)')

        # 添加数据点标签 (动态偏移防遮挡)
        for i, (val_h, val_v) in enumerate(zip(data_h, data_v)):
            offset_h = 0.0025 if val_h >= val_v else -0.0035
            offset_v = -0.0035 if val_h >= val_v else 0.0025
            
            va_h = 'bottom' if offset_h > 0 else 'top'
            va_v = 'bottom' if offset_v > 0 else 'top'

            ax.text(i, val_h + offset_h, f"{val_h:.3f}", ha='center', va=va_h, 
                    color=COLOR_H, fontproperties=font_en_anno, fontweight='bold')
            ax.text(i, val_v + offset_v, f"{val_v:.3f}", ha='center', va=va_v, 
                    color=COLOR_V, fontproperties=font_en_anno, fontweight='bold')

        # 样式美化 (已去除 Title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontproperties=font_en_tick)
        ax.set_ylabel(ylabel, fontproperties=font_en_label)
        
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en_tick)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3) 

        # 将图例放置在右下角 (因为折线整体向右上角增长，右下角通常是完美的空白区域)
        ax.legend(frameon=True, prop=font_zh_legend, loc='lower right')

        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_filename}")

    # ==========================================
    # 2. 分别调用函数生成两张独立图像
    # ==========================================
    draw_and_save_figure(
        yolo_h_50, yolo_v_50, static_h_50, static_v_50, 
        ylabel='mAP@50', save_filename='plot_iter_map50.png'
    )
    
    draw_and_save_figure(
        yolo_h_95, yolo_v_95, static_h_95, static_v_95, 
        ylabel='mAP@50:95', save_filename='plot_iter_map50_95.png'
    )

if __name__ == "__main__":
    # ================== 核心参数输入区 ==================
    # 1. 配置路径
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/数据飞轮主实验'
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

    # 2. 输入实验数据 
    # 列表顺序：[Baseline, Iter 1, Iter 2, Iter 3]
    # --- mAP@50 数据 ---
    YOLO_H_50 = [0.737, 0.757, 0.764, 0.760] 
    YOLO_V_50 = [0.693, 0.716, 0.748, 0.745]
    STATIC_H_50 = 0.755
    STATIC_V_50 = 0.731

    # --- mAP@50:95 数据 ---
    YOLO_H_95 = [0.373, 0.385, 0.395, 0.398]
    YOLO_V_95 = [0.333, 0.352, 0.370, 0.371]
    STATIC_H_95 = 0.382
    STATIC_V_95 = 0.366
    # ====================================================

    plot_iteration_metrics(
        output_dir=OUTPUT_DIR,
        times_font_path=TIMES_FONT_PATH,
        zh_font_path=ZH_FONT_PATH,
        yolo_h_50=YOLO_H_50, yolo_v_50=YOLO_V_50, static_h_50=STATIC_H_50, static_v_50=STATIC_V_50,
        yolo_h_95=YOLO_H_95, yolo_v_95=YOLO_V_95, static_h_95=STATIC_H_95, static_v_95=STATIC_V_95
    )