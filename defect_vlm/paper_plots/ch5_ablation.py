"""
第五章的消融实验，验证α=1.0和没有ema的效果
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_ablation_metrics_split(
    output_dir, times_font_path, zh_font_path,
    ours_50, no_ema_50, no_alpha_50, base_50,
    ours_95, no_ema_95, no_alpha_95, base_95
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # 1. 字体与全局样式设置
    # ==========================================
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 13
    LEGEND_SIZE = 11
    ANNOTATE_SIZE = 11

    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_en_label = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)
    font_en_anno = fm.FontProperties(fname=times_font_path, size=ANNOTATE_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    # 启用 STIX 引擎处理数学/英文混排，视觉等效于 Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 学术配色：与你之前确认的高级色系保持一致
    COLOR_OURS = '#E64B35'    # 玫瑰红
    COLOR_NO_ALP = '#4DBBD5'  # 湖蓝
    COLOR_NO_EMA = '#8D6E63'  # 质感棕灰
    COLOR_BASE = '#9E9E9E'    # 基础灰
    
    x_labels = ['Baseline\n(GT)', 'Iter 1', 'Iter 2', 'Iter 3']
    x = np.arange(len(x_labels))

    # ==========================================
    # 2. 内部独立绘图函数
    # ==========================================
    def draw_single_figure(d_ours, d_no_ema, d_no_alp, val_base, ylabel, filename):
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        
        # 补齐长度 (Baseline 的点为索引 0，后面三轮拼接上列表)
        y_ours = [val_base] + d_ours
        y_no_ema = [val_base] + d_no_ema
        y_no_alp = [val_base] + d_no_alp

        # 画折线
        ax.plot(x, y_ours, marker='o', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                linewidth=2.5, color=COLOR_OURS, label=r'$\mathrm{Ours}$ (完整动态迭代)')
                
        ax.plot(x, y_no_alp, marker='^', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                linewidth=2.5, linestyle='-.', color=COLOR_NO_ALP, label=r'$w/o$ 动态 $\alpha$')
                
        ax.plot(x, y_no_ema, marker='X', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                linewidth=2.5, linestyle='--', color=COLOR_NO_EMA, label=r'$w/o$ $\mathrm{EMA}$ 更新')

        # Baseline 单独画一个灰色的正方形点
        ax.plot(0, val_base, marker='s', markersize=8, color=COLOR_BASE)

        # ---------------------------------------------------------
        # 【核心智能标注算法】：彩色数值 & 防重叠防遮挡
        # ---------------------------------------------------------
        # 1. 标注 Baseline (统一放在下方)
        ax.annotate(f"{val_base:.3f}", xy=(0, val_base), xytext=(0, -8), 
                    textcoords="offset points", ha='center', va='top', 
                    color=COLOR_BASE, fontproperties=font_en_anno, fontweight='bold')

        # 2. 动态标注 Iter 1, Iter 2, Iter 3
        for i in range(1, 4):
            pts = [
                {'val': y_ours[i], 'color': COLOR_OURS},
                {'val': y_no_ema[i], 'color': COLOR_NO_EMA},
                {'val': y_no_alp[i], 'color': COLOR_NO_ALP}
            ]
            # 根据数值从大到小排序，用来分配上下位置
            pts_sorted = sorted(pts, key=lambda item: item['val'], reverse=True)
            
            # 碰撞检测：判断中间的点离上面近还是离下面近
            if pts_sorted[0]['val'] == pts_sorted[1]['val']:
                pos = ['up', 'down', 'down']
            elif pts_sorted[1]['val'] == pts_sorted[2]['val']:
                pos = ['up', 'up', 'down']
            else:
                dist_to_top = pts_sorted[0]['val'] - pts_sorted[1]['val']
                dist_to_bot = pts_sorted[1]['val'] - pts_sorted[2]['val']
                pos = ['up', 'down', 'down'] if dist_to_top > dist_to_bot else ['up', 'up', 'down']
            
            # 执行带有同色字体属性的标注
            for p, position in zip(pts_sorted, pos):
                offset = 6 if position == 'up' else -7
                va = 'bottom' if position == 'up' else 'top'
                ax.annotate(f"{p['val']:.3f}", xy=(i, p['val']), xytext=(0, offset), 
                            textcoords="offset points", ha='center', va=va, 
                            color=p['color'], fontproperties=font_en_anno, fontweight='bold')
        # ---------------------------------------------------------

        # 样式美化
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontproperties=font_en_tick)
        ax.set_ylabel(ylabel, fontproperties=font_en_label)
        
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en_tick)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 扩大Y轴上限留出图例空间
        ylim_max = max([max(y_ours), max(y_no_ema), max(y_no_alp)])
        ylim_min = min([min(y_ours), min(y_no_ema), min(y_no_alp)])
        margin = (ylim_max - ylim_min) * 0.25
        ax.set_ylim(ylim_min - margin/2, ylim_max + margin)

        # 添加图例
        ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.12), ncol=3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存消融图: {filename}")

    # ==========================================
    # 3. 生成两张独立图像
    # ==========================================
    draw_single_figure(ours_50, no_ema_50, no_alpha_50, base_50, 
                       ylabel='mAP@50', filename='plot_ablation_map50.png')
    
    draw_single_figure(ours_95, no_ema_95, no_alpha_95, base_95, 
                       ylabel='mAP@50:95', filename='plot_ablation_map50_95.png')


if __name__ == "__main__":
    # ================== 参数输入区 ==================
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/动态伪标签分析'
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

    # 1. Baseline 数值
    BASE_50 = 0.737
    BASE_95 = 0.373

    # 2. mAP@50 消融数据 (Iter 1, 2, 3)
    OURS_50     = [0.757, 0.764, 0.760] 
    NO_ALP_50   = [0.757, 0.735, 0.750] 
    NO_EMA_50   = [0.750, 0.764, 0.753] 

    # 3. mAP@50:95 消融数据 (Iter 1, 2, 3)
    OURS_95     = [0.385, 0.395, 0.398]
    NO_ALP_95   = [0.385, 0.370, 0.388]
    NO_EMA_95   = [0.380, 0.392, 0.392]

    plot_ablation_metrics_split(
        output_dir=OUTPUT_DIR,
        times_font_path=TIMES_FONT_PATH,
        zh_font_path=ZH_FONT_PATH,
        ours_50=OURS_50, no_ema_50=NO_EMA_50, no_alpha_50=NO_ALP_50, base_50=BASE_50,
        ours_95=OURS_95, no_ema_95=NO_EMA_95, no_alpha_95=NO_ALP_95, base_95=BASE_95
    )