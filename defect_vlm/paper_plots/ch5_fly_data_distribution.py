"""
统计数据飞轮三次迭代和静态半监督的伪标签数据分布
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from math import pi

def parse_pseudo_labels(file_list, base_dir):
    """解析伪标签，提取类别和置信度(weight)"""
    data = []
    for f in file_list:
        if not f.endswith('.txt'):
            continue
        file_path = os.path.join(base_dir, f)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    weight = float(parts[5])
                    data.append({
                        'class_id': class_id, 
                        'confidence': weight
                    })
    return pd.DataFrame(data)

def analyze_flywheel_comprehensive(
    gt_dir, iter_dirs_dict, output_dir, 
    times_font_path, zh_font_path
):
    """
    综合分析数据飞轮：小提琴图 + 雷达图
    :param iter_dirs_dict: 字典形式传入，例如 {'Iter 0': path0, 'Iter 1': path1, ...}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================
    # 1. 字体与全局样式设置
    # ==========================================
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_en_label = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)
    font_zh_label = fm.FontProperties(fname=zh_font_path, size=AXIS_LABEL_SIZE)
    font_zh_tick = fm.FontProperties(fname=zh_font_path, size=TICK_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    plt.rcParams['mathtext.fontset'] = 'stix'

    # 学术渐变配色：冷色(早期) -> 暖色(后期)
    # 对应 4 个 Iter 的颜色，如果你有 5 个 Iter，可以再加一个颜色
    COLORS = ['#4DBBD5', '#00A087', '#F39B7F', '#E64B35', '#8491B4'] 
    
    class_map = {0: '破损', 1: '漆渣', 2: '划痕', 3: '凹陷', 4: '流挂', 5: '凸起'}
    classes_zh = list(class_map.values())

    # ==========================================
    # 2. 提取并清洗数据
    # ==========================================
    gt_files = set([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
    
    all_dfs = []
    iter_names = list(iter_dirs_dict.keys())
    
    for iter_name in iter_names:
        iter_dir = iter_dirs_dict[iter_name]
        # 剔除 GT 真值文件，只保留生成的伪标签
        pseudo_files = set([f for f in os.listdir(iter_dir) if f.endswith('.txt')]) - gt_files
        
        df = parse_pseudo_labels(pseudo_files, iter_dir)
        df['Iteration'] = iter_name
        df['class_name'] = df['class_id'].map(class_map)
        all_dfs.append(df)
        
        print(f"✅ 解析 {iter_name} 完成，提取纯伪标签数: {len(pseudo_files)}")

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # ==========================================
    # 3. 图A：置信度演进小提琴图 (Violin Plot)
    # ==========================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    
    sns.violinplot(
        data=df_combined, x='Iteration', y='confidence', 
        palette=COLORS[:len(iter_names)], inner='box', linewidth=1.5, ax=ax, alpha=0.8
    )

    ax.set_xlabel('迭代轮次', fontproperties=font_zh_label)
    ax.set_ylabel('伪标签生成置信度', fontproperties=font_zh_label)
    
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_en_tick)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_en_tick)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path_violin = os.path.join(output_dir, 'plot_comprehensive_violin.png')
    plt.savefig(save_path_violin, bbox_inches='tight')
    plt.close()
    print(f"✅ 置信度演进小提琴图已保存: {save_path_violin}")

    # ==========================================
    # 4. 图B：全周期类别挖掘边界扩张雷达图
    # ==========================================
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

    angles = [n / float(len(classes_zh)) * 2 * pi for n in range(len(classes_zh))]
    angles += angles[:1] # 闭合多边形

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], classes_zh, fontproperties=font_zh_tick, size=14)
    ax.tick_params(pad=15) 

    ax.set_yticklabels([]) # 隐藏中间的圈圈刻度文字
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#E0E0E0', linestyle='--', linewidth=1)

    # 循环绘制每一个 Iter 的雷达多边形
    for idx, (iter_name, df_iter) in enumerate(zip(iter_names, all_dfs)):
        counts = df_iter['class_name'].value_counts()
        vals = [counts.get(c, 0) for c in classes_zh]
        vals += vals[:1] # 闭合

        color = COLORS[idx % len(COLORS)]
        
        # 考虑到 4 层叠加可能看不清，前面的 Iter 用虚线或细线，最后的 Iter 用粗实线加强调
        linewidth = 2.5 if idx == len(iter_names) - 1 else 1.8
        linestyle = '-' if idx == len(iter_names) - 1 else '--'
        alpha_fill = 0.15 if idx == len(iter_names) - 1 else 0.05 # 只高亮最后一层的填充

        # 名字替换，让图例好看 (带上数学字体)
        label_str = r'$\mathrm{' + iter_name.replace(' ', '\ ') + '}$'
        
        ax.plot(angles, vals, linewidth=linewidth, linestyle=linestyle, color=color, label=label_str)
        ax.fill(angles, vals, color=color, alpha=alpha_fill)

    # 图例配置：放在底部居中
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), 
              prop=font_zh_legend, frameon=False, ncol=len(iter_names))

    plt.tight_layout()
    save_path_radar = os.path.join(output_dir, 'plot_comprehensive_radar.png')
    plt.savefig(save_path_radar, bbox_inches='tight')
    plt.close()
    print(f"✅ 挖掘边界扩张雷达图已保存: {save_path_radar}")

if __name__ == "__main__":
    # ================== 核心参数输入区 ==================
    GT_DIR = '/data/ZS/v11_input/datasets/col3/train/labels'
    
    # 将你所有的 Iter 路径放在这个字典里，键名就是图例上的文字
    # 如果你想画 5 个（到 Iter 4），直接在字典里多加一行即可，代码会自动适应！
    ITER_DIRS = {
        'Iter 0': '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter0/row3/train/labels',
        'Iter 1': '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter1_2/row3/train/labels',
        'Iter 2': '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter2_1/row3/train/labels',
        'Iter 3': '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_10/row3/train/labels'
    }

    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/动态伪标签分析'
    
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

    analyze_flywheel_comprehensive(
        gt_dir=GT_DIR,
        iter_dirs_dict=ITER_DIRS,
        output_dir=OUTPUT_DIR,
        times_font_path=TIMES_FONT_PATH, 
        zh_font_path=ZH_FONT_PATH
    )