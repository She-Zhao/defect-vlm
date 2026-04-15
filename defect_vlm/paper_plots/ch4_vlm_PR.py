"""
计算消融实验中不同微调策略下，各类缺陷 Precision 和 Recall 的对比图
(优化为紧凑版，适合双图并排，剔除背景类)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def parse_vlm_metrics_txt(filepath):
    """解析 VLM 导出的详细分类报告 txt 文件"""
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "详细分类报告" in line:
            start_parsing = True
            continue
        
        if start_parsing:
            parts = line.strip().split()
            # 【修改点 1】：去掉了 'background'
            if len(parts) >= 5 and parts[0] in ['breakage', 'inclusion', 'crater', 'bulge', 'scratch', 'run']:
                class_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                metrics[class_name] = {'P': precision, 'R': recall}
                
    return metrics

def plot_multi_method_comparison(
    methods_dict, output_dir, 
    times_font_path, zh_font_path
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 【修改点 2】：移除了背景的映射
    class_map = {
        'breakage': '破损', 
        'inclusion': '漆渣', 
        'crater': '凹陷', 
        'bulge': '凸起',
        'scratch': '划痕', 
        'run': '流挂'
    }
    
    classes_en = list(class_map.keys())
    classes_zh = list(class_map.values())
    
    parsed_data = {}
    for method_name, filepath in methods_dict.items():
        parsed_data[method_name] = parse_vlm_metrics_txt(filepath)

    # 【修改点 3】：字号微调，适应更小的画布
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 11
    ANNOTATE_SIZE = 9  # 数据标签略微缩小，防止溢出

    font_en_label = fm.FontProperties(fname=times_font_path, size=AXIS_LABEL_SIZE)
    font_en_tick = fm.FontProperties(fname=times_font_path, size=TICK_SIZE)
    font_en_anno = fm.FontProperties(fname=times_font_path, size=ANNOTATE_SIZE)
    font_zh_tick = fm.FontProperties(fname=zh_font_path, size=TICK_SIZE)
    font_zh_legend = fm.FontProperties(fname=zh_font_path, size=LEGEND_SIZE)

    plt.rcParams['mathtext.fontset'] = 'stix'

    COLORS = ['#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#E64B35']
    
    x = np.arange(len(classes_zh))
    num_methods = len(methods_dict)
    total_width = 0.82 
    width = total_width / num_methods 

    def draw_and_save_bar_chart(metric_key, ylabel, save_filename):
        # 【修改点 4】：画布调整为 9x4.5，更方正紧凑，完美适合左右排版
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

        rects_list = []
        method_names = list(methods_dict.keys())
        
        offsets = [ (i - num_methods/2 + 0.5) * width for i in range(num_methods) ]

        for i, method_name in enumerate(method_names):
            data_values = [parsed_data[method_name].get(c, {}).get(metric_key, 0) for c in classes_en]
            
            rects = ax.bar(x + offsets[i], data_values, width, 
                           label=method_name, 
                           color=COLORS[i], alpha=0.9, edgecolor='white')
            rects_list.append(rects)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 2),  
                                textcoords="offset points",
                                ha='center', va='bottom', rotation=90, 
                                fontproperties=font_en_anno, color='#333333')

        for rects in rects_list:
            autolabel(rects)

        ax.set_xticks(x)
        ax.set_xticklabels(classes_zh, fontproperties=font_zh_tick)
        
        ax.set_ylabel(ylabel, fontproperties=font_en_label)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_en_tick)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        ax.set_ylim(0, 1.00) # 留出空间给 90 度旋转的标签

        # 【修改点 5】：图例折成 3 列（ncol=3），位置略微上移，防止遮挡图表
        ax.legend(frameon=True, prop=font_zh_legend, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.25), ncol=3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已成功保存紧凑版图像: {save_path}")

    draw_and_save_bar_chart(metric_key='P', ylabel='Precision', save_filename='Ablation_Precision_Comparison.png')
    draw_and_save_bar_chart(metric_key='R', ylabel='Recall', save_filename='Ablation_Recall_Comparison.png')


if __name__ == "__main__":
    # ================= 路径配置 =================
    BASE_DIR = '/data/ZS/defect-vlm/output/figures/蒸馏模型效果_val_merged_general_recall1'
    OUTPUT_DIR = '/data/ZS/defect-vlm/output/figures/ch4_VLM消融实验对比图'
    
    TIMES_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 
    
    METHODS_DICT = {
        'Baseline (仅名称)': os.path.join(BASE_DIR, 'v1_LM.txt'),
        '+ 图像描述': os.path.join(BASE_DIR, 'v4_LM_defect_only.txt'),
        '+ 思维链推理': os.path.join(BASE_DIR, 'v5_LM_cot_defect.txt'),
        '端到端混合微调': os.path.join(BASE_DIR, 'v6_LM_step12_defect.txt'),
        'Ours (课程学习)': os.path.join(BASE_DIR, 'v11_stage3_LM_PRO_VIT_26k.txt')
    }
    # ============================================

    plot_multi_method_comparison(
        methods_dict=METHODS_DICT, 
        output_dir=OUTPUT_DIR, 
        times_font_path=TIMES_FONT_PATH, 
        zh_font_path=ZH_FONT_PATH
    )