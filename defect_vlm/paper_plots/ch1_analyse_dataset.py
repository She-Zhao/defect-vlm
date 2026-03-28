import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def load_and_process_data(json_path):
    # 此处逻辑与你原代码一致，保持不变
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_info = {img['id']: (img['width'], img['height']) for img in data['images']}
    all_img_ids = list(img_info.keys())
    random.seed(42) 
    random.shuffle(all_img_ids)
    split_idx = int(len(all_img_ids) * 0.8)
    train_ids = set(all_img_ids[:split_idx])
    val_ids = set(all_img_ids[split_idx:])
    
    train_stats = {'xc': [], 'yc': [], 'w': [], 'h': []}
    val_stats = {'xc': [], 'yc': [], 'w': [], 'h': []}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_info: continue
            
        img_w, img_h = img_info[img_id]
        x_min, y_min, bbox_w, bbox_h = ann['bbox']
        
        xc = (x_min + bbox_w / 2.0) / img_w
        yc = (y_min + bbox_h / 2.0) / img_h
        w_norm = bbox_w / img_w
        h_norm = bbox_h / img_h
        
        target_dict = train_stats if img_id in train_ids else val_stats
        target_dict['xc'].append(xc)
        target_dict['yc'].append(yc)
        target_dict['w'].append(w_norm)
        target_dict['h'].append(h_norm)
        
    return train_stats, val_stats


def save_individual_plots(json_path, output_dir, simsun_path, times_path):
    """
    将数据集的各项特征分布单独保存为独立的图片
    """
    FONT_SIZE = 10.5 
    
    if not os.path.exists(simsun_path): raise FileNotFoundError(f"找不到中文字体: {simsun_path}")
    if not os.path.exists(times_path): raise FileNotFoundError(f"找不到英文字体: {times_path}")

    font_zh = fm.FontProperties(fname=simsun_path, size=FONT_SIZE)
    font_en = fm.FontProperties(fname=times_path, size=FONT_SIZE)

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['xtick.labelsize'] = FONT_SIZE
    plt.rcParams['ytick.labelsize'] = FONT_SIZE

    print("正在加载并解析 JSON 数据...")
    train_stats, val_stats = load_and_process_data(json_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    datasets = [('train', '训练集', train_stats), ('val', '验证集', val_stats)]
    ACADEMIC_BLUE = '#1f77b4'
    
    # 定义要画的图表类型清单
    plot_tasks = [
        ('center', '缺陷中心空间分布', '$y$'),
        ('size', '缺陷尺寸分布', '$h$'),
        ('x', '$x$轴坐标分布', 'Frequency'),
        ('y', '$y$轴坐标分布', 'Frequency'),
        ('h', '高度区间分布', 'Frequency'),
        ('w', '宽度区间分布', 'Frequency')
    ]

    for split_name, prefix_zh, stats in datasets:
        xc, yc = np.array(stats['xc']), np.array(stats['yc'])
        w, h = np.array(stats['w']), np.array(stats['h'])
        
        for task_id, title_zh, ylabel in plot_tasks:
            # 单图尺寸设置：宽3.5英寸，高2.8英寸，非常适合LaTeX拼图
            fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
            
            if task_id == 'center':
                ax.scatter(xc, yc, s=2, alpha=0.3, color=ACADEMIC_BLUE)
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_xlabel(f'{prefix_zh}{title_zh}', fontproperties=font_zh)
                ax.set_ylabel(ylabel, fontproperties=font_en)
            elif task_id == 'size':
                ax.scatter(w, h, s=2, alpha=0.3, color=ACADEMIC_BLUE)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel(f'{prefix_zh}{title_zh}', fontproperties=font_zh)
                ax.set_ylabel(ylabel, fontproperties=font_en)
            else:
                # 直方图
                data_dict = {'x': xc, 'y': yc, 'w': w, 'h': h}
                data = data_dict[task_id]
                ax.hist(data, bins=50, color=ACADEMIC_BLUE, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel(f'{prefix_zh}{title_zh}', fontproperties=font_zh)
                # 移除了物理量子标题，让图面更纯粹
            
            # 保存单图，命名规则：train_x.png, val_h.png 等
            save_path = os.path.join(output_dir, f'{split_name}_{task_id}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ 已保存单图: {save_path}")

if __name__ == "__main__":
    json_path = '/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/train_val.json'
    output_dir = '/data/ZS/defect-vlm/output/figures/数据分布'
    
    simsun_font_path = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc'
    times_font_path = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf'
    
    save_individual_plots(json_path, output_dir, simsun_font_path, times_font_path)
