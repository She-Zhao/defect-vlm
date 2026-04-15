"""
计算横向和纵向条纹光融合后的结果（fusion.json）的各项指标
输入：多流主干网络的推理结果 + 原始验证集的数据（coco格式）
输出：PR 曲线、F1 曲线、混淆矩阵（自定义新罗马字体），精准阈值下的P/R/F1指标，并同步保存
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

# ================== 字体配置 ==================
TIMES_FONT_PATH = "/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/times.ttf"
ZH_FONT_PATH = "/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc"

# 创建字体属性对象
font_times_large = fm.FontProperties(fname=TIMES_FONT_PATH, size=16)  # 坐标轴大字
font_times_tick = fm.FontProperties(fname=TIMES_FONT_PATH, size=14)   # 刻度英文字
font_times_annot = fm.FontProperties(fname=TIMES_FONT_PATH, size=12)  # 矩阵内数字 (相当于五号字)
font_zh_large = fm.FontProperties(fname=ZH_FONT_PATH, size=16)        # 中文大字
# ==============================================

def evaluate_fusion_results(pred_json, gt_json, output_dir, target_conf=0.1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 解析 Ground Truth
    print(f"📖 正在加载真实标签: {gt_json}")
    with open(gt_json, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    names_dict = {cat['id']: cat['name'] for cat in coco_gt['categories']}
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    
    gt_dict = {img['file_name']: [] for img in coco_gt['images']}
    for ann in coco_gt['annotations']:
        img_name = imgid2name[ann['image_id']]
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt_dict[img_name].append([cat_id, x, y, x + w, y + h])
        
    # 2. 解析 Predictions
    print(f"📖 正在加载预测结果: {pred_json}")
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_dict = json.load(f)

    # 3. 准备评估工具
    iouv = torch.linspace(0.5, 0.95, 10)
    cm = ConfusionMatrix(nc=len(names_dict), conf=target_conf, iou_thres=0.45)
    stats = []

    # 4. 逐张图对比计算
    print("⚙️ 正在对比每一张图像的预测框与真实框...")
    all_images = set(gt_dict.keys()).union(set(pred_dict.keys()))
    
    for filename in all_images:
        gts = gt_dict.get(filename, [])
        preds = pred_dict.get(filename, [])
        
        labels = torch.tensor(gts, dtype=torch.float32) if len(gts) > 0 else torch.empty(0, 5)
        
        if len(preds) > 0:
            detections = torch.tensor(
                [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3], p['confidence'], p['class_id']]
                 for p in preds], dtype=torch.float32
            )
            detections = detections[detections[:, 4].argsort(descending=True)]
        else:
            detections = torch.empty(0, 6)

        tp = torch.zeros((detections.shape[0], len(iouv)), dtype=torch.bool)
        if labels.shape[0] > 0 and detections.shape[0] > 0:
            ious = box_iou(labels[:, 1:], detections[:, :4])
            correct_class = labels[:, 0:1] == detections[:, 5]
            
            for i, threshold in enumerate(iouv):
                matches = torch.nonzero((ious >= threshold) & correct_class)
                if matches.shape[0] > 0:
                    matches_iou = ious[matches[:, 0], matches[:, 1]]
                    matches = torch.cat([matches, matches_iou[:, None]], dim=1)
                    matches = matches[matches[:, 2].argsort(descending=True)]
                    
                    matched_labels, matched_detections = set(), set()
                    for m in matches:
                        l_idx, d_idx = int(m[0]), int(m[1])
                        if l_idx not in matched_labels and d_idx not in matched_detections:
                            matched_labels.add(l_idx)
                            matched_detections.add(d_idx)
                            tp[d_idx, i] = True
                            
        cm.process_batch(detections, labels[:, 1:], labels[:, 0])
        
        stats.append((
            tp,
            detections[:, 4] if len(detections) > 0 else torch.empty(0),
            detections[:, 5] if len(detections) > 0 else torch.empty(0),
            labels[:, 0] if len(labels) > 0 else torch.empty(0)
        ))

    # 5. 计算指标
    print("📊 正在计算 mAP 并提取指定阈值的精准指标...")
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]

    tp_arr, fp_arr, p_max, r_max, f1_max, all_ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, 
        plot=True, 
        save_dir=output_dir, 
        names=names_dict,
        prefix="Fusion_"  
    )
    
    ap50 = all_ap[:, 0]
    ap = all_ap.mean(1)

    # 精准查表
    idx = np.abs(x - target_conf).argmin()
    actual_conf = x[idx]
    
    p_at_conf = p_curve[:, idx]
    r_at_conf = r_curve[:, idx]
    f1_at_conf = f1_curve[:, idx]
    
    macro_p = p_at_conf.mean()
    macro_r = r_at_conf.mean()
    macro_f1 = f1_at_conf.mean()
    
    # ================== 【自定义高级混淆矩阵绘制】 ==================
    matrix = cm.matrix.cpu().numpy() if isinstance(cm.matrix, torch.Tensor) else cm.matrix
    class_names = list(names_dict.values()) + ['background']
    
    # 计算归一化矩阵
    matrix_r = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-15)
    matrix_p = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-15)

    def plot_custom_cm(mat, title_zh, title_en, save_name, is_norm=False):
        plt.figure(figsize=(10, 8))
        
        # 核心：控制方块内部文字的字体和大小 (fmt 动态调整，归一化用.2f，绝对值用.0f)
        annot_fmt = ".2f" if is_norm else ".0f"
        ax = sns.heatmap(mat, annot=True, fmt=annot_fmt, cmap="Blues", 
                         xticklabels=class_names, yticklabels=class_names,
                         vmin=0.0, vmax=1.0 if is_norm else None,
                         annot_kws={"fontproperties": font_times_annot}) # <--- 强制内部数字为新罗马五号
        
        # 设置标题和轴标签字体
        plt.title(title_zh, fontproperties=font_zh_large, pad=15)
        plt.ylabel("Predicted Class", fontproperties=font_times_large)
        plt.xlabel("True Class", fontproperties=font_times_large)
        
        # 设置坐标轴刻度的字体（缺陷英文名使用新罗马）
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_times_tick)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_times_tick)
            
        plt.tight_layout()
        plt.savefig(output_dir / save_name, dpi=300)
        plt.close()

    print("🎨 正在绘制新罗马字体的高清混淆矩阵...")
    plot_custom_cm(matrix, "绝对数量混淆矩阵", "Absolute Confusion Matrix", "Custom_Confusion_Matrix_Abs.png", is_norm=False)
    plot_custom_cm(matrix_r, "按行归一化混淆矩阵 (Recall)", "Recall Confusion Matrix", "Custom_Confusion_Matrix_R.png", is_norm=True)
    plot_custom_cm(matrix_p, "按列归一化混淆矩阵 (Precision)", "Precision Confusion Matrix", "Custom_Confusion_Matrix_P.png", is_norm=True)
    # ==============================================================
    
    # 记录日志
    log_lines = []
    log_lines.append("=" * 75)
    log_lines.append(f"🏆 评估结果 (YOLO Fusion) 已保存至: {output_dir}")
    log_lines.append(f"【严格控制阈值测试】 目标阈值: {target_conf:.4f} (实际查表点: {actual_conf:.4f})")
    log_lines.append(f"Macro-Precision: {macro_p:.4f}")
    log_lines.append(f"Macro-Recall:    {macro_r:.4f}")
    log_lines.append(f"Macro-F1:        {macro_f1:.4f}")
    log_lines.append("-" * 75)
    log_lines.append(f"{'Class':>15} {'Prec.':>8} {'Recall':>8} {'F1':>8} {'mAP@50':>10} {'mAP@50-95':>10}")
    log_lines.append("-" * 75)
    log_lines.append(f"{'all':>15} {macro_p:>8.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {ap50.mean():>10.4f} {ap.mean():>10.4f}")
    for i, c in enumerate(ap_class):
        log_lines.append(f"{names_dict[c]:>15} {p_at_conf[i]:>8.4f} {r_at_conf[i]:>8.4f} {f1_at_conf[i]:>8.4f} {ap50[i]:>10.4f} {ap[i]:>10.4f}")
    log_lines.append("=" * 75)
    
    log_content = "\n".join(log_lines)
    print(log_content)
    
    metric_file = output_dir / f"metric_th{target_conf}.txt"
    with open(metric_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"📄 日志已成功同步保存至: {metric_file}")

if __name__ == '__main__':
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    PRED_JSON = "/data/ZS/defect_dataset/9_yolo_preds/自己手写的推理脚本/val_0p001/nms_fusion_conf_0p01.json" 
    
    TARGET_CONF = 0.15
    OUTPUT_DIR = f"/data/ZS/defect-vlm/output/figures/ch4_cascade_final/ch4_yolo_th{TARGET_CONF}" 
    
    evaluate_fusion_results(PRED_JSON, GT_JSON, OUTPUT_DIR, target_conf=TARGET_CONF)