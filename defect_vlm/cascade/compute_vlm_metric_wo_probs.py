"""
计算 VLM (Qwen-VL) 推理结果的各项检测指标
输入：VLM 的 JSONL 结果 + 中间映射 JSON + 原始 COCO 格式 GT
输出：PR 曲线、F1 曲线、混淆矩阵（自定义新罗马字体），全局宏平均 P、R、F1，并同步保存到 metric.txt
"""
import os
import re
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

def parse_vlm_prediction(pred_text):
    """鲁棒地解析 VLM 输出的 JSON 文本，提取缺陷类别"""
    try:
        # 去除可能存在的 markdown 代码块包裹
        clean_text = pred_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("defect", "background").lower()
    except Exception:
        # 如果 json 解析失败，使用正则兜底提取
        match = re.search(r'"defect"\s*:\s*"([^"]+)"', pred_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "background"

def evaluate_vlm_results(vlm_jsonl, inter_json, gt_json, output_dir, conf_threshold=0.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 解析 Ground Truth (COCO 格式)
    print(f"📖 正在加载真实标签: {gt_json}")
    with open(gt_json, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    names_dict = {cat['id']: cat['name'] for cat in coco_gt['categories']}
    name2id = {cat['name']: cat['id'] for cat in coco_gt['categories']}
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    
    gt_dict = {img['file_name']: [] for img in coco_gt['images']}
    for ann in coco_gt['annotations']:
        img_name = imgid2name[ann['image_id']]
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt_dict[img_name].append([cat_id, x, y, x + w, y + h])

    # 2. 加载中间映射文件 (用于 ID 溯源到原图名称)
    print(f"🔗 正在加载 ID 映射文件: {inter_json}")
    id2filename = {}
    with open(inter_json, 'r', encoding='utf-8') as f:
        inter_data = json.load(f)
        for item in inter_data:
            # 获取原始图像名称 (去掉路径和光源等前缀/后缀)
            orig_path = item["original_image_paths"][0]
            filename = Path(orig_path).name 
            id2filename[item["id"]] = filename

    # 3. 解析 VLM 的预测结果
    print(f"🧠 正在解析 VLM 预测结果: {vlm_jsonl}")
    if conf_threshold > 0:
        print(f"🛡️ 已开启前置置信度过滤，阈值: {conf_threshold}")
        
    pred_dict = {img['file_name']: [] for img in coco_gt['images']}
    
    with open(vlm_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            origin_id = item["meta_info"]["origin_id"]
            if origin_id not in id2filename: continue
            
            filename = id2filename[origin_id]
            pred_cls_name = parse_vlm_prediction(item.get("pred", ""))
            
            # 如果 VLM 认为是背景 (无缺陷)，我们就不作为正样本推入 detections 中
            if pred_cls_name == "background" or pred_cls_name not in name2id:
                continue
                
            class_id = name2id[pred_cls_name]
            
            # 提取前级置信度
            prob = item["meta_info"]["confidence"]
            
            # 阈值过滤机制
            if prob < conf_threshold:
                continue
            
            # xywh 转换为 xyxy
            x, y, w, h = item["meta_info"]["bbox"]
            pred_dict[filename].append({
                'bbox': [x, y, x + w, y + h],
                'confidence': prob,
                'class_id': class_id
            })

    # 4. 准备 Ultralytics 评估矩阵
    iouv = torch.linspace(0.5, 0.95, 10)
    cm = ConfusionMatrix(nc=len(names_dict), conf=0.0, iou_thres=0.45)
    stats = []

    # 5. 逐张图对比计算
    print("⚙️ 正在执行边界框匹配与 IoU 计算...")
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
            # 必须按照置信度降序排列
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

    # 6. 生成最终指标与图表
    print("📊 正在生成 PR 曲线、混淆矩阵及计算 mAP...")
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]

    if len(tp) == 0:
        print("⚠️ 警告：VLM 没有输出任何有效的正样本预测，无法生成指标。")
        return

    tp_arr, fp_arr, p, r, f1, all_ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, 
        plot=True, 
        save_dir=output_dir, 
        names=names_dict,
        prefix="VLM_"  # 前缀区分
    )
    
    ap50 = all_ap[:, 0]
    ap = all_ap.mean(1)
    
    # 提取并计算宏平均指标
    macro_p = p.mean()
    macro_r = r.mean()
    macro_f1 = f1.mean()

    # ================== 【自定义高级混淆矩阵绘制】 ==================
    matrix = cm.matrix.cpu().numpy() if isinstance(cm.matrix, torch.Tensor) else cm.matrix
    class_names = list(names_dict.values()) + ['background']
    
    # 计算归一化矩阵
    matrix_r = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-15)
    matrix_p = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-15)

    def plot_custom_cm(mat, title_zh, title_en, save_name, is_norm=False):
        plt.figure(figsize=(10, 8))
        
        # 控制方块内部文字的字体和大小 (fmt 动态调整，归一化用.2f，绝对值用.0f)
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
    plot_custom_cm(matrix, "绝对数量混淆矩阵", "Absolute Confusion Matrix", "Custom_VLM_Confusion_Matrix_Abs.png", is_norm=False)
    plot_custom_cm(matrix_r, "按行归一化混淆矩阵 (Recall)", "Recall Confusion Matrix", "Custom_VLM_Confusion_Matrix_R.png", is_norm=True)
    plot_custom_cm(matrix_p, "按列归一化混淆矩阵 (Precision)", "Precision Confusion Matrix", "Custom_VLM_Confusion_Matrix_P.png", is_norm=True)
    # ==============================================================
        
    # ================== 【将结果保存至 metric.txt】 ==================
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"🏆 评估结果 (VLM Model) 已保存至: {output_dir}")
    log_lines.append(f"【全局最佳宏平均指标 (At Max F1)】")
    log_lines.append(f"Macro-Precision: {macro_p:.4f}")
    log_lines.append(f"Macro-Recall:    {macro_r:.4f}")
    log_lines.append(f"Macro-F1:        {macro_f1:.4f}")
    log_lines.append("-" * 70)
    log_lines.append(f"{'Class':>15} {'Prec.':>8} {'Recall':>8} {'F1':>8} {'mAP@50':>10} {'mAP@50-95':>10}")
    log_lines.append("-" * 70)
    log_lines.append(f"{'all':>15} {macro_p:>8.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {ap50.mean():>10.4f} {ap.mean():>10.4f}")
    for i, c in enumerate(ap_class):
        log_lines.append(f"{names_dict[c]:>15} {p[i]:>8.4f} {r[i]:>8.4f} {f1[i]:>8.4f} {ap50[i]:>10.4f} {ap[i]:>10.4f}")
    log_lines.append("=" * 70)
    
    # 拼装字符串
    log_content = "\n".join(log_lines)
    
    # 1. 打印到终端
    print(log_content)
    
    # 2. 写入到文件
    metric_file = output_dir / "metric.txt"
    with open(metric_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"📄 日志已成功同步保存至: {metric_file}")
    # =======================================================================

if __name__ == '__main__':
    # 1. 真实的标注 JSON
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    
    # 2. 我们拼图那一步生成的中间 JSON
    INTER_JSON = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p01.json"
    
    # 3. VLM 跑出来的最终 JSONL
    VLM_JSONL = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v11_val_0p01.jsonl"
    
    # 4. 图表和曲线存放的目录
    OUTPUT_DIR = "/data/ZS/defect-vlm/output/figures/ch4_cascade_final/ch4_vlm_0p01_th0p2"
    
    # 5. 执行评估（开启置信度过滤，这里可以按需修改）
    evaluate_vlm_results(VLM_JSONL, INTER_JSON, GT_JSON, OUTPUT_DIR, conf_threshold=0.2)