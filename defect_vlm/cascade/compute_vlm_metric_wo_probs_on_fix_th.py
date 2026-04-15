"""
计算 VLM (Qwen-VL) 推理结果的各项检测指标
输入：VLM 的 JSONL 结果 + 中间映射 JSON + 原始 COCO 格式 GT
输出：PR 曲线、F1 曲线、混淆矩阵，严格阈值下的全局宏平均 P、R、F1，并同步保存
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

# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

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

def evaluate_vlm_results(vlm_jsonl, inter_json, gt_json, output_dir, target_conf=0.2):
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
            orig_path = item["original_image_paths"][0]
            filename = Path(orig_path).name 
            id2filename[item["id"]] = filename

    # 3. 解析 VLM 的预测结果
    print(f"🧠 正在解析 VLM 预测结果: {vlm_jsonl}")
    pred_dict = {img['file_name']: [] for img in coco_gt['images']}
    
    with open(vlm_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            origin_id = item["meta_info"]["origin_id"]
            if origin_id not in id2filename: continue
            
            filename = id2filename[origin_id]
            pred_cls_name = parse_vlm_prediction(item.get("pred", ""))
            
            # 如果 VLM 认为是背景 (无缺陷)，抛弃该候选框
            if pred_cls_name == "background" or pred_cls_name not in name2id:
                continue
                
            class_id = name2id[pred_cls_name]
            # 提取前级置信度
            prob = item["meta_info"]["confidence"]
            
            # 注：这里去掉了硬过滤，把完整的概率流送给底层评估器，才能画出完整的 PR 曲线！
            
            x, y, w, h = item["meta_info"]["bbox"]
            pred_dict[filename].append({
                'bbox': [x, y, x + w, y + h],
                'confidence': prob,
                'class_id': class_id
            })

    # 4. 准备 Ultralytics 评估矩阵
    iouv = torch.linspace(0.5, 0.95, 10)
    # 【核心：让混淆矩阵内置接收 target_conf】
    cm = ConfusionMatrix(nc=len(names_dict), conf=target_conf, iou_thres=0.45)
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

    tp_arr, fp_arr, p_max, r_max, f1_max, all_ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, 
        plot=True, 
        save_dir=output_dir, 
        names=names_dict,
        prefix="VLM_"  
    )
    
    ap50 = all_ap[:, 0]
    ap = all_ap.mean(1)
    
    # ================== 【核心黑科技：精准查表提取特定阈值的指标】 ==================
    idx = np.abs(x - target_conf).argmin()
    actual_conf = x[idx]
    
    p_at_conf = p_curve[:, idx]
    r_at_conf = r_curve[:, idx]
    f1_at_conf = f1_curve[:, idx]
    
    macro_p = p_at_conf.mean()
    macro_r = r_at_conf.mean()
    macro_f1 = f1_at_conf.mean()
    # =================================================================================

    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=True)
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=False)
    
    matrix = cm.matrix.cpu().numpy() if isinstance(cm.matrix, torch.Tensor) else cm.matrix
    class_names = list(names_dict.values()) + ['background']

    matrix_r = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-15)
    matrix_p = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-15)

    def plot_custom_cm(mat, title, save_name):
        plt.figure(figsize=(10, 8))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0.0, vmax=1.0)
        plt.title(title, pad=15)
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(output_dir / save_name, dpi=300)
        plt.close()

    print("🎨 正在绘制双向归一化混淆矩阵...")
    plot_custom_cm(matrix_r, "Confusion Matrix (Row-Normalized / Recall)", "VLM_Confusion_Matrix_R.png")
    plot_custom_cm(matrix_p, "Confusion Matrix (Col-Normalized / Precision)", "VLM_Confusion_Matrix_P.png")
        
    # ================== 【将结果保存至 metric.txt】 ==================
    log_lines = []
    log_lines.append("=" * 75)
    log_lines.append(f"🏆 评估结果 (VLM Cascade Model) 已保存至: {output_dir}")
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
    # =======================================================================

if __name__ == '__main__':
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    INTER_JSON = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p01.json"
    VLM_JSONL = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v11_val_0p01.jsonl"
    
    # 修改这里的阈值即可，比如你想测 0.2，就写 0.2
    TARGET_CONF = 0.35
    OUTPUT_DIR = f"/data/ZS/defect-vlm/output/figures/ch4_cascade/ch4_vlm_th{TARGET_CONF}"
    
    evaluate_vlm_results(VLM_JSONL, INTER_JSON, GT_JSON, OUTPUT_DIR, target_conf=TARGET_CONF)