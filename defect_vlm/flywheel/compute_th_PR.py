"""
计算模型在验证集上的全局 P-R 曲线，并支持灵活的阈值/召回率/精确率反查。
用于确定 VLM 伪标签挖掘的高确信阈值 (th_h) 和低召回阈值 (th_l)。
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# =====================================================================
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

from ultralytics.utils.metrics import box_iou

def calculate_pr_thresholds(pred_json, gt_json, output_dir, target_thresholds, target_recalls, target_precisions):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 解析 Ground Truth
    print(f"📖 正在加载真实标签: {gt_json}")
    with open(gt_json, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    gt_dict = {img['file_name']: [] for img in coco_gt['images']}
    total_gts = 0
    for ann in coco_gt['annotations']:
        img_name = imgid2name[ann['image_id']]
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt_dict[img_name].append([cat_id, x, y, x + w, y + h])
        total_gts += 1
        
    # 2. 解析 Predictions
    print(f"📖 正在加载预测结果: {pred_json}")
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_dict = json.load(f)
    if 'config' in pred_dict:
        del pred_dict['config']

    # 3. 逐图计算 TP 和 FP (基于 IoU 0.5)
    print("⚙️ 正在计算全局 P-R 数据...")
    all_images = set(gt_dict.keys()).union(set(pred_dict.keys()))
    
    all_tps = []
    all_confs = []
    
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
            # 限制最多框数，对齐 YOLO 评测标准
            if detections.shape[0] > 300:
                detections = detections[:300]
        else:
            detections = torch.empty(0, 6)

        tp = torch.zeros(detections.shape[0], dtype=torch.bool)
        
        if labels.shape[0] > 0 and detections.shape[0] > 0:
            ious = box_iou(labels[:, 1:], detections[:, :4])
            correct_class = labels[:, 0:1] == detections[:, 5]
            
            # IoU 阈值设为 0.5
            matches = torch.nonzero((ious >= 0.5) & correct_class)
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
                        tp[d_idx] = True
                        
        if detections.shape[0] > 0:
            all_tps.append(tp.cpu().numpy())
            all_confs.append(detections[:, 4].cpu().numpy())

    # 4. 合并所有图像的数据并全局排序
    if len(all_tps) == 0:
        print("❌ 警告：未检测到任何预测框！")
        return
        
    global_tps = np.concatenate(all_tps)
    global_confs = np.concatenate(all_confs)
    
    # 按照置信度降序排序
    sort_idx = np.argsort(-global_confs)
    global_tps = global_tps[sort_idx]
    global_confs = global_confs[sort_idx]
    
    # 计算累加 TP 和 FP
    tpc = np.cumsum(global_tps)
    fpc = np.cumsum(1 - global_tps)
    
    # 计算每一点的 P 和 R
    recalls = tpc / (total_gts + 1e-16)
    precisions = tpc / (tpc + fpc + 1e-16)

    # ================= 5. 开始各种指标查询 =================
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("📊 YOLO 伪标签阈值挖掘分析报告")
    report_lines.append("=" * 60)
    report_lines.append(f"数据集总真实框 (GT) 数量: {total_gts}")
    report_lines.append(f"模型总预测框 (Pred) 数量: {len(global_confs)}")
    report_lines.append("-" * 60)

    # 任务 1: 给定阈值列表，计算 P 和 R
    report_lines.append("【任务 1: 给定 Conf 阈值 -> 查询 Precision & Recall】")
    report_lines.append(f"{'Target Conf':>15} | {'Precision':>15} | {'Recall':>15}")
    for th in target_thresholds:
        valid_idxs = np.where(global_confs >= th)[0]
        if len(valid_idxs) > 0:
            idx = valid_idxs[-1]  # 满足该阈值的最后一个框（最低置信度）
            p, r = precisions[idx], recalls[idx]
        else:
            p, r = 1.0, 0.0
        report_lines.append(f"{th:>15.4f} | {p:>15.4f} | {r:>15.4f}")
    
    report_lines.append("-" * 60)
    
    # 任务 2: 给定召回率 R，计算阈值和 P (通常用于确定 th_l)
    report_lines.append("【任务 2: 给定目标 Recall -> 查询 阈值 & Precision】")
    report_lines.append(f"{'Target Recall':>15} | {'Req. Conf(th_l)':>15} | {'Precision':>15}")
    for tr in target_recalls:
        valid_idxs = np.where(recalls >= tr)[0]
        if len(valid_idxs) > 0:
            idx = valid_idxs[0]  # 首次达到该召回率的索引
            th, p = global_confs[idx], precisions[idx]
        else:
            th, p = 0.0, precisions[-1] # 达不到该召回率，取最低阈值
        report_lines.append(f"{tr:>15.4f} | {th:>15.4f} | {p:>15.4f}")

    report_lines.append("-" * 60)
    
    # 任务 3: 给定精确率 P，计算阈值和 R (通常用于确定 th_h)
    report_lines.append("【任务 3: 给定目标 Precision -> 查询 阈值 & Recall】")
    report_lines.append(f"{'Target Precision':>16} | {'Req. Conf(th_h)':>15} | {'Recall':>15}")
    for tp_val in target_precisions:
        # 为了稳定，寻找置信度尽可能低，且往后看 Precision 始终不低于目标的那个点
        valid_idxs = np.where(precisions >= tp_val)[0]
        if len(valid_idxs) > 0:
            idx = valid_idxs[-1] # 能维持该精度的最低置信度
            th, r = global_confs[idx], recalls[idx]
        else:
            th, r = 1.0, 0.0
        report_lines.append(f"{tp_val:>16.4f} | {th:>15.4f} | {r:>15.4f}")
        
    report_lines.append("=" * 60)

    # 6. 打印并保存
    report_str = "\n".join(report_lines)
    print(report_str)
    
    txt_save_path = output_dir / "threshold_mining_report.txt"
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"\n📄 阈值分析报告已成功导出至: {txt_save_path}")

if __name__ == '__main__':
    # 文件路径
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    # PRED_JSON = "/data/ZS/defect_dataset/9_yolo_preds/val_official/decision_fusion.json"
    PRED_JSON = "/data/ZS/flywheel_dataset/10_preds_on_gt_val/iter1/decision_fusion.json"
    OUTPUT_DIR = "/data/ZS/defect-vlm/output"
    
    # ============ 配置你需要查询的目标值 ============
    # 1. 想看的置信度阈值 (看看 YOLO 在这些常规阈值下的表现)
    TEST_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.85]
    
    # 2. 目标召回率 (用于确定 th_l，确保极少漏检，比如 95%, 98%)
    TARGET_RECALLS = [0.65, 0.70, 0.75, 0.80]
    
    # 3. 目标精确率 (用于确定 th_h，确保直接生成的伪标签极度纯净，比如 95%, 98%)
    TARGET_PRECISIONS = [0.85, 0.90, 0.95, 0.98]
    # ===============================================
    
    calculate_pr_thresholds(PRED_JSON, GT_JSON, OUTPUT_DIR, 
                            TEST_THRESHOLDS, TARGET_RECALLS, TARGET_PRECISIONS)