"""
计算横向和纵向条纹光融合后的结果（fusion.json）的各项指标
输入：多流主干网络的推理结果 + 原始验证集的数据（coco格式）
输出：一个文件夹，里面包括一系列指标
"""
import os
import sys
import json
import torch
from pathlib import Path

# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

# 导入 Ultralytics 的核心评估和绘图工具
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

def evaluate_fusion_results(pred_json, gt_json, output_dir):
    """
    计算并绘制融合结果的各项检测指标
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 解析 Ground Truth (COCO 格式)
    print(f"📖 正在加载真实标签: {gt_json}")
    with open(gt_json, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    # 建立映射字典
    names_dict = {cat['id']: cat['name'] for cat in coco_gt['categories']}
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    
    # 将 COCO 的 [x,y,w,h] 转换为 YOLO 的 [x1,y1,x2,y2]
    gt_dict = {img['file_name']: [] for img in coco_gt['images']}
    for ann in coco_gt['annotations']:
        img_name = imgid2name[ann['image_id']]
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt_dict[img_name].append([cat_id, x, y, x + w, y + h])
        
    # 2. 解析 Predictions (你的 fusion.json)
    print(f"📖 正在加载预测结果: {pred_json}")
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_dict = json.load(f)

    # 3. 准备 Ultralytics 的评估工具
    # 设置 10 个 IoU 阈值，从 0.5 到 0.95 (用于计算 mAP@50-95)
    iouv = torch.linspace(0.5, 0.95, 10)
    
    # 初始化混淆矩阵 (置信度阈值设为 0.25，这是 YOLO 生成混淆矩阵的标准设定)
    cm = ConfusionMatrix(nc=len(names_dict), conf=0.25, iou_thres=0.45)
    
    stats = []  # 用于收集所有的 (True_Positives, Confidence, Pred_Class, Target_Class)

    # 4. 逐张图对比计算
    print("⚙️ 正在对比每一张图像的预测框与真实框...")
    all_images = set(gt_dict.keys()).union(set(pred_dict.keys()))
    
    for filename in all_images:
        gts = gt_dict.get(filename, [])
        preds = pred_dict.get(filename, [])
        
        # 转换为 Tensor 格式
        # labels: [M, 5] (class_id, x1, y1, x2, y2)
        labels = torch.tensor(gts, dtype=torch.float32) if len(gts) > 0 else torch.empty(0, 5)
        
        # detections: [N, 6] (x1, y1, x2, y2, conf, class_id)
        if len(preds) > 0:
            detections = torch.tensor(
                [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3], p['confidence'], p['class_id']]
                 for p in preds], dtype=torch.float32
            )
            # 必须按照置信度降序排列
            detections = detections[detections[:, 4].argsort(descending=True)]
        else:
            detections = torch.empty(0, 6)

        # 核心逻辑：匹配 TP (True Positives)
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
                            
        # 更新混淆矩阵 (拆分传入坐标和类别)
        cm.process_batch(detections, labels[:, 1:], labels[:, 0])
        
        # 记录给 mAP 计算器的数据
        stats.append((
            tp,
            detections[:, 4] if len(detections) > 0 else torch.empty(0),
            detections[:, 5] if len(detections) > 0 else torch.empty(0),
            labels[:, 0] if len(labels) > 0 else torch.empty(0)
        ))

    # 5. 拼合所有统计数据
    print("📊 正在计算 mAP 并绘制曲线...")
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]


    # 6. 生成所有指标并绘图
    # 新版 YOLO 的 ap_per_class 返回 12 个变量
    tp_arr, fp_arr, p, r, f1, all_ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, 
        plot=True, 
        save_dir=output_dir, 
        names=names_dict,
        prefix="Fusion_"  # 给图表加上前缀，防止和普通评估混淆
    )
    
    # 提取需要的 mAP@50 和 mAP@50-95 (all_ap 是所有阈值下的准确率，形状为 [nc, 10])
    ap50 = all_ap[:, 0]
    ap = all_ap.mean(1)

    # 绘制并保存混淆矩阵
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=True)
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=False)
    
    # 打印最终指标
    print("=" * 60)
    print(f"🏆 评估结果 (Fusion Model) 已保存至: {output_dir}")
    print(f"{'Class':>15} {'mAP@50':>10} {'mAP@50-95':>12}")
    print("-" * 60)
    print(f"{'all':>15} {ap50.mean():>10.4f} {ap.mean():>12.4f}")
    for i, c in enumerate(ap_class):
        print(f"{names_dict[c]:>15} {ap50[i]:>10.4f} {ap[i]:>12.4f}")
    print("=" * 60)

if __name__ == '__main__':
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    PRED_JSON = "/data/ZS/defect_dataset/9_yolo_preds/val/row3.json"        # 修改
    
    # 输出的图片会全部保存在这个文件夹里
    OUTPUT_DIR = "/data/ZS/defect-vlm/output/figures/检测模型指标/row3"      # 修改
    
    evaluate_fusion_results(PRED_JSON, GT_JSON, OUTPUT_DIR)
