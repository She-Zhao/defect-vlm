
"""
根据yolo推理得到的runs/detect/predictions.json，直接计算各项指标
省的再转换了
"""
"""
评估 YOLO 标准 COCO JSON 推理结果的各项检测指标
计算 mAP@0.5, mAP@0.75, mAP@[0.5:0.95]，并生成混淆矩阵
"""
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

from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

def evaluate_yolo_coco_results(pred_json, gt_json, output_dir):
    """
    计算并绘制标准 COCO 推理结果的各项检测指标
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
        
    # ================= 核心修改区域：解析新的 JSON 格式 =================
    print(f"📖 正在加载预测结果: {pred_json}")
    with open(pred_json, 'r', encoding='utf-8') as f:
        preds_list = json.load(f)

    pred_dict = {img_name: [] for img_name in gt_dict.keys()}
    
    # 建立一个去后缀的映射字典，防止预测结果里的 image_id 不带 .png
    base2filename = {Path(k).stem: k for k in gt_dict.keys()}
    
    for p in preds_list:
        img_id_str = str(p['image_id'])
        
        # 鲁棒的文件名匹配逻辑
        if img_id_str in gt_dict:
            file_name = img_id_str
        elif img_id_str in base2filename:
            file_name = base2filename[img_id_str]
        elif f"{img_id_str}.png" in gt_dict:
            file_name = f"{img_id_str}.png"
        else:
            continue  # 匹配不上 GT 的忽略
            
        cat_id = p['category_id']
        conf = p['score']
        x, y, w, h = p['bbox']
        
        # 【关键】：将绝对坐标 [x, y, w, h] 转为 [x1, y1, x2, y2]
        pred_dict[file_name].append({
            'bbox': [x, y, x + w, y + h],
            'confidence': conf,
            'class_id': cat_id
        })
    # ==================================================================

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
            
            # Top-K 截断机制，完美对齐官方评测
            max_det = 300  
            if detections.shape[0] > max_det:
                detections = detections[:max_det]

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
    tp_arr, fp_arr, p, r, f1, all_ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, 
        plot=True, 
        save_dir=output_dir, 
        names=names_dict,
        prefix="Eval_"  # 给图表加上前缀
    )
    
    # 提取 mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
    ap50 = all_ap[:, 0]       
    ap75 = all_ap[:, 5]       
    ap_mean = all_ap.mean(1)  

    # 绘制并保存混淆矩阵
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=True)
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=False)
    
    # 保存所有的指标并打印
    report_str = "=" * 70 + "\n"
    report_str += f"🏆 评估结果已保存至: {output_dir}\n"
    report_str += f"{'Class':>15} {'mAP@0.5':>10} {'mAP@0.75':>10} {'mAP@[0.5:0.95]':>15}\n"
    report_str += "-" * 70 + "\n"
    report_str += f"{'all':>15} {ap50.mean():>10.4f} {ap75.mean():>10.4f} {ap_mean.mean():>15.4f}\n"
    
    for i, c in enumerate(ap_class):
        report_str += f"{names_dict[c]:>15} {ap50[i]:>10.4f} {ap75[i]:>10.4f} {ap_mean[i]:>15.4f}\n"
    report_str += "=" * 70 + "\n"

    # 1. 打印到控制台
    print(report_str)

    # 2. 保存到 txt 文件
    txt_save_path = output_dir / "metrics_report.txt"
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"📄 指标表格已成功导出至: {txt_save_path}")

if __name__ == '__main__':
    # 1. 真实的标签 (COCO 格式)
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    PRED_JSON = "/data/ZS/v11_input/runs/detect/val2/predictions.json"
    
    # 3. 输出目录
    OUTPUT_DIR = "/data/ZS/defect-vlm/output/figures/检测模型指标"
    
    evaluate_yolo_coco_results(PRED_JSON, GT_JSON, OUTPUT_DIR)