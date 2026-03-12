"""
计算fusion.json经过vlm过滤后的指标
输入：vlm推理后的jsonl文件 + 原始验证集的数据（coco格式）
输出：一个文件夹，里面包括一系列指标
"""
"""
计算 VLM (Qwen-VL) 推理结果的各项检测指标
输入：VLM 的 JSONL 结果 + 中间映射 JSON + 原始 COCO 格式 GT
输出：PR 曲线、F1 曲线、混淆矩阵等
"""
import os
import re
import sys
import json
import torch
import numpy as np
from pathlib import Path

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

def extract_probability(token_probs, target_class):
    """从概率流中倒序提取目标类别的真实概率"""
    for t_info in reversed(token_probs):
        token_str = t_info["token"].strip().strip('"').strip("'").lower()
        # 只要 token 包含类别的核心部分，或者是类别名的一部分
        if token_str in target_class or target_class in token_str:
            return t_info["probability"]
    return 0.5  # 极端情况下的兜底概率

def evaluate_vlm_results(vlm_jsonl, inter_json, gt_json, output_dir):
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
            prob = extract_probability(item.get("pred_token_probs", []), pred_cls_name)
            
            # xywh 转换为 xyxy
            x, y, w, h = item["meta_info"]["bbox"]
            pred_dict[filename].append({
                'bbox': [x, y, x + w, y + h],
                'confidence': prob,
                'class_id': class_id
            })

    # 4. 准备 Ultralytics 评估矩阵
    iouv = torch.linspace(0.5, 0.95, 10)
    cm = ConfusionMatrix(nc=len(names_dict), conf=0.25, iou_thres=0.45)
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

    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=True)
    cm.plot(save_dir=output_dir, names=tuple(names_dict.values()), normalize=False)
    
    print("=" * 60)
    print(f"🏆 评估结果 (VLM Model) 已保存至: {output_dir}")
    print(f"{'Class':>15} {'mAP@50':>10} {'mAP@50-95':>12}")
    print("-" * 60)
    print(f"{'all':>15} {ap50.mean():>10.4f} {ap.mean():>12.4f}")
    for i, c in enumerate(ap_class):
        print(f"{names_dict[c]:>15} {ap50[i]:>10.4f} {ap[i]:>12.4f}")
    print("=" * 60)

if __name__ == '__main__':
    # 1. 真实的标注 JSON (用于提取答案和图片名称)
    GT_JSON = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    
    # 2. 我们拼图那一步生成的中间 JSON (包含了 origin_id 和 original_image_path 的映射关系)
    INTER_JSON = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p1.json"
    
    # 3. VLM 跑出来的最终 JSONL
    VLM_JSONL = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v2_qwen3_4b_LM.jsonl"
    
    # 4. 图表和曲线存放的目录
    OUTPUT_DIR = "/data/ZS/defect-vlm/output/figures/"
    
    evaluate_vlm_results(VLM_JSONL, INTER_JSON, GT_JSON, OUTPUT_DIR)
     