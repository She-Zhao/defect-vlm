"""
实验一：级联效能对比实验 (Base YOLO vs. YOLO+VLM Cascade)
目的：计算固定阈值下，Base YOLO 与 Cascade 系统的 Precision, Recall, F1
输入：GT JSON, YOLO 预测 JSON, VLM 预测 JSONL
输出：各类别及宏观平均的 P, R, F1 对比表 (同时打印到终端并保存为 txt)
"""
import os
import re
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# ================= 工具函数 =================
def parse_vlm_prediction(pred_text):
    """鲁棒地解析 VLM 预测的类别"""
    try:
        clean_text = pred_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("defect", "background").lower()
    except Exception:
        match = re.search(r'"defect"\s*:\s*"([^"]+)"', pred_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "background"

def compute_iou(box1, box2):
    """计算两个 [x1, y1, x2, y2] 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

# ================= 核心评估引擎 =================
def evaluate_predictions(preds_dict, gt_dict, num_classes, iou_thresh=0.45):
    """
    计算 TP, FP, FN 并返回 P, R, F1
    """
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    
    all_files = set(preds_dict.keys()).union(set(gt_dict.keys()))
    
    for filename in all_files:
        gts = gt_dict.get(filename, [])
        preds = preds_dict.get(filename, [])
        
        matched_gt_indices = set()
        
        # 遍历每一个预测框寻找匹配
        for p in preds:
            p_cls = p['class_id']
            p_box = p['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            
            for idx, g in enumerate(gts):
                if idx in matched_gt_indices:
                    continue
                if g['class_id'] != p_cls:
                    continue
                    
                iou = compute_iou(p_box, g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # 如果最高 IoU 大于阈值，则记为 TP，否则为 FP
            if best_iou >= iou_thresh:
                tp[p_cls] += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp[p_cls] += 1
                
        # 剩下的未匹配的 GT 记为 FN
        for idx, g in enumerate(gts):
            if idx not in matched_gt_indices:
                fn[g['class_id']] += 1
                
    # 计算指标，避免除以0
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) != 0)
    
    return p, r, f1, tp, fp, fn

# ================= 主控流 =================
def main(gt_json_path, yolo_json_path, vlm_jsonl_path, output_dir,
         base_yolo_conf_thresh=0.25, 
         cascade_yolo_conf_thresh=0.1, # <-- 暴露出来的级联基础阈值
         iou_thresh=0.45):
    
    # 创建输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"experiment1_cascade_results_base{base_yolo_conf_thresh}_casc{cascade_yolo_conf_thresh}.txt"

    # 用于双向输出（终端 + 文件）的辅助函数
    def log_print(text, file_handle):
        print(text)
        file_handle.write(text + "\n")

    # 1. 加载 GT
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    names_dict = {cat['id']: cat['name'] for cat in coco_gt['categories']}
    name2id = {cat['name']: cat['id'] for cat in coco_gt['categories']}
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    num_classes = len(names_dict)
    
    gt_dict = defaultdict(list)
    for ann in coco_gt['annotations']:
        filename = imgid2name[ann['image_id']]
        c_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt_dict[filename].append({'class_id': c_id, 'bbox': [x, y, x + w, y + h]})
        
    # 2. 加载 YOLO 预测并建立 ID 映射
    with open(yolo_json_path, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)
        
    id2filename = {}
    base_yolo_preds = defaultdict(list)
    cascade_yolo_inputs = {}  
    
    for item in yolo_data:
        origin_id = item["id"]
        orig_path = item["original_image_paths"][0]
        filename = os.path.basename(orig_path)
        id2filename[origin_id] = filename
        
        conf = item["confidence"]
        prior_cls_name = item["prior_label"].lower()
        if prior_cls_name not in name2id:
            continue
        c_id = name2id[prior_cls_name]
        
        x, y, w, h = item["bbox"]
        bbox = [x, y, x + w, y + h]
        
        # 记录给 Cascade 系统的底层提议框 (增加显式的阈值过滤)
        if conf >= cascade_yolo_conf_thresh:
            cascade_yolo_inputs[origin_id] = {'filename': filename, 'bbox': bbox}
        
        # 记录 Base YOLO 的基线预测 (常规较高阈值)
        if conf >= base_yolo_conf_thresh:
            base_yolo_preds[filename].append({'class_id': c_id, 'bbox': bbox, 'conf': conf})

    # 3. 加载 VLM 预测 (构建 Cascade 结果)
    cascade_preds = defaultdict(list)
    with open(vlm_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            origin_id = item["meta_info"]["origin_id"]
            if origin_id not in cascade_yolo_inputs: continue
            
            filename = cascade_yolo_inputs[origin_id]['filename']
            bbox = cascade_yolo_inputs[origin_id]['bbox']
            
            vlm_cls_name = parse_vlm_prediction(item.get("pred", ""))
            
            if vlm_cls_name != "background" and vlm_cls_name in name2id:
                c_id = name2id[vlm_cls_name]
                cascade_preds[filename].append({'class_id': c_id, 'bbox': bbox})

    # 4. 执行评估
    with open(out_file, 'w', encoding='utf-8') as log_f:
        log_print("="*90, log_f)
        log_print(f"📊 实验一：基线 YOLO (Conf>{base_yolo_conf_thresh}) vs. 级联 VLM (YOLO Conf>{cascade_yolo_conf_thresh} + VLM)", log_f)
        log_print("="*90, log_f)
        
        p_base, r_base, f1_base, _, _, _ = evaluate_predictions(base_yolo_preds, gt_dict, num_classes, iou_thresh)
        p_casc, r_casc, f1_casc, _, _, _ = evaluate_predictions(cascade_preds, gt_dict, num_classes, iou_thresh)

        # 5. 打印并保存对比表格
        header = f"{'Category':<12} | {'Base Precision':<15} {'Base Recall':<15} {'Base F1':<10} | {'VLM Precision':<15} {'VLM Recall':<15} {'VLM F1':<10}"
        log_print(header, log_f)
        log_print("-" * len(header), log_f)
        
        for c_id in range(num_classes):
            c_name = names_dict[c_id]
            row_str = (f"{c_name:<12} | {p_base[c_id]:<15.4f} {r_base[c_id]:<15.4f} {f1_base[c_id]:<10.4f} | "
                       f"{p_casc[c_id]:<15.4f} {r_casc[c_id]:<15.4f} {f1_casc[c_id]:<10.4f}")
            log_print(row_str, log_f)
            
        log_print("-" * len(header), log_f)
        
        macro_str = (f"{'Macro-Avg':<12} | {np.mean(p_base):<15.4f} {np.mean(r_base):<15.4f} {np.mean(f1_base):<10.4f} | "
                     f"{np.mean(p_casc):<15.4f} {np.mean(r_casc):<15.4f} {np.mean(f1_casc):<10.4f}")
        log_print(macro_str, log_f)
        log_print("="*90, log_f)
        log_print(f"💾 评估报告已保存至: {out_file}", log_f)

if __name__ == "__main__":
    # 配置输入路径
    GT_JSON_PATH = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    YOLO_JSON_PATH = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p1.json"
    VLM_JSONL_PATH = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v2_qwen3_4b_LM.jsonl"
    
    # 配置输出文件夹路径 (自由指定)
    OUTPUT_DIRECTORY = "/data/ZS/defect-vlm/output/figures/"
    
    main(
        gt_json_path=GT_JSON_PATH,
        yolo_json_path=YOLO_JSON_PATH,
        vlm_jsonl_path=VLM_JSONL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        base_yolo_conf_thresh=0.25,      # 基线 YOLO 部署卡 0.25
        cascade_yolo_conf_thresh=0.1,    # VLM 级联系统底层放宽到 0.1
        iou_thresh=0.45                  # 与验证指标保持一致
    )