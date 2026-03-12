"""
实验二：VLM 仲裁流向与降噪漏斗分析 (Arbitration Flow Analysis)
目的：量化 VLM 在级联系统中的具体作用（排伪、矫正、顺从），证明其“二验大脑”的价值。
输入：GT JSON, YOLO 预测 JSON, VLM 预测 JSONL
输出：详细的流向统计报告 (同时打印到终端并保存为 txt)
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

# ================= 主控流 =================
def analyze_arbitration_flow(gt_json_path, yolo_json_path, vlm_jsonl_path, output_dir,
                             cascade_yolo_conf_thresh=0.1, 
                             iou_thresh=0.45):
    
    # 准备输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"experiment2_flow_analysis_casc{cascade_yolo_conf_thresh}.txt"

    def log_print(text, file_handle):
        print(text)
        file_handle.write(text + "\n")

    # 1. 加载 GT
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    names_dict = {cat['id']: cat['name'] for cat in coco_gt['categories']}
    imgid2name = {img['id']: img['file_name'] for img in coco_gt['images']}
    
    gt_dict = defaultdict(list)
    for ann in coco_gt['annotations']:
        filename = imgid2name[ann['image_id']]
        c_name = names_dict[ann['category_id']]
        x, y, w, h = ann['bbox']
        gt_dict[filename].append({'class_name': c_name, 'bbox': [x, y, x + w, y + h]})

    # 2. 加载 YOLO 预测并分配 "上帝视角" 的真实标签 (True Label)
    with open(yolo_json_path, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)
        
    # 按图片名将 YOLO 预测分组，并按置信度降序排序 (IoU匹配的标准做法)
    yolo_preds_by_img = defaultdict(list)
    for item in yolo_data:
        if item["confidence"] < cascade_yolo_conf_thresh:
            continue
        origin_id = item["id"]
        orig_path = item["original_image_paths"][0]
        filename = os.path.basename(orig_path)
        
        x, y, w, h = item["bbox"]
        bbox = [x, y, x + w, y + h]
        
        yolo_preds_by_img[filename].append({
            'origin_id': origin_id,
            'prior_label': item["prior_label"].lower(),
            'conf': item["confidence"],
            'bbox': bbox
        })
        
    for filename in yolo_preds_by_img:
        yolo_preds_by_img[filename].sort(key=lambda x: x['conf'], reverse=True)

    # 给每个 YOLO 框匹配 GT 标签
    yolo_ground_truth_map = {} # origin_id -> true_label
    for filename, preds in yolo_preds_by_img.items():
        gts = gt_dict.get(filename, [])
        matched_gt_indices = set()
        
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, g in enumerate(gts):
                iou = compute_iou(p['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_thresh and best_gt_idx not in matched_gt_indices:
                true_label = gts[best_gt_idx]['class_name']
                matched_gt_indices.add(best_gt_idx)
            else:
                true_label = "background"
                
            yolo_ground_truth_map[p['origin_id']] = true_label

    # 将 YOLO 框信息展平，方便后续查询
    yolo_proposals = {}
    for preds in yolo_preds_by_img.values():
        for p in preds:
            yolo_proposals[p['origin_id']] = {
                'prior_label': p['prior_label'],
                'true_label': yolo_ground_truth_map[p['origin_id']]
            }

    # 3. 加载 VLM 预测并开始流向统计
    flow_stats = {
        'total_analyzed': 0,
        'yolo_fake_alarms': 0, # 实际是背景的 YOLO 框
        'agreement': {'total': 0, 'correct': 0, 'wrong': 0},
        'noise_reduction': {'total': 0, 'true_background_killed': 0, 'real_defect_killed': 0},
        'rectification': {'total': 0, 'vlm_correct': 0, 'yolo_was_correct': 0, 'both_wrong': 0}
    }

    with open(vlm_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            origin_id = item["meta_info"]["origin_id"]
            if origin_id not in yolo_proposals: 
                continue
                
            yolo_info = yolo_proposals[origin_id]
            yolo_cls = yolo_info['prior_label']
            true_cls = yolo_info['true_label']
            vlm_cls = parse_vlm_prediction(item.get("pred", ""))
            
            flow_stats['total_analyzed'] += 1
            if true_cls == "background":
                flow_stats['yolo_fake_alarms'] += 1

            # --- 流向 1：降噪 (VLM 预测为 background) ---
            if vlm_cls == "background":
                flow_stats['noise_reduction']['total'] += 1
                if true_cls == "background":
                    flow_stats['noise_reduction']['true_background_killed'] += 1
                else:
                    flow_stats['noise_reduction']['real_defect_killed'] += 1
                    
            # --- 流向 2：顺从 (VLM 同意 YOLO) ---
            elif vlm_cls == yolo_cls:
                flow_stats['agreement']['total'] += 1
                if vlm_cls == true_cls:
                    flow_stats['agreement']['correct'] += 1
                else:
                    flow_stats['agreement']['wrong'] += 1
                    
            # --- 流向 3：矫正 (VLM 提出了新的缺陷类别) ---
            else:
                flow_stats['rectification']['total'] += 1
                if vlm_cls == true_cls:
                    flow_stats['rectification']['vlm_correct'] += 1
                elif yolo_cls == true_cls:
                    flow_stats['rectification']['yolo_was_correct'] += 1
                else:
                    flow_stats['rectification']['both_wrong'] += 1

    # 4. 打印报告
    with open(out_file, 'w', encoding='utf-8') as log_f:
        log_print("="*80, log_f)
        log_print(f"🌊 实验二：VLM 仲裁流向与降噪漏斗分析 (YOLO Conf > {cascade_yolo_conf_thresh})", log_f)
        log_print("="*80, log_f)
        
        log_print(f"📦 【初始提议池】", log_f)
        log_print(f"  - YOLO 共推送了 {flow_stats['total_analyzed']} 个疑似缺陷框给 VLM。", log_f)
        log_print(f"  - 经上帝视角(GT)校验，其中有 {flow_stats['yolo_fake_alarms']} 个是纯粹的背景噪点（假阳性）。", log_f)
        log_print("-" * 80, log_f)
        
        # 降噪分析
        nr = flow_stats['noise_reduction']
        log_print(f"🛡️ 【流向 A：噪点过滤 (Noise Reduction)】 - VLM 认为这是背景，予以丢弃", log_f)
        log_print(f"  - 触发次数：{nr['total']} 次", log_f)
        if nr['total'] > 0:
            success_rate = nr['true_background_killed'] / nr['total'] * 100
            log_print(f"  - ✅ 成功拦截假阳性：{nr['true_background_killed']} 个 (降噪准确率: {success_rate:.1f}%)", log_f)
            log_print(f"  - ❌ 误杀真实缺陷：{nr['real_defect_killed']} 个", log_f)
        
        # 矫正分析
        rect = flow_stats['rectification']
        log_print(f"\n🔧 【流向 B：类别矫正 (Rectification)】 - VLM 推翻 YOLO 结论，修改为其他缺陷", log_f)
        log_print(f"  - 触发次数：{rect['total']} 次", log_f)
        if rect['total'] > 0:
            v_acc = rect['vlm_correct'] / rect['total'] * 100
            log_print(f"  - ✅ VLM 成功纠错：{rect['vlm_correct']} 个 (矫正有效率: {v_acc:.1f}%)", log_f)
            log_print(f"  - ❌ VLM 帮倒忙 (YOLO原本是对的)：{rect['yolo_was_correct']} 个", log_f)
            log_print(f"  - ⚠️ 双方都猜错了：{rect['both_wrong']} 个", log_f)
            
        # 顺从分析
        agree = flow_stats['agreement']
        log_print(f"\n🤝 【流向 C：共识确认 (Agreement)】 - VLM 同意 YOLO 的预测类别", log_f)
        log_print(f"  - 触发次数：{agree['total']} 次", log_f)
        if agree['total'] > 0:
            a_acc = agree['correct'] / agree['total'] * 100
            log_print(f"  - ✅ 确认正确：{agree['correct']} 个 (共识准确率: {a_acc:.1f}%)", log_f)
            log_print(f"  - ❌ 确认错误 (共同踩坑)：{agree['wrong']} 个", log_f)
            
        log_print("="*80, log_f)
        
        # 核心论点提取
        yolo_false_positives = flow_stats['yolo_fake_alarms']
        if yolo_false_positives > 0:
            fprr = nr['true_background_killed'] / yolo_false_positives * 100
            log_print("💡 核心学术结论：", log_f)
            log_print(f"  - 误报消除率 (FPRR): VLM 成功清除了 YOLO 产生的 {fprr:.1f}% 的背景噪点框！", log_f)
        log_print(f"💾 详细流向报告已保存至: {out_file}", log_f)

if __name__ == "__main__":
    GT_JSON_PATH = "/data/ZS/defect_dataset/0_defect_dataset_raw/paint_stripe/labels/val.json"
    YOLO_JSON_PATH = "/data/ZS/defect_dataset/11_composite_yolo_preds/stripe_phase012/labels/val_0p1.json"
    VLM_JSONL_PATH = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v2_qwen3_4b_LM.jsonl"
    OUTPUT_DIRECTORY = "/data/ZS/defect-vlm/output/figures/"
    
    analyze_arbitration_flow(
        gt_json_path=GT_JSON_PATH,
        yolo_json_path=YOLO_JSON_PATH,
        vlm_jsonl_path=VLM_JSONL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        cascade_yolo_conf_thresh=0.1,  
        iou_thresh=0.45                
    )