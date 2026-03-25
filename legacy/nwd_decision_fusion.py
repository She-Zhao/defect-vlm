"""
决策融合实现
输入：横向、纵向网络检测结果的json文件
输出：决策融合后的结果
"""
import json
import os
import numpy as np
from ensemble_boxes import soft_nms

# 类别 ID 到 名称 的映射 (用于融合后还原)
CLASS_ID_TO_NAME = {
    0: 'breakage', 1: 'inclusion', 2: 'scratch',
    3: 'crater', 4: 'run', 5: 'bulge'
}

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_area_ratio(bbox, img_w, img_h):
    """计算边界框占全图的相对面积比 R_A"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return (w * h) / (img_w * img_h)

def calculate_nwd(box1, box2, C):
    """计算两个边界框之间的归一化 Wasserstein 距离 (NWD)"""
    # 计算框1的中心点和宽高
    cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    
    # 计算框2的中心点和宽高
    cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    # 计算二阶 Wasserstein 距离平方 (W_2^2)
    w2_dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2 + ((w1 - w2)**2 + (h1 - h2)**2) / 4.0
    
    # 指数归一化得到 NWD 相似度 (0 到 1 之间)
    nwd = np.exp(-np.sqrt(w2_dist_sq) / C)
    return nwd

def nwd_consensus_fusion(preds_A, preds_B, nwd_thr, C):
    """基于 NWD 和联合概率的微小缺陷跨源共识融合机制"""
    fused_preds = []
    used_B = set()
    
    # 按照置信度降序排列，优先匹配高分框
    preds_A = sorted(preds_A, key=lambda x: x['confidence'], reverse=True)
    preds_B = sorted(preds_B, key=lambda x: x['confidence'], reverse=True)
    
    # 以网络 A 为基准去匹配网络 B
    for pA in preds_A:
        best_nwd = 0.0
        best_idx = -1
        
        # 寻找网络 B 中同类别且 NWD 最高的框
        for j, pB in enumerate(preds_B):
            if j in used_B: continue
            if pA['class_id'] != pB['class_id']: continue
            
            nwd = calculate_nwd(pA['bbox'], pB['bbox'], C)
            if nwd > best_nwd:
                best_nwd = nwd
                best_idx = j
                
        # 如果找到的最高 NWD 大于阈值，说明双流网络达成了共识
        if best_nwd > nwd_thr:
            used_B.add(best_idx)
            pB = preds_B[best_idx]
            
            # 1. 置信度提权：概率联合公式 (A + B - A*B)
            new_conf = pA['confidence'] + pB['confidence'] - (pA['confidence'] * pB['confidence'])
            
            # 2. 最优坐标保留：不求平均，直接取置信度高的那个框的坐标
            best_box = pA['bbox'] if pA['confidence'] > pB['confidence'] else pB['bbox']
            
            fused_preds.append({
                "class_id": pA['class_id'],
                "class_name": pA['class_name'],
                "bbox": best_box,
                "confidence": new_conf,
                "model_source": "fused_NWD_consensus" # 标记为共识强化框
            })
        else:
            # 单流检出：网络 A 有，网络 B 没有
            pA_copy = pA.copy()
            pA_copy['model_source'] = "fused_NWD_single"
            fused_preds.append(pA_copy)
            
    # 单流检出：网络 B 有，网络 A 没有
    for j, pB in enumerate(preds_B):
        if j not in used_B:
            pB_copy = pB.copy()
            pB_copy['model_source'] = "fused_NWD_single"
            fused_preds.append(pB_copy)
            
    return fused_preds

def format_for_ensemble(preds_list_1, preds_list_2, img_w, img_h):
    """(仅供 Soft-NMS 使用) 格式化为归一化格式"""
    boxes_list, scores_list, labels_list = [[], []], [[], []], [[], []]
    for i, preds in enumerate([preds_list_1, preds_list_2]):
        for p in preds:
            x1 = max(0.0, min(1.0, p['bbox'][0] / img_w))
            y1 = max(0.0, min(1.0, p['bbox'][1] / img_h))
            x2 = max(0.0, min(1.0, p['bbox'][2] / img_w))
            y2 = max(0.0, min(1.0, p['bbox'][3] / img_h))
            if x2 <= x1 or y2 <= y1: continue
            boxes_list[i].append([x1, y1, x2, y2])
            scores_list[i].append(p['confidence'])
            labels_list[i].append(p['class_id'])
    return boxes_list, scores_list, labels_list

def denormalize_boxes(boxes, scores, labels, strategy_name, img_w, img_h):
    """(仅供 Soft-NMS 使用) 还原为绝对坐标格式"""
    fused_preds = []
    for box, score, label in zip(boxes, scores, labels):
        abs_bbox = [box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h]
        class_id = int(label)
        fused_preds.append({
            "class_id": class_id,
            "class_name": CLASS_ID_TO_NAME.get(class_id, "unknown"),
            "bbox": abs_bbox,
            "confidence": float(score),
            "model_source": f"fused_{strategy_name}"
        })
    return fused_preds

def filter_empty_predictions(boxes, scores, labels, weights):
    """(仅供 Soft-NMS 使用) 过滤空分支"""
    v_boxes, v_scores, v_labels, v_weights = [], [], [], []
    for b, s, l, w in zip(boxes, scores, labels, weights):
        if len(b) > 0:
            v_boxes.append(b)
            v_scores.append(s)
            v_labels.append(l)
            v_weights.append(w)
    return v_boxes, v_scores, v_labels, v_weights

def process_scale_aware_fusion(json1_path, json2_path, out_path, config):  
    print("加载预测结果 JSON 文件...")
    preds_col3 = load_json(json1_path)
    preds_row3 = load_json(json2_path)
    
    img_w, img_h = config['IMG_W'], config['IMG_H']
    area_th = config['AREA_TH']
    iou_thr_soft = config['IOU_THR_SOFT']
    sigma_soft = config['SIGMA_SOFT']
    skip_box_thr = config['SKIP_BOX_THR']
    
    # NWD 独占参数
    nwd_thr = config['NWD_THR']
    nwd_c = config['NWD_C']

    all_images = set(preds_col3.keys()).union(set(preds_row3.keys()))
    fused_results = {}
    
    print(f"共发现 {len(all_images)} 张图像，开始自适应尺度融合 (NWD + SoftNMS)...")
    
    for img_name in all_images:
        img_preds_col3 = preds_col3.get(img_name, [])
        img_preds_row3 = preds_row3.get(img_name, [])
        
        # --- 1. 尺度路由 ---
        small_col3, large_col3 = [], []
        for p in img_preds_col3:
            (small_col3 if get_area_ratio(p['bbox'], img_w, img_h) <= area_th else large_col3).append(p)
            
        small_row3, large_row3 = [], []
        for p in img_preds_row3:
            (small_row3 if get_area_ratio(p['bbox'], img_w, img_h) <= area_th else large_row3).append(p)
            
        final_img_preds = []
        
        # --- 2. 微小缺陷分支：使用 NWD 共识强化融合 (无需归一化，直接算) ---
        fused_small = nwd_consensus_fusion(small_col3, small_row3, nwd_thr=nwd_thr, C=nwd_c)
        # 对单边保留下来的极低置信度噪点做个过滤
        fused_small = [p for p in fused_small if p['confidence'] >= skip_box_thr]
        final_img_preds.extend(fused_small)

        # --- 3. 大尺度缺陷分支：使用 Soft-NMS ---
        b_l, s_l, l_l = format_for_ensemble(large_col3, large_row3, img_w, img_h)
        vb_l, vs_l, vl_l, vw_l = filter_empty_predictions(b_l, s_l, l_l, [1, 1])
        
        if len(vb_l) > 0:
            fused_b, fused_s, fused_l = soft_nms(
                vb_l, vs_l, vl_l, 
                weights=vw_l, iou_thr=iou_thr_soft, sigma=sigma_soft, thresh=skip_box_thr
            )
            final_img_preds.extend(denormalize_boxes(fused_b, fused_s, fused_l, "SoftNMS", img_w, img_h))
            
        # --- 4. 汇总该图像的最终预测 ---
        final_img_preds.sort(key=lambda x: x['confidence'], reverse=True)
        fused_results[img_name] = final_img_preds

    print(f"融合完成！正在保存至 {out_path} ...")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fused_results, f, indent=2, ensure_ascii=False)
    print("处理完毕！")

if __name__ == "__main__":
    
    config = {
        'IMG_W': 300.0,             
        'IMG_H': 300.0,             
        'SIGMA_SOFT': 0.05,         
        'SKIP_BOX_THR': 0.15,       
        'AREA_TH': 0,          
        'IOU_THR_SOFT': 0.45,       
        
        # --- 新增：NWD 核心调参区 ---
        'NWD_THR': 0.8,            # NWD 相似度匹配阈值 (越接近1越严格，通常取 0.8~0.9)
        'NWD_C': 50.0               # 常数 C，通常设为数据集微小目标的平均绝对尺寸(像素)
    }

    # 路径配置
    json_col3 = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/col3.json"
    json_row3 = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/row3.json"
    output_json = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/decision_fusion_nwd.json"
    
    process_scale_aware_fusion(json_col3, json_row3, output_json, config)