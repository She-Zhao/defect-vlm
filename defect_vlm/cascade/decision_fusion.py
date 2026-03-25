"""
决策融合实现
输入：横向、纵向网络检测结果的json文件
输出：决策融合后的结果
"""
import json
import os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion, soft_nms

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

def format_for_ensemble(preds_list_1, preds_list_2, img_w, img_h):
    """将预测结果格式化为 ensemble_boxes 所需的归一化格式"""
    boxes_list = [[], []]
    scores_list = [[], []]
    labels_list = [[], []]
    
    for i, preds in enumerate([preds_list_1, preds_list_2]):
        for p in preds:
            # ensemble_boxes 严格要求坐标在 [0, 1] 之间，因此必须做归一化和截断
            x1 = max(0.0, min(1.0, p['bbox'][0] / img_w))
            y1 = max(0.0, min(1.0, p['bbox'][1] / img_h))
            x2 = max(0.0, min(1.0, p['bbox'][2] / img_w))
            y2 = max(0.0, min(1.0, p['bbox'][3] / img_h))
            
            # 防止无效框
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes_list[i].append([x1, y1, x2, y2])
            scores_list[i].append(p['confidence'])
            labels_list[i].append(p['class_id'])
            
    return boxes_list, scores_list, labels_list

def denormalize_boxes(boxes, scores, labels, strategy_name, img_w, img_h):
    """将融合后的归一化框还原为字典格式的绝对坐标框"""
    fused_preds = []
    for box, score, label in zip(boxes, scores, labels):
        # 将 [0, 1] 的坐标还原回原始像素尺寸
        abs_bbox = [
            box[0] * img_w,
            box[1] * img_h,
            box[2] * img_w,
            box[3] * img_h
        ]
        class_id = int(label)
        fused_preds.append({
            "class_id": class_id,
            "class_name": CLASS_ID_TO_NAME.get(class_id, "unknown"),
            "bbox": abs_bbox,
            "confidence": float(score),
            "model_source": f"fused_{strategy_name}" # 标记是由哪个算法融合出来的
        })
    return fused_preds

def filter_empty_predictions(boxes, scores, labels, weights):
    """过滤掉预测框为空的模型分支，既避免 ensemble_boxes 报错，又防止互补光源互相压低置信度"""
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
    
    img_w = config['IMG_W']
    img_h = config['IMG_H']
    area_th = config['AREA_TH']
    iou_thr_wbf = config['IOU_THR_WBF']
    iou_thr_soft = config['IOU_THR_SOFT']
    sigma_soft = config['SIGMA_SOFT']
    skip_box_thr = config['SKIP_BOX_THR']

    all_images = set(preds_col3.keys()).union(set(preds_row3.keys()))
    fused_results = {}
    fused_results['config'] = config
    
    print(f"共发现 {len(all_images)} 张图像，开始自适应尺度融合...")
    
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
        
        # --- 2. 微小缺陷分支：使用 WBF ---
        b_s, s_s, l_s = format_for_ensemble(small_col3, small_row3, img_w, img_h)
        # 【修改点】过滤掉没有预测出小缺陷的模型分支
        vb_s, vs_s, vl_s, vw_s = filter_empty_predictions(b_s, s_s, l_s, [1, 1])
        
        if len(vb_s) > 0: # 如果过滤后还有有效的模型分支
            fused_b, fused_s, fused_l = weighted_boxes_fusion(
                vb_s, vs_s, vl_s, 
                weights=vw_s, iou_thr=iou_thr_wbf, skip_box_thr=skip_box_thr
            )
            final_img_preds.extend(denormalize_boxes(fused_b, fused_s, fused_l, "WBF", img_w, img_h))

        # --- 3. 大尺度缺陷分支：使用 Soft-NMS ---
        b_l, s_l, l_l = format_for_ensemble(large_col3, large_row3, img_w, img_h)
        # 【修改点】过滤掉没有预测出大缺陷的模型分支
        vb_l, vs_l, vl_l, vw_l = filter_empty_predictions(b_l, s_l, l_l, [1, 1])
        
        if len(vb_l) > 0: # 如果过滤后还有有效的模型分支
            fused_b, fused_s, fused_l = soft_nms(
                vb_l, vs_l, vl_l, 
                weights=vw_l, iou_thr=iou_thr_soft, sigma=sigma_soft, thresh=skip_box_thr
            )
            final_img_preds.extend(denormalize_boxes(fused_b, fused_s, fused_l, "SoftNMS", img_w, img_h))
            
        # --- 4. 汇总该图像的最终预测 ---
        final_img_preds.sort(key=lambda x: x['confidence'], reverse=True)
        fused_results[img_name] = final_img_preds

    # 保存最终 JSON
    print(f"融合完成！正在保存至 {out_path} ...")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fused_results, f, indent=2, ensure_ascii=False)
    print("处理完毕！")

if __name__ == "__main__":
    
    # 核心配置文件，方便后续修改或遍历调参
    config = {
        'IMG_W': 300.0,             # 图像的宽
        'IMG_H': 300.0,             # 图像的高
        'SIGMA_SOFT': 0.05,         # Soft-NMS 的高斯衰减方差
        'SKIP_BOX_THR': 0.15,       # 过滤掉sft-nms降分之后置信度极低的废框          
        'AREA_TH': 0.0004,          # 尺度感知面积阈值 (th): (6*6像素作为界限)
        'IOU_THR_WBF': 0.45,        # WBF 的聚类 IoU 阈值
        'IOU_THR_SOFT': 0.45,       # Soft-NMS 的重叠阈值
    }

    # 配置你的文件路径
    json_col3 = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/col3.json"
    json_row3 = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/row3.json"
    output_json = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/decision_fusion.json"
    
    # 执行尺度感知混合融合
    process_scale_aware_fusion(json_col3, json_row3, output_json, config)