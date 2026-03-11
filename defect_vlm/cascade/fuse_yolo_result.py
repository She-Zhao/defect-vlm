"""
对使用多流主干网络推理得到的结果进行NMS,得到融合后的检测结果
输入：
/data/ZS/defect_dataset/9_yolo_preds/val/col3.json
/data/ZS/defect_dataset/9_yolo_preds/val/row3.json
输出：
/data/ZS/defect_dataset/9_yolo_preds/val/fusion.json
"""
import json
import torch
import torchvision
from pathlib import Path

def fusion_nms(json_col3, json_row3, output_json, iou_thres=0.45):
    """
    对两个模型的预测结果进行 NMS 融合
    """
    # 1. 加载两个 JSON 文件
    print(f"📂 正在加载预测结果...\n -> {json_col3}\n -> {json_row3}")
    with open(json_col3, 'r', encoding='utf-8') as f:
        col3_data = json.load(f)
    with open(json_row3, 'r', encoding='utf-8') as f:
        row3_data = json.load(f)
        
    # 获取所有的图片名（取并集，防止某个模型漏掉某些图片）
    all_images = set(col3_data.keys()).union(set(row3_data.keys()))
    print(f"🖼️ 共发现 {len(all_images)} 张图片待融合。")
    
    fusion_data = {}
    total_original_boxes = 0
    total_fused_boxes = 0
    
    # 2. 逐图进行处理
    for img_name in all_images:
        preds_col3 = col3_data.get(img_name, [])
        preds_row3 = row3_data.get(img_name, [])
        
        # 将两个模型的预测框直接拼接在同一个列表里
        combined_preds = preds_col3 + preds_row3
        total_original_boxes += len(combined_preds)
        
        if not combined_preds:
            fusion_data[img_name] = []
            continue
            
        # 3. 按类别将预测框分组 (NMS 需要按类独立进行)
        class_to_preds = {}
        for p in combined_preds:
            cid = p['class_id']
            if cid not in class_to_preds:
                class_to_preds[cid] = []
            class_to_preds[cid].append(p)
            
        fused_img_preds = []
        
        # 4. 对每个类别分别执行 NMS
        for cid, preds in class_to_preds.items():
            if len(preds) == 1:
                fused_img_preds.extend(preds)
                continue
                
            # 提取坐标和置信度，转换为 PyTorch Tensor (借助 torchvision 的高性能 nms 算子)
            boxes = torch.tensor([p['bbox'] for p in preds], dtype=torch.float32)
            scores = torch.tensor([p['confidence'] for p in preds], dtype=torch.float32)
            
            # 调用 NMS: 返回需要保留的框的索引
            keep_indices = torchvision.ops.nms(boxes, scores, iou_thres)
            
            # 把保留下来的框存入结果
            for idx in keep_indices.tolist():
                fused_img_preds.append(preds[idx])
                
        fusion_data[img_name] = fused_img_preds
        total_fused_boxes += len(fused_img_preds)

    # 5. 保存融合结果
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fusion_data, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print("✅ NMS 融合完成！")
    print(f"📊 统计信息: 融合前共有 {total_original_boxes} 个预测框，NMS 剔除了 {total_original_boxes - total_fused_boxes} 个冗余框。")
    print(f"最终保留了 {total_fused_boxes} 个框，已保存至: {output_path}")

if __name__ == '__main__':
    # 配置文件路径
    COL3_JSON = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/col3.json"
    ROW3_JSON = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/row3.json"
    FUSION_JSON = "/data/ZS/defect_dataset/9_yolo_preds/val_0p1/fusion1.json"
    
    # 融合的 IoU 阈值 (如果两个同类框的重合度大于 0.45，就剔除置信度低的那个)
    # 可以根据你的缺陷密集程度微调，通常设为 0.45 ~ 0.6
    IOU_THRESHOLD = 0.45
    
    fusion_nms(COL3_JSON, ROW3_JSON, FUSION_JSON, iou_thres=IOU_THRESHOLD)
