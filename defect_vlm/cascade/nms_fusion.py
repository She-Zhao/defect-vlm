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

def fusion_nms(json_col3, json_row3, output_json, iou_thres=0.45, conf_thres=0.0):
    """
    对两个模型的预测结果进行 NMS 融合，并添加置信度过滤
    :param json_col3: 模型1的预测结果
    :param json_row3: 模型2的预测结果
    :param output_json: 融合后的输出路径
    :param iou_thres: NMS 的 IoU 阈值
    :param conf_thres: 置信度过滤阈值，默认 0.0 (不过滤)
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
    print(f"⚙️ 当前配置: 置信度阈值(Conf) >= {conf_thres}, NMS重合度阈值(IoU) = {iou_thres}")
    
    fusion_data = {}
    total_original_boxes = 0       # 最原始的框总数
    total_after_conf_filter = 0    # 经过置信度过滤后的框总数
    total_fused_boxes = 0          # 经过 NMS 后的最终保留框总数
    
    # 2. 逐图进行处理
    for img_name in all_images:
        preds_col3 = col3_data.get(img_name, [])
        preds_row3 = row3_data.get(img_name, [])
        
        # 原始预测框拼接
        raw_combined = preds_col3 + preds_row3
        total_original_boxes += len(raw_combined)
        
        # 2.1 置信度过滤 (Confidence Filtering)
        combined_preds = [p for p in raw_combined if p['confidence'] >= conf_thres]
        total_after_conf_filter += len(combined_preds)
        
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
    print("✅ 过滤与 NMS 融合完成！")
    print(f"📊 统计信息:")
    print(f"   - 初始预测框总数: {total_original_boxes}")
    print(f"   - 置信度过滤剔除: {total_original_boxes - total_after_conf_filter} 个 (剩余 {total_after_conf_filter})")
    print(f"   - NMS 冗余剔除  : {total_after_conf_filter - total_fused_boxes} 个")
    print(f"   - 最终保留框总数: {total_fused_boxes}")
    print(f"📁 结果已保存至: {output_path}")

if __name__ == '__main__':
    # 配置文件路径
    COL3_JSON = "/data/ZS/defect_dataset/9_yolo_preds/自己手写的推理脚本/val_0p001/col3.json"
    ROW3_JSON = "/data/ZS/defect_dataset/9_yolo_preds/自己手写的推理脚本/val_0p001/row3.json"
    FUSION_JSON = "/data/ZS/defect_dataset/9_yolo_preds/自己手写的推理脚本/val_0p001/nms_fusion_conf_0p25.json"
    
    # 超参数配置
    CONF_THRESHOLD = 0.25  # 在这里设定卡的置信度阈值
    IOU_THRESHOLD = 0.45   # NMS 的 IoU 阈值
    
    fusion_nms(COL3_JSON, ROW3_JSON, FUSION_JSON, iou_thres=IOU_THRESHOLD, conf_thres=CONF_THRESHOLD)