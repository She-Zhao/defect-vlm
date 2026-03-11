"""
对/data/ZS/defect_dataset/9_yolo_infer_result/val里面的而推理结果进行可视化
该推理结果由训练多流主干网络推理 `/data/ZS/v11_input/datasets/col3（row3）`得到的
"""
import os
import cv2
import json
import random
from pathlib import Path

def visualize_json_predictions(json_path, image_base_dir, output_dir, num_samples=5):
    """
    随机抽取 JSON 中的预测结果，画框并保存
    """
    json_path = Path(json_path)
    image_base_dir = Path(image_base_dir)
    output_dir = Path(output_dir)
    
    # 创建保存结果的文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 正在读取 JSON 文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        preds_dict = json.load(f)
        
    # 过滤掉没有检测到任何缺陷的图片（只看有框的）
    valid_images = [filename for filename, bboxes in preds_dict.items() if len(bboxes) > 0]
    
    if not valid_images:
        print("⚠️ 没找到任何带预测框的图片！")
        return
        
    # 随机抽样
    sample_size = min(num_samples, len(valid_images))
    sampled_filenames = random.sample(valid_images, sample_size)
    print(f"🎯 从 {len(valid_images)} 张有缺陷的图中，随机抽取了 {sample_size} 张进行可视化...")
    
    for filename in sampled_filenames:
        # 读取原图 (我们只取 val 这个视角的图作为底图来画框即可)
        img_path = image_base_dir / filename
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"❌ 找不到图片: {img_path}")
            continue
            
        # 遍历这张图里的所有预测框
        for box_info in preds_dict[filename]:
            x_min, y_min, x_max, y_max = [int(v) for v in box_info['bbox']]
            cls_name = box_info['class_name']
            conf = box_info['confidence']
            
            # 1. 画矩形框 (红色: BGR格式为 (0, 0, 255), 线条粗细 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # 2. 准备标签文字
            label = f"{cls_name} {conf:.2f}"
            
            # 3. 画标签背景框 (为了让文字更清晰可见)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x_min, y_min - text_h - 5), (x_min + text_w, y_min), (0, 0, 255), -1)
            
            # 4. 画文字 (白色文字)
            cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        # 保存可视化结果
        save_path = output_dir / f"vis_{filename}"
        cv2.imwrite(str(save_path), img)
        print(f"🖼️ 已保存: {save_path}")

if __name__ == '__main__':
    # ==========================
    # 请确认这里的路径
    # ==========================
    JSON_FILE = "/data/ZS/defect_dataset/9_yolo_preds/val/row3.json"
    
    # 只需要拿其中一个视角（比如基础的 val）作为画图的“底片”就行
    IMAGE_DIR = "/data/ZS/v11_input/datasets/row3/val/images"  
    
    # 画好的图存在哪
    OUTPUT_DIR = "/data/ZS/temp/yolo_preds/row3"
    
    # 执行可视化 (抽取 10 张看看)
    visualize_json_predictions(JSON_FILE, IMAGE_DIR, OUTPUT_DIR, num_samples=10)

