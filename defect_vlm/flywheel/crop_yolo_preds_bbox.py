"""
根据双流主干网络的推理结果，从伪rgb文件夹里面裁剪出bbox并保存，同步保存json文件
输入：融合后的 JSON 文件和伪 RGB 图像所在目录
输出：抠取图像的保存目录和元数据 JSON 的保存路径
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_dynamic_context_ratio(bbox_size: float) -> float:
    """根据bbox的尺寸动态计算context_ratio (保持与训练集完全对齐)"""
    sizes = [20, 100]
    ratios = [2, 0.5]
    return float(np.interp(bbox_size, sizes, ratios))

def get_region_proposal(image: np.ndarray, bbox: list, bbox_format='xyxy', fixed_context_ratio: float = None) -> np.ndarray:
    """根据bbox截取图像局部"""
    image_h, image_w = image.shape[:2]
    
    if bbox_format == 'coco':
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
    elif bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
    else:
        raise ValueError(f"不支持的格式: {bbox_format}")
    
    if fixed_context_ratio is not None:
        h_context_ratio = w_context_ratio = fixed_context_ratio
    else:
        h_context_ratio = get_dynamic_context_ratio(h)
        w_context_ratio = get_dynamic_context_ratio(w)

    new_w = w * (1 + w_context_ratio)
    new_h = h * (1 + h_context_ratio)
    
    x_min = int(max(0, round(cx - new_w / 2)))
    y_min = int(max(0, round(cy - new_h / 2)))
    x_max = int(min(image_w, round(cx + new_w / 2)))
    y_max = int(min(image_h, round(cy + new_h / 2)))
      
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return image[y_min:y_max, x_min:x_max].copy()

def main(fusion_json_path, rgb_image_root, save_img_dir, out_json_path, data_root):
    # 将传入的字符串路径转为 Path 对象，方便代码内部处理
    fusion_json_path = Path(fusion_json_path)
    rgb_image_root = Path(rgb_image_root)
    save_img_dir = Path(save_img_dir)
    out_json_path = Path(out_json_path)
    data_root = Path(data_root)
    
    # 自动创建输出文件夹
    save_img_dir.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    light_sources = ['16col', '16row', '32col', '32row']

    # ================= 加载数据 =================
    print(f"📖 正在加载融合预测文件: {fusion_json_path}")
    with open(fusion_json_path, 'r', encoding='utf-8') as f:
        fusion_preds = json.load(f)

    if 'config' in fusion_preds:
        print(f"决策融合的配置参数:{fusion_preds['config']}")
        del fusion_preds['config']

    metadata_list = []
    global_id = 0
    missing_images = 0

    # ================= 开始处理 =================
    for img_name, bboxes in tqdm(fusion_preds.items(), desc="裁剪预测框"):
        if not bboxes:
            continue  # 跳过没有预测框的图像
            
        file_stem = Path(img_name).stem
        
        # 遍历4个光源
        for light in light_sources:
            original_img_path = rgb_image_root / light / img_name
            
            if not original_img_path.exists():
                print(original_img_path)
                missing_images += 1
                continue
                
            raw_image = cv2.imread(str(original_img_path))
            if raw_image is None:
                continue
                
            # 遍历该图像上的所有预测框
            for box_idx, pred in enumerate(bboxes):
                # fusion.json 里的 bbox 格式是 [x_min, y_min, x_max, y_max]
                bbox_xyxy = pred['bbox']
                prior_label = pred['class_name']
                confidence = pred['confidence']
                model_source = pred['model_source']
                
                # 执行裁剪 (必须指定 xyxy 格式)
                crop = get_region_proposal(raw_image, bbox_xyxy, bbox_format='xyxy')
                
                if crop is None or crop.size == 0:
                    continue
                    
                # 构造保存文件名: 原文件名_光源_先验类别_序号.png
                save_name = f"{file_stem}_{light}_{prior_label}_{box_idx}.png"
                save_path = save_img_dir / save_name
                
                cv2.imwrite(str(save_path), crop)
                
                # ================= 记录元数据 =================
                # 尝试计算相对路径，如果跨盘符失败则直接保留绝对路径
                try:
                    rel_original_path = original_img_path.relative_to(data_root)
                    rel_crop_path = save_path.relative_to(data_root)
                except ValueError:
                    rel_original_path = original_img_path
                    rel_crop_path = save_path
                
                # 将 xyxy 转换为 coco 的 xywh 格式保存
                x1, y1, x2, y2 = bbox_xyxy
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]
                
                metadata = {
                    "id": global_id,
                    "original_image_path": str(rel_original_path),
                    "crop_image_path": str(rel_crop_path),
                    "bbox": bbox_coco,  # 转成了 [x, y, w, h] 格式
                    "prior_label": prior_label,
                    "confidence": confidence,
                    "model_source": model_source,
                    "light_source": light
                }
                metadata_list.append(metadata)
                global_id += 1
                
    # ================= 保存输出 JSON =================
    print(f"\n💾 裁剪完成！共提取了 {global_id} 个局部特征图。")
    if missing_images > 0:
        print(f"⚠️ 警告: 有 {missing_images} 次原图读取失败，请检查路径。")
        
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"📄 元数据已保存至: {out_json_path}")

if __name__ == "__main__":
    # 1. 输入：融合后的 JSON 文件和伪 RGB 图像所在目录
    FUSION_JSON_PATH = "/data/ZS/flywheel_dataset/2_yolo_preds/iter2/decision_fusion_0p1_chunk12.json"
    RGB_IMAGE_ROOT   = "/data/ZS/flywheel_dataset/1_paint_rgb/stripe_phase012/images"     # 不用动
    
    # 2. 输出：抠取图像的保存目录和元数据 JSON 的保存路径
    SAVE_IMG_DIR  = "/data/ZS/flywheel_dataset/3_yolo_preds_bbox/iter2/images/0p1_chunk12"
    OUT_JSON_PATH = "/data/ZS/flywheel_dataset/3_yolo_preds_bbox/iter2/labels/0p1_chunk12.json"
    
    # 3. 根目录锚点 (仅用于在 JSON 中精简路径，比如变成 '1_paint_rgb/...')
    DATA_ROOT = "/data/ZS/flywheel_dataset"

    main(
        fusion_json_path=FUSION_JSON_PATH,
        rgb_image_root=RGB_IMAGE_ROOT,
        save_img_dir=SAVE_IMG_DIR,
        out_json_path=OUT_JSON_PATH,
        data_root=DATA_ROOT
    )
