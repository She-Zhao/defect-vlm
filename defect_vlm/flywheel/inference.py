"""
个人手写的推理，卡阈值做数据飞轮的时候使用，计算指标的时候不要用。
计算指标请使用官方的val.py去实现，对应inference_official.py
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path
import argparse

# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics，而不是系统官方包
# 假设你的项目根目录是 /data/ZS/v11_input
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 保证最高优先级
# =====================================================================

import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def run_multistream_inference(model_path, input_dir, output_json, conf_thres=0.001, nms_iou=0.6):
    """
    多流 YOLO 模型推理脚本，并将结果保存为 JSON
    
    参数:
        model_path: best.pt 的绝对路径
        input_dir: 输入的根目录 (例如: '/data/ZS/v11_input/datasets/col3')
        output_json: 保存的 json 文件路径
        conf_thres: 置信度阈值。级联融合前建议设低一点(如0.1)，把决策权交给后续的 NMS
    """
    input_dir = Path(input_dir)
    model_source = input_dir.name  # 提取模型来源名称，比如 'col3' 或 'row3'
    
    print(f"🚀 初始化推理任务: 模型来源 [{model_source}]")
    print(f"📦 加载权重: {model_path}")
    
    # 1. 加载模型
    model = YOLO(model_path)
    
    # 获取类别映射字典 (例如 {0: 'breakage', 1: 'inclusion', ...})
    names_dict = model.names 
    
    # 2. 探测有多少个输入流 (val, val_1, val_2 ...)
    # 寻找所有以 val 开头的文件夹
    val_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith('val') and (input_dir / d).is_dir()])
    print(f"🔍 探测到 {len(val_dirs)} 个多流数据文件夹: {val_dirs}")
    
    # 3. 获取所有验证集图片文件名 (以基础的 val 文件夹为准)
    base_images_dir = input_dir / "val" / "images"
    image_filenames = [f for f in os.listdir(base_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📸 共找到 {len(image_filenames)} 张待推理图片。")
    
    # 结果字典
    results_dict = {}
    
    # 4. 遍历所有图片进行推理
    for filename in tqdm(image_filenames, desc="推理进度"):
        ims_list = []
        
        # 依次读取各个视角的光照图像
        for val_dir in val_dirs:
            img_path = input_dir / val_dir / "images" / filename
            if not img_path.exists():
                print(f"\n⚠️ 警告: 缺失图像分支 {img_path}")
                continue
                
            # 读取灰度图 (保持与你之前 base.py 中 load_image 的逻辑一致)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ims_list.append(img)
        
        if not ims_list:
            results_dict[filename] = []
            continue
            
        # 沿着通道维度拼接，形成 (H, W, N) 的多流 Numpy 数组
        # 如果是 6 个视角，则变成 (300, 300, 6)
        stacked_img = np.dstack(ims_list)
        # stacked_img = stacked_img[..., ::-1]
        
        # 5. 执行推理
        # 直接传入 Numpy 数组，YOLO 内部会调用我们之前修复好的 LetterBox 补边，再转 Tensor
        # 这里关闭 verbose 防止进度条被刷屏，设置 imgsz=300 保持与训练一致
        preds = model.predict(
            source=stacked_img,
            imgsz=300,
            conf=conf_thres,
            iou=nms_iou,
            max_det=3000,
            verbose=False,
            augment=False
        )
        
        # 6. 解析结果
        img_results = []
        for r in preds:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            # 获取坐标、置信度、类别
            xyxys = boxes.xyxy.cpu().numpy()  # [N, 4] -> [x_min, y_min, x_max, y_max]
            confs = boxes.conf.cpu().numpy()  # [N]
            clss = boxes.cls.cpu().numpy()    # [N]
            
            for box, conf, cls_id in zip(xyxys, confs, clss):
                img_results.append({
                    "class_id": int(cls_id),
                    "class_name": names_dict[int(cls_id)],
                    "bbox": [float(val) for val in box],  # 转换为普通 float 以便 json 序列化
                    "confidence": float(conf),
                    "model_source": model_source
                })
                
        # 存入结果字典，即使为空也要保存一个 []
        results_dict[filename] = img_results
        
    # 7. 导出为 JSON 文件
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
    print(f"✅ 推理完成！结果已成功保存至: {output_path}")

def main():
    """利用 argparse 进行命令行传参"""
    parser = argparse.ArgumentParser(description="多流 YOLO 推理脚本")
    parser.add_argument('--model_path', type=str, required=True, help='模型权重')
    parser.add_argument('--input_dir', type=str, required=True, help='待推理数据集所在文件夹')
    parser.add_argument('--output_json', type=str, required=True, help='输出json路径')
    parser.add_argument('--conf_thres', type=float, required=True, help='置信度阈值')
    parser.add_argument('--nms_iou', type=float, required=True, default=0.6, help='NMS的默认IoU阈值')
    args = parser.parse_args()

    run_multistream_inference(
        model_path = args.model_path,       
        input_dir = args.input_dir,
        output_json = args.output_json,
        conf_thres = args.conf_thres,            # 设低一点，让下一步的 NMS 去做决策
        nms_iou = args.nms_iou
    )
    
if __name__ == '__main__':
    # main()
    
    # 推理无标注的数据
    run_multistream_inference(
        model_path="/data/ZS/defect-vlm/output/yolo_weights/iter1_col3_0p1_ema0p01.pt",  # 填入你昨晚训练出来的 col3 权重
        input_dir="/data/ZS/flywheel_dataset/0_multi_input/iter3/col3",
        output_json="/data/ZS/flywheel_dataset/2_yolo_preds/iter3_weight_iter1ema/col3_0p1_chunk123.json",
        conf_thres=0.05            # 故意设低一点，让下一步的 NMS 去做决策
    )

    # 推理 GT 的验证集
    # run_multistream_inference(
    #     model_path="/data/ZS/defect-vlm/output/yolo_weights/gt_col3_part_cbam_max.pt",  # 填入你昨晚训练出来的 col3 权重
    #     input_dir="/data/ZS/v11_input/datasets/col3",
    #     output_json="/data/ZS/flywheel_dataset/10_preds_on_gt_val/gt/col3.json",
    #     conf_thres=0.05            # 故意设低一点，让下一步的 NMS 去做决策
    # )

    