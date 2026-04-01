import json
import argparse
from pathlib import Path

def convert_and_filter_results(yolo_json_path, output_json_path, model_source, conf_thres=0.0):
    """
    将 YOLO 官方输出的 COCO 格式 JSON 转换为自定义融合评估格式，并支持阈值过滤。
    """
    yolo_json_path = Path(yolo_json_path)
    output_json_path = Path(output_json_path)
    
    if not yolo_json_path.exists():
        raise FileNotFoundError(f"❌ 找不到官方预测文件: {yolo_json_path}")
        
    print(f"📥 正在读取官方预测结果: {yolo_json_path}")
    with open(yolo_json_path, 'r', encoding='utf-8') as f:
        yolo_preds = json.load(f)
        
    # YOLO 类别映射字典 (根据你数据集的 yaml 配置)
    names_dict = {
        0: 'breakage', 
        1: 'inclusion', 
        2: 'scratch', 
        3: 'crater', 
        4: 'run', 
        5: 'bulge'
    }
    
    results_dict = {}
    filtered_count = 0
    total_count = len(yolo_preds)
    
    for p in yolo_preds:
        # 1. 阈值过滤 (拦截低分垃圾框)
        if p['score'] < conf_thres:
            filtered_count += 1
            continue
            
        # 2. 文件名还原 (YOLO 的 image_id 可能是去掉了后缀的 stem)
        img_id = str(p['image_id'])
        img_name = img_id if img_id.endswith('.png') else f"{img_id}.png"
        
        if img_name not in results_dict:
            results_dict[img_name] = []
            
        # 3. 坐标转换: COCO 的 [x_min, y_min, width, height] -> Custom 的 [x_min, y_min, x_max, y_max]
        x_min, y_min, w, h = p['bbox']
        x_max = x_min + w
        y_max = y_min + h
        
        # 4. 组装数据
        results_dict[img_name].append({
            "class_id": p['category_id'],
            "class_name": names_dict[p['category_id']],
            "bbox": [x_min, y_min, x_max, y_max], 
            "confidence": p['score'],
            "model_source": model_source
        })
        
    # 5. 保存结果
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print(f"✅ 转换与清洗完成！")
    print(f"📊 原始检测框总数: {total_count}")
    print(f"✂️  阈值 {conf_thres} 过滤掉的框: {filtered_count}")
    print(f"📦 最终保留的有效框: {total_count - filtered_count}")
    print(f"💾 结果已保存至: {output_json_path}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="YOLO 官方 JSON 转自定义融合格式清洗脚本")
    parser.add_argument('--input_json', type=str, required=True, help='官方预测生成的 predictions.json 路径')
    parser.add_argument('--output_json', type=str, required=True, help='输出的自定义格式 json 路径')
    parser.add_argument('--source', type=str, required=True, help='模型来源标记，例如 row3 或 col3')
    parser.add_argument('--conf_thres', type=float, default=0.0, help='置信度过滤阈值，低于该值的框将被丢弃 (默认 0.0 保留所有)')
    
    args = parser.parse_args()
    
    convert_and_filter_results(
        yolo_json_path=args.input_json,
        output_json_path=args.output_json,
        model_source=args.source,
        conf_thres=args.conf_thres
    )

if __name__ == '__main__':
    # main()
    input_json = '/data/ZS/v11_input/runs/detect/val10/predictions.json'
    output_json = '/data/ZS/defect_dataset/9_yolo_preds/val_official/row3_new.json'
    # output_json = '/data/ZS/flywheel_dataset/10_semi_yolo_preds/iter0/row3.json'
    source = 'row3'
    conf_thres = 0
    
    convert_and_filter_results(
        yolo_json_path = input_json,
        output_json_path = output_json,
        model_source = source,
        conf_thres = conf_thres
    )