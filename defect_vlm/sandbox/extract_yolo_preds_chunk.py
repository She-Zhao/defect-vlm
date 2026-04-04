"""
从全量 YOLO 预测结果中，提取指定 Chunk （预计只有chunk1）的检测框数据。
"""
import json
import time
from pathlib import Path

def extract_preds_for_chunk(pred_json_path, chunk_json_path, target_chunk, output_json_path):
    pred_file = Path(pred_json_path)
    chunk_file = Path(chunk_json_path)
    output_file = Path(output_json_path)

    # 1. 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📖 正在加载分块名单: {chunk_file.name}")
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    if target_chunk not in chunks_data:
        raise ValueError(f"❌ 找不到指定的 {target_chunk}，请检查分块 JSON 文件！")

    # 使用 set 来存储目标文件名，查询时间复杂度为 O(1)，极其快速
    target_filenames = set(chunks_data[target_chunk])
    print(f"🎯 目标 {target_chunk} 包含图像数量: {len(target_filenames)}")

    print(f"📖 正在加载全量预测结果: {pred_file.name}")
    start_time = time.time()
    with open(pred_file, 'r', encoding='utf-8') as f:
        all_preds = json.load(f)
    print(f"📦 全量预测文件中共有 {len(all_preds)} 张图像的检测记录。")

    # 2. 核心过滤逻辑
    filtered_preds = {}
    match_count = 0
    box_count = 0

    print("⚙️ 正在执行高效匹配过滤...")
    for img_name, bboxes in all_preds.items():
        if img_name in target_filenames:
            filtered_preds[img_name] = bboxes
            match_count += 1
            box_count += len(bboxes)

    # 3. 保存过滤后的结果
    print(f"💾 正在保存提取结果至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_preds, f, ensure_ascii=False, indent=2)

    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"🎉 提取完成! 耗时: {elapsed_time:.2f} 秒")
    print(f"✅ 成功匹配到 {match_count} 张图像，共提取出 {box_count} 个预测框。")
    print(f"   (未在预测文件中找到的 Chunk 图像数量: {len(target_filenames) - match_count})")
    print("="*50)


if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. Iter 0 的全量预测结果
    PRED_JSON = "/data/ZS/flywheel_dataset/2_yolo_preds/iter0/sp012_decision_fusion_0p1.json"
    
    # 2. 切分好的 Chunk 名单
    CHUNK_JSON = "/data/ZS/flywheel_dataset/chunks/chunk_splits.json"      
    
    # 3. 本次要提取的目标 Chunk(应该也只有chunk1)
    TARGET_CHUNK = "chunk1"
    
    # 4. 提取后的输出路径
    OUTPUT_JSON = f"/data/ZS/flywheel_dataset/2_yolo_preds/iter1/sp012_decision_fusion_0p1_{TARGET_CHUNK}.json"
    # ==========================================

    extract_preds_for_chunk(
        pred_json_path=PRED_JSON,
        chunk_json_path=CHUNK_JSON,
        target_chunk=TARGET_CHUNK,
        output_json_path=OUTPUT_JSON
    )