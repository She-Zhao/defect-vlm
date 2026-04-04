"""
利用 Bbox 作为 Hash Key，从全量 VLM JSONL 中精准提取 Chunk 的判断结果。
"""
import json
import time
from pathlib import Path

def match_vlm_by_bbox(chunk_json_path, vlm_jsonl_path, output_jsonl_path):
    chunk_file = Path(chunk_json_path)
    vlm_file = Path(vlm_jsonl_path)
    output_file = Path(output_jsonl_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. 解析 Chunk 数据，构建 bbox 的哈希集合 (Hash Set)
    print(f"📖 正在加载 Chunk 预测结果: {chunk_file.name}")
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)

    # 将 bbox 统一保留 4 位小数，存入 Set 以实现 O(1) 的极速匹配
    target_bboxes = set()
    total_chunk_boxes = 0

    for key, value in chunk_data.items():
        if key == "config":
            continue  # 跳过配置项
        for pred in value:
            # 四舍五入保留4位小数，转为 tuple 作为 key
            bbox_tuple = tuple(round(x, 4) for x in pred["bbox"])
            target_bboxes.add(bbox_tuple)
            total_chunk_boxes += 1

    print(f"🎯 Chunk 中共解析出 {total_chunk_boxes} 个预测框，去重后哈希键数量: {len(target_bboxes)}")

    # 2. 遍历 VLM 全量 JSONL 进行匹配
    print(f"📖 正在扫描全量 VLM 数据: {vlm_file.name}")
    start_time = time.time()
    
    match_count = 0
    total_vlm_lines = 0

    with open(vlm_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            total_vlm_lines += 1
            item = json.loads(line)
            
            # 提取 VLM 的 bbox，同样保留 4 位小数
            vlm_bbox_tuple = tuple(round(x, 4) for x in item["bbox"])
            
            # --- 💡 防坑机制：如果你发现匹配结果是 0，说明 VLM 存的是 [x, y, w, h] ---
            # 如果需要转换，可以取消下面三行的注释：
            x, y, w, h = item["bbox"]
            vlm_bbox_tuple = (round(x, 4), round(y, 4), round(x+w, 4), round(y+h, 4))
            # ------------------------------------------------------------------------

            if vlm_bbox_tuple in target_bboxes:
                fout.write(line)
                match_count += 1

    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"🎉 匹配提取完成! 耗时: {elapsed_time:.2f} 秒")
    print(f"📊 扫描 VLM 数据总量: {total_vlm_lines} 条")
    print(f"✅ 成功匹配并导出: {match_count} 条记录")
    print(f"💾 输出文件保存至: {output_file}")
    print("="*60)
    
    # 安全性检查警告
    if match_count == 0:
        print("⚠️ 警告：命中率为 0！请务必检查两个文件中的 bbox 格式是否一致（[x1,y1,x2,y2] vs [x,y,w,h]）。")
        print("⚠️ 建议取消代码中 '防坑机制' 处的注释再试一次。")


if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. 刚刚提取出来的 Chunk 1 的 YOLO 预测结果
    CHUNK_JSON = "/data/ZS/flywheel_dataset/2_yolo_preds/iter1/sp012_decision_fusion_0p1_chunk1.json"
    
    # 2. Iter 0 全量的 VLM 判别结果
    VLM_JSONL = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/iter0/sp012_0p1_v1_LM.jsonl"
    
    # 3. 本次过滤后要输出的 VLM Chunk 结果
    # 建议放到 iter1 的专门目录下
    OUTPUT_JSONL = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/iter1/sp012_0p1_v1_LM_chunk1.jsonl"
    # ==========================================

    match_vlm_by_bbox(
        chunk_json_path=CHUNK_JSON,
        vlm_jsonl_path=VLM_JSONL,
        output_jsonl_path=OUTPUT_JSONL
    )