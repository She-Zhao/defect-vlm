"""
提取VLM输出(JSONL)的结果，转换为只包含检测任务需要的数据的格式
输入：/data/ZS/flywheel_dataset/6_vlm_response里面vlm针对bbox推理的结果
输出："/data/ZS/flywheel_dataset/7_vlm_extracted_data提取后的关键信息，名称与输入文件一样
"""
import os
import json
from pathlib import Path

def extract_vlm_core_data(input_jsonl, output_dir):
    input_path = Path(input_jsonl)
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    
    print(f"📖 开始读取 VLM 推理文件: {input_path}")
    print("⚙️ 正在执行严格的字段提取...")
    
    extracted_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_idx, line in enumerate(fin, 1):
            if not line.strip():
                continue
            
            try:
                # 1. 读取原数据
                data = json.loads(line)
                
                # 2. 强匹配提取所需字段 (如果键不存在，直接触发 KeyError)
                vlm_id = data["id"]
                images = data["images"]
                
                meta = data["meta_info"]
                bbox = meta["bbox"]
                prior_label = meta["prior_label"]
                confidence = meta["confidence"]
                model_source = meta["model_source"]
                
                # 3. 将 pred 字符串反序列化为字典，并提取 defect
                pred_str = data["pred"]
                pred_dict = json.loads(pred_str)
                defect_result = pred_dict["defect"]
                
                # 4. 组装清洗后的纯净数据
                clean_data = {
                    "id": vlm_id,
                    "images": images,
                    "bbox": bbox,
                    "confidence": confidence,
                    "model_source": model_source,
                    "prior_label": prior_label,
                    "vlm_pred": defect_result
                }
                
                # 5. 写入新文件
                fout.write(json.dumps(clean_data, ensure_ascii=False) + '\n')
                extracted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ [格式错误] 第 {line_idx} 行 JSON 解析失败 (可能是 VLM 没按格式输出 pred): {e}")
                error_count += 1
            except KeyError as e:
                print(f"❌ [字段缺失] 第 {line_idx} 行缺少关键字段 {e}")
                error_count += 1
            except Exception as e:
                print(f"❌ [未知异常] 第 {line_idx} 行发生错误: {e}")
                error_count += 1

    # 汇总报告
    print("=" * 60)
    print("✅ VLM 数据清洗完成！")
    print(f"📊 成功清洗并保存: {extracted_count} 条")
    if error_count > 0:
        print(f"⚠️ 发现格式不规范数据: {error_count} 条 (详情见上方日志)")
    print(f"📄 精简后数据路径: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    # 配置输入输出路径
    INPUT_JSONL = "/data/ZS/flywheel_dataset/6_vlm_response/iter0/sp012_0p1_v1_LM.jsonl"
    OUTPUT_DIR = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/iter0"       # 输出文件的名称和输入名称保持一致
    
    extract_vlm_core_data(INPUT_JSONL, OUTPUT_DIR)
