"""
将 `/data/ZS/flywheel_dataset/4_composite_yolo_preds/iter0/labels/sp012_0p1.json` 里面的数据转化为swift推理需要的格式
输入：
上一步生成的 2x2 拼图 JSON 结果：/data/ZS/flywheel_dataset/4_composite_yolo_preds/iter0/labels/sp012_0p1.json
提示词模板文件：/val_0p1.json/data/ZS/defect-vlm/defect_vlm/pe/prompts.json

输出：可以用ms swift直接进行推理的数据 12_vlm_message/stripe_phase012/val_0p1.jsonl
"""
import json
from pathlib import Path
from typing import Generator, Any
from tqdm import tqdm

def load_prompt_text(prompt_json_path: Path, prompt_idx: int) -> str:
    """加载提示词模板"""
    if not prompt_json_path.exists():
        raise FileNotFoundError(f"找不到 Prompt 文件: {prompt_json_path}")
        
    content = prompt_json_path.read_text(encoding='utf-8')
    data = json.loads(content)
    
    key = f"prompt{prompt_idx}"
    if key not in data:
        raise KeyError(f"在 {prompt_json_path} 中未找到 key: {key}")
        
    return data[key]['prompt_text']

def generate_message_entries(
    source_data: list[dict],
    prompt_template: str,
    data_root_dir: Path,
    dataset_name: str,
    split_name: str
) -> Generator[dict[str, Any], None, None]:
    
    for idx, item in enumerate(source_data):
        try:
            rel_global_path = item['composite_global_path'].lstrip('/')
            rel_local_path = item['composite_local_path'].lstrip('/')
            
            global_abs_path = (data_root_dir / rel_global_path).as_posix()
            local_abs_path = (data_root_dir / rel_local_path).as_posix()
            
            origin_id = item['id']
            message_id = f"sp012_gt_pred_{origin_id}"
            
            prior_label = item['prior_label']
            
            # 【核心修正】直接使用 replace 替换所有的 {} 即可，不要再加 <image> 了！
            # 因为 prompts.json 里已经写了 <image>\n<image>\n
            final_content = prompt_template.replace('{}', prior_label)
            
            yield {
                "id": message_id,
                "images": [global_abs_path, local_abs_path],
                "messages": [
                    {
                        "role": "user", 
                        "content": final_content
                    }
                ],
                "meta_info": {
                    "bbox": item['bbox'],
                    "prior_label": prior_label,
                    "confidence": item['confidence'],
                    "model_source": item['model_source'],
                    "origin_id": origin_id,
                }
            }
        except Exception as e:
            print(f"⚠️ Skipping index {idx} due to error: {e}")

def main(data_root: Path, input_json: Path, output_jsonl: Path, prompt_json: Path, prompt_idx: int):
    print(f"📂 正在加载数据: {input_json}")
    raw_data = json.loads(input_json.read_text(encoding='utf-8'))
    
    print(f"📝 正在加载 Prompt 模板 (Index: {prompt_idx})...")
    prompt_text = load_prompt_text(prompt_json, prompt_idx)
    
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = input_json.parent.parent.name 
    split_name = input_json.stem                 

    print(f"🚀 正在写入大模型输入文件: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        iterator = generate_message_entries(
            source_data=raw_data,
            prompt_template=prompt_text,
            data_root_dir=data_root,
            dataset_name=dataset_name,
            split_name=split_name
        )
        for entry in tqdm(iterator, total=len(raw_data), desc="构建 VLM Message"):
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ 转换成功！准备启动 VLM 推理！")

if __name__ == "__main__":
    # Prompt 模板所在的 JSON 路径 
    PROMPT_JSON = Path('/data/ZS/defect-vlm/defect_vlm/pe/prompts.json')
    PROMPT_IDX = 1 

    # 上一步生成的 2x2 拼图 JSON 结果
    INPUT_JSON = Path('/data/ZS/flywheel_dataset/4_composite_yolo_preds/iter0/labels/sp123_0p1.json')
    
    # 最终输出给 MS Swift 推理用的 JSONL 路径
    OUTPUT_JSONL = Path('/data/ZS/flywheel_dataset/5_vlm_message/iter0/sp123_0p1.jsonl')
    
    main(
        data_root=Path('/data/ZS/defect_dataset'), 
        input_json=INPUT_JSON, 
        output_jsonl=OUTPUT_JSONL, 
        prompt_json=PROMPT_JSON, 
        prompt_idx=PROMPT_IDX
    )

