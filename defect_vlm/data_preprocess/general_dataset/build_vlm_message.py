"""
将拼接好的 2x2 图像 JSON 数据转化为 ms-swift 推理/微调需要的格式
支持透传 GT Label 以及各类难例抖动属性 (IoU, neg_type 等)
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
            # 动态生成 message_id，避免写死前缀
            message_id = f"{dataset_name}_{split_name}_{origin_id}"
            
            prior_label = item['prior_label']
            
            # 【核心修正】直接使用 replace 替换所有的 {} 即可，不要再加 <image> 了！
            # 因为 prompts.json 里已经写了 <image>\n<image>\n
            final_content = prompt_template.replace('{}', prior_label)
            
            # ================= 动态构建 meta_info =================
            meta_info = {
                "bbox": item['bbox'],
                "prior_label": prior_label,
                "label": item.get('label', 'unknown'),             # <--- 新增的真实标签
                "sample_type": item.get('sample_type', 'unknown'), # 样本类型 (positive/negative/rectification)
                "origin_id": origin_id,
            }
            
            # 动态透传其他所有可能有用的评测字段（如果有的话）
            optional_keys = ['confidence', 'model_source', 'overlap_iou', 'neg_type', 'jitter_iou']
            for opt_key in optional_keys:
                if opt_key in item:
                    meta_info[opt_key] = item[opt_key]
            # ======================================================

            yield {
                "id": message_id,
                "images": [global_abs_path, local_abs_path],
                "messages": [
                    {
                        "role": "user", 
                        "content": final_content
                    }
                ],
                "meta_info": meta_info
            }
        except Exception as e:
            print(f"⚠️ Skipping index {idx} due to error: {e}")

def main(data_root: Path, input_json: Path, output_jsonl: Path, prompt_json: Path, prompt_idx: int):
    print(f"📂 正在加载数据: {input_json}")
    raw_data = json.loads(input_json.read_text(encoding='utf-8'))
    
    print(f"📝 正在加载 Prompt 模板 (Index: {prompt_idx})...")
    prompt_text = load_prompt_text(prompt_json, prompt_idx)
    
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # 自动提取项目名称和 split (例如: sp012_gt_negative 和 val)
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
            
    print(f"✅ 转换成功！准备启动 VLM 推理/微调！")

if __name__ == "__main__":
    # ================= 配置区 =================
    # 1. 提示词模板路径
    PROMPT_JSON = Path('/data/ZS/defect-vlm/defect_vlm/pe/prompts.json')
    PROMPT_IDX = 1 

    # 2. 数据集根目录 (用于拼接出绝对路径)
    DATA_ROOT = Path('/data/ZS/defect_dataset')

    # 3. 输入：上一步拼接生成的 2x2 拼图 JSON 结果
    INPUT_JSON = Path('/data/ZS/defect_dataset/3_composite_images_general/sp123_gt_negative/labels/val.json')
    
    # 4. 输出：给 MS Swift 推理/评测用的 JSONL 路径
    OUTPUT_JSONL = Path('/data/ZS/defect_dataset/7_swift_dataset_general/val/sp123_gt_negative.jsonl')
    
    main(
        data_root=DATA_ROOT, 
        input_json=INPUT_JSON, 
        output_jsonl=OUTPUT_JSONL, 
        prompt_json=PROMPT_JSON, 
        prompt_idx=PROMPT_IDX
    )