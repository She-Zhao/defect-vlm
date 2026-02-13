"""
将图像转换为调用LLM API的格式

输入包含图像的文件夹和提示词（建议从pe.json文件中提取），自动构造json格式数据.

author:zhaoshe
"""
import json
from pathlib import Path
from typing import Generator, Any
from tqdm import tqdm

def load_prompt_text(pe_json_path: Path, idx: int) -> str:
    """加载 Prompt 文本"""
    # 1. 读取文件内容为字符串
    content = pe_json_path.read_text(encoding='utf-8')
    
    # 2. 【修正点】使用 json.loads 解析字符串，而不是 json.load
    data = json.loads(content)
    
    key = f"prompt{idx}"
    if key not in data:
        raise KeyError(f"在 {pe_json_path} 中未找到 key: {key}")
        
    return data[key]['prompt_text']

def generate_sft_entries(
    source_data: list[dict],
    prompt_template: str,
    root_dir: Path
) -> Generator[dict[str, Any], None, None]:
    """
    生成器函数：只负责产出数据字典，不负责写文件。
    这样代码解耦，方便测试，也方便以后改成写入数据库或其他格式。
    """
    for idx, item in enumerate(source_data):
        try:
            # 防御性处理路径：去掉可能存在的开头的 '/'，确保路径拼接正确
            rel_global_path = item['composite_global_path'].lstrip('/')
            rel_local_path = item['composite_local_path'].lstrip('/')
            
            # 使用 Path 拼接，最后统一转 str
            global_path = (root_dir / rel_global_path).as_posix()
            local_path = (root_dir / rel_local_path).as_posix()
            
            # 构造 ID
            api_id = item['id']
            
            # import pdb; pdb.set_trace()
            # 构造包含先验信息的提示词
            prior_label = item['prior_label']
            prompt_text = prompt_template.replace('{}', prior_label)
            
            # 构造 Entry
            yield {
                'id': api_id,
                'image': [global_path, local_path],
                'conversation': [
                    {'from': 'human', 'value': prompt_text},
                    {'from': 'assistant', 'value': ''}
                ],
                'meta_info': {
                    "label": item['label'],
                    "prior_label": item['prior_label'],
                    "sample_type": item['sample_type'],
                    "bbox": item['bbox'],
                }
            }
        except Exception as e:
            # 在生成器内部打印错误，不中断整个流，但让调用者知道
            print(f"Skipping index {idx}: {e}")

def main(base_dir: Path, config: dict):
    print(f"Loading data from {config['input_json']}...")
    raw_data = json.loads(config['input_json'].read_text(encoding='utf-8'))
    
    prompt_text = load_prompt_text(Path('./pe/prompts.json'), config['prompt_idx'])
    
    # 确保输出目录存在
    config['output_file'].parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {config['output_file']}...")
    with open(config['output_file'], 'w', encoding='utf-8') as f_out:
        # 使用 tqdm 包装生成器，显示进度
        iterator = generate_sft_entries(
            source_data=raw_data,
            prompt_template=prompt_text,
            root_dir=base_dir
        )
        
        for entry in tqdm(iterator, total=len(raw_data), desc="Building SFT Data"):
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 使用 Path 对象，操作系统无关
    base_dir = Path('/data/ZS/defect_dataset')
    config = {
        "input_json": base_dir / '3_composite_images/stripe_phase012_gt_rectification/labels/val.json',
        "output_file": base_dir / '4_api_request/val/stripe_phase012_gt_rectification.jsonl',
        "prompt_idx": 1,
    }   
    main(base_dir, config)
