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
    prompt_text: str,
    root_dir: Path,
    prefix: str,
    meta_defaults: dict[str, str]
) -> Generator[dict[str, Any], None, None]:
    """
    生成器函数：只负责产出数据字典，不负责写文件。
    这样代码解耦，方便测试，也方便以后改成写入数据库或其他格式。
    """
    for idx, item in enumerate(source_data):
        try:
            # 防御性处理路径：去掉可能存在的开头的 '/'，确保路径拼接正确
            rel_original = item['original_image_path'].lstrip('/')
            rel_crop = item['crop_image_path'].lstrip('/')
            
            # 使用 Path 拼接，最后统一转 str
            original_path = (root_dir / rel_original).as_posix()
            crop_path = (root_dir / rel_crop).as_posix()
            
            # 构造 ID
            sft_id = f"{prefix}_{idx:06d}"
            
            # 构造 Entry
            yield {
                'id': sft_id,
                'image': [original_path, crop_path],
                'conversation': [
                    {'from': 'human', 'value': prompt_text},
                    {'from': 'assistant', 'value': ''}
                ],
                'meta_info': {
                    "label": item.get('label'), # 使用 .get 防止缺少字段报错
                    "light_source": item.get('light_source'),
                    **meta_defaults # 优雅地合并字典
                }
            }
        except Exception as e:
            # 在生成器内部打印错误，不中断整个流，但让调用者知道
            print(f"Skipping index {idx}: {e}")

def main():
    # --- 配置区域 ---
    # 使用 Path 对象，操作系统无关
    base_dir = Path('/data/ZS/defect_dataset')
    
    config = {
        "image_json": base_dir / '1_defect_dataset_bbox/paint_ap_gt_negative/labels/val.json',
        "prompt_json": Path('./pe/prompts.json'),
        "output_file": base_dir / '2_sft_dataset/paint_ap_gt_negative_val.jsonl',
        "prompt_idx": 0,
        "prefix": 'gt_neg',
        "meta_defaults": {
            "source": 'gt',
            "sample_type": 'negative' # 根据输入文件手动修改这里
        }
    }

    # --- 执行逻辑 ---
    print(f"Loading data from {config['image_json']}...")
    raw_data = json.loads(config['image_json'].read_text(encoding='utf-8'))
    
    prompt_text = load_prompt_text(config['prompt_json'], config['prompt_idx'])
    
    # 确保输出目录存在
    config['output_file'].parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {config['output_file']}...")
    with open(config['output_file'], 'w', encoding='utf-8') as f_out:
        # 使用 tqdm 包装生成器，显示进度
        iterator = generate_sft_entries(
            source_data=raw_data,
            prompt_text=prompt_text,
            root_dir=base_dir,
            prefix=config['prefix'],
            meta_defaults=config['meta_defaults']
        )
        
        for entry in tqdm(iterator, total=len(raw_data), desc="Building SFT Data"):
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
