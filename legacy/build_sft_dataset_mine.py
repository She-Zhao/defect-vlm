"""
将图像转换为调用LLM API的格式

输入包含图像的文件夹和提示词（建议从pe.json文件中提取），自动构造json格式数据.
自己写的，不是很优雅

author:zhaoshe
"""
import os
from pathlib import Path
from typing import Literal
from collections import defaultdict
import json
from tqdm import tqdm
import argparse

def load_prompt(pe_json_path: str='./pe/prompts.json', idx: int=0) -> str:
    """根据输入的idx，加载相应的prompt

    Args:
        pe_json_path(str): 存储prompt的json文件路径
        idx (int): prompt的编号

    Returns:
        str: 对应的prompt
    """
    with open(pe_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompt = f"prompt{idx}"
    return data[prompt]['prompt_text']

def build_item(
    image_json: str,
    prompt: str,
    output_file: str,
    root_dir: Path,
    prefix: str,
    source: str='gt',
    sample_type: str='positive'
) -> None:
    """为图像和prompt生成相应的json数据,单图模式

    Args:
        image_json (str): 存储bbox和对应原始图像的json文件
        prompt (str): 对应的提示词
        output_file(str): 保存的json文件路径
        prefix(str): id的前缀(e.g: stripe_gt_neg)
    """

    try:
        with open(image_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except ValueError as e:
        raise ValueError(f"读取文件 {image_json} 时发生错误: {e}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for idx, item in tqdm(enumerate(data), desc="Processing bbox: "):
                try:
                    original_image_path = str(root_dir / item['original_image_path'])
                    crop_image_path = str(root_dir / item['crop_image_path'])
                    
                    new_entry = {
                        'id': f"{prefix}_{idx:06d}",
                        'image': [original_image_path, crop_image_path],
                        'conversation': [
                            {'from': 'human', 'value': prompt},
                            {'from': 'assistant', 'value': ''}
                        ],
                        'meta_info':{
                            "label": item['label'],
                            "light_source": item['light_source'],
                            "source": source,
                            "sample_type": sample_type
                        }
                    }                    
                    f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
            
                except Exception as e:
                    print(f"处理 {item['id']} 时发生错误 {e}, 已跳过")

    except Exception as e:
        print(f"写入 JSONL 时出错: {e}")

if __name__ == "__main__":
    image_json = '/data/ZS/defect_dataset/1_defect_dataset_bbox/paint_stripe_gt_negative/labels/val.json'
    prompt = load_prompt(pe_json_path='./example/pe.json', idx=2)       # 从pe.json中加载相应的prompt
    output_file = '/data/ZS/defect_dataset/2_sft_dataset/paint_stripe_gt_negative_val.jsonl'          # 输出文件的路径 
    root_dir = Path('/data/ZS/defect_dataset')
    prefix = 'gt_neg'
    
    # 输入单张图像，生成sft格式数据
    build_item(
        image_json = image_json,
        prompt = prompt, 
        output_file = output_file,
        root_dir = root_dir,
        prefix = prefix
    )
