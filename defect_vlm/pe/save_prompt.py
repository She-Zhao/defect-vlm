import json
import os
import re
from pathlib import Path
import argparse

def get_next_prompt_idx(data: dict) -> int:
    indices = []
    for key in data.keys():
        match = re.match(r'prompt(\d+)', key)
        if match:
            indices.append(int(match.group(1)))
    
    return max(indices)+1 if indices else 0
    
def save_prompt(prompt_text_txt: Path, pe_json_path: Path) -> None:
    with open(prompt_text_txt, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    # 如果文件不存在，创建空字典
    if not os.path.exists(pe_json_path):
        data = {}
    else:
        # 如果文件存在，尝试读取内容
        try:
            with open(pe_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # 如果文件内容不是有效的 JSON，创建空字典
            data = {}

    idx = get_next_prompt_idx(data)
    new_prompt =f'prompt{idx}'
    new_entry = {
        'comment': '',
        'prompt_text': prompt_text
    }
    
    data[new_prompt] = new_entry

    with open(pe_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"🚀🚀🚀 prompt已成功保存!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '保存txt文件中提示词到相应的json文件')
    parser.add_argument('--input_file', type=str, default='./pe/prompt_text.txt', help='撰写提示词的txt文件的路径')
    parser.add_argument('--output_file', type=str, default='./pe/prompts.json', help='存储各版本prompt的json文件')

    args = parser.parse_args()
        
    save_prompt(args.input_file, args.output_file)