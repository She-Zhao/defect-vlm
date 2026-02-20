"""
对调用api返回的结果进行清洗，并划分为回复符合要求的和不符合要求的两部分
"""
import os
import json
from pathlib import Path
from tqdm import tqdm

REQUIRED_KEYS = set(['step1','step2','step3', 'defect'])
REQUIRED_DEFECTS = set(['breakage', 'inclusion', 'crater', 'bulge', 'scratch', 'run', 'none'])

def check_ai_response(item: dict) -> bool:
    """
    判断调用api的得到的回复是否符合规则
    """
    ai_response_str = item['conversation'][1]['value']
    
    # 检查是否为 API 调用失败
    if ai_response_str.startswith("ERROR"):
        if 'meta_info' not in item: item['meta_info'] = {}
        item['meta_info']['fail_reason'] = f"API调用阶段报错: {ai_response_str}"
        return False
        
    # 提取纯 JSON
    # 寻找第一个左大括号和最后一个右大括号的位置
    start_idx = ai_response_str.find('{')
    end_idx = ai_response_str.rfind('}')
    
    # 确保同时找到了 { 和 }，并且 { 在 } 的前面
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        ai_response_str = ai_response_str[start_idx:end_idx+1]
    else:
        if 'meta_info' not in item: item['meta_info'] = {}
        item['meta_info']['fail_reason'] = "模型回复中未找到有效的JSON大括号结构"
        return False

    try:
        # 1. 允许真实换行符
        ai_response = json.loads(ai_response_str, strict=False)
        
        # 2. 将解析出的字典重新转为标准严格的 JSON 字符串
        clean_response_str = json.dumps(ai_response, ensure_ascii=False, indent=4)
        
        # 3. 覆盖原始脏数据
        item['conversation'][1]['value'] = clean_response_str
        
    except json.JSONDecodeError as e:
        if 'meta_info' not in item: item['meta_info'] = {}
        item['meta_info']['fail_reason'] = "JSON解析失败，非标准JSON格式"
        return False

    # 检查回复的key是否正确
    ai_response_keys = set(ai_response.keys())
    if ai_response_keys != REQUIRED_KEYS:
        if 'meta_info' not in item: item['meta_info'] = {}
        item['meta_info']['fail_reason'] = f"模型的回复缺少必须字段, 当前为 {ai_response_keys}"
        return False
    
    # 检查缺陷类别是否在允许的列表中
    ai_response_defect = ai_response.get('defect', '').lower()
    if ai_response_defect not in REQUIRED_DEFECTS:
        if 'meta_info' not in item: item['meta_info'] = {}
        item['meta_info']['fail_reason'] = f"模型回复的缺陷类型不符合要求, 当前为 {ai_response_defect}"
        return False
    
    # 走到这里说明一切完美
    return True
    
def main(api_response_path: str, save_path: str, retry_path: str) -> None:
    save_path = Path(save_path)
    retry_path = Path(retry_path)
    
    save_path.parent.mkdir(exist_ok=True, parents=True)
    retry_path.parent.mkdir(exist_ok=True, parents=True)

    good_cnt = 0
    bad_cnt = 0
    
    with open(api_response_path, 'r', encoding='utf-8') as f_in, \
        open(save_path, 'a', encoding='utf-8') as f_good, \
        open(retry_path, 'w', encoding='utf-8') as f_bad:
        
        for line in tqdm(f_in, desc='Processing item'):
            try:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                
                if check_ai_response(item):
                    f_good.write(json.dumps(item, ensure_ascii=False) + '\n')
                    good_cnt += 1
                    
                else:
                    f_bad.write(json.dumps(item, ensure_ascii=False) + '\n')
                    bad_cnt += 1
                
            except Exception as e:
                print(f"处理id: {item['id']} 时发生错误{e}")
                continue

    print(f"处理完成!")
    print(f"✅ 合格样本 (Good): {good_cnt} -> 保存至 {save_path}")
    print(f"❌ 错误样本 (Bad) : {bad_cnt} -> 保存至 {retry_path}")
        
if __name__ == "__main__":
    main(
        api_response_path = "/data/ZS/defect_dataset/5_api_response/retry/qwen3-vl-235b-a22b-instruct.jsonl",
        save_path = "/data/ZS/defect_dataset/6_sft_dataset/test/qwen3-vl-235b-a22b-instruct.jsonl" ,
        retry_path = "/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-235b-a22b-instruct.jsonl"
    )
 