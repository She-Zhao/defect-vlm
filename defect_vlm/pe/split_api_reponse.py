"""
对调用api返回的结果进行清洗和划分
"""
import os
import json
from pathlib import Path
from tqdm import tqdm

REQUIRED_KEYS = set(['step1','step2','step3', 'defect'])
REQUIRED_DEFECTS = set(['breakage', 'inclusion', 'crater', 'bulge', 'scratch', 'run', 'None'])

def check_ai_response(item: dict) -> bool:
    """
    判断调用api的得到的回复是否符合规则
    
    具体规则：
    """
    ai_response_str = item['conversation'][1]['value']
    try:
        # 检查是否为 API 调用失败（调用错误会自动保存为ERROR）
        if ai_response_str.startswith("ERROR"):
             raise ValueError(f"API调用阶段报错: {ai_response_str}")
         
        # 检查是否能直接被load来判断是否为json格式
        try:
            ai_response = json.loads(ai_response_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败，非标准JSON格式")

        # 检查回复的key是否正确
        ai_response_keys = set(ai_response.keys())
        if ai_response_keys != REQUIRED_KEYS:
            raise ValueError(f"模型的回复缺少必须字段, 当前为 {ai_response_keys}")
        
        ai_response_defect = ai_response['defect']
        if ai_response_defect not in REQUIRED_DEFECTS:
            raise ValueError(f"模型回复的缺陷类型不符合要求, 当前为 {ai_response_defect}")
        
        return True
    
    except Exception as e:
        item['meta_info']['fail_reason'] = str(e)
        return False
    
def main(api_response_path: str, save_path: str, retry_path: str) -> None:
    save_path = Path(save_path)
    retry_path = Path(retry_path)
    
    save_path.parent.mkdir(exist_ok=True, parent=True)
    retry_path.parent.mkdir(exist_ok=True, parent=True)

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
        api_response_path = "",
        save_path = "" ,
        retry_path = ""
    )
 