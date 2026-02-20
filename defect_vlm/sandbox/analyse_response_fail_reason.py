"""
查看模型回复结果，总结错误原因
"""
import json

def main(retry_path):
    fail_reasons = {}
    with open(retry_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            fail_reason = item['meta_info']['fail_reason']
            fail_reasons[fail_reason] = fail_reasons.get(fail_reason, 0) + 1
    
    print(fail_reasons)

if __name__ == "__main__":
    retry_path = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-plus.jsonl'
    main(retry_path)
