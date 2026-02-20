import json

# def main(json_path, idx=0):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f if line.strip()]
    
#     item = data[idx]
#     ai_response_str = item['conversation'][1]['value']
    
#     try:
#         ai_response_obj = json.loads(ai_response_str, strict=False)
#     except json.JSONDecodeError as e:
#         print(f"错误位置: 第 {e.lineno} 行, 第 {e.colno} 列")
#         print(f"错误字符: {ai_response_str[e.pos]}")
#         # 显示错误位置前后的内容
#         start = max(0, e.pos - 50)
#         end = min(len(ai_response_str), e.pos + 50)
#         print(f"附近内容: ...{ai_response_str[start:end]}...")
#         return
    
#     ai_response = json.dumps(ai_response_obj, indent=4, ensure_ascii=False)
#     print(ai_response)
#     print(ai_response_str)

def count_ids(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    all_id = {}
    for item in data:
        id_ = item['id']
        all_id[id_] = item.get(id_, 0) + 1
    
    return all_id

if __name__ == "__main__":
    # json_path = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-plus.jsonl'
    json_path1 = '/data/ZS/defect_dataset/4_api_request/test/012_pos200_neg150_rect150.jsonl'
    json_path2 = '/data/ZS/defect_dataset/backup/5_api_response/qwen3-vl-235b-a22b-thinking.jsonl'
    
    ids1 = count_ids(json_path1)
    ids2 = count_ids(json_path2)
  
    print(ids1 == ids2) 
    # print(ids1)
    # print(ids2)
