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

# def count_ids(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f if line.strip()]
    
#     all_id = {}
#     for item in data:
#         id_ = item['id']
#         all_id[id_] = item.get(id_, 0) + 1
    
#     return all_id

import json

def get_id(js):
    all_id = set()
    with open(js, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            item = json.loads(line)
            all_id.add(item['id'])

    return all_id

def copy_same_item(input_json, output_json, same_id):
    cnt = 0
    with open(input_json, 'r', encoding='utf-8') as f_in, \
        open(output_json, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            
            item = json.loads(line)
            if item['id'] in same_id:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                cnt += 1
    
    print(f"已从 {input_json} 中复制 {cnt} 条数据到 {output_json}中")

def copy_different_item(input_json, output_json, same_id):
    cnt = 0
    with open(input_json, 'r', encoding='utf-8') as f_in, \
        open(output_json, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            
            item = json.loads(line)
            if item['id'] not in same_id:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                cnt += 1
    
    print(f"已从 {input_json} 中复制 {cnt} 条数据到 {output_json}中")

if __name__ == "__main__":
    # json_path = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-plus.jsonl'
    js1 = '/data/ZS/defect_dataset/6_sft_dataset/test/gemini-3-pro-preview.jsonl'
    js2 = '/data/ZS/defect_dataset/backup/5_api_response/gemini-3-pro-preview.jsonl'
    
    all_id1 = get_id(js1)
    all_id2 = get_id(js2)
  
    same_id = all_id1 & all_id2
    print(f"相同id数量: {len(same_id)}")


    input_json = '/data/ZS/defect_dataset/backup/5_api_response/qwen3-vl-235b-a22b-thinking.jsonl'
    output_json = '/data/ZS/defect_dataset/backup/5_api_response/qwen3-vl-235b-a22b-thinking_not_in_test1000.jsonl'
    copy_different_item(input_json, output_json, same_id)
    # print(ids1)
    # print(ids2)
