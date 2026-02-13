"""

"""


def extract_ai_pred(ai_response):
    try:
        result = json.loads(ai_response)
        return result['defect']
    
    except json.JSONDecodeError as e:
        print((f"❌ 提取VLM判断结果失败: {e}"))
        return 'ERROR'
    

# sample['meta_info']['ai_pred'] = extract_ai_pred(ai_response)    
