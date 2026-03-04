# debug_json_error.py
import json

file_path = '/data/ZS/defect_dataset/8_model_reponse/student/test/qwen3-vl-8b-instrusct.jsonl'

print("=== 检查文件前5行 ===")
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print(f"总行数: {len(lines)}")

for i in range(min(5, len(lines))):
    line_num = i + 1
    line = lines[i].strip()
    print(f"\n--- 第 {line_num} 行 ---")
    print(f"长度: {len(line)}")
    print(f"前100字符: {line[:100]}")
    
    # 检查整行 JSON
    try:
        item = json.loads(line)
        print("✅ 整行 JSON 解析成功")
        
        # 检查 pred 字段
        if 'pred' in item:
            pred_content = item['pred']
            print(f"pred 类型: {type(pred_content)}")
            print(f"pred 前50字符: {str(pred_content)[:50]}")
            
            # 如果是字符串，尝试解析
            if isinstance(pred_content, str):
                try:
                    pred_json = json.loads(pred_content)
                    print("✅ pred 字段 JSON 解析成功")
                except json.JSONDecodeError as e:
                    print(f"❌ pred 字段 JSON 解析失败: {e}")
                    print(f"问题内容: {pred_content[:200]}")
    except json.JSONDecodeError as e:
        print(f"❌ 整行 JSON 解析失败: {e}")