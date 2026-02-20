"""
将rectification中的sample_type从negative矫正为rectification
"""
import json

def rectify_jsonl(input_file, output_file):
    cnt = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            # 解析 JSON
            data = json.loads(line)
            
            # 检查 image 列表中的任何一个路径是否包含 "rectification"
            images = data['image']
            has_rectification = any("rectification" in img_path for img_path in images)
            
            if has_rectification:
                data["meta_info"]["sample_type"] = "rectification"
                cnt += 1
            # 写入新文件，确保中文不转义
            json.dump(data, f_out, ensure_ascii=False)
            f_out.write('\n')

    print(f"修改 {cnt} 条文件")
if __name__ == "__main__":
    # 在这里替换你的文件名
    input_path = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-plus.jsonl'
    output_path = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-plus1.jsonl'
    
    rectify_jsonl(input_path, output_path)
    print(f"处理完成！修正后的文件已保存至: {output_path}")
