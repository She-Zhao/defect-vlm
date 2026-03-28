"""
用于查看jsonl的脚本，从jsonl里面随机抽取三条并打印出来
"""
import json
from pathlib import Path
import random

random.seed(42)

def inspect_jsonl(file_path: str, num_samples: int = 3):
    path = Path(file_path)
    
    # 1. 检查文件是否存在
    if not path.exists():
        print(f"❌ 错误: 文件不存在 -> {path}")
        return

    print(f"🔍 正在检查文件: {path.name}")
    print(f"📂 完整路径: {path}")
    print("=" * 60)

    try:
        # 2. 读取所有行
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"📊 总数据条数: {total_lines}")
        
        # 3. 随机抽取 num_samples 条
        if total_lines < num_samples:
            print(f"⚠️  警告: 总数据条数({total_lines})小于要抽取的数量({num_samples})，将返回所有数据")
            sampled_indices = list(range(total_lines))
        else:
            sampled_indices = sorted(random.sample(range(total_lines), num_samples))
        
        print(f"🎲 随机抽取的索引: {sampled_indices}")
        print("=" * 60)
        
        # 4. 解析并打印抽取的数据
        for i, line_idx in enumerate(sampled_indices):
            line = lines[line_idx].strip()
            if not line:
                continue
                
            # 解析 JSON
            entry = json.loads(line)
            
            # 打印数据
            print(f"📄 [第 {line_idx + 1} 条数据] (随机抽取第 {i+1} 条):")
            print(json.dumps(entry, indent=4, ensure_ascii=False))
            
            # --- 自动检查逻辑 ---
            human_text = entry['conversation'][0]['value']
            if "{}" in human_text:
                print("⚠️  警告: 提示词中仍然包含 '{}'，替换未生效！")
            else:
                print("✅ 提示词格式检查通过 (无 '{}' 占位符)")
            
            print(entry['conversation'][0]['value'])
            
            # 检查图片路径
            img_paths = entry.get('image', [])
            if not img_paths or not all(isinstance(p, str) for p in img_paths):
                print("⚠️  警告: 图片路径格式看似不正确")
            
            print("-" * 60)
                
    except Exception as e:
        print(f"❌ 读取过程中发生错误: {e}")

if __name__ == "__main__":
    # 这里填入你刚才生成的文件路径
    target_file = '/data/ZS/flywheel_dataset/6_vlm_response/iter0/sp012_0p1_v1_LM.jsonl'
    
    inspect_jsonl(target_file, num_samples=1)