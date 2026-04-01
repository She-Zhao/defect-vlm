"""
从/data/ZS/defect_dataset/4_api_request/val下面的文件里面抽取一些样本，组成一个新的测试集，用于选取教师模型
"""
import os
import json
import random
from typing import List

# 固定种子以保证每次抽样结果一致
random.seed(42)

def main(jsonl_list: List[str], sample_num_list: List[int], prefixes: List[str], save_path: str):
    """
    Args:
        jsonl_list: 源文件路径列表
        sample_num_list: 对应的抽样数量
        prefixes: 对应的ID前缀列表 (例如 ['pos', 'neg', 'rect'])
        save_path: 保存路径
    """
    if not (len(jsonl_list) == len(sample_num_list) == len(prefixes)):
        raise ValueError("jsonl_list, sample_num_list, prefixes 长度必须一致")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_samples = []
    
    print(f"🚀 开始抽样合并...")
    
    for jsonl_path, sample_num, prefix in zip(jsonl_list, sample_num_list, prefixes):
        print(f"正在处理: {os.path.basename(jsonl_path)} -> 抽取 {sample_num} 条 (前缀: {prefix})")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 1. 解析所有数据
        data = []
        for line in lines:
            line = line.strip()
            if line:
                data.append(json.loads(line))
        
        # 2. 随机打乱并切片
        # 注意：如果文件行数少于 sample_num，则取全部
        if len(data) < sample_num:
            print(f"⚠️ 警告: 文件行数 ({len(data)}) 少于请求数量 ({sample_num})，将使用全部数据。")
            current_samples = data
        else:
            # random.sample 比 shuffle 更高效，且不改变原列表
            current_samples = random.sample(data, sample_num)
            
        # 3. 修改 ID 逻辑
        for item in current_samples:
            original_id = item.get('id')
            
            # 方法 A: 字符串拼接前缀 (推荐)
            # 例如: "pos_1000001"
            new_id = f"{prefix}_{original_id}"
            
            # 更新 ID
            item['id'] = new_id
            
            # 建议：在 meta_info 中备份原始 ID，方便追溯
            if 'meta_info' not in item:
                item['meta_info'] = {}
            item['meta_info']['origin_id'] = original_id
            item['meta_info']['origin_source'] = os.path.basename(jsonl_path)
            
            all_samples.append(item)
            
    # 4. 再次打乱整体顺序（可选，防止同类样本聚集）
    random.shuffle(all_samples)
    
    # 5. 保存
    with open(save_path, 'w', encoding='utf-8') as f:     
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 合并完成！共 {len(all_samples)} 条样本。")
    print(f"📂 保存路径: {save_path}")

    
if __name__ == "__main__":
    jsonl_list = [
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase012_gt_positive.jsonl',
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase012_gt_negative.jsonl',
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase012_gt_rectification.jsonl',
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase123_gt_positive.jsonl',
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase123_gt_negative.jsonl',
        '/data/ZS/defect_dataset/4_api_request/teacher/val/stripe_phase123_gt_rectification.jsonl'        
    ]
    
    sample_num_list = [4000, 3000, 3000, 4000, 3000, 3000]
    
    # 定义对应的前缀，建议简短一点
    prefixes = ['sp012_gt_pos_tea', 'sp012_gt_neg_tea', 'sp012_gt_rect_tea',
                'sp123_gt_pos_tea', 'sp123_gt_neg_tea', 'sp123_gt_rect_tea']
    
    save_path = '/data/ZS/defect_dataset/4_api_request/teacher/test/merged_all.jsonl'
    
    main(jsonl_list, sample_num_list, prefixes, save_path)
