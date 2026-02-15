"""
从/data/ZS/defect_dataset/4_api_request/val下面的文件里面抽取一些样本，组成一个新的测试集，用于选取教师模型
"""
import os
import json
import random
random.seed(42)

def main(jsonl_list, sample_num_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_samples = []
    for jsonl, sample_num in zip(jsonl_list, sample_num_list):
        with open(jsonl, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            random.shuffle(data)
            samples = data[:sample_num]
            all_samples.extend(samples)
    
    with open(save_path, 'w', encoding='utf-8') as f:     
        for line in all_samples:
            f.write(json.dumps(line) + '\n')
    
    
if __name__ == "__main__":
    jsonl_list = [
        '/data/ZS/defect_dataset/4_api_request/val/stripe_phase012_gt_positive.jsonl',
        '/data/ZS/defect_dataset/4_api_request/val/stripe_phase012_gt_negative.jsonl',
        '/data/ZS/defect_dataset/4_api_request/val/stripe_phase012_gt_rectification.jsonl'
    ]
    
    sample_num_list = [200, 150, 150]
    save_path = '/data/ZS/defect_dataset/4_api_request/test/012_pos200_neg150_rect150.jsonl'
    
    main(jsonl_list, sample_num_list, save_path)