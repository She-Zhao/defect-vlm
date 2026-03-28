"""
统计精简后的 JSONL 文件中，不同置信度阈值下的数据分布情况。
帮助确定最佳的 th_l (低召回过滤线) 和 th_h (高确信采纳线)。
"""
import json
import numpy as np
from pathlib import Path

def analyze_confidence_distribution(jsonl_path, thresholds):
    input_path = Path(jsonl_path)
    
    if not input_path.exists():
        print(f"❌ 找不到文件: {input_path}")
        return

    print(f"📖 正在读取文件: {input_path.name}")
    
    confidences = []
    
    # 1. 提取所有置信度
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    confidences.append(data['confidence'])
                except KeyError:
                    print("⚠️ 警告：发现缺少 'confidence' 字段的行，已跳过。")
                except json.JSONDecodeError:
                    print("⚠️ 警告：发现 JSON 格式错误的行，已跳过。")

    total_items = len(confidences)
    if total_items == 0:
        print("❌ 未提取到任何有效的置信度数据！")
        return
        
    conf_array = np.array(confidences)
    print(f"\n✅ 数据加载完毕，总计 bbox 数量: {total_items}")
    
    # 确保阈值列表是有序的
    thresholds = sorted(thresholds)
    
    print("\n" + "=" * 50)
    print("📊 累积数据量统计 (大于等于当前阈值)")
    print("=" * 50)
    print(f"{'Threshold (>=)':<15} | {'Count':<10} | {'Percentage':<12}")
    print("-" * 50)
    
    for th in thresholds:
        count = np.sum(conf_array >= th)
        percentage = (count / total_items) * 100
        print(f"{th:<15.4f} | {count:<10} | {percentage:>6.2f} %")

    # === 附加福利：区间分布统计 ===
    # 这对于你评估送给 VLM 的“疑难杂症”数量非常有帮助
    print("\n" + "=" * 50)
    print("📦 区间数据量统计 (介于两个阈值之间)")
    print("=" * 50)
    print(f"{'Interval':<20} | {'Count':<10} | {'Percentage':<12}")
    print("-" * 50)
    
    # 添加 0.0 和 1.0 以闭合区间
    interval_th = [0.0] + thresholds + [1.0]
    interval_th = sorted(list(set(interval_th))) # 去重并排序
    
    for i in range(len(interval_th) - 1):
        low_th = interval_th[i]
        high_th = interval_th[i+1]
        
        # 左闭右开区间，除了最后一个是全闭合
        if i == len(interval_th) - 2:
            count = np.sum((conf_array >= low_th) & (conf_array <= high_th))
            interval_str = f"[{low_th:.3f}, {high_th:.3f}]"
        else:
            count = np.sum((conf_array >= low_th) & (conf_array < high_th))
            interval_str = f"[{low_th:.3f}, {high_th:.3f})"
            
        percentage = (count / total_items) * 100
        print(f"{interval_str:<20} | {count:<10} | {percentage:>6.2f} %")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # 你上一步刚刚跑出来的精简数据路径
    INPUT_JSONL = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/sp012_0p1_v1_LM.jsonl"
    
    # 你想测试的阈值列表 (可以随意增删修改)
    # 建议包含极低阈值(看噪点)、中间阈值(看模糊地带)和高阈值(看强确信样本)
    TEST_THRESHOLDS = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]
    
    analyze_confidence_distribution(INPUT_JSONL, TEST_THRESHOLDS)