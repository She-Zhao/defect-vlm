import os
import glob
from collections import defaultdict

# 类别字典
CLASS_NAME2ID = {
    "breakage": 0,
    "inclusion": 1,
    "scratch": 2,
    "crater": 3,
    "run": 4,
    "bulge": 5
}
# 翻转字典方便打印
ID2CLASS_NAME = {v: k for k, v in CLASS_NAME2ID.items()}

def analyze_pseudo_weights(label_dir, gt_dir, exclude_list_file=None):
    print(f"🚀 开始分析纯伪标签权重")
    print(f"📁 分析目录: {label_dir}")
    print(f"🛡️  GT参照目录: {gt_dir}")
    if exclude_list_file:
        print(f"🚫 额外排除文件: {exclude_list_file}")
    print("=" * 60)
    
    # 1. 建立 GT 文件名的黑名单集合
    skip_filenames = set()
    if os.path.exists(gt_dir):
        gt_filenames = set(f for f in os.listdir(gt_dir) if f.endswith('.txt'))
        skip_filenames.update(gt_filenames)
        print(f"   [INFO] 成功加载 GT 参照集，将排除 {len(gt_filenames)} 个真实标签文件。")
    else:
        print(f"❌ 找不到 GT 目录: {gt_dir}，请检查路径！")
        return

    # 2. 如果提供了额外的排除列表 (如 Chunk3)，则加入黑名单
    if exclude_list_file and os.path.exists(exclude_list_file):
        with open(exclude_list_file, 'r') as f:
            exclude_count = 0
            for line in f:
                line = line.strip()
                if line:
                    # 获取文件名(无论文件里写的是绝对路径还是纯文件名)
                    # 确保如果文件里没有 .txt 后缀，就自动补上
                    basename = os.path.basename(line)
                    if not basename.endswith('.txt'):
                        # 如果 chunk3.txt 里存的是图像名或无后缀名，转为 txt
                        basename = os.path.splitext(basename)[0] + '.txt'
                    
                    skip_filenames.add(basename)
                    exclude_count += 1
        print(f"   [INFO] 成功加载额外排除列表，额外排除了 {exclude_count} 个文件。")

    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    if not txt_files:
        print(f"❌ 未找到任何 .txt 文件在目录: {label_dir}")
        return

    # 初始化统计数据
    total_pseudo_files = 0
    total_boxes = 0
    total_weight = 0.0
    
    # 记录每个类别的: [数量, 权重总和, 最小权重, 最大权重]
    class_stats = defaultdict(lambda: {"count": 0, "sum_weight": 0.0, "min_w": float('inf'), "max_w": 0.0})

    invalid_lines = 0

    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        
        # [核心逻辑]：如果这个文件在 GT 或 排除列表 里，直接跳过！
        if basename in skip_filenames:
            continue
            
        total_pseudo_files += 1

        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    weight = float(parts[5])
                elif len(parts) == 5:
                    class_id = int(parts[0])
                    weight = 1.0
                else:
                    invalid_lines += 1
                    continue

                total_boxes += 1
                total_weight += weight
                
                stats = class_stats[class_id]
                stats["count"] += 1
                stats["sum_weight"] += weight
                if weight < stats["min_w"]:
                    stats["min_w"] = weight
                if weight > stats["max_w"]:
                    stats["max_w"] = weight

    if total_boxes == 0:
        print("⚠️ 排除指定文件后，没有解析到任何伪标签边界框！")
        return

    # ================= 打印整体报告 =================
    overall_avg_weight = total_weight / total_boxes
    print("-" * 60)
    print(f"📊 【纯伪标签】整体统计:")
    print(f"   - 分析的伪标签文件数: {total_pseudo_files}")
    print(f"   - 伪缺陷框总数: {total_boxes}")
    print(f"   - 纯伪标签平均权重: {overall_avg_weight:.4f}  <--- 重点看这个！")
    if invalid_lines > 0:
        print(f"   - ⚠️ 解析失败的行数: {invalid_lines}")
    print("-" * 60)

    # ================= 打印各类明细 =================
    print(f"{'类别 (ID)':<18} | {'新增数量':<8} | {'平均权重':<10} | {'最小权重':<10} | {'最大权重':<10}")
    print("-" * 60)
    
    # 按 class_id 排序输出
    for class_id in sorted(class_stats.keys()):
        class_name = ID2CLASS_NAME.get(class_id, f"Unknown({class_id})")
        display_name = f"{class_name} ({class_id})"
        
        stats = class_stats[class_id]
        count = stats["count"]
        avg_weight = stats["sum_weight"] / count if count > 0 else 0
        
        # 处理可能某个类没数据的情况
        min_w = stats["min_w"] if count > 0 else 0.0
        max_w = stats["max_w"] if count > 0 else 0.0
        
        print(f"{display_name:<18} | {count:<8} | {avg_weight:<10.4f} | {min_w:<10.4f} | {max_w:<10.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    # GT 参照目录和 Chunk3 文件列表路径
    GT_DIR = "/data/ZS/v11_input/datasets/col3/train/labels" 
    CHUNK3_TXT = "/data/ZS/flywheel_dataset/chunks/chunk3.txt"

    # 待对比文件夹
    iter2 = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter2_1/row3/train/labels"
    iter3 = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_1/row3/train/labels"
    
    # 待对比文件夹
    # iter2 = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter2_1/row3/train/labels"
    # iter3 = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_1/row3/train/labels"

    # 分析 Iter 2 (本身只有 Chunk1+2)
    analyze_pseudo_weights(iter2, GT_DIR)
    
    # 分析 Iter 3 (剔除 Chunk3，仅保留 Chunk1+2)
    analyze_pseudo_weights(iter3, GT_DIR, exclude_list_file=CHUNK3_TXT)
    
    # 不剔除iter3
    # analyze_pseudo_weights(iter3, GT_DIR)
