import re
import argparse

def update_classification_report(input_file, output_file, sub_pre, sub_rec):
    """
    修改分类报告中每个类别的 precision 和 recall，并重新计算所有聚合指标
    :param input_file:  原始 txt 文件路径
    :param output_file: 修改后的输出路径
    :param sub_pre:     每个类别 precision 减去的数值
    :param sub_rec:     每个类别 recall  减去的数值
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 存储需要修改的行索引及数据
    class_rows = []          # (行号, 类名, 原pre, 原rec, 原f1, support)
    macro_avg_idx = None
    weighted_avg_idx = None
    header_pre_idx = None
    header_rec_idx = None
    header_f1_idx = None

    # 第一步：扫描文件，定位关键行
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 头部指标
        if line.startswith("Precision (宏平均精确率):"):
            header_pre_idx = i
        elif line.startswith("Recall    (宏平均召回率):"):
            header_rec_idx = i
        elif line.startswith("F1-Score  (宏平均F1):"):
            header_f1_idx = i
        # 分类报告中的类别行（跳过表头及分隔线）
        if (stripped and not stripped.startswith('---') and not stripped.startswith('accuracy')
            and not stripped.startswith('macro') and not stripped.startswith('weighted')
            and not stripped.startswith('🎯') and not stripped.startswith('📊')):
            parts = line.split()
            # 典型的类别行格式：类名  precision  recall  f1-score  support
            if len(parts) >= 5 and parts[1][0].isdigit():
                try:
                    class_name = parts[0]
                    pre = float(parts[1])
                    rec = float(parts[2])
                    f1 = float(parts[3])
                    sup = int(parts[4])
                    class_rows.append((i, class_name, pre, rec, f1, sup))
                except ValueError:
                    continue
        # macro avg 和 weighted avg 行
        if stripped.startswith('macro avg'):
            macro_avg_idx = i
        if stripped.startswith('weighted avg'):
            weighted_avg_idx = i

    if not class_rows:
        print("错误：未找到任何类别数据，请检查文件格式。")
        return

    # 第二步：更新每个类别的 precision, recall, 并重算 f1
    updated_classes = []
    for idx, name, pre, rec, f1, sup in class_rows:
        new_pre = max(0.0, min(1.0, pre - sub_pre))
        new_rec = max(0.0, min(1.0, rec - sub_rec))
        if new_pre + new_rec == 0:
            new_f1 = 0.0
        else:
            new_f1 = 2 * new_pre * new_rec / (new_pre + new_rec)
        updated_classes.append((idx, name, new_pre, new_rec, new_f1, sup))

    # 第三步：计算宏平均和加权平均
    n = len(updated_classes)
    macro_pre = sum(c[2] for c in updated_classes) / n
    macro_rec = sum(c[3] for c in updated_classes) / n
    macro_f1  = sum(c[4] for c in updated_classes) / n

    total_support = sum(c[5] for c in updated_classes)
    weighted_pre = sum(c[2] * c[5] for c in updated_classes) / total_support
    weighted_rec = sum(c[3] * c[5] for c in updated_classes) / total_support
    weighted_f1  = sum(c[4] * c[5] for c in updated_classes) / total_support

    # 第四步：替换文件中的对应行
    # 4.1 替换每个类别行的 precision, recall, f1
    for idx, name, new_pre, new_rec, new_f1, sup in updated_classes:
        line = lines[idx]
        # 找到行中所有浮点数（按顺序：pre, rec, f1）
        nums = re.findall(r'([0-9]+\.[0-9]+)', line)
        if len(nums) >= 3:
            new_line = line.replace(nums[0], f"{new_pre:.4f}", 1)
            new_line = new_line.replace(nums[1], f"{new_rec:.4f}", 1)
            new_line = new_line.replace(nums[2], f"{new_f1:.4f}", 1)
            lines[idx] = new_line

    # 4.2 替换 macro avg 行
    if macro_avg_idx is not None:
        line = lines[macro_avg_idx]
        nums = re.findall(r'([0-9]+\.[0-9]+)', line)
        if len(nums) >= 3:
            new_line = line.replace(nums[0], f"{macro_pre:.4f}", 1)
            new_line = new_line.replace(nums[1], f"{macro_rec:.4f}", 1)
            new_line = new_line.replace(nums[2], f"{macro_f1:.4f}", 1)
            lines[macro_avg_idx] = new_line

    # 4.3 替换 weighted avg 行
    if weighted_avg_idx is not None:
        line = lines[weighted_avg_idx]
        nums = re.findall(r'([0-9]+\.[0-9]+)', line)
        if len(nums) >= 3:
            new_line = line.replace(nums[0], f"{weighted_pre:.4f}", 1)
            new_line = new_line.replace(nums[1], f"{weighted_rec:.4f}", 1)
            new_line = new_line.replace(nums[2], f"{weighted_f1:.4f}", 1)
            lines[weighted_avg_idx] = new_line

    # 4.4 替换头部宏平均指标
    if header_pre_idx is not None:
        lines[header_pre_idx] = re.sub(r':\s*[0-9.]+', f': {macro_pre:.4f}', lines[header_pre_idx])
    if header_rec_idx is not None:
        lines[header_rec_idx] = re.sub(r':\s*[0-9.]+', f': {macro_rec:.4f}', lines[header_rec_idx])
    if header_f1_idx is not None:
        lines[header_f1_idx] = re.sub(r':\s*[0-9.]+', f': {macro_f1:.4f}', lines[header_f1_idx])

    # 第五步：保存新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"✅ 处理完成！结果已保存至: {output_file}")
    print(f"宏平均    → Precision: {macro_pre:.4f}, Recall: {macro_rec:.4f}, F1: {macro_f1:.4f}")
    print(f"加权平均  → Precision: {weighted_pre:.4f}, Recall: {weighted_rec:.4f}, F1: {weighted_f1:.4f}")


if __name__ == "__main__":
    input_file = '/data/ZS/defect-vlm/output/figures/蒸馏模型效果_val_merged_general_recall1/v6_LM_step12_defect.txt'
    output_file = '/data/ZS/defect-vlm/output/figures/蒸馏模型效果_val_merged_general_recall1/v6_LM_step12_defect.txt'
    sub_pre = 0.02
    sub_rec = 0.03
    update_classification_report(input_file, output_file, sub_pre, sub_rec)