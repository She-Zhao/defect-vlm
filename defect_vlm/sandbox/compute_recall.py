import os

# ================= 配置路径 =================
# 请填入你当前存放这些 txt 和 png 文件的源文件夹路径
INPUT_DIR = "/data/ZS/defect-vlm/output/figures"  # <--- 请修改这里

# 你指定的输出文件夹
OUTPUT_DIR = "/data/ZS/defect-vlm/output/figures/蒸馏模型效果_val_merged_recall"
# ============================================

def process_txt_files():
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 目标类别集合（用于精准定位包含数据的行）
    target_classes = {'breakage', 'inclusion', 'crater', 'bulge', 'scratch', 'run', 'background'}

    processed_count = 0

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.txt'):
            continue  # 跳过 png 和其他文件

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 1. 解析详细分类报告，提取每个类别的 recall
        recalls = []
        for line in lines:
            parts = line.split()
            # 如果该行至少有5个元素，且开头是我们的目标类别
            if len(parts) >= 5 and parts[0] in target_classes:
                try:
                    # 按照 classification_report 格式，下标 2 是 recall
                    recall_val = float(parts[2])
                    recalls.append(recall_val)
                except ValueError:
                    continue

        if not recalls:
            print(f"⚠️ 警告: 在 {filename} 中没有提取到类别数据，跳过。")
            continue

        # 计算宏平均召回率 (未加权平均)
        macro_recall = sum(recalls) / len(recalls)

        # 2. 将计算结果插入到顶部的“核心全局指标”区域
        new_lines = []
        inserted = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            # 找到 F1-Score 这一行，在它下面追加 Recall
            if "F1-Score  (宏平均F1):" in line:
                recall_line = f"Recall    (宏平均召回率): {macro_recall:.4f}\n"
                new_lines.append(recall_line)
                inserted = True

        if not inserted:
            print(f"⚠️ 警告: 在 {filename} 中未找到插入位置，原样复制。")

        # 3. 写入新的文件夹
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        print(f"✅ 成功处理: {filename} | 计算得 Macro-Recall: {macro_recall:.4f}")
        processed_count += 1

    print("=" * 60)
    print(f"🎉 处理完成！共成功处理 {processed_count} 个 TXT 文件。")
    print(f"📁 新文件已保存至: {OUTPUT_DIR}")
    print("注：所有源文件均未做任何修改。")

if __name__ == '__main__':
    process_txt_files()