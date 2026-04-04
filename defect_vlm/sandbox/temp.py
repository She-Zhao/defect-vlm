import os
import glob

def check_empty_labels(label_dir):
    print(f"🔍 正在检查目录: {label_dir}")
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    if not txt_files:
        print("❌ 警告：该目录下没有找到任何 .txt 文件！")
        return

    empty_files = []
    
    for f in txt_files:
        # 读取文件，剥离前后的空格和换行符
        with open(f, 'r') as file:
            content = file.read().strip()
            # 如果剥离空白后什么都没剩下，说明是无效标签
            if not content:
                empty_files.append(f)

    print("-" * 50)
    print(f"📊 检查完毕！共扫描 {len(txt_files)} 个标签文件。")
    
    if empty_files:
        print(f"⚠️ 发现 {len(empty_files)} 个空标签（代表纯背景图）！")
        print("以下是前 10 个空标签的示例：")
        for ef in empty_files[:10]:
            print(f"   - {os.path.basename(ef)}")
    else:
        print("✅ 完美！所有标签文件都包含有效的标注数据，没有空文件！")

if __name__ == "__main__":
    # 你可以把需要检查的目录都加进这个列表里
    directories_to_check = [
        "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_1/col3/train/labels",
        "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_1/row3/train/labels"  # 顺手把 row3 也查了
    ]
    
    for d in directories_to_check:
        check_empty_labels(d)
        print("\n")