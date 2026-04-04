"""
从iter3的yolo格式数据集中提取出来只有chunk12的数据
"""
import os

def create_filtered_dataset(src_base, dst_base, chunk3_txt):
    print(f"🚀 开始构建剥离 Chunk3 的对比数据集...")
    print(f"📁 源目录  : {src_base}")
    print(f"🎯 目标目录: {dst_base}")
    print("=" * 60)

    # 1. 解析 chunk3 黑名单
    blacklist = set()
    if os.path.exists(chunk3_txt):
        with open(chunk3_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 剥离可能存在的扩展名，只保留核心文件名
                    base_name = os.path.splitext(os.path.basename(line))[0]
                    blacklist.add(base_name)
        print(f"   [INFO] 成功加载 Chunk3 黑名单，共计 {len(blacklist)} 个基础文件名。")
    else:
        print(f"❌ 找不到 Chunk3 列表文件: {chunk3_txt}")
        return

    # 2. 遍历源目录并建立软链接
    skipped_cache = 0
    skipped_chunk3 = 0
    linked_files = 0
    
    # 确保目标基础目录存在
    os.makedirs(dst_base, exist_ok=True)

    for root, dirs, files in os.walk(src_base):
        # 计算相对路径并创建对应的目标子目录
        rel_path = os.path.relpath(root, src_base)
        target_dir = os.path.join(dst_base, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            # 规则 1：跳过 YOLO 生成的缓存文件
            if file.endswith('.cache'):
                skipped_cache += 1
                continue

            # 规则 2：检查是否在黑名单中 (无论是 .txt 还是 .jpg/.png，只要名字匹配就踢掉)
            file_base = os.path.splitext(file)[0]
            if file_base in blacklist:
                skipped_chunk3 += 1
                continue

            # 规则 3：建立绝对路径软链接 (保护物理磁盘空间)
            src_file_path = os.path.realpath(os.path.join(root, file))
            dst_file_path = os.path.abspath(os.path.join(target_dir, file))

            if not os.path.exists(dst_file_path):
                os.symlink(src_file_path, dst_file_path)
                linked_files += 1

    print("-" * 60)
    print(f"📊 处理完成总结:")
    print(f"   - ✅ 成功软链接文件数: {linked_files} (包含 GT, Chunk1, Chunk2 的图像和标签)")
    print(f"   - 🗑️  跳过的 .cache 数 : {skipped_cache}")
    print(f"   - 🚫 拦截的 Chunk3 数: {skipped_chunk3}")
    print(f"🎯 新数据集已就绪: {dst_base}")

if __name__ == "__main__":
    SRC_DIR = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_7"
    DST_DIR = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_7_chunk12"
    CHUNK3_TXT = "/data/ZS/flywheel_dataset/chunks/chunk3.txt"
    
    create_filtered_dataset(SRC_DIR, DST_DIR, CHUNK3_TXT)
