"""
根据 `/data/ZS/flywheel_dataset/chunks/chunk_splits.json` 里面的chunk分组，
从 `/data/ZS/flywheel_dataset/0_multi_input/sp012` 里面复制（软链接）图像到每次迭代当中（支持多个chunk联合）
输入：
`/data/ZS/flywheel_dataset/chunks/chunk_splits.json`
`/data/ZS/flywheel_dataset/0_multi_input/sp012`

输出：
能够直接被 `/data/ZS/v11_input/inference.py` 调用的格式
"""
import os
import json
import shutil
from pathlib import Path

def build_symlinks_for_chunks(chunk_json, source_dir, target_dir, target_chunks=["chunk1"]):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print(f"📖 正在加载分块名单: {chunk_json}")
    with open(chunk_json, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        
    # === 新增逻辑：将多个 chunk 的名单合并 ===
    target_filenames = []
    for chunk_name in target_chunks:
        if chunk_name not in chunks_data:
            raise ValueError(f"❌ 找不到指定的 {chunk_name}，请检查 JSON 文件！")
        target_filenames.extend(chunks_data[chunk_name])
        
    print(f"🎯 目标 Chunks: {target_chunks} | 总计图像数量: {len(target_filenames)}")

    # 定义多流的 6 个文件夹后缀
    sub_folders = ['val', 'val_1', 'val_2', 'val_3', 'val_4', 'val_5']
    
    for sub in sub_folders:
        src_img_dir = source_dir / sub / 'images'
        tgt_img_dir = target_dir / sub / 'images'
        
        # 1. 安全清理旧的软链接目录
        if tgt_img_dir.exists():
            shutil.rmtree(tgt_img_dir)
            
        # 2. 重新创建干净的目标目录
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 建立软链接
        success_count = 0
        missing_count = 0
        
        for filename in target_filenames:
            src_file = src_img_dir / filename
            tgt_file = tgt_img_dir / filename
            
            if src_file.exists():
                os.symlink(str(src_file), str(tgt_file))
                success_count += 1
            else:
                missing_count += 1
                
        print(f"✅ 视角 [{sub}] 处理完毕: 成功链接 {success_count} 张, 缺失 {missing_count} 张。")

    print("\n" + "="*50)
    print(f"🎉 推理数据集软链接已构建完成！")
    print(f"📁 目标路径: {target_dir}")
    print("="*50)

if __name__ == '__main__':
    # ================= 配置区 =================
    CHUNK_JSON_PATH = "/data/ZS/flywheel_dataset/chunks/chunk_splits.json"              # 不用动
    VIEW_NAME = "col3"  
    SOURCE_BASE_DIR = f"/data/ZS/flywheel_dataset/0_multi_input/sp012/{VIEW_NAME}"      # 不用动
    TARGET_BASE_DIR = f"/data/ZS/flywheel_dataset/0_multi_input/iter3/{VIEW_NAME}"      # 修改文件夹名称 iter1/2/3
    
    # 【完美适配迭代】
    # Iter 1: ["chunk1"]
    # Iter 2: ["chunk1", "chunk2"]
    # Iter 3: ["chunk1", "chunk2", "chunk3"]
    TARGET_CHUNKS = ["chunk1", "chunk2", "chunk3"]
    # ==========================================

    build_symlinks_for_chunks(
        chunk_json=CHUNK_JSON_PATH,
        source_dir=SOURCE_BASE_DIR,
        target_dir=TARGET_BASE_DIR,
        target_chunks=TARGET_CHUNKS
    )
