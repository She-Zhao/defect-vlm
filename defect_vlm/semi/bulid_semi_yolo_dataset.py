"""
构建用于 YOLO 多流半监督训练的数据集 (GT + Pseudo Labels)。
采用 Bbox 作为唯一键进行图像溯源，规避 ID 冲突问题。

输入：
原始纯粹的GT的数据集目录
无标注的数据集图像目录
组合不同光源的结果json文件
对VLM处理结果过滤后的jsonl文件

输出：
最终的输出文件夹
"""

import os
import json
from pathlib import Path
from PIL import Image

def get_bbox_key(bbox, decimals=4):
    """
    将 bbox 转化为保留固定小数位数的元组，作为字典匹配的唯一 Key。
    解决浮点数精度漂移问题。
    """
    return tuple(round(float(x), decimals) for x in bbox)

def convert_to_yolo_format(bbox, img_w, img_h):
    """
    将绝对坐标 [x_min, y_min, w, h] 转换为 YOLO 归一化中心点坐标 [x_center, y_center, w_norm, h_norm]
    """
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    
    # 归一化并限制在 0-1 之间，防止越界报错
    x_center = max(0.0, min(1.0, x_center / img_w))
    y_center = max(0.0, min(1.0, y_center / img_h))
    w_norm = max(0.0, min(1.0, w / img_w))
    h_norm = max(0.0, min(1.0, h / img_h))
    
    return [x_center, y_center, w_norm, h_norm]

def build_semi_dataset(
    gt_dir, 
    unlabeled_dir, 
    mapping_json, 
    refined_jsonl, 
    output_dir, 
    class_name2id
):
    gt_dir = Path(gt_dir)
    unlabeled_dir = Path(unlabeled_dir)
    output_dir = Path(output_dir)
    
    print(f"🚀 开始构建半监督快照数据集: {output_dir.name}")
    
    # ================= 1. 解析映射文件，建立 Bbox -> Filename 的字典 =================
    print("🔗 正在解析映射文件，建立 Bbox 索引...")
    bbox2filename = {}
    with open(mapping_json, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        for item in mapping_data:
            # 使用保留 4 位小数的 bbox 元组作为唯一键
            bbox_key = get_bbox_key(item["bbox"])
            
            # 取出第一张原图路径提取文件名
            orig_path = item["original_image_paths"][0]
            filename = Path(orig_path).name
            bbox2filename[bbox_key] = filename
            
    print(f"   [状态] 成功建立 {len(bbox2filename)} 个 Bbox 映射关系。")
            
    # ================= 2. 解析 JSONL 伪标签，按图片重组数据 =================
    print("📖 正在解析精炼伪标签...")
    # pseudo_dict 结构: {filename: [ [cls_id, xc, yc, w, h, final_weight], ... ]}
    pseudo_dict = {}
    miss_count = 0
    
    with open(refined_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 提取并转换 bbox 作为查询键
            bbox_key = get_bbox_key(data["bbox"])
            filename = bbox2filename[bbox_key]
            
            if not filename:
                miss_count += 1
                # 仅打印前几个找不到的用于排错，防止刷屏
                if miss_count <= 5:
                    print(f"⚠️ 映射表中找不到 Bbox {bbox_key}，跳过该伪标签。")
                continue
                
            final_label = data["final_label"]
            final_weight = data["final_weight"]
            bbox = data["bbox"]         # 使用原始的bbox，最后保存的时候再阶段6位，和GT保持一致
            
            if final_label not in class_name2id:
                continue
                
            cls_id = class_name2id[final_label]
            
            if filename not in pseudo_dict:
                pseudo_dict[filename] = []
            pseudo_dict[filename].append({
                "bbox": bbox, 
                "cls_id": cls_id, 
                "weight": final_weight
            })
            
    if miss_count > 0:
        print(f"⚠️ 总共有 {miss_count} 个 Bbox 无法在映射表中找到对应文件。")

    # ================= 3. 构建多流目录结构 =================
    print("📂 正在生成全新目录结构...")
    sub_dirs = [f"train_{i}" if i > 0 else "train" for i in range(6)] + \
               [f"val_{i}" if i > 0 else "val" for i in range(6)]
    
    for sub in sub_dirs:
        (output_dir / sub / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / sub / "labels").mkdir(parents=True, exist_ok=True)

    # ================= 4. 处理纯净 GT 数据 =================
    print("🖼️ 正在处理并软链接 GT 数据...")
    gt_img_count = 0
    
    for split in ["train", "val"]:
        for i in range(6):
            folder = f"{split}_{i}" if i > 0 else split
            src_img_dir = gt_dir / folder / "images"
            dst_img_dir = output_dir / folder / "images"
            
            if not src_img_dir.exists(): continue
            
            for img_path in src_img_dir.glob("*.png"):
                dst_img_path = dst_img_dir / img_path.name
                if not dst_img_path.exists():
                    os.symlink(img_path, dst_img_path)
                    
                # 【核心】：仅在主分支生成 labels
                if i == 0:
                    src_lbl_path = gt_dir / split / "labels" / f"{img_path.stem}.txt"
                    dst_lbl_path = output_dir / split / "labels" / f"{img_path.stem}.txt"
                    
                    if src_lbl_path.exists() and not dst_lbl_path.exists():
                        # GT 数据末尾追加权重 1.0
                        with open(src_lbl_path, 'r') as fin, open(dst_lbl_path, 'w') as fout:
                            for line in fin:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    fout.write(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} 1.0\n")
                    gt_img_count += 1

    # ================= 5. 处理伪标签数据 =================
    print("🤖 正在注入 VLM 伪标签数据...")
    pseudo_img_count = 0
    
    for filename, anns in pseudo_dict.items():
        src_main_img = unlabeled_dir / "val" / "images" / filename
        if not src_main_img.exists():
            print(f"⚠️ 找不到无标签原图: {src_main_img}，跳过该图。")
            continue
            
        # 动态读取真实宽高
        with Image.open(src_main_img) as img:
            img_w, img_h = img.size
            
        # 多流图像软链接
        for i in range(6):
            src_folder = f"val_{i}" if i > 0 else "val"
            dst_folder = f"train_{i}" if i > 0 else "train"  
            
            src_img = unlabeled_dir / src_folder / "images" / filename
            dst_img = output_dir / dst_folder / "images" / filename
            
            if src_img.exists() and not dst_img.exists():
                os.symlink(src_img, dst_img)
                
        # 生成带 final_weight 的 TXT
        dst_lbl_path = output_dir / "train" / "labels" / f"{Path(filename).stem}.txt"
        with open(dst_lbl_path, 'w') as fout:
            for ann in anns:
                norm_bbox = convert_to_yolo_format(ann["bbox"], img_w, img_h)
                fout.write(f"{ann['cls_id']} {norm_bbox[0]:.6f} {norm_bbox[1]:.6f} {norm_bbox[2]:.6f} {norm_bbox[3]:.6f} {ann['weight']:.4f}\n")
        
        pseudo_img_count += 1

    print("\n" + "=" * 50)
    print("✅ 半监督超级数据集构建完成！")
    print("=" * 50)
    print(f"📁 存放路径: {output_dir}")
    print(f"📊 数据规模统计:")
    print(f"  ├── 真实标注 (GT) 图像数  : {gt_img_count} 张")
    print(f"  └── 注入伪标签 (Pseudo) 图像数: {pseudo_img_count} 张")
    print(f"  └── 融合后训练集总规模  : {gt_img_count + pseudo_img_count} 张")
    print("=" * 50)


if __name__ == '__main__':
    # 类别映射字典
    CLASS_NAME2ID = {
        "breakage": 0,
        "inclusion": 1,
        "scratch": 2,
        "crater": 3,
        "run": 4,
        "bulge": 5
    }

    # 路径配置1，尽量不用sp123
    MAPPING_JSON = "/data/ZS/flywheel_dataset/4_composite_yolo_preds/iter0/labels/sp012_0p1.json"       # 尽量只用012，因为GT是在sp012上训练的
    REFINED_JSONL = "/data/ZS/flywheel_dataset/8_pseudo_labels/iter0/sp012_0p1_v1_LM_1.jsonl"           # 尽量只用012，因为GT是在sp012上训练的

    # 路径配置2，修改横向还是纵向条纹
    GT_DIR = "/data/ZS/v11_input/datasets/col3"                                         # 这里用的是创建软链接的方式，这个文件夹里面的内容一定不能动！
    UNLABELED_DIR = "/data/ZS/flywheel_dataset/0_multi_input/sp012/col3"                # 这里用的是创建软链接的方式，这个文件夹里面的内容一定不能动！
    OUTPUT_DIR = "/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter0/col3"
    
    build_semi_dataset(
        gt_dir=GT_DIR,
        unlabeled_dir=UNLABELED_DIR,
        mapping_json=MAPPING_JSON,
        refined_jsonl=REFINED_JSONL,
        output_dir=OUTPUT_DIR,
        class_name2id=CLASS_NAME2ID
    )
