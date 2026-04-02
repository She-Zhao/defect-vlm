"""
消融实验：传统半监督伪标签生成 (Baseline)
基于单一高阈值 (th) 对 YOLO 打标结果进行过滤，完全去除 VLM 决策及所有权重机制。
"""
import json
from pathlib import Path

def generate_baseline_pseudo_labels(
    input_jsonl,
    output_jsonl,
    th
):
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 开始生成传统半监督基线伪标签 (无 VLM)...")
    print(f"   [配置] 统一高置信度阈值 th: {th}")
    
    # 极简决策统计器
    stats = {
        "total_input": 0,
        "discard_low_conf": 0,    # P < th 被抛弃
        "final_saved": 0          # P >= th 直接采信
    }
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            data = json.loads(line)
            stats["total_input"] += 1
            
            # 只提取 YOLO 相关的初筛字段，无视 VLM 的输出
            p_yolo = data["confidence"]
            c_yolo = data["prior_label"]
            
            # ================= 极简决策树 =================
            if p_yolo < th:
                # 抛弃低于高阈值的框
                stats["discard_low_conf"] += 1
                continue
            else:
                # 直接信任 YOLO 的高分框
                final_label = c_yolo
                decision_source = "YOLO_Only_Baseline"
                stats["final_saved"] += 1
            # ==============================================
            
            # 组装兼容下游 Pipeline 的结构
            refined_item = {
                "id": data["id"],
                "images": data["images"],
                "bbox": data["bbox"],
                "model_source": data["model_source"],
                "final_label": final_label,
                # 【关键点】强制将权重设为 1.0，兼容下游 dataset 和 loss 代码，代表传统的 Hard Label
                "final_weight": 1.0, 
                "decision_source": decision_source
            }
            
            fout.write(json.dumps(refined_item, ensure_ascii=False) + '\n')

    # 保存元数据
    meta_info = {
        "hyperparameters": {
            "th": th,
            "note": "Ablation Study 1: Traditional SSL (No VLM, Hard Labels, Single High Threshold)"
        },
        "statistics": stats
    }
    
    meta_path = output_path.parent / f"{output_path.stem}_meta.json"
    
    with open(meta_path, 'w', encoding='utf-8') as fmeta:
        json.dump(meta_info, fmeta, ensure_ascii=False, indent=4)
    # ================================================================

    # 打印极简漏斗报告
    print("\n" + "=" * 50)
    print("🎯 基线伪标签过滤统计报告 (YOLO High Threshold)")
    print("=" * 50)
    print(f"📥 初始输入框总数 : {stats['total_input']}")
    print(f"  ├── 🗑️  [过滤] 因置信度低于 {th} 抛弃 : {stats['discard_low_conf']}")
    print(f"  └── 🟢  [保留] YOLO 高置信度直通 : {stats['final_saved']}")
    print("-" * 50)
    print(f"💾 最终落盘保存伪标签数 : {stats['final_saved']}")
    print(f"   保留率 : {(stats['final_saved'] / stats['total_input']) * 100:.2f}%")
    print("=" * 50)
    print(f"✅ 基线伪标签已保存至: {output_path}")
    print(f"📄 元数据已保存至: {meta_path}")

if __name__ == '__main__':
    # 1. 路径配置
    input_jsonl = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/iter0/sp012_0p1_v1_LM.jsonl"
    # 建议放到专门的 baseline 文件夹下，以免和完整版混淆
    output_jsonl = "/data/ZS/flywheel_dataset/8_pseudo_labels/iter0/wo_vlm/sp012_0p25_wo_vlm.jsonl"
    
    # 2. 超参数配置 (对于传统半监督，通常直接卡你测出来的 P95 即可)
    th = 0.25
    
    generate_baseline_pseudo_labels(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        th=th
    )