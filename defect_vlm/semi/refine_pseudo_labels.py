"""
基于双阈值 (th_l, th_h) ,对vlm的打标结果进行过滤
同时保存各部分的 Loss 权重: 半监督权重alpha(lambda)、类别级权重gamma_c、实例级权重beta 
"""
import json
from pathlib import Path

def generate_refined_pseudo_labels(
    input_jsonl,
    output_jsonl,
    th_l,
    th_h,
    alpha,
    eta,
    class_weights,
    class_method,
    class_args
):
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 开始生成精炼伪标签...")
    print(f"   [配置] th_l: {th_l}, th_h: {th_h}, eta: {eta}")
    print(f"   [类别权重] {class_weights}")
    
    # 决策漏斗统计器
    stats = {
        "total_input": 0,
        "discard_low_conf": 0,    # P < th_l 被抛弃
        "trust_yolo_high": 0,     # P >= th_h 直接信任 YOLO
        "vlm_discard_bg": 0,      # VLM 判为背景被抛弃
        "vlm_agreed": 0,          # VLM 赞同 YOLO
        "vlm_corrected": 0,       # VLM 纠正 YOLO
        "final_saved": 0
    }
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            data = json.loads(line)
            stats["total_input"] += 1
            
            # 提取关键字段
            p_yolo = data["confidence"]
            c_yolo = data["prior_label"]
            c_vlm = data["vlm_pred"]
            
            # 初始化决策变量
            final_label = None
            beta = 0.0
            decision_source = ""
            
            # ================= 核心决策树 =================
            if p_yolo < th_l:
                # 规则 1：直接抛弃底座噪点
                stats["discard_low_conf"] += 1
                continue
                
            elif p_yolo >= th_h:
                # 规则 2：完全信任底座小模型
                final_label = c_yolo
                beta = p_yolo
                decision_source = "YOLO_Trusted"
                stats["trust_yolo_high"] += 1
                
            else:
                # 规则 3：进入 VLM 决策域 (th_l <= p_yolo < th_h)
                if c_vlm == "background":                   # 如果VLM判定是背景，直接舍弃
                    stats["vlm_discard_bg"] += 1
                    continue
                elif c_yolo == c_vlm:                       # 如果VLM和YOLO判断结果一致，直接保留
                    final_label = c_yolo
                    beta = p_yolo
                    decision_source = "VLM_Agreed"
                    stats["vlm_agreed"] += 1
                else:                                       # 如果VLM纠正了YOLO，需要给beta乘上一个惩罚系数。
                    final_label = c_vlm
                    beta = p_yolo * eta         
                    decision_source = "VLM_Corrected"
                    stats["vlm_corrected"] += 1
            # ==============================================
            
            # 获取类别宏观权重 (gamma)，如果字典里没写，容错默认为 1.0
            gamma = class_weights[final_label]
            
            # 计算最终的联合权重
            final_weight = alpha * beta * gamma
            
            # 组装全新的纯净结构
            refined_item = {
                "id": data["id"],
                "images": data["images"],
                "bbox": data["bbox"],
                "model_source": data["model_source"],  # 保留是为了溯源
                "final_label": final_label,
                'alpha_weight': round(alpha, 2),
                "beta_weight": round(beta, 2),
                "gamma_weight": round(gamma, 2),
                "final_weight": round(final_weight, 4),
                "decision_source": decision_source     # 记录这打标是怎么来的，方便分析
            }
            
            fout.write(json.dumps(refined_item, ensure_ascii=False) + '\n')
            stats["final_saved"] += 1

    meta_info = {
        "hyperparameters": {
            "th_l": th_l,
            "th_h": th_h,
            "eta": eta,
            "alpha": alpha,
            "class_weights": class_weights,
            "class_method": class_method,
            "class_args": class_args 
        },
        "statistics": stats
    }
    
    # 自动生成同名的 meta 文件名，例如：sp012_refined_meta.json
    meta_path = output_path.parent / f"{output_path.stem}_meta.json"
    
    with open(meta_path, 'w', encoding='utf-8') as fmeta:
        json.dump(meta_info, fmeta, ensure_ascii=False, indent=4)
    # ================================================================

    # 打印超级直观的决策漏斗报告
    print("\n" + "=" * 50)
    print("🎯 伪标签决策漏斗统计报告")
    print("=" * 50)
    print(f"📥 初始输入框总数 : {stats['total_input']}")
    print(f"  ├── 🗑️  [过滤] 因置信度过低抛弃 : {stats['discard_low_conf']}")
    print(f"  ├── 🗑️  [过滤] VLM 判定为背景抛弃 : {stats['vlm_discard_bg']}")
    print(f"  │")
    print(f"  ├── 🟢  [保留] YOLO 高置信度直通 : {stats['trust_yolo_high']}")
    print(f"  ├── 🟢  [保留] VLM 验证一致通过   : {stats['vlm_agreed']}")
    print(f"  └── 🟠  [保留] VLM 语义纠正通过   : {stats['vlm_corrected']} (打 {eta} 折扣)")
    print("-" * 50)
    print(f"💾 最终落盘保存伪标签数 : {stats['final_saved']}")
    print(f"   保留率 : {(stats['final_saved'] / stats['total_input']) * 100:.2f}%")
    print("=" * 50)
    print(f"✅ 伪标签已保存至: {output_path}")
    print(f"📄 元数据与配置已保存至: {meta_path}")

if __name__ == '__main__':
    # 1. 路径配置
    input_jsonl = "/data/ZS/flywheel_dataset/7_vlm_extracted_data/iter0/sp012_0p1_v1_LM.jsonl"   # 仅提取了vlm的判断结果，没做任何筛选
    output_jsonl = "/data/ZS/flywheel_dataset/8_pseudo_labels/iter0/sp012_0p1_v1_LM_1.jsonl"
    
    # 2. 超参数配置 (这些需要根据你上一个脚本测出来的值来填)
    th_l = 0.1     # R72
    th_h = 0.83     # P95
    alpha = 1.0     # 半监督的权重系数 
    eta = 0.5       # VLM 纠正时的 β 惩罚系数
    
    # 3. 类别权重配置 (由 semi/compute_cls_weight.py 得到，对应γ_c)
    class_weights = {
        "breakage": 0.79,
        "inclusion": 0.64,
        "scratch": 1.09,
        "crater": 1.14,
        "run": 1.34,
        "bulge": 1.01
    }
    class_method = 'power'          # 计算 class_weights 的方法
    class_args = 2                  # 计算 class_weights 的方法对应的参数
    
    generate_refined_pseudo_labels(
        input_jsonl = input_jsonl,
        output_jsonl = output_jsonl,
        th_l = th_l,
        th_h = th_h,
        alpha = alpha,
        eta = eta,
        class_weights = class_weights,
        class_method = class_method,
        class_args = class_args
    )
