"""
模型 EMA 权重融合脚本 (Teacher-Student EMA Fusion)
用于将上一代 Teacher 模型与本代训练出的 Student 模型进行参数级融合。
"""
import torch
from pathlib import Path
import sys
PROJECT_ROOT = '/data/ZS/v11_input'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def apply_ema_fusion(teacher_ckpt_path, student_ckpt_path, output_ckpt_path, alpha=0.5):
    """
    执行 EMA 权重融合
    :param teacher_ckpt_path: 上一代 Teacher 模型路径 (例如 yolo_gt.pt)
    :param student_ckpt_path: 刚刚训练好的 Student 模型路径 (例如 yolo_iter1.pt)
    :param output_ckpt_path: 融合后的输出路径 (例如 teacher_iter1.pt 或 iter1_ema.pt)
    :param alpha: EMA 动量参数 (保留多少上一代 Teacher 的知识)
    """
    print("=" * 60)
    print("🔄 开始执行模型 EMA 权重融合...")
    print(f"👨‍🏫 载入 Teacher (t-1): {teacher_ckpt_path}")
    print(f"🎓 载入 Student (t)  : {student_ckpt_path}")
    
    # 1. 加载模型 (映射到 CPU 防止显存爆炸)
    ckpt_t = torch.load(teacher_ckpt_path, map_location='cpu')
    ckpt_s = torch.load(student_ckpt_path, map_location='cpu')
    
    # 2. 提取真正的模型对象 (YOLO 官方通常把最优权重放在 'ema' 或 'model' 键值下)
    model_t = ckpt_t.get('ema') or ckpt_t['model']
    model_s = ckpt_s.get('ema') or ckpt_s['model']
    
    state_dict_t = model_t.state_dict()
    state_dict_s = model_s.state_dict()
    
    # 3. 逐层进行 EMA 加权求和
    print(f"⚙️  正在进行参数空间融合 (alpha = {alpha})...")
    fused_layers = 0
    
    with torch.no_grad():
        for k, v_t in state_dict_t.items():
            if k in state_dict_s:
                v_s = state_dict_s[k]
                
                # 统一转为 FP32 进行高精度计算，防止 FP16 溢出
                if v_t.dtype != torch.float32:
                    v_t = v_t.float()
                if v_s.dtype != torch.float32:
                    v_s = v_s.float()
                
                # 核心公式: Teacher_new = alpha * Teacher_old + (1 - alpha) * Student
                fused_v = alpha * v_t + (1.0 - alpha) * v_s
                
                # 将融合后的权重覆盖回 Teacher 的状态字典，并恢复原有数据类型 (通常是 FP16)
                state_dict_t[k].copy_(fused_v.to(state_dict_t[k].dtype))
                fused_layers += 1
                
    # 4. 清理冗余数据，瘦身打包 (剔除 optimizer，因为新 Teacher 只用于推理或作为下一轮的起点)
    ckpt_t['model'] = model_t
    ckpt_t['ema'] = None  
    if 'optimizer' in ckpt_t:
        ckpt_t['optimizer'] = None
        
    # 5. 保存新生代 Teacher 模型
    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(ckpt_t, output_path)
    
    print("-" * 60)
    print(f"✅ 融合完成！共处理了 {fused_layers} 个网络层。")
    print(f"💾 新代模型已保存至: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    # ================= 配置区 =================
    # 1. 上一回合的模型
    TEACHER_WEIGHT = "/data/ZS/defect-vlm/output/yolo_weights/iter1_row3_0p1_ema0p01.pt" 
    
    # 2. 你刚刚用伪标签训练出来的模型
    STUDENT_WEIGHT = "/data/ZS/defect-vlm/output/yolo_weights/iter2_row3_0p1.pt"
    
    # 3. 融合后的输出路径
    OUTPUT_WEIGHT = "/data/ZS/defect-vlm/output/yolo_weights/temp/iter2_row3_0p1_ema0p01.pt"
    
    # 4. EMA 动量因子 (关键超参数！)
    ALPHA = 0.01
    # ==========================================
    
    apply_ema_fusion(
        teacher_ckpt_path=TEACHER_WEIGHT,
        student_ckpt_path=STUDENT_WEIGHT,
        output_ckpt_path=OUTPUT_WEIGHT,
        alpha=ALPHA
    )