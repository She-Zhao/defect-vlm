"""
验证集上推理，调用官方的实现，和训练时候验证的效果完全一致
"""
import sys
# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

from ultralytics import YOLO

def generate_official_json():
    # 1. 加载你训练好的权重
    model = YOLO("/data/ZS/v11_input/weights/col3_part_max_cbam_iter0.pt")
    
    # 2. 直接调用 val 方法，开启 save_json
    # 这将 100% 复刻跑出 0.737 时的所有底层逻辑！
    print("🚀 正在使用官方评估逻辑生成高精度预测框...")
    model.val(
        data="/data/ZS/v11_input/ultralytics/cfg/datasets/paint/col3.yaml",
        conf=0.001,       # 极限阈值保召回
        iou=0.6,          # 官方验证阈值
        max_det=3000,     # 防止低分框被截断
        save_json=True,   # 【核心】强制保存 COCO 格式的 JSON
        plots=False       # 不需要画图，省点时间
    )

if __name__ == '__main__':
    generate_official_json()