"""
查看训练好的 yolo 模型的网络结构
"""

import sys
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) 保证最高优先级

from ultralytics import YOLO

model = YOLO("/data/ZS/v11_input/runs/train/exp_col3_part_max/weights/best.pt")

# 1. 打印详细的网络层级结构摘要（类似 PyTorch 的 torchsummary）
print("========== 网络层级摘要 ==========")
model.info(detailed=True)

# 2. 如果你想看原生的 PyTorch nn.Module 结构（每一层的具体算子，如 Conv2d, BatchNorm 等）
print("\n========== 原生 PyTorch 结构 ==========")
print(model.model)

# 3. 查看里面保存的类别字典
print("\n========== 类别字典 ==========")
print(model.names)