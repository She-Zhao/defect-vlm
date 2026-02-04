import cv2
from pathlib import Path
import json
import numpy as np

json_path = "/data/ZS/defect_dataset/1_defect_dataset_bbox/paint_stripe_gt_positive/labels/val.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

item = data[4027]

# 计算坐标
x, y, w, h = item["bbox"]
y1, y2 = y, int(y+h)
x1, x2 = x, int(x+w)
pt1 = (x, y)
pt2 = (int(x+w), int(y+h))

# 读取图像饼绘制bbox
root_dir = Path('/data/ZS/defect_dataset')
original_image_path = root_dir / item["original_image_path"]
img = cv2.imread(str(original_image_path))
# import pdb; pdb.set_trace()
raw_image = img.copy()
save_img = cv2.rectangle(raw_image, pt1, pt2, color=(0, 0, 255), thickness=1, lineType=8)

# 保存原始图像
save_img_name = item['label']
save_img_path = f'/data/ZS/defect-vlm/example/{save_img_name}_1.png'
cv2.imwrite(save_img_path, save_img)

# 保存bbox的局部图像
bbox = img[y1:y2, x1:x2]
save_bbox_path = f'/data/ZS/defect-vlm/example/bbox_{save_img_name}_1.png'
cv2.imwrite(save_bbox_path, bbox)

