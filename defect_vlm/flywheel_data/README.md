## 功能说明
该目录下的脚本主要用于构建半监督和飞轮数据。逻辑和 `/data/ZS/defect-vlm/defect_vlm/data_preprocess` 基本一致，区别是操作对象从真实数据变成了YOLO预测结果。

包括批量推断预测概率、构建VLM消息、从YOLO预测中生成复合图像、裁剪YOLO预测的边界框、决策融合、提取未标记数据以及非极大值抑制融合等功能。

/data/ZS/defect-vlm/defect_vlm/flywheel_data
├── batch_infer_preds_probs.py
├── build_vlm_message.py
├── composite_images_from_yolo_preds.py
├── crop_yolo_preds_bbox.py
├── decision_fusion.py
├── extract_unlabeled_data.py
├── nms_fusion.py
└── README.md