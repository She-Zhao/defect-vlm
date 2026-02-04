## 缺陷检测VLM

### 整体目录结构
```
VLM-Defect-Inspection/        <-- Git 仓库根目录 (Root)
├── pyproject.toml            <-- 配置文件
├── uv.lock
├── README.md
├── .gitignore
│
├── defect_vlm/               <-- Python 源码包 (Package)
│   ├── __init__.py  
│   ├── data_preprocess/      <-- 数据预处理
│   │   ├── concat_image.py
│   │   ├── get_negative_bbox_from_gt.py
│   │   ├── get_positive_bbox_from_gt.py
│   │   ├── split_train_val_coco.py
│   │   ├── split_train_val_yolo.py
│   │   ├── bbox_converter
│   │   │   ├── csvmask2Rect.py
│   │   │   └── mask2Rect.py
│   │   ├── bbox_converter
│   │   │   ├── labelme2coco.py
│   │   │   ├── xmlme2coco.py
│   │   │   └── yolo2coco.py
│   │
│   ├── pe/              <-- 提示词工程相关
│   │   ├── save_prompt.py
│   │   ├── build_sft_dataset.py
│   │   ├── prompt_text.py
│   │   └── prompts.py
│   │
│   ├── sandbox/         <-- 一次性的实验脚本
│   │   └── ...
│   │
│   └──tools/            <-- 独立脚本如绘图、可视化等
│       └── ...
│
├── scripts/                  <-- 启动脚本
│   └── train.sh
│
├── label/
└── outputs/
```

### 数据集
```
defect_datasets_raw/        <-- 原始数据集图像
├── paint_stripe/           <-- 条纹光漆面数据集
│   ├── images                  # 一次性脚本
│   ├── labels_yolo
│   

defect_datasets_vlm/        <-- 用于微调VLM的数据集，只存储了region proposal
├── paint_stripe_gt_positive/           <-- 从标注的数据集里构造的正样本
├── paint_stripe_gt_negative/           <-- 从标注的数据集里构造的负样本
│  
│   
```

### TODO 已完成
- [x] 数据预处理脚本
    - [x] 将不同格式的标注转换为 COCO 格式的脚本
    - [x] 将数据集划分为训练集和验证集的脚本
    - [x] 将单通道灰度图拼接为三通道伪BGR图像的脚本
    - [x] 从标注的缺陷框中提取正样本的脚本
    - [ ] 从标注的缺陷框中提取负样本的脚本


