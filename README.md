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
│   ├── core/                 <-- 核心训练代码
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── ...
│   ├── modules/              <-- 组件
│   │   ├── __init__.py
│   │   └── ...
│   └── utils/                <-- 一些通用函数，比如坐标转换函数
│       └── __init__.py
│
├── tools/                    <-- 独立脚本保持在外
│   ├── convert2coco.py       # 一次性脚本
│   └── visualization.py
│
├── scripts/                  <-- 启动脚本
│   └── train.sh
│
├── data/
└── outputs/
```

### 数据集
defect_datasets_raw/        <-- 原始数据集图像
├──  
├── 
├── 
├── 
├── paint_stripe/           <-- 条纹光漆面数据集
│   ├── images                  # 一次性脚本
│   ├── labels_yolo
│   

defect_datasets_vlm/        <-- 用于微调VLM的数据集，只存储了region proposal
├── paint_stripe_gt_positive/           <-- 从标注的数据集里构造的正样本
├── paint_stripe_gt_negative/           <-- 从标注的数据集里构造的负样本
│  
│   

### TODO
- [ ]

