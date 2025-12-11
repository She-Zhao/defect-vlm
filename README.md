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

