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

### 数据预处理流程
1. 将原始数据集中的图像和标注转换为 COCO 格式
```
defect-vlm/defect_vlm/data_preprocess/label_converter
```

2. 将数据集划分为训练集和验证集
```
defect-vlm/defect_vlm/data_preprocess/split_train_val_coco.py
```

3. 将单通道灰度图拼接为三通道伪BGR图像
```
defect-vlm/defect_vlm/data_preprocess/concat_image.py
```

4. 从标注的缺陷框中提取正样本
```
defect-vlm/defect_vlm/data_preprocess/get_positive_bbox_from_gt.py
```

5. 从标注的缺陷框中提取两类负样本
```
defect-vlm/defect_vlm/data_preprocess/get_negative_bbox_from_gt.py

defect-vlm/defect_vlm/data_preprocess/get_rectification_bbox_from_gt.py
```

6. 拼接获得2*2全局图像和2*2的局部图像
```
defect-vlm/defect_vlm/data_preprocess/composite_images_from_gt.py
```

### 提示词工程流程

### TODO 已完成
- [x] 数据预处理脚本
    - [x] 将不同格式的标注转换为 COCO 格式的脚本
    - [x] 将数据集划分为训练集和验证集的脚本
    - [x] 将单通道灰度图拼接为三通道伪BGR图像的脚本
    - [x] 从标注的缺陷框中提取正样本的脚本
    - [x] 从标注的缺陷框中提取两类负样本的脚本
- [x] 提示词工程脚本
    - [x] 撰写多图推理的提示词
    - [x] 将多图推理的提示词和数据集构成调用API的SFT格式数据集
    - [x] 编写调用api进行打标的脚本，打标输入放在4_api_request目录下，打标结果放在5_api_response目录下
    - [x] 编写对打标结果进行过滤、筛选其中错误样本的脚本
        - [x] 对api返回结果进行解析，并判断是否符合要求
        - [x] 将模型的返回结果分成两部分，没问题的直接保存到6_sft_dataset，不符合要求的样本，保存到4_api_request/retry
        - [x] 对于4_api_request/retry中的样本，重新调用打标脚本，生成新的结果，保存到5_api_response/retry中
        - [x] 重复上述流程，直到没有错误样本为止
    - [x] 从val里面抽取一部分数据，用于选择教师模型
    - [x] 调用api进行打标
    - [ ] 编写评估脚本，对打标结果进行评估，绘制混淆矩阵，计算准确率、召回率、F1分数等指标
    - [ ] 测试不同模型效果（教师模型和学生模型）
    - [ ] 撰写可视化脚本，根据打标得到的结果，对打标结果和对应图像进行可视化


