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

/data/ZS/defect_dataset/7_swift_dataset        <-- 转化为ms-swift格式的数据集，里面模型的回答都是教师模型的
/data/ZS/defect_dataset/8_model_reponse        <-- 调用模型进行推理得到的结果，里面模型的回答都是学生型的

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
    - [x] 修改/data/ZS/defect_dataset/4_api_request/test /data/ZS/defect_dataset/5_api_response/test /data/ZS/defect_dataset/backup/5_api_response 里面对于rectification项sample_type的名称
    - [x] 查看打标结果，看下错误的数量及原因。(原因：缺陷名称有的是Bulge，有的是bulge；回复的json字符串里面有的有转义字符，可以通过json.loads(strict=False)来解决该问题)
    - [x] 分析下重复打标的原因，将/data/ZS/defect_dataset/5_api_response/test里面打标的id进行矫正，并看是否和/data/ZS/defect_dataset/4_api_request/test有所重复。（012_pos200_neg150_rect150和012_pos400_neg300_rect300里面不是完全重复的，因为前者用的是random.shuffle，后者用的是random.sample）
    - [x] 撰写可视化脚本，根据打标得到的结果，对打标结果和对应图像进行可视化
    - [x] 编写评估脚本，对打标结果进行评估，绘制混淆矩阵，计算准确率、召回率、F1分数等指标
    - [x] 将gpt5.1、5.2的500条打标结果融入到test里面
    - [x] 测试gpt5.1 5.2 qwen3.5 以及8b模型的打标结果
    - [x] 用新的提示词生成对应的sft文件，并简单测试下效果（抽样一部分人工看一下）
    - [x] 整理下目前打标的这部分数据，作为第二章的对比实验
    - [x] 将数据集里面的none统一改成background, 从/data/ZS/defect_dataset/4_api_request/student/train开始
    - [x] 根据测试结果进行大规模打标
    - [x] 将打标结果进行分析，抽一部分看看效果
- [x] 训练脚本
    - [x] train/val/test数据集转化为ms-swift格式的脚本
    - [x] 读取ms-swift格式数据集中的数据，调用模型进行推理，并保存推理结果的脚本
    - [x] 评估脚本，绘制混淆矩阵，计算准确率、召回率、F1
    - [x] 评估下不同模型的效果
    - [x] 微调模型的代码
    - [x] 合并权重，并评估模型效果的代码

- [ ] 级联检测脚本
    - [ ] 设置数据切分的脚本，保持源数据不变，单独切出来一个未标注的数据
    - [ ] 恢复多流主干网络
    - [ ] 在验证集上进行推理yolo得到结果（存储为json格式的bbox即可）
    - [ ] 取原图像+json里面的bbox，送入VLM进行推理，要添加保存logits的方法
    - [ ] 将VLM的推理结果保存起来，同时保存logits
    - [ ] 撰写根据logits计算各项指标的代码