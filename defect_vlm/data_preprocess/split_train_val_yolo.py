"""
将yolo格式的数据集划分为训练集和测试集
"""
import os
import random
import shutil
random.seed(42)

labels_dir = './all_labels'
train_dir = './train_labels'
val_dir = './val_labels'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

all_labels = os.listdir(labels_dir)
random.shuffle(all_labels)

ratio = 0.2
all_labels_cnt = len(all_labels)
val_labels = all_labels[ : int(all_labels_cnt*ratio)]
train_labels = all_labels[int(all_labels_cnt*ratio): ]

for train_label in train_labels:
    src_train_label_path = os.path.join(labels_dir, train_label)
    dst_train_label_path = os.path.join(train_dir, train_label)
    shutil.copy2(src_train_label_path, dst_train_label_path)

for val_label in val_labels:
    src_val_label_path = os.path.join(labels_dir, val_label)
    dst_val_label_path = os.path.join(val_dir, val_label)
    shutil.copy2(src_val_label_path, dst_val_label_path)
