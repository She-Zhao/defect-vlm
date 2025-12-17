import os
import shutil

# 将    D:\Project\Multi_input\my_v11\datasets\PaintDatasets\01\camera\00\sin1.png
# 变成  D:\Project\Multi_input\my_v11\datasets\PaintDatasets\01\images_sin1\00.png

def move_img_to_newfolder(source_dir, target_dir, img_name):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历00-73的子文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 确保当前路径是文件夹
        if os.path.isdir(folder_path):
            # 查找sin1.png文件
            sin1_path = os.path.join(folder_path, img_name)
            if os.path.exists(sin1_path):
                # 目标文件路径（以子文件夹名称命名）
                target_file_path = os.path.join(target_dir, f"{folder_name}.png")

                # 复制文件到目标文件夹，并重命名
                shutil.copy(sin1_path, target_file_path)
                print(f"已复制 {sin1_path} 到 {target_file_path}")
            else:
                print(f"未找到 {folder_name} 下的 sin3.png")

img_list = ['gc0', 'gc1', 'gc2', 'gc3', 'gc4', 'sin0', 'sin1', 'sin2', 'sin3']
source_dir = r"D:\Project\Defect_Dataset\PaintDatasets24\Camera\Enhanced\stripe16_photo"  
target_dir = r"D:\Project\Defect_Dataset\PaintDatasets24\Camera_single\stripe16_photo"


for img_name in img_list:
    target_dir1 = os.path.join(target_dir,img_name)
    move_img_to_newfolder(source_dir, target_dir1, img_name+'.png')

