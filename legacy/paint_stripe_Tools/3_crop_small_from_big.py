import os
from PIL import Image

# 根据txt，从大图中裁剪小图，将小图保存到一个新的文件夹中
# 输入txt路径、大图所在文件夹、小图拟保存的文件夹
def extract_and_save_images(txt_file, images_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取txt文件中的坐标信息
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    cnt=0
    for line in lines:
        # 去除行末的空白字符
        line = line.strip()
        if line:
            # 解析每一行的内容，例如 00_1_1050_210
            parts = line.split('_')
            img_name = parts[0]  # 00 作为图像名
            y1, x1 = int(parts[2]), int(parts[3])  # 左上角坐标 (x1, y1)
            
            # 计算右下角坐标，假设每个小图的尺寸为 300x300
            x2, y2 = x1 + 300, y1 + 300
            
            # 构建对应的大图文件路径
            img_file_path = os.path.join(images_folder, f"{img_name}.png")
            
            # 打开大图
            try:
                img = Image.open(img_file_path)
            except FileNotFoundError:
                print(f"文件 {img_file_path} 未找到，跳过...")
                continue
            
            # 截取小图
            small_img = img.crop((x1, y1, x2, y2))
            
            # 生成小图的文件名
            # small_img_name = f"{img_name}_{x1}_{y1}_{x2 - x1}_{y2 - y1}_0_2432_2048.png"
            small_img_name = f"{line}_0_2048_2448.png"
            small_img_path = os.path.join(output_folder, small_img_name)
            
            # 保存小图
            small_img.save(small_img_path)
            
            cnt+=1
            if cnt%10==0:
                print(f"保存小图 {small_img_path}")

# 文件路径
txt_file = r"D:\Project\Defect_Dataset\PaintDatasets24\small_high_coordinate84.txt"
img_list = ['gc0', 'gc1', 'gc2', 'gc3', 'gc4', 'sin0', 'sin1', 'sin2', 'sin3']
# img_list = ['gc4', 'sin0', 'sin1', 'sin2', 'sin3']

for img_name in img_list:
    root_path = r"D:\Project\Defect_Dataset\PaintDatasets24\Camera_single\pmd_photo"
    images_folder = os.path.join(root_path, img_name)
    output_folder = os.path.join(root_path, "cropped_" + img_name)
    extract_and_save_images(txt_file, images_folder, output_folder)

