from PIL import Image, ImageEnhance
import cv2
import numpy as np

import os
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor, as_completed

# 增加亮度和对比度的函数
def increase_brightness_and_contrast(image_path, brightness_factor, contrast_factor, new_file_path):
    try:
        # 读取图像
        img = Image.open(image_path)

        # 使用 ImageEnhance 调整亮度
        enhancer_brightness = ImageEnhance.Brightness(img)
        bright_img = enhancer_brightness.enhance(brightness_factor)

        # 使用 ImageEnhance 调整对比度
        enhancer_contrast = ImageEnhance.Contrast(bright_img)
        contrast_img = enhancer_contrast.enhance(contrast_factor)

        # 保存增强后的图像
        contrast_img.save(new_file_path)
        print(f"已保存：{new_file_path}")
    except Exception as e:
        print(f"无法处理文件 {image_path}: {e}")

# 遍历文件夹及其子文件夹
def process_images_in_folder(root_folder, output_folder, brightness_factor, contrast_factor):
    # 创建输出根文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建线程池，提升性能
    with ThreadPoolExecutor() as executor:
        futures = []  # 存储所有任务

        # 遍历根目录下的所有文件夹
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)

            # 只处理文件夹，跳过以字母开头的文件夹
            if os.path.isdir(folder_path) and folder_name.isdigit():
                # 创建与源文件夹结构相同的新文件夹
                new_folder_path = os.path.join(output_folder, folder_name)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)

                # 遍历每个文件夹中的所有文件
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # 检查文件是否为图像文件（你可以根据需要调整文件类型）
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        # 新文件名，添加后缀以避免覆盖
                        new_file_name = file_name.split('.')[0] + '_enhanced.' + file_name.split('.')[1]
                        new_file_path = os.path.join(new_folder_path, new_file_name)

                        # 提交任务到线程池
                        futures.append(executor.submit(increase_brightness_and_contrast, file_path, brightness_factor, contrast_factor, new_file_path))

        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()

# 输入图像文件夹，对文件夹内的所有子文件夹内的图像进行增强
# 注意比如A下面有 a b c三个文件夹，输入的是A，会对abc文件夹内的图像进行增强
# 如果想对某个文件夹进行增强，将该文件夹放到一个父文件夹内，输入父文件夹即可，例如A/a，输入A即可对a内所有图像增强
root_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\test\cropped_stripe16'  # 输入图像文件夹路径
output_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\test\cropped_stripe16_brighten5'  # 输出文件夹路径
brightness_factor = 5  # 增加亮度的因子
contrast_factor = 1    # 增加对比度的因子

# 处理文件夹中的所有图像
process_images_in_folder(root_folder, output_folder, brightness_factor, contrast_factor)







# def increase_brightness_and_contrast(image_path, brightness_factor, contrast_factor):
#     # 读取图像
#     img = Image.open(image_path)

#     # 使用 ImageEnhance 调整亮度
#     enhancer_brightness = ImageEnhance.Brightness(img)
#     bright_img = enhancer_brightness.enhance(brightness_factor)

#     # 使用 ImageEnhance 调整对比度
#     enhancer_contrast = ImageEnhance.Contrast(bright_img)
#     contrast_img = enhancer_contrast.enhance(contrast_factor)

#     return contrast_img

# # 示例使用
# image_path = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera\stripe16_photo\0000\gc0.png'
# brightness_factor = 3  # 增加亮度的因子，1.0 是原始亮度
# contrast_factor = 1  # 增加对比度的因子，1.0 是原始对比度

# # 调整亮度和对比度
# result_img = increase_brightness_and_contrast(image_path, brightness_factor, contrast_factor)

# # 显示图像
# result_img.show()

