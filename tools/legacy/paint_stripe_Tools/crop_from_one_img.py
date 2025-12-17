import cv2
import cv2

def extract_cropped_region(new_image_path, original_y, original_x, crop_size, crop_save_path):
    """
    从新图像中截取指定区域并保存。

    :param new_image_path: 新图像的路径
    :param original_y: 裁剪区域的 y 坐标
    :param original_x: 裁剪区域的 x 坐标
    :param crop_size: 裁剪区域的大小（假设为正方形）
    :param crop_save_path: 保存截取区域的路径
    """
    # 读取新图像
    new_image = cv2.imread(new_image_path)

    # 检查图像是否成功读取
    if new_image is None:
        print(f'无法读取图像: {new_image_path}')
        return

    # 从新图像中截取对应区域
    cropped_region = new_image[original_y:original_y + crop_size, original_x:original_x + crop_size]

    # 保存截取的区域
    cv2.imwrite(crop_save_path, cropped_region)
    print(f'成功截取并保存区域到: {crop_save_path}')

# 示例调用
new_image_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera/Enhanced/stripe32_photo/0000/Absolute_pha.png"  # 替换为实际新图像路径
original_y = 200    # 从裁剪图像文件名提取的 y 坐标
original_x = 200    # 从裁剪图像文件名提取的 x 坐标
crop_size = 300     # 裁剪的尺寸
crop_save_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera/Enhanced/stripe32_photo/0000/cropped_image.png" 

# 调用函数
extract_cropped_region(new_image_path, original_y, original_x, crop_size, crop_save_path)
    
    
# # 使用示例
# input_image_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera/Enhanced/stripe32_photo/0000/Absolute_pha.png"  # 输入图像路径
# x = 400  # 左上角x坐标
# y = 200  # 左上角y坐标
# a = 300  # 截取区域的大小（宽度和高度都是a）
# output_image_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera/Enhanced/stripe32_photo/0000/cropped_image.png"  # 输出图像路径

# crop_image(input_image_path, x, y, a, output_image_path)
