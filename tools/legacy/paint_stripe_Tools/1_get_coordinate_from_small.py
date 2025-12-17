from PIL import Image
import os

# 输入文件夹，文件夹内为图像名称，此函数将所有图像的名称保存到.txt中
def save_img_information(filefolder, output_txt):
    # 获取文件夹中的所有照片的位置信息
    files = os.listdir(filefolder)
    img_information_list = []
    # 打开txt文件（以写入模式，如果文件存在则覆盖）
    with open(output_txt, 'w') as f:
        for file in files:
            # 分割文件名并提取需要的信息
            filename_list = file.split('_')
            # temp = [filename_list[0]] + filename_list[2:6]
            img_information = '_'.join(filename_list[0:4])
            img_information_list.append(img_information)
            # 将 img_information 写入文本文件，每个信息占一行
            f.write(img_information + '\n')
            
    return img_information_list

# 使用示例
filefolder = r'D:\Project\Defect_Dataset\PaintDatasets24\small_high\cropped'  # 你存放文件的文件夹路径
output_txt = r'D:\Project\Defect_Dataset\PaintDatasets24\small_high\output.txt'  # 结果保存到的文本文件路径

# [00001_200_200_300_300, 00002_200_400_300_300, ...]
img_information_list = save_img_information(filefolder, output_txt)     