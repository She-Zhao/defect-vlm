import os

def rename_enhanced_images(base_dir):
    """
    去掉文件名中的 '_enhanced' 后缀的图像文件。

    :param base_dir: 存放图像的文件夹路径
    """
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if '_enhanced' in file_name:
                    old_file_path = os.path.join(folder_path, file_name)
                    new_file_name = file_name.replace('_enhanced', '')
                    new_file_path = os.path.join(folder_path, new_file_name)

                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed: {old_file_path} to {new_file_path}')

    print('Enhanced images renaming completed!')

import os

def rename_images(folder_path):
    """
    修改文件名，去掉 '_300_300' 并将 '2432_2048' 改为 '2048_2448'。

    :param folder_path: 存放图像的文件夹路径
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            # 分割文件名为各部分
            parts = file_name.split('_')
            
            # 确保文件名格式符合要求
            if len(parts) >= 9 and parts[-2] == '2432' and parts[-1].split('.')[0] == '2048':
                try:
                    # 去掉一个 '_300_300'
                    idx_300 = parts.index('300')  # 找到第一个 '300'
                    del parts[idx_300:idx_300+2]  # 删除连续两个 '300'

                    # 修改最后两个部分为 '2048' 和 '2448'
                    parts[-2] = '2048'
                    parts[-1] = '2448.png'  # 保留 .png 扩展名

                    # 构造新的文件名
                    new_file_name = '_'.join(parts)

                    # 构造完整路径
                    old_file_path = os.path.join(folder_path, file_name)
                    new_file_path = os.path.join(folder_path, new_file_name)

                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    # print(f'Renamed: {old_file_path} to {new_file_path}')
                except ValueError:
                    # 如果找不到 '_300_300'，跳过此文件
                    print(f"Skipped: {file_name} (no '_300_300' found)")
    print('Renaming completed!')

# 示例调用
folder_path = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera_single\stripe32_photo\cropped_sin3'  # 替换为实际文件夹路径
rename_images(folder_path)
