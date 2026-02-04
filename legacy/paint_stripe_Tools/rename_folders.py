import os

# 将09 23 123改成0009 0023 0123
def rename_folders(base_folder):
    # 获取文件夹中的所有文件和子文件夹
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        # 检查是否是文件夹且文件夹名称为数字
        if os.path.isdir(folder_path) and folder_name.isdigit():
            # 生成新的四位数字名称，前面补零
            new_folder_name = folder_name.zfill(4)  # zfill(4) 将数字填充为四位
            
            # 如果文件夹名称需要更改
            if folder_name != new_folder_name:
                # 生成新的文件夹路径
                new_folder_path = os.path.join(base_folder, new_folder_name)
                
                # 重命名文件夹
                os.rename(folder_path, new_folder_path)
                print(f"Renamed '{folder_name}' to '{new_folder_name}'")
        else:
            # 如果是以字符开头的文件夹，跳过
            print(f"Skipping folder '{folder_name}' (not a valid number)")

# 示例调用
base_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera\brighten3\pmd_photo_enhanced_3'
rename_folders(base_folder)

base_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera\brighten3\stripe9_photo_enhanced_3'
rename_folders(base_folder)

base_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera\brighten3\stripe16_photo_enhanced_3'
rename_folders(base_folder)

base_folder = r'D:\Project\Defect_Dataset\PaintDatasets24\Camera\brighten3\stripe32_photo_enhanced_3'
rename_folders(base_folder)
