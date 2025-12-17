from pathlib import Path
import shutil
import json

# root_dir = Path(r'D:\Project\Defect_Dataset\AL\决赛_train_1011\单瑕疵图片')
# dst_dir = Path(r'D:\Project\Defect_Dataset\AL\决赛_train_1011\single_all_images')

# defect_dirs = [f for f in root_dir.iterdir() if f.is_dir()]

# for defect_dir in defect_dirs:
#     json_files = [f for f in defect_dir.glob('*.jpg')]
#     for json_file in json_files:
#         dst_path = dst_dir / json_file.name
        
#         shutil.copy2(json_file, dst_path)
#         print(f'Copied: {json_file.name}')
all_defects  = 0
data_dir = Path('data')
files = [f for f in data_dir.glob('*.json')]
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"处理 {file} 时遇到错误 {e}")
    all_defects += data['info']['total_defects']

print(all_defects)

