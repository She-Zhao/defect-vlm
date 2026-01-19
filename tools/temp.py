from pathlib import Path
import shutil
import json
from tqdm import tqdm
import PIL

# save_dir = Path(r'D:\Project\Defect_Dataset\al_final\new_labels')
# labels_dir = Path(r'D:\Project\Defect_Dataset\al_final\labels')
# labels = list(labels_dir.glob('*.json'))

# save_dir.mkdir(exist_ok=True, parents=True)

# for label in tqdm(labels, desc='Processing'):
#     try:
#         with open(label, 'r', encoding='utf-8') as f:
#             data = json.load(f)    
#         data.pop('imageData', None)
#         save_path = save_dir / label.name
#         with open(save_path, 'w', encoding='utf-8') as f: 
#             json.dump(data, f, indent=2, ensure_ascii=False)
#     except Exception as e:
#         print(f"处理 {label} 时发生错误: {e}")

