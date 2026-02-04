import os

defects_map = {
    '0': 'breakage',
    '1': 'inclusion',
    '2': 'scratch',
    '3': 'crater',
    '4': 'run',
    '5': 'bulge',
    '6': 'pinhole',
    '7': 'condensate'
}

def analyse_labels(labels_dir):
    defects_count = {}
    for file_name in os.listdir(labels_dir):
        if file_name == 'classes.txt':
            continue
        label_path = os.path.join(labels_dir, file_name)
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:

            c, x, y, w, h = line.strip().split(' ')
            defects_count[defects_map[c]] = defects_count.get(defects_map[c], 0) + 1

    print(defects_count)

# labels_dir = r'D:\Project\Defect_Dataset\paint_stripe\My_datasets\all_labels'
train_dir = r'D:\Project\Defect_Dataset\paint_stripe\增强后的所有300300大小图像\train_labels'
val_dir = r'D:\Project\Defect_Dataset\paint_stripe\增强后的所有300300大小图像\val_labels'

# analyse_labels(labels_dir)
analyse_labels(train_dir)
analyse_labels(val_dir)
