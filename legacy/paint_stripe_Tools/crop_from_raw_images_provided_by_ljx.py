"""
师姐提供的从原相机大图中裁剪300*300小图的脚本
"""
import os
import cv2

xkeys = [0,200,400,600,800,1000,1200,1400,1600,1800,2000,2148]
ykeys = [0,200,400,600,800,1000,1200,1400,1600,1748]
crop_size = 300

parent_dir = './output'
save_dir = './cropped'
cnt = 0

for folder in os.listdir(parent_dir):
    if folder[0] != 's':
        img_path = os.path.join(parent_dir,folder+'/Absolute_pha.png')
        if not os.path.exists(img_path):
            print(img_path)
        else:
            img = cv2.imread(img_path, 0)
            h,w = img.shape
            for y in ykeys:
                for x in xkeys:
                    crop_img = img[y:y+crop_size, x:x+crop_size]
                    cnt += 1
                    str_cnt = str(cnt)
                    while len(str_cnt) < 5:
                        str_cnt = '0' + str_cnt
                        
                    idx = folder
                    imgid = str_cnt
                    y_str = str(y)
                    x_str = str(x)
                    y_crop = str(crop_size)
                    x_crop = str(crop_size)
                    r = str(0)
                    h_str = str(h)
                    w_str = str(w)
                    crop_name = f'{idx}_{imgid}_{y_str}_{x_str}_{y_crop}_{x_crop}_{r}_{h_str}_{w_str}.png'
                    save_path = os.path.join(save_dir,crop_name)
                    cv2.imwrite(save_path, crop_img)