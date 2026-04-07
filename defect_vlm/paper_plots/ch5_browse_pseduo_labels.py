"""
可视化迭代后的模型生成的伪标签
"""
import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import math

# ==========================================
# 1. 页面与全局配置
# ==========================================
st.set_page_config(layout="wide", page_title="Iter 3 伪标签可视化")

CLASS_MAP = {0: '破损', 1: '漆渣', 2: '划痕', 3: '凹陷', 4: '流挂', 5: '凸起'}
COLOR_MAP = {
    0: '#E64B35', # 玫瑰红
    1: '#4DBBD5', # 克莱因蓝
    2: '#00A087', # 翠绿
    3: '#F39B7F', # 珊瑚橘
    4: '#8491B4', # 灰蓝
    5: '#91D1C2'  # 浅青
}

# ==========================================
# 2. 路径配置区
# ==========================================
# Iter 3 的训练集图片与标签路径
ITER3_BASE = '/data/ZS/flywheel_dataset/9_semi_yolo_dataset/iter3_10/row3/train'
IMG_DIR = os.path.join(ITER3_BASE, 'images')
LBL_DIR = os.path.join(ITER3_BASE, 'labels')

# GT 真值路径 (用于过滤，此处已修正为 train/labels 以匹配上述的 train 目录)
GT_LBL_DIR = '/data/ZS/v11_input/datasets/row3/train/labels'

# 中文字体路径 (防止乱码方块)
ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc' 

# ==========================================
# 3. 数据加载与过滤 (提取纯伪标签)
# ==========================================
@st.cache_data
def get_pseudo_image_ids():
    if not os.path.exists(LBL_DIR):
        return []
        
    all_labels = set([f for f in os.listdir(LBL_DIR) if f.endswith('.txt')])
    
    # 获取 GT 文件列表进行过滤
    gt_labels = set()
    if os.path.exists(GT_LBL_DIR):
        gt_labels = set([f for f in os.listdir(GT_LBL_DIR) if f.endswith('.txt')])
        
    # 纯伪标签 = 所有标签 - GT标签
    pseudo_labels = list(all_labels - gt_labels)
    
    # 提取无后缀的 image_id
    image_ids = sorted([os.path.splitext(f)[0] for f in pseudo_labels])
    return image_ids

pseudo_image_ids = get_pseudo_image_ids()

if not pseudo_image_ids:
    st.error("没有找到伪标签数据，请检查路径配置！")
    st.stop()

# ==========================================
# 4. 绘图函数 (解析 YOLO 归一化坐标)
# ==========================================
def draw_yolo_pseudo_labels(img_path, lbl_path, threshold):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None
        
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(ZH_FONT_PATH, 20)
    except IOError:
        font = ImageFont.load_default()

    has_valid_boxes = False

    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cat_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # 伪标签通常有第6列作为置信度
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    
                    if conf < threshold:
                        continue
                        
                    has_valid_boxes = True
                    
                    # 反归一化为绝对坐标
                    w_abs = w * img_w
                    h_abs = h * img_h
                    x_min = (xc - w / 2) * img_w
                    y_min = (yc - h / 2) * img_h
                    
                    cat_name = CLASS_MAP.get(cat_id, str(cat_id))
                    color = COLOR_MAP.get(cat_id, '#FFFFFF')
                    
                    # 画框
                    draw.rectangle([x_min, y_min, x_min+w_abs, y_min+h_abs], outline=color, width=3)
                    
                    # 画文字
                    label_text = f"{cat_name} {conf:.2f}"
                    text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
                    draw.rectangle([x_min, y_min - (text_bbox[3] - text_bbox[1]), 
                                  x_min + (text_bbox[2] - text_bbox[0]), y_min], fill=color)
                    draw.text((x_min, y_min - (text_bbox[3] - text_bbox[1])), label_text, fill="white", font=font)
                    
    return img, has_valid_boxes

# ==========================================
# 5. UI 与分页逻辑
# ==========================================
st.sidebar.title("⚙️ 伪标签可视化控制台")
st.sidebar.markdown(f"**总计过滤出伪标签图像:** `{len(pseudo_image_ids)}` 张")

conf_threshold = st.sidebar.slider("🎚️ 置信度过滤阈值", 0.0, 1.0, 0.40, 0.05)

IMAGES_PER_PAGE = 20  # 每页显示 20 张 (4行5列)
total_pages = math.ceil(len(pseudo_image_ids) / IMAGES_PER_PAGE)

if 'page' not in st.session_state:
    st.session_state.page = 1

col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col1:
    if st.button("⬅️") and st.session_state.page > 1:
        st.session_state.page -= 1
with col2:
    st.markdown(f"<div style='text-align: center; margin-top: 8px;'>第 {st.session_state.page} / {total_pages} 页</div>", unsafe_allow_html=True)
with col3:
    if st.button("➡️") and st.session_state.page < total_pages:
        st.session_state.page += 1

st.sidebar.markdown("---")
# 仅展示有框的图像选项
only_show_objects = st.sidebar.checkbox("👀 仅显示包含缺陷(大于阈值)的图像", value=True)

# ==========================================
# 6. 主区域网格渲染 (一行 5 列)
# ==========================================
st.title("动态数据飞轮 Iter 3 伪标签生成效果")

# 计算当前页的数据
start_idx = (st.session_state.page - 1) * IMAGES_PER_PAGE
end_idx = start_idx + IMAGES_PER_PAGE
current_ids = pseudo_image_ids[start_idx:end_idx]

# 构建一行5列的网格
cols_per_row = 5

# 遍历当前页的图像
current_row_cols = st.columns(cols_per_row)
col_idx = 0

for img_id in current_ids:
    img_path_jpg = os.path.join(IMG_DIR, f"{img_id}.jpg")
    img_path_png = os.path.join(IMG_DIR, f"{img_id}.png")
    img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
    
    lbl_path = os.path.join(LBL_DIR, f"{img_id}.txt")
    
    if os.path.exists(img_path):
        result_img, has_boxes = draw_yolo_pseudo_labels(img_path, lbl_path, conf_threshold)
        
        if result_img:
            # 如果勾选了"仅显示有框图像"且该图没框，则跳过
            if only_show_objects and not has_boxes:
                continue
                
            # 在对应的列中绘制图像
            with current_row_cols[col_idx]:
                st.image(result_img, use_container_width=True)
                st.caption(f"{img_id}")
            
            # 列索引+1，如果达到 5 列，就换行重新创建 5 列
            col_idx += 1
            if col_idx >= cols_per_row:
                col_idx = 0
                current_row_cols = st.columns(cols_per_row)

if only_show_objects:
    st.info("💡 当前已开启【过滤空图像】模式。如果你发现有些行没有填满 5 张图，是因为该页存在低于置信度阈值的背景图被隐藏了。翻页即可查看更多！")
    