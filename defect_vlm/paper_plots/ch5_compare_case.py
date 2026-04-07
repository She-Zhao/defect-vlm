"""
可视化不同模型在验证集上的检测效果
"""
import streamlit as st
import json
import os
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# ==========================================
# 1. 页面与全局配置
# ==========================================
st.set_page_config(layout="wide", page_title="数据飞轮 Case 对比工具")
ZH_FONT_PATH = '/data/ZS/defect-vlm/defect_vlm/paper_plots/fonts/simsun.ttc'

CLASS_MAP = {0: '破损', 1: '漆渣', 2: '划痕', 3: '凹陷', 4: '流挂', 5: '凸起'}
# 顶会风格的类别配色 (Hex)
COLOR_MAP = {
    0: '#E64B35', # 玫瑰红 (破损)
    1: '#4DBBD5', # 克莱因蓝 (漆渣)
    2: '#00A087', # 翠绿 (划痕)
    3: '#F39B7F', # 珊瑚橘 (凹陷)
    4: '#8491B4', # 灰蓝 (流挂)
    5: '#91D1C2'  # 浅青 (凸起)
}

# 路径配置
IMG_DIR = '/data/ZS/v11_input/datasets/row3/val/images'
GT_LABEL_DIR = '/data/ZS/v11_input/datasets/row3/val/labels' # 新增：GT YOLO txt 目录

# JSON 预测路径 (移除了 GT)
JSON_PATHS = {
    "Baseline (仅 GT 训练)": "/data/ZS/defect-vlm/output/yolo_decect/gt_row3/predictions.json",
    "静态半监督 (Iter 0)": "/data/ZS/defect-vlm/output/yolo_decect/iter0_row3/predictions.json",
    "动态迭代 (Iter 1)": "/data/ZS/defect-vlm/output/yolo_decect/iter1_row3_ema0p01/predictions.json",
    "动态迭代 (Iter 2)": "/data/ZS/defect-vlm/output/yolo_decect/iter2_1_row3_ema0p01/predictions.json",
    "动态迭代 (Iter 3)": "/data/ZS/defect-vlm/output/yolo_decect/iter3_10_row3_ema0p01/predictions.json"
}

# 用于下拉菜单展示的所有模型列
ALL_MODELS = ["GT (真实标签)"] + list(JSON_PATHS.keys())

# ==========================================
# 2. 数据加载 (缓存 JSON 预测与扫描图像 ID)
# ==========================================
@st.cache_data
def load_predictions():
    model_preds = {}
    valid_image_ids = set()
    
    # 1. 解析 JSON 预测文件
    for model_name, path in JSON_PATHS.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                preds = json.load(f)
            
            img_dict = defaultdict(list)
            for p in preds:
                img_dict[p['image_id']].append(p)
                valid_image_ids.add(p['image_id'])
            
            model_preds[model_name] = img_dict
        else:
            st.sidebar.warning(f"⚠️ 文件未找到: {path}")
            
    # 2. 扫描 GT txt 文件，确保那些没有被预测出来的图像也能被浏览
    if os.path.exists(GT_LABEL_DIR):
        for f in os.listdir(GT_LABEL_DIR):
            if f.endswith('.txt'):
                valid_image_ids.add(os.path.splitext(f)[0])
                
    return model_preds, sorted(list(valid_image_ids))

model_preds, all_image_ids = load_predictions()

if not all_image_ids:
    st.error("没有找到任何预测数据或真值数据，请检查路径是否正确！")
    st.stop()

# ==========================================
# 3. 会话状态管理 (翻页与下拉框同步)
# ==========================================
if 'img_idx' not in st.session_state:
    st.session_state.img_idx = 0
if 'img_selector' not in st.session_state:
    st.session_state.img_selector = all_image_ids[0]

def prev_img():
    if st.session_state.img_idx > 0:
        st.session_state.img_idx -= 1
        st.session_state.img_selector = all_image_ids[st.session_state.img_idx]

def next_img():
    if st.session_state.img_idx < len(all_image_ids) - 1:
        st.session_state.img_idx += 1
        st.session_state.img_selector = all_image_ids[st.session_state.img_idx]

def update_idx_from_select():
    selected = st.session_state.img_selector
    st.session_state.img_idx = all_image_ids.index(selected)

# ==========================================
# 4. 侧边栏 UI 设置
# ==========================================
st.sidebar.title("⚙️ 控制面板")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("⬅️ 上一张", on_click=prev_img, use_container_width=True)
with col2:
    st.button("下一张 ➡️", on_click=next_img, use_container_width=True)

selected_img_id = st.sidebar.selectbox(
    "🖼️ 选择图像 (Image ID)", 
    options=all_image_ids, 
    key='img_selector',
    on_change=update_idx_from_select
)

st.sidebar.markdown("---")

img_path = os.path.join(IMG_DIR, f"{selected_img_id}.jpg")
if not os.path.exists(img_path):
    img_path = os.path.join(IMG_DIR, f"{selected_img_id}.png")

conf_threshold = st.sidebar.slider("🎚️ 置信度过滤阈值 (Confidence)", 0.0, 1.0, 0.25, 0.01)

selected_models = st.sidebar.multiselect(
    "📊 选择要对比的模型", 
    options=ALL_MODELS, 
    default=["GT (真实标签)", "Baseline (仅 GT 训练)", "静态半监督 (Iter 0)", "动态迭代 (Iter 3)"]
)

# ==========================================
# 5. 绘图核心函数 (新增 YOLO TXT 实时解析)
# ==========================================
def draw_boxes(image_path, predictions, threshold, is_gt=False, gt_txt_path=None):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(ZH_FONT_PATH, 20) 
    except IOError:
        st.warning("⚠️ 未找到中文字体，文字可能显示为方块！")
        font = ImageFont.load_default()
        
    # 如果是 GT，实时解析归一化的 YOLO txt 并转化为绝对坐标
    if is_gt and gt_txt_path and os.path.exists(gt_txt_path):
        img_w, img_h = img.size
        predictions = []
        with open(gt_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cat_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # 反归一化：将中心点和宽高转化为左上角坐标和绝对宽高
                    w_abs = w * img_w
                    h_abs = h * img_h
                    x_min = (xc - w / 2) * img_w
                    y_min = (yc - h / 2) * img_h
                    
                    predictions.append({
                        'category_id': cat_id,
                        'bbox': [x_min, y_min, w_abs, h_abs]
                        # GT 不含 score 字段
                    })

    for p in predictions:
        score = p.get('score', 1.0)
        # 只有存在 score 字段时才卡阈值 (保护 GT 数据不被过滤)
        if 'score' in p and score < threshold:
            continue
            
        cat_id = p['category_id']
        cat_name = CLASS_MAP.get(cat_id, str(cat_id))
        color = COLOR_MAP.get(cat_id, 'red')
        
        x, y, w, h = p['bbox']
        
        draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
        
        label_text = f"{cat_name} {score:.2f}" if 'score' in p else cat_name
        
        text_bbox = draw.textbbox((x, y), label_text, font=font)
        draw.rectangle([x, y - (text_bbox[3] - text_bbox[1]), x + (text_bbox[2] - text_bbox[0]), y], fill=color)
        draw.text((x, y - (text_bbox[3] - text_bbox[1])), label_text, fill="white", font=font)
        
    return img

# ==========================================
# 6. 主区域渲染
# ==========================================
st.title("视觉缺陷检测 Case 对比看板")
st.markdown(f"**当前图像:** `{selected_img_id}` ({st.session_state.img_idx + 1} / {len(all_image_ids)})")

if not os.path.exists(img_path):
    st.error(f"找不到图像文件: {img_path}")
else:
    if selected_models:
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            with col:
                st.subheader(model_name)
                
                # 区分 GT 的加载逻辑与 JSON 的加载逻辑
                if model_name == "GT (真实标签)":
                    txt_path = os.path.join(GT_LABEL_DIR, f"{selected_img_id}.txt")
                    result_img = draw_boxes(img_path, [], conf_threshold, is_gt=True, gt_txt_path=txt_path)
                else:
                    preds = model_preds.get(model_name, {}).get(selected_img_id, [])
                    result_img = draw_boxes(img_path, preds, conf_threshold, is_gt=False)
                
                if result_img:
                    st.image(result_img, use_container_width=True)
                else:
                    st.write("图像加载失败")