"""
输入模型回复的jsonl文件(例如 /data/ZS/defect_dataset/5_api_response/test/gemini-3-pro-preview.jsonl)
通过Streamlit进行可视化:
- stream run browse_api_response.py
- 点击出现的下面出现的 `Local URL: http://localhost:8501` 即可
"""

import streamlit as st
import json
import os

# 页面配置设置，使用宽屏模式查看图片更爽
st.set_page_config(layout="wide", page_title="Defect Data Viewer")

@st.cache_data
def load_jsonl(file_path):
    """读取并缓存数据，避免每次拖动进度条都重新读取文件"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

# --- 侧边栏：配置与导航 ---
st.sidebar.title("数据导航")
file_path = st.sidebar.text_input("1. 输入 JSONL 绝对/相对路径:", "your_data.jsonl")

data = load_jsonl(file_path)

if not data:
    st.info("👆 请在左侧输入正确的文件路径以加载数据。")
    st.stop()

# --- 引入 Session State 来管理当前样本索引 ---
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

st.sidebar.markdown("---")
st.sidebar.subheader("2. 样本导航")

# --- 添加上一条/下一条按钮 ---
col1, col2 = st.sidebar.columns(2)
with col1:
    # 点击上一条，且当前不是第一条时，索引减 1
    if st.button("⬅️ 上一条"):
        if st.session_state.sample_idx > 0:
            st.session_state.sample_idx -= 1
with col2:
    # 点击下一条，且当前不是最后一条时，索引加 1
    if st.button("下一条 ➡️"):
        if st.session_state.sample_idx < len(data) - 1:
            st.session_state.sample_idx += 1

# --- 保留滑块，并通过 key 绑定 Session State ---
# 注意这里加了 key="sample_idx"，这样滑块和上面的按钮就能联动了
st.sidebar.slider(
    "滑动/输入序号选择:", 
    min_value=0, 
    max_value=len(data) - 1, 
    key="sample_idx"
)

# 从 session_state 获取当前应该展示的 item
item = data[st.session_state.sample_idx]

st.sidebar.markdown("---")
st.sidebar.write(f"**当前 ID:** `{item.get('id', 'Unknown')}`")
st.sidebar.write(f"**Prior Label:** `{item.get('meta_info', {}).get('prior_label', 'N/A')}`")
st.sidebar.write(f"**Real Label:** `{item.get('meta_info', {}).get('label', 'N/A')}`")
# --- 主界面：图像展示 ---
st.header(f"样本可视化 ID: {item.get('id')}")

images = item.get("image", [])
if len(images) == 2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global View [Image 1]")
        if os.path.exists(images[0]):
            st.image(images[0], use_container_width=True)
        else:
            st.error(f"图片不存在: {images[0]}")
            
    with col2:
        st.subheader("Local Detail [Image 2]")
        if os.path.exists(images[1]):
            st.image(images[1], use_container_width=True)
        else:
            st.error(f"图片不存在: {images[1]}")
else:
    st.warning("该样本没有包含两张图片。")

st.markdown("---")

# --- 主界面：模型回复展示 ---
st.subheader("🤖 模型诊断回复 (Assistant Value)")

# 确保 conversation 存在且至少有两条（第一条 human，第二条 assistant）
if "conversation" in item and len(item["conversation"]) > 1:
    assistant_reply = item["conversation"][1].get("value", "")
    
    # 尝试按 JSON 解析展示，如果失败（如你数据里的 fail_reason）则用纯文本展示
    try:
        reply_json = json.loads(assistant_reply)
        st.json(reply_json)
    except json.JSONDecodeError:
        st.warning("⚠️ 该回复为非标准 JSON 格式，已降级为纯文本代码块展示：")
        st.code(assistant_reply, language="json")
else:
    st.info("该样本没有包含模型的回复。")

# --- 底部：Meta Info 展示 ---
with st.expander("查看完整 Meta Info 详情"):
    st.json(item.get("meta_info", {}))