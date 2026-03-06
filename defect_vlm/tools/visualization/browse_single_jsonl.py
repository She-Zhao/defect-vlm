"""
输入一个jsonl文件，用streamlit进行可视化
"""
import streamlit as st
import json
import os

# 1. 页面配置
st.set_page_config(page_title="JSONL 单文件查看器", layout="wide")

st.title("📄 工业级 JSONL 单文件浏览工具")

# 2. 侧边栏：用于输入服务器上的绝对路径
with st.sidebar:
    st.header("📁 文件路径配置")
    # 默认填入你刚刚生成的精美数据集路径之一
    file_path = st.text_input("文件绝对路径:", 
                              value="/data/ZS/defect_dataset/7_swift_dataset/teacher/train/stripe_phase012_gt_negative.jsonl")
    load_btn = st.button("🚀 加载数据", type="primary")

# 3. 初始化 Session State (保持翻页时不丢失数据)
if 'single_data' not in st.session_state:
    st.session_state.single_data = []
if 'single_current_idx' not in st.session_state:
    st.session_state.single_current_idx = 0

# 4. 数据加载函数
def load_data(path):
    if not os.path.exists(path):
        st.error("❌ 文件路径不存在，请检查！")
        return
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            st.session_state.single_data = [json.loads(line) for line in f if line.strip()]
            
        st.session_state.single_current_idx = 0 # 重置索引
        st.sidebar.success(f"✅ 加载成功！共计 {len(st.session_state.single_data)} 条数据")
    except Exception as e:
        st.error(f"❌ 解析 JSONL 失败: {e}")

if load_btn:
    load_data(file_path)

# 5. 主界面渲染与交互
if st.session_state.single_data:
    max_len = len(st.session_state.single_data)
    
    # 顶部控制台 (翻页器)
    st.markdown("---")
    col_prev, col_input, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️ 上一条", use_container_width=True) and st.session_state.single_current_idx > 0:
            st.session_state.single_current_idx -= 1
            st.rerun()
            
    with col_input:
        # 直接输入数字跳转
        new_idx = st.number_input(f"当前进度: {st.session_state.single_current_idx + 1} / {max_len} (输入行号跳转)", 
                                  min_value=1, max_value=max_len, 
                                  value=st.session_state.single_current_idx + 1, 
                                  label_visibility="collapsed")
        if new_idx - 1 != st.session_state.single_current_idx:
            st.session_state.single_current_idx = new_idx - 1
            st.rerun()
            
    with col_next:
        if st.button("下一条 ➡️", use_container_width=True) and st.session_state.single_current_idx < max_len - 1:
            st.session_state.single_current_idx += 1
            st.rerun()

    st.markdown("---")
    
    # 单文件数据展示区
    st.subheader("📄 数据详情")
    idx = st.session_state.single_current_idx
    
    # 使用 st.json 渲染，自带美观的语法高亮和折叠
    st.json(st.session_state.single_data[idx], expanded=True)

else:
    st.info("👈 请在左侧边栏输入 JSONL 文件的绝对路径，并点击“加载数据”。")
