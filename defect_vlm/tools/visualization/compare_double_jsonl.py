"""
输入两个jsonl文件，用streamlit进行对比。
"""
import streamlit as st
import json
import os

# 1. 页面配置：必须开启 wide 模式，否则左右对比会非常拥挤
st.set_page_config(page_title="JSONL 数据对比工具", layout="wide")

st.title("🔍 工业级 JSONL 双栏对比核对工具")

# 2. 侧边栏：用于输入服务器上的绝对路径
with st.sidebar:
    st.header("📁 文件路径配置")
    file1_path = st.text_input("左侧文件路径 (如学生 prompt):", 
                               value="/data/ZS/defect_dataset/4_api_request/student/train/stripe_phase012_gt_negative.jsonl")
    file2_path = st.text_input("右侧文件路径 (如老师 response):", 
                               value="/data/ZS/defect_dataset/6_sft_dataset/teacher/train/stripe_phase012_gt_negative.jsonl")
    load_btn = st.button("🚀 加载数据", type="primary")

# 3. 初始化 Session State（保持翻页时数据不丢失）
if 'data1' not in st.session_state:
    st.session_state.data1 = []
if 'data2' not in st.session_state:
    st.session_state.data2 = []
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# 4. 数据加载函数
def load_data(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        st.error("❌ 某个文件路径不存在，请检查绝对路径！")
        return
    
    try:
        with open(path1, 'r', encoding='utf-8') as f:
            st.session_state.data1 = [json.loads(line) for line in f if line.strip()]
        with open(path2, 'r', encoding='utf-8') as f:
            st.session_state.data2 = [json.loads(line) for line in f if line.strip()]
            
        st.session_state.current_idx = 0 # 重置索引
        st.sidebar.success(f"✅ 加载成功！\n左: {len(st.session_state.data1)} 条\n右: {len(st.session_state.data2)} 条")
    except Exception as e:
        st.error(f"❌ 解析 JSONL 失败: {e}")

if load_btn:
    load_data(file1_path, file2_path)

# 5. 主界面渲染与交互
if st.session_state.data1 or st.session_state.data2:
    max_len = max(len(st.session_state.data1), len(st.session_state.data2))
    
    # 顶部控制台 (翻页器)
    st.markdown("---")
    col_prev, col_input, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️ 上一条", use_container_width=True) and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
            
    with col_input:
        # 直接输入数字跳转
        new_idx = st.number_input(f"当前进度: {st.session_state.current_idx + 1} / {max_len} (输入行号跳转)", 
                                  min_value=1, max_value=max_len, 
                                  value=st.session_state.current_idx + 1, 
                                  label_visibility="collapsed")
        if new_idx - 1 != st.session_state.current_idx:
            st.session_state.current_idx = new_idx - 1
            st.rerun()
            
    with col_next:
        if st.button("下一条 ➡️", use_container_width=True) and st.session_state.current_idx < max_len - 1:
            st.session_state.current_idx += 1
            st.rerun()

    st.markdown("---")
    
    # 双栏对比展示
    col_left, col_right = st.columns(2)
    idx = st.session_state.current_idx
    
    with col_left:
        st.subheader("📄 左侧数据")
        if idx < len(st.session_state.data1):
            # st.json 渲染自带代码高亮、折叠功能
            st.json(st.session_state.data1[idx], expanded=True)
        else:
            st.warning("⚠️ 越界：左侧文件已到底")

    with col_right:
        st.subheader("📄 右侧数据")
        if idx < len(st.session_state.data2):
            st.json(st.session_state.data2[idx], expanded=True)
        else:
            st.warning("⚠️ 越界：右侧文件已到底")
else:
    st.info("👈 请在左侧边栏输入文件路径，并点击加载数据。")
