"""
对ms-swift需要的message格式jsonl文件进行可视化
输入举例：/data/ZS/defect_dataset/7_swift_dataset/train/stripe_phase012_gt_negative.jsonl
"""

import streamlit as st
import json
import os
import re
import glob
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide", page_title="VLM Result Explorer")

# ================= 工具函数 =================
def parse_assistant_content(content):
    """尝试解析 Assistant 的输出（处理可能带有的 markdown 标记）"""
    try:
        clean_text = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("defect", "background").lower(), data
    except Exception:
        # 如果不是标准 JSON，尝试正则硬提取
        match = re.search(r'"defect"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
        defect = match.group(1).lower() if match else "unknown"
        return defect, {"raw_output": content}

@st.cache_data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 兼容 messages 格式的数据集/推理结果
            messages = item.get("messages", [])
            user_content = ""
            ast_content = ""
            
            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    ast_content = msg.get("content", "")
            
            # 如果存在额外的 pred 字段 (来自你之前的推理脚本)，则优先使用 pred
            if "pred" in item and item["pred"]:
                ast_content = item["pred"]

            # 1. 提取先验提示 (Prior Hint)
            prior_match = re.search(r'Prior Hint:\s*([^\n]+)', user_content)
            prior = prior_match.group(1).strip().lower() if prior_match else "unknown"

            # 2. 提取 Assistant 结果 (微调数据里的 GT，或者推理脚本里的 Pred)
            if ast_content:
                parsed_defect, parsed_dict = parse_assistant_content(ast_content)
            else:
                parsed_defect = "no_output"
                parsed_dict = {}

            item["_prior"] = prior
            item["_parsed_defect"] = parsed_defect
            item["_parsed_dict"] = parsed_dict
            item["_is_match"] = (parsed_defect == prior)
            item["_user_content"] = user_content
            
            data.append(item)
    return data

# ================= 顶部配置与文件选择 =================
st.title("👁️ VLM 数据集与推理结果深度看板")

col_path, col_file = st.columns([2, 1])

with col_path:
    # 默认路径指向你的训练集或推理结果目录
    base_dir = st.text_input("📁 请输入包含 JSONL 文件的文件夹路径 (或具体文件):", "/data/ZS/defect_dataset/7_swift_dataset/train")

jsonl_path = ""
if os.path.isdir(base_dir):
    # 扫描目录下所有的 jsonl 文件
    jsonl_files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    if not jsonl_files:
        st.warning("该目录下没有找到 .jsonl 文件！")
        st.stop()
    with col_file:
        selected_file = st.selectbox("📄 请选择要查看的文件:", [os.path.basename(f) for f in sorted(jsonl_files)])
        jsonl_path = os.path.join(base_dir, selected_file)
elif os.path.isfile(base_dir) and base_dir.endswith(".jsonl"):
    jsonl_path = base_dir
else:
    st.warning("找不到指定的路径，请检查！")
    st.stop()

# 当切换文件时，自动重置页码
if 'last_file' not in st.session_state or st.session_state.last_file != jsonl_path:
    st.session_state.current_idx = 0
    st.session_state.last_file = jsonl_path

# 加载数据
with st.spinner('正在极速加载解析数据...'):
    all_data = load_jsonl(jsonl_path)

# ================= 筛选控制器 =================
st.write("### 🎛️ 数据筛选")
filter_option = st.radio(
    "根据【先验提示】与【VLM结论】的一致性进行过滤：", 
    ["全部数据", "✅ 一致 (VLM 同意先验)", "❌ 不一致 (VLM 进行了矫正)", "⚠️ 无输出"], 
    horizontal=True
)

if "✅" in filter_option:
    display_data = [item for item in all_data if item["_is_match"] and item["_parsed_defect"] != "no_output"]
elif "❌" in filter_option:
    display_data = [item for item in all_data if not item["_is_match"] and item["_parsed_defect"] != "no_output"]
elif "⚠️" in filter_option:
    display_data = [item for item in all_data if item["_parsed_defect"] == "no_output"]
else:
    display_data = all_data

total_samples = len(display_data)

if total_samples == 0:
    st.warning("🕵️‍♂️ 当前筛选条件下没有找到任何数据！")
    st.stop()

# ================= 分页控制器 =================
if st.session_state.current_idx >= total_samples:
    st.session_state.current_idx = max(0, total_samples - 1)

col_prev, col_idx, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ 上一条", use_container_width=True):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
with col_next:
    if st.button("下一条 ➡️", use_container_width=True):
        st.session_state.current_idx = min(total_samples - 1, st.session_state.current_idx + 1)
with col_idx:
    # 居中显示进度
    st.markdown(f"<h3 style='text-align: center;'>当前样本: {st.session_state.current_idx + 1} / {total_samples}</h3>", unsafe_allow_html=True)

# 获取当前样本
item = display_data[st.session_state.current_idx]
pred_cls = item["_parsed_defect"]
prior_label = item["_prior"]

# ================= 数据展示区 =================
st.divider()
col_img, col_text = st.columns([1.2, 1.8])

# 1. 图像展示
with col_img:
    st.subheader("🖼️ 模型输入图像")
    if "images" in item and len(item["images"]) >= 2:
        global_img_path = item["images"][0]
        local_img_path = item["images"][1]
        try:
            st.image(Image.open(global_img_path), caption=f"Global View | {os.path.basename(global_img_path)}", use_container_width=True)
            st.image(Image.open(local_img_path), caption=f"Local Detail | {os.path.basename(local_img_path)}", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载图像 (请检查服务器路径是否正确): \n{global_img_path}\n{e}")
    else:
        st.warning("当前数据没有完整的 images 字段记录。")

# 2. 预测与对比
with col_text:
    st.subheader("🧠 模型推理结果 / 数据集 Ground Truth")
    
    # 核心指标卡片
    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric("先验提示 (Prior Hint)", prior_label)
    
    if pred_cls == "no_output":
         metrics_col2.metric("VLM 目标输出", "无回答 (尚未推理)", delta="待推理", delta_color="off")
    else:
         metrics_col2.metric("VLM 目标输出", pred_cls, delta="一致 (匹配先验)" if item["_is_match"] else "矫正 (推翻先验)", delta_color="normal" if item["_is_match"] else "inverse")

    # 结构化展示 JSON
    st.markdown("#### 📝 Assistant 结构化输出")
    if pred_cls != "no_output":
        st.json(item["_parsed_dict"])
    else:
        st.info("该条数据中 Assistant 为空。")
        
    # 可折叠的 Prompt
    with st.expander("🔍 查看完整的 User Prompt (输入指令)"):
        st.text(item["_user_content"])