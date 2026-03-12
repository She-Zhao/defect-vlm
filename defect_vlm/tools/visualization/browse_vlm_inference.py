import streamlit as st
import json
import os
import re
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide", page_title="VLM Result Explorer")

# ================= 工具函数 =================
def parse_vlm_prediction(pred_text):
    try:
        clean_text = pred_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("defect", "background").lower()
    except Exception:
        match = re.search(r'"defect"\s*:\s*"([^"]+)"', pred_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "background"

def extract_probability(token_probs, target_class):
    for t_info in reversed(token_probs):
        token_str = t_info["token"].strip().strip('"').strip("'").lower()
        if token_str in target_class or target_class in token_str:
            return t_info["probability"]
    return 0.5

@st.cache_data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # ================= 新增：预处理一致性标识 =================
                # 在加载数据时就提前算好 prior 和 pred，方便后续极速筛选
                parsed_pred = parse_vlm_prediction(item.get("pred", ""))
                prior = item.get("meta_info", {}).get("prior_label", "unknown")
                item["_parsed_pred"] = parsed_pred
                item["_prior"] = prior
                item["_is_match"] = (parsed_pred == prior)
                # =======================================================
                data.append(item)
    return data

# ================= 主界面 =================
st.title("👁️ VLM 推理结果深度可视化看板")

# 输入文件路径
default_path = "/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v2_qwen3_4b_LM.jsonl"
jsonl_path = st.text_input("请输入 JSONL 文件路径:", default_path)

if not os.path.exists(jsonl_path):
    st.warning("找不到文件，请检查路径！")
    st.stop()

# 加载数据
all_data = load_jsonl(jsonl_path)

# ================= 新增：筛选控制器 =================
st.write("### 🎛️ 数据筛选")
filter_option = st.radio(
    "根据【先验提示】与【VLM结论】的一致性进行过滤：", 
    ["全部数据", "✅ 一致 (VLM 同意先验)", "❌ 不一致 (VLM 进行了矫正)"], 
    horizontal=True
)

# 根据选择过滤数据
if "✅" in filter_option:
    display_data = [item for item in all_data if item["_is_match"]]
elif "❌" in filter_option:
    display_data = [item for item in all_data if not item["_is_match"]]
else:
    display_data = all_data

total_samples = len(display_data)

if total_samples == 0:
    st.warning("🕵️‍♂️ 当前筛选条件下没有找到任何数据！")
    st.stop()

# ================= 分页控制器 =================
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# 如果因为改变了筛选条件导致索引越界，自动重置回合法范围
if st.session_state.current_idx >= total_samples:
    st.session_state.current_idx = max(0, total_samples - 1)

col_prev, col_idx, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ 上一个"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
with col_next:
    if st.button("下一个 ➡️"):
        st.session_state.current_idx = min(total_samples - 1, st.session_state.current_idx + 1)
with col_idx:
    st.markdown(f"### 当前样本: {st.session_state.current_idx + 1} / {total_samples}")

# 获取当前要展示的样本
item = display_data[st.session_state.current_idx]

# 直接使用预处理好的分类结果
pred_cls = item["_parsed_pred"]
prior_label = item["_prior"]
prob = extract_probability(item.get("pred_token_probs", []), pred_cls)

# ================= 数据展示区 =================
st.divider()

col1, col2, col3 = st.columns([1.5, 1.5, 1])

# 1. 图像展示
with col1:
    st.subheader("🖼️ 模型输入图像")
    global_img_path = item["images"][0]
    local_img_path = item["images"][1]
    
    try:
        st.image(Image.open(global_img_path), caption="Global View (Image 1)", use_container_width=True)
        st.image(Image.open(local_img_path), caption="Local Detail (Image 2)", use_container_width=True)
    except Exception as e:
        st.error(f"无法加载图像: {e}")

# 2. 预测与对比
with col2:
    st.subheader("🧠 模型推理结果")
    
    # 核心指标卡片
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("先验提示 (Prior)", prior_label)
    
    # 根据是否匹配给出不同颜色
    metrics_col2.metric("VLM 预测结论", pred_cls, delta="匹配" if item["_is_match"] else "已矫正", delta_color="normal" if item["_is_match"] else "inverse")
    metrics_col3.metric("最终提取置信度", f"{prob:.4f}")

    st.markdown("#### 📝 思维链 (CoT) 输出原文")
    raw_pred = item.get("pred", "无输出")
    try:
        # 尝试去掉可能存在的 markdown 标记并解析为字典，使用 st.json 渲染实现缩进折叠
        clean_text = raw_pred.replace("```json", "").replace("```", "").strip()
        parsed_pred = json.loads(clean_text)
        st.json(parsed_pred)
    except Exception:
        st.info(raw_pred)
    
    st.markdown("#### 📍 BBox 坐标")
    st.code(item["meta_info"]["bbox"])

# 3. Logprobs 探查
with col3:
    st.subheader("🎲 Token 概率流 (末尾 20 个)")
    
    tokens_data = item.get("pred_token_probs", [])
    if tokens_data:
        df_list = []
        for t in tokens_data:
            df_list.append({
                "Token": repr(t["token"]), 
                "Prob": round(t["probability"], 4),
                "LogProb": round(t["logprob"], 4)
            })
        
        df = pd.DataFrame(df_list)
        
        # 高亮目标 Token
        def highlight_target(row):
            target_clean = pred_cls.lower()
            token_clean = row["Token"].strip("'\"").lower()
            if target_clean in token_clean or token_clean in target_clean:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.style.apply(highlight_target, axis=1), height=500, use_container_width=True)
    else:
        st.warning("该样本没有 Logprobs 数据。")