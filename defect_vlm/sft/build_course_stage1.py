"""
构建课程学习第一阶段需要的训练数据
"""
import os
import json
import re
import random
from tqdm import tqdm

def process_assistant_content(content):
    """
    剥离 Assistant 的逻辑推理和结论，仅保留 step1 和 step2
    """
    try:
        # 清理可能存在的 markdown 标记
        clean_text = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        # 仅保留纯视觉描述
        new_data = {
            "step1": data.get("step1", ""),
            "step2": data.get("step2", "")
        }
        # 格式化输出为漂亮的 JSON 字符串
        return json.dumps(new_data, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ 解析 Assistant JSON 失败，跳过该条数据: {e}")
        return None

def process_user_content(content):
    """
    对 User 的指令进行外科手术式大改造：去分类、去逻辑，只留观察
    """
    # 1. 剔除先验提示 (Prior Hint)
    content = re.sub(r"Prior Hint:\s*[^\n]+\n\n", "", content)
    
    # 2. 篡改任务目标 (Task Objectives)
    task_obj_pattern = r"# Task Objectives & Notes[\s\S]*?(?=# Notes)"
    task_obj_repl = (
        "# Task Objectives & Notes\n"
        "Your task is to utilize the visual information from [Image 1] and [Image 2] to **carefully observe and describe the macroscopic background texture and the microscopic abnormal features in the red box area**. You do not need to diagnose the defect type.\n\n"
    )
    content = re.sub(task_obj_pattern, task_obj_repl, content)
    
    # 3. 彻底删除“缺陷定义”库 (Defect Definitions)
    def_pattern = r"# Defect Definitions & Imaging Characteristics[\s\S]*?(?=# Analysis Steps)"
    content = re.sub(def_pattern, "", content)
    
    # 4. 裁剪分析步骤 (Analysis Steps) 中的 Step 3
    step3_pattern = r"3\.\s+\*\*Step 3 \(Feature Matching & Comprehensive Analysis\)\*\*:[^\n]*\n"
    content = re.sub(step3_pattern, "", content)
    
    # 5. 修改输出格式约束 (Output Format)
    out_format_pattern = r"{\n\s+\"step1\":[\s\S]*?\"defect\":[\s\S]*?\n}"
    out_format_repl = (
        "{\n"
        "    \"step1\": \"Analyze the distribution of the four light sources and the background texture in the global view...\",\n"
        "    \"step2\": \"Describe the abnormal features found in any patch of the local view (e.g., obvious stripe breakage visible in the top-right patch)...\"\n"
        "}"
    )
    content = re.sub(out_format_pattern, out_format_repl, content)
    
    return content

def build_stage1_data(input_dir, target_files, output_path, sample_size, random_seed):
    """
    构建 Stage 1 数据集的主控流
    """
    random.seed(random_seed) # 保证多次运行抽样结果完全一致
    all_valid_data = []

    print(f"🔍 开始扫描目录: {input_dir}")
    print(f"🎯 目标文件白名单: {target_files}")

    # 1. 遍历并加载目标文件中的所有数据
    for filename in os.listdir(input_dir):
        if filename in target_files:
            file_path = os.path.join(input_dir, filename)
            print(f"📖 正在读取: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_valid_data.append(json.loads(line))

    total_loaded = len(all_valid_data)
    print(f"\n🌊 成功将 {total_loaded} 条数据载入全局水池！")

    # 2. 随机抽样
    if total_loaded < sample_size:
        print(f"⚠️ 警告: 水池数据总量 ({total_loaded}) 小于期望采样量 ({sample_size})，将使用全部数据。")
        sample_size = total_loaded
    
    sampled_data = random.sample(all_valid_data, sample_size)
    print(f"🎲 已随机抽取 {sample_size} 条数据准备改造...")

    # 3. 执行清洗与重构
    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(sampled_data, desc="🛠️ 重构数据中"):
            messages = item.get("messages", [])
            valid_item = True
            
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = process_user_content(msg["content"])
                elif msg["role"] == "assistant":
                    new_content = process_assistant_content(msg["content"])
                    if new_content:
                        msg["content"] = new_content
                    else:
                        valid_item = False # 如果解析失败则丢弃这条数据
            
            if valid_item:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                processed_count += 1

    print(f"\n🎉 大功告成！Stage 1 数据集构建完毕。")
    print(f"💾 最终有效写入: {processed_count} 条数据")
    print(f"📁 保存路径: {output_path}")

if __name__ == "__main__":
    # ================= 🎛️ 核心参数配置区 🎛️ =================
    
    # 1. 原始数据集目录
    INPUT_DIR = "/data/ZS/defect_dataset/7_swift_dataset/val"
    
    # 2. 目标文件列表 (我们只提取正样本和负样本，避开同源的纠错样本)
    TARGET_FILES = [
        "stripe_phase012_gt_positive.jsonl",
        "stripe_phase012_gt_negative.jsonl",
        "stripe_phase123_gt_positive.jsonl",
        "stripe_phase123_gt_negative.jsonl"
    ]
    
    # 3. 输出文件路径 (建议专门建一个 stage1 的文件夹)
    OUTPUT_FILE = "/data/ZS/defect_dataset/7_swift_dataset/course/stage1_vision_only_15k_val.jsonl"
    
    # 4. 采样数量
    SAMPLE_SIZE = 2000
    
    # 5. 随机种子 (非常重要，写在论文里可以说明实验具备严格可复现性)
    RANDOM_SEED = 42 
    
    # ========================================================
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 启动任务
    build_stage1_data(
        input_dir=INPUT_DIR,
        target_files=TARGET_FILES,
        output_path=OUTPUT_FILE,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )
