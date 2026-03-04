import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import re

CLASSES = ['breakage', 'inclusion', 'crater', 'bulge', 'scratch', 'run', 'background']

def main(json_path: str, cm_name: str) -> None:
    y_pred = [] 
    y_true = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            item = json.loads(line)
            ai_response_str = item['pred']
            
            # ################################################################################
            # --- 使用正则鲁棒提取 ---
            true_label = item['meta_info']['label'].lower()
        
            # 匹配格式如 "defect": "Breakage" 或 "defect":"background"
            match = re.search(r'"defect"\s*:\s*"([^"]+)"', ai_response_str, re.IGNORECASE)
            
            if match:
                # 提取出 defect 冒号后面的字符串
                pred_label = match.group(1).lower()
            else:
                # 兜底容错：如果连正则都没匹配到，说明模型输出完全崩了
                print(f"⚠️ 警告: 样本 {item['id']} 未找到合法的 defect 字段。原串: {ai_response_str[-50:]}")
                pred_label = "unknown" # 赋一个未知标签，防止列表长度错位
            
            y_pred.append(pred_label)            
            y_true.append(true_label)  
            # ################################################################################
            
            
            # ################################################################################
            # 直接使用json.loads提取
            # try:
            #     ai_response = json.loads(ai_response_str, strict=False)
            
            #     y_pred.append(ai_response['defect'].lower())            
            #     y_true.append(item['meta_info']['label'].lower())
            
            # except Exception as e:
            #     print(f"处理 {item['id']} 时出错: {e}")
            #     continue
            # ################################################################################
            
    # 1. 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    pre_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    stats_summary = (
        f"Accuracy  (准确率):      {acc:.4f}\n"
        f"Precision (宏平均精确率): {pre_macro:.4f}\n"
        f"F1-Score  (宏平均F1):    {f1_macro:.4f}"
    )
    
    print("🎯 --- 核心全局指标 ---")
    print(stats_summary + "\n")
    
    # 2. 生成并保存详细分类报告 (.txt)
    print("📊 --- 详细分类报告 ---")
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0)
    print(report)
    
    txt_save_path = f'/data/ZS/defect-vlm/output/figures/{cm_name}.txt'
    with open(txt_save_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write("🎯 --- 核心全局指标 ---\n")
        f_txt.write(stats_summary + "\n\n")
        f_txt.write("📊 --- 详细分类报告 ---\n")
        f_txt.write(report)
    print(f"✅ 分类报告已保存为 {cm_name}.txt")

    # 3. 绘制混淆矩阵并添加指标文字
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    # 稍微增加高度 (figsize 从 8 改为 9) 给底部文字留空间
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax
    )

    ax.set_title('Defect Classification Confusion Matrix', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    # --- 新增：在图片底部写入全局指标 ---
    stats_text = f"Accuracy: {acc:.4f}  |  Precision (Macro): {pre_macro:.4f}  |  F1-Score (Macro): {f1_macro:.4f}"
    # transform=fig.transFigure 表示坐标是相对于整张图的 (0.5 是中间，0.02 是底部)
    plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5}, fontweight='bold')
    
    # 调整布局以确保文字不被裁切 (增加 bottom 间距)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    plt.savefig(f'/data/ZS/defect-vlm/output/figures/{cm_name}.png', dpi=300)
    print(f"✅ 混淆矩阵已保存为 {cm_name}.png")
    
    
if __name__ == "__main__":
    json_path = '/data/ZS/defect_dataset/8_model_reponse/student/test/qwen3-vl-4b-instrusct.jsonl'
    # 自动获取文件名作为保存名称
    cm_name = Path(json_path).stem
    
    main(json_path, cm_name)