"""
计算半监督训练中类别的先验权重系数 (Gamma)
包含三种不同的缩放策略：线性、指数、幂函数
"""
import numpy as np

def compute_cls_weight_linear(confusion_matrix_p: list) -> list:
    """
    1. 线性缩放 (Linear Scaling)
    最温和稳定的策略。直接除以平均值。
    """
    p_arr = np.array(confusion_matrix_p)
    mean_p = np.mean(p_arr)
    weights = p_arr / mean_p
    return [round(w, 2) for w in weights]

def compute_cls_weight_exponential(confusion_matrix_p: list, tau: float = 0.1) -> list:
    """
    2. 指数缩放 (Exponential Scaling / Temperature Softmax)
    最激进的策略 (马太效应强)。
    :param tau: 温度系数。tau 越小，两极分化越严重；tau 越大，越趋近于平均分配 (1.0)。
    """
    p_arr = np.array(confusion_matrix_p)
    C = len(p_arr)
    
    # 减去最大值防止指数爆炸 (数值稳定性技巧)
    exp_p = np.exp((p_arr - np.max(p_arr)) / tau)
    softmax_p = exp_p / np.sum(exp_p)
    
    # 乘以类别数 C，使生成的权重数组均值依然为 1.0
    weights = C * softmax_p
    return [round(w, 2) for w in weights]

def compute_cls_weight_power(confusion_matrix_p: list, alpha: float = 2.0) -> list:
    """
    3. 幂函数缩放 (Power Scaling)
    折中策略，常用于处理长尾分布与伪标签降噪。
    :param alpha: 指数因子。alpha=1 时等同于线性缩放；alpha>1 时逐渐拉开差距。
    """
    p_arr = np.array(confusion_matrix_p)
    power_p = np.power(p_arr, alpha)
    mean_power = np.mean(power_p)
    
    # 除以幂值的均值，确保最终权重均值为 1.0
    weights = power_p / mean_power
    return [round(w, 2) for w in weights]


if __name__ == "__main__":
    # 你的 VLM 精度先验 (对应: breakage, inclusion, scratch, crater, run, bulge)
    confusion_matrix_p = [0.69, 0.62, 0.81, 0.83, 0.90, 0.78]
    
    print("原始 Precision 矩阵 :", confusion_matrix_p)
    print("-" * 60)
    
    # 1. 线性缩放测试
    w_linear = compute_cls_weight_linear(confusion_matrix_p)
    print(f"1. 线性缩放 (均值=1) : {w_linear}")
    
    # 2. 幂函数缩放测试 (alpha=2.0)
    w_power = compute_cls_weight_power(confusion_matrix_p, alpha=2.0)
    print(f"2. 幂函数缩放 (alpha=2): {w_power}")
    
    # 3. 指数缩放测试 (tau=0.2)
    w_exp = compute_cls_weight_exponential(confusion_matrix_p, tau=0.2)
    print(f"3. 指数缩放 (tau=0.1)  : {w_exp}")
    
    print("-" * 60)
    print(f"注意观察 inclusion(第2项) 和 run(第5项) 的权重差距变化。")