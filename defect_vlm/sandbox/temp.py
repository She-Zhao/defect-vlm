# 定义精确率 (P) 和召回率 (R) 的列表
P = [0.8531, 0.8189, 0.8439, 0.8203, 0.8730, 0.8261]
R = [0.5281, 0.4180, 0.7223, 0.8655, 0.8730, 0.7909]

# 检查两个列表长度是否一致
if len(P) != len(R):
    print("错误：P 和 R 的长度不一致！")
else:
    print("索引\tP\t\tR\t\tF1")
    print("-" * 50)
    
    # 遍历每一对 P 和 R 计算 F1
    for i in range(len(P)):
        p_val = P[i]
        r_val = R[i]
        
        # 处理分母为0的情况
        if p_val + r_val == 0:
            f1 = 0.0
        else:
            f1 = 2 * (p_val * r_val) / (p_val + r_val)
        
        # 打印结果（保留4位小数）
        print(f"{i}\t{p_val:.4f}\t{r_val:.4f}\t{f1:.4f}")