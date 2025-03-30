import numpy as np
import pandas as pd
from Estimator.bayesian_estimator import BayesianOptimalStopping
from Estimator.logistic_estimator import estimate_accept_logistic
from simulate_multiple import simulate_multiple

def load_excel_data(input_excel_path, sheet_name=0):
    """
    读取 Excel 文件中的数据，并返回特征 X 和 true_w（从第一行提取）。

    要求：
    - 第一行是权重行（sample_id 为 'weight'）；
    - 剩下是样本数据；
    - 第一列为 sample_id（不参与建模）；
    - 其余列为特征。

    参数：
    - input_excel_path: str, Excel 文件路径
    - sheet_name: Excel 工作表名称或索引（默认第一个）

    返回：
    - X: numpy.ndarray，特征矩阵（不含权重行）
    - true_w: numpy.ndarray，第一行的权重向量
    """
    # 读取数据，第一列作为索引
    df = pd.read_excel(input_excel_path, sheet_name=sheet_name)

    # 提取权重行（第一行）
    true_w = df.iloc[0, 1:].values.astype(float)

    # 提取样本数据（从第二行开始，不包括 sample_id 列）
    X = df.iloc[1:, 1:].values.astype(float)

    return X, true_w


# === 1. 从 Excel 文件加载数据 ===
input_excel_path = "C:/File/Best_Candidate_Estimator/test.xlsx"  
X_train, true_w = load_excel_data(input_excel_path)
    
print(f"从 {input_excel_path} 读取样本：{len(X_train)} 行, {X_train.shape[1]} 个特征")
print("读取到的 true_w 向量：", true_w)

# === 2. 根据 true_w 构造目标变量 y_train ===
noise_std = 0.1
y_train = X_train.dot(true_w) + np.random.normal(0, noise_std, size=len(X_train))


if __name__ == '__main__':
    # === 3. 多次贝叶斯最优停止模拟 ===
    expected_stop, df_rounds = simulate_multiple(
        BayesianOptimalStopping,
        X_train,
        y_train,
        true_w,
        max_rounds=30,
        num_simulations=100  # 可以根据需要调整模拟次数
    )

    # === 4. 输出结果 ===
    print("\n模拟结果：")
    print("估计的期望停止轮次:", expected_stop)

    df_prob = df_rounds.groupby("round")["predicted_prob_better"].mean().reset_index()
    print("\n各轮次的平均下一次更优概率:")
    print(df_prob)

    print("\n部分模拟详细信息：")
    print(df_rounds.head())


# # ------------------------------
# # 6. 保存详细模拟决策信息到 CSV 文件中
# output_csv = "simulation_rounds_info.csv"
# df_rounds.to_csv(output_csv, index=False)
# print(f"详细决策信息已保存到 {output_csv}")

# ToDo:
'''
1. 用户输入候选人特征数据的方式（CSV、手动输入等）
2. 真实权重向量的生成方式（随机、用户输入等）
3. 模拟参数的配置（如最大轮次、模拟次数等），给出不同预设
4. 结果的可视化（如停止轮次分布图、效用分布图等）
5. 用户GUI，前段交互界面
6. 模拟算法的优化（如使用显示贝叶斯）
7. 模拟过程的优化（如并行计算）
8. 结果的存储与导出（如保存为 Excel、数据库等）'''