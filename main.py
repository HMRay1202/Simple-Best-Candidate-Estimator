import numpy as np
import pandas as pd
from bootstrap_utils import generate_bootstrap_sample
from Estimator.bayesian_estimator import BayesianOptimalStopping
from simulate_multiple import simulate_multiple
# 设置候选人 CSV 文件路径（或是以后的数据源）
input_csv = "candidates.csv"

# 生成 bootstrap 样本，作为训练数据的候选人特征
bootstrap_df = generate_bootstrap_sample(input_csv)
print(f"从 {input_csv} 生成 bootstrap sample，共 {len(bootstrap_df)} 行。")

# 将 bootstrap 样本转换为 numpy 数组作为 X_train ===
X_train = bootstrap_df.values

# 获取特征维度，根据 X_train 的列数自动确定真实权重的长度
n_features = X_train.shape[1]
print(f"候选人特征的数量：{n_features}")

# 定义真实权重向量 true_w（需要想想怎么构建，给定还是随机还是用户输入）
# 这下面示例随机生成一个权重向量（范围可调）
true_w = np.random.uniform(0.5, 1.5, size=n_features)
print("生成的真实权重向量 true_w：", true_w)

# === 根据 X_train 计算 y_train：采用线性模型 y = X * true_w + 噪音 ===
y_train = X_train.dot(true_w) + np.random.normal(0, 0.1, size=len(X_train))

# 5. 调用整个模拟流程（多次贝叶斯模拟）
# 模拟过程会多次运行贝叶斯最优停止模型，记录每次的停止轮次及详细决策信息
expected_stop, df_rounds = simulate_multiple(
    BayesianOptimalStopping,
    X_train,
    y_train,
    true_w,
    max_rounds=30,
    num_simulations=1000
)

print("\n模拟结果：")
print("估计的期望停止轮次:", expected_stop)

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