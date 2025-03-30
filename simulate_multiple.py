# === 文件 2: simulate_multiple.py ===
import numpy as np
import pandas as pd
from tqdm import tqdm
from Estimator.bayesian_estimator import BayesianOptimalStopping

def simulate_multiple(bos_class, X_train, y_train, true_weights, max_rounds=30, num_simulations=1000,
                      decision_threshold=0.6, noise_std=0.1, num_bootstrap=500):
    """
    多次模拟贝叶斯最优停止策略。

    参数：
    - bos_class: 贝叶斯估计类（BayesianOptimalStopping）
    - X_train, y_train: 初始训练数据
    - true_weights: 用于生成真实效用值的权重
    - max_rounds: 每次模拟最多观察多少个候选人
    - num_simulations: 模拟次数
    - decision_threshold, noise_std, num_bootstrap: 传入模型的参数

    返回：
    - expected_stop: 平均停止轮次
    - df_rounds: 所有模拟轮次的详细记录（DataFrame）
    """
    all_rounds_info = []
    stop_rounds = []

    for sim in tqdm(range(num_simulations), desc="Simulations"):
        bos = bos_class(X_train, y_train,
                        decision_threshold=decision_threshold,
                        noise_std=noise_std,
                        num_bootstrap=num_bootstrap)

        result = bos.simulate(max_rounds=max_rounds, true_weights=true_weights)
        stop_round = result['stopped_round'] if result['stopped_round'] is not None else max_rounds
        stop_rounds.append(stop_round)

        for round_info in result['rounds_info']:
            round_info['simulation_run'] = sim
            all_rounds_info.append(round_info)

    df_rounds = pd.DataFrame(all_rounds_info)
    expected_stop = np.mean(stop_rounds)
    return expected_stop, df_rounds


# === 测试模块调用 ===
if __name__ == '__main__':
    from Estimator.logistic_estimator import estimate_accept_logistic
    np.random.seed(42)

    n_samples = 10
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    true_w = np.array([0.5, 1.2, -0.7])
    true_utility = X @ true_w + np.random.normal(0, 0.1, size=n_samples)

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    accept_prob = sigmoid(true_utility)
    accept = np.random.binomial(1, accept_prob)

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
    df["accept"] = accept
    model, probas, report, acc = estimate_accept_logistic(df)
    print(f"逻辑回归训练完成，准确率: {acc:.3f}")

    X_train = X[:10]
    y_train = true_utility[:10]

    expected_stop, df_rounds = simulate_multiple(
        BayesianOptimalStopping,
        X_train, y_train, true_w,
        max_rounds=30, num_simulations=20  # 缩小轮数和次数以便快速测试
    )

    print("\n估计的期望停止轮次:", expected_stop)
    df_prob = df_rounds.groupby("round")["predicted_prob_better"].mean().reset_index()
    print("\n各轮次的平均下一次更优概率:")
    print(df_prob)
    print("\n部分模拟详细信息：")
    print(df_rounds.head())
