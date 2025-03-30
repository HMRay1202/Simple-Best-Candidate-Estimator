import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from Estimator.bayesian_estimator import BayesianOptimalStopping

def run_single_simulation(sim, bos_class, X_train, y_train, true_weights, max_rounds,
                          decision_threshold, noise_std, num_bootstrap):
    """
    执行一次模拟，并返回停止轮次和每一轮的详细信息。
    """
    bos = bos_class(X_train, y_train,
                    decision_threshold=decision_threshold,
                    noise_std=noise_std,
                    num_bootstrap=num_bootstrap)
    result = bos.simulate(max_rounds=max_rounds, true_weights=true_weights)
    stop_round = result['stopped_round'] if result['stopped_round'] is not None else max_rounds
    # 为每一轮记录加入模拟编号
    for round_info in result['rounds_info']:
        round_info['simulation_run'] = sim
    return stop_round, result['rounds_info']

def simulate_multiple(bos_class, X_train, y_train, true_weights, max_rounds=30, num_simulations=1000,
                      decision_threshold=0.6, noise_std=0.1, num_bootstrap=500):
    """
    多次模拟贝叶斯最优停止策略（并行运行）。

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

    # 利用 ProcessPoolExecutor 实现并行模拟
    with ProcessPoolExecutor() as executor:
        func = partial(run_single_simulation, bos_class=bos_class, X_train=X_train, y_train=y_train,
                       true_weights=true_weights, max_rounds=max_rounds,
                       decision_threshold=decision_threshold, noise_std=noise_std,
                       num_bootstrap=num_bootstrap)
        results = list(tqdm(executor.map(func, range(num_simulations)), total=num_simulations, desc="Simulations"))

    for stop_round, rounds_info in results:
        stop_rounds.append(stop_round)
        all_rounds_info.extend(rounds_info)

    df_rounds = pd.DataFrame(all_rounds_info)
    expected_stop = np.mean(stop_rounds)
    return expected_stop, df_rounds

# 如果直接运行此模块，则进行简单测试
if __name__ == '__main__':
    # 构造简单的随机数据进行测试
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    true_weights = np.random.rand(5)
    expected_stop, df_rounds = simulate_multiple(BayesianOptimalStopping, X_train, y_train, true_weights,
                                                   max_rounds=30, num_simulations=100, decision_threshold=0.6,
                                                   noise_std=0.1, num_bootstrap=500)
    print("平均停止轮次：", expected_stop)
    print(df_rounds.head())
