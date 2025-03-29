import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

class BayesianOptimalStopping:
    def __init__(self, X, y, candidate_generator=None, decision_threshold=0.6, noise_std=0.1, num_bootstrap=500):
        """
        参数说明：
        - X: 训练数据特征，形状 (n_samples, n_features)
        - y: 训练数据响应（效用值），形状 (n_samples,)
        - candidate_generator: 生成新候选人特征的函数，
              若为 None，则默认基于当前的 X_obs 进行 bootstrap 抽样并加入微扰；
        - decision_threshold: 当预测新候选人超越当前最好效用的概率达到该阈值时，作出停止选择决策；
        - noise_std: 模拟候选人效用时的噪音标准差；
        - num_bootstrap: bootstrap 重采样次数。
        """
        self.X_obs = X.copy()
        self.y_obs = y.copy()
        self.decision_threshold = decision_threshold
        self.noise_std = noise_std
        self.num_bootstrap = num_bootstrap
        self.n_features = X.shape[1]
        
        # 如果用户未传入 candidate_generator，则使用默认生成器
        if candidate_generator is None:
            self.candidate_generator = self.default_candidate_generator
        else:
            self.candidate_generator = candidate_generator
            
        # 利用初始数据通过 bootstrap 构造模型参数分布（经验先验）
        self.update_prior()
        
    def default_candidate_generator(self):
        """
        默认候选人生成器：
        从当前的 X_obs 中随机抽取一个样本（bootstrap），
        并在其上加入微扰以模拟真实中候选人的轻微变化。
        """
        idx = np.random.choice(self.X_obs.shape[0], size=1, replace=True)
        candidate = self.X_obs[idx][0]
        # 添加微扰，噪音服从均值0、标准差0.05的正态分布
        noise = np.random.normal(loc=0, scale=0.05, size=candidate.shape)
        return candidate + noise
    
    def update_prior(self):
        """
        通过 bootstrap 重采样，对当前观测数据构造线性回归模型，
        得到回归系数的样本分布，用作经验先验。
        """
        n_samples = self.X_obs.shape[0]
        bootstrap_weights = []
        for _ in range(self.num_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_obs[idx]
            y_boot = self.y_obs[idx]
            model = LinearRegression().fit(X_boot, y_boot)
            bootstrap_weights.append(model.coef_)
        self.bootstrap_weights = np.array(bootstrap_weights)  # 形状 (num_bootstrap, n_features)
        
    def predict_candidate(self, x_candidate):
        """
        给定候选人特征向量 x_candidate（形状 (n_features,)），
        利用 bootstrap 得到的各个模型参数预测其效用，返回预测效用的一维数组。
        """
        predictions = np.dot(self.bootstrap_weights, x_candidate)
        return predictions
    
    def decide(self, x_candidate, current_best):
        """
        给定候选人 x_candidate 和当前最佳效用 current_best，
        计算候选人效用超过 current_best 的概率，并判断是否满足决策阈值。
        返回 (概率, 是否满足停止决策)。
        """
        preds = self.predict_candidate(x_candidate)
        prob_better = np.mean(preds > current_best)
        return prob_better, prob_better >= self.decision_threshold
    
    def simulate(self, max_rounds=50, true_weights=None):
        """
        模拟候选人依次到来，利用最优停止策略作出决策。
        
        参数：
          - max_rounds: 模拟的最大候选人数；
          - true_weights: 若提供，用于模拟候选人的真实效用，
              假设真实模型为：y = dot(x, true_weights) + 噪音。
              
        返回一个字典，包含：
          - 'stopped_round': 停止选择的轮次（若未停止则为 None）；
          - 'candidate_x': 被选中的候选人特征（如果作出选择）；
          - 'candidate_utility': 被选中候选人的真实效用（如果提供 true_weights）；
          - 'rounds_info': 每一轮决策信息列表，每项为一个字典。
        """
        rounds_info = []
        stopped_round = None
        chosen_candidate = None
        chosen_utility = None
        
        # 初始化当前最佳效用：取已有训练数据中的最大值
        current_best = np.max(self.y_obs)
        
        for t in range(1, max_rounds + 1):
            x_candidate = self.candidate_generator()
            # 若提供真实权重，则模拟候选人真实效用（线性模型 + 噪音）
            if true_weights is not None:
                candidate_utility = np.dot(x_candidate, true_weights) + np.random.normal(0, self.noise_std)
            else:
                candidate_utility = None
            
            prob_better, decision = self.decide(x_candidate, current_best)
            
            round_dict = {
                'round': t,
                'x_candidate': x_candidate,
                'predicted_prob_better': prob_better,
                'decision': decision,
                'current_best': current_best,
                'candidate_utility': candidate_utility
            }
            rounds_info.append(round_dict)
            
            # 如果决策满足条件，则停止搜索
            if decision:
                stopped_round = t
                chosen_candidate = x_candidate
                if candidate_utility is not None:
                    chosen_utility = candidate_utility
                break
            else:
                # 若未选择，则将候选人信息加入观测数据，并更新当前最佳效用与先验
                if candidate_utility is not None:
                    self.X_obs = np.vstack([self.X_obs, x_candidate])
                    self.y_obs = np.append(self.y_obs, candidate_utility)
                    if candidate_utility > current_best:
                        current_best = candidate_utility
                    self.update_prior()
                    
        return {
            'stopped_round': stopped_round,
            'candidate_x': chosen_candidate,
            'candidate_utility': chosen_utility,
            'rounds_info': rounds_info
        }

def simulate_multiple(bos_class, X_train, y_train, true_weights, max_rounds=30, num_simulations=1000):
    """
    对 BayesianOptimalStopping 模型进行多次模拟，
    返回期望停止轮次以及所有模拟的轮次详细信息（DataFrame）。
    """
    all_rounds_info = []
    stop_rounds = []
    
    for sim in tqdm(range(num_simulations), desc="Simulations"):
        # 每次模拟前重新实例化模型，避免数据污染
        bos = bos_class(X_train, y_train, decision_threshold=0.6, noise_std=0.1, num_bootstrap=500)
        result = bos.simulate(max_rounds=max_rounds, true_weights=true_weights)
        stop_round = result['stopped_round'] if result['stopped_round'] is not None else max_rounds
        stop_rounds.append(stop_round)
        
        for round_info in result['rounds_info']:
            round_info['simulation_run'] = sim
            all_rounds_info.append(round_info)
    
    df_rounds = pd.DataFrame(all_rounds_info)
    expected_stop = np.mean(stop_rounds)
    return expected_stop, df_rounds

if __name__ == '__main__':
    np.random.seed(42)
    n_samples = 10
    n_features = 3
    # 构造初始训练数据：10个样本、3个特征
    X_train = np.random.rand(n_samples, n_features)
    # 定义真实权重向量，用于模拟候选人真实效用
    true_w = np.array([0.5, 1.2, -0.7])
    y_train = X_train.dot(true_w) + np.random.normal(0, 0.1, size=n_samples)
    
    # 进行多次模拟，获取期望停止轮次以及各轮次的详细信息
    expected_stop, df_rounds = simulate_multiple(BayesianOptimalStopping, X_train, y_train, true_w, max_rounds=30, num_simulations=100)
    
    print("估计的期望停止轮次:", expected_stop)
    # 按轮次计算各轮次的平均下一次更优概率
    df_prob = df_rounds.groupby('round')['predicted_prob_better'].mean().reset_index()
    print("\n各轮次的平均下一次更优概率:")
    print(df_prob)
    print("\n部分模拟详细信息：")
    print(df_rounds.head())