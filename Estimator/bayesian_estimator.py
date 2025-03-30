# === 文件 1: bayesian_estimator.py ===
import numpy as np
from sklearn.linear_model import LinearRegression
from Estimator.logistic_estimator import logistic_model_global, predict_accept_probability

class BayesianOptimalStopping:
    def __init__(self, X, y, candidate_generator=None, decision_threshold=0.6, noise_std=0.1, num_bootstrap=500):
        self.X_obs = X.copy()
        self.y_obs = y.copy()
        self.decision_threshold = decision_threshold
        self.noise_std = noise_std
        self.num_bootstrap = num_bootstrap
        self.n_features = X.shape[1]

        if candidate_generator is None:
            self.candidate_generator = self.default_candidate_generator
        else:
            self.candidate_generator = candidate_generator

        self.update_prior()

    def default_candidate_generator(self):
        idx = np.random.choice(self.X_obs.shape[0], size=1, replace=True)
        candidate = self.X_obs[idx][0]
        noise = np.random.normal(loc=0, scale=0.05, size=candidate.shape)
        return candidate + noise

    def update_prior(self):
        n_samples = self.X_obs.shape[0]
        bootstrap_weights = []
        for _ in range(self.num_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_obs[idx]
            y_boot = self.y_obs[idx]
            model = LinearRegression().fit(X_boot, y_boot)
            bootstrap_weights.append(model.coef_)
        self.bootstrap_weights = np.array(bootstrap_weights)

    def predict_candidate(self, x_candidate):
        return np.dot(self.bootstrap_weights, x_candidate)

    def decide(self, x_candidate, current_best):
        preds = self.predict_candidate(x_candidate)
        prob_better = np.mean(preds > current_best)
        return prob_better, prob_better >= self.decision_threshold

    def simulate(self, max_rounds=50, true_weights=None):
        rounds_info = []
        stopped_round = None
        chosen_candidate = None
        chosen_utility = None
        current_best = np.max(self.y_obs)

        for t in range(1, max_rounds + 1):
            x_candidate = self.candidate_generator()
            if true_weights is not None:
                base_utility = np.dot(x_candidate, true_weights)
                accept_prob = predict_accept_probability(logistic_model_global, x_candidate) if logistic_model_global else 1.0
                candidate_utility = base_utility * accept_prob + np.random.normal(0, self.noise_std)
            else:
                candidate_utility = None

            prob_better, decision = self.decide(x_candidate, current_best)

            rounds_info.append({
                'round': t,
                'x_candidate': x_candidate,
                'predicted_prob_better': prob_better,
                'decision': decision,
                'current_best': current_best,
                'candidate_utility': candidate_utility,
                'accept_prob': accept_prob if true_weights is not None else None
            })

            if decision:
                stopped_round = t
                chosen_candidate = x_candidate
                chosen_utility = candidate_utility
                break
            else:
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
