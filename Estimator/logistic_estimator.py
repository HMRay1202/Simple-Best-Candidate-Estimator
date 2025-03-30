# === logistic_estimator.py ===
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score

logistic_model_global = None  # 可供贝叶斯模拟器调用的全局模型


def estimate_accept_logistic(candidate_df, target_column='accept', num_bootstrap_samples=1000):
    """
    使用候选人数据中的其他特征和 bootstrap 构造的大量样本来预测 'accept' （0/1）列。

    参数：
    - candidate_df: 包含所有候选人信息的 DataFrame，需包含 'accept' 列。
    - target_column: 默认为 'accept'，为预测目标。
    - num_bootstrap_samples: bootstrap 总样本数量（默认 1000）

    返回：
    - model: 训练好的 LogisticRegression 模型
    - probas: 测试集样本被接受的概率（0-1 之间）
    - report: classification report
    - accuracy: 准确率
    """
    global logistic_model_global

    # 原始数据
    X = candidate_df.drop(columns=[target_column])
    y = candidate_df[target_column]
    data = candidate_df.copy()

    # bootstrap 重采样，构造放大量样本
    bootstrap_data = resample(data, replace=True, n_samples=num_bootstrap_samples, random_state=42)
    X_boot = bootstrap_data.drop(columns=[target_column])
    y_boot = bootstrap_data[target_column]

    # 拆分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]  # 返回为1的概率
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    logistic_model_global = model  # 存入全局变量供外部模块使用

    return model, probas, report, accuracy


def predict_accept_probability(model, x_input):
    """
    使用训练好的模型对一个新候选人的特征向量进行预测，返回其被接受的概率。
    参数：
    - model: 训练好的 LogisticRegression 模型
    - x_input: 一维 numpy 数组，表示候选人特征，形如 (n_features,) 或二维 (1, n_features)

    返回：
    - 概率值：介于 0 和 1 之间
    """
    x_input = np.array(x_input).reshape(1, -1)
    return model.predict_proba(x_input)[0, 1]
