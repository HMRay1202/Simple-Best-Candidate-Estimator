import pandas as pd
import numpy as np

def generate_bootstrap_sample(input_csv_path, n_samples=None, random_state=42):
    """
    读取 CSV 文件，并返回一个 bootstrap sample 的 DataFrame。
    
    参数：
    - input_csv_path: str, 输入 CSV 文件路径
    - n_samples: int, 抽样数量（默认为原数据行数）
    - random_state: int, 随机种子
    
    返回：
    - bootstrap_df: DataFrame，bootstrap 样本
    """
    df = pd.read_csv(input_csv_path)
    if n_samples is None:
        n_samples = len(df)
    bootstrap_df = df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True)
    return bootstrap_df