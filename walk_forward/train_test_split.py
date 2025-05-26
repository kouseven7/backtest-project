"""
Module: Train-Test Split for Walk-Forward Testing
File: train_test_split.py
Description: 
  日次データをトレーニング期間とテスト期間に分割するための関数を提供します。

Author: imega
Created: 2025-04-29

Dependencies:
  - pandas
"""

import pandas as pd
from typing import List, Tuple

def split_data_for_walk_forward(data: pd.DataFrame, train_size: int, test_size: int):
    """
    ウォークフォワードテスト用に日次データをトレーニング期間とテスト期間に分割する。

    Parameters:
        data (pd.DataFrame): 株価データ
        train_size (int): トレーニング期間の長さ（日数）
        test_size (int): テスト期間の長さ（日数）

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: トレーニング期間とテスト期間のペアのリスト
    """
    splits = []
    total_size = len(data)
    if total_size < train_size + test_size:
        print("Warning: Not enough data for the specified train and test sizes.")
        return splits

    for start in range(0, total_size - train_size - test_size + 1, test_size):
        train_idx = data.index[start:start + train_size]
        test_idx = data.index[start + train_size:start + train_size + test_size]
        splits.append((train_idx, test_idx))
    return splits

if __name__ == "__main__":
    # テスト用のダミーデータ
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    test_data = pd.DataFrame({
        'Close': np.random.random(100) * 100
    }, index=dates)

    # トレーニング期間: 252日、テスト期間: 63日
    splits = split_data_for_walk_forward(test_data, train_size=252, test_size=63)

    for i, (train, test) in enumerate(splits):
        print(f"Split {i + 1}:")
        print(f"Train Data: {train.index[0]} to {train.index[-1]}")
        print(f"Test Data: {test.index[0]} to {test.index[-1]}")
        print()