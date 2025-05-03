"""
Module: Test for Train-Test Split Function
File: test_train_test_split.py
Description:
  ウォークフォワードテスト用のデータ分割関数 (split_data_for_walk_forward) のテストスクリプト。

Purpose:
  - 境界条件や一般的なケースでの関数の動作を検証する。
  - データが不足している場合や、トレーニング期間とテスト期間が異なる場合の挙動を確認する。

Test Cases:
  1. データが不足している場合:
     - データ長がトレーニング期間 + テスト期間より短い場合。
     - 期待結果: 分割数が 0、警告が表示される。

  2. データがちょうどトレーニング + テスト期間の場合:
     - データ長がトレーニング期間 + テスト期間と同じ場合。
     - 期待結果: 分割数が 1。

  3. データが少し長い場合:
     - データ長がトレーニング期間 + テスト期間より少し長い場合。
     - 期待結果: 分割数が 1（余剰データは無視される）。

  4. データが中途半端な長さの場合:
     - データ長がトレーニング期間 + テスト期間の倍数ではない場合。
     - 期待結果: 分割数が複数（余剰データは無視される）。

  5. データが倍数の場合:
     - データ長がトレーニング期間 + テスト期間の正確な倍数の場合。
     - 期待結果: 分割数が正確に計算される。

  6. データが非常に少ない場合:
     - データ長がトレーニング期間 + テスト期間よりも極端に短い場合。
     - 期待結果: 分割数が 0 または 1（条件に応じて）。

Author: imega
Created: 2025-05-03
"""

import pandas as pd
import numpy as np
from train_test_split import split_data_for_walk_forward

# テストケース
test_cases = [
    {"data_length": 200, "train_size": 252, "test_size": 63, "description": "データが不足している場合"},
    {"data_length": 315, "train_size": 252, "test_size": 63, "description": "データがちょうどトレーニング+テスト期間の場合"},
    {"data_length": 316, "train_size": 252, "test_size": 63, "description": "データが少し長い場合"},
    {"data_length": 500, "train_size": 252, "test_size": 63, "description": "データが中途半端な長さの場合"},
    {"data_length": 630, "train_size": 252, "test_size": 63, "description": "データが倍数の場合"},
    {"data_length": 10, "train_size": 5, "test_size": 5, "description": "データが非常に少ない場合"},
]

# テスト実行
for case in test_cases:
    print(f"--- {case['description']} ---")
    dates = pd.date_range(start="2022-01-01", periods=case["data_length"], freq='B')
    test_data = pd.DataFrame({
        'Close': np.random.random(case["data_length"]) * 100
    }, index=dates)

    splits = split_data_for_walk_forward(test_data, train_size=case["train_size"], test_size=case["test_size"])
    print(f"データ長: {case['data_length']}, 分割数: {len(splits)}")
    for i, (train, test) in enumerate(splits):
        print(f"  Split {i + 1}: Train {train.index[0]} to {train.index[-1]}, Test {test.index[0]} to {test.index[-1]}")
    print()