"""
Module: test_data
File: test_data.py
Description: 
  様々な戦略のテスト用データを生成するユーティリティモジュール。
  各戦略の挙動をテストするための特定のパターンを持ったデータを提供します。

Author: kouseven7
Created: 2025-04-05
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_vwap_bounce_test_data(days=100, start_date=None):
    """
    VWAP_Bounce戦略が反応するテストデータを生成します。
    VWAP_Bounce戦略は、価格がVWAP（出来高加重平均価格）に近づいた後に反発するパターンを検出します。
    
    Parameters:
        days (int): 生成するデータの日数
        start_date (str): 開始日（デフォルトは現在の日付-days日）
        
    Returns:
        pd.DataFrame: テスト用の株価データ
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 日付インデックスの作成
    dates = pd.date_range(start=start_date, periods=days, freq='B')
    
    # 基本的な価格トレンドを作成（上昇トレンド + ノイズ）
    base_price = 100
    trend = np.linspace(0, 30, days)  # 0から30までの直線（上昇トレンド）
    noise = np.random.normal(0, 2, days)  # ランダムノイズ
    prices = base_price + trend + noise
    
    # VWAP_Bounce戦略に反応するパターンを作成
    # 1. 価格がVWAP付近に下落する期間を作成
    vwap_pattern_start = days // 3
    vwap_pattern_length = 5
    vwap_values = prices.copy()
    
    # 30日おきにVWAP付近への下落と反発パターンを追加
    for i in range(vwap_pattern_start, days, 30):
        if i + vwap_pattern_length >= days:
            break
        
        # VWAPに近づく（下落）パターン
        prices[i:i+3] = prices[i] * 0.98  # 価格を少し下げてVWAPに近づける
        
        # VWAPから反発するパターン
        prices[i+3:i+vwap_pattern_length] = prices[i+3] * 1.03  # 価格を上げて反発
    
    # データフレームの作成
    df = pd.DataFrame({
        'Open': prices * 0.99,  # 始値は終値より少し低く
        'High': prices * 1.02,  # 高値は終値より少し高く
        'Low': prices * 0.98,   # 安値は終値より少し低く
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000, 10000, days)  # 出来高
    }, index=dates)
    
    # VWAPの計算を事前に行い、データに追加
    df['VWAP'] = calculate_vwap(df)
    
    # テスト対象の戦略が使用するインジケーターを事前に計算
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    return df

def calculate_vwap(data):
    """
    VWAPを計算します。
    
    Parameters:
        data (pd.DataFrame): 株価データ
        
    Returns:
        pd.Series: VWAPの値
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def add_test_data_to_strategy(strategy_name='VWAP_Bounce'):
    """
    指定された戦略のテストデータを生成し、既存のデータセットに統合します。
    
    Parameters:
        strategy_name (str): 生成するテストデータの対象戦略名
        
    Returns:
        pd.DataFrame: テスト用の株価データ
    """
    if strategy_name == 'VWAP_Bounce':
        return generate_vwap_bounce_test_data()
    else:
        raise ValueError(f"未対応の戦略名: {strategy_name}")

# テスト実行
if __name__ == "__main__":
    test_data = generate_vwap_bounce_test_data()
    print(test_data.head())
    print("VWAPバウンス戦略用テストデータの生成完了。")
