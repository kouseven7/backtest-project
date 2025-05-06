"""
Module: Returns Calculation
File: returns.py
Description: 
  株価データからリターンを計算するためのモジュールです。
  日次リターン、累積リターン、対数リターンなどを計算します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - numpy
"""

#日次リターンと累積リターンを計算して追加モジュール
# returns.py
import pandas as pd
import numpy as np
import logging

def add_returns(data, price_column="Adj Close"):
    """
    株価データにリターン（収益率）列を追加する
    
    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 株価列の名前
        
    Returns:
        pd.DataFrame: リターン列が追加された株価データ
    """
    # 'Adj Close'がなく'Close'がある場合は'Close'を使用
    if price_column not in data.columns and 'Close' in data.columns:
        data[price_column] = data['Close']
        logging.info(f"'{price_column}'カラムがないため、'Close'カラムで代用します")
    
    data[price_column] = pd.to_numeric(data[price_column], errors='coerce')
    
    # 日次リターンの計算
    data['Daily Return'] = data[price_column].pct_change()
    
    # 対数リターンの計算
    data['Log Return'] = np.log(data[price_column] / data[price_column].shift(1))
    
    # リターンのローリング統計量
    data['Return_MA5'] = data['Daily Return'].rolling(window=5).mean()
    data['Return_Std5'] = data['Daily Return'].rolling(window=5).std()
    
    return data


if __name__ == "__main__":
    # テスト用コード：ダミーデータで計算結果を確認
    dates = pd.date_range(start="2022-01-01", periods=10, freq='B')
    test_data = pd.DataFrame({
        'Close': np.random.random(10) * 100
    }, index=dates)
    
    test_data = add_returns(test_data, price_column='Close')
    print("【日次リターン、累積リターンのテスト結果】")
    print(test_data[['Close', 'Daily Return', 'Log Return', 'Return_MA5', 'Return_Std5']])
