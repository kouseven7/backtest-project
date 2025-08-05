"""
Module: Volatility Indicators
File: volatility_indicators.py
Description: 
  ボラティリティ指標（ATRなど）を計算するモジュールです。
  株価データに基づいてボラティリティを測定します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
"""

import pandas as pd

def calculate_atr(data: pd.DataFrame, price_column: str, period: int = 14) -> pd.Series:
    """
    ATR（平均的な値幅）を計算する。

    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 株価カラム名
        period (int): ATRの計算期間（デフォルトは14）

    Returns:
        pd.Series: ATR値
    """
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data[price_column].shift()).abs()
    low_close = (data['Low'] - data[price_column].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr