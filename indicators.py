# indicators.py

#テクニカル指標計算の関数群をまとめたモジュールです。
#ここでは、単純移動平均（SMA）、追加指標（ボラティリティ、ATR ）、出来高指標を実装しています。

import pandas as pd

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均 (SMA) を計算する関数"""
    return series.rolling(window=window).mean()

def calc_additional_indicators(data: pd.DataFrame, vol_window: int = 20, atr_window: int = 14) -> pd.DataFrame:
    """
    追加指標の計算:
      - ボラティリティ: 'Adj Close' があればそれ、なければ 'Close' の日次パーセント変化率のローリング標準偏差
      - High-Low レンジ、True Range、ATR の計算
    """
    # 列名をすべて小文字かつ空白除去したリストを作成
    normalized_cols = {col.lower().replace(" ", ""): col for col in data.columns}
    
    # "adjclose" というキーがあれば、それを価格列として使用。なければ "close" を使う
    if "adjclose" in normalized_cols:
        price_col = normalized_cols["adjclose"]
    elif "close" in normalized_cols:
        price_col = normalized_cols["close"]
    else:
        raise KeyError("価格データとして利用できる 'Adj Close' も 'Close' も存在しません。")
    
    # 日次リターンとボラティリティの計算
    data['Daily Return'] = data[price_col].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=vol_window).std()
    
    # High-Low レンジの計算
    data['Range'] = data['High'] - data['Low']
    
    # True Range の計算に必要な前日終値
    data['Previous Close'] = data['Close'].shift(1)
    data['TR1'] = data['High'] - data['Low']
    data['TR2'] = (data['High'] - data['Previous Close']).abs()
    data['TR3'] = (data['Low'] - data['Previous Close']).abs()
    data['True Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # ATR の計算
    data['ATR'] = data['True Range'].rolling(window=atr_window).mean()
    
    # 中間計算用の列を削除
    data.drop(columns=['Previous Close', 'TR1', 'TR2', 'TR3'], inplace=True)
    
    return data

def calc_volume_indicators(data: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    出来高関連指標の計算:
      - 出来高変動率: Volume の日次パーセント変化率
      - 出来高移動平均: Volume の単純移動平均 (SMA)
    """
    data['Volume Change'] = data['Volume'].pct_change()
    data['Volume SMA'] = calc_sma(data['Volume'], vol_window)
    return data
