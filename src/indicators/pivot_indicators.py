"""
Module: Pivot Indicators
File: pivot_indicators.py
Description: 
  ピボットポイントやボリンジャーバンド関連の指標を計算するモジュールです。
  サポート・レジスタンスレベルやバンド幅などを提供します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - numpy
"""

# pivot_indicators.py
import pandas as pd

def add_pivot_and_bb_indicators(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    ピボットポイント、サポート・レジスタンスレベル、及び
    ボリンジャーバンド派生指標（バンド幅、%B）をDataFrameに追加します。
    
    この関数は、以下のカラムが既に存在していることを前提とします：
      - 'Prev_High'：前日の高値
      - 'Prev_Low' : 前日の安値
      - 'BB_Middle', 'BB_Upper', 'BB_Lower'：ボリンジャーバンドの計算結果
    また、price_column（例："Close"または"Adj Close"）を使用して計算を行います。
    
    Parameters:
        data (pd.DataFrame): 入力の株価データ。必要なカラムは、
                             'Prev_High', 'Prev_Low', 'BB_Middle', 'BB_Upper', 'BB_Lower'、
                             およびprice_columnで指定するカラム。
        price_column (str): ピボットポイントや%Bの計算に用いる価格カラム名。
    
    Returns:
        pd.DataFrame: 指標が追加されたDataFrame
    """
    # 前日のCloseが存在しない場合は追加
    if 'Prev_Close' not in data.columns:
        data['Prev_Close'] = data[price_column].shift(1)
    
    # ピボットポイントの算出（前日のHigh, Low, Closeを使用）
    data['Pivot'] = (data['Prev_High'] + data['Prev_Low'] + data['Prev_Close']) / 3
    
    # サポート・レジスタンスレベルの計算
    data['R1'] = 2 * data['Pivot'] - data['Prev_Low']
    data['S1'] = 2 * data['Pivot'] - data['Prev_High']
    data['R2'] = data['Pivot'] + (data['Prev_High'] - data['Prev_Low'])
    data['S2'] = data['Pivot'] - (data['Prev_High'] - data['Prev_Low'])
    # （オプション）さらに3段階まで
    data['R3'] = data['Prev_High'] + 2 * (data['Pivot'] - data['Prev_Low'])
    data['S3'] = data['Prev_Low'] - 2 * (data['Prev_High'] - data['Pivot'])
    
    # ボリンジャーバンド派生指標の計算
    # ボリンジャーバンド幅（%）：中心線に対する上部・下部の幅の割合
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle'] * 100
    # %Bの算出：現在の価格がバンド内のどの位置にあるか
    data['Percent_B'] = (data[price_column] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower']) * 100
    
    return data

if __name__ == '__main__':
    # テスト用ダミーデータの作成
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=10, freq='B')
    test_data = pd.DataFrame({
        'Prev_High': np.random.random(10) * 100,
        'Prev_Low': np.random.random(10) * 100,
        'Close': np.random.random(10) * 100,
        'BB_Middle': np.random.random(10) * 100 + 50,   # 仮の値（50～150程度）
        'BB_Upper': np.random.random(10) * 100 + 150,     # 仮に中間値より上になるように
        'BB_Lower': np.random.random(10) * 50            # 仮に中間値より下になるように
    }, index=dates)
    
    # モジュール関数を適用
    test_data = add_pivot_and_bb_indicators(test_data, price_column='Close')
    
    # テスト結果の確認
    print("【ピボットポイント・サポート・レジスタンスおよびボリンジャーバンド派生指標 テスト結果】")
    print(test_data[['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'BB_Width', 'Percent_B']])
