#日次リターンと累積リターンを計算して追加モジュール
# returns.py
import pandas as pd

def add_returns(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    与えられたDataFrameに対して、日次リターンと累積リターンを計算して追加します。

    Parameters:
        data (pd.DataFrame): 株価データを含むDataFrame。price_column（例："Close" や "Adj Close"）が存在する必要があります。
        price_column (str): 日次リターン計算に使用するカラム名。

    Returns:
        pd.DataFrame: 'Daily Return' と 'Cumulative Return' カラムが追加されたDataFrame。
    """
    # 数値型に変換（変換できない値は NaN に）
    data[price_column] = pd.to_numeric(data[price_column], errors='coerce')
    # 日次リターンの計算（'Daily Return' として追加）
    data['Daily Return'] = data[price_column].pct_change(fill_method=None)
    # 累積リターンの計算
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod() - 1
    return data


if __name__ == "__main__":
    # テスト用コード：ダミーデータで計算結果を確認
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=10, freq='B')
    test_data = pd.DataFrame({
        'Close': np.random.random(10) * 100
    }, index=dates)
    
    test_data = add_returns(test_data, price_column='Close')
    print("【日次リターン、累積リターンのテスト結果】")
    print(test_data[['Close', 'Daily Return', 'Cumulative Return']])
