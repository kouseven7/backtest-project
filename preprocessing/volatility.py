#ローリング標準偏差と年率換算ボラティリティ（変動率）を計算モジュール
import numpy as np
import pandas as pd

def add_volatility(data: pd.DataFrame, daily_return_col: str = 'Daily Return', window: int = 20, trading_days: int = 252) -> pd.DataFrame:
    """
    指定されたDataFrameに対して、ローリング標準偏差と年率換算ボラティリティ（変動率）を計算して追加します。
    
    Parameters:
        data (pd.DataFrame): 'Daily Return' を含むデータフレーム
        daily_return_col (str): 日次リターンのカラム名（デフォルトは 'Daily Return'）
        window (int): ローリング標準偏差を計算する期間（日数）。デフォルトは20
        trading_days (int): 年間の営業日数。デフォルトは252
    
    Returns:
        pd.DataFrame: 'Rolling Std' と 'Annualized Volatility' のカラムが追加されたDataFrame
    """
    # ローリング標準偏差の計算
    data['Rolling Std'] = data[daily_return_col].rolling(window=window).std()
    # 年率換算のボラティリティ（変動率）の計算
    data['Annualized Volatility'] = data['Rolling Std'] * np.sqrt(trading_days)
    return data

if __name__ == '__main__':
    # テスト用のダミーデータ
    dates = pd.date_range(start="2022-01-01", periods=10, freq='B')
    test_data = pd.DataFrame({
        'Daily Return': np.random.random(10) * 0.02  # ランダムな日次リターン（例: 0～2%の範囲）
    }, index=dates)
    
    test_data = add_volatility(test_data)
    print("【ボラティリティ計算のテスト結果】")
    print(test_data[['Daily Return', 'Rolling Std', 'Annualized Volatility']])
