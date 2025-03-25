# gap_indicators.py
import pandas as pd

def add_gap_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame にギャップ関連の指標を追加します。
    
    具体的には、前日の高値、安値、始値を抽出し、
    当日の始値と前日の各値との差分の割合（パーセンテージ）を計算して、
    'Gap_High_Rate', 'Gap_Low_Rate', 'Gap_Open_Rate' として追加します。
    
    Parameters:
        data (pd.DataFrame): 'High', 'Low', 'Open' のカラムを含むデータフレーム
        
    Returns:
        pd.DataFrame: ギャップ関連指標が追加された DataFrame
    """
    # --- 必要に応じてカラムをフラット化 ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # --- 前日の高値、安値、始値を抽出 ---
    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    data['Prev_Open'] = data['Open'].shift(1)
    
    # --- ギャップ率の計算（パーセンテージ表示） ---
    data['Gap_High_Rate'] = (data['Open'] - data['Prev_High']) / data['Prev_High'] * 100
    data['Gap_Low_Rate'] = (data['Open'] - data['Prev_Low']) / data['Prev_Low'] * 100
    data['Gap_Open_Rate'] = (data['Open'] - data['Prev_Open']) / data['Prev_Open'] * 100
    
    return data

if __name__ == '__main__':
    # テスト用のダミーデータ
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=10, freq='B')
    test_data = pd.DataFrame({
        'High': np.random.rand(10) * 100,
        'Low': np.random.rand(10) * 100,
        'Open': np.random.rand(10) * 100,
        'Close': np.random.rand(10) * 100
    }, index=dates)
    
    test_data = add_gap_indicators(test_data)
    print("【ギャップ関連指標テスト結果】")
    print(test_data[['Prev_High', 'Prev_Low', 'Prev_Open', 'Gap_High_Rate', 'Gap_Low_Rate', 'Gap_Open_Rate']])
