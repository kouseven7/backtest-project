# volume_indicators.py
import numpy as np
import pandas as pd

def calculate_volume_ma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    出来高移動平均を計算します。

    Parameters:
        data (pd.DataFrame): 'Volume'カラムを含むデータフレーム
        period (int): ローリングウィンドウ（日数）。デフォルトは20日

    Returns:
        pd.Series: 出来高移動平均
    """
    return data['Volume'].rolling(window=period).mean()

def calculate_obv(data: pd.DataFrame, price_column: str) -> pd.Series:
    """
    OBV (On Balance Volume) を計算します。
    Parameters:
        data (pd.DataFrame): 価格と出来高のデータを含むデータフレーム
        price_column (str): OBV計算に使用する価格のカラム名（例："Close"または"Adj Close"）
    Returns:
        pd.Series: OBVの算出結果
    """
    # price_column が DataFrame の場合は 1 次元の Series に変換
    price_series = data[price_column]
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    
    delta = price_series.diff()  # 前日との差分
    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv = pd.Series(direction * data['Volume'].values, index=data.index).cumsum()
    return obv


def calculate_volume_change_rate(data: pd.DataFrame) -> pd.Series:
    """
    出来高変化率（前日比のパーセンテージ変化）を計算します。

    Parameters:
        data (pd.DataFrame): 'Volume'カラムを含むデータフレーム

    Returns:
        pd.Series: 出来高変化率
    """
    return data['Volume'].pct_change()

def calculate_volume_oscillator(data: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """
    ボリュームオシレーターを計算します。
    短期（例：5日間）と長期（例：20日間）の出来高移動平均の乖離率（%）を算出します。

    Parameters:
        data (pd.DataFrame): 'Volume'カラムを含むデータフレーム
        short_period (int): 短期移動平均の日数。デフォルトは5
        long_period (int): 長期移動平均の日数。デフォルトは20

    Returns:
        pd.Series: ボリュームオシレーター（パーセンテージ）
    """
    vol_ma_short = data['Volume'].rolling(window=short_period).mean()
    vol_ma_long = data['Volume'].rolling(window=long_period).mean()
    oscillator = ((vol_ma_short - vol_ma_long) / vol_ma_long) * 100
    return oscillator

def add_volume_indicators(data: pd.DataFrame, price_column: str, short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
    """
    DataFrameに出来高関連の指標を追加します。
    追加する指標は以下の通りです：
      - 出来高移動平均（長期、例：20日）
      - OBV
      - 出来高変化率
      - 出来高移動平均（短期、例：5日）
      - ボリュームオシレーター

    Parameters:
        data (pd.DataFrame): 'Volume'と、price_column（例："Close"または"Adj Close"）を含むデータフレーム
        price_column (str): OBV計算に使用する価格カラム名
        short_period (int): 短期移動平均の日数。デフォルトは5
        long_period (int): 長期移動平均の日数。デフォルトは20

    Returns:
        pd.DataFrame: 出来高関連指標が追加されたデータフレーム
    """
    data['Volume_MA_20'] = calculate_volume_ma(data, period=long_period)
    data['OBV'] = calculate_obv(data, price_column=price_column)
    data['Volume_Change_Rate'] = calculate_volume_change_rate(data)
    data['Volume_MA_5'] = data['Volume'].rolling(window=short_period).mean()
    data['Volume_Oscillator'] = calculate_volume_oscillator(data, short_period=short_period, long_period=long_period)
    return data

if __name__ == "__main__":
    # テスト用ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    test_data = pd.DataFrame({
        'Volume': np.random.randint(1000, 10000, size=50),
        'Close': np.random.random(50) * 100
    }, index=dates)
    
    # 'Close'を価格カラムとしてOBVを計算
    test_data = add_volume_indicators(test_data, price_column='Close', short_period=5, long_period=20)
    print("【出来高関連指標の計算結果】")
    print(test_data[['Volume', 'Volume_MA_20', 'OBV', 'Volume_Change_Rate', 'Volume_MA_5', 'Volume_Oscillator']].tail())
