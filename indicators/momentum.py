# momentum.py
import pandas as pd

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """
    指定したSeriesの指数移動平均(EMA)を計算する関数

    Parameters:
        series (pd.Series): 計算対象の価格データ
        span (int): EMAの期間

    Returns:
        pd.Series: EMAの算出結果
    """
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    MACDとMACDシグナルを計算してDataFrameに追加する関数

    Parameters:
        data (pd.DataFrame): 価格データを含むDataFrame
        price_column (str): EMA/MACD計算に用いる価格カラム（例："Close"または"Adj Close"）

    Returns:
        pd.DataFrame: 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal' が追加されたDataFrame
    """
    data['EMA_12'] = calculate_ema(data[price_column], span=12)
    data['EMA_26'] = calculate_ema(data[price_column], span=26)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def calculate_stochastics(data: pd.DataFrame) -> pd.DataFrame:
    """
    ストキャスティクスオシレーター(%K, %D)を計算してDataFrameに追加する関数

    Parameters:
        data (pd.DataFrame): 'Low', 'High', 'Close' カラムを含むDataFrame

    Returns:
        pd.DataFrame: 'Stoch_%K' と 'Stoch_%D' が追加されたDataFrame
    """
    # 14日間の最低値と最高値を計算
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    data['Stoch_%K'] = ((data['Close'] - low14) / (high14 - low14)) * 100
    data['Stoch_%D'] = data['Stoch_%K'].rolling(window=3).mean()
    return data

def calculate_roc(data: pd.DataFrame, price_column: str, period: int = 12) -> pd.DataFrame:
    """
    ROC (Rate of Change) を計算してDataFrameに追加する関数

    Parameters:
        data (pd.DataFrame): 価格データを含むDataFrame
        price_column (str): ROC計算に用いる価格カラム
        period (int): ROCの計算に用いる期間（日数）。デフォルトは12

    Returns:
        pd.DataFrame: 'ROC_12' が追加されたDataFrame
    """
    data['ROC_12'] = (data[price_column] / data[price_column].shift(period) - 1) * 100
    return data

def add_momentum_indicators(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    補助的なモメンタム指標（EMA、MACD、ストキャスティクス、ROC）をDataFrameに追加する関数

    Parameters:
        data (pd.DataFrame): 'Low', 'High', 'Close'（または'Adj Close'）などのカラムを含むDataFrame
        price_column (str): 指標計算に使用する価格カラム（例："Close"または"Adj Close"）

    Returns:
        pd.DataFrame: 各モメンタム指標が追加されたDataFrame
    """
    data = calculate_macd(data, price_column)
    data = calculate_stochastics(data)
    data = calculate_roc(data, price_column, period=12)
    return data

if __name__ == '__main__':
    # テスト用コード：ダミーデータで各指標の計算結果を確認
    import numpy as np
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    test_data = pd.DataFrame({
        'Close': np.random.random(50) * 100,
        'Low': np.random.random(50) * 100,
        'High': np.random.random(50) * 100
    }, index=dates)
    
    # DataFrame内のデータ順序の整合性（High >= Low となるようにソートするなどの前処理が必要な場合もあります）
    test_data = add_momentum_indicators(test_data, price_column='Close')
    print("【補助的なモメンタム指標計算結果】")
    print(test_data[['Close', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'Stoch_%K', 'Stoch_%D', 'ROC_12']].tail())
