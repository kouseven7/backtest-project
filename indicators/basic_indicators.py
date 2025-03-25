# basic_indicators.py
import pandas as pd
import numpy as np

def calculate_sma(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    指定したカラムの単純移動平均 (SMA) を計算する関数

    Parameters:
        data (pd.DataFrame): 指標計算対象のデータフレーム
        column (str): 移動平均を計算する対象のカラム名（例：'Close' または 'Adj Close'）
        window (int): 移動平均の期間（日数）

    Returns:
        pd.Series: 移動平均の算出結果
    """
    return data[column].rolling(window=window).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（Relative Strength Index）を計算する関数

    Parameters:
        series (pd.Series): 価格データのSeries（例：終値）
        period (int): RSIの計算期間（デフォルトは14日）

    Returns:
        pd.Series: RSIの算出結果
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    VWAP（Volume Weighted Average Price）を計算する関数

    VWAPは典型価格（(High + Low + Close) / 3）と出来高を用いて、
    指定期間内の加重平均として算出されます。

    Parameters:
        data (pd.DataFrame): 銘柄データ。'High', 'Low', 'Close', 'Volume' のカラムが必要
        window (int): VWAPの計算に用いるローリングウィンドウ（日数）

    Returns:
        pd.Series: VWAPの算出結果
    """
    # 対象カラムを数値型に変換（変換できない場合は NaN に）
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    # VWAPの計算
    vwap = (typical_price * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return vwap


def add_basic_indicators(stock_data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    DataFrame に基本インジケーター（SMA、RSI、VWAP）を追加する関数

    Parameters:
        stock_data (pd.DataFrame): 銘柄の株価データ。必要なカラムは、price_column, 'High', 'Low', 'Close', 'Volume'
        price_column (str): 移動平均やRSIの計算に使用する価格カラム名（例："Adj Close" または "Close"）

    Returns:
        pd.DataFrame: 基本インジケーターが追加された DataFrame
    """
    # 移動平均の計算
    stock_data['SMA_5'] = calculate_sma(stock_data, price_column, 5)
    stock_data['SMA_25'] = calculate_sma(stock_data, price_column, 25)
    stock_data['SMA_75'] = calculate_sma(stock_data, price_column, 75)
    
    # RSIの計算
    stock_data['RSI_14'] = calculate_rsi(stock_data[price_column], period=14)
    
    # VWAPの計算（20日間のローリングウィンドウ）
    stock_data['VWAP'] = calculate_vwap(stock_data, window=20)
    
    return stock_data

# テスト用コード（このファイルを直接実行した場合のみ動作）
if __name__ == "__main__":
    # ダミーのデータ作成（実際はyfinanceなどで取得したstock_dataを使用）
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'Adj Close': np.random.random(100) * 100,
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Close': np.random.random(100) * 100,
        'Volume': np.random.randint(1000, 10000, size=100)
    }, index=dates)
    
    # ここでは 'Adj Close' を使用
    df = add_basic_indicators(df, price_column='Adj Close')
    print(df.tail())
