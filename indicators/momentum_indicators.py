import pandas as pd
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

def calculate_macd(data: pd.DataFrame, price_column: str, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    """
    MACD（移動平均収束拡散法）を計算する。

    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 株価カラム名
        short_window (int): 短期EMAの期間（デフォルトは12）
        long_window (int): 長期EMAの期間（デフォルトは26）
        signal_window (int): シグナルラインの期間（デフォルトは9）

    Returns:
        tuple: MACDライン（pd.Series）とシグナルライン（pd.Series）
    """
    short_ema = data[price_column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[price_column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line