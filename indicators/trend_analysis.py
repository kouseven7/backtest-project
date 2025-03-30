import pandas as pd
from .basic_indicators import calculate_sma
from .bollinger_atr import calculate_atr

def detect_trend(data: pd.DataFrame, price_column: str) -> str:
    """
    SMAを用いてトレンドを判定する関数。

    Args:
        data (pd.DataFrame): 株価データを含むDataFrame。
        price_column (str): SMA計算に使用する価格カラム名。

    Returns:
        str: "uptrend"（上昇トレンド）、"downtrend"（下降トレンド）、または"range-bound"（レンジ相場）。
    """
    # 短期、中期、長期のSMAを計算
    data['SMA_5'] = calculate_sma(data, price_column, 5)
    data['SMA_25'] = calculate_sma(data, price_column, 25)
    data['SMA_75'] = calculate_sma(data, price_column, 75)

    # 最新のSMA値を取得
    latest_sma_5 = data['SMA_5'].iloc[-1]
    latest_sma_25 = data['SMA_25'].iloc[-1]
    latest_sma_75 = data['SMA_75'].iloc[-1]
    latest_price = data[price_column].iloc[-1]

    # トレンド判定 上昇トレンド、下降トレンド、レンジ相場
    # 上昇トレンド: 最新価格 > SMA5 > SMA25 > SMA75
    if latest_price > latest_sma_5 > latest_sma_25 > latest_sma_75:
        return "uptrend"
    elif latest_price < latest_sma_5 < latest_sma_25 < latest_sma_75:
        return "downtrend"
    else:
        return "range-bound"

def detect_high_volatility(data: pd.DataFrame, price_column: str, atr_threshold: float) -> str:
    """
    ATRを用いて高ボラティリティ相場を判定する関数。

    Args:
        data (pd.DataFrame): 株価データを含むDataFrame。
        price_column (str): ATR計算に使用する価格カラム名。
        atr_threshold (float): 高ボラティリティを判定するATRの閾値。

    Returns:
        str: "high volatility"（高ボラティリティ）または"normal volatility"（通常のボラティリティ）。
    """
    # ATRを計算
    data = calculate_atr(data, price_column)

    # 最新のATR値を取得
    latest_atr = data['ATR'].iloc[-1]

    # 高ボラティリティ判定
    if latest_atr > atr_threshold:
        return "high volatility"
    else:
        return "normal volatility"

# テストコード
if __name__ == "__main__":
    # ダミーのデータ作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': [i + 5 for i in range(100)],
        'Low': [i for i in range(100)],
        'Adj Close': [i + (i % 5) * 2 for i in range(100)]  # ダミー価格データ
    }, index=dates)

    trend = detect_trend(df, price_column='Adj Close')
    print(f"トレンド判定: {trend}")

    volatility = detect_high_volatility(df, price_column='Adj Close', atr_threshold=10)
    print(f"ボラティリティ判定: {volatility}")
