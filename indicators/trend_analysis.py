import pandas as pd
from .basic_indicators import calculate_sma
from .bollinger_atr import calculate_atr

def detect_trend(data: pd.DataFrame, price_column: str, lookback_period: int = 5, 
                short_period: int = 5, medium_period: int = 25, long_period: int = 75) -> str:
    """
    SMAを用いてトレンドを判定する関数。複数のポイントを考慮し、より堅牢なトレンド判定を行います。
    """
    # データのコピーを作成して元のデータを変更しないようにする
    data_copy = data.copy()
    
    # 短期、中期、長期のSMAを計算
    data_copy.loc[:, 'SMA_short'] = calculate_sma(data_copy, price_column, short_period)
    data_copy.loc[:, 'SMA_medium'] = calculate_sma(data_copy, price_column, medium_period)
    data_copy.loc[:, 'SMA_long'] = calculate_sma(data_copy, price_column, long_period)
    
    # 十分なデータがない場合はレンジ相場と判定
    if len(data_copy) < max(lookback_period, long_period):
        return "range-bound"
    
    # 直近の値を取得
    latest_data = data_copy.iloc[-lookback_period:]
    
    # SMAの傾き（方向性）を計算
    short_slope = (latest_data['SMA_short'].iloc[-1] - latest_data['SMA_short'].iloc[0]) / lookback_period
    medium_slope = (latest_data['SMA_medium'].iloc[-1] - latest_data['SMA_medium'].iloc[0]) / lookback_period
    long_slope = (latest_data['SMA_long'].iloc[-1] - latest_data['SMA_long'].iloc[0]) / lookback_period
    
    # トレンド判定の緩和：スコアリングシステム
    uptrend_score = 0
    downtrend_score = 0
    
    # 位置関係のスコア
    if latest_data['SMA_short'].iloc[-1] > latest_data['SMA_medium'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if latest_data['SMA_medium'].iloc[-1] > latest_data['SMA_long'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if latest_data[price_column].iloc[-1] > latest_data['SMA_short'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
    
    # 傾きのスコア
    if short_slope > 0:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if medium_slope > 0:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if long_slope > 0:
        uptrend_score += 1
    else:
        downtrend_score += 1
    
    # スコアに基づくトレンド判定
    if uptrend_score >= 5:  # 6点中5点以上で上昇トレンド
        return "uptrend"
    elif downtrend_score >= 5:  # 6点中5点以上で下降トレンド
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
