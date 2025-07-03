import pandas as pd
import preprocessing.returns as returns
import preprocessing.volatility as volatility
import logging

logger = logging.getLogger(__name__)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR（Average True Range）を計算する関数"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)  # 前日の終値
    
    # True Range計算
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """データの前処理を行う関数"""
    
    # インデックスのチェックと修正
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning(f"インデックスが日付型ではありません: {data.index[0]}")
        
        # 'Ticker'などのヘッダー行がインデックスになっている場合の対処
        if data.index[0] == 'Ticker' or 'Date' in data.columns:
            logger.info("ヘッダー行が含まれています。インデックスをリセットして再設定します。")
            # インデックスをリセットして、Date列があればそれをインデックスに設定
            data = data.reset_index(drop=True)
            if 'Date' in data.columns:
                data = data.set_index('Date')
                
        # インデックスを日付型に変換
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"インデックスを日付型に変換できませんでした: {e}")
            raise ValueError("日付型に変換できないデータです。データ形式を確認してください。")
    
    # 数値型カラムへの変換
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 欠損値を補完または削除
    data = data.fillna(method='ffill').fillna(method='bfill')

    # ATRを計算
    if all(col in data.columns for col in ['High', 'Low', 'Close']):
        data['ATR'] = calculate_atr(data)
        logger.info("ATR計算完了")
    else:
        logger.warning("High, Low, Closeカラムが揃っていないためATRを計算できません")

    # リターンとボラティリティの計算
    data = returns.add_returns(data, price_column="Adj Close")
    data = volatility.add_volatility(data)
    logger.info("前処理（リターン、ボラティリティ計算）完了")
    
    return data