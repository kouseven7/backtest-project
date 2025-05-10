import pandas as pd
import preprocessing.returns as returns
import preprocessing.volatility as volatility
import logging

logger = logging.getLogger(__name__)

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

    # リターンとボラティリティの計算
    data = returns.add_returns(data, price_column="Adj Close")
    data = volatility.add_volatility(data)
    logger.info("前処理（リターン、ボラティリティ計算）完了")
    
    return data