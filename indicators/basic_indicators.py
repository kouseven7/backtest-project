# basic_indicators.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    try:
        if column not in data.columns:
            logger.error(f"カラム '{column}' がデータフレームに存在しません。利用可能なカラム: {data.columns.tolist()}")
            raise KeyError(f"カラム '{column}' がデータフレームに存在しません。")
        
        # カラムが数値型かチェック
        if not pd.api.types.is_numeric_dtype(data[column]):
            logger.warning(f"カラム '{column}' が数値型ではありません。型変換を試みます。現在の型: {data[column].dtype}")
            try:
                # 数値型に変換を試みる
                data[column] = pd.to_numeric(data[column], errors='coerce')
                logger.info(f"カラム '{column}' を数値型に変換しました。")
            except Exception as e:
                logger.error(f"カラム '{column}' の数値変換に失敗しました: {e}")
                # 失敗した場合はデフォルト値（NaN）のSeriesを返す
                return pd.Series(index=data.index, dtype=float)
        
        return data[column].rolling(window=window).mean()
    except Exception as e:
        logger.error(f"SMA計算中にエラーが発生しました: {e}")
        # エラーが発生した場合はデフォルト値（NaN）のSeriesを返す
        return pd.Series(index=data.index, dtype=float)

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（相対力指数）を計算する。

    Parameters:
        data (pd.Series): 株価データ（通常は終値）
        period (int): RSIの計算期間（デフォルトは14）

    Returns:
        pd.Series: RSI値
    """
    try:
        # データが数値型かチェック
        if not pd.api.types.is_numeric_dtype(data):
            data = pd.to_numeric(data, errors='coerce')
            
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # ゼロ除算を防ぐ
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logger.error(f"RSI計算中にエラーが発生しました: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_vwap(data: pd.DataFrame, price_column: str, volume_column: str) -> pd.Series:
    """
    VWAPを計算する。

    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 株価カラム名
        volume_column (str): 出来高カラム名

    Returns:
        pd.Series: VWAP値
    """
    try:
        # 必要なカラムが存在するか確認
        required_columns = ['High', 'Low', price_column, volume_column]
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"カラム '{col}' がデータフレームに存在しません。利用可能なカラム: {data.columns.tolist()}")
                raise KeyError(f"カラム '{col}' がデータフレームに存在しません。")
            
            # 数値型に変換
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        typical_price = (data['High'] + data['Low'] + data[price_column]) / 3
        cumulative_vwap = (typical_price * data[volume_column]).cumsum()
        cumulative_volume = data[volume_column].cumsum()
        
        # ゼロ除算を防ぐ
        cumulative_volume = cumulative_volume.replace(0, np.nan)
        return cumulative_vwap / cumulative_volume
    except Exception as e:
        logger.error(f"VWAP計算中にエラーが発生しました: {e}")
        return pd.Series(index=data.index, dtype=float)

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
    stock_data['VWAP'] = calculate_vwap(stock_data, price_column=price_column, volume_column='Volume')
    
    return stock_data

class VWAPBounceStrategy:
    def __init__(self, data: pd.DataFrame):
        """
        VWAP反発戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
        """
        self.data = data

        # VWAPを計算してデータに追加
        self.data['VWAP'] = calculate_vwap(self.data, price_column='Close', volume_column='Volume')
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], 14)

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
