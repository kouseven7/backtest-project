# bollinger_atr.py
import pandas as pd

def calculate_bollinger_bands(data: pd.DataFrame, price_column: str, window: int = 20, k: float = 2) -> pd.DataFrame:
    """
    指定したDataFrameにボリンジャーバンドを計算して追加します。

    Parameters:
        data (pd.DataFrame): 価格データを含むDataFrame。price_columnが存在すること。
        price_column (str): ボリンジャーバンドの計算に使用する価格のカラム名（例："Close" または "Adj Close"）。
        window (int): 移動平均および標準偏差を計算するローリングウィンドウ（日数）。デフォルトは20。
        k (float): 標準偏差の係数（バンド幅係数）。デフォルトは2。

    Returns:
        pd.DataFrame: 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower'が追加されたDataFrame。
    """
    data['BB_Middle'] = data[price_column].rolling(window=window).mean()
    data['BB_Std'] = data[price_column].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + k * data['BB_Std']
    data['BB_Lower'] = data['BB_Middle'] - k * data['BB_Std']
    return data

def calculate_atr(data: pd.DataFrame, price_column: str, atr_period: int = 14) -> pd.DataFrame:
    """
    指定したDataFrameにATR（Average True Range）を計算して追加します。

    Parameters:
        data (pd.DataFrame): 'High', 'Low'、およびprice_column（例："Close" または "Adj Close"）を含むDataFrame。
        price_column (str): ATR計算に使用する価格のカラム名。
        atr_period (int): True Rangeのローリング平均を計算する期間。デフォルトは14。

    Returns:
        pd.DataFrame: 'ATR'が追加されたDataFrame。
    """
    # True Rangeの各構成要素を計算
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data[price_column].shift(1)).abs()
    data['L-PC'] = (data['Low'] - data[price_column].shift(1)).abs()
    
    # 上記3つの中で最大の値をTrue Rangeとして採用
    data['True Range'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # ATRはTrue Rangeの指定期間の移動平均
    data['ATR'] = data['True Range'].rolling(window=atr_period).mean()
    
    # 一時的な計算用カラムを削除（不要な場合）
    data.drop(columns=['H-L', 'H-PC', 'L-PC'], inplace=True)
    
    return data

# テスト用コード：このモジュールを直接実行した場合にダミーデータで計算結果を確認
if __name__ == '__main__':
    import numpy as np
    # ダミーデータの作成（例：50営業日分のデータ）
    dates = pd.date_range(start="2022-01-01", periods=50, freq='B')
    test_data = pd.DataFrame({
        'High': np.random.random(50) * 100,
        'Low': np.random.random(50) * 100,
        'Close': np.random.random(50) * 100,
        'Adj Close': np.random.random(50) * 100,
        'Volume': np.random.randint(1000, 10000, size=50)
    }, index=dates)
    
    # ここでは 'Close' を使用して計算
    test_data = calculate_bollinger_bands(test_data, price_column='Close')
    test_data = calculate_atr(test_data, price_column='Close')
    print(test_data[['Close', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR']].tail())

# エイリアスを追加して、main.py で bollinger_atr としてインポート可能にする
bollinger_atr = calculate_bollinger_bands
