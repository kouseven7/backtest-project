import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from indicators.basic_indicators import calculate_sma, calculate_vwap
from indicators.volume_analysis import detect_volume_increase
from indicators.momentum_indicators import calculate_macd
from indicators.basic_indicators import calculate_rsi

class VWAPBreakoutStrategy:
    def __init__(self, data: pd.DataFrame, index_data: pd.DataFrame, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        VWAPアウトブレイク戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            index_data (pd.DataFrame): 市場全体のインデックスデータ
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        self.data = data
        self.index_data = index_data
        self.price_column = price_column
        self.volume_column = volume_column

        # 必要なインジケーターを計算してデータに追加
        self.data['SMA_20'] = calculate_sma(self.data, price_column, 20)
        self.data['SMA_50'] = calculate_sma(self.data, price_column, 50)
        self.data['VWAP'] = calculate_vwap(self.data, price_column, volume_column)
        self.data['RSI'] = calculate_rsi(self.data[price_column], 14)
        self.data['MACD'], self.data['Signal_Line'] = calculate_macd(self.data, price_column)

        # 市場全体のトレンドを確認するためのインデックスの移動平均線
        self.index_data['SMA_20'] = calculate_sma(self.index_data, price_column, 20)
        self.index_data['SMA_50'] = calculate_sma(self.index_data, price_column, 50)

    def is_market_uptrend(self, idx: int) -> bool:
        """
        市場全体が上昇トレンドにあるかを確認する。

        Parameters:
            idx (int): 現在のインデックス

        Returns:
            bool: 市場全体が上昇トレンドにある場合は True、それ以外は False
        """
        if idx < 50:  # 50日分のデータが必要
            return False

        index_price = self.index_data[self.price_column].iloc[idx]
        index_sma_20 = self.index_data['SMA_20'].iloc[idx]
        index_sma_50 = self.index_data['SMA_50'].iloc[idx]

        # 市場全体が上昇トレンドにある条件
        return index_price > index_sma_20 > index_sma_50 and \
               self.index_data['SMA_20'].iloc[idx] > self.index_data['SMA_20'].iloc[idx - 1] and \
               self.index_data['SMA_50'].iloc[idx] > self.index_data['SMA_50'].iloc[idx - 1]

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        """
        if idx < 50:  # 50日分のデータが必要
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        sma_20 = self.data['SMA_20'].iloc[idx]
        sma_50 = self.data['SMA_50'].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        previous_vwap = self.data['VWAP'].iloc[idx - 1]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # 市場全体が上昇トレンドにあるか確認
        if not self.is_market_uptrend(idx):
            return 0

        # 株価が20日移動平均線や50日移動平均線の上に位置している
        if not (current_price > sma_20 > sma_50):
            return 0

        # 移動平均線が上昇している
        if not (sma_20 > self.data['SMA_20'].iloc[idx - 1] and sma_50 > self.data['SMA_50'].iloc[idx - 1]):
            return 0

        # VWAPが上昇している
        if not (vwap > previous_vwap):
            return 0

        # VWAPを上抜けしている
        if not (current_price > vwap and self.data[self.price_column].iloc[idx - 1] <= vwap):
            return 0

        # 出来高が増加している
        if not detect_volume_increase(current_volume, previous_volume, threshold=1.2):
            return 0

        return 1

    def generate_exit_signal(self, entry_price: float, current_price: float, atr: float, idx: int) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - VWAPを下回った場合
        - エントリー価格から3～5%の下落でストップロス
        - 利益が伸びた場合、トレーリングストップを設定
        - RSIやMACDの反転、または短期移動平均線のブレイクダウン

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
        vwap = self.data['VWAP'].iloc[idx]

        # VWAPを下回った場合
        if current_price < vwap:
            return -1

        # ストップロス条件（ATRベースまたはパーセンテージベース）
        if current_price <= entry_price - atr or current_price <= entry_price * 0.95:
            return -1

        # 利益確定条件（目標利益）
        if current_price >= entry_price * 1.10:
            return -1

        # トレーリングストップ条件
        trailing_stop = self.data['High'].iloc[:idx].max() * 0.97  # 直近高値の3%下
        if current_price <= trailing_stop:
            return -1

        # RSIやMACDの反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:  # RSIが70以上から急落
            return -1
        if macd < signal_line:  # MACDがシグナルラインを下抜け
            return -1

        return 0

    def backtest(self):
        """
        VWAPアウトブレイク戦略のバックテストを実行する。
        """
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        entry_price = None

        for idx in range(len(self.data)):
            if entry_price is None:
                # エントリーシグナルを確認
                entry_signal = self.generate_entry_signal(idx)
                self.data.at[idx, 'Entry_Signal'] = entry_signal
                if entry_signal == 1:
                    entry_price = self.data[self.price_column].iloc[idx]
            else:
                # エグジットシグナルを確認
                current_price = self.data[self.price_column].iloc[idx]
                atr = self.data['VWAP'].iloc[idx] * 0.02  # ATRの代わりにVWAPの2%を使用
                exit_signal = self.generate_exit_signal(entry_price, current_price, atr, idx)
                self.data.at[idx, 'Exit_Signal'] = exit_signal
                if exit_signal == -1:
                    entry_price = None

        return self.data

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=200, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(200) * 100,
        'High': np.random.random(200) * 100,
        'Low': np.random.random(200) * 100,
        'Adj Close': np.random.random(200) * 100,
        'Volume': np.random.randint(100, 1000, 200)
    }, index=dates)

    # 市場全体のインデックスデータの作成
    index_data = pd.DataFrame({
        'Adj Close': np.random.random(200) * 100
    }, index=dates)

    # VWAPアウトブレイク戦略の実行
    strategy = VWAPBreakoutStrategy(df, index_data)
    result = strategy.backtest()
    print(result)