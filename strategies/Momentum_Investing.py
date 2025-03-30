import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from indicators.basic_indicators import calculate_sma, calculate_rsi
from indicators.momentum_indicators import calculate_macd
from indicators.volume_analysis import detect_volume_increase
from indicators.volatility_indicators import calculate_atr

class MomentumInvestingStrategy:
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        モメンタム戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        self.data = data
        self.price_column = price_column
        self.volume_column = volume_column

        # 必要なインジケーターを計算してデータに追加
        self.data['SMA_20'] = calculate_sma(self.data, price_column, 20)
        self.data['SMA_50'] = calculate_sma(self.data, price_column, 50)
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], 14)
        self.data['MACD'], self.data['Signal_Line'] = calculate_macd(self.data, price_column)
        self.data['ATR'] = calculate_atr(self.data, price_column)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 株価が20日MAおよび50日MAの上にある
        - 20日MAが50日MAを上抜けしており、両者が上昇傾向にある
        - RSIが50以上で過熱状態ではない
        - MACDラインがシグナルラインを上抜けしている
        - 出来高が増加している
        - プルバックからの反発、またはブレイクアウト

        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 50:  # 50日分のデータが必要
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        sma_20 = self.data['SMA_20'].iloc[idx]
        sma_50 = self.data['SMA_50'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # 株価が20日MAおよび50日MAの上にある
        if not (current_price > sma_20 > sma_50):
            return 0

        # 20日MAが50日MAを上抜けしており、両者が上昇傾向にある
        if not (sma_20 > sma_50 and self.data['SMA_20'].iloc[idx - 1] < self.data['SMA_50'].iloc[idx - 1]):
            return 0

        # RSIが50以上で過熱状態ではない
        if not (50 <= rsi < 70):
            return 0

        # MACDラインがシグナルラインを上抜けしている
        if not (macd > signal_line):
            return 0

        # 出来高が増加している
        if not detect_volume_increase(current_volume, previous_volume, threshold=1.2):
            return 0

        # プルバックからの反発
        if current_price < sma_20 and current_price > sma_20 * 0.98:
            return 1

        # ブレイクアウト（直近の抵抗線を上抜け）
        recent_high = self.data['High'].iloc[idx - 10:idx].max()
        if current_price > recent_high:
            return 1

        return 0

    def generate_exit_signal(self, entry_price: float, current_price: float, atr: float, idx: int) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - エントリー価格から3～5%の下落でストップロス
        - ATRに基づいたストップロス
        - 利益が伸びた場合、トレーリングストップを設定
        - 株価が短期MA（20日MA）を下回った場合
        - RSIが70以上から急落、またはMACDがシグナルラインを下抜けした場合
        - 高値更新が止まり、ダイバージェンスが発生した場合

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
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

        # 移動平均線のブレイク
        sma_20 = self.data['SMA_20'].iloc[idx]
        if current_price < sma_20:
            return -1

        # モメンタム指標の反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:  # RSIが70以上から急落
            return -1
        if macd < signal_line:  # MACDがシグナルラインを下抜け
            return -1

        # チャートパターンの崩壊（高値更新が止まる）
        recent_high = self.data['High'].iloc[idx - 10:idx].max()
        if current_price < recent_high * 0.97:  # 直近高値の3%下
            return -1

        return 0

    def backtest(self):
        """
        モメンタム戦略のバックテストを実行する。
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
                atr = self.data['ATR'].iloc[idx]
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

    # モメンタム戦略の実行
    strategy = MomentumInvestingStrategy(df)
    result = strategy.backtest()
    print(result)