import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
from indicators.basic_indicators import calculate_rsi
from indicators.trend_analysis import detect_trend

class ContrarianStrategy:
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close"):
        """
        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): RSIや価格計算に使用する価格カラム（デフォルトは "Adj Close"）
        """
        self.data = data
        self.price_column = price_column

        # RSIを計算してデータに追加
        self.data['RSI'] = calculate_rsi(self.data[self.price_column])

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - RSIが30以下
        - 株価が前日終値より-5%以上のギャップダウン
        - またはピンバーが出現
        - レンジ相場であること

        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 5:  # 過去5日間のデータが必要
            return 0

        rsi = self.data['RSI'].iloc[idx]
        current_price = self.data[self.price_column].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]

        # ギャップダウンの判定
        gap_down = (current_price < previous_close * 0.95)

        # ピンバーの判定（簡易的に高値と安値の差で判定）
        high = self.data['High'].iloc[idx]
        low = self.data['Low'].iloc[idx]
        pin_bar = (high - current_price) > 2 * (current_price - low)

        # レンジ相場の判定
        trend = detect_trend(self.data.iloc[:idx + 1], price_column=self.price_column)
        range_market = (trend == "range-bound")

        # エントリー条件
        if rsi <= 30 and gap_down and range_market:
            return 1
        if pin_bar and range_market:
            return 1

        return 0

    def generate_exit_signal(self, entry_price: float, current_price: float, previous_close: float) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - 利益確定（前日終値に達したら利確）
        - 損切（-4%で損切）

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
        # 利益確定条件: 現在の価格が前日終値以上
        if current_price >= previous_close:
            return -1  # 利益確定

        # 損切条件: 現在の価格がエントリー価格の96%以下
        if current_price <= entry_price * 0.96:
            return -1  # 損切

        return 0

    def backtest(self):
        """
        逆張り戦略のバックテストを実行する。
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
                previous_close = self.data[self.price_column].iloc[idx - 1]
                exit_signal = self.generate_exit_signal(entry_price, current_price, previous_close)
                self.data.at[idx, 'Exit_Signal'] = exit_signal
                if exit_signal == -1:
                    entry_price = None

        return self.data

# テストコード
if __name__ == "__main__":
    import numpy as np
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100
    }, index=dates)

    strategy = ContrarianStrategy(df, price_column='Adj Close')
    result = strategy.backtest()
    print(result[['Adj Close', 'RSI', 'Entry_Signal', 'Exit_Signal']].tail())