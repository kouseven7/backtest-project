import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
from indicators.trend_analysis import detect_high_volatility

class OpeningGapStrategy:
    def __init__(self, data: pd.DataFrame, dow_data: pd.DataFrame, price_column: str = "Adj Close"):
        """
        Opening Gap Strategy の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            dow_data (pd.DataFrame): ダウ平均データ
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
        """
        self.data = data
        self.dow_data = dow_data
        self.price_column = price_column

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 高ボラティリティであること
        - 前日ダウ平均が上昇していること
        - 前日終値より始値が高いこと

        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0

        # 高ボラティリティの判定
        volatility = detect_high_volatility(self.data.iloc[:idx + 1], price_column=self.price_column, atr_threshold=2.0)
        if volatility != "high volatility":
            return 0

        # 前日ダウ平均が上昇していること
        dow_close_today = self.dow_data['Close'].iloc[idx]
        dow_close_yesterday = self.dow_data['Close'].iloc[idx - 1]
        if dow_close_today <= dow_close_yesterday:
            return 0

        # 前日終値より始値が高いこと
        open_price = self.data['Open'].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]
        if open_price <= previous_close:
            return 0

        return 1

    def generate_exit_signal(self, idx: int) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - エントリーと同時に始値に逆指値を設定する
        - 5分足で上昇が止まった場合に利益確定する

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
        # 5分足で上昇が止まった場合に利益確定
        if idx < 1:  # 前のデータが必要
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        previous_price = self.data[self.price_column].iloc[idx - 1]

        if current_price <= previous_price:  # 上昇が止まった場合
            return -1

        return 0

    def backtest(self):
        """
        Opening Gap Strategy のバックテストを実行する。
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
                    entry_price = self.data['Open'].iloc[idx]  # 始値でエントリー
            else:
                # エグジットシグナルを確認
                exit_signal = self.generate_exit_signal(idx)
                self.data.at[idx, 'Exit_Signal'] = exit_signal
                if exit_signal == -1:
                    entry_price = None

        return self.data

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(100) * 100,
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=dates)

    # ダウ平均のダミーデータ
    dow_dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    dow_data = pd.DataFrame({
        'Close': np.random.random(100) * 100
    }, index=dow_dates)

    # Opening Gap Strategy の実行
    strategy = OpeningGapStrategy(df, dow_data)
    result = strategy.backtest()
    print(result)