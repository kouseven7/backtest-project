import pandas as pd
import numpy as np

class BreakoutStrategy:
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        ブレイクアウト戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        self.data = data
        self.price_column = price_column
        self.volume_column = volume_column

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 前日高値を上抜けた場合
        - 出来高が前日比で20%増加している

        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        previous_high = self.data['High'].iloc[idx - 1]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # 前日高値を上抜けた場合
        price_breakout = current_price > previous_high

        # 出来高が前日比で20%増加している
        volume_increase = current_volume > previous_volume * 1.2

        if price_breakout and volume_increase:
            return 1

        return 0

    def generate_exit_signal(self, entry_price: float, current_price: float, high_price: float) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - エントリー株価を3%超えた場合に利確
        - 高値を下回った場合に損切

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
        # 利確条件
        if current_price >= entry_price * 1.03:
            return -1

        # 損切条件
        if current_price < high_price:
            return -1

        return 0

    def backtest(self):
        """
        ブレイクアウト戦略のバックテストを実行する。
        """
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        entry_price = None
        high_price = None

        for idx in range(len(self.data)):
            if entry_price is None:
                # エントリーシグナルを確認
                entry_signal = self.generate_entry_signal(idx)
                self.data.at[idx, 'Entry_Signal'] = entry_signal
                if entry_signal == 1:
                    entry_price = self.data[self.price_column].iloc[idx]
                    high_price = self.data['High'].iloc[idx]  # エントリー時の高値を記録
            else:
                # エグジットシグナルを確認
                current_price = self.data[self.price_column].iloc[idx]
                exit_signal = self.generate_exit_signal(entry_price, current_price, high_price)
                self.data.at[idx, 'Exit_Signal'] = exit_signal
                if exit_signal == -1:
                    entry_price = None
                    high_price = None

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

    # ブレイクアウト戦略の実行
    strategy = BreakoutStrategy(df)
    result = strategy.backtest()
    print(result)