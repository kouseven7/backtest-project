import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
from indicators.trend_analysis import detect_trend
from indicators.basic_indicators import calculate_vwap

class VWAPBounceStrategy:
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        VWAP反発戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        self.data = data
        self.price_column = price_column
        self.volume_column = volume_column

        # VWAPを計算してデータに追加
        self.data['VWAP'] = calculate_vwap(self.data, price_column=self.price_column, volume_column=self.volume_column)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - レンジ相場であること
        - 株価がVWAPから-1%以内で反発の兆候を示す（陽線形成、出来高増加）

        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0

        # レンジ相場の判定
        trend = detect_trend(self.data.iloc[:idx + 1], price_column=self.price_column)
        if trend != "range-bound":
            return 0  # レンジ相場でない場合はエントリーしない

        current_price = self.data[self.price_column].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # VWAPから-1%以内で反発の兆候（陽線形成、出来高増加）
        price_near_vwap = (vwap * 0.99 <= current_price <= vwap)
        bullish_candle = current_price > previous_close
        volume_increase = current_volume > previous_volume

        if price_near_vwap and bullish_candle and volume_increase:
            return 1

        return 0

    def generate_exit_signal(self, current_price: float, vwap: float) -> int:
        """
        エグジットシグナルを生成する。
        条件:
        - VWAPから+2%で利確
        - VWAPから-1%で損切

        Returns:
            int: エグジットシグナル（-1: エグジット, 0: なし）
        """
        if current_price >= vwap * 1.02:  # 利確条件
            return -1
        if current_price <= vwap * 0.99:  # 損切条件
            return -1

        return 0

    def backtest(self):
        """
        VWAP反発戦略のバックテストを実行する。
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
                vwap = self.data['VWAP'].iloc[idx]
                exit_signal = self.generate_exit_signal(current_price, vwap)
                self.data.at[idx, 'Exit_Signal'] = exit_signal
                if exit_signal == -1:
                    entry_price = None

        return self.data

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=dates)

    # VWAP反発戦略の実行
    strategy = VWAPBounceStrategy(df)
    result = strategy.backtest()
    print(result)