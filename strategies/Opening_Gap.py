"""
Module: Opening_Gap
File: Opening_Gap.py
Description: 
  寄り付きのギャップ（前日終値と当日始値の差）を利用した戦略を実装します。
  特に海外市場の影響による上昇ギャップに着目し、高ボラティリティ環境での
  取引機会を検出します。適切なエントリー条件とリスク管理戦略で利益を最大化します。

Author: kouseven7
Created: 2023-03-05
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.trend_analysis
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from indicators.trend_analysis import detect_high_volatility

class OpeningGapStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, dow_data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        Opening Gap Strategy の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            dow_data (pd.DataFrame): ダウ平均データ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
        """
        # 戦略固有の属性を先に設定
        self.dow_data = dow_data
        self.price_column = price_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "atr_threshold": 2.0,  # 高ボラティリティ判定の閾値
            "stop_loss": 0.02,     # ストップロス（2%）
            "take_profit": 0.05    # 利益確定（5%）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 高ボラティリティであること
        - 前日ダウ平均が上昇していること
        - 前日終値より始値が高いこと

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0

        # 高ボラティリティの判定
        volatility = detect_high_volatility(self.data.iloc[:idx + 1], price_column=self.price_column, 
                                            atr_threshold=self.params["atr_threshold"])
        if volatility != "high volatility":
            return 0

        # 前日ダウ平均が上昇していること
        if idx >= len(self.dow_data):
            return 0  # ダウデータが十分にない場合
            
        dow_close_today = self.dow_data['Close'].iloc[min(idx, len(self.dow_data)-1)]
        dow_close_yesterday = self.dow_data['Close'].iloc[min(idx-1, len(self.dow_data)-1)]
        if dow_close_today <= dow_close_yesterday:
            return 0

        # 前日終値より始値が高いこと
        if 'Open' not in self.data.columns:
            return 0  # 始値データがない場合

        open_price = self.data['Open'].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]
        if open_price <= previous_close:
            return 0

        # エントリー価格を記録
        self.entry_prices[idx] = open_price  # 始値でエントリー
        self.log_trade(f"Opening Gap エントリーシグナル: 日付={self.data.index[idx]}, 始値={open_price}, 前日終値={previous_close}")
        return 1

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        条件:
        - 上昇が止まった場合に利益確定する
        - ストップロスやテイクプロフィットを設定

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0
            
        # エントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        if latest_entry_idx not in self.entry_prices:
            # 記録されていない場合は始値を使用
            if 'Open' in self.data.columns:
                self.entry_prices[latest_entry_idx] = self.data['Open'].iloc[latest_entry_idx]
            else:
                self.entry_prices[latest_entry_idx] = self.data[self.price_column].iloc[latest_entry_idx]
            
        entry_price = self.entry_prices[latest_entry_idx]
        current_price = self.data[self.price_column].iloc[idx]
        previous_price = self.data[self.price_column].iloc[idx - 1]

        # 上昇が止まった場合
        if current_price < previous_price:
            self.log_trade(f"Opening Gap イグジットシグナル: 上昇停止 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # エントリー価格からのストップロス
        if current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"Opening Gap イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # エントリー価格からの利益確定
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"Opening Gap イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        return 0

    def backtest(self):
        """
        Opening Gap Strategy のバックテストを実行する。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        # 各日にちについてシグナルを計算
        for idx in range(len(self.data)):
            # Entry_Signalがまだ立っていない場合のみエントリーシグナルをチェック
            if not self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1].any():
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
            
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

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