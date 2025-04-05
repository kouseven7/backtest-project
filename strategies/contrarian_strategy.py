"""
Module: contrarian_strategy
File: contrarian_strategy.py
Description: 
  過度な売られ場面で反発を狙う逆張り戦略を実装しています。
  RSIの過売り状態やギャップダウン、ピンバー形成などの反転サインを検出し、
  レンジ相場でこれらの条件が揃った際にエントリーします。短期の利食いと
  適切な損切り設定で勝率とリスクリワード比の向上を図ります。

Author: kouseven7
Created: 2023-04-10
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.trend_analysis
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_rsi
from indicators.trend_analysis import detect_trend

class ContrarianStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        逆張り戦略の初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): RSIや価格計算に使用する価格カラム（デフォルトは "Adj Close"）
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "rsi_period": 14,        # RSI計算期間
            "rsi_oversold": 30,      # RSI過売り閾値
            "gap_threshold": 0.05,   # ギャップダウン閾値（5%）
            "stop_loss": 0.04,       # ストップロス（4%）
            "pin_bar_ratio": 2.0     # ピンバー判定比率
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # RSIを計算してデータに追加
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], period=self.params["rsi_period"])

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - RSIが30以下
        - 株価が前日終値より-5%以上のギャップダウン
        - またはピンバーが出現
        - レンジ相場であること

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 5:  # 過去5日間のデータが必要
            return 0

        rsi = self.data['RSI'].iloc[idx]
        current_price = self.data[self.price_column].iloc[idx]
        
        if idx <= 0 or idx >= len(self.data):
            return 0  # インデックスが範囲外
            
        previous_close = self.data[self.price_column].iloc[idx - 1]

        # ギャップダウンの判定
        gap_threshold = 1.0 - self.params["gap_threshold"]
        gap_down = (current_price < previous_close * gap_threshold)

        # ピンバーの判定（簡易的に高値と安値の差で判定）
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            high = self.data['High'].iloc[idx]
            low = self.data['Low'].iloc[idx]
            pin_bar = (high - current_price) > self.params["pin_bar_ratio"] * (current_price - low)
        else:
            pin_bar = False  # 高値・安値データがない場合

        # レンジ相場の判定
        trend = detect_trend(self.data.iloc[:idx + 1], price_column=self.price_column)
        range_market = (trend == "range-bound")

        # エントリー条件
        if rsi <= self.params["rsi_oversold"] and gap_down and range_market:
            self.entry_prices[idx] = current_price
            self.log_trade(f"Contrarian エントリーシグナル (過売り+ギャップダウン): 日付={self.data.index[idx]}, 価格={current_price}, RSI={rsi}")
            return 1
            
        if pin_bar and range_market:
            self.entry_prices[idx] = current_price
            self.log_trade(f"Contrarian エントリーシグナル (ピンバー): 日付={self.data.index[idx]}, 価格={current_price}")
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        条件:
        - 利益確定（前日終値に達したら利確）
        - 損切（-4%で損切）

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 過去データが必要
            return 0
            
        # エントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        if latest_entry_idx not in self.entry_prices:
            self.entry_prices[latest_entry_idx] = self.data[self.price_column].iloc[latest_entry_idx]
            
        entry_price = self.entry_prices[latest_entry_idx]
        current_price = self.data[self.price_column].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]
        
        # 利益確定条件: 現在の価格が前日終値以上
        if current_price >= previous_close:
            self.log_trade(f"Contrarian イグジットシグナル: 前日終値到達 日付={self.data.index[idx]}, 価格={current_price}")
            return -1  # 利益確定

        # 損切条件: 現在の価格がエントリー価格の(1-stop_loss)%以下
        stop_loss_level = entry_price * (1.0 - self.params["stop_loss"])
        if current_price <= stop_loss_level:
            self.log_trade(f"Contrarian イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}, 損失率={round((current_price/entry_price-1)*100, 2)}%")
            return -1  # 損切

        return 0

    def backtest(self):
        """
        逆張り戦略のバックテストを実行する。
        
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