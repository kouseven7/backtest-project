"""
Module: VWAP_Bounce
File: VWAP_Bounce.py
Description: 
  出来高加重平均価格(VWAP)からの反発を検出する戦略クラスを実装しています。
  レンジ相場で株価がVWAP付近から反発する場面を捉え、出来高増加と陽線形成を
  確認してエントリーし、VWAPからの乖離や利益確定・損切りポイントでイグジットします。

Author: kouseven7
Created: 2023-02-15
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.trend_analysis
  - indicators.basic_indicators
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from indicators.trend_analysis import detect_trend
from indicators.basic_indicators import calculate_vwap

class VWAPBounceStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        VWAP反発戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.volume_column = volume_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "vwap_lower_threshold": 0.99,  # VWAPから下方の許容値（-1%）
            "vwap_upper_threshold": 1.02,  # VWAPから上方の許容値（+2%）
            "stop_loss": 0.02,             # ストップロス（2%）
            "take_profit": 0.05            # 利益確定（5%）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # VWAPを計算してデータに追加
        self.data['VWAP'] = calculate_vwap(self.data, price_column=self.price_column, volume_column=self.volume_column)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - レンジ相場であること
        - 株価がVWAPから-1%以内で反発の兆候を示す（陽線形成、出来高増加）
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
        vwap_lower = vwap * self.params["vwap_lower_threshold"]
        price_near_vwap = (vwap_lower <= current_price <= vwap)
        bullish_candle = current_price > previous_close
        volume_increase = current_volume > previous_volume

        if price_near_vwap and bullish_candle and volume_increase:
            self.entry_prices[idx] = current_price
            self.log_trade(f"VWAP Bounce エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}, VWAP={vwap}")
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        条件:
        - VWAPから+2%で利確
        - VWAPから-1%で損切り

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 必要な履歴データがない場合
            return 0
            
        # エントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        if latest_entry_idx not in self.entry_prices:
            # 記録されていない場合は価格を取得して記録
            self.entry_prices[latest_entry_idx] = self.data[self.price_column].iloc[latest_entry_idx]
            
        entry_price = self.entry_prices[latest_entry_idx]
        current_price = self.data[self.price_column].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        
        # VWAPから上方の許容値で利確
        vwap_upper = vwap * self.params["vwap_upper_threshold"]
        if current_price >= vwap_upper:
            self.log_trade(f"VWAP Bounce イグジットシグナル: VWAP上方到達 日付={self.data.index[idx]}, 価格={current_price}, VWAP={vwap}")
            return -1
            
        # VWAPから下方の許容値で損切り
        vwap_lower = vwap * self.params["vwap_lower_threshold"]
        if current_price <= vwap_lower:
            self.log_trade(f"VWAP Bounce イグジットシグナル: VWAP下方損切り 日付={self.data.index[idx]}, 価格={current_price}, VWAP={vwap}")
            return -1
            
        # エントリー価格からのストップロス
        if current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"VWAP Bounce イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # エントリー価格からの利益確定
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"VWAP Bounce イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        return 0

    def backtest(self):
        """
        VWAP反発戦略のバックテストを実行する。
        
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
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=dates)

    # VWAP反発戦略の実行
    strategy = VWAPBounceStrategy(df)
    result = strategy.backtest()
    print(result)