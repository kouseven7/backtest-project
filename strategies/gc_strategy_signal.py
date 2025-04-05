"""
Module: gc_strategy_signal
File: gc_strategy_signal.py
Description: 
  移動平均線のゴールデンクロス（短期線が長期線を上抜け）とデッドクロス（短期線が長期線を下抜け）を
  検出して取引シグナルを生成する戦略を実装しています。上昇トレンドの確認と合わせて使用することで
  精度を高め、適切な利確・損切り条件も設定しています。

Author: kouseven7
Created: 2023-02-25
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.trend_analysis
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import logging
from strategies.base_strategy import BaseStrategy
from indicators.trend_analysis import detect_trend

class GCStrategy(BaseStrategy):
    """
    GC戦略（ゴールデンクロス戦略）の実装クラス。
    短期移動平均と長期移動平均のゴールデンクロス／デッドクロスを基にエントリー／イグジットシグナルを生成し、
    Excelから取得した戦略パラメータ（例: 利益確定％、損切割合％、短期・長期移動平均期間）を反映させます。
    """
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（例: {"short_window": 5, "long_window": 25, ...}）
            price_column (str): インジケーター計算に使用する価格カラム（デフォルトは "Adj Close"）
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
        # 指定された価格カラムが存在するか確認、なければ 'Close' を代用
        if self.price_column not in data.columns:
            price_column = "Close"
            self.price_column = price_column
        
        # デフォルトパラメータの設定
        default_params = {
            "short_window": 5,       # 短期移動平均期間
            "long_window": 25,       # 長期移動平均期間
            "take_profit": 0.05,     # 利益確定（5%）
            "stop_loss": 0.03        # ストップロス（3%）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # 戦略パラメータの読み込み
        self.short_window = int(self.params.get("short_window", 5))
        self.long_window = int(self.params.get("long_window", 25))
        
        self.logger.info(
            f"GCStrategy initialized with short_window={self.short_window}, long_window={self.long_window}, "
            f"take_profit={self.params['take_profit']}, stop_loss={self.params['stop_loss']}"
        )
        
        # 移動平均の計算（指定した価格カラムを使用）
        self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean()
        self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean()

    def generate_entry_signal(self, idx: int) -> int:
        """
        指定されたインデックス位置でのエントリーシグナルを生成する。
        短期移動平均が長期移動平均を上回り、かつトレンドが上昇トレンドの場合、1を返す。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < self.long_window:  # 長期移動平均の計算に必要な日数分のデータがない場合
            return 0
        
        short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
        long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
        
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0

        # 前日のSMA値を取得してクロス判定
        prev_short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx-1]
        prev_long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx-1]
        
        # ゴールデンクロス（短期MAが長期MAを下から上に抜けた）
        golden_cross = short_sma > long_sma and prev_short_sma <= prev_long_sma

        # トレンド判定
        trend = detect_trend(self.data.iloc[:idx + 1], price_column=self.price_column)
        
        if golden_cross and trend == "uptrend":
            current_price = self.data[self.price_column].iloc[idx]
            self.entry_prices[idx] = current_price
            self.log_trade(f"GC Strategy エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}, 短期MA={short_sma}, 長期MA={long_sma}")
            return 1
            
        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        指定されたインデックス位置でのイグジットシグナルを生成する。
        短期移動平均が長期移動平均を下回った場合、または利益確定・損切条件に達した場合に-1を返す。
        
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
        
        # デッドクロス判定
        short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
        long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
        
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0
            
        prev_short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx-1]
        prev_long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx-1]
        
        # デッドクロス（短期MAが長期MAを上から下に抜けた）
        dead_cross = short_sma < long_sma and prev_short_sma >= prev_long_sma
        
        if dead_cross:
            self.log_trade(f"GC Strategy イグジットシグナル: デッドクロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # 利益確定条件
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"GC Strategy イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}, 利益率={round((current_price/entry_price-1)*100, 2)}%")
            return -1
            
        # 損切条件
        if current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"GC Strategy イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}, 損失率={round((current_price/entry_price-1)*100, 2)}%")
            return -1
            
        return 0

    def backtest(self):
        """
        GC戦略のバックテストを実行する。
        
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
        'Adj Close': np.random.random(100) * 100
    }, index=dates)

    # GC戦略の実行
    strategy = GCStrategy(df)