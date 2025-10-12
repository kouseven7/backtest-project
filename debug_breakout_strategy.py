#!/usr/bin/env python3
"""
デバッグ版BreakoutStrategy
Exit_Signal列更新処理の詳細追跡版
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from src.strategies.Breakout import BreakoutStrategy

class DebugBreakoutStrategy(BreakoutStrategy):
    def __init__(self, data, params=None, price_column="Adj Close", volume_column="Volume"):
        super().__init__(data, params, price_column, volume_column)
        self.debug_log = []
        
    def debug_print(self, message):
        """デバッグメッセージ出力"""
        print(f"[DEBUG] {message}")
        self.debug_log.append(message)
    
    def generate_exit_signal(self, idx: int) -> int:
        """デバッグ版エグジットシグナル生成"""
        self.debug_print(f"generate_exit_signal called: idx={idx}")
        
        if idx < 1:
            self.debug_print(f"idx < 1, returning 0")
            return 0
            
        # エントリー価格と高値を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        self.debug_print(f"entry_indices: {entry_indices.tolist()}")
        
        if len(entry_indices) == 0:
            self.debug_print(f"No entry indices, returning 0")
            return 0
            
        if entry_indices[-1] >= self.data.index[idx]:
            self.debug_print(f"Latest entry >= current date, returning 0")
            return 0
            
        # 最新のエントリーインデックス（日付）を取得
        latest_entry_date = entry_indices[-1]
        latest_entry_pos = self.data.index.get_loc(latest_entry_date)
        
        self.debug_print(f"latest_entry_date: {latest_entry_date}, latest_entry_pos: {latest_entry_pos}")

        if latest_entry_date not in self.entry_prices:
            self.entry_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        if latest_entry_date not in self.high_prices and 'High' in self.data.columns:
            self.high_prices[latest_entry_date] = self.data['High'].iloc[latest_entry_pos]
        elif latest_entry_date not in self.high_prices:
            self.high_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        entry_price = self.entry_prices[latest_entry_date]
        high_price = self.high_prices[latest_entry_date]
        current_price = self.data[self.price_column].iloc[idx]
        
        self.debug_print(f"entry_price: {entry_price}, high_price: {high_price}, current_price: {current_price}")
        
        # 現在の高値を更新（トレーリングストップのために）
        if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
            high_price = self.data['High'].iloc[idx]
            self.high_prices[latest_entry_date] = high_price
            self.debug_print(f"High price updated: {high_price}")

        # 利確条件
        take_profit_level = entry_price * (1 + self.params["take_profit"])
        self.debug_print(f"take_profit_level: {take_profit_level}")
        
        if current_price >= take_profit_level:
            self.debug_print(f"利確条件成立: current_price({current_price}) >= take_profit_level({take_profit_level})")
            self.log_trade(f"Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            self.debug_print(f"Returning -1 for profit taking")
            return -1

        # 損切条件（高値からの反落）
        trailing_stop_level = 1 - self.params["trailing_stop"]
        trailing_stop_price = high_price * trailing_stop_level
        self.debug_print(f"trailing_stop_price: {trailing_stop_price}")
        
        if current_price < trailing_stop_price:
            self.debug_print(f"損切条件成立: current_price({current_price}) < trailing_stop_price({trailing_stop_price})")
            self.log_trade(f"Breakout イグジットシグナル: 高値から反落 日付={self.data.index[idx]}, 価格={current_price}, 高値={high_price}")
            self.debug_print(f"Returning -1 for trailing stop")
            return -1

        self.debug_print(f"No exit condition met, returning 0")
        return 0
    
    def backtest(self):
        """デバッグ版バックテスト"""
        self.debug_print("backtest() started")
        
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        self.debug_print("Signal columns initialized")

        # 各日にちについてシグナルを計算
        for idx in range(len(self.data)):
            current_date = self.data.index[idx]
            self.debug_print(f"Processing idx={idx}, date={current_date}")
            
            # Entry_Signalがまだ立っていない場合のみエントリーシグナルをチェック
            if not self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1].any():
                entry_signal = self.generate_entry_signal(idx)
                self.debug_print(f"entry_signal result: {entry_signal}")
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    self.debug_print(f"Set Entry_Signal = 1 at idx={idx}, date={current_date}")
            
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            self.debug_print(f"exit_signal result: {exit_signal}")
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                self.debug_print(f"Set Exit_Signal = -1 at idx={idx}, date={current_date}")
                
            # 現在の状態確認
            current_entry = self.data['Entry_Signal'].iloc[idx]
            current_exit = self.data['Exit_Signal'].iloc[idx]
            self.debug_print(f"Final state: Entry_Signal={current_entry}, Exit_Signal={current_exit}")

        self.debug_print("backtest() completed")
        return self.data
