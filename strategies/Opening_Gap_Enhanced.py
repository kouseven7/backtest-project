"""
Module: opening_gap_enhanced
File: Opening_Gap_Enhanced.py
Description: 
  拡張基底クラスを使用したOpeningGapStrategyの強化版。
  同日のエントリー/エグジット問題を解決し、ポジション管理を改善した戦略。

Author: imega
Created: 2025-10-15
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from strategies.enhanced_base_strategy import EnhancedBaseStrategy

class OpeningGapEnhancedStrategy(EnhancedBaseStrategy):
    """
    OpeningGapEnhancedStrategyは、EnhancedBaseStrategyを継承し、
    ギャップを利用した取引戦略を実装します。
    強化されたポジション管理機能を持ち、同日のエントリー/エグジット問題を解決します。
    """
    
    def __init__(self, data: pd.DataFrame, dow_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None, price_column: str = "Adj Close"):
        """
        OpeningGapEnhancedStrategyの初期化
        
        Parameters:
            data (pd.DataFrame): 株価データ
            dow_data (pd.DataFrame): ダウ平均などの指数データ
            params (dict, optional): 戦略パラメータ
            price_column (str): 価格列の名前
        """
        # パラメータがNoneの場合は空のディクショナリを使用
        if params is None:
            params = {}
            
        # 親クラスに渡すパラメータを準備
        enhanced_params = params.copy()
        enhanced_params["price_column"] = price_column
        
        # 親クラスの初期化
        super().__init__(data, enhanced_params)
        
        # 追加データの保存
        self.dow_data = dow_data
        self.price_column = price_column
        
        # ポジション管理用のデータ構造
        self.entry_prices = {}  # エントリー価格を保存するディクショナリ {インデックス: 価格}
        self.high_prices = {}   # トレーリングストップ用の高値を保存するディクショナリ {インデックス: 高値}
        self.current_position = 0.0  # 現在のポジションサイズ
        
        # ギャップ戦略のデフォルトパラメータ
        self.gap_threshold = params.get("gap_threshold", 0.01)  # ギャップの閾値
        self.profit_target = params.get("profit_target", 0.03)  # 利益確定の閾値
        self.stop_loss = params.get("stop_loss", 0.02)          # 損切りの閾値
        
    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する
        
        Parameters:
            idx (int): バックテスト中の現在のインデックス
        
        Returns:
            int: エントリーシグナル (1: エントリー, 0: なし)
        """
        try:
            # データが不足している場合はシグナルなし
            if idx < 1:
                return 0
                
            # 前日と当日の価格を取得
            prev_close = float(self.data[self.price_column].iloc[idx-1])
            current_open = float(self.data["Open"].iloc[idx])
            
            # ギャップアップを検出
            gap_percent = (current_open / prev_close) - 1
            
            # 対応する日付のインデックス平均を取得
            current_date = self.data.index[idx].date()
            
            # ダウが上昇している場合のみエントリー
            dow_rising = False
            
            try:
                # インデックスデータから同じ日付のデータを探す（日付ベースの検索）
                matching_dates = []
                for i, date in enumerate(self.dow_data.index):
                    if hasattr(date, 'date') and date.date() == current_date:
                        matching_dates.append(i)
                
                if matching_dates:
                    dow_date_idx = matching_dates[0]
                    if dow_date_idx > 0:
                        # 前日と当日のダウの終値を取得
                        dow_prev_close = float(self.dow_data[self.price_column].iloc[dow_date_idx - 1])
                        dow_current = float(self.dow_data[self.price_column].iloc[dow_date_idx])
                        dow_rising = dow_current > dow_prev_close
                else:
                    # 日付が見つからない場合はデフォルトでTrueとする
                    dow_rising = True
            except Exception as e:
                self.logger.warning(f"ダウデータ取得エラー: {e}")
                dow_rising = True  # エラーが発生した場合、デフォルトでTrueとする
                
            # ギャップが閾値を超え、かつダウが上昇している場合はエントリーシグナルを発生
            if gap_percent > self.gap_threshold and dow_rising:
                # エントリー価格を記録
                self.record_entry_price(idx, current_open)
                return 1
                
            return 0
        except Exception as e:
            self.logger.warning(f"エントリーシグナル計算エラー: {e}")
            return 0  # エラーが発生した場合はシグナルなし
            
        return 0
        
    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する
        
        Parameters:
            idx (int): バックテスト中の現在のインデックス
        
        Returns:
            int: イグジットシグナル (-1: イグジット, 0: なし)
        """
        try:
            # 直接データからエントリー情報を取得
            if 'Entry_Signal' not in self.data.columns:
                return 0  # まだエントリーシグナル列が存在しない場合は早期リターン
            
            # エントリーのあるインデックスを検索
            entry_rows = []
            for i in range(idx):
                if i < len(self.data) and self.data['Entry_Signal'].iloc[i] == 1:
                    entry_rows.append(i)
            
            if not entry_rows:
                return 0  # エントリーがない場合はイグジットシグナルなし
                
            # 最新のエントリーインデックスと価格を取得
            latest_entry_idx = entry_rows[-1]
            entry_price = self.entry_prices.get(latest_entry_idx, None)
            
            if entry_price is None:
                # エントリー価格が不明な場合は、エントリー日のオープン価格を使用
                entry_price = float(self.data["Open"].iloc[latest_entry_idx])
            
            # 現在の価格を取得
            current_price = float(self.data[self.price_column].iloc[idx])
            current_high = float(self.data["High"].iloc[idx])
            current_low = float(self.data["Low"].iloc[idx])
            
            # トレーリングストップの更新
            self.update_trailing_high(idx, current_high)
            
            # 損益率を計算
            profit_pct = (current_price / entry_price) - 1
            
            # 利益確定条件: 利益率が利益目標を超えた場合
            if profit_pct >= self.profit_target:
                return -1
                
            # 損切り条件: 損失率が損切り閾値を超えた場合
            if profit_pct <= -self.stop_loss:
                return -1
                
            # トレーリングストップ条件: 高値からの下落が一定以上の場合
            trailing_threshold = self.params.get("trailing_threshold", 0.015)  # デフォルト1.5%
            trailing_high = self.high_prices.get(idx-1, entry_price)
            if trailing_high > entry_price and (current_low / trailing_high) < (1 - trailing_threshold):
                return -1
                
            # クローズタイミングの追加条件（例：指定日数経過後に自動決済）
            max_hold_days = self.params.get("max_hold_days", 5)
            days_held = idx - latest_entry_idx
            if days_held >= max_hold_days:
                return -1
                
            return 0
        except Exception as e:
            self.logger.warning(f"イグジットシグナル計算エラー: {e}")
            return 0  # エラーが発生した場合はシグナルなし
                
            return 0
        except Exception as e:
            self.logger.warning(f"イグジットシグナル計算エラー (2): {e}")
            return 0  # エラーが発生した場合はシグナルなし
        
        return 0
        
    def update_trailing_high(self, idx: int, current_high: float) -> None:
        """
        トレーリングストップのための最高値を更新する
        
        Parameters:
            idx (int): 現在のインデックス
            current_high (float): 現在の高値
        """
        try:
            # 前回の最高値を取得
            prev_high = self.high_prices.get(idx-1, 0)
            
            # 現在の高値が前回より高ければ更新
            if current_high > prev_high:
                self.high_prices[idx] = current_high
            else:
                # そうでなければ前回の高値を維持
                self.high_prices[idx] = prev_high
        except Exception as e:
            self.logger.warning(f"トレーリング更新エラー: {e}")
            # エラー時は現在の高値をそのまま設定
            self.high_prices[idx] = current_high
    
    def record_entry_price(self, idx: int, price: float) -> None:
        """
        エントリー価格を記録する
        
        Parameters:
            idx (int): エントリーのインデックス
            price (float): エントリー価格
        """
        try:
            self.entry_prices[idx] = price
            self.current_position = 1.0  # 1単位のポジションを取る
            self.log_trade(f"エントリー: インデックス={idx}, 価格={price:.2f}")
        except Exception as e:
            self.logger.warning(f"エントリー価格記録エラー: {e}")
    
    def log_trade(self, message: str) -> None:
        """
        トレード情報をログに記録する
        
        Parameters:
            message (str): ログメッセージ
        """
        self.logger.info(f"[{self.__class__.__name__}] {message}")