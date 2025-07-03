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
from typing import Dict, Any, Optional
from strategies.base_strategy import BaseStrategy
from config.optimized_parameters import OptimizedParameterManager
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend, detect_unified_trend_with_confidence

class OpeningGapStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, dow_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None, price_column: str = "Adj Close"):
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
        self.high_prices = {}  # トレーリングストップ用の最高値を記録する辞書
        
        # デフォルトパラメータの設定
        default_params: Dict[str, Any] = {
            # 既存パラメータ
            "atr_threshold": 2.0,  # 高ボラティリティ判定の閾値
            "stop_loss": 0.02,     # ストップロス（2%）
            "take_profit": 0.05,   # 利益確定（5%）
            "gap_threshold": 0.01, # ギャップアップ判定の閾値（1%）
            
            # 新規パラメータ
            "entry_delay": 0,                 # エントリー遅延（0=当日エントリー）
            "gap_direction": "both",          # ギャップ方向（"up", "down", "both"）
            "dow_filter_enabled": False,      # ダウトレンドフィルター有効/無効
            "dow_trend_days": 5,              # ダウトレンド判定期間
            "min_vol_ratio": 1.0,             # 最小出来高倍率（前日比）
            "volatility_filter": False,       # 高ボラ環境でのみ取引
            
            # トレンドフィルター設定
            "trend_filter_enabled": True,     # 統一トレンド判定の有効化
            "allowed_trends": ["uptrend"],    # 許可するトレンド（上昇トレンドのみ）

            # イグジット関連の新規パラメータ
            "max_hold_days": 5,              # 最大保有期間
            "consecutive_down_days": 1,      # 連続下落日数でイグジット
            "trailing_stop_pct": 0.02,       # トレーリングストップ割合
            "atr_stop_multiple": 1.5,        # ATRベースストップロス乗数
            "partial_exit_enabled": False,   # 一部利確機能
            "partial_exit_threshold": 0.03,  # 一部利確の閾値
            "partial_exit_portion": 0.5      # 一部利確の割合
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params: Dict[str, Any] = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # 統一トレンド検出器の初期化
        # 最新時点でのトレンド判定をコンソールに出力
        if len(self.data) > 0:
            try:
                trend, confidence = detect_unified_trend_with_confidence(
                    self.data, self.price_column, strategy="Opening_Gap"
                )
                self.logger.info(f"現在のトレンド: {trend}, 信頼度: {confidence:.1%}")
            except Exception as e:
                self.logger.warning(f"トレンド判定エラー: {e}")

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 前日終値から当日始値にかけて1%以上のギャップアップ
        - トレンド判定が上昇トレンド（オプション）
        - ダウ平均も上昇トレンド（オプション）

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < 1:  # 前日データが必要
            return 0
            
        # トレンド確認（統一トレンド判定を使用）
        use_trend_filter = self.params.get("trend_filter_enabled", False)
        if use_trend_filter:
            trend = detect_unified_trend(self.data.iloc[:idx + 1], self.price_column, strategy="Opening_Gap")
            allowed_trends = self.params.get("allowed_trends", ["uptrend"])
            # 許可されたトレンドでのみエントリー
            if trend not in allowed_trends:
                return 0  # トレンド不適合
                
        # ギャップアップ/ダウン判定
        open_price = self.data['Open'].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]

        # データ型をログに記録
        self.logger.info(f"データ型チェック: Open={type(open_price)}, Close={type(previous_close)}")

        # データ型を数値型に変換
        open_price = pd.to_numeric(open_price, errors='coerce')
        previous_close = pd.to_numeric(previous_close, errors='coerce')

        # ギャップアップをチェック
        gap_up = open_price > previous_close * (1 + self.params["gap_threshold"])
        gap_down = open_price < previous_close * (1 - self.params["gap_threshold"])

        # ダウトレンドフィルター
        if self.params.get("dow_filter_enabled", False) and self.dow_data is not None:
            # ダウのトレンド判定
            trend_days = self.params.get("dow_trend_days", 5)
            
            # ダウデータの確認
            current_date = self.data.index[idx]
            dow_before_date = self.dow_data.index[self.dow_data.index < current_date]
            
            if len(dow_before_date) >= trend_days:
                # 直近のダウデータを取得
                recent_dow = self.dow_data.loc[dow_before_date[-trend_days:]].copy()
                
                # トレンド判定（単純な場合：終値の方向性）
                dow_trend = "neutral"
                first_close = recent_dow['Close'].iloc[0]
                last_close = recent_dow['Close'].iloc[-1]
                
                if last_close > first_close * 1.01:  # 1%以上上昇
                    dow_trend = "up"
                elif last_close < first_close * 0.99:  # 1%以上下落
                    dow_trend = "down"
                
                self.logger.info(f"ダウトレンド: {dow_trend}, 日付={current_date}")
                
                # ギャップアップの場合、ダウトレンドがアップの時のみエントリー
                if gap_up and self.params.get("gap_direction") != "down":
                    if dow_trend != "up" and self.params.get("gap_direction") == "up":
                        self.logger.info(f"ダウトレンドフィルターでスキップ: ギャップアップだがダウトレンドが{dow_trend}")
                        return 0
                        
                # ギャップダウンの場合、ダウトレンドがダウンの時のみエントリー
                if gap_down and self.params.get("gap_direction") != "up":
                    if dow_trend != "down" and self.params.get("gap_direction") == "down":
                        self.logger.info(f"ダウトレンドフィルターでスキップ: ギャップダウンだがダウトレンドが{dow_trend}")
                        return 0

        if gap_up:
            self.entry_prices[idx] = open_price
            self.logger.info(f"Opening Gap エントリーシグナル: ギャップアップ 日付={self.data.index[idx]}, 始値={open_price}, 前日終値={previous_close}")
            return 1
        elif gap_down:
            self.entry_prices[idx] = open_price
            self.logger.info(f"Opening Gap エントリーシグナル: ギャップダウン 日付={self.data.index[idx]}, 始値={open_price}, 前日終値={previous_close}")
            return -1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        拡張されたイグジットシグナル生成。
        条件:
        - 上昇が止まった場合に利益確定する
        - ストップロスやテイクプロフィットを設定
        - 最大保有期間、連続下落日数、ATRベースストップロス、トレーリングストップを追加
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

        # 最大保有期間チェック
        if "max_hold_days" in self.params:
            entry_idx_loc = self.data.index.get_loc(entry_indices[-1])
            days_held = idx - entry_idx_loc
            if days_held >= self.params["max_hold_days"]:
                self.log_trade(f"Opening Gap イグジットシグナル: 最大保有期間超過 日付={self.data.index[idx]}, 価格={current_price}")
                return -1

        # 連続下落日数チェック
        if "consecutive_down_days" in self.params and idx >= self.params["consecutive_down_days"]:
            consecutive_days = self.params["consecutive_down_days"]
            is_consecutive_down = True
            for i in range(consecutive_days):
                if self.data[self.price_column].iloc[idx-i] >= self.data[self.price_column].iloc[idx-i-1]:
                    is_consecutive_down = False
                    break
            if is_consecutive_down:
                self.log_trade(f"Opening Gap イグジットシグナル: 連続下落日数超過 日付={self.data.index[idx]}, 価格={current_price}")
                return -1

        # ATRベースのストップロス
        if "atr_stop_multiple" in self.params:
            atr_value = self.data['ATR'].iloc[latest_entry_idx]
            atr_stop = entry_price - (atr_value * self.params["atr_stop_multiple"])
            if current_price <= atr_stop:
                self.log_trade(f"Opening Gap イグジットシグナル: ATRベースストップロス 日付={self.data.index[idx]}, 価格={current_price}")
                return -1

        # トレーリングストップ
        if "trailing_stop_pct" in self.params:
            # エントリー後の最高値を更新
            if idx not in self.high_prices:
                self.high_prices[idx] = current_price
            else:
                self.high_prices[idx] = max(self.high_prices[idx], current_price)
            
            # 最高値からの下落率でイグジット
            trailing_stop = self.high_prices[idx] * (1 - self.params["trailing_stop_pct"])
            if current_price <= trailing_stop:
                self.log_trade(f"Opening Gap イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
                return -1

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
        """バックテストに一部利確機能を追加"""
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        self.data['Position_Size'] = 0.0  # ポジションサイズ追跡用
        self.data['Partial_Exit'] = 0     # 一部利確用
        
        # バックテストループ
        for idx in range(len(self.data)):
            # Entry_Signalがまだ立っていない場合のみエントリーシグナルをチェック
            entry_signal_window = self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1]
            if not bool(entry_signal_window.values.any()):
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    self.data.at[self.data.index[idx], 'Position_Size'] = 1.0  # エントリー時にポジションサイズを1に設定
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                self.data.at[self.data.index[idx], 'Position_Size'] = 0.0  # イグジット時にポジションサイズを0に設定

            # 一部利確処理
            if self.params.get("partial_exit_enabled", False) and idx > 0:
                if self.data['Position_Size'].iloc[idx-1] > 0 and self.data['Partial_Exit'].iloc[idx-1] == 0:
                    entry_indices = self.data[self.data['Entry_Signal'] == 1].index
                    if len(entry_indices) > 0:
                        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
                        entry_price = self.entry_prices.get(latest_entry_idx)
                        current_price = self.data[self.price_column].iloc[idx]
                        
                        # 利益率が閾値を超えたら一部利確
                        if entry_price and (current_price / entry_price - 1) >= self.params["partial_exit_threshold"]:
                            portion = self.params["partial_exit_portion"]
                            self.data.at[self.data.index[idx], 'Partial_Exit'] = portion
                            self.data.at[self.data.index[idx], 'Position_Size'] -= portion
                            self.log_trade(f"一部利確 {portion*100}%: 日付={self.data.index[idx]}")

        return self.data

    def load_optimized_parameters(self, ticker: Optional[str] = None, strategy_name: str = "OpeningGapStrategy") -> bool:
        manager = OptimizedParameterManager()
        params = manager.load_approved_params(strategy_name, ticker)
        if params:
            self.params.update(params)
            return True
        return False

    def run_optimized_strategy(self, ticker: Optional[str] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.load_optimized_parameters(ticker):
            print("最適化パラメータが見つからないためデフォルトで実行します")
        return self.backtest()

    def get_optimization_info(self) -> Dict[str, Any]:
        return {
            "current_params": self.params,
            "optimization_mode": getattr(self, "optimization_mode", False)
        }

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