"""
Module: Momentum_Investing
File: Momentum_Investing.py
Description: 
  モメンタム（勢い）に基づいた投資戦略を実装します。トレンドが強く出ている銘柄に投資し、
  移動平均線やRSI、MACD、出来高などの複合的な指標を用いて、上昇トレンドの継続性を
  判断します。適切なエントリー・イグジットポイントを算出し、リスク管理も考慮しています。

Author: kouseven7
Created: 2023-03-10
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.momentum_indicators
  - indicators.volume_analysis
  - indicators.volatility_indicators
"""

from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_sma, calculate_rsi
from indicators.momentum_indicators import calculate_macd
from indicators.volume_analysis import detect_volume_increase
from indicators.volatility_indicators import calculate_atr
from indicators.unified_trend_detector import detect_unified_trend, detect_unified_trend_with_confidence

class MomentumInvestingStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None, 
                 price_column: str = "Adj Close", 
                 volume_column: str = "Volume", 
                 optimization_mode: Optional[str] = None):
        """モメンタム戦略の初期化（拡張版）"""        # 戦略固有の属性
        self.price_column = price_column
        self.volume_column = volume_column
        self.entry_prices: Dict[int, float] = {}  # エントリー価格を記録する辞書
        self.optimization_mode = optimization_mode  # 最適化モード
        
        # 最適化モード用の属性
        self._parameter_manager = None
        self._approved_params = None
        
        # デフォルトパラメータの拡張
        default_params = {
            # 既存パラメータ
            "sma_short": 20,
            "sma_long": 50,
            "rsi_period": 14,
            "rsi_lower": 50,
            "rsi_upper": 68,
            "volume_threshold": 1.18,
            "take_profit": 0.12,
            "stop_loss": 0.06,
            "trailing_stop": 0.04,
            
            # 新規パラメータ
            "ma_type": "SMA",               # 移動平均タイプ (SMA/EMA)
            "max_hold_days": 15,            # 最大保有期間
            "atr_multiple": 2.0,            # ATRストップロス倍率
            "partial_exit_pct": 0.5,        # 一部利確率 (0〜1)
            "partial_exit_threshold": 0.08, # 一部利確の発動閾値
            "momentum_exit_threshold": -0.03, # モメンタム失速閾値
            "volume_exit_threshold": 0.7,   # 出来高減少イグジット閾値
            "trend_filter": True,           # トレンドフィルターの使用
            
            # Phase 2: スリッページ・取引コスト（2025-12-23追加）
            "slippage": 0.001,              # スリッページ0.1%
            "transaction_cost": 0.0,        # 取引コスト0.1%（デフォルト0、オプション）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
        # データは必ずコピーして保持
        self.data = data.copy()

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        ma_type = self.params.get("ma_type", "SMA")
        sma_short = self.params["sma_short"]
        sma_long = self.params["sma_long"]

        # ルックアヘッドバイアス修正: 既に指標列がある場合は再計算しない
        if f'MA_{sma_short}' not in self.data.columns:
            if ma_type == "SMA":
                self.data[f'MA_{sma_short}'] = calculate_sma(self.data, self.price_column, sma_short).shift(1)
            elif ma_type == "EMA":
                self.data[f'MA_{sma_short}'] = self.data[self.price_column].ewm(span=sma_short, adjust=False).mean().shift(1)
        if f'MA_{sma_long}' not in self.data.columns:
            if ma_type == "SMA":
                self.data[f'MA_{sma_long}'] = calculate_sma(self.data, self.price_column, sma_long).shift(1)
            elif ma_type == "EMA":
                self.data[f'MA_{sma_long}'] = self.data[self.price_column].ewm(span=sma_long, adjust=False).mean().shift(1)
        if 'RSI' not in self.data.columns:
            self.data['RSI'] = calculate_rsi(self.data[self.price_column], self.params["rsi_period"]).shift(1)
        if 'MACD' not in self.data.columns or 'Signal_Line' not in self.data.columns:
            macd_raw, signal_raw = calculate_macd(self.data, self.price_column)
            self.data['MACD'] = macd_raw.shift(1)
            self.data['Signal_Line'] = signal_raw.shift(1)
        if 'ATR' not in self.data.columns:
            self.data['ATR'] = calculate_atr(self.data, self.price_column).shift(1)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。さらに厳しいエントリー条件。
        条件:
        - 株価が20日MAおよび50日MAの上にある
        - RSIが50以上68未満の範囲内
        - MACDラインがシグナルラインを上抜け
        - 出来高増加または価格の明確なブレイクアウト

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        sma_short_key = 'MA_' + str(self.params["sma_short"])
        sma_long_key = 'MA_' + str(self.params["sma_long"])
        rsi_lower = self.params["rsi_lower"]
        rsi_upper = self.params["rsi_upper"]

        if idx < self.params["sma_long"]:
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        sma_short = self.data[sma_short_key].iloc[idx]
        sma_long = self.data[sma_long_key].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]

        # 条件カウント方式（やや厳しめ）
        condition_count = 0

        # 必須条件：株価が短期MAの上
        if current_price > sma_short:
            condition_count += 1
        else:
            return 0  # 必須条件

        # 株価が長期MAの上
        if current_price > sma_long:
            condition_count += 1

        # 20日MAが50日MAの上
        if sma_short > sma_long:
            condition_count += 1

        # RSIが条件範囲内（範囲を広げる）
        if rsi_lower <= rsi <= rsi_upper:
            condition_count += 1

        # MACDがシグナルラインを上抜け or MACD > 0
        if (macd > signal_line and self.data['MACD'].iloc[idx - 1] <= self.data['Signal_Line'].iloc[idx - 1]) or (macd > 0):
            condition_count += 1

        # 出来高条件（緩和：前日比または平均の1.05倍）
        if self.volume_column in self.data.columns:
            current_volume = self.data[self.volume_column].iloc[idx]
            previous_volume = self.data[self.volume_column].iloc[idx - 1]
            avg_volume = self.data[self.volume_column].iloc[max(0, idx-10):idx].mean()
            if detect_volume_increase(current_volume, previous_volume, threshold=self.params["volume_threshold"]) or current_volume > avg_volume * 1.05:
                condition_count += 1

        # プルバックまたはブレイクアウト（緩和）
        pullback_or_breakout = False
        if current_price < sma_short * 0.99 and current_price > sma_short * 0.97:
            pullback_or_breakout = True
        recent_high = self.data['High'].iloc[max(0, idx - 15):idx].max()
        if current_price > recent_high * 1.01:  # 1%以上のブレイクアウト
            pullback_or_breakout = True
        if pullback_or_breakout:
            condition_count += 1

        # エントリー条件 - 必須条件＋2つ以上（合計3つ以上）でエントリー
        if condition_count >= 3:
            # Phase 1修正: エントリー価格記録を削除（backtest()で翌日始値を記録するため）
            # self.entry_prices[idx] = current_price  # ← 削除
            self.log_trade(f"Momentum Investing エントリーシグナル: 日付={self.data.index[idx]}, 判断価格={current_price:.2f}, 条件数={condition_count}/7")
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する拡張版。
        条件:
        - 最大保有期間によるイグジット
        - ATRベースのストップロス（ATR倍率を導入）
        - モメンタム失速によるイグジット
        - 出来高減少によるイグジット
        - 既存の条件（ストップロス、利益確定、トレーリングストップ、移動平均線ブレイク、モメンタム指標の反転、チャートパターン崩壊）
        """
        if idx < 1:  # 必要な履歴データがない場合
            return 0
            
        # エントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx_raw = self.data.index.get_loc(entry_indices[-1])
        # 型エラー対策: latest_entry_idxをintに変換（sliceやndarrayの場合は例外）
        if not isinstance(latest_entry_idx_raw, int):
            raise TypeError(f"latest_entry_idx is not int: {type(latest_entry_idx_raw)}")
        latest_entry_idx_int = latest_entry_idx_raw
        
        if latest_entry_idx_int not in self.entry_prices:
            # Phase 1修正: フォールバック処理も翌日始値を使用（ルックアヘッドバイアス修正）
            next_day_pos = latest_entry_idx_int + 1
            if next_day_pos < len(self.data):
                next_day_open = self.data['Open'].iloc[next_day_pos]
            else:
                # 最終日の場合は当日始値を使用（境界条件の妥協案）
                next_day_open = self.data['Open'].iloc[latest_entry_idx_int]
            
            # Phase 2修正: スリッページ・取引コスト考慮（2025-12-23追加）
            slippage = self.params.get("slippage", 0.001)
            transaction_cost = self.params.get("transaction_cost", 0.0)
            self.entry_prices[latest_entry_idx_int] = next_day_open * (1 + slippage + transaction_cost)
            
        entry_price = self.entry_prices[latest_entry_idx_int]
        
        # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でイグジット
        current_price_val = self.data['Open'].iloc[idx + 1]
        if isinstance(current_price_val, pd.Series):
            current_price_val = current_price_val.values[0]
        current_price = current_price_val
        
        atr = self.data['ATR'].iloc[latest_entry_idx_int]  # エントリー時点のATR
        sma_short_key = 'MA_' + str(self.params["sma_short"])
            
        # 最大保有期間によるイグジット
        max_hold_days = self.params.get("max_hold_days")
        if max_hold_days is not None:
            days_held = idx - latest_entry_idx_int
            if days_held >= max_hold_days:
                self.log_trade(f"保有期間超過イグジット: {days_held}日/{max_hold_days}日 日付={self.data.index[idx]}")
                return -1

        # ATRベースのストップロス（ATR倍率を導入）
        atr_multiple = self.params.get("atr_multiple", 2.0)
        atr_stop_loss = entry_price - (atr * atr_multiple)
        if current_price <= atr_stop_loss or current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"ストップロスイグジット: 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # モメンタム失速によるイグジット
        momentum_exit_threshold = self.params.get("momentum_exit_threshold", -0.03)
        if idx > 1:
            rsi = self.data['RSI'].iloc[idx]
            rsi_prev = self.data['RSI'].iloc[idx-1]
            momentum_change = rsi - rsi_prev
            if momentum_change <= momentum_exit_threshold and rsi < 60:
                self.log_trade(f"モメンタム失速イグジット: 変化量={momentum_change} 日付={self.data.index[idx]}")
                return -1

        # 出来高減少によるイグジット
        if self.volume_column in self.data.columns:
            volume_exit_threshold = self.params.get("volume_exit_threshold", 0.7)
            current_volume = self.data[self.volume_column].iloc[idx]
            avg_volume = self.data[self.volume_column].iloc[max(0, idx-5):idx].mean()
            if current_volume < avg_volume * volume_exit_threshold:
                self.log_trade(f"出来高減少イグジット: 日付={self.data.index[idx]}, 比率={current_volume/avg_volume:.2f}")
                return -1

        # 利益確定条件（目標利益）
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"Momentum Investing イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # トレーリングストップ条件
        high_since_entry = self.data['High'].iloc[latest_entry_idx_int:idx+1].max()
        trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
        if current_price <= trailing_stop:
            self.log_trade(f"Momentum Investing イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # 移動平均線のブレイク
        sma_short = self.data[sma_short_key].iloc[idx]
        if current_price < sma_short:
            self.log_trade(f"Momentum Investing イグジットシグナル: 移動平均線ブレイク 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # モメンタム指標の反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > self.params["rsi_upper"] and rsi < self.data['RSI'].iloc[idx - 1]:  # RSIが70以上から急落
            self.log_trade(f"Momentum Investing イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        if macd < signal_line and self.data['MACD'].iloc[idx-1] >= self.data['Signal_Line'].iloc[idx-1]:  # MACDがシグナルラインを下抜け
            self.log_trade(f"Momentum Investing イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # チャートパターンの崩壊（高値更新が止まる）
        recent_high = self.data['High'].iloc[max(0, idx - 10):idx].max()
        if current_price < recent_high * (1 - self.params["trailing_stop"]):  # 直近高値の3%下
            self.log_trade(f"Momentum Investing イグジットシグナル: チャートパターン崩壊 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        return 0

    def backtest(self, trading_start_date=None, trading_end_date=None):
        """モメンタム戦略のバックテストを実行（部分利確機能付き + ウォームアップ期間対応）
        
        Parameters:
            trading_start_date (datetime, optional): 取引開始日（この日以降にシグナル生成開始）
            trading_end_date (datetime, optional): 取引終了日（この日以前までシグナル生成）
        """
        # シグナル列の初期化
        self.data.loc[:, 'Entry_Signal'] = 0
        self.data.loc[:, 'Exit_Signal'] = 0
        self.data.loc[:, 'Position'] = 0
        self.data.loc[:, 'Partial_Exit'] = 0
        self.data.loc[:, 'Profit_Pct'] = 0
        self.data.loc[:, 'Strategy'] = 'MomentumInvestingStrategy'  # 戦略名を明示的に追加
        
        # ポジション状態の追跡
        in_position = False
        entry_idx = -1

        # Phase 1修正: 最終日を除外してidx+1アクセスを安全に（ルックアヘッドバイアス修正）
        # 理由: エントリー価格を翌日始値（idx+1）に変更するため、最終日でのIndexError回避
        for idx in range(len(self.data) - 1):
            # 取引期間フィルタリング（BaseStrategy.backtest()と同じロジック）
            if trading_start_date is not None or trading_end_date is not None:
                current_date = self.data.index[idx]
                in_trading_period = True
                
                if trading_start_date is not None and current_date < trading_start_date:
                    in_trading_period = False
                if trading_end_date is not None and current_date > trading_end_date:
                    in_trading_period = False
                
                if not in_trading_period:
                    # 取引期間外はシグナル生成をスキップ
                    continue
            # ポジションを持っていない場合のみエントリーシグナルを検討
            if not in_position:
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    self.data.at[self.data.index[idx], 'Position'] = 1
                    in_position = True
                    entry_idx = idx
                    # Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
                    # 理由: idx日の終値を見てからidx日の終値で買うことは不可能
                    # リアルトレードでは翌日（idx+1日目）の始値でエントリー
                    next_day_open = self.data['Open'].iloc[idx + 1]
                    
                    # Phase 2修正: スリッページ・取引コスト考慮（2025-12-23追加）
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = next_day_open * (1 + slippage + transaction_cost)
                    
                    self.entry_prices[idx] = entry_price
                    self.log_trade(f"モメンタム エントリー: 日付={self.data.index[idx]}, 翌日始値={next_day_open:.2f}, エントリー価格={entry_price:.2f} (スリッページ+コスト={slippage+transaction_cost:.4f})")
            
            # ポジションを持っている場合のみイグジットシグナルを検討
            elif in_position:
                # ポジションを前日から引き継ぐ
                if idx > 0:
                    self.data.at[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                
                exit_signal = self.generate_exit_signal(idx)
                if exit_signal == -1:
                    self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                    self.data.at[self.data.index[idx], 'Position'] = 0
                    in_position = False
                    # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
                    exit_price = self.data['Open'].iloc[idx + 1]
                    self.log_trade(f"モメンタム イグジット: 日付={self.data.index[idx]}, 価格={exit_price}, エントリー日={self.data.index[entry_idx]}")
                    entry_idx = -1
                else:
                    # 一部利確の処理（ポジションがある場合のみ）
                    partial_exit_pct = self.params.get("partial_exit_pct", 0.0)
                    partial_exit_threshold = self.params.get("partial_exit_threshold", 0.08)
                    if partial_exit_pct > 0 and self.data['Partial_Exit'].iloc[idx-1 if idx > 0 else idx] == 0:
                        entry_price = self.entry_prices.get(entry_idx)
                        if entry_price:
                            current_price = self.data[self.price_column].iloc[idx]
                            profit_pct = (current_price - entry_price) / entry_price
                            self.data.at[self.data.index[idx], 'Profit_Pct'] = profit_pct
                            if profit_pct >= partial_exit_threshold:
                                self.data.at[self.data.index[idx], 'Partial_Exit'] = partial_exit_pct
                                self.data.at[self.data.index[idx], 'Position'] -= partial_exit_pct
                                self.log_trade(f"一部利確 {partial_exit_pct*100}%: 日付={self.data.index[idx]}, 価格={current_price}, 利益={profit_pct:.2%}")
        
        # バックテスト終了時に未決済のポジションがある場合は、最終日に強制決済
        if in_position and entry_idx >= 0:
            last_idx = len(self.data) - 1
            self.data.at[self.data.index[last_idx], 'Exit_Signal'] = -1
            self.data.at[self.data.index[last_idx], 'Position'] = 0
            entry_price = self.entry_prices.get(entry_idx, 0)
            exit_price = self.data[self.price_column].iloc[last_idx]
            profit_pct = 0
            if entry_price > 0:
                profit_pct = (exit_price - entry_price) / entry_price * 100
            
            self.log_trade(f"バックテスト終了時のオープンポジションを強制決済: エントリー日={self.data.index[entry_idx]}, 決済日={self.data.index[last_idx]}, 損益={profit_pct:.2f}%")
        
        # エントリーとエグジットの回数を検証
        entry_count = (self.data['Entry_Signal'] == 1).sum()
        exit_count = (self.data['Exit_Signal'] == -1).sum()
        
        if entry_count != exit_count:
            self.log_trade(f"警告: エントリー ({entry_count}) とエグジット ({exit_count}) の回数が一致しません！")
        
        return self.data

    def load_optimized_parameters(self) -> bool:
        """
        最適化されたパラメータを読み込む
        
        Returns:
            bool: パラメータが正常に読み込まれた場合True
        """
        if not self.optimization_mode:
            return False
            
        try:
            from config.optimized_parameters import OptimizedParameterManager
            
            if self._parameter_manager is None:
                self._parameter_manager = OptimizedParameterManager()
            
            strategy_name = "MomentumInvestingStrategy"
            
            if self.optimization_mode == "interactive":
                # 対話式選択
                selected_params = self._parameter_manager.select_parameters_interactive(strategy_name)
                if selected_params:
                    self.params.update(selected_params['parameters'])
                    self._approved_params = selected_params
                    return True
                    
            elif self.optimization_mode == "best_sharpe":
                # 最高シャープレシオの自動選択
                best_params = self._parameter_manager.get_best_parameters(
                    strategy_name, 
                    metric='sharpe_ratio', 
                    status_filter='approved'
                )
                if best_params:
                    self.params.update(best_params['parameters'])
                    self._approved_params = best_params
                    return True
                    
            elif self.optimization_mode == "best_return":
                # 最高リターンの自動選択
                best_params = self._parameter_manager.get_best_parameters(
                    strategy_name, 
                    metric='total_return', 
                    status_filter='approved'
                )
                if best_params:
                    self.params.update(best_params['parameters'])
                    self._approved_params = best_params
                    return True
                    
            elif self.optimization_mode == "latest_approved":
                # 最新の承認済みパラメータ
                latest_params = self._parameter_manager.get_latest_parameters(
                    strategy_name, 
                    status_filter='approved'
                )
                if latest_params:
                    self.params.update(latest_params['parameters'])
                    self._approved_params = latest_params
                    return True
                    
        except Exception as e:
            print(f"最適化パラメータの読み込みエラー: {e}")
            
        return False
    
    def run_optimized_strategy(self) -> pd.DataFrame:
        """
        最適化されたパラメータを使用して戦略を実行
        
        Returns:
            pd.DataFrame: 戦略実行結果
        """
        # 最適化パラメータの読み込み
        if self.optimization_mode and not self.load_optimized_parameters():
            print(f"[WARNING] 最適化パラメータの読み込みに失敗しました。デフォルトパラメータを使用します。")
        
        # 使用するパラメータの表示
        if self._approved_params:
            print(f"[OK] 最適化パラメータを使用:")
            print(f"   パラメータID: {self._approved_params.get('parameter_id', 'N/A')}")
            print(f"   作成日時: {self._approved_params.get('created_at', 'N/A')}")
            print(f"   シャープレシオ: {self._approved_params.get('sharpe_ratio', 'N/A')}")
            print(f"   パラメータ: {self._approved_params.get('parameters', {})}")
        else:
            print(f"[CHART] デフォルトパラメータを使用: {self.params}")
          # 戦略実行
        return self.backtest()
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        最適化情報を取得
        
        Returns:
            dict: 最適化情報
        """
        info = {
            'optimization_mode': self.optimization_mode,
            'using_optimized_params': self._approved_params is not None,
            'default_params': {
                "sma_short": 20,
                "sma_long": 50,
                "rsi_period": 14,
                "rsi_lower": 50,
                "rsi_upper": 68,
                "volume_threshold": 1.18,
                "take_profit": 0.12,
                "stop_loss": 0.06,
                "trailing_stop": 0.04,
                "ma_type": "SMA",
                "max_hold_days": 15,
                "atr_multiple": 2.0,
                "partial_exit_pct": 0.5,
                "partial_exit_threshold": 0.08,
                "momentum_exit_threshold": -0.03,
                "volume_exit_threshold": 0.7,
                "trend_filter": True
            },
            'current_params': self.params,
            'approved_params_info': self._approved_params
        }
        
        return info

    def backtest_daily(self, current_date, stock_data, existing_position=None, **kwargs):
        """
        MomentumInvestingStrategy 日次バックテスト実行
        
        Phase 3-B Step B3実装: Momentum戦略での実証実装
        
        Cycle 26修正: **kwargs追加
        - 理由: force_close時にentry_symbol_dataがkwargsで渡される（Cycle 7修正）
        
        Sprint 1.5修正 (2026-02-09): force_close強制決済実装
        - force_close=True の場合、エグジット条件に関わらず強制決済
        - 銘柄切替時の旧ポジション自動決済を保証
        
        Parameters:
            current_date (datetime): 判定対象日
            stock_data (pd.DataFrame): 最新の株価データ
            existing_position (dict, optional): 既存ポジション情報
                {
                    'symbol': str,           # 保有銘柄コード
                    'quantity': int,         # 保有株数
                    'entry_price': float,    # エントリー価格
                    'entry_date': datetime,  # エントリー日
                    'entry_idx': int,        # エントリー時のインデックス（オプション）
                    'force_close': bool,     # Sprint 1.5: 強制決済フラグ
                    'entry_symbol': str      # Cycle 7: エントリー銘柄コード
                }
            **kwargs: 追加引数
                - entry_symbol_data (pd.DataFrame): force_close時の元の銘柄データ
                
        Returns:
            dict: {
                'action': 'entry'|'exit'|'hold',
                'signal': 1|-1|0,
                'price': float,
                'shares': int,
                'reason': str
            }
            
        実装内容:
        1. current_dateのindexを特定
        2. ウォームアップ期間考慮（150日）
        3. 前日データのみでMomentumインジケーター計算（shift(1)適用済み）
        4. エントリー/エグジット判定（ルックアヘッドバイアス防止）
        5. 翌日始値エントリー/エグジット価格設定
        
        copilot-instructions.md遵守:
        - バックテスト実行必須
        - フォールバック禁止（実データのみ使用）
        - ルックアヘッドバイアス防止3原則
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Phase 1: current_dateの型変換・検証
        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        elif not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
            
        # Phase 2: データ整合性チェック
        if current_date not in stock_data.index:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'MomentumInvesting: No data available for {current_date.strftime("%Y-%m-%d")}'
            }
            
        # Phase 3: ウォームアップ期間考慮（150日推奨）
        current_idx = stock_data.index.get_loc(current_date)
        warmup_period = 150  # copilot-instructions.mdで推奨される値
        
        # Momentum戦略の最小要求期間も考慮
        min_required = max(warmup_period, self.params.get("sma_long", 50))
        
        if current_idx < min_required:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'MomentumInvesting: Insufficient warmup data. Required: {min_required}, Available: {current_idx}'
            }
        
        # Phase 4: データ更新（Option B方式を活用）
        # 既存のself.dataを一時保存
        original_data = self.data.copy()
        
        try:
            # BaseStrategy.backtest_daily()の Option B ロジックを活用
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    # インデックスが一致する部分のみ安全に更新
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            logger.debug(f"[MomentumInvesting.backtest_daily] Data updated: {updated_columns}")
            
            # Phase 5: 既存ポジション処理分岐
            if existing_position is not None:
                # 【既存ポジションあり: エグジット判定】
                # エントリーインデックスの取得
                entry_idx = existing_position.get('entry_idx', current_idx)
                
                # entry_pricesに記録（エグジット判定で使用）
                if entry_idx not in self.entry_prices:
                    self.entry_prices[entry_idx] = existing_position.get('entry_price', 0.0)
                
                # Cycle 27修正: entry_symbol_dataをkwargsから取得
                entry_symbol_data = kwargs.get('entry_symbol_data', None)
                
                # 最終日チェック
                if current_idx + 1 >= len(stock_data):
                    # 最終日の場合は当日終値でエグジット（境界条件）
                    # Cycle 27修正: entry_symbol_data優先
                    data_for_exit = entry_symbol_data if entry_symbol_data is not None else stock_data
                    exit_price = data_for_exit.iloc[current_idx]['Close']
                    logger.warning(f"[MomentumInvesting.exit] Final day exit: {current_date}")
                    
                    return {
                        'action': 'exit',
                        'signal': -1,
                        'price': float(exit_price),
                        'shares': existing_position.get('quantity', 0),
                        'reason': f'MomentumInvesting: Final day exit on {current_date.strftime("%Y-%m-%d")}'
                    }
                
                # エグジット判定（簡易実装: entry_prices辞書を使用）
                # Sprint 1.5修正: _handle_exit_logic()にentry_symbol_data渡す
                from strategies.Momentum_Investing_backtest_daily import _handle_exit_logic
                return _handle_exit_logic(self, current_idx, existing_position, stock_data, current_date, entry_symbol_data=entry_symbol_data)
            else:
                # 【既存ポジションなし: エントリー判定】
                entry_signal = self.generate_entry_signal(current_idx)
                
                if entry_signal == 1:
                    if current_idx + 1 < len(stock_data):
                        entry_price = stock_data.iloc[current_idx + 1]['Open']
                        if isinstance(entry_price, pd.Series):
                            entry_price = entry_price.values[0]
                        
                        slippage = self.params.get("slippage", 0.001)
                        transaction_cost = self.params.get("transaction_cost", 0.0)
                        entry_price = entry_price * (1 + slippage + transaction_cost)
                        
                        # ポジションサイズ計算
                        target_amount = self.params.get("position_amount", 100000)
                        shares = int(target_amount / entry_price) if entry_price > 0 else 0
                        shares = max(100, shares // 100 * 100)
                        
                        return {
                            'action': 'entry',
                            'signal': 1,
                            'price': float(entry_price),
                            'shares': shares,
                            'reason': f'MomentumInvesting: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                        }
                    else:
                        return {
                            'action': 'hold',
                            'signal': 0,
                            'price': 0.0,
                            'shares': 0,
                            'reason': f'MomentumInvesting: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                        }
                else:
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': f'MomentumInvesting: No entry signal on {current_date.strftime("%Y-%m-%d")}'
                    }
        
        except Exception as e:
            logger.error(f"[MomentumInvesting.backtest_daily] Error: {e}", exc_info=True)
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'MomentumInvesting: Error: {str(e)}'
            }
        
        finally:
            # データの復元
            self.data = original_data


# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=200, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(200) * 100,
        'High': np.random.random(200) * 100,
        'Low': np.random.random(200) * 100,
        'Adj Close': np.random.random(200) * 100,
        'Volume': np.random.randint(100, 1000, 200)
    }, index=dates)

    # モメンタム戦略の実行
    strategy = MomentumInvestingStrategy(df)
    result = strategy.backtest()
    print(result)