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

import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_sma, calculate_rsi
from indicators.momentum_indicators import calculate_macd
from indicators.volume_analysis import detect_volume_increase
from indicators.volatility_indicators import calculate_atr

class MomentumInvestingStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """モメンタム戦略の初期化（拡張版）"""
        # 戦略固有の属性
        self.price_column = price_column
        self.volume_column = volume_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
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
            "trend_filter": True            # トレンドフィルターの使用
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

        # 既に指標列がある場合は再計算しない
        if f'MA_{sma_short}' not in self.data.columns:
            if ma_type == "SMA":
                self.data[f'MA_{sma_short}'] = calculate_sma(self.data, self.price_column, sma_short)
            elif ma_type == "EMA":
                self.data[f'MA_{sma_short}'] = self.data[self.price_column].ewm(span=sma_short, adjust=False).mean()
        if f'MA_{sma_long}' not in self.data.columns:
            if ma_type == "SMA":
                self.data[f'MA_{sma_long}'] = calculate_sma(self.data, self.price_column, sma_long)
            elif ma_type == "EMA":
                self.data[f'MA_{sma_long}'] = self.data[self.price_column].ewm(span=sma_long, adjust=False).mean()
        if 'RSI' not in self.data.columns:
            self.data['RSI'] = calculate_rsi(self.data[self.price_column], self.params["rsi_period"])
        if 'MACD' not in self.data.columns or 'Signal_Line' not in self.data.columns:
            self.data['MACD'], self.data['Signal_Line'] = calculate_macd(self.data, self.price_column)
        if 'ATR' not in self.data.columns:
            self.data['ATR'] = calculate_atr(self.data, self.price_column)

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
            self.entry_prices[idx] = current_price
            self.log_trade(f"Momentum Investing エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}, 条件数={condition_count}/7")
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
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        if latest_entry_idx not in self.entry_prices:
            # 記録されていない場合は価格を取得して記録
            self.entry_prices[latest_entry_idx] = self.data[self.price_column].iloc[latest_entry_idx]
            
        entry_price = self.entry_prices[latest_entry_idx]
        current_price = self.data[self.price_column].iloc[idx]
        atr = self.data['ATR'].iloc[latest_entry_idx]  # エントリー時点のATR
        sma_short_key = 'MA_' + str(self.params["sma_short"])
            
        # 最大保有期間によるイグジット
        max_hold_days = self.params.get("max_hold_days")
        if max_hold_days is not None:
            days_held = idx - latest_entry_idx
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
        high_since_entry = self.data['High'].iloc[latest_entry_idx:idx+1].max()
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

    def backtest(self):
        """モメンタム戦略のバックテストを実行（部分利確機能付き）"""
        # シグナル列の初期化
        self.data.loc[:, 'Entry_Signal'] = 0
        self.data.loc[:, 'Exit_Signal'] = 0
        self.data.loc[:, 'Position'] = 0
        self.data.loc[:, 'Partial_Exit'] = 0
        self.data.loc[:, 'Profit_Pct'] = 0

        for idx in range(len(self.data)):
            # 既存のコード（エントリー/イグジットのロジック）
            
            # ポジションの更新
            if idx > 0:
                self.data.at[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                
                # エントリーシグナルでポジションを1に設定
                if self.data['Entry_Signal'].iloc[idx] == 1:
                    self.data.at[self.data.index[idx], 'Position'] = 1
                    # エントリー価格を記録
                    entry_price = self.data[self.price_column].iloc[idx]
                    self.entry_prices[idx] = entry_price
                
                # イグジットシグナルでポジションを0に設定
                if self.data['Exit_Signal'].iloc[idx] == -1:
                    self.data.at[self.data.index[idx], 'Position'] = 0
            
            # 一部利確の処理（ポジションがある場合のみ）
            if idx > 0 and self.data['Position'].iloc[idx-1] > 0:
                # 一部利確のパラメータ
                partial_exit_pct = self.params.get("partial_exit_pct", 0.0)
                partial_exit_threshold = self.params.get("partial_exit_threshold", 0.08)
                
                # 一部利確が有効で、まだ実行されていない場合
                if partial_exit_pct > 0 and self.data['Partial_Exit'].iloc[idx-1] == 0:
                    # エントリー価格を取得
                    entry_idx = self.data.index.get_loc(self.data[self.data['Entry_Signal'] == 1].index[-1])
                    entry_price = self.entry_prices.get(entry_idx)
                    
                    if entry_price:
                        current_price = self.data[self.price_column].iloc[idx]
                        profit_pct = (current_price - entry_price) / entry_price
                        
                        # 利益率を記録
                        self.data.at[self.data.index[idx], 'Profit_Pct'] = profit_pct
                        
                        # 閾値を超えたら一部利確を実行
                        if profit_pct >= partial_exit_threshold:
                            self.data.at[self.data.index[idx], 'Partial_Exit'] = partial_exit_pct
                            self.data.at[self.data.index[idx], 'Position'] -= partial_exit_pct
                            self.log_trade(f"一部利確 {partial_exit_pct*100}%: 日付={self.data.index[idx]}, 価格={current_price}, 利益={profit_pct:.2%}")

        return self.data

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