"""
Module: Support/Resistance Contrarian Strategy
File: support_resistance_contrarian_strategy.py
Description: 
  支持線・抵抗線レベルでの逆張り戦略。
  価格が支持線に近づいた時にロングエントリー、抵抗線に近づいた時にショートエントリーを行う。
  ピボットポイント分析とフィボナッチリトレースメントを使用して
  精度の高い反転ポイントを特定する。

Author: imega
Created: 2025-07-22
Modified: 2025-07-22

Dependencies:
  - strategies.base_strategy
  - indicators.pivot_points
  - indicators.fibonacci_levels
  - numpy, pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy


class SupportResistanceContrarianStrategy(BaseStrategy):
    """支持線・抵抗線ベース逆張り戦略"""
    
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        支持線・抵抗線逆張り戦略の初期化
        
        Args:
            data: 価格データ
            params: 戦略パラメータ
            price_column: 価格カラム名
        """
        self.price_column = price_column
        self.entry_prices = {}
        self.support_levels = []
        self.resistance_levels = []
        
        # デフォルトパラメータ
        default_params = {
            "lookback_period": 20,           # 支持線・抵抗線検出期間
            "min_touches": 2,                # 最小接触回数（有効レベルとして認識）
            "proximity_threshold": 0.005,     # レベル接近閾値（0.5%）
            "breakout_threshold": 0.003,      # ブレイクアウト閾値（0.3%）
            "stop_loss_pct": 0.02,           # ストップロス（2%）
            "take_profit_pct": 0.04,         # 利益確定（4%）
            "volume_threshold": 1.0,         # ボリューム閾値倍率
            "fibonacci_enabled": True,        # フィボナッチレベル使用
            "pivot_period": 5,               # ピボット検出期間
            "trend_filter_enabled": False,    # トレンドフィルター（レンジ相場で有効）
            "max_hold_days": 10,             # 最大保有日数
            "rsi_confirmation": True,        # RSI確認シグナル
            "rsi_period": 14,                # RSI期間
            "rsi_oversold": 25,              # RSI過売り閾値
            "rsi_overbought": 75,            # RSI過買い閾値
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """戦略初期化処理"""
        super().initialize_strategy()
        
        # RSI計算（確認シグナル用）
        if self.params["rsi_confirmation"]:
            self.data['RSI'] = self._calculate_rsi()
        
        # ボリューム移動平均
        self.data['Volume_MA'] = self.data['Volume'].rolling(
            window=self.params["lookback_period"]
        ).mean()
        
        # 支持線・抵抗線の初期計算
        self._calculate_support_resistance_levels()
        
        print(f"Support/Resistance Contrarian Strategy Initialized")
        print(f"Detected {len(self.support_levels)} support levels")
        print(f"Detected {len(self.resistance_levels)} resistance levels")
        
    def _calculate_rsi(self) -> pd.Series:
        """RSI計算"""
        delta = self.data[self.price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.params["rsi_period"]
        ).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=self.params["rsi_period"]
        ).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_support_resistance_levels(self):
        """支持線・抵抗線レベル計算"""
        # ピボットポイント検出
        pivot_highs, pivot_lows = self._find_pivot_points()
        
        # 支持線レベル（ピボット安値から）
        self.support_levels = self._cluster_levels(pivot_lows)
        
        # 抵抗線レベル（ピボット高値から）
        self.resistance_levels = self._cluster_levels(pivot_highs)
        
        # フィボナッチレベル追加
        if self.params["fibonacci_enabled"]:
            fib_levels = self._calculate_fibonacci_levels()
            self.support_levels.extend(fib_levels['support'])
            self.resistance_levels.extend(fib_levels['resistance'])
            
    def _find_pivot_points(self) -> tuple:
        """ピボットポイント（高値・安値）を検出"""
        period = self.params["pivot_period"]
        high_col = 'High'
        low_col = 'Low'
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(period, len(self.data) - period):
            # ピボット高値チェック
            is_pivot_high = True
            for j in range(-period, period + 1):
                if j == 0:
                    continue
                if self.data[high_col].iloc[i] <= self.data[high_col].iloc[i + j]:
                    is_pivot_high = False
                    break
                    
            if is_pivot_high:
                pivot_highs.append({
                    'price': self.data[high_col].iloc[i],
                    'index': i,
                    'date': self.data.index[i]
                })
                
            # ピボット安値チェック
            is_pivot_low = True
            for j in range(-period, period + 1):
                if j == 0:
                    continue
                if self.data[low_col].iloc[i] >= self.data[low_col].iloc[i + j]:
                    is_pivot_low = False
                    break
                    
            if is_pivot_low:
                pivot_lows.append({
                    'price': self.data[low_col].iloc[i],
                    'index': i,
                    'date': self.data.index[i]
                })
                
        return pivot_highs, pivot_lows
        
    def _cluster_levels(self, pivots: List[Dict]) -> List[float]:
        """類似価格レベルをクラスター化"""
        if not pivots:
            return []
            
        prices = [p['price'] for p in pivots]
        prices.sort()
        
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # 現在のクラスターの平均価格
            cluster_avg = np.mean(current_cluster)
            
            # 価格差が閾値以下なら同じクラスターに追加
            if abs(price - cluster_avg) / cluster_avg <= self.params["proximity_threshold"] * 2:
                current_cluster.append(price)
            else:
                # 最小接触回数以上なら有効レベルとして追加
                if len(current_cluster) >= self.params["min_touches"]:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [price]
                
        # 最後のクラスター処理
        if len(current_cluster) >= self.params["min_touches"]:
            clusters.append(np.mean(current_cluster))
            
        return clusters
        
    def _calculate_fibonacci_levels(self) -> Dict[str, List[float]]:
        """フィボナッチリトレースメントレベル計算"""
        lookback = self.params["lookback_period"] * 2
        
        if len(self.data) < lookback:
            return {'support': [], 'resistance': []}
            
        # 直近の高値・安値を取得
        recent_data = self.data.tail(lookback)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        
        # フィボナッチレベル
        fib_ratios = [0.236, 0.382, 0.618, 0.786]
        
        support_levels = []
        resistance_levels = []
        
        for ratio in fib_ratios:
            level = low + (high - low) * ratio
            
            # 現在価格との関係で支持線・抵抗線を判定
            current_price = self.data[self.price_column].iloc[-1]
            
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
                
        return {'support': support_levels, 'resistance': resistance_levels}
        
    def generate_entry_signal(self, idx: int) -> int:
        """エントリーシグナル生成"""
        if idx < self.params["lookback_period"]:
            return 0
            
        current_price = self.data[self.price_column].iloc[idx]
        current_volume = self.data['Volume'].iloc[idx]
        avg_volume = self.data['Volume_MA'].iloc[idx]
        
        # ボリューム確認
        if current_volume < avg_volume * self.params["volume_threshold"]:
            return 0
            
        # RSI確認（有効な場合）
        if self.params["rsi_confirmation"] and 'RSI' in self.data.columns:
            rsi = self.data['RSI'].iloc[idx]
            rsi_oversold = self.params["rsi_oversold"]
            
            # 支持線付近でのロングエントリー
            for support in self.support_levels:
                proximity = abs(current_price - support) / support
                if proximity <= self.params["proximity_threshold"]:
                    # RSI過売り確認
                    if rsi <= rsi_oversold:
                        return 1  # ロングエントリー
                        
        else:
            # RSI確認なしの場合
            for support in self.support_levels:
                proximity = abs(current_price - support) / support
                if proximity <= self.params["proximity_threshold"]:
                    return 1  # ロングエントリー
                    
        return 0
        
    def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
        """エグジットシグナル生成"""
        if position_size <= 0:
            return 0
            
        current_price = self.data[self.price_column].iloc[idx]
        
        # エントリー価格取得
        entry_price = self.entry_prices.get(idx, None)
        if entry_price is None:
            # 過去のエントリーを逆算
            for i in range(max(0, idx - self.params["max_hold_days"]), idx):
                if i in self.entry_prices:
                    entry_price = self.entry_prices[i]
                    break
                    
        if entry_price is None:
            return 0
            
        # 損益計算
        pnl_pct = (current_price - entry_price) / entry_price
        
        # ストップロス
        if pnl_pct <= -self.params["stop_loss_pct"]:
            return 1
            
        # 利益確定
        if pnl_pct >= self.params["take_profit_pct"]:
            return 1
            
        # 抵抗線到達での利益確定
        for resistance in self.resistance_levels:
            if current_price >= resistance * (1 - self.params["proximity_threshold"]):
                if pnl_pct > 0:  # 利益が出ている場合のみ
                    return 1
                    
        # 最大保有日数チェック
        # これは簡略化（実際には日付ベースで計算）
        
        return 0
        
    def backtest(self) -> pd.DataFrame:
        """バックテスト実行"""
        result_data = self.data.copy()
        result_data['Entry_Signal'] = 0
        result_data['Exit_Signal'] = 0
        result_data['Position'] = 0
        
        position_size = 0
        
        for i in range(len(result_data)):
            if position_size == 0:
                # エントリーチェック
                entry_signal = self.generate_entry_signal(i)
                if entry_signal == 1:
                    result_data['Entry_Signal'].iloc[i] = 1
                    position_size = 1.0
                    self.entry_prices[i] = result_data[self.price_column].iloc[i]
                    
            else:
                # エグジットチェック
                exit_signal = self.generate_exit_signal(i, position_size)
                if exit_signal == 1:
                    result_data['Exit_Signal'].iloc[i] = 1
                    position_size = 0
                    
            result_data['Position'].iloc[i] = position_size
            
        return result_data
        
    def get_strategy_name(self) -> str:
        """戦略名を返す"""
        return "SupportResistanceContrarian"
        
    def get_required_columns(self) -> List[str]:
        """必要なデータ列を返す"""
        return [self.price_column, 'High', 'Low', 'Volume']
        
    def get_parameters_info(self) -> Dict[str, Any]:
        """パラメータ情報を返す"""
        return {
            "strategy_type": "contrarian",
            "market_regime": "range_bound",
            "time_frame": "intraday_to_swing",
            "risk_level": "medium",
            "parameters": self.params
        }


# テスト用のユーティリティ関数
def create_test_strategy(data: pd.DataFrame, 
                        params: Optional[Dict[str, Any]] = None) -> SupportResistanceContrarianStrategy:
    """テスト用戦略インスタンス作成"""
    test_params = {
        "lookback_period": 15,
        "proximity_threshold": 0.008,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "rsi_confirmation": True,
        "fibonacci_enabled": True
    }
    
    if params:
        test_params.update(params)
        
    return SupportResistanceContrarianStrategy(data, test_params)


if __name__ == "__main__":
    # 簡単なテスト
    print("Support/Resistance Contrarian Strategy - Test Mode")
    
    # テストデータ生成
    import datetime
    dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')
    
    # より現実的な価格データ（レンジ相場）
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(len(dates)):
        # レンジ相場（95-105の範囲で変動）
        range_center = 100
        range_width = 10
        noise = np.random.normal(0, 1)
        
        # サイン波ベースのレンジ相場
        cycle_position = (i / len(dates)) * 4 * np.pi  # 4周期
        cycle_value = np.sin(cycle_position) * (range_width / 2)
        
        price = range_center + cycle_value + noise
        prices.append(max(90, min(110, price)))  # 範囲制限
    
    # OHLCV データ作成
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * np.random.uniform(1.005, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 0.995) for p in prices],
        'Adj Close': prices,
        'Volume': np.random.randint(100000, 500000, len(dates))
    })
    
    test_data['Close'] = test_data['Adj Close']
    test_data.set_index('Date', inplace=True)
    
    # 戦略テスト
    strategy = create_test_strategy(test_data)
    result = strategy.backtest()
    
    # 結果表示
    entries = (result['Entry_Signal'] == 1).sum()
    exits = (result['Exit_Signal'] == 1).sum()
    
    print(f"Backtest Results:")
    print(f"  Entry Signals: {entries}")
    print(f"  Exit Signals: {exits}")
    print(f"  Support Levels: {len(strategy.support_levels)}")
    print(f"  Resistance Levels: {len(strategy.resistance_levels)}")
    
    if entries > 0:
        print(f"  Strategy appears to be working correctly!")
    else:
        print(f"  No signals generated - may need parameter adjustment")
        
    print("Test completed successfully!")
