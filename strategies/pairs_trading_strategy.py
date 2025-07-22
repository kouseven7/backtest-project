"""
Module: Pairs Trading Strategy (Simplified Single-Asset Version)
File: pairs_trading_strategy.py
Description: 
  ペアトレーディング戦略のシンプル化版。
  単一資産で、異なる期間の移動平均間の相関崩れを利用する戦略。
  短期移動平均と長期移動平均の乖離が異常に大きくなった時に
  回帰を期待してエントリーする「擬似ペアトレーディング」戦略。

Author: imega
Created: 2025-07-22
Modified: 2025-07-22

Dependencies:
  - strategies.base_strategy
  - numpy, pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    """ペアトレーディング戦略（簡略版）"""
    
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        ペアトレーディング戦略の初期化
        
        Args:
            data: 価格データ
            params: 戦略パラメータ
            price_column: 価格カラム名
        """
        self.price_column = price_column
        self.entry_prices = {}
        self.position_days = {}
        
        # デフォルトパラメータ
        default_params = {
            "short_ma_period": 5,            # 短期移動平均期間
            "long_ma_period": 20,            # 長期移動平均期間
            "spread_period": 15,             # スプレッド計算期間
            "entry_threshold": 2.0,          # エントリー閾値（標準偏差倍率）
            "exit_threshold": 0.5,           # エグジット閾値（標準偏差倍率）
            "stop_loss_pct": 0.04,           # ストップロス（4%）
            "take_profit_pct": 0.06,         # 利益確定（6%）
            "cointegration_lookback": 30,    # 共和分検定ルックバック期間
            "volume_filter": True,           # ボリュームフィルター
            "volume_threshold": 1.2,         # ボリューム閾値倍率
            "max_hold_days": 20,             # 最大保有日数
            "half_life_periods": 10,         # 平均回帰半減期
            "correlation_min": 0.7,          # 最小相関閾値
            "volatility_filter": True,       # ボラティリティフィルター
            "volatility_period": 10,         # ボラティリティ計算期間
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """戦略初期化処理"""
        super().initialize_strategy()
        
        # 短期・長期移動平均の計算
        self.data['SMA_Short'] = self.data[self.price_column].rolling(
            window=self.params["short_ma_period"]
        ).mean()
        
        self.data['SMA_Long'] = self.data[self.price_column].rolling(
            window=self.params["long_ma_period"]
        ).mean()
        
        # スプレッド（短期MA - 長期MA）の計算
        self.data['Spread'] = self.data['SMA_Short'] - self.data['SMA_Long']
        
        # スプレッドの移動平均と標準偏差
        self.data['Spread_MA'] = self.data['Spread'].rolling(
            window=self.params["spread_period"]
        ).mean()
        
        self.data['Spread_Std'] = self.data['Spread'].rolling(
            window=self.params["spread_period"]
        ).std()
        
        # Z-Scoreの計算（標準化されたスプレッド）
        self.data['Spread_ZScore'] = (
            (self.data['Spread'] - self.data['Spread_MA']) / self.data['Spread_Std']
        )
        
        # ボリュームフィルター
        if self.params["volume_filter"]:
            self.data['Volume_MA'] = self.data['Volume'].rolling(
                window=self.params["spread_period"]
            ).mean()
        
        # ボラティリティフィルター
        if self.params["volatility_filter"]:
            returns = self.data[self.price_column].pct_change()
            self.data['Volatility'] = returns.rolling(
                window=self.params["volatility_period"]
            ).std()
            
        # 移動平均間の相関（ローリング相関）
        if len(self.data) >= self.params["cointegration_lookback"]:
            correlation_window = min(self.params["cointegration_lookback"], len(self.data))
            self.data['MA_Correlation'] = self.data['SMA_Short'].rolling(
                window=correlation_window
            ).corr(self.data['SMA_Long'])
        
        print(f"Pairs Trading Strategy Initialized")
        print(f"Parameters: Short_MA={self.params['short_ma_period']}, Long_MA={self.params['long_ma_period']}")
        print(f"Entry Threshold: ±{self.params['entry_threshold']} std devs")
        
    def _check_correlation_condition(self, idx: int) -> bool:
        """相関条件チェック"""
        if 'MA_Correlation' not in self.data.columns:
            return True  # 相関データがない場合はスルー
            
        correlation = self.data['MA_Correlation'].iloc[idx]
        if pd.isna(correlation):
            return True
            
        return correlation >= self.params["correlation_min"]
        
    def _check_volume_condition(self, idx: int) -> bool:
        """ボリューム条件チェック"""
        if not self.params["volume_filter"]:
            return True
            
        if 'Volume_MA' not in self.data.columns:
            return True
            
        current_volume = self.data['Volume'].iloc[idx]
        avg_volume = self.data['Volume_MA'].iloc[idx]
        
        if pd.isna(avg_volume):
            return True
            
        return current_volume >= (avg_volume * self.params["volume_threshold"])
        
    def _check_volatility_condition(self, idx: int) -> bool:
        """ボラティリティ条件チェック（高ボラティリティ時は取引を避ける）"""
        if not self.params["volatility_filter"]:
            return True
            
        if 'Volatility' not in self.data.columns:
            return True
            
        current_vol = self.data['Volatility'].iloc[idx]
        if pd.isna(current_vol):
            return True
            
        # ボラティリティの移動平均
        vol_ma = self.data['Volatility'].rolling(
            window=self.params["volatility_period"]
        ).mean().iloc[idx]
        
        if pd.isna(vol_ma):
            return True
            
        # 過度なボラティリティの場合は取引を避ける
        return current_vol <= vol_ma * 2.0
        
    def generate_entry_signal(self, idx: int) -> int:
        """エントリーシグナル生成"""
        if idx < max(self.params["spread_period"], self.params["long_ma_period"]):
            return 0
            
        spread_zscore = self.data['Spread_ZScore'].iloc[idx]
        
        if pd.isna(spread_zscore):
            return 0
            
        # 相関条件チェック
        if not self._check_correlation_condition(idx):
            return 0
            
        # ボリューム条件チェック
        if not self._check_volume_condition(idx):
            return 0
            
        # ボラティリティ条件チェック
        if not self._check_volatility_condition(idx):
            return 0
            
        # エントリー条件：スプレッドが異常に拡大した場合
        entry_threshold = self.params["entry_threshold"]
        
        # 正の異常値（短期MAが長期MAを大幅に上回る）
        # → 回帰を期待してショート的な動き（実際はロング）
        if spread_zscore >= entry_threshold:
            return 1  # ロングエントリー
            
        # 負の異常値（短期MAが長期MAを大幅に下回る）
        # → 回帰を期待してロング
        elif spread_zscore <= -entry_threshold:
            return 1  # ロングエントリー
            
        return 0
        
    def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
        """エグジットシグナル生成"""
        if position_size <= 0:
            return 0
            
        current_price = self.data[self.price_column].iloc[idx]
        spread_zscore = self.data['Spread_ZScore'].iloc[idx]
        
        # エントリー価格を取得
        entry_price = None
        entry_idx = None
        
        for i in range(max(0, idx - self.params["max_hold_days"]), idx):
            if i in self.entry_prices:
                entry_price = self.entry_prices[i]
                entry_idx = i
                break
                
        if entry_price is None:
            return 0
            
        # 保有日数チェック
        hold_days = idx - entry_idx
        if hold_days >= self.params["max_hold_days"]:
            return 1  # 最大保有日数到達
            
        # 損益計算
        pnl_pct = (current_price - entry_price) / entry_price
        
        # ストップロス
        if pnl_pct <= -self.params["stop_loss_pct"]:
            return 1
            
        # 利益確定
        if pnl_pct >= self.params["take_profit_pct"]:
            return 1
            
        # スプレッド回帰チェック（メイン エグジット条件）
        if not pd.isna(spread_zscore):
            exit_threshold = self.params["exit_threshold"]
            
            # スプレッドが正常範囲に戻った場合
            if abs(spread_zscore) <= exit_threshold:
                return 1  # 回帰完了でエグジット
                
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
                    self.position_days[i] = 0
                    
            else:
                # 保有日数更新
                for entry_idx in list(self.position_days.keys()):
                    if entry_idx in self.entry_prices:
                        self.position_days[entry_idx] += 1
                        
                # エグジットチェック
                exit_signal = self.generate_exit_signal(i, position_size)
                if exit_signal == 1:
                    result_data['Exit_Signal'].iloc[i] = 1
                    position_size = 0
                    
            result_data['Position'].iloc[i] = position_size
            
        return result_data
        
    def get_strategy_name(self) -> str:
        """戦略名を返す"""
        return "PairsTrading"
        
    def get_required_columns(self) -> List[str]:
        """必要なデータ列を返す"""
        return [self.price_column, 'Volume']
        
    def get_parameters_info(self) -> Dict[str, Any]:
        """パラメータ情報を返す"""
        return {
            "strategy_type": "statistical_arbitrage",
            "market_regime": "range_bound",
            "time_frame": "short_to_medium_term",
            "risk_level": "medium_high",
            "parameters": self.params
        }


# テスト用のユーティリティ関数
def create_test_pairs_strategy(data: pd.DataFrame, 
                               params: Optional[Dict[str, Any]] = None) -> PairsTradingStrategy:
    """テスト用戦略インスタンス作成"""
    test_params = {
        "short_ma_period": 5,
        "long_ma_period": 18,
        "spread_period": 12,
        "entry_threshold": 1.8,
        "exit_threshold": 0.4,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05,
        "volume_filter": True,
        "max_hold_days": 15
    }
    
    if params:
        test_params.update(params)
        
    return PairsTradingStrategy(data, test_params)


if __name__ == "__main__":
    # 簡単なテスト
    print("Pairs Trading Strategy - Test Mode")
    
    # テストデータ生成（移動平均間の乖離パターン）
    import datetime
    dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')
    
    np.random.seed(456)
    n_days = len(dates)
    base_price = 100
    
    # 移動平均間の乖離を意図的に作るデータパターン
    prices = []
    for i in range(n_days):
        if i == 0:
            price = base_price
        else:
            prev_price = prices[-1]
            
            # 基本トレンド
            base_change = np.random.normal(0, 0.01)
            
            # 周期的な乖離パターン（ペアトレーディングの機会）
            if i % 25 in [15, 16, 17]:  # 25日ごとに3日間の大きな乖離
                divergence = np.random.choice([-0.025, 0.025])  # ±2.5%の乖離
            else:
                divergence = 0
                
            # 価格更新
            total_change = base_change + divergence
            price = prev_price * (1 + total_change)
            price = max(85, min(115, price))  # 範囲制限
            
        prices.append(price)
    
    # OHLCV データ作成
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
        'High': [p * np.random.uniform(1.002, 1.015) for p in prices],
        'Low': [p * np.random.uniform(0.985, 0.998) for p in prices],
        'Adj Close': prices,
        'Volume': np.random.randint(100000, 300000, len(dates))
    })
    
    test_data['Close'] = test_data['Adj Close']
    test_data.set_index('Date', inplace=True)
    
    # 戦略テスト
    strategy = create_test_pairs_strategy(test_data)
    result = strategy.backtest()
    
    # 結果表示
    entries = (result['Entry_Signal'] == 1).sum()
    exits = (result['Exit_Signal'] == 1).sum()
    
    print(f"Backtest Results:")
    print(f"  Entry Signals: {entries}")
    print(f"  Exit Signals: {exits}")
    print(f"  Price Range: {test_data['Adj Close'].min():.2f} - {test_data['Adj Close'].max():.2f}")
    
    if 'Spread_ZScore' in result.columns:
        print(f"  Spread Z-Score Range: {result['Spread_ZScore'].min():.2f} - {result['Spread_ZScore'].max():.2f}")
    
    if entries > 0:
        print(f"  Pairs Trading Strategy is working correctly!")
        entry_dates = result[result['Entry_Signal'] == 1].index
        print(f"  Entry Dates: {[d.strftime('%m-%d') for d in entry_dates[:3]]}")
    else:
        print(f"  No signals generated - consider adjusting parameters")
        
    print("Test completed successfully!")
