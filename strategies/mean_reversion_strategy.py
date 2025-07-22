"""
Module: Mean Reversion Strategy
File: mean_reversion_strategy.py
Description: 
  平均回帰戦略 - 価格が移動平均から大きく乖離した際に
  平均回帰を期待してエントリーする戦略。
  ボリンジャーバンドとZ-scoreを使用して統計的な異常値を検出し、
  逆張りエントリーを実行する。

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


class MeanReversionStrategy(BaseStrategy):
    """平均回帰戦略"""
    
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        平均回帰戦略の初期化
        
        Args:
            data: 価格データ
            params: 戦略パラメータ
            price_column: 価格カラム名
        """
        self.price_column = price_column
        self.entry_prices = {}
        self.position_days = {}  # 保有日数追跡
        
        # デフォルトパラメータ
        default_params = {
            "sma_period": 20,                 # 移動平均期間
            "bb_period": 20,                  # ボリンジャーバンド期間
            "bb_std_dev": 2.0,               # ボリンジャーバンド標準偏差倍率
            "zscore_period": 15,             # Z-score計算期間
            "zscore_entry_threshold": -1.8,   # Z-scoreエントリー閾値（負の値で売られすぎ）
            "zscore_exit_threshold": -0.3,    # Z-scoreイグジット閾値
            "stop_loss_pct": 0.03,           # ストップロス（3%）
            "take_profit_pct": 0.05,         # 利益確定（5%）
            "volume_confirmation": True,      # ボリューム確認
            "volume_threshold": 0.8,         # ボリューム倍率閾値
            "rsi_filter": True,              # RSIフィルター使用
            "rsi_period": 14,                # RSI期間
            "rsi_oversold": 30,              # RSI過売り閾値
            "max_hold_days": 15,             # 最大保有日数
            "atr_filter": True,              # ATRボラティリティフィルター
            "atr_period": 14,                # ATR期間
            "atr_multiplier": 1.5,           # ATRストップ倍率
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """戦略初期化処理"""
        super().initialize_strategy()
        
        # 移動平均の計算
        self.data['SMA'] = self.data[self.price_column].rolling(
            window=self.params["sma_period"]
        ).mean()
        
        # ボリンジャーバンドの計算
        bb_sma = self.data[self.price_column].rolling(
            window=self.params["bb_period"]
        ).mean()
        bb_std = self.data[self.price_column].rolling(
            window=self.params["bb_period"]
        ).std()
        
        self.data['BB_Upper'] = bb_sma + (bb_std * self.params["bb_std_dev"])
        self.data['BB_Lower'] = bb_sma - (bb_std * self.params["bb_std_dev"])
        self.data['BB_Middle'] = bb_sma
        
        # Z-score計算（統計的異常値検出）
        z_sma = self.data[self.price_column].rolling(
            window=self.params["zscore_period"]
        ).mean()
        z_std = self.data[self.price_column].rolling(
            window=self.params["zscore_period"]
        ).std()
        
        self.data['Z_Score'] = (self.data[self.price_column] - z_sma) / z_std
        
        # RSIフィルター（オプション）
        if self.params["rsi_filter"]:
            self.data['RSI'] = self._calculate_rsi()
        
        # ATR（ボラティリティベースのストップロス）
        if self.params["atr_filter"]:
            self.data['ATR'] = self._calculate_atr()
            
        # ボリューム移動平均
        if self.params["volume_confirmation"]:
            self.data['Volume_MA'] = self.data['Volume'].rolling(
                window=self.params["sma_period"]
            ).mean()
            
        print(f"Mean Reversion Strategy Initialized")
        print(f"Parameters: SMA={self.params['sma_period']}, BB_StdDev={self.params['bb_std_dev']}, Z-Score_Threshold={self.params['zscore_entry_threshold']}")
        
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
        
    def _calculate_atr(self) -> pd.Series:
        """ATR（Average True Range）計算"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data[self.price_column].shift())
        low_close = np.abs(self.data['Low'] - self.data[self.price_column].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.params["atr_period"]).mean()
        return atr
        
    def _is_statistical_anomaly(self, idx: int) -> bool:
        """統計的異常値判定"""
        if idx < max(self.params["zscore_period"], self.params["bb_period"]):
            return False
            
        z_score = self.data['Z_Score'].iloc[idx]
        current_price = self.data[self.price_column].iloc[idx]
        bb_lower = self.data['BB_Lower'].iloc[idx]
        
        # Z-scoreベースの判定
        z_score_condition = z_score <= self.params["zscore_entry_threshold"]
        
        # ボリンジャーバンドベースの判定
        bb_condition = current_price <= bb_lower
        
        return z_score_condition and bb_condition
        
    def _volume_confirmation_check(self, idx: int) -> bool:
        """ボリューム確認"""
        if not self.params["volume_confirmation"]:
            return True
            
        if 'Volume_MA' not in self.data.columns:
            return True
            
        current_volume = self.data['Volume'].iloc[idx]
        avg_volume = self.data['Volume_MA'].iloc[idx]
        
        if pd.isna(avg_volume):
            return True
            
        return current_volume >= (avg_volume * self.params["volume_threshold"])
        
    def generate_entry_signal(self, idx: int) -> int:
        """エントリーシグナル生成"""
        if idx < max(self.params["sma_period"], self.params["zscore_period"]):
            return 0
            
        # 統計的異常値チェック
        if not self._is_statistical_anomaly(idx):
            return 0
            
        # ボリューム確認
        if not self._volume_confirmation_check(idx):
            return 0
            
        # RSIフィルター（オプション）
        if self.params["rsi_filter"] and 'RSI' in self.data.columns:
            rsi = self.data['RSI'].iloc[idx]
            if pd.notna(rsi) and rsi > self.params["rsi_oversold"]:
                return 0  # RSIが過売り状態でない場合はエントリーしない
                
        return 1  # ロングエントリー
        
    def generate_exit_signal(self, idx: int, position_size: float = 0) -> int:
        """エグジットシグナル生成"""
        if position_size <= 0:
            return 0
            
        current_price = self.data[self.price_column].iloc[idx]
        
        # エントリー価格とエントリー日を特定
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
            
        # ATRベースのストップロス（オプション）
        if self.params["atr_filter"] and 'ATR' in self.data.columns:
            atr = self.data['ATR'].iloc[idx]
            if pd.notna(atr):
                atr_stop_loss = self.params["atr_multiplier"] * atr / entry_price
                if pnl_pct <= -atr_stop_loss:
                    return 1
                    
        # 平均回帰完了チェック（Z-scoreベース）
        if 'Z_Score' in self.data.columns:
            z_score = self.data['Z_Score'].iloc[idx]
            if pd.notna(z_score) and z_score >= self.params["zscore_exit_threshold"]:
                if pnl_pct > 0:  # 利益が出ている場合のみ
                    return 1
                    
        # 移動平均回帰チェック
        if 'SMA' in self.data.columns:
            sma = self.data['SMA'].iloc[idx]
            if pd.notna(sma) and current_price >= sma * 0.995:  # 移動平均の99.5%まで戻った
                if pnl_pct > 0:  # 利益が出ている場合のみ
                    return 1
                    
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
                for entry_idx in self.position_days:
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
        return "MeanReversion"
        
    def get_required_columns(self) -> List[str]:
        """必要なデータ列を返す"""
        return [self.price_column, 'High', 'Low', 'Volume']
        
    def get_parameters_info(self) -> Dict[str, Any]:
        """パラメータ情報を返す"""
        return {
            "strategy_type": "mean_reversion",
            "market_regime": "range_bound",
            "time_frame": "short_to_medium_term",
            "risk_level": "medium",
            "parameters": self.params
        }


# テスト用のユーティリティ関数
def create_test_mean_reversion_strategy(data: pd.DataFrame, 
                                       params: Optional[Dict[str, Any]] = None) -> MeanReversionStrategy:
    """テスト用戦略インスタンス作成"""
    test_params = {
        "sma_period": 15,
        "zscore_entry_threshold": -1.5,
        "zscore_exit_threshold": -0.2,
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.04,
        "volume_confirmation": True,
        "rsi_filter": True,
        "rsi_oversold": 25
    }
    
    if params:
        test_params.update(params)
        
    return MeanReversionStrategy(data, test_params)


if __name__ == "__main__":
    # 簡単なテスト
    print("Mean Reversion Strategy - Test Mode")
    
    # テストデータ生成（平均回帰パターン）
    import datetime
    dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')
    
    np.random.seed(123)
    base_price = 100
    
    # 平均回帰パターン（中心値100を軸にした振動）
    n_days = len(dates)
    mean_price = 100
    
    prices = []
    for i in range(n_days):
        # 平均回帰プロセス（Ornstein-Uhlenbeck process に近似）
        if i == 0:
            price = base_price
        else:
            prev_price = prices[-1]
            # 平均回帰力（価格が100から離れるほど強く戻る力）
            revert_force = (mean_price - prev_price) * 0.05
            
            # ランダムショック
            shock = np.random.normal(0, 1.5)
            
            # 時々大きな逸脱（平均回帰の機会）
            if i % 20 == 15:
                shock += np.random.choice([-8, 8])  # 大きな逸脱
                
            price = prev_price + revert_force + shock
            price = max(85, min(115, price))  # 範囲制限
            
        prices.append(price)
    
    # OHLCV データ作成
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
        'High': [p * np.random.uniform(1.005, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 0.995) for p in prices],
        'Adj Close': prices,
        'Volume': np.random.randint(100000, 500000, len(dates))
    })
    
    test_data['Close'] = test_data['Adj Close']
    test_data.set_index('Date', inplace=True)
    
    # 戦略テスト
    strategy = create_test_mean_reversion_strategy(test_data)
    result = strategy.backtest()
    
    # 結果表示
    entries = (result['Entry_Signal'] == 1).sum()
    exits = (result['Exit_Signal'] == 1).sum()
    
    print(f"Backtest Results:")
    print(f"  Entry Signals: {entries}")
    print(f"  Exit Signals: {exits}")
    print(f"  Price Range: {test_data['Adj Close'].min():.2f} - {test_data['Adj Close'].max():.2f}")
    print(f"  Z-Score Range: {result['Z_Score'].min():.2f} - {result['Z_Score'].max():.2f}")
    
    if entries > 0:
        print(f"  Mean Reversion Strategy is working correctly!")
        entry_dates = result[result['Entry_Signal'] == 1].index
        print(f"  Entry Dates: {[d.strftime('%m-%d') for d in entry_dates[:3]]}")
    else:
        print(f"  No signals generated - consider adjusting parameters")
        
    print("Test completed successfully!")
