"""
Module: contrarian_strategy
File: contrarian_strategy.py
Description: 
  過度な売られ場面で反発を狙う逆張り戦略を実装しています。
  RSIの過売り状態やギャップダウン、ピンバー形成などの反転サインを検出し、
  レンジ相場でこれらの条件が揃った際にエントリーします。短期の利食いと
  適切な損切り設定で勝率とリスクリワード比の向上を図ります。

Author: kouseven7
Created: 2023-04-10
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.trend_analysis
  - config.optimized_parameters
  - validation.validators.contrarian_validator
"""

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_rsi
from indicators.trend_analysis import detect_trend
from config.optimized_parameters import OptimizedParameterManager
from validation.validators.contrarian_validator import ContrarianParameterValidator
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend, detect_unified_trend_with_confidence

class ContrarianStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        逆張り戦略の初期化。
        """
        self.price_column = price_column
        self.entry_prices = {}  # エントリー価格を記録
        self.high_prices = {}   # トレーリングストップ用の最高値を記録

        # デフォルトパラメータ
        default_params = {
            "rsi_period": 14,        # RSI計算期間
            "rsi_oversold": 30,      # RSI過売り閾値
            "gap_threshold": 0.02,   # ギャップダウン閾値（A: 5%→2%に緩和）
            "stop_loss": 0.04,       # ストップロス
            "take_profit": 0.05,     # 利益確定
            "pin_bar_ratio": 2.0,    # ピンバー判定比率
            "max_hold_days": 5,      # 最大保有日数
            "rsi_exit_level": 50,    # RSI中立域でのイグジット
            "trailing_stop_pct": 0.02,  # トレーリングストップ率
            
            # トレンドフィルター設定（Option 4: レンジ相場のみに戻す）
            "trend_filter_enabled": True,  # トレンドフィルター有効化
            "allowed_trends": ["range-bound"]  # レンジ相場のみ許可
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        # RSIを計算してデータに追加
        # ルックアヘッドバイアス修正: shift(1)を追加して前日のRSIを使用
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], period=self.params["rsi_period"]).shift(1)
        
        # Openカラムの確認（ピンバー判定に必要）
        if 'Open' not in self.data.columns:
            raise ValueError("ピンバー判定にはOpenカラムが必要です")
        
        # 統一トレンド判定の初期結果を表示（データがある場合）
        if len(self.data) > 20:  # データが十分ある場合のみ
            try:
                # 現在のトレンドを判定
                trend = detect_unified_trend(
                    self.data,
                    price_column=self.price_column,
                    strategy="contrarian_strategy",
                    method="combined"
                )
                print(f"初期トレンド判定: {trend} (contrarian_strategy)")
                
                # 信頼度付き判定
                trend_detector = UnifiedTrendDetector(
                    self.data,
                    price_column=self.price_column,
                    strategy_name="contrarian_strategy",
                    method="combined"
                )
                _, confidence = trend_detector.detect_trend_with_confidence()
                print(f"トレンド判定信頼度: {confidence:.2f}")
            except Exception as e:
                print(f"トレンド判定初期化エラー: {e}")

    def _is_valid_pinbar(self, high: float, low: float, close: float, open_price: float) -> bool:
        """
        改善版ピンバー判定（案B - 緩和版）
        
        条件:
        1. 下ヒゲの最小値チェック（1.0円以上）← 2.0円から緩和
        2. 上ヒゲが下ヒゲの2倍以上
        3. 上ヒゲが全体レンジの50%以上
        4. 実体が小さい（全体レンジの30%以下）
        
        Args:
            high: 高値
            low: 安値
            close: 終値
            open_price: 始値
            
        Returns:
            bool: ピンバーと判定された場合True
        """
        upper_shadow = high - max(close, open_price)
        lower_shadow = min(close, open_price) - low
        body = abs(close - open_price)
        total_range = high - low
        
        # 条件1: 下ヒゲの最小値チェック（緩和: 2.0円 → 1.0円）
        if lower_shadow < 1.0:
            return False
        
        # 条件2: 上ヒゲが下ヒゲの2倍以上
        if upper_shadow <= self.params["pin_bar_ratio"] * lower_shadow:
            return False
        
        # 条件3: 上ヒゲが全体レンジの50%以上
        if total_range > 0 and upper_shadow < 0.5 * total_range:
            return False
        
        # 条件4: 実体が小さい（全体レンジの30%以下）
        if total_range > 0 and body > 0.3 * total_range:
            return False
        
        return True

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        """
        if idx < 5:  # 過去データが不足している場合
            return 0

        rsi = self.data['RSI'].iloc[idx]
        current_price = self.data[self.price_column].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]

        # ギャップダウンの判定
        gap_down = current_price < previous_close * (1.0 - self.params["gap_threshold"])

        # ピンバーの判定（改善版）
        if 'High' in self.data.columns and 'Low' in self.data.columns and 'Open' in self.data.columns:
            high = self.data['High'].iloc[idx]
            low = self.data['Low'].iloc[idx]
            open_price = self.data['Open'].iloc[idx]
            pin_bar = self._is_valid_pinbar(high, low, current_price, open_price)
        else:
            pin_bar = False

        # トレンド判定（統一トレンド判定インターフェースを使用）
        if self.params["trend_filter_enabled"]:
            # 統一トレンド判定インターフェースを使用
            trend = detect_unified_trend(
                self.data.iloc[:idx + 1], 
                price_column=self.price_column,
                strategy="contrarian_strategy",
                method="combined"  # 複合メソッドを使用
            )
            # 許可されたトレンド内にあるか確認
            if trend not in self.params["allowed_trends"]:
                return 0
        # trend_filter_enabled=Falseの場合、トレンドチェックをスキップ

        # エントリー条件（B: RSI条件を両方に適用）
        # 条件1: RSI過売り + ギャップダウン
        if rsi <= self.params["rsi_oversold"] and gap_down:
            self.entry_prices[idx] = current_price
            return 1
        # 条件2: RSI過売り + ピンバー（RSI条件追加）
        if rsi <= self.params["rsi_oversold"] and pin_bar:
            self.entry_prices[idx] = current_price
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        """
        if idx < 1:
            return 0

        # ポジション状態管理を追加
        # 現在までのエントリー・エグジット数を計算
        current_entries = (self.data['Entry_Signal'].iloc[:idx+1] == 1).sum()
        current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())
        
        # アクティブなポジションがない場合はエグジット不可
        if current_entries <= current_exits:
            return 0

        # 最新のエントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0

        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        entry_price = self.entry_prices.get(latest_entry_idx)
        if entry_price is None:
            return 0

        current_price = self.data[self.price_column].iloc[idx]

        # RSIによるイグジット
        current_rsi = self.data['RSI'].iloc[idx]
        if current_rsi >= self.params["rsi_exit_level"]:
            return -1

        # トレーリングストップ
        if latest_entry_idx not in self.high_prices:
            self.high_prices[latest_entry_idx] = entry_price
        self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], current_price)
        trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
        if current_price <= trailing_stop_price:
            return -1

        # 利益確定
        if current_price >= entry_price * (1.0 + self.params["take_profit"]):
            return -1

        # ストップロス
        if current_price <= entry_price * (1.0 - self.params["stop_loss"]):
            return -1

        # 最大保有日数
        days_held = idx - latest_entry_idx
        if days_held >= self.params["max_hold_days"]:
            return -1

        return 0

    def backtest(self):
        """
        バックテストを実行する。
        """
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        for idx in range(len(self.data)):
            # エントリーシグナル
            if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1

            # イグジットシグナル
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

        return self.data

    def load_optimized_parameters(self, ticker: str = None):
        """
        承認済みの最適化パラメータを自動適用
        """
        manager = OptimizedParameterManager()
        params = manager.load_approved_params("ContrarianStrategy", ticker)
        if params:
            self.params.update(params)
        return self.params

    def run_optimized_strategy(self, ticker: str = None):
        """
        最適化パラメータを自動適用してバックテスト実行
        """
        self.load_optimized_parameters(ticker)
        return self.backtest()

    def get_optimization_info(self, ticker: str = None):
        """
        最適化パラメータのメタ情報を取得
        """
        manager = OptimizedParameterManager()
        configs = manager.list_available_configs(strategy_name="ContrarianStrategy", ticker=ticker)
        return configs

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100
    }, index=dates)

    strategy = ContrarianStrategy(df, price_column='Adj Close')
    result = strategy.backtest()
    print(result[['Adj Close', 'RSI', 'Entry_Signal', 'Exit_Signal']].tail())