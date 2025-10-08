"""
Module: Switching Performance Calculator  
File: switching_performance_calculator.py
Description: 
  5-1-1「戦略切替のタイミング分析ツール」
  切替前後のパフォーマンス計算と比較分析

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# プロジェクトパスの追加  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

@dataclass
class SwitchingPerformanceResult:
    """切替パフォーマンス結果"""
    switch_timestamp: datetime
    from_strategy: str
    to_strategy: str
    pre_switch_metrics: PerformanceMetrics
    post_switch_metrics: PerformanceMetrics
    performance_improvement: Dict[str, float]
    switching_cost: float
    opportunity_cost: float
    net_benefit: float
    success: bool
    confidence_score: float

@dataclass  
class ComparativeAnalysisResult:
    """比較分析結果"""
    analysis_period: Tuple[datetime, datetime]
    strategy_performances: Dict[str, PerformanceMetrics]
    switching_scenario_performance: PerformanceMetrics
    buy_and_hold_performance: PerformanceMetrics
    best_single_strategy_performance: PerformanceMetrics
    switching_effectiveness: float
    total_switching_costs: float
    net_switching_benefit: float
    optimal_switching_frequency: float

class PerformanceCalculationMethod(Enum):
    """パフォーマンス計算方法"""
    SIMPLE_RETURNS = "simple_returns"
    LOG_RETURNS = "log_returns" 
    COMPOUND_RETURNS = "compound_returns"

class BenchmarkType(Enum):
    """ベンチマーク種別"""
    BUY_AND_HOLD = "buy_and_hold"
    BEST_SINGLE_STRATEGY = "best_single_strategy"
    EQUAL_WEIGHT_PORTFOLIO = "equal_weight_portfolio"
    RISK_PARITY = "risk_parity"

class SwitchingPerformanceCalculator:
    """戦略切替パフォーマンス計算器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Parameters:
            config: 設定辞書
        """
        self.config = config or self._get_default_config()
        self.calculation_method = PerformanceCalculationMethod(
            self.config.get('calculation_method', 'simple_returns')
        )
        
        # 取引コスト設定
        self.transaction_cost_bps = self.config.get('transaction_cost_bps', 5.0)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        # 計算履歴
        self.calculation_history: List[SwitchingPerformanceResult] = []
        
        logger.info("SwitchingPerformanceCalculator initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'calculation_method': 'simple_returns',
            'transaction_cost_bps': 5.0,
            'risk_free_rate': 0.02,
            'performance_lookback': 30,
            'annualization_factor': 252,
            'min_periods_for_metrics': 10
        }

    def calculate_switching_performance(
        self,
        data: pd.DataFrame,
        switch_timestamp: datetime,
        from_strategy: str,
        to_strategy: str,
        lookback_periods: int = 30,
        lookahead_periods: int = 30
    ) -> SwitchingPerformanceResult:
        """
        特定の切替に対するパフォーマンス計算
        
        Parameters:
            data: 価格・リターンデータ
            switch_timestamp: 切替時刻
            from_strategy: 切替前戦略
            to_strategy: 切替後戦略  
            lookback_periods: 切替前評価期間
            lookahead_periods: 切替後評価期間
            
        Returns:
            切替パフォーマンス結果
        """
        try:
            # データの準備
            processed_data = self._prepare_performance_data(data, switch_timestamp)
            
            # 切替前パフォーマンスの計算
            pre_switch_metrics = self._calculate_strategy_performance(
                processed_data, from_strategy, switch_timestamp, 
                lookback_periods, is_pre_switch=True
            )
            
            # 切替後パフォーマンスの計算
            post_switch_metrics = self._calculate_strategy_performance(
                processed_data, to_strategy, switch_timestamp,
                lookahead_periods, is_pre_switch=False
            )
            
            # パフォーマンス改善の計算
            improvement = self._calculate_performance_improvement(
                pre_switch_metrics, post_switch_metrics
            )
            
            # 切替コストの計算
            switching_cost = self._calculate_switching_cost(
                from_strategy, to_strategy, processed_data, switch_timestamp
            )
            
            # 機会損失の計算
            opportunity_cost = self._calculate_opportunity_cost(
                processed_data, from_strategy, to_strategy, 
                switch_timestamp, lookahead_periods
            )
            
            # 純利益の計算
            net_benefit = improvement.get('total_return', 0.0) - switching_cost - opportunity_cost
            
            # 成功判定
            success = net_benefit > 0 and improvement.get('sharpe_ratio', 0.0) > 0
            
            # 信頼度スコアの計算
            confidence_score = self._calculate_confidence_score(
                pre_switch_metrics, post_switch_metrics, improvement
            )
            
            # 結果の構築
            result = SwitchingPerformanceResult(
                switch_timestamp=switch_timestamp,
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                pre_switch_metrics=pre_switch_metrics,
                post_switch_metrics=post_switch_metrics,
                performance_improvement=improvement,
                switching_cost=switching_cost,
                opportunity_cost=opportunity_cost,
                net_benefit=net_benefit,
                success=success,
                confidence_score=confidence_score
            )
            
            # 履歴に追加
            self.calculation_history.append(result)
            
            logger.debug(f"Switching performance calculated: net_benefit={net_benefit:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Switching performance calculation failed: {e}")
            raise

    def _prepare_performance_data(
        self, 
        data: pd.DataFrame, 
        switch_timestamp: datetime
    ) -> pd.DataFrame:
        """パフォーマンス計算用データの準備"""
        if data.empty:
            return data
            
        processed_data = data.copy()
        
        # リターン計算（存在しない場合）
        if 'close' in processed_data.columns and 'returns' not in processed_data.columns:
            if self.calculation_method == PerformanceCalculationMethod.LOG_RETURNS:
                processed_data['returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1))
            else:
                processed_data['returns'] = processed_data['close'].pct_change()
                
        # 戦略固有のリターン計算（簡易版）
        if 'returns' in processed_data.columns:
            processed_data = self._add_strategy_returns(processed_data)
            
        # 欠損値処理
        processed_data = processed_data.fillna(method='ffill').fillna(0)
        
        return processed_data

    def _add_strategy_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """戦略固有リターンの追加"""
        # 基本リターン
        base_returns = data['returns']
        
        # 戦略別リターンの簡易シミュレーション
        strategies = ['momentum', 'mean_reversion', 'vwap', 'breakout']
        
        for strategy in strategies:
            if strategy == 'momentum':
                # モメンタム戦略: トレンドフォロー
                momentum_signal = base_returns.rolling(5).mean()
                data[f'{strategy}_returns'] = base_returns * np.where(momentum_signal > 0, 1.2, 0.8)
                
            elif strategy == 'mean_reversion':
                # 平均回帰戦略: 逆張り
                z_score = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
                data[f'{strategy}_returns'] = base_returns * np.where(abs(z_score) > 1, 1.3, 0.9)
                
            elif strategy == 'vwap':
                # VWAP戦略: ボリューム加重
                volume_factor = data.get('volume', pd.Series(1, index=data.index))
                volume_weight = volume_factor / volume_factor.rolling(10).mean()
                data[f'{strategy}_returns'] = base_returns * np.clip(volume_weight, 0.8, 1.2)
                
            elif strategy == 'breakout':
                # ブレイクアウト戦略: ボラティリティブレイク
                volatility = base_returns.rolling(20).std()
                breakout_signal = abs(base_returns) / volatility.shift(1)
                data[f'{strategy}_returns'] = base_returns * np.where(breakout_signal > 2, 1.4, 0.7)
                
        return data

    def _calculate_strategy_performance(
        self,
        data: pd.DataFrame,
        strategy: str,
        reference_timestamp: datetime,
        periods: int,
        is_pre_switch: bool = True
    ) -> PerformanceMetrics:
        """戦略パフォーマンスの計算"""
        try:
            # 参照時点のインデックスを取得
            ref_index = data.index.get_loc(reference_timestamp) if reference_timestamp in data.index else -1
            
            if ref_index == -1:
                logger.warning(f"Reference timestamp not found: {reference_timestamp}")
                return PerformanceMetrics()
                
            # 期間の設定
            if is_pre_switch:
                start_idx = max(0, ref_index - periods)
                end_idx = ref_index
            else:
                start_idx = ref_index
                end_idx = min(len(data), ref_index + periods)
                
            if start_idx >= end_idx:
                return PerformanceMetrics()
                
            # 戦略リターンの取得
            strategy_returns_col = f'{strategy}_returns'
            if strategy_returns_col in data.columns:
                returns = data[strategy_returns_col].iloc[start_idx:end_idx]
            else:
                # フォールバック: 基本リターンを使用
                returns = data['returns'].iloc[start_idx:end_idx] if 'returns' in data.columns else pd.Series()
                
            if returns.empty or len(returns) < self.config.get('min_periods_for_metrics', 10):
                return PerformanceMetrics()
                
            # パフォーマンス指標の計算
            metrics = PerformanceMetrics()
            
            # 基本リターン指標
            metrics.total_return = (1 + returns).prod() - 1
            metrics.annualized_return = ((1 + metrics.total_return) ** (self.config.get('annualization_factor', 252) / len(returns))) - 1
            metrics.volatility = returns.std() * np.sqrt(self.config.get('annualization_factor', 252))
            
            # シャープレシオ
            if metrics.volatility > 0:
                excess_return = metrics.annualized_return - self.risk_free_rate
                metrics.sharpe_ratio = excess_return / metrics.volatility
            else:
                metrics.sharpe_ratio = 0.0
                
            # ソルティーノレシオ
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(self.config.get('annualization_factor', 252))
                if downside_vol > 0:
                    metrics.sortino_ratio = (metrics.annualized_return - self.risk_free_rate) / downside_vol
                    
            # ドローダウン計算
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics.max_drawdown = abs(drawdown.min())
            
            # カルマーレシオ
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
                
            # 勝率と利益ファクター
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            total_trades = len(returns[returns != 0])
            if total_trades > 0:
                metrics.win_rate = len(winning_returns) / total_trades
                
            if len(winning_returns) > 0:
                metrics.average_win = winning_returns.mean()
                metrics.largest_win = winning_returns.max()
                
            if len(losing_returns) > 0:
                metrics.average_loss = abs(losing_returns.mean())
                metrics.largest_loss = abs(losing_returns.min())
                
                # プロフィットファクター
                total_gains = winning_returns.sum()
                total_losses = abs(losing_returns.sum())
                if total_losses > 0:
                    metrics.profit_factor = total_gains / total_losses
                    
            # 連続勝敗の計算
            metrics.consecutive_wins, metrics.consecutive_losses = self._calculate_consecutive_streaks(returns)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance calculation failed for strategy {strategy}: {e}")
            return PerformanceMetrics()

    def _calculate_consecutive_streaks(self, returns: pd.Series) -> Tuple[int, int]:
        """連続勝敗の計算"""
        if returns.empty:
            return 0, 0
            
        # 勝敗判定
        wins = returns > 0
        losses = returns < 0
        
        # 連続勝ちの最大値
        max_consecutive_wins = 0
        current_win_streak = 0
        
        for win in wins:
            if win:
                current_win_streak += 1
                max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
            else:
                current_win_streak = 0
                
        # 連続負けの最大値  
        max_consecutive_losses = 0
        current_loss_streak = 0
        
        for loss in losses:
            if loss:
                current_loss_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            else:
                current_loss_streak = 0
                
        return max_consecutive_wins, max_consecutive_losses

    def _calculate_performance_improvement(
        self,
        pre_metrics: PerformanceMetrics,
        post_metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """パフォーマンス改善の計算"""
        improvement = {}
        
        # 各指標の改善度を計算
        metrics_to_compare = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'win_rate', 'profit_factor'
        ]
        
        for metric in metrics_to_compare:
            pre_value = getattr(pre_metrics, metric, 0.0)
            post_value = getattr(post_metrics, metric, 0.0)
            
            if pre_value != 0:
                improvement[metric] = (post_value - pre_value) / abs(pre_value)
            else:
                improvement[metric] = post_value
                
        # ドローダウンは改善が小さい方が良い
        pre_dd = pre_metrics.max_drawdown
        post_dd = post_metrics.max_drawdown
        if pre_dd > 0:
            improvement['max_drawdown'] = (pre_dd - post_dd) / pre_dd
        else:
            improvement['max_drawdown'] = -post_dd
            
        # ボラティリティも改善が小さい方が良い
        pre_vol = pre_metrics.volatility  
        post_vol = post_metrics.volatility
        if pre_vol > 0:
            improvement['volatility'] = (pre_vol - post_vol) / pre_vol
        else:
            improvement['volatility'] = -post_vol
            
        return improvement

    def _calculate_switching_cost(
        self,
        from_strategy: str,
        to_strategy: str,
        data: pd.DataFrame,
        timestamp: datetime
    ) -> float:
        """切替コストの計算"""
        # 基本取引コスト
        base_cost = self.transaction_cost_bps / 10000
        
        # 戦略間の切替コスト（戦略の差異に基づく）
        strategy_distance_multiplier = self._get_strategy_distance_multiplier(from_strategy, to_strategy)
        
        # 市場条件による調整
        market_impact = self._estimate_market_impact_cost(data, timestamp)
        
        total_cost = base_cost * strategy_distance_multiplier * (1 + market_impact)
        
        return total_cost

    def _get_strategy_distance_multiplier(self, from_strategy: str, to_strategy: str) -> float:
        """戦略間距離に基づくコスト倍率"""
        # 戦略間の「距離」を定義（類似度の逆数）
        distance_matrix = {
            ('momentum', 'breakout'): 1.2,
            ('momentum', 'trend_following'): 1.1,
            ('momentum', 'mean_reversion'): 2.5,
            ('momentum', 'vwap'): 2.0,
            ('mean_reversion', 'vwap'): 1.5,
            ('mean_reversion', 'breakout'): 2.8,
            ('vwap', 'breakout'): 2.2,
            ('breakout', 'trend_following'): 1.3
        }
        
        key1 = (from_strategy, to_strategy)
        key2 = (to_strategy, from_strategy)
        
        return distance_matrix.get(key1, distance_matrix.get(key2, 2.0))

    def _estimate_market_impact_cost(self, data: pd.DataFrame, timestamp: datetime) -> float:
        """マーケットインパクトコストの推定"""
        try:
            if timestamp not in data.index or 'returns' not in data.columns:
                return 0.1  # デフォルトインパクト
                
            # ボラティリティベースのマーケットインパクト
            recent_volatility = data['returns'].loc[:timestamp].tail(10).std()
            
            # ボリュームベースの調整（利用可能な場合）
            volume_factor = 1.0
            if 'volume' in data.columns:
                recent_volume = data['volume'].loc[:timestamp].tail(10).mean()
                avg_volume = data['volume'].mean()
                if avg_volume > 0:
                    volume_factor = max(0.5, min(2.0, avg_volume / recent_volume))
                    
            market_impact = recent_volatility * volume_factor * 0.5
            return min(0.5, market_impact)  # 最大50%の追加コスト
            
        except Exception as e:
            logger.warning(f"Market impact cost estimation failed: {e}")
            return 0.1

    def _calculate_opportunity_cost(
        self,
        data: pd.DataFrame,
        from_strategy: str,
        to_strategy: str,
        timestamp: datetime,
        periods: int
    ) -> float:
        """機会損失の計算"""
        try:
            ref_index = data.index.get_loc(timestamp) if timestamp in data.index else -1
            if ref_index == -1 or ref_index + periods >= len(data):
                return 0.0
                
            # 切替せずに元戦略を続けた場合のパフォーマンス
            from_strategy_col = f'{from_strategy}_returns'
            if from_strategy_col in data.columns:
                continuing_returns = data[from_strategy_col].iloc[ref_index:ref_index+periods]
                continuing_performance = (1 + continuing_returns).prod() - 1
            else:
                continuing_performance = 0.0
                
            # 切替後戦略のパフォーマンス
            to_strategy_col = f'{to_strategy}_returns'
            if to_strategy_col in data.columns:
                switching_returns = data[to_strategy_col].iloc[ref_index:ref_index+periods]
                switching_performance = (1 + switching_returns).prod() - 1
            else:
                switching_performance = 0.0
                
            # 機会損失（負の場合は機会利益）
            opportunity_cost = max(0, continuing_performance - switching_performance)
            
            return opportunity_cost
            
        except Exception as e:
            logger.warning(f"Opportunity cost calculation failed: {e}")
            return 0.0

    def _calculate_confidence_score(
        self,
        pre_metrics: PerformanceMetrics,
        post_metrics: PerformanceMetrics,
        improvement: Dict[str, float]
    ) -> float:
        """信頼度スコアの計算"""
        # 改善指標の重み付け平均
        key_improvements = ['total_return', 'sharpe_ratio', 'max_drawdown']
        weights = [0.4, 0.4, 0.2]
        
        weighted_improvement = 0.0
        for metric, weight in zip(key_improvements, weights):
            improvement_value = improvement.get(metric, 0.0)
            # 改善値を0-1の範囲にマッピング
            normalized_improvement = max(0, min(1, (improvement_value + 1) / 2))
            weighted_improvement += normalized_improvement * weight
            
        # 統計的有意性の考慮（簡易版）
        statistical_significance = 0.5  # 中立値
        
        if hasattr(pre_metrics, 'total_return') and hasattr(post_metrics, 'total_return'):
            # リターンの差の相対的な大きさ
            return_diff = abs(post_metrics.total_return - pre_metrics.total_return)
            avg_return = (abs(pre_metrics.total_return) + abs(post_metrics.total_return)) / 2
            if avg_return > 0:
                statistical_significance = min(1.0, return_diff / avg_return)
                
        # 最終信頼度スコア
        confidence = (weighted_improvement * 0.7 + statistical_significance * 0.3)
        return max(0.1, min(1.0, confidence))

    def calculate_comparative_analysis(
        self,
        data: pd.DataFrame,
        switching_timestamps: List[datetime],
        strategy_sequence: List[str],
        benchmark_strategies: Optional[List[str]] = None
    ) -> ComparativeAnalysisResult:
        """
        複数戦略とのパフォーマンス比較分析
        
        Parameters:
            data: 価格データ
            switching_timestamps: 切替時刻のリスト
            strategy_sequence: 戦略の順序
            benchmark_strategies: ベンチマーク戦略リスト
            
        Returns:
            比較分析結果
        """
        try:
            if benchmark_strategies is None:
                benchmark_strategies = ['momentum', 'mean_reversion', 'vwap', 'breakout']
                
            analysis_start = data.index[0] if not data.empty else datetime.now()
            analysis_end = data.index[-1] if not data.empty else datetime.now()
            
            # 各戦略の個別パフォーマンス計算
            strategy_performances = {}
            for strategy in benchmark_strategies:
                performance = self._calculate_strategy_performance(
                    data, strategy, analysis_end, len(data), is_pre_switch=True
                )
                strategy_performances[strategy] = performance
                
            # 切替戦略のパフォーマンス計算
            switching_performance = self._calculate_switching_scenario_performance(
                data, switching_timestamps, strategy_sequence
            )
            
            # Buy & Hold パフォーマンス
            buy_hold_performance = self._calculate_buy_hold_performance(data)
            
            # 最高単一戦略のパフォーマンス
            best_single_strategy = max(
                strategy_performances.keys(),
                key=lambda s: strategy_performances[s].total_return
            )
            best_single_performance = strategy_performances[best_single_strategy]
            
            # 切替効果の評価
            switching_effectiveness = self._evaluate_switching_effectiveness(
                switching_performance, best_single_performance, buy_hold_performance
            )
            
            # 切替コストの合計
            total_switching_costs = self._calculate_total_switching_costs(
                data, switching_timestamps, strategy_sequence
            )
            
            # 純利益
            net_benefit = switching_effectiveness - total_switching_costs
            
            # 最適切替頻度の推定
            optimal_frequency = self._estimate_optimal_switching_frequency(
                data, switching_timestamps, switching_effectiveness
            )
            
            result = ComparativeAnalysisResult(
                analysis_period=(analysis_start, analysis_end),
                strategy_performances=strategy_performances,
                switching_scenario_performance=switching_performance,
                buy_and_hold_performance=buy_hold_performance,
                best_single_strategy_performance=best_single_performance,
                switching_effectiveness=switching_effectiveness,
                total_switching_costs=total_switching_costs,
                net_switching_benefit=net_benefit,
                optimal_switching_frequency=optimal_frequency
            )
            
            logger.info(f"Comparative analysis completed: net_benefit={net_benefit:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise

    def _calculate_switching_scenario_performance(
        self,
        data: pd.DataFrame,
        switching_timestamps: List[datetime],
        strategy_sequence: List[str]
    ) -> PerformanceMetrics:
        """切替シナリオのパフォーマンス計算"""
        if not switching_timestamps or not strategy_sequence:
            return PerformanceMetrics()
            
        try:
            # 期間別のリターンを計算
            total_returns = []
            
            # 最初の期間
            first_switch = switching_timestamps[0]
            start_data = data[data.index < first_switch]
            if not start_data.empty and len(strategy_sequence) > 0:
                initial_strategy = strategy_sequence[0]
                initial_returns = self._get_strategy_returns(start_data, initial_strategy)
                total_returns.extend(initial_returns.tolist())
                
            # 各切替期間
            for i, (timestamp, strategy) in enumerate(zip(switching_timestamps, strategy_sequence[1:])):
                # 前回の切替から今回の切替まで
                start_time = switching_timestamps[i-1] if i > 0 else data.index[0]
                period_data = data[(data.index >= start_time) & (data.index < timestamp)]
                
                if not period_data.empty:
                    period_returns = self._get_strategy_returns(period_data, strategy)
                    total_returns.extend(period_returns.tolist())
                    
            # 最後の期間
            if switching_timestamps:
                last_switch = switching_timestamps[-1]
                last_strategy = strategy_sequence[-1] if strategy_sequence else strategy_sequence[0]
                end_data = data[data.index >= last_switch]
                
                if not end_data.empty:
                    final_returns = self._get_strategy_returns(end_data, last_strategy)
                    total_returns.extend(final_returns.tolist())
                    
            # パフォーマンス指標の計算
            if total_returns:
                returns_series = pd.Series(total_returns)
                return self._calculate_metrics_from_returns(returns_series)
            else:
                return PerformanceMetrics()
                
        except Exception as e:
            logger.warning(f"Switching scenario performance calculation failed: {e}")
            return PerformanceMetrics()

    def _get_strategy_returns(self, data: pd.DataFrame, strategy: str) -> pd.Series:
        """戦略リターンの取得"""
        strategy_col = f'{strategy}_returns'
        if strategy_col in data.columns:
            return data[strategy_col].dropna()
        elif 'returns' in data.columns:
            return data['returns'].dropna()
        else:
            return pd.Series()

    def _calculate_metrics_from_returns(self, returns: pd.Series) -> PerformanceMetrics:
        """リターンからパフォーマンス指標を計算"""
        if returns.empty:
            return PerformanceMetrics()
            
        metrics = PerformanceMetrics()
        
        # 基本指標
        metrics.total_return = (1 + returns).prod() - 1
        if len(returns) > 0:
            metrics.annualized_return = ((1 + metrics.total_return) ** (252 / len(returns))) - 1
        metrics.volatility = returns.std() * np.sqrt(252)
        
        # シャープレシオ
        if metrics.volatility > 0:
            metrics.sharpe_ratio = (metrics.annualized_return - self.risk_free_rate) / metrics.volatility
            
        # ドローダウン
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = abs(drawdown.min())
        
        # その他の指標
        winning_returns = returns[returns > 0]
        metrics.win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        
        return metrics

    def _calculate_buy_hold_performance(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Buy & Hold パフォーマンスの計算"""
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
            return self._calculate_metrics_from_returns(returns)
        else:
            return PerformanceMetrics()

    def _evaluate_switching_effectiveness(
        self,
        switching_perf: PerformanceMetrics,
        best_single_perf: PerformanceMetrics,
        buy_hold_perf: PerformanceMetrics
    ) -> float:
        """切替効果の評価"""
        # ベースライン（最高単一戦略とBuy & Holdの平均）
        baseline_return = (best_single_perf.total_return + buy_hold_perf.total_return) / 2
        
        # 切替による超過リターン
        excess_return = switching_perf.total_return - baseline_return
        
        return excess_return

    def _calculate_total_switching_costs(
        self,
        data: pd.DataFrame,
        switching_timestamps: List[datetime],
        strategy_sequence: List[str]
    ) -> float:
        """総切替コストの計算"""
        total_cost = 0.0
        
        for i, timestamp in enumerate(switching_timestamps):
            if i < len(strategy_sequence) - 1:
                from_strategy = strategy_sequence[i]
                to_strategy = strategy_sequence[i + 1]
                cost = self._calculate_switching_cost(from_strategy, to_strategy, data, timestamp)
                total_cost += cost
                
        return total_cost

    def _estimate_optimal_switching_frequency(
        self,
        data: pd.DataFrame,
        switching_timestamps: List[datetime],
        effectiveness: float
    ) -> float:
        """最適切替頻度の推定"""
        if not switching_timestamps or data.empty:
            return 0.0
            
        # 実際の切替頻度（年間）
        analysis_days = (data.index[-1] - data.index[0]).days
        actual_frequency = len(switching_timestamps) / (analysis_days / 365.25)
        
        # 効果性に基づく調整
        if effectiveness > 0.05:  # 5%以上の改善
            optimal_frequency = actual_frequency * 1.2  # 20%増加
        elif effectiveness > 0.02:  # 2%以上の改善
            optimal_frequency = actual_frequency  # 現状維持
        else:  # 改善が小さい
            optimal_frequency = actual_frequency * 0.5  # 半減
            
        return max(0.0, optimal_frequency)

    def export_performance_summary(
        self,
        results: List[SwitchingPerformanceResult],
        file_path: str
    ):
        """パフォーマンスサマリーのエクスポート"""
        try:
            if not results:
                logger.warning("No results to export")
                return
                
            # サマリーデータの作成
            summary_data = []
            
            for result in results:
                summary_data.append({
                    'switch_timestamp': result.switch_timestamp.isoformat(),
                    'from_strategy': result.from_strategy,
                    'to_strategy': result.to_strategy,
                    'pre_total_return': result.pre_switch_metrics.total_return,
                    'post_total_return': result.post_switch_metrics.total_return,
                    'pre_sharpe_ratio': result.pre_switch_metrics.sharpe_ratio,
                    'post_sharpe_ratio': result.post_switch_metrics.sharpe_ratio,
                    'switching_cost': result.switching_cost,
                    'opportunity_cost': result.opportunity_cost,
                    'net_benefit': result.net_benefit,
                    'success': result.success,
                    'confidence_score': result.confidence_score
                })
                
            # DataFrame作成とエクスポート
            df = pd.DataFrame(summary_data)
            
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_json(file_path, orient='records', indent=2)
                
            logger.info(f"Performance summary exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    calculator = SwitchingPerformanceCalculator()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    test_data['returns'] = test_data['close'].pct_change()
    
    try:
        # 切替パフォーマンスの計算
        switch_timestamp = test_data.index[150]  # 年央での切替
        
        result = calculator.calculate_switching_performance(
            test_data,
            switch_timestamp=switch_timestamp,
            from_strategy='momentum',
            to_strategy='mean_reversion',
            lookback_periods=30,
            lookahead_periods=30
        )
        
        print("\n=== 戦略切替パフォーマンス計算結果 ===")
        print(f"切替時刻: {result.switch_timestamp.strftime('%Y-%m-%d')}")
        print(f"切替戦略: {result.from_strategy} → {result.to_strategy}")
        print(f"切替前リターン: {result.pre_switch_metrics.total_return:.2%}")
        print(f"切替後リターン: {result.post_switch_metrics.total_return:.2%}")
        print(f"切替前シャープ: {result.pre_switch_metrics.sharpe_ratio:.3f}")
        print(f"切替後シャープ: {result.post_switch_metrics.sharpe_ratio:.3f}")
        print(f"切替コスト: {result.switching_cost:.4f}")
        print(f"機会損失: {result.opportunity_cost:.4f}")
        print(f"純利益: {result.net_benefit:.4f}")
        print(f"成功: {'Yes' if result.success else 'No'}")
        print(f"信頼度: {result.confidence_score:.1%}")
        
        print("計算成功")
        
    except Exception as e:
        print(f"計算エラー: {e}")
        raise
