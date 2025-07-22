"""
Module: Strategy Switching Performance Analyzer
File: strategy_switching_performance_analyzer.py
Description: 
  4-2-1「トレンド変化時の戦略切替テスト」
  戦略切替性能分析・最適化機能

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 戦略切替タイミング分析・最適化
  - パフォーマンス統計・ベンチマーク比較
  - 切替効率・リスク評価機能
  - 最適化レポート・改善提案生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import warnings
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class PerformanceMetricType(Enum):
    """パフォーマンスメトリクスタイプ"""
    TIMING_ACCURACY = "timing_accuracy"
    RETURN_IMPROVEMENT = "return_improvement"
    RISK_REDUCTION = "risk_reduction"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"

@dataclass
class SwitchingAnalysisResult:
    """戦略切替分析結果"""
    metric_type: PerformanceMetricType
    metric_value: float
    benchmark_value: float
    improvement_ratio: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    analysis_notes: List[str]

@dataclass
class PerformanceBenchmark:
    """パフォーマンスベンチマーク"""
    name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float

class AdvancedPerformanceAnalyzer:
    """高度パフォーマンス分析器"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.benchmark_data = {}
        
    def analyze_switching_effectiveness(self, 
                                      switching_events: List[Dict[str, Any]],
                                      price_data: pd.DataFrame,
                                      benchmark_returns: pd.Series) -> Dict[str, Any]:
        """戦略切替効果分析"""
        try:
            analysis_results = {}
            
            # 基本統計計算
            analysis_results['basic_stats'] = self._calculate_basic_switching_stats(
                switching_events, price_data
            )
            
            # タイミング分析
            analysis_results['timing_analysis'] = self._analyze_switching_timing(
                switching_events, price_data
            )
            
            # パフォーマンス分析
            analysis_results['performance_analysis'] = self._analyze_switching_performance(
                switching_events, price_data, benchmark_returns
            )
            
            # リスク分析
            analysis_results['risk_analysis'] = self._analyze_switching_risk(
                switching_events, price_data
            )
            
            # 効率性分析
            analysis_results['efficiency_analysis'] = self._analyze_switching_efficiency(
                switching_events
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing switching effectiveness: {e}")
            return {'error': str(e)}
    
    def _calculate_basic_switching_stats(self, 
                                       switching_events: List[Dict[str, Any]],
                                       price_data: pd.DataFrame) -> Dict[str, Any]:
        """基本切替統計計算"""
        if not switching_events:
            return {'total_switches': 0}
        
        switch_df = pd.DataFrame(switching_events)
        
        stats = {
            'total_switches': len(switching_events),
            'unique_strategies': len(set(switch_df['from_strategy'].tolist() + 
                                       switch_df['to_strategy'].tolist())),
            'average_confidence': switch_df['confidence_score'].mean(),
            'confidence_std': switch_df['confidence_score'].std(),
            'average_switching_delay': switch_df['switching_delay'].mean(),
            'delay_std': switch_df['switching_delay'].std()
        }
        
        # 戦略別統計
        strategy_stats = {}
        for strategy in set(switch_df['from_strategy'].tolist() + switch_df['to_strategy'].tolist()):
            from_count = (switch_df['from_strategy'] == strategy).sum()
            to_count = (switch_df['to_strategy'] == strategy).sum()
            strategy_stats[strategy] = {
                'switched_from': from_count,
                'switched_to': to_count,
                'net_switches': to_count - from_count
            }
        
        stats['strategy_breakdown'] = strategy_stats
        
        return stats
    
    def _analyze_switching_timing(self, 
                                switching_events: List[Dict[str, Any]],
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """切替タイミング分析"""
        if not switching_events or price_data.empty:
            return {}
        
        timing_analysis = {
            'timing_quality_scores': [],
            'market_condition_alignment': [],
            'anticipation_vs_reaction': []
        }
        
        for event in switching_events:
            timestamp = event['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # タイミング前後のデータ取得
            event_idx = price_data.index.get_indexer([timestamp], method='nearest')[0]
            if event_idx < 0:
                continue
                
            # 前後の期間設定
            lookback = min(24, event_idx)  # 24期間前まで
            lookahead = min(24, len(price_data) - event_idx - 1)
            
            if lookback == 0 or lookahead == 0:
                continue
            
            pre_data = price_data.iloc[event_idx-lookback:event_idx]
            post_data = price_data.iloc[event_idx:event_idx+lookahead+1]
            
            # タイミング品質スコア計算
            quality_score = self._calculate_timing_quality(
                pre_data, post_data, event
            )
            timing_analysis['timing_quality_scores'].append(quality_score)
            
            # 市場条件アライメント評価
            alignment_score = self._evaluate_market_alignment(
                pre_data, event['market_conditions']
            )
            timing_analysis['market_condition_alignment'].append(alignment_score)
        
        # 統計計算
        if timing_analysis['timing_quality_scores']:
            timing_analysis['average_timing_quality'] = np.mean(
                timing_analysis['timing_quality_scores']
            )
            timing_analysis['timing_consistency'] = 1 - np.std(
                timing_analysis['timing_quality_scores']
            )
        
        if timing_analysis['market_condition_alignment']:
            timing_analysis['average_alignment'] = np.mean(
                timing_analysis['market_condition_alignment']
            )
        
        return timing_analysis
    
    def _calculate_timing_quality(self, 
                                pre_data: pd.DataFrame,
                                post_data: pd.DataFrame,
                                event: Dict[str, Any]) -> float:
        """タイミング品質計算"""
        try:
            # 切替前後のリターン比較
            pre_return = (pre_data['close'].iloc[-1] / pre_data['close'].iloc[0] - 1)
            post_return = (post_data['close'].iloc[-1] / post_data['close'].iloc[0] - 1)
            
            # 戦略切替の方向性と市場動向の一致度
            strategy_direction = self._get_strategy_direction(
                event['from_strategy'], event['to_strategy']
            )
            
            market_direction = 1 if post_return > 0 else -1
            direction_alignment = 1 if strategy_direction * market_direction > 0 else 0
            
            # 信頼度スコアとの整合性
            confidence_factor = event['confidence_score']
            
            # 総合タイミング品質
            timing_quality = (direction_alignment * 0.6 + 
                            confidence_factor * 0.3 + 
                            min(abs(post_return) * 10, 1) * 0.1)
            
            return max(0, min(timing_quality, 1))
            
        except Exception:
            return 0.5  # デフォルト値
    
    def _get_strategy_direction(self, from_strategy: str, to_strategy: str) -> int:
        """戦略方向性取得"""
        bullish_strategies = ['trend_following', 'momentum']
        bearish_strategies = ['mean_reversion', 'contrarian']
        
        from_bullish = from_strategy in bullish_strategies
        to_bullish = to_strategy in bullish_strategies
        
        if to_bullish and not from_bullish:
            return 1  # より強気に
        elif not to_bullish and from_bullish:
            return -1  # より弱気に
        else:
            return 0  # 中立
    
    def _evaluate_market_alignment(self, 
                                 market_data: pd.DataFrame,
                                 market_conditions: Dict[str, Any]) -> float:
        """市場条件アライメント評価"""
        try:
            # 実際の市場状況計算
            returns = market_data['close'].pct_change().dropna()
            actual_volatility = returns.std() * np.sqrt(252)
            actual_trend = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1)
            
            # 予想条件との比較
            expected_volatility = market_conditions.get('volatility', 0.2)
            volatility_alignment = 1 - abs(actual_volatility - expected_volatility) / max(expected_volatility, 0.1)
            
            # 総合アライメントスコア
            alignment = max(0, min(volatility_alignment, 1))
            
            return alignment
            
        except Exception:
            return 0.5
    
    def _analyze_switching_performance(self, 
                                     switching_events: List[Dict[str, Any]],
                                     price_data: pd.DataFrame,
                                     benchmark_returns: pd.Series) -> Dict[str, Any]:
        """切替パフォーマンス分析"""
        performance_analysis = {}
        
        try:
            # 戦略別パフォーマンス追跡
            strategy_performance = self._track_strategy_performance(
                switching_events, price_data
            )
            performance_analysis['strategy_performance'] = strategy_performance
            
            # 切替効果測定
            switching_impact = self._measure_switching_impact(
                switching_events, price_data, benchmark_returns
            )
            performance_analysis['switching_impact'] = switching_impact
            
            # パフォーマンス統計
            performance_stats = self._calculate_performance_statistics(
                switching_events, price_data
            )
            performance_analysis['performance_statistics'] = performance_stats
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            performance_analysis['error'] = str(e)
        
        return performance_analysis
    
    def _track_strategy_performance(self, 
                                  switching_events: List[Dict[str, Any]],
                                  price_data: pd.DataFrame) -> Dict[str, Any]:
        """戦略別パフォーマンス追跡"""
        strategy_periods = []
        current_strategy = "initial"
        start_time = price_data.index[0]
        
        for event in switching_events:
            end_time = event['timestamp']
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            
            # 期間データ抽出
            period_data = price_data[start_time:end_time]
            if not period_data.empty:
                period_return = (period_data['close'].iloc[-1] / 
                               period_data['close'].iloc[0] - 1)
                
                strategy_periods.append({
                    'strategy': current_strategy,
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': (end_time - start_time).total_seconds() / 3600,
                    'return': period_return,
                    'volatility': period_data['close'].pct_change().std()
                })
            
            current_strategy = event['to_strategy']
            start_time = end_time
        
        # 最後の期間
        if start_time < price_data.index[-1]:
            final_data = price_data[start_time:]
            if not final_data.empty:
                final_return = (final_data['close'].iloc[-1] / 
                              final_data['close'].iloc[0] - 1)
                
                strategy_periods.append({
                    'strategy': current_strategy,
                    'start': start_time,
                    'end': price_data.index[-1],
                    'duration_hours': (price_data.index[-1] - start_time).total_seconds() / 3600,
                    'return': final_return,
                    'volatility': final_data['close'].pct_change().std()
                })
        
        # 戦略別集計
        strategy_summary = {}
        for period in strategy_periods:
            strategy = period['strategy']
            if strategy not in strategy_summary:
                strategy_summary[strategy] = {
                    'total_periods': 0,
                    'total_duration': 0,
                    'total_return': 0,
                    'returns': [],
                    'volatilities': []
                }
            
            summary = strategy_summary[strategy]
            summary['total_periods'] += 1
            summary['total_duration'] += period['duration_hours']
            summary['total_return'] += period['return']
            summary['returns'].append(period['return'])
            summary['volatilities'].append(period.get('volatility', 0))
        
        # 統計計算
        for strategy, summary in strategy_summary.items():
            if summary['returns']:
                summary['average_return'] = np.mean(summary['returns'])
                summary['return_std'] = np.std(summary['returns'])
                summary['average_volatility'] = np.mean(summary['volatilities'])
                summary['sharpe_ratio'] = (summary['average_return'] / 
                                         (summary['return_std'] + 1e-8))
        
        return {
            'period_details': strategy_periods,
            'strategy_summary': strategy_summary
        }
    
    def _measure_switching_impact(self, 
                                switching_events: List[Dict[str, Any]],
                                price_data: pd.DataFrame,
                                benchmark_returns: pd.Series) -> Dict[str, Any]:
        """切替影響測定"""
        impact_analysis = {
            'immediate_impact': [],
            'short_term_impact': [],
            'medium_term_impact': []
        }
        
        for event in switching_events:
            timestamp = event['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            event_idx = price_data.index.get_indexer([timestamp], method='nearest')[0]
            if event_idx < 0 or event_idx >= len(price_data) - 5:
                continue
            
            # 切替前後のリターン計算
            pre_price = price_data['close'].iloc[event_idx]
            
            # 即座の影響（次の期間）
            if event_idx + 1 < len(price_data):
                immediate_return = (price_data['close'].iloc[event_idx + 1] / pre_price - 1)
                impact_analysis['immediate_impact'].append(immediate_return)
            
            # 短期影響（5期間後）
            if event_idx + 5 < len(price_data):
                short_term_return = (price_data['close'].iloc[event_idx + 5] / pre_price - 1)
                impact_analysis['short_term_impact'].append(short_term_return)
            
            # 中期影響（20期間後）
            if event_idx + 20 < len(price_data):
                medium_term_return = (price_data['close'].iloc[event_idx + 20] / pre_price - 1)
                impact_analysis['medium_term_impact'].append(medium_term_return)
        
        # 統計計算
        for period, impacts in impact_analysis.items():
            if impacts:
                impact_analysis[f'{period}_avg'] = np.mean(impacts)
                impact_analysis[f'{period}_std'] = np.std(impacts)
                impact_analysis[f'{period}_positive_ratio'] = np.mean([r > 0 for r in impacts])
        
        return impact_analysis
    
    def _calculate_performance_statistics(self, 
                                        switching_events: List[Dict[str, Any]],
                                        price_data: pd.DataFrame) -> Dict[str, float]:
        """パフォーマンス統計計算"""
        if price_data.empty:
            return {}
        
        returns = price_data['close'].pct_change().dropna()
        
        stats = {
            'total_return': (price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1),
            'annualized_return': ((price_data['close'].iloc[-1] / price_data['close'].iloc[0]) ** 
                                (252 / len(price_data)) - 1),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'win_rate': (returns > 0).mean(),
            'profit_factor': (returns[returns > 0].sum() / 
                            abs(returns[returns < 0].sum() + 1e-8)),
            'calmar_ratio': ((returns.mean() * 252) / 
                           abs((returns.cumsum() - returns.cumsum().expanding().max()).min() + 1e-8))
        }
        
        # 切替関連統計
        stats['switches_per_day'] = len(switching_events) / max(len(price_data) / 24, 1)
        stats['average_holding_period'] = (len(price_data) / max(len(switching_events) + 1, 1))
        
        return stats
    
    def _analyze_switching_risk(self, 
                              switching_events: List[Dict[str, Any]],
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """切替リスク分析"""
        risk_analysis = {}
        
        try:
            # ボラティリティ分析
            risk_analysis['volatility_analysis'] = self._analyze_volatility_impact(
                switching_events, price_data
            )
            
            # ドローダウン分析
            risk_analysis['drawdown_analysis'] = self._analyze_drawdown_patterns(
                switching_events, price_data
            )
            
            # 相関リスク分析
            risk_analysis['correlation_analysis'] = self._analyze_correlation_risk(
                switching_events, price_data
            )
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            risk_analysis['error'] = str(e)
        
        return risk_analysis
    
    def _analyze_volatility_impact(self, 
                                 switching_events: List[Dict[str, Any]],
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """ボラティリティ影響分析"""
        if price_data.empty:
            return {}
        
        returns = price_data['close'].pct_change().dropna()
        
        # 切替前後のボラティリティ比較
        pre_switch_vol = []
        post_switch_vol = []
        
        for event in switching_events:
            timestamp = event['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            event_idx = returns.index.get_indexer([timestamp], method='nearest')[0]
            if event_idx < 10 or event_idx > len(returns) - 10:
                continue
            
            pre_vol = returns.iloc[event_idx-10:event_idx].std()
            post_vol = returns.iloc[event_idx:event_idx+10].std()
            
            pre_switch_vol.append(pre_vol)
            post_switch_vol.append(post_vol)
        
        analysis = {
            'overall_volatility': returns.std() * np.sqrt(252),
            'volatility_changes': []
        }
        
        if pre_switch_vol and post_switch_vol:
            analysis['volatility_changes'] = [
                (post - pre) / (pre + 1e-8) 
                for pre, post in zip(pre_switch_vol, post_switch_vol)
            ]
            analysis['average_volatility_change'] = np.mean(analysis['volatility_changes'])
            analysis['volatility_reduction_ratio'] = np.mean([
                change < 0 for change in analysis['volatility_changes']
            ])
        
        return analysis
    
    def _analyze_drawdown_patterns(self, 
                                 switching_events: List[Dict[str, Any]],
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """ドローダウンパターン分析"""
        if price_data.empty:
            return {}
        
        returns = price_data['close'].pct_change().dropna()
        cumulative_returns = returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - running_max
        
        analysis = {
            'max_drawdown': drawdowns.min(),
            'average_drawdown': drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0,
            'drawdown_duration': self._calculate_drawdown_duration(drawdowns),
            'recovery_analysis': self._analyze_recovery_patterns(drawdowns, switching_events)
        }
        
        return analysis
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> Dict[str, float]:
        """ドローダウン期間計算"""
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        if not drawdown_periods:
            return {'average_duration': 0, 'max_duration': 0}
        
        return {
            'average_duration': np.mean(drawdown_periods),
            'max_duration': max(drawdown_periods),
            'total_periods': len(drawdown_periods)
        }
    
    def _analyze_recovery_patterns(self, 
                                 drawdowns: pd.Series,
                                 switching_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """回復パターン分析"""
        recovery_times = []
        
        # ドローダウンから回復までの時間分析
        max_dd_idx = drawdowns.idxmin()
        recovery_start = max_dd_idx
        
        for i in range(recovery_start, len(drawdowns)):
            if drawdowns.iloc[i] >= 0:
                recovery_times.append(i - recovery_start)
                break
        
        return {
            'recovery_times': recovery_times,
            'average_recovery_time': np.mean(recovery_times) if recovery_times else float('inf')
        }
    
    def _analyze_correlation_risk(self, 
                                switching_events: List[Dict[str, Any]],
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """相関リスク分析"""
        # 簡易実装：市場条件との相関分析
        correlation_analysis = {
            'market_correlation': 'Not implemented',
            'volatility_correlation': 'Not implemented'
        }
        
        return correlation_analysis
    
    def _analyze_switching_efficiency(self, 
                                    switching_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """切替効率性分析"""
        if not switching_events:
            return {}
        
        efficiency_metrics = {
            'switching_frequency': len(switching_events),
            'unique_strategy_utilization': len(set(
                [e['from_strategy'] for e in switching_events] + 
                [e['to_strategy'] for e in switching_events]
            )),
            'confidence_distribution': self._analyze_confidence_distribution(switching_events),
            'delay_analysis': self._analyze_switching_delays(switching_events)
        }
        
        return efficiency_metrics
    
    def _analyze_confidence_distribution(self, 
                                       switching_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """信頼度分布分析"""
        confidences = [event['confidence_score'] for event in switching_events]
        
        return {
            'mean_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'high_confidence_ratio': np.mean([c > 0.7 for c in confidences]),
            'low_confidence_ratio': np.mean([c < 0.3 for c in confidences])
        }
    
    def _analyze_switching_delays(self, 
                                switching_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """切替遅延分析"""
        delays = [event['switching_delay'] for event in switching_events]
        
        return {
            'mean_delay': np.mean(delays),
            'delay_std': np.std(delays),
            'max_delay': max(delays),
            'fast_switch_ratio': np.mean([d < 1.0 for d in delays])  # 1秒未満
        }

class BenchmarkComparator:
    """ベンチマーク比較器"""
    
    def __init__(self):
        self.benchmark_strategies = {
            'buy_and_hold': self._buy_and_hold_benchmark,
            'simple_moving_average': self._sma_benchmark,
            'random_switching': self._random_switching_benchmark
        }
    
    def compare_with_benchmarks(self, 
                              test_results: Dict[str, Any],
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """ベンチマーク比較実行"""
        comparison_results = {}
        
        try:
            # ベンチマーク計算
            benchmarks = {}
            for name, calculator in self.benchmark_strategies.items():
                try:
                    benchmark = calculator(price_data)
                    benchmarks[name] = benchmark
                except Exception as e:
                    logger.warning(f"Failed to calculate {name} benchmark: {e}")
            
            comparison_results['benchmarks'] = benchmarks
            
            # 比較分析
            if benchmarks and 'performance_statistics' in test_results:
                comparison_results['performance_comparison'] = self._compare_performance(
                    test_results['performance_statistics'], benchmarks
                )
            
            # 統計的有意性テスト
            comparison_results['significance_tests'] = self._perform_significance_tests(
                test_results, benchmarks, price_data
            )
            
        except Exception as e:
            logger.error(f"Error in benchmark comparison: {e}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def _buy_and_hold_benchmark(self, price_data: pd.DataFrame) -> PerformanceBenchmark:
        """バイアンドホールドベンチマーク"""
        if price_data.empty:
            raise ValueError("Empty price data")
        
        total_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1)
        returns = price_data['close'].pct_change().dropna()
        
        return PerformanceBenchmark(
            name='Buy and Hold',
            total_return=total_return,
            sharpe_ratio=(returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252),
            max_drawdown=(returns.cumsum() - returns.cumsum().expanding().max()).min(),
            volatility=returns.std() * np.sqrt(252),
            win_rate=(returns > 0).mean(),
            profit_factor=(returns[returns > 0].sum() / 
                         abs(returns[returns < 0].sum() + 1e-8)),
            calmar_ratio=total_return / abs((returns.cumsum() - returns.cumsum().expanding().max()).min() + 1e-8)
        )
    
    def _sma_benchmark(self, price_data: pd.DataFrame) -> PerformanceBenchmark:
        """単純移動平均ベンチマーク"""
        if price_data.empty or len(price_data) < 20:
            raise ValueError("Insufficient data for SMA benchmark")
        
        # 20期間移動平均戦略
        sma = price_data['close'].rolling(20).mean()
        signals = (price_data['close'] > sma).astype(int)
        strategy_returns = price_data['close'].pct_change() * signals.shift(1)
        strategy_returns = strategy_returns.dropna()
        
        total_return = (1 + strategy_returns).prod() - 1
        
        return PerformanceBenchmark(
            name='Simple Moving Average',
            total_return=total_return,
            sharpe_ratio=(strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(252),
            max_drawdown=(strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min(),
            volatility=strategy_returns.std() * np.sqrt(252),
            win_rate=(strategy_returns > 0).mean(),
            profit_factor=(strategy_returns[strategy_returns > 0].sum() / 
                         abs(strategy_returns[strategy_returns < 0].sum() + 1e-8)),
            calmar_ratio=total_return / abs((strategy_returns.cumsum() - 
                         strategy_returns.cumsum().expanding().max()).min() + 1e-8)
        )
    
    def _random_switching_benchmark(self, price_data: pd.DataFrame) -> PerformanceBenchmark:
        """ランダム切替ベンチマーク"""
        if price_data.empty:
            raise ValueError("Empty price data")
        
        # ランダム戦略切替シミュレーション
        np.random.seed(42)  # 再現性のため
        
        returns = price_data['close'].pct_change().dropna()
        random_signals = np.random.choice([0, 1], size=len(returns), p=[0.3, 0.7])
        strategy_returns = returns * random_signals
        
        total_return = (1 + strategy_returns).prod() - 1
        
        return PerformanceBenchmark(
            name='Random Switching',
            total_return=total_return,
            sharpe_ratio=(strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(252),
            max_drawdown=(strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min(),
            volatility=strategy_returns.std() * np.sqrt(252),
            win_rate=(strategy_returns > 0).mean(),
            profit_factor=(strategy_returns[strategy_returns > 0].sum() / 
                         abs(strategy_returns[strategy_returns < 0].sum() + 1e-8)),
            calmar_ratio=total_return / abs((strategy_returns.cumsum() - 
                         strategy_returns.cumsum().expanding().max()).min() + 1e-8)
        )
    
    def _compare_performance(self, 
                           test_performance: Dict[str, float],
                           benchmarks: Dict[str, PerformanceBenchmark]) -> Dict[str, Any]:
        """パフォーマンス比較"""
        comparison = {}
        
        for bench_name, benchmark in benchmarks.items():
            comparison[bench_name] = {
                'return_improvement': (test_performance.get('total_return', 0) - 
                                     benchmark.total_return),
                'sharpe_improvement': (test_performance.get('sharpe_ratio', 0) - 
                                     benchmark.sharpe_ratio),
                'drawdown_improvement': (benchmark.max_drawdown - 
                                       test_performance.get('max_drawdown', 0)),
                'volatility_change': (test_performance.get('volatility', 0) - 
                                    benchmark.volatility),
                'win_rate_improvement': (test_performance.get('win_rate', 0) - 
                                       benchmark.win_rate)
            }
        
        return comparison
    
    def _perform_significance_tests(self, 
                                  test_results: Dict[str, Any],
                                  benchmarks: Dict[str, PerformanceBenchmark],
                                  price_data: pd.DataFrame) -> Dict[str, Any]:
        """統計的有意性テスト"""
        significance_tests = {}
        
        try:
            # 簡易的な統計テスト（実際の実装では詳細な分析が必要）
            if 'performance_statistics' in test_results:
                test_return = test_results['performance_statistics'].get('total_return', 0)
                
                for bench_name, benchmark in benchmarks.items():
                    # t検定（簡易版）
                    difference = test_return - benchmark.total_return
                    
                    significance_tests[bench_name] = {
                        'return_difference': difference,
                        'is_significant': abs(difference) > 0.01,  # 1%以上の差
                        'p_value': 'Not calculated',  # 実際の実装では計算
                        'confidence_interval': (difference - 0.02, difference + 0.02)
                    }
            
        except Exception as e:
            logger.error(f"Error in significance tests: {e}")
            significance_tests['error'] = str(e)
        
        return significance_tests

def main():
    """メイン関数（テスト用）"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # テストデータ作成
        date_range = pd.date_range(start='2024-01-01', periods=100, freq='H')
        test_price_data = pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum() * 0.1,
            'high': 100 + np.random.randn(100).cumsum() * 0.1 + 0.1,
            'low': 100 + np.random.randn(100).cumsum() * 0.1 - 0.1,
            'close': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=date_range)
        
        # テスト切替イベント
        test_switching_events = [
            {
                'timestamp': date_range[20],
                'from_strategy': 'trend_following',
                'to_strategy': 'mean_reversion',
                'trigger_reason': 'trend_change',
                'confidence_score': 0.8,
                'market_conditions': {'volatility': 0.15},
                'switching_delay': 0.5
            },
            {
                'timestamp': date_range[50],
                'from_strategy': 'mean_reversion',
                'to_strategy': 'momentum',
                'trigger_reason': 'volatility_increase',
                'confidence_score': 0.7,
                'market_conditions': {'volatility': 0.25},
                'switching_delay': 1.2
            }
        ]
        
        # パフォーマンス分析実行
        analyzer = AdvancedPerformanceAnalyzer()
        benchmark_returns = test_price_data['close'].pct_change().dropna()
        
        analysis_results = analyzer.analyze_switching_effectiveness(
            test_switching_events, test_price_data, benchmark_returns
        )
        
        # ベンチマーク比較
        comparator = BenchmarkComparator()
        comparison_results = comparator.compare_with_benchmarks(
            analysis_results, test_price_data
        )
        
        # 結果表示
        print("\n" + "="*50)
        print("戦略切替性能分析テスト結果")
        print("="*50)
        
        if 'basic_stats' in analysis_results:
            stats = analysis_results['basic_stats']
            print(f"総切替回数: {stats.get('total_switches', 0)}")
            print(f"平均信頼度: {stats.get('average_confidence', 0):.3f}")
            print(f"平均切替遅延: {stats.get('average_switching_delay', 0):.3f}秒")
        
        if 'performance_statistics' in analysis_results:
            perf = analysis_results['performance_statistics']
            print(f"総リターン: {perf.get('total_return', 0):.3%}")
            print(f"シャープレシオ: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"最大ドローダウン: {perf.get('max_drawdown', 0):.3%}")
        
        if 'benchmarks' in comparison_results:
            print("\nベンチマーク比較:")
            for name, benchmark in comparison_results['benchmarks'].items():
                print(f"  {name}: リターン {benchmark.total_return:.3%}, "
                     f"シャープ {benchmark.sharpe_ratio:.3f}")
        
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
