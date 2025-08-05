"""
パフォーマンスサマリー生成器
Phase 2.A.2: 拡張トレンド切替テスター用パフォーマンス分析モジュール
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """パフォーマンスメトリクス計算"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_comprehensive_metrics(self, 
                                      data: pd.DataFrame,
                                      benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """包括的パフォーマンスメトリクス計算"""
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid data for metrics calculation")
                return {}
            
            # リターン計算
            returns = data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            # 基本メトリクス
            metrics = {}
            
            # トータルリターン
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
            metrics['total_return'] = total_return
            
            # 年率リターン
            days = (data.index[-1] - data.index[0]).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365.25 / days) - 1
                metrics['annualized_return'] = annualized_return
            else:
                metrics['annualized_return'] = 0.0
            
            # ボラティリティ（年率）
            periods_per_year = self._get_periods_per_year(data.index)
            volatility = returns.std() * np.sqrt(periods_per_year)
            metrics['volatility'] = volatility
            
            # シャープレシオ
            excess_returns = returns.mean() * periods_per_year - self.risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # ソルティノレシオ
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
            metrics['sortino_ratio'] = sortino_ratio
            
            # 最大ドローダウン
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / running_max - 1
            max_drawdown = drawdowns.min()
            metrics['max_drawdown'] = max_drawdown
            
            # カルマーレシオ
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            metrics['calmar_ratio'] = calmar_ratio
            
            # 勝率
            win_rate = (returns > 0).mean()
            metrics['win_rate'] = win_rate
            
            # プロフィットファクター
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # VaR (95%信頼区間)
            var_95 = np.percentile(returns, 5)
            metrics['var_95'] = var_95
            
            # 最大連続勝数・負数
            win_streak, loss_streak = self._calculate_streaks(returns)
            metrics['max_win_streak'] = win_streak
            metrics['max_loss_streak'] = loss_streak
            
            # ベンチマーク比較（提供されている場合）
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_data)
                metrics.update(benchmark_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _get_periods_per_year(self, index: pd.DatetimeIndex) -> float:
        """年間期間数計算"""
        if len(index) < 2:
            return 252  # デフォルト（日次）
        
        # 平均時間間隔から推定
        avg_timedelta = (index[-1] - index[0]) / len(index)
        hours = avg_timedelta.total_seconds() / 3600
        
        if hours <= 1.5:  # 1時間足
            return 24 * 365
        elif hours <= 6:   # 4時間足
            return 6 * 365
        else:              # 日足
            return 252
    
    def _calculate_streaks(self, returns: pd.Series) -> Tuple[int, int]:
        """連続勝数・負数計算"""
        try:
            win_streak = 0
            loss_streak = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for ret in returns:
                if ret > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    win_streak = max(win_streak, current_win_streak)
                elif ret < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    loss_streak = max(loss_streak, current_loss_streak)
                else:
                    current_win_streak = 0
                    current_loss_streak = 0
            
            return win_streak, loss_streak
            
        except Exception as e:
            logger.warning(f"Error calculating streaks: {e}")
            return 0, 0
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series, 
                                   benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """ベンチマーク比較メトリクス"""
        try:
            if 'close' not in benchmark_data.columns:
                return {}
            
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            
            # 共通期間に調整
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return {}
            
            aligned_returns = returns.loc[common_index]
            aligned_benchmark = benchmark_returns.loc[common_index]
            
            # ベータ計算
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # アルファ計算（年率）
            periods_per_year = self._get_periods_per_year(pd.DatetimeIndex(common_index))
            portfolio_return = aligned_returns.mean() * periods_per_year
            benchmark_return = aligned_benchmark.mean() * periods_per_year
            alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
            
            # 情報レシオ
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
            information_ratio = excess_returns.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0
            
            # 相関係数
            correlation = aligned_returns.corr(aligned_benchmark)
            
            return {
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'correlation_with_benchmark': correlation,
                'tracking_error': tracking_error
            }
            
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {e}")
            return {}

class PerformanceSummaryGenerator:
    """パフォーマンスサマリー生成器"""
    
    def __init__(self, output_dir: str = "output/performance_summaries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calculator = PerformanceMetrics()
        
        # チャート出力ディレクトリ
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(exist_ok=True)
        
        logger.info(f"PerformanceSummaryGenerator initialized (output: {self.output_dir})")
    
    def generate_comprehensive_summary(self, 
                                     results: Dict[str, Any],
                                     config: Optional[Dict] = None) -> Dict[str, Any]:
        """包括的サマリー生成"""
        try:
            summary = {
                'generation_time': datetime.now().isoformat(),
                'overview': self._generate_overview(results),
                'detailed_analysis': self._generate_detailed_analysis(results),
                'performance_ranking': self._generate_performance_ranking(results),
                'risk_analysis': self._generate_risk_analysis(results),
                'strategy_switching_analysis': self._generate_switching_analysis(results),
                'recommendations': self._generate_recommendations(results)
            }
            
            # チャート生成
            if config and config.get("output", {}).get("charts", {}).get("enabled", True):
                chart_paths = self._generate_charts(results)
                summary['chart_paths'] = chart_paths
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            return {'error': str(e)}
    
    def _generate_overview(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """概要生成"""
        try:
            test_summary = results.get('test_summary', {})
            
            overview = {
                'total_tests': test_summary.get('total_scenarios', 0),
                'successful_tests': test_summary.get('successful_scenarios', 0),
                'success_rate': test_summary.get('success_rate', 0.0),
                'total_execution_time': test_summary.get('total_execution_time', 0.0),
                'test_efficiency': self._calculate_test_efficiency(results)
            }
            
            # パフォーマンス概要
            if 'performance_analysis' in results:
                perf_stats = results['performance_analysis'].get('performance_statistics', {})
                overview.update({
                    'avg_total_return': perf_stats.get('avg_total_return', 0.0),
                    'avg_sharpe_ratio': perf_stats.get('avg_sharpe_ratio', 0.0),
                    'avg_max_drawdown': perf_stats.get('avg_max_drawdown', 0.0),
                    'best_performer': self._identify_best_performer(results),
                    'worst_performer': self._identify_worst_performer(results)
                })
            
            return overview
            
        except Exception as e:
            logger.warning(f"Error generating overview: {e}")
            return {}
    
    def _generate_detailed_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """詳細分析生成"""
        try:
            detailed_results = results.get('detailed_results', [])
            
            if not detailed_results:
                return {}
            
            # 成功したテストのみ分析
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            
            if not successful_tests:
                return {'error': 'No successful tests to analyze'}
            
            analysis = {
                'performance_distribution': self._analyze_performance_distribution(successful_tests),
                'temporal_analysis': self._analyze_temporal_patterns(successful_tests),
                'volatility_analysis': self._analyze_volatility_patterns(successful_tests),
                'return_characteristics': self._analyze_return_characteristics(successful_tests)
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error generating detailed analysis: {e}")
            return {}
    
    def _generate_performance_ranking(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """パフォーマンスランキング生成"""
        try:
            detailed_results = results.get('detailed_results', [])
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            
            if not successful_tests:
                return []
            
            # 複合スコアでランキング
            ranked_tests = []
            
            for test in successful_tests:
                metrics = test.get('performance_metrics', {})
                
                # 複合スコア計算（シャープレシオ重視）
                sharpe = metrics.get('sharpe_ratio', 0)
                total_return = metrics.get('total_return', 0)
                max_dd = abs(metrics.get('max_drawdown', 0))
                
                composite_score = (
                    sharpe * 0.4 +
                    total_return * 0.3 +
                    (1 / (1 + max_dd)) * 0.3
                )
                
                ranked_tests.append({
                    'scenario_id': test.get('scenario_id', 'unknown'),
                    'composite_score': composite_score,
                    'sharpe_ratio': sharpe,
                    'total_return': total_return,
                    'max_drawdown': max_dd,
                    'execution_time': test.get('execution_time', 0)
                })
            
            # スコア順でソート
            ranked_tests.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return ranked_tests[:10]  # 上位10件
            
        except Exception as e:
            logger.warning(f"Error generating performance ranking: {e}")
            return []
    
    def _generate_risk_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """リスク分析生成"""
        try:
            detailed_results = results.get('detailed_results', [])
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            
            if not successful_tests:
                return {}
            
            # リスクメトリクス収集
            risk_metrics = []
            for test in successful_tests:
                metrics = test.get('performance_metrics', {})
                risk_metrics.append({
                    'volatility': metrics.get('volatility', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'var_95': metrics.get('var_95', 0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0)
                })
            
            if not risk_metrics:
                return {}
            
            # リスク統計
            risk_df = pd.DataFrame(risk_metrics)
            
            risk_analysis = {
                'risk_statistics': {
                    'avg_volatility': risk_df['volatility'].mean(),
                    'max_volatility': risk_df['volatility'].max(),
                    'avg_max_drawdown': risk_df['max_drawdown'].mean(),
                    'worst_drawdown': risk_df['max_drawdown'].min(),
                    'avg_var_95': risk_df['var_95'].mean(),
                    'volatility_distribution': self._calculate_distribution_stats(risk_df['volatility'])
                },
                'risk_categories': self._categorize_risk_levels(risk_df),
                'risk_return_efficiency': self._analyze_risk_return_efficiency(successful_tests)
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.warning(f"Error generating risk analysis: {e}")
            return {}
    
    def _generate_switching_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """戦略切替分析生成"""
        try:
            if 'performance_analysis' not in results:
                return {}
            
            switching_data = results['performance_analysis'].get('switching_analysis', {})
            detailed_results = results.get('detailed_results', [])
            
            # 切替イベント詳細分析
            all_events = []
            for test in detailed_results:
                if test.get('success', False):
                    events = test.get('switching_events', [])
                    all_events.extend(events)
            
            if not all_events:
                return switching_data
            
            # 詳細分析
            switching_analysis = switching_data.copy()
            
            # 信頼度分布
            confidences = [event.get('confidence_score', 0) for event in all_events]
            switching_analysis['confidence_distribution'] = self._calculate_distribution_stats(confidences)
            
            # 切替タイミング分析
            switching_analysis['timing_analysis'] = self._analyze_switching_timing(all_events)
            
            # 戦略効果分析
            switching_analysis['strategy_effectiveness'] = self._analyze_strategy_effectiveness(all_events, detailed_results)
            
            return switching_analysis
            
        except Exception as e:
            logger.warning(f"Error generating switching analysis: {e}")
            return {}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        try:
            test_summary = results.get('test_summary', {})
            success_rate = test_summary.get('success_rate', 0)
            
            # 成功率に基づく推奨
            if success_rate < 0.7:
                recommendations.append("成功率が低いため、戦略切替ロジックの見直しを推奨します。")
            
            # パフォーマンス分析結果に基づく推奨
            if 'performance_analysis' in results:
                perf_stats = results['performance_analysis'].get('performance_statistics', {})
                
                avg_sharpe = perf_stats.get('avg_sharpe_ratio', 0)
                if avg_sharpe < 0.5:
                    recommendations.append("シャープレシオが低いため、リスク調整後リターンの改善を推奨します。")
                
                avg_drawdown = perf_stats.get('avg_max_drawdown', 0)
                if avg_drawdown < -0.2:
                    recommendations.append("最大ドローダウンが大きいため、リスク管理の強化を推奨します。")
                
                # 切替頻度に基づく推奨
                switching_data = results['performance_analysis'].get('switching_analysis', {})
                avg_switches = switching_data.get('avg_events_per_test', 0)
                
                if avg_switches > 10:
                    recommendations.append("戦略切替が頻繁すぎるため、切替しきい値の調整を推奨します。")
                elif avg_switches < 2:
                    recommendations.append("戦略切替が少なすぎるため、感度の向上を推奨します。")
            
            # 一般的な推奨
            if not recommendations:
                recommendations.append("総合的なパフォーマンスは良好です。継続的な監視と微調整を推奨します。")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["分析中にエラーが発生しました。詳細な検証を推奨します。"]
    
    def _calculate_test_efficiency(self, results: Dict[str, Any]) -> float:
        """テスト効率計算"""
        try:
            test_summary = results.get('test_summary', {})
            total_time = test_summary.get('total_execution_time', 0)
            total_tests = test_summary.get('total_scenarios', 0)
            
            if total_tests > 0 and total_time > 0:
                return total_tests / total_time  # テスト/秒
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _identify_best_performer(self, results: Dict[str, Any]) -> Optional[str]:
        """最高パフォーマンステスト特定"""
        try:
            ranking = self._generate_performance_ranking(results)
            if ranking:
                return ranking[0]['scenario_id']
            return None
        except Exception:
            return None
    
    def _identify_worst_performer(self, results: Dict[str, Any]) -> Optional[str]:
        """最低パフォーマンステスト特定"""
        try:
            ranking = self._generate_performance_ranking(results)
            if ranking:
                return ranking[-1]['scenario_id']
            return None
        except Exception:
            return None
    
    def _analyze_performance_distribution(self, successful_tests: List[Dict]) -> Dict[str, Any]:
        """パフォーマンス分布分析"""
        try:
            returns = [test.get('performance_metrics', {}).get('total_return', 0) for test in successful_tests]
            sharpe_ratios = [test.get('performance_metrics', {}).get('sharpe_ratio', 0) for test in successful_tests]
            
            return {
                'return_distribution': self._calculate_distribution_stats(returns),
                'sharpe_distribution': self._calculate_distribution_stats(sharpe_ratios),
                'correlation_analysis': self._calculate_metric_correlations(successful_tests)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing performance distribution: {e}")
            return {}
    
    def _calculate_distribution_stats(self, values: List[float]) -> Dict[str, float]:
        """分布統計計算"""
        try:
            if not values:
                return {}
            
            values_array = np.array(values)
            
            return {
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
                'skewness': float(self._calculate_skewness(values_array)),
                'kurtosis': float(self._calculate_kurtosis(values_array))
            }
            
        except Exception as e:
            logger.warning(f"Error calculating distribution stats: {e}")
            return {}
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """歪度計算"""
        try:
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return 0.0
            
            skewness = n / ((n - 1) * (n - 2)) * np.sum(((values - mean) / std) ** 3)
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """尖度計算"""
        try:
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return 0.0
            
            kurtosis = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((values - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
            return kurtosis
            
        except Exception:
            return 0.0
    
    def _analyze_temporal_patterns(self, successful_tests: List[Dict]) -> Dict[str, Any]:
        """時間的パターン分析"""
        # 簡易実装 - 実際の実装では実行時間やテスト期間の分析
        return {
            'execution_time_analysis': self._calculate_distribution_stats(
                [test.get('execution_time', 0) for test in successful_tests]
            )
        }
    
    def _analyze_volatility_patterns(self, successful_tests: List[Dict]) -> Dict[str, Any]:
        """ボラティリティパターン分析"""
        volatilities = [test.get('performance_metrics', {}).get('volatility', 0) for test in successful_tests]
        return {
            'volatility_distribution': self._calculate_distribution_stats(volatilities)
        }
    
    def _analyze_return_characteristics(self, successful_tests: List[Dict]) -> Dict[str, Any]:
        """リターン特性分析"""
        returns = [test.get('performance_metrics', {}).get('total_return', 0) for test in successful_tests]
        win_rates = [test.get('performance_metrics', {}).get('win_rate', 0) for test in successful_tests]
        
        return {
            'return_distribution': self._calculate_distribution_stats(returns),
            'win_rate_distribution': self._calculate_distribution_stats(win_rates)
        }
    
    def _calculate_metric_correlations(self, successful_tests: List[Dict]) -> Dict[str, float]:
        """メトリクス相関分析"""
        try:
            metrics_data = []
            for test in successful_tests:
                metrics = test.get('performance_metrics', {})
                metrics_data.append({
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'volatility': metrics.get('volatility', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                })
            
            if len(metrics_data) < 2:
                return {}
            
            df = pd.DataFrame(metrics_data)
            correlation_matrix = df.corr()
            
            return {
                'return_sharpe_corr': correlation_matrix.loc['total_return', 'sharpe_ratio'],
                'return_volatility_corr': correlation_matrix.loc['total_return', 'volatility'],
                'sharpe_volatility_corr': correlation_matrix.loc['sharpe_ratio', 'volatility']
            }
            
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            return {}
    
    def _categorize_risk_levels(self, risk_df: pd.DataFrame) -> Dict[str, int]:
        """リスクレベル分類"""
        try:
            categories = {'low': 0, 'medium': 0, 'high': 0}
            
            volatility_threshold_low = 0.15
            volatility_threshold_high = 0.35
            
            for _, row in risk_df.iterrows():
                vol = row['volatility']
                if vol < volatility_threshold_low:
                    categories['low'] += 1
                elif vol < volatility_threshold_high:
                    categories['medium'] += 1
                else:
                    categories['high'] += 1
            
            return categories
            
        except Exception:
            return {'low': 0, 'medium': 0, 'high': 0}
    
    def _analyze_risk_return_efficiency(self, successful_tests: List[Dict]) -> Dict[str, float]:
        """リスクリターン効率分析"""
        try:
            efficiencies = []
            for test in successful_tests:
                metrics = test.get('performance_metrics', {})
                ret = metrics.get('total_return', 0)
                vol = metrics.get('volatility', 0)
                
                if vol > 0:
                    efficiency = ret / vol
                    efficiencies.append(efficiency)
            
            if efficiencies:
                return self._calculate_distribution_stats(efficiencies)
            
            return {}
            
        except Exception:
            return {}
    
    def _analyze_switching_timing(self, all_events: List[Dict]) -> Dict[str, Any]:
        """切替タイミング分析"""
        try:
            delays = [event.get('switching_delay', 0) for event in all_events]
            
            return {
                'delay_distribution': self._calculate_distribution_stats(delays),
                'total_events': len(all_events)
            }
            
        except Exception:
            return {}
    
    def _analyze_strategy_effectiveness(self, all_events: List[Dict], detailed_results: List[Dict]) -> Dict[str, Any]:
        """戦略効果分析"""
        try:
            # 戦略別パフォーマンス（簡易実装）
            strategy_counts = {}
            for event in all_events:
                to_strategy = event.get('to_strategy', 'unknown')
                strategy_counts[to_strategy] = strategy_counts.get(to_strategy, 0) + 1
            
            return {
                'strategy_usage': strategy_counts,
                'most_used_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None
            }
            
        except Exception:
            return {}
    
    def _generate_charts(self, results: Dict[str, Any]) -> List[str]:
        """チャート生成"""
        chart_paths = []
        
        try:
            # パフォーマンス分布チャート
            perf_chart = self._create_performance_distribution_chart(results)
            if perf_chart:
                chart_paths.append(perf_chart)
            
            # リスクリターン散布図
            risk_return_chart = self._create_risk_return_scatter(results)
            if risk_return_chart:
                chart_paths.append(risk_return_chart)
            
            # 戦略切替分析チャート
            switching_chart = self._create_switching_analysis_chart(results)
            if switching_chart:
                chart_paths.append(switching_chart)
            
        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
        
        return chart_paths
    
    def _create_performance_distribution_chart(self, results: Dict[str, Any]) -> Optional[str]:
        """パフォーマンス分布チャート作成"""
        try:
            detailed_results = results.get('detailed_results', [])
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            
            if len(successful_tests) < 2:
                return None
            
            returns = [test.get('performance_metrics', {}).get('total_return', 0) for test in successful_tests]
            sharpe_ratios = [test.get('performance_metrics', {}).get('sharpe_ratio', 0) for test in successful_tests]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # リターン分布
            ax1.hist(returns, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Total Return')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Return Distribution')
            ax1.grid(True, alpha=0.3)
            
            # シャープレシオ分布
            ax2.hist(sharpe_ratios, bins=10, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Sharpe Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Sharpe Ratio Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.chart_dir / f"performance_distribution_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.warning(f"Error creating performance distribution chart: {e}")
            return None
    
    def _create_risk_return_scatter(self, results: Dict[str, Any]) -> Optional[str]:
        """リスクリターン散布図作成"""
        try:
            detailed_results = results.get('detailed_results', [])
            successful_tests = [r for r in detailed_results if r.get('success', False)]
            
            if len(successful_tests) < 3:
                return None
            
            returns = [test.get('performance_metrics', {}).get('total_return', 0) for test in successful_tests]
            volatilities = [test.get('performance_metrics', {}).get('volatility', 0) for test in successful_tests]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(volatilities, returns, alpha=0.6, s=50)
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Total Return')
            ax.set_title('Risk-Return Scatter Plot')
            ax.grid(True, alpha=0.3)
            
            # 効率フロンティア風の参考線
            if returns and volatilities:
                ax.axhline(y=np.mean(returns), color='red', linestyle='--', alpha=0.5, label='Mean Return')
                ax.axvline(x=np.mean(volatilities), color='blue', linestyle='--', alpha=0.5, label='Mean Volatility')
                ax.legend()
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.chart_dir / f"risk_return_scatter_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.warning(f"Error creating risk-return scatter: {e}")
            return None
    
    def _create_switching_analysis_chart(self, results: Dict[str, Any]) -> Optional[str]:
        """戦略切替分析チャート作成"""
        try:
            if 'performance_analysis' not in results:
                return None
            
            switching_data = results['performance_analysis'].get('switching_analysis', {})
            transitions = switching_data.get('strategy_transitions', {})
            
            if not transitions:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            strategies = list(transitions.keys())
            counts = list(transitions.values())
            
            bars = ax.bar(range(len(strategies)), counts, alpha=0.7, color='orange')
            ax.set_xlabel('Strategy Transitions')
            ax.set_ylabel('Count')
            ax.set_title('Strategy Switching Frequency')
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.chart_dir / f"switching_analysis_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.warning(f"Error creating switching analysis chart: {e}")
            return None
    
    def save_summary_to_file(self, summary: Dict[str, Any], filename: Optional[str] = None) -> str:
        """サマリーをファイルに保存"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"performance_summary_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Performance summary saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            return ""
