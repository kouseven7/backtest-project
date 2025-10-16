"""
パフォーマンス集計システム
フェーズ2：戦略×市場環境でのパフォーマンス分析・集約

複数戦略の実行結果を市場環境別に集約し、
時系列分析・相関分析・統計的評価を行います。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import sys
import os
import json
import pickle
import warnings
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.logger_config import setup_logger

# 統計分析用
try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Scipyが利用できません。一部の統計分析機能が制限されます。")

# プロット用
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@dataclass
class AggregationConfig:
    """集約設定"""
    market_environments: List[str] = None
    performance_metrics: List[str] = None
    time_granularity: str = "monthly"  # daily, weekly, monthly, quarterly
    correlation_threshold: float = 0.7
    confidence_level: float = 0.95
    clustering_enable: bool = True
    pca_components: int = 5
    outlier_detection: bool = True
    outlier_threshold: float = 2.0
    
    def __post_init__(self):
        if self.market_environments is None:
            self.market_environments = [
                "bull_market", "bear_market", "sideways", 
                "high_volatility", "low_volatility",
                "uptrend", "downtrend"
            ]
        
        if self.performance_metrics is None:
            self.performance_metrics = [
                "total_return", "sharpe_ratio", "win_rate", 
                "max_drawdown", "volatility", "alpha", "beta"
            ]

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

class PerformanceAggregator:
    """パフォーマンス集計・分析システム"""
    
    def __init__(self, config: AggregationConfig = None):
        """
        初期化
        
        Args:
            config: 集約設定
        """
        self.config = config or AggregationConfig()
        self.logger = setup_logger(__name__)
        
        # 集約データの保存
        self.strategy_performance = defaultdict(lambda: defaultdict(list))
        self.market_performance = defaultdict(lambda: defaultdict(list))
        self.time_series_data = defaultdict(list)
        self.correlation_matrices = {}
        self.clustering_results = {}
        
        self.logger.info("パフォーマンス集計システム初期化完了")

    def aggregate_walkforward_results(self, walkforward_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ウォークフォワード結果の集約
        
        Args:
            walkforward_results: ウォークフォワードテスト結果リスト
            
        Returns:
            集約結果
        """
        self.logger.info(f"ウォークフォワード結果集約開始: {len(walkforward_results)}件")
        
        try:
            # データの前処理と分類
            preprocessed_data = self._preprocess_results(walkforward_results)
            
            # 戦略×市場環境でのパフォーマンス集約
            strategy_market_aggregation = self._aggregate_by_strategy_market(preprocessed_data)
            
            # 時系列分析
            time_series_analysis = self._perform_time_series_analysis(preprocessed_data)
            
            # 相関分析
            correlation_analysis = self._perform_correlation_analysis(preprocessed_data)
            
            # 統計的分析
            statistical_analysis = self._perform_statistical_analysis(preprocessed_data)
            
            # クラスタリング分析
            clustering_analysis = self._perform_clustering_analysis(preprocessed_data) if self.config.clustering_enable else {}
            
            # パフォーマンスランキング
            performance_rankings = self._calculate_performance_rankings(strategy_market_aggregation)
            
            # 集約結果の構築
            aggregated_results = {
                'summary': {
                    'total_results': len(walkforward_results),
                    'strategies_analyzed': len(set(r['combination']['strategy'] for r in walkforward_results)),
                    'symbols_analyzed': len(set(r['combination']['symbol'] for r in walkforward_results)),
                    'aggregation_timestamp': datetime.now().isoformat()
                },
                'strategy_market_performance': strategy_market_aggregation,
                'time_series_analysis': time_series_analysis,
                'correlation_analysis': correlation_analysis,
                'statistical_analysis': statistical_analysis,
                'clustering_analysis': clustering_analysis,
                'performance_rankings': performance_rankings,
                'data_quality_assessment': self._assess_aggregation_quality(preprocessed_data)
            }
            
            self.logger.info("ウォークフォワード結果集約完了")
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"ウォークフォワード結果集約失敗: {e}")
            raise

    def _preprocess_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """結果の前処理"""
        preprocessed = []
        
        for result in results:
            try:
                processed_result = {
                    'combination': result.get('combination', {}),
                    'metrics': self._extract_metrics(result),
                    'market_classification': self._extract_market_classification(result),
                    'time_period': self._extract_time_period(result),
                    'data_quality': result.get('data_quality', {}),
                    'raw_result': result
                }
                
                # データ品質チェック
                if self._validate_result_quality(processed_result):
                    preprocessed.append(processed_result)
                else:
                    self.logger.warning(f"データ品質不足によりスキップ: {result.get('combination', {})}")
                    
            except Exception as e:
                self.logger.warning(f"結果前処理失敗: {e}")
                continue
        
        self.logger.info(f"前処理完了: {len(preprocessed)}/{len(results)}件が有効")
        return preprocessed

    def _extract_metrics(self, result: Dict[str, Any]) -> PerformanceMetrics:
        """メトリクスの抽出と計算"""
        try:
            summary_metrics = result.get('summary_metrics', {})
            
            # 基本メトリクス
            metrics = PerformanceMetrics(
                total_return=summary_metrics.get('avg_return', 0.0),
                volatility=summary_metrics.get('std_return', 0.0),
                sharpe_ratio=summary_metrics.get('avg_sharpe_ratio', 0.0),
                win_rate=summary_metrics.get('avg_win_rate', 0.0)
            )
            
            # 拡張メトリクスの計算
            walkforward_results = result.get('walkforward_results', [])
            if walkforward_results:
                metrics = self._calculate_extended_metrics(metrics, walkforward_results)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"メトリクス抽出失敗: {e}")
            return PerformanceMetrics()

    def _calculate_extended_metrics(self, base_metrics: PerformanceMetrics, 
                                  walkforward_results: List[Dict[str, Any]]) -> PerformanceMetrics:
        """拡張メトリクスの計算"""
        try:
            # 全ウィンドウのリターンデータを収集
            all_returns = []
            for window_result in walkforward_results:
                if 'metrics' in window_result:
                    ret = window_result['metrics'].get('total_return', 0)
                    all_returns.append(ret)
            
            if not all_returns:
                return base_metrics
            
            returns_array = np.array(all_returns)
            
            # 年率換算リターン（仮定：252取引日）
            if len(returns_array) > 0:
                avg_return = np.mean(returns_array)
                base_metrics.annualized_return = avg_return * 252
            
            # ソルティノ比率（下方偏差使用）
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns)
                if downside_deviation > 0:
                    base_metrics.sortino_ratio = base_metrics.annualized_return / downside_deviation
            
            # 最大ドローダウン
            cumulative_returns = np.cumprod(1 + returns_array) - 1
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / (1 + running_max)
            base_metrics.max_drawdown = np.abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # プロフィットファクター
            positive_returns = returns_array[returns_array > 0]
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0 and np.sum(negative_returns) != 0:
                base_metrics.profit_factor = np.sum(positive_returns) / np.abs(np.sum(negative_returns))
            
            # カルマー比率
            if base_metrics.max_drawdown > 0:
                base_metrics.calmar_ratio = base_metrics.annualized_return / base_metrics.max_drawdown
            
            return base_metrics
            
        except Exception as e:
            self.logger.warning(f"拡張メトリクス計算失敗: {e}")
            return base_metrics

    def _extract_market_classification(self, result: Dict[str, Any]) -> Dict[str, str]:
        """市場分類の抽出"""
        try:
            market_classification = result.get('market_classification', {})
            
            extracted = {}
            
            # A-B市場分類
            if 'ab_classification' in market_classification:
                ab_result = market_classification['ab_classification']
                extracted['market_state'] = ab_result.get('market_state', 'unknown')
                extracted['trend_direction'] = ab_result.get('trend_direction', 'unknown')
            
            # 拡張市場検出
            if 'enhanced_detection' in market_classification:
                enhanced_result = market_classification['enhanced_detection']
                extracted['volatility_regime'] = enhanced_result.get('volatility_regime', 'unknown')
                extracted['market_regime'] = enhanced_result.get('market_regime', 'unknown')
            
            # デフォルト値の設定
            if not extracted:
                extracted = {'market_state': 'unknown', 'trend_direction': 'unknown'}
            
            return extracted
            
        except Exception as e:
            self.logger.warning(f"市場分類抽出失敗: {e}")
            return {'market_state': 'unknown'}

    def _extract_time_period(self, result: Dict[str, Any]) -> Dict[str, str]:
        """時間期間の抽出"""
        try:
            combination = result.get('combination', {})
            
            return {
                'start_date': combination.get('start_date', ''),
                'end_date': combination.get('end_date', ''),
                'period_length_days': self._calculate_period_length(
                    combination.get('start_date'), 
                    combination.get('end_date')
                )
            }
            
        except Exception as e:
            self.logger.warning(f"時間期間抽出失敗: {e}")
            return {}

    def _calculate_period_length(self, start_date: str, end_date: str) -> int:
        """期間長の計算"""
        try:
            if not start_date or not end_date:
                return 0
            
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            return (end - start).days
            
        except Exception as e:
            return 0

    def _validate_result_quality(self, result: Dict[str, Any]) -> bool:
        """結果品質の検証"""
        try:
            # 必須フィールドの確認
            if not result.get('combination'):
                return False
            
            metrics = result.get('metrics')
            if not metrics:
                return False
            
            # メトリクスの妥当性チェック
            if isinstance(metrics, PerformanceMetrics):
                # 異常値チェック
                if abs(metrics.total_return) > 10:  # 1000%以上のリターンは異常
                    return False
                
                if metrics.sharpe_ratio < -5 or metrics.sharpe_ratio > 5:  # 極端なシャープ比率は異常
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"品質検証エラー: {e}")
            return False

    def _aggregate_by_strategy_market(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """戦略×市場環境での集約"""
        try:
            aggregation = defaultdict(lambda: defaultdict(list))
            
            # 戦略×市場環境でのグループ化
            for result in results:
                strategy = result['combination']['strategy']
                market_state = result['market_classification'].get('market_state', 'unknown')
                metrics = result['metrics']
                
                aggregation[strategy][market_state].append(metrics)
            
            # 統計の計算
            aggregated_stats = {}
            for strategy, market_data in aggregation.items():
                aggregated_stats[strategy] = {}
                for market_state, metrics_list in market_data.items():
                    if metrics_list:
                        stats = self._calculate_aggregated_statistics(metrics_list)
                        aggregated_stats[strategy][market_state] = stats
            
            return aggregated_stats
            
        except Exception as e:
            self.logger.error(f"戦略×市場環境集約失敗: {e}")
            return {}

    def _calculate_aggregated_statistics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """集約統計の計算"""
        try:
            if not metrics_list:
                return {}
            
            # メトリクスを辞書のリストに変換
            metrics_dicts = [m.to_dict() if isinstance(m, PerformanceMetrics) else m for m in metrics_list]
            
            statistics = {}
            
            # 各メトリクスの統計量を計算
            for metric_name in self.config.performance_metrics:
                values = [m.get(metric_name, 0) for m in metrics_dicts]
                values = [v for v in values if v is not None and not np.isnan(v)]
                
                if values:
                    statistics[metric_name] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values),
                        'percentile_25': np.percentile(values, 25),
                        'percentile_75': np.percentile(values, 75)
                    }
                    
                    # 信頼区間の計算
                    if SCIPY_AVAILABLE and len(values) > 1:
                        confidence_interval = stats.t.interval(
                            self.config.confidence_level,
                            len(values) - 1,
                            loc=np.mean(values),
                            scale=stats.sem(values)
                        )
                        statistics[metric_name]['confidence_interval'] = confidence_interval
            
            return statistics
            
        except Exception as e:
            self.logger.warning(f"集約統計計算失敗: {e}")
            return {}

    def _perform_time_series_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時系列分析"""
        try:
            # 時間軸でのグループ化
            time_groups = defaultdict(list)
            
            for result in results:
                time_period = result.get('time_period', {})
                start_date = time_period.get('start_date', '')
                
                if start_date:
                    # 粒度に応じた時間グループの決定
                    time_key = self._get_time_group_key(start_date)
                    time_groups[time_key].append(result)
            
            # 時系列データの構築
            time_series_data = {}
            for time_key, group_results in time_groups.items():
                if group_results:
                    # 戦略別のパフォーマンス
                    strategy_performance = defaultdict(list)
                    for result in group_results:
                        strategy = result['combination']['strategy']
                        metrics = result['metrics']
                        strategy_performance[strategy].append(metrics)
                    
                    # 各戦略の統計を計算
                    time_series_data[time_key] = {}
                    for strategy, metrics_list in strategy_performance.items():
                        stats = self._calculate_aggregated_statistics(metrics_list)
                        time_series_data[time_key][strategy] = stats
            
            # トレンド分析
            trend_analysis = self._analyze_performance_trends(time_series_data)
            
            return {
                'time_series_data': time_series_data,
                'trend_analysis': trend_analysis,
                'time_granularity': self.config.time_granularity
            }
            
        except Exception as e:
            self.logger.error(f"時系列分析失敗: {e}")
            return {}

    def _get_time_group_key(self, date_str: str) -> str:
        """時間グループキーの取得"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            if self.config.time_granularity == 'daily':
                return date.strftime('%Y-%m-%d')
            elif self.config.time_granularity == 'weekly':
                # 週の開始日（月曜日）を取得
                week_start = date - timedelta(days=date.weekday())
                return week_start.strftime('%Y-W%U')
            elif self.config.time_granularity == 'monthly':
                return date.strftime('%Y-%m')
            elif self.config.time_granularity == 'quarterly':
                quarter = (date.month - 1) // 3 + 1
                return f"{date.year}-Q{quarter}"
            else:
                return date.strftime('%Y-%m')
                
        except Exception as e:
            self.logger.warning(f"時間グループキー取得失敗: {e}")
            return 'unknown'

    def _analyze_performance_trends(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンストレンド分析"""
        try:
            trends = {}
            
            # 時系列順にソート
            sorted_periods = sorted(time_series_data.keys())
            
            # 各戦略のトレンド分析
            all_strategies = set()
            for period_data in time_series_data.values():
                all_strategies.update(period_data.keys())
            
            for strategy in all_strategies:
                strategy_trends = {}
                
                # 各メトリクスのトレンドを分析
                for metric in self.config.performance_metrics:
                    values = []
                    periods = []
                    
                    for period in sorted_periods:
                        period_data = time_series_data.get(period, {})
                        strategy_data = period_data.get(strategy, {})
                        metric_data = strategy_data.get(metric, {})
                        
                        if 'mean' in metric_data:
                            values.append(metric_data['mean'])
                            periods.append(period)
                    
                    if len(values) >= 2:
                        # 線形トレンドの計算
                        if SCIPY_AVAILABLE:
                            x = np.arange(len(values))
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                            
                            strategy_trends[metric] = {
                                'slope': slope,
                                'r_squared': r_value ** 2,
                                'p_value': p_value,
                                'trend_direction': 'improving' if slope > 0 else 'declining',
                                'significance': 'significant' if p_value < 0.05 else 'not_significant'
                            }
                
                trends[strategy] = strategy_trends
            
            return trends
            
        except Exception as e:
            self.logger.warning(f"トレンド分析失敗: {e}")
            return {}

    def _perform_correlation_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """相関分析"""
        try:
            # 戦略別のパフォーマンスマトリックス構築
            strategy_metrics = defaultdict(list)
            
            for result in results:
                strategy = result['combination']['strategy']
                metrics = result['metrics']
                
                if isinstance(metrics, PerformanceMetrics):
                    metric_values = []
                    for metric_name in self.config.performance_metrics:
                        value = getattr(metrics, metric_name, 0)
                        metric_values.append(value if value is not None else 0)
                    strategy_metrics[strategy].append(metric_values)
            
            # 戦略間相関の計算
            correlation_results = {}
            
            if len(strategy_metrics) >= 2:
                strategies = list(strategy_metrics.keys())
                
                # 各戦略の平均パフォーマンス
                strategy_means = {}
                for strategy, metrics_list in strategy_metrics.items():
                    if metrics_list:
                        strategy_means[strategy] = np.mean(metrics_list, axis=0)
                
                # 相関行列の計算
                correlation_matrix = self._calculate_correlation_matrix(strategy_means)
                correlation_results['strategy_correlation_matrix'] = correlation_matrix
                
                # 高相関ペアの特定
                high_correlation_pairs = self._find_high_correlation_pairs(
                    correlation_matrix, threshold=self.config.correlation_threshold
                )
                correlation_results['high_correlation_pairs'] = high_correlation_pairs
            
            # メトリクス間相関
            metrics_correlation = self._analyze_metrics_correlation(results)
            correlation_results['metrics_correlation'] = metrics_correlation
            
            return correlation_results
            
        except Exception as e:
            self.logger.error(f"相関分析失敗: {e}")
            return {}

    def _calculate_correlation_matrix(self, strategy_means: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """相関行列の計算"""
        try:
            strategies = list(strategy_means.keys())
            n_strategies = len(strategies)
            
            # 相関行列の初期化
            correlation_matrix = np.eye(n_strategies)
            p_values = np.eye(n_strategies)
            
            for i in range(n_strategies):
                for j in range(i + 1, n_strategies):
                    strategy1 = strategies[i]
                    strategy2 = strategies[j]
                    
                    metrics1 = strategy_means[strategy1]
                    metrics2 = strategy_means[strategy2]
                    
                    if SCIPY_AVAILABLE and len(metrics1) == len(metrics2) and len(metrics1) > 1:
                        # ピアソン相関係数の計算
                        corr, p_val = pearsonr(metrics1, metrics2)
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
            
            return {
                'matrix': correlation_matrix.tolist(),
                'p_values': p_values.tolist(),
                'strategies': strategies
            }
            
        except Exception as e:
            self.logger.warning(f"相関行列計算失敗: {e}")
            return {}

    def _find_high_correlation_pairs(self, correlation_data: Dict[str, Any], 
                                   threshold: float) -> List[Dict[str, Any]]:
        """高相関ペアの特定"""
        try:
            high_pairs = []
            
            matrix = correlation_data.get('matrix', [])
            strategies = correlation_data.get('strategies', [])
            p_values = correlation_data.get('p_values', [])
            
            n = len(strategies)
            for i in range(n):
                for j in range(i + 1, n):
                    corr = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
                    p_val = p_values[i][j] if i < len(p_values) and j < len(p_values[i]) else 1
                    
                    if abs(corr) >= threshold:
                        high_pairs.append({
                            'strategy1': strategies[i],
                            'strategy2': strategies[j],
                            'correlation': corr,
                            'p_value': p_val,
                            'significance': 'significant' if p_val < 0.05 else 'not_significant'
                        })
            
            # 相関の絶対値でソート
            high_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return high_pairs
            
        except Exception as e:
            self.logger.warning(f"高相関ペア特定失敗: {e}")
            return []

    def _analyze_metrics_correlation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """メトリクス間相関の分析"""
        try:
            # 全結果からメトリクス値を抽出
            metrics_data = defaultdict(list)
            
            for result in results:
                metrics = result['metrics']
                if isinstance(metrics, PerformanceMetrics):
                    for metric_name in self.config.performance_metrics:
                        value = getattr(metrics, metric_name, 0)
                        metrics_data[metric_name].append(value if value is not None else 0)
            
            # メトリクス間相関行列
            metric_names = list(metrics_data.keys())
            n_metrics = len(metric_names)
            
            if n_metrics >= 2 and SCIPY_AVAILABLE:
                correlation_matrix = np.eye(n_metrics)
                
                for i in range(n_metrics):
                    for j in range(i + 1, n_metrics):
                        metric1_data = metrics_data[metric_names[i]]
                        metric2_data = metrics_data[metric_names[j]]
                        
                        if len(metric1_data) == len(metric2_data) and len(metric1_data) > 1:
                            corr, _ = pearsonr(metric1_data, metric2_data)
                            correlation_matrix[i, j] = corr
                            correlation_matrix[j, i] = corr
                
                return {
                    'correlation_matrix': correlation_matrix.tolist(),
                    'metric_names': metric_names
                }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"メトリクス間相関分析失敗: {e}")
            return {}

    def _perform_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """統計的分析"""
        try:
            statistical_results = {}
            
            # 戦略別の統計テスト
            strategy_stats = self._perform_strategy_statistical_tests(results)
            statistical_results['strategy_tests'] = strategy_stats
            
            # 市場環境別の統計テスト
            market_stats = self._perform_market_statistical_tests(results)
            statistical_results['market_tests'] = market_stats
            
            # アウトライヤー検出
            if self.config.outlier_detection:
                outliers = self._detect_outliers(results)
                statistical_results['outlier_analysis'] = outliers
            
            # パフォーマンス分布分析
            distribution_analysis = self._analyze_performance_distributions(results)
            statistical_results['distribution_analysis'] = distribution_analysis
            
            return statistical_results
            
        except Exception as e:
            self.logger.error(f"統計的分析失敗: {e}")
            return {}

    def _perform_strategy_statistical_tests(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """戦略別統計テスト"""
        try:
            if not SCIPY_AVAILABLE:
                return {}
            
            # 戦略別のリターンデータ
            strategy_returns = defaultdict(list)
            
            for result in results:
                strategy = result['combination']['strategy']
                metrics = result['metrics']
                if isinstance(metrics, PerformanceMetrics):
                    strategy_returns[strategy].append(metrics.total_return)
            
            statistical_tests = {}
            strategies = list(strategy_returns.keys())
            
            # 戦略間のt検定
            if len(strategies) >= 2:
                pairwise_tests = []
                
                for i, strategy1 in enumerate(strategies):
                    for j, strategy2 in enumerate(strategies[i+1:], i+1):
                        returns1 = strategy_returns[strategy1]
                        returns2 = strategy_returns[strategy2]
                        
                        if len(returns1) > 1 and len(returns2) > 1:
                            # Welchのt検定（等分散を仮定しない）
                            t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)
                            
                            pairwise_tests.append({
                                'strategy1': strategy1,
                                'strategy2': strategy2,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'better_strategy': strategy1 if np.mean(returns1) > np.mean(returns2) else strategy2
                            })
                
                statistical_tests['pairwise_t_tests'] = pairwise_tests
            
            # 正規性検定
            normality_tests = {}
            for strategy, returns in strategy_returns.items():
                if len(returns) >= 8:  # Shapiro-Wilkテストには最低8サンプル必要
                    shapiro_stat, shapiro_p = stats.shapiro(returns)
                    normality_tests[strategy] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
            
            statistical_tests['normality_tests'] = normality_tests
            
            return statistical_tests
            
        except Exception as e:
            self.logger.warning(f"戦略統計テスト失敗: {e}")
            return {}

    def _perform_market_statistical_tests(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """市場環境別統計テスト"""
        try:
            if not SCIPY_AVAILABLE:
                return {}
            
            # 市場環境別のパフォーマンスデータ
            market_performance = defaultdict(list)
            
            for result in results:
                market_state = result['market_classification'].get('market_state', 'unknown')
                metrics = result['metrics']
                if isinstance(metrics, PerformanceMetrics):
                    market_performance[market_state].append(metrics.total_return)
            
            # 市場環境間の分散分析（ANOVA）
            market_states = list(market_performance.keys())
            if len(market_states) >= 2:
                returns_by_market = [market_performance[state] for state in market_states]
                
                # 各市場環境に十分なデータがある場合のみANOVA実行
                valid_markets = [returns for returns in returns_by_market if len(returns) >= 2]
                
                if len(valid_markets) >= 2:
                    f_stat, p_value = stats.f_oneway(*valid_markets)
                    
                    return {
                        'anova_test': {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'market_states': market_states
                        }
                    }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"市場統計テスト失敗: {e}")
            return {}

    def _detect_outliers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """アウトライヤー検出"""
        try:
            outliers = {
                'z_score_outliers': [],
                'iqr_outliers': [],
                'outlier_summary': {}
            }
            
            # 各メトリクスのアウトライヤー検出
            for metric_name in self.config.performance_metrics:
                values = []
                result_indices = []
                
                for i, result in enumerate(results):
                    metrics = result['metrics']
                    if isinstance(metrics, PerformanceMetrics):
                        value = getattr(metrics, metric_name, 0)
                        if value is not None and not np.isnan(value):
                            values.append(value)
                            result_indices.append(i)
                
                if len(values) >= 3:
                    values_array = np.array(values)
                    
                    # Z-scoreによるアウトライヤー検出
                    z_scores = np.abs(stats.zscore(values_array))
                    z_outlier_mask = z_scores > self.config.outlier_threshold
                    
                    # IQRによるアウトライヤー検出
                    q1 = np.percentile(values_array, 25)
                    q3 = np.percentile(values_array, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    iqr_outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
                    
                    # アウトライヤーの記録
                    for i, (z_outlier, iqr_outlier) in enumerate(zip(z_outlier_mask, iqr_outlier_mask)):
                        result_idx = result_indices[i]
                        result = results[result_idx]
                        
                        if z_outlier:
                            outliers['z_score_outliers'].append({
                                'result_index': result_idx,
                                'metric': metric_name,
                                'value': values[i],
                                'z_score': z_scores[i],
                                'combination': result['combination']
                            })
                        
                        if iqr_outlier:
                            outliers['iqr_outliers'].append({
                                'result_index': result_idx,
                                'metric': metric_name,
                                'value': values[i],
                                'combination': result['combination']
                            })
            
            # アウトライヤーサマリー
            outliers['outlier_summary'] = {
                'total_z_score_outliers': len(outliers['z_score_outliers']),
                'total_iqr_outliers': len(outliers['iqr_outliers']),
                'outlier_rate': (len(outliers['z_score_outliers']) + len(outliers['iqr_outliers'])) / (2 * len(results))
            }
            
            return outliers
            
        except Exception as e:
            self.logger.warning(f"アウトライヤー検出失敗: {e}")
            return {}

    def _analyze_performance_distributions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """パフォーマンス分布分析"""
        try:
            distribution_analysis = {}
            
            for metric_name in self.config.performance_metrics:
                values = []
                
                for result in results:
                    metrics = result['metrics']
                    if isinstance(metrics, PerformanceMetrics):
                        value = getattr(metrics, metric_name, 0)
                        if value is not None and not np.isnan(value):
                            values.append(value)
                
                if len(values) >= 5:
                    values_array = np.array(values)
                    
                    # 基本統計
                    distribution_stats = {
                        'mean': np.mean(values_array),
                        'median': np.median(values_array),
                        'std': np.std(values_array),
                        'skewness': stats.skew(values_array) if SCIPY_AVAILABLE else 0,
                        'kurtosis': stats.kurtosis(values_array) if SCIPY_AVAILABLE else 0,
                        'min': np.min(values_array),
                        'max': np.max(values_array),
                        'range': np.max(values_array) - np.min(values_array),
                        'percentiles': {
                            '5th': np.percentile(values_array, 5),
                            '25th': np.percentile(values_array, 25),
                            '75th': np.percentile(values_array, 75),
                            '95th': np.percentile(values_array, 95)
                        }
                    }
                    
                    distribution_analysis[metric_name] = distribution_stats
            
            return distribution_analysis
            
        except Exception as e:
            self.logger.warning(f"分布分析失敗: {e}")
            return {}

    def _perform_clustering_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """クラスタリング分析"""
        try:
            # 戦略のパフォーマンス特徴量を抽出
            feature_matrix = []
            strategy_names = []
            
            strategy_metrics = defaultdict(list)
            for result in results:
                strategy = result['combination']['strategy']
                metrics = result['metrics']
                if isinstance(metrics, PerformanceMetrics):
                    metric_values = [getattr(metrics, metric, 0) for metric in self.config.performance_metrics]
                    strategy_metrics[strategy].append(metric_values)
            
            # 戦略別の平均パフォーマンス
            for strategy, metrics_list in strategy_metrics.items():
                if metrics_list:
                    avg_metrics = np.mean(metrics_list, axis=0)
                    feature_matrix.append(avg_metrics)
                    strategy_names.append(strategy)
            
            if len(feature_matrix) < 2:
                return {}
            
            feature_matrix = np.array(feature_matrix)
            
            # 特徴量の標準化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            clustering_results = {}
            
            # K-meansクラスタリング
            if len(feature_matrix) >= 3:
                optimal_k = min(4, len(feature_matrix) - 1)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                clustering_results['kmeans'] = {
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'strategy_names': strategy_names,
                    'n_clusters': optimal_k
                }
                
                # クラスター別の戦略
                clusters = defaultdict(list)
                for strategy, label in zip(strategy_names, cluster_labels):
                    clusters[f'cluster_{label}'].append(strategy)
                
                clustering_results['kmeans']['clusters'] = dict(clusters)
            
            # PCA分析
            if len(feature_matrix) >= self.config.pca_components:
                pca = PCA(n_components=min(self.config.pca_components, len(feature_matrix)))
                pca_features = pca.fit_transform(scaled_features)
                
                clustering_results['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'transformed_features': pca_features.tolist(),
                    'strategy_names': strategy_names
                }
            
            return clustering_results
            
        except Exception as e:
            self.logger.warning(f"クラスタリング分析失敗: {e}")
            return {}

    def _calculate_performance_rankings(self, strategy_market_aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンスランキング計算"""
        try:
            rankings = {}
            
            # 総合ランキング
            overall_scores = {}
            for strategy, market_data in strategy_market_aggregation.items():
                total_score = 0
                count = 0
                
                for market_state, stats in market_data.items():
                    # 主要メトリクスの重み付きスコア
                    score = 0
                    weights = {
                        'total_return': 0.3,
                        'sharpe_ratio': 0.25,
                        'win_rate': 0.2,
                        'max_drawdown': -0.15,  # 負の重み（小さいほど良い）
                        'volatility': -0.1       # 負の重み（小さいほど良い）
                    }
                    
                    for metric, weight in weights.items():
                        if metric in stats and 'mean' in stats[metric]:
                            score += stats[metric]['mean'] * weight
                    
                    total_score += score
                    count += 1
                
                if count > 0:
                    overall_scores[strategy] = total_score / count
            
            rankings['overall'] = dict(sorted(overall_scores.items(), key=lambda x: x[1], reverse=True))
            
            # 市場環境別ランキング
            market_rankings = {}
            for market_state in self.config.market_environments:
                market_scores = {}
                
                for strategy, market_data in strategy_market_aggregation.items():
                    if market_state in market_data:
                        stats = market_data[market_state]
                        
                        # シャープレシオベースのランキング
                        if 'sharpe_ratio' in stats and 'mean' in stats['sharpe_ratio']:
                            market_scores[strategy] = stats['sharpe_ratio']['mean']
                
                if market_scores:
                    market_rankings[market_state] = dict(
                        sorted(market_scores.items(), key=lambda x: x[1], reverse=True)
                    )
            
            rankings['by_market_environment'] = market_rankings
            
            # メトリクス別ランキング
            metric_rankings = {}
            for metric in self.config.performance_metrics:
                metric_scores = {}
                
                for strategy, market_data in strategy_market_aggregation.items():
                    total_score = 0
                    count = 0
                    
                    for market_state, stats in market_data.items():
                        if metric in stats and 'mean' in stats[metric]:
                            total_score += stats[metric]['mean']
                            count += 1
                    
                    if count > 0:
                        metric_scores[strategy] = total_score / count
                
                if metric_scores:
                    # max_drawdownやvolatilityは小さいほど良い
                    reverse = metric not in ['max_drawdown', 'volatility']
                    metric_rankings[metric] = dict(
                        sorted(metric_scores.items(), key=lambda x: x[1], reverse=reverse)
                    )
            
            rankings['by_metric'] = metric_rankings
            
            return rankings
            
        except Exception as e:
            self.logger.warning(f"ランキング計算失敗: {e}")
            return {}

    def _assess_aggregation_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """集約品質の評価"""
        try:
            quality_assessment = {
                'data_completeness': {},
                'coverage_analysis': {},
                'quality_scores': {}
            }
            
            # データ完全性
            total_results = len(results)
            valid_results = len([r for r in results if self._validate_result_quality(r)])
            
            quality_assessment['data_completeness'] = {
                'total_results': total_results,
                'valid_results': valid_results,
                'completeness_rate': valid_results / total_results if total_results > 0 else 0
            }
            
            # カバレッジ分析
            strategies_covered = set(r['combination']['strategy'] for r in results)
            symbols_covered = set(r['combination']['symbol'] for r in results)
            markets_covered = set(r['market_classification'].get('market_state', 'unknown') for r in results)
            
            quality_assessment['coverage_analysis'] = {
                'strategies_count': len(strategies_covered),
                'symbols_count': len(symbols_covered),
                'market_states_count': len(markets_covered),
                'strategies_covered': list(strategies_covered),
                'symbols_covered': list(symbols_covered),
                'market_states_covered': list(markets_covered)
            }
            
            # 品質スコア
            quality_score = (
                (valid_results / total_results) * 0.4 +
                (min(len(strategies_covered), 5) / 5) * 0.3 +
                (min(len(symbols_covered), 10) / 10) * 0.2 +
                (min(len(markets_covered), 5) / 5) * 0.1
            ) if total_results > 0 else 0
            
            quality_assessment['quality_scores'] = {
                'overall_quality_score': quality_score,
                'quality_grade': self._get_quality_grade(quality_score)
            }
            
            return quality_assessment
            
        except Exception as e:
            self.logger.warning(f"集約品質評価失敗: {e}")
            return {}

    def _get_quality_grade(self, score: float) -> str:
        """品質スコアからグレードを取得"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'

    def save_aggregated_results(self, results: Dict[str, Any], output_path: str = None):
        """集約結果の保存"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"output/performance_aggregation_{timestamp}.json"
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"集約結果保存完了: {output_file}")
            
        except Exception as e:
            self.logger.error(f"集約結果保存失敗: {e}")

def create_aggregation_config(
    market_environments: List[str] = None,
    performance_metrics: List[str] = None,
    time_granularity: str = "monthly",
    **kwargs
) -> AggregationConfig:
    """集約設定の作成ヘルパー"""
    return AggregationConfig(
        market_environments=market_environments,
        performance_metrics=performance_metrics,
        time_granularity=time_granularity,
        **kwargs
    )

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="パフォーマンス集計システム")
    parser.add_argument("--input", required=True, help="ウォークフォワード結果ファイル")
    parser.add_argument("--output", help="出力ファイルパス")
    parser.add_argument("--granularity", default="monthly", choices=["daily", "weekly", "monthly", "quarterly"], help="時間粒度")
    
    args = parser.parse_args()
    
    try:
        # 結果ファイルの読み込み
        input_path = Path(args.input)
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                walkforward_results = json.load(f)
        elif input_path.suffix == '.pkl':
            with open(input_path, 'rb') as f:
                walkforward_results = pickle.load(f)
        else:
            raise ValueError("サポートされていないファイル形式")
        
        # 設定の作成
        config = create_aggregation_config(time_granularity=args.granularity)
        
        # 集約実行
        aggregator = PerformanceAggregator(config)
        
        # walkforward_resultsから実際の結果リストを抽出
        if isinstance(walkforward_results, dict):
            # 統合結果から個別結果を抽出する必要がある
            results_list = []
            # ここで結果の構造に応じて適切に抽出
            # 例: walkforward_results.get('results', [])
            print("辞書形式の結果が渡されました。結果リストの抽出が必要です。")
            return 1
        else:
            results_list = walkforward_results
        
        aggregated_results = aggregator.aggregate_walkforward_results(results_list)
        
        # 結果の保存
        output_path = args.output or f"output/performance_aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        aggregator.save_aggregated_results(aggregated_results, output_path)
        
        print(f"\n=== パフォーマンス集約完了 ===")
        print(f"入力結果数: {len(results_list)}")
        print(f"出力ファイル: {output_path}")
        
        # サマリーの表示
        summary = aggregated_results.get('summary', {})
        print(f"分析戦略数: {summary.get('strategies_analyzed', 0)}")
        print(f"分析シンボル数: {summary.get('symbols_analyzed', 0)}")
        
    except Exception as e:
        print(f"パフォーマンス集約失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
