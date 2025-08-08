"""
フェーズ3B メトリクス収集モジュール

このモジュールは、リアルタイムデータシステムの各種メトリクスを
収集・集計・分析する機能を提供します。
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import statistics
import numpy as np

# プロジェクト内インポート
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logger_config import setup_logger
from src.data.data_feed_integration import DataQualityMetrics, DataQualityLevel
from src.error_handling.exception_handler import UnifiedExceptionHandler


@dataclass 
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    timestamp: datetime
    operation: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class NetworkMetrics:
    """ネットワークメトリクス"""
    timestamp: datetime
    source: str
    request_count: int
    response_time_ms: float
    success_rate: float
    error_count: int
    timeout_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CacheMetrics:
    """キャッシュメトリクス"""
    timestamp: datetime
    cache_type: str  # memory, disk
    hit_count: int
    miss_count: int
    hit_rate: float
    size_bytes: int
    item_count: int
    eviction_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ErrorMetrics:
    """エラーメトリクス"""
    timestamp: datetime
    error_type: str
    error_level: str  # critical, error, warning, info
    source: str
    message: str
    recovery_attempted: bool
    recovery_successful: bool
    impact_score: float  # 0.0 - 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsAggregator:
    """メトリクス集計器"""
    
    def __init__(self, window_size: int = 300):  # 5分間のウィンドウ
        self.window_size = window_size
        self.logger = setup_logger(f"{__name__}.MetricsAggregator")
        
    def aggregate_quality_metrics(self, metrics_list: List[DataQualityMetrics]) -> Dict[str, Any]:
        """品質メトリクス集計"""
        if not metrics_list:
            return {}
            
        try:
            # 各軸のスコア集計
            completeness_scores = [m.completeness_score for m in metrics_list]
            accuracy_scores = [m.accuracy_score for m in metrics_list]
            timeliness_scores = [m.timeliness_score for m in metrics_list]
            consistency_scores = [m.consistency_score for m in metrics_list]
            overall_scores = [m.overall_score for m in metrics_list]
            
            # 品質レベル分布
            level_distribution = defaultdict(int)
            for m in metrics_list:
                level_distribution[m.quality_level.value] += 1
                
            # 問題カテゴリ分析
            issue_categories = defaultdict(int)
            for m in metrics_list:
                for issue in m.issues:
                    # 問題をカテゴリ分類
                    category = self._categorize_issue(issue)
                    issue_categories[category] += 1
                    
            return {
                'period': {
                    'start': min(m.timestamp for m in metrics_list).isoformat(),
                    'end': max(m.timestamp for m in metrics_list).isoformat(),
                    'count': len(metrics_list)
                },
                'completeness': {
                    'mean': statistics.mean(completeness_scores),
                    'median': statistics.median(completeness_scores),
                    'std': statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0.0,
                    'min': min(completeness_scores),
                    'max': max(completeness_scores)
                },
                'accuracy': {
                    'mean': statistics.mean(accuracy_scores),
                    'median': statistics.median(accuracy_scores),
                    'std': statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0,
                    'min': min(accuracy_scores),
                    'max': max(accuracy_scores)
                },
                'timeliness': {
                    'mean': statistics.mean(timeliness_scores),
                    'median': statistics.median(timeliness_scores),
                    'std': statistics.stdev(timeliness_scores) if len(timeliness_scores) > 1 else 0.0,
                    'min': min(timeliness_scores),
                    'max': max(timeliness_scores)
                },
                'consistency': {
                    'mean': statistics.mean(consistency_scores),
                    'median': statistics.median(consistency_scores),
                    'std': statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0,
                    'min': min(consistency_scores),
                    'max': max(consistency_scores)
                },
                'overall': {
                    'mean': statistics.mean(overall_scores),
                    'median': statistics.median(overall_scores),
                    'std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
                    'min': min(overall_scores),
                    'max': max(overall_scores)
                },
                'quality_distribution': dict(level_distribution),
                'issue_categories': dict(issue_categories),
                'trends': self._calculate_quality_trends(metrics_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating quality metrics: {e}")
            return {}
            
    def aggregate_performance_metrics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """パフォーマンスメトリクス集計"""
        if not metrics_list:
            return {}
            
        try:
            # 成功・失敗分離
            successful = [m for m in metrics_list if m.success]
            failed = [m for m in metrics_list if not m.success]
            
            success_rate = len(successful) / len(metrics_list) if metrics_list else 0.0
            
            # 応答時間統計（成功のみ）
            response_times = [m.duration_ms for m in successful]
            
            # オペレーション別統計
            operation_stats = defaultdict(list)
            for m in metrics_list:
                operation_stats[m.operation].append(m)
                
            operation_summary = {}
            for op, op_metrics in operation_stats.items():
                op_successful = [m for m in op_metrics if m.success]
                op_response_times = [m.duration_ms for m in op_successful]
                
                operation_summary[op] = {
                    'count': len(op_metrics),
                    'success_rate': len(op_successful) / len(op_metrics) if op_metrics else 0.0,
                    'avg_response_time': statistics.mean(op_response_times) if op_response_times else 0.0,
                    'p95_response_time': np.percentile(op_response_times, 95) if op_response_times else 0.0
                }
                
            return {
                'period': {
                    'start': min(m.timestamp for m in metrics_list).isoformat(),
                    'end': max(m.timestamp for m in metrics_list).isoformat(),
                    'count': len(metrics_list)
                },
                'success_rate': success_rate,
                'response_time': {
                    'mean': statistics.mean(response_times) if response_times else 0.0,
                    'median': statistics.median(response_times) if response_times else 0.0,
                    'p95': np.percentile(response_times, 95) if response_times else 0.0,
                    'p99': np.percentile(response_times, 99) if response_times else 0.0,
                    'min': min(response_times) if response_times else 0.0,
                    'max': max(response_times) if response_times else 0.0
                },
                'error_analysis': {
                    'total_errors': len(failed),
                    'error_rate': 1.0 - success_rate,
                    'error_types': self._analyze_error_types(failed)
                },
                'operation_breakdown': operation_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating performance metrics: {e}")
            return {}
            
    def aggregate_network_metrics(self, metrics_list: List[NetworkMetrics]) -> Dict[str, Any]:
        """ネットワークメトリクス集計"""
        if not metrics_list:
            return {}
            
        try:
            # ソース別統計
            source_stats = defaultdict(list)
            for m in metrics_list:
                source_stats[m.source].append(m)
                
            source_summary = {}
            for source, source_metrics in source_stats.items():
                total_requests = sum(m.request_count for m in source_metrics)
                total_errors = sum(m.error_count for m in source_metrics)
                total_timeouts = sum(m.timeout_count for m in source_metrics)
                avg_response_time = statistics.mean([m.response_time_ms for m in source_metrics])
                avg_success_rate = statistics.mean([m.success_rate for m in source_metrics])
                
                source_summary[source] = {
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'total_timeouts': total_timeouts,
                    'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
                    'timeout_rate': total_timeouts / total_requests if total_requests > 0 else 0.0,
                    'avg_response_time': avg_response_time,
                    'avg_success_rate': avg_success_rate
                }
                
            # 全体統計
            total_requests = sum(m.request_count for m in metrics_list)
            total_errors = sum(m.error_count for m in metrics_list)
            total_timeouts = sum(m.timeout_count for m in metrics_list)
            response_times = [m.response_time_ms for m in metrics_list]
            
            return {
                'period': {
                    'start': min(m.timestamp for m in metrics_list).isoformat(),
                    'end': max(m.timestamp for m in metrics_list).isoformat(),
                    'count': len(metrics_list)
                },
                'overall': {
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'total_timeouts': total_timeouts,
                    'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
                    'timeout_rate': total_timeouts / total_requests if total_requests > 0 else 0.0,
                    'avg_response_time': statistics.mean(response_times) if response_times else 0.0,
                    'p95_response_time': np.percentile(response_times, 95) if response_times else 0.0
                },
                'by_source': source_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating network metrics: {e}")
            return {}
            
    def _categorize_issue(self, issue: str) -> str:
        """問題をカテゴリ分類"""
        issue_lower = issue.lower()
        
        if 'missing' in issue_lower or 'empty' in issue_lower:
            return 'completeness'
        elif 'invalid' in issue_lower or 'accuracy' in issue_lower or 'outlier' in issue_lower:
            return 'accuracy'
        elif 'stale' in issue_lower or 'age' in issue_lower or 'timeout' in issue_lower:
            return 'timeliness'
        elif 'inconsistent' in issue_lower or 'mismatch' in issue_lower:
            return 'consistency'
        else:
            return 'other'
            
    def _calculate_quality_trends(self, metrics_list: List[DataQualityMetrics]) -> Dict[str, float]:
        """品質トレンド計算"""
        try:
            if len(metrics_list) < 2:
                return {}
                
            # 時系列順でソート
            sorted_metrics = sorted(metrics_list, key=lambda x: x.timestamp)
            
            # 前半と後半で比較
            mid_point = len(sorted_metrics) // 2
            first_half = sorted_metrics[:mid_point]
            second_half = sorted_metrics[mid_point:]
            
            if not first_half or not second_half:
                return {}
                
            # 各軸のトレンド計算
            trends = {}
            for axis in ['completeness', 'accuracy', 'timeliness', 'consistency', 'overall']:
                first_avg = statistics.mean([getattr(m, f"{axis}_score") for m in first_half])
                second_avg = statistics.mean([getattr(m, f"{axis}_score") for m in second_half])
                trend = (second_avg - first_avg) / first_avg if first_avg > 0 else 0.0
                trends[f"{axis}_trend"] = trend
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating quality trends: {e}")
            return {}
            
    def _analyze_error_types(self, failed_metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """エラータイプ分析"""
        error_types = defaultdict(int)
        
        for m in failed_metrics:
            if m.error_message:
                # エラーメッセージからタイプを推定
                error_type = self._classify_error(m.error_message)
                error_types[error_type] += 1
            else:
                error_types['unknown'] += 1
                
        return dict(error_types)
        
    def _classify_error(self, error_message: str) -> str:
        """エラー分類"""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network'
        elif 'auth' in error_lower or 'permission' in error_lower:
            return 'authentication'
        elif 'rate' in error_lower or 'limit' in error_lower:
            return 'rate_limit'
        elif 'parse' in error_lower or 'format' in error_lower:
            return 'data_format'
        else:
            return 'other'


class MetricsCollector:
    """メトリクス収集器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = setup_logger(__name__)
        self.exception_handler = UnifiedExceptionHandler()
        
        # メトリクス保存
        self.quality_metrics: deque = deque(maxlen=self.config.get('max_metrics', 10000))
        self.performance_metrics: deque = deque(maxlen=self.config.get('max_metrics', 10000))
        self.network_metrics: deque = deque(maxlen=self.config.get('max_metrics', 10000))
        self.cache_metrics: deque = deque(maxlen=self.config.get('max_metrics', 10000))
        self.error_metrics: deque = deque(maxlen=self.config.get('max_metrics', 10000))
        
        # 集計器
        self.aggregator = MetricsAggregator(
            window_size=self.config.get('aggregation_window', 300)
        )
        
        # 収集統計
        self.collection_stats = {
            'start_time': datetime.now(),
            'total_collected': 0,
            'collection_errors': 0
        }
        
        # コールバック
        self.metric_callbacks: List[Callable] = []
        
        self.logger.info("Metrics collector initialized")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'max_metrics': 10000,
            'aggregation_window': 300,  # 5分
            'enable_auto_cleanup': True,
            'cleanup_interval': 3600,   # 1時間
            'retention_hours': 24,
            'export_interval': 1800     # 30分
        }
        
    def record_quality_metrics(self, metrics: DataQualityMetrics):
        """品質メトリクス記録"""
        try:
            self.quality_metrics.append(metrics)
            self.collection_stats['total_collected'] += 1
            
            # コールバック実行
            for callback in self.metric_callbacks:
                try:
                    callback('quality', metrics)
                except Exception as e:
                    self.logger.warning(f"Metric callback error: {e}")
                    
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error recording quality metrics: {e}")
            self.exception_handler.handle_data_error(
                e, context={'operation': 'record_quality_metrics'}
            )
            
    def record_performance_metrics(self, operation: str, duration_ms: float, 
                                 success: bool, error_message: Optional[str] = None):
        """パフォーマンスメトリクス記録"""
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )
            
            self.performance_metrics.append(metrics)
            self.collection_stats['total_collected'] += 1
            
            # コールバック実行
            for callback in self.metric_callbacks:
                try:
                    callback('performance', metrics)
                except Exception as e:
                    self.logger.warning(f"Metric callback error: {e}")
                    
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error recording performance metrics: {e}")
            
    def record_network_metrics(self, source: str, request_count: int, 
                             response_time_ms: float, success_rate: float,
                             error_count: int = 0, timeout_count: int = 0):
        """ネットワークメトリクス記録"""
        try:
            metrics = NetworkMetrics(
                timestamp=datetime.now(),
                source=source,
                request_count=request_count,
                response_time_ms=response_time_ms,
                success_rate=success_rate,
                error_count=error_count,
                timeout_count=timeout_count
            )
            
            self.network_metrics.append(metrics)
            self.collection_stats['total_collected'] += 1
            
            # コールバック実行
            for callback in self.metric_callbacks:
                try:
                    callback('network', metrics)
                except Exception as e:
                    self.logger.warning(f"Metric callback error: {e}")
                    
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error recording network metrics: {e}")
            
    def record_cache_metrics(self, cache_type: str, hit_count: int, miss_count: int,
                           size_bytes: int, item_count: int, eviction_count: int = 0):
        """キャッシュメトリクス記録"""
        try:
            hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0.0
            
            metrics = CacheMetrics(
                timestamp=datetime.now(),
                cache_type=cache_type,
                hit_count=hit_count,
                miss_count=miss_count,
                hit_rate=hit_rate,
                size_bytes=size_bytes,
                item_count=item_count,
                eviction_count=eviction_count
            )
            
            self.cache_metrics.append(metrics)
            self.collection_stats['total_collected'] += 1
            
            # コールバック実行
            for callback in self.metric_callbacks:
                try:
                    callback('cache', metrics)
                except Exception as e:
                    self.logger.warning(f"Metric callback error: {e}")
                    
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error recording cache metrics: {e}")
            
    def record_error_metrics(self, error_type: str, error_level: str, source: str,
                           message: str, recovery_attempted: bool = False,
                           recovery_successful: bool = False, impact_score: float = 0.5):
        """エラーメトリクス記録"""
        try:
            metrics = ErrorMetrics(
                timestamp=datetime.now(),
                error_type=error_type,
                error_level=error_level,
                source=source,
                message=message,
                recovery_attempted=recovery_attempted,
                recovery_successful=recovery_successful,
                impact_score=impact_score
            )
            
            self.error_metrics.append(metrics)
            self.collection_stats['total_collected'] += 1
            
            # コールバック実行
            for callback in self.metric_callbacks:
                try:
                    callback('error', metrics)
                except Exception as e:
                    self.logger.warning(f"Metric callback error: {e}")
                    
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"Error recording error metrics: {e}")
            
    def get_quality_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """品質サマリー取得"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_metrics = [
                m for m in self.quality_metrics 
                if m.timestamp > cutoff_time
            ]
            
            return self.aggregator.aggregate_quality_metrics(recent_metrics)
            
        except Exception as e:
            self.logger.error(f"Error getting quality summary: {e}")
            return {}
            
    def get_performance_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp > cutoff_time
            ]
            
            return self.aggregator.aggregate_performance_metrics(recent_metrics)
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
            
    def get_network_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """ネットワークサマリー取得"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_metrics = [
                m for m in self.network_metrics 
                if m.timestamp > cutoff_time
            ]
            
            return self.aggregator.aggregate_network_metrics(recent_metrics)
            
        except Exception as e:
            self.logger.error(f"Error getting network summary: {e}")
            return {}
            
    def get_all_metrics_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """全メトリクスサマリー取得"""
        try:
            return {
                'quality': self.get_quality_summary(time_window_minutes),
                'performance': self.get_performance_summary(time_window_minutes),
                'network': self.get_network_summary(time_window_minutes),
                'collection_stats': self.collection_stats,
                'summary_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all metrics summary: {e}")
            return {}
            
    def add_metric_callback(self, callback: Callable):
        """メトリクスコールバック追加"""
        self.metric_callbacks.append(callback)
        
    def remove_metric_callback(self, callback: Callable):
        """メトリクスコールバック削除"""
        if callback in self.metric_callbacks:
            self.metric_callbacks.remove(callback)
            
    def export_metrics(self, filepath: str, time_window_hours: int = 24):
        """メトリクスエクスポート"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'time_window_hours': time_window_hours,
                    'cutoff_time': cutoff_time.isoformat()
                },
                'quality_metrics': [
                    m.to_dict() for m in self.quality_metrics 
                    if m.timestamp > cutoff_time
                ],
                'performance_metrics': [
                    m.to_dict() for m in self.performance_metrics 
                    if m.timestamp > cutoff_time
                ],
                'network_metrics': [
                    m.to_dict() for m in self.network_metrics 
                    if m.timestamp > cutoff_time
                ],
                'cache_metrics': [
                    m.to_dict() for m in self.cache_metrics 
                    if m.timestamp > cutoff_time
                ],
                'error_metrics': [
                    m.to_dict() for m in self.error_metrics 
                    if m.timestamp > cutoff_time
                ],
                'collection_stats': self.collection_stats
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            self.exception_handler.handle_system_error(
                e, context={'operation': 'export_metrics', 'filepath': filepath}
            )
            
    def cleanup_old_metrics(self, retention_hours: int = 24):
        """古いメトリクスクリーンアップ"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=retention_hours)
            
            # 各メトリクスタイプのクリーンアップ
            metrics_collections = [
                self.quality_metrics,
                self.performance_metrics,
                self.network_metrics,
                self.cache_metrics,
                self.error_metrics
            ]
            
            total_removed = 0
            for collection in metrics_collections:
                original_length = len(collection)
                
                # 古いメトリクスを削除
                while collection and collection[0].timestamp < cutoff_time:
                    collection.popleft()
                    
                removed = original_length - len(collection)
                total_removed += removed
                
            self.logger.info(f"Cleaned up {total_removed} old metrics")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {e}")


if __name__ == "__main__":
    # テスト用デモ
    collector = MetricsCollector()
    
    # サンプルメトリクス記録
    collector.record_performance_metrics("data_fetch", 150.5, True)
    collector.record_network_metrics("yahoo_finance", 10, 120.3, 0.95, 1, 0)
    collector.record_cache_metrics("memory", 850, 150, 1024*1024*50, 1000, 5)
    
    # サマリー取得
    summary = collector.get_all_metrics_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
