"""
DSSMS Phase 3 Task 3.1: Performance Monitor
パフォーマンス監視クラス

高度ランキングシステムのパフォーマンスを監視し、
最適化の指標を提供します。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import threading
import psutil
import asyncio
from collections import deque, defaultdict

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class MetricType(Enum):
    """メトリクスタイプ定義"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    CACHE_PERFORMANCE = "cache_performance"
    ERROR_RATE = "error_rate"
    SYSTEM_HEALTH = "system_health"

class AlertLevel(Enum):
    """アラートレベル定義"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricData:
    """メトリクスデータ"""
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceThreshold:
    """パフォーマンス閾値"""
    metric_type: MetricType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"  # ">", "<", ">=", "<=", "=="

@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False

@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    period_start: datetime
    period_end: datetime
    summary_metrics: Dict[str, float]
    detailed_metrics: List[MetricData]
    alerts: List[PerformanceAlert]
    recommendations: List[str]
    system_health_score: float

@dataclass
class MonitoringConfig:
    """監視設定"""
    enable_performance_monitoring: bool = True
    enable_system_monitoring: bool = True
    enable_alerts: bool = True
    monitoring_interval_seconds: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 80.0,
        'disk_usage': 90.0,
        'execution_time': 30.0
    })
    log_level: str = "INFO"

class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    
    機能:
    - リアルタイムメトリクス収集
    - パフォーマンス閾値監視
    - アラート生成と通知
    - システムヘルス監視
    - パフォーマンスレポート生成
    - 最適化推奨事項提供
    """
    
    def __init__(self, monitoring_interval: int = 60):
        """
        初期化
        
        Args:
            monitoring_interval: 監視間隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logger
        
        # メトリクス保存
        self._metrics_history: Dict[MetricType, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._real_time_metrics: Dict[str, MetricData] = {}
        
        # 閾値設定
        self._thresholds: Dict[MetricType, PerformanceThreshold] = {}
        self._initialize_default_thresholds()
        
        # アラート管理
        self._active_alerts: List[PerformanceAlert] = []
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # 監視スレッド
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # パフォーマンス統計
        self._performance_stats = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0.0,
            'error_count': 0,
            'last_health_check': None
        }
        
        # システム情報
        self._system_info = self._get_system_info()
        
        self.logger.info("Performance Monitor initialized")
    
    def _initialize_default_thresholds(self):
        """デフォルト閾値初期化"""
        self._thresholds = {
            MetricType.EXECUTION_TIME: PerformanceThreshold(
                MetricType.EXECUTION_TIME,
                warning_threshold=5.0,
                error_threshold=10.0,
                critical_threshold=30.0
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                MetricType.MEMORY_USAGE,
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0
            ),
            MetricType.CPU_USAGE: PerformanceThreshold(
                MetricType.CPU_USAGE,
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                MetricType.ERROR_RATE,
                warning_threshold=1.0,
                error_threshold=5.0,
                critical_threshold=10.0
            ),
            MetricType.CACHE_PERFORMANCE: PerformanceThreshold(
                MetricType.CACHE_PERFORMANCE,
                warning_threshold=70.0,
                error_threshold=50.0,
                critical_threshold=30.0,
                comparison_operator="<"
            )
        }
    
    def start_monitoring(self):
        """監視開始"""
        try:
            if not self._monitoring_active:
                self._monitoring_active = True
                self._monitoring_thread = threading.Thread(
                    target=self._monitoring_worker, 
                    daemon=True
                )
                self._monitoring_thread.start()
                self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """監視停止"""
        try:
            self._monitoring_active = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5)
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def record_operation_start(self, operation_name: str) -> str:
        """操作開始記録"""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        self._real_time_metrics[f"{operation_id}_start"] = MetricData(
            metric_type=MetricType.EXECUTION_TIME,
            name=f"{operation_name}_start",
            value=start_time,
            unit="timestamp",
            timestamp=datetime.now(),
            metadata={"operation_id": operation_id, "operation_name": operation_name}
        )
        
        return operation_id
    
    def record_operation_end(self, operation_id: str, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """操作終了記録"""
        try:
            end_time = time.time()
            start_metric = self._real_time_metrics.get(f"{operation_id}_start")
            
            if start_metric:
                execution_time = end_time - start_metric.value
                operation_name = start_metric.metadata.get("operation_name", "unknown")
                
                # 実行時間メトリクス
                self.record_metric(
                    MetricType.EXECUTION_TIME,
                    f"{operation_name}_duration",
                    execution_time,
                    "seconds",
                    metadata={
                        "operation_id": operation_id,
                        "operation_name": operation_name,
                        "success": success,
                        **(metadata or {})
                    }
                )
                
                # 統計更新
                self._performance_stats['total_operations'] += 1
                self._performance_stats['total_execution_time'] += execution_time
                self._performance_stats['average_execution_time'] = (
                    self._performance_stats['total_execution_time'] / 
                    self._performance_stats['total_operations']
                )
                
                if not success:
                    self._performance_stats['error_count'] += 1
                
                # スタートメトリクス削除
                del self._real_time_metrics[f"{operation_id}_start"]
                
        except Exception as e:
            self.logger.warning(f"Failed to record operation end: {e}")
    
    def record_metric(
        self, 
        metric_type: MetricType, 
        name: str, 
        value: float, 
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """メトリクス記録"""
        try:
            metric = MetricData(
                metric_type=metric_type,
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # 履歴に追加
            self._metrics_history[metric_type].append(metric)
            
            # リアルタイムメトリクス更新
            self._real_time_metrics[name] = metric
            
            # 閾値チェック
            self._check_threshold(metric)
            
        except Exception as e:
            self.logger.warning(f"Failed to record metric {name}: {e}")
    
    def record_system_metrics(self):
        """システムメトリクス記録"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(
                MetricType.CPU_USAGE,
                "system_cpu_usage",
                cpu_percent,
                "percent"
            )
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            self.record_metric(
                MetricType.MEMORY_USAGE,
                "system_memory_usage",
                memory.percent,
                "percent"
            )
            
            # プロセス固有メモリ
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.record_metric(
                MetricType.MEMORY_USAGE,
                "process_memory_usage",
                process_memory,
                "MB"
            )
            
            # ピークメモリ更新
            if process_memory > self._performance_stats['peak_memory_usage']:
                self._performance_stats['peak_memory_usage'] = process_memory
            
        except Exception as e:
            self.logger.warning(f"Failed to record system metrics: {e}")
    
    def calculate_throughput(self, operation_name: str, time_window_minutes: int = 5) -> float:
        """スループット計算"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            # 指定期間の操作数をカウント
            operation_count = 0
            
            for metric in self._metrics_history[MetricType.EXECUTION_TIME]:
                if (metric.metadata.get("operation_name") == operation_name and
                    start_time <= metric.timestamp <= end_time):
                    operation_count += 1
            
            # 1分あたりの操作数
            throughput = operation_count / time_window_minutes
            
            self.record_metric(
                MetricType.THROUGHPUT,
                f"{operation_name}_throughput",
                throughput,
                "operations/minute",
                metadata={"time_window_minutes": time_window_minutes}
            )
            
            return throughput
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate throughput for {operation_name}: {e}")
            return 0.0
    
    def calculate_error_rate(self, operation_name: str, time_window_minutes: int = 30) -> float:
        """エラー率計算"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            total_operations = 0
            failed_operations = 0
            
            for metric in self._metrics_history[MetricType.EXECUTION_TIME]:
                if (metric.metadata.get("operation_name") == operation_name and
                    start_time <= metric.timestamp <= end_time):
                    total_operations += 1
                    if not metric.metadata.get("success", True):
                        failed_operations += 1
            
            error_rate = (failed_operations / total_operations * 100) if total_operations > 0 else 0.0
            
            self.record_metric(
                MetricType.ERROR_RATE,
                f"{operation_name}_error_rate",
                error_rate,
                "percent",
                metadata={
                    "time_window_minutes": time_window_minutes,
                    "total_operations": total_operations,
                    "failed_operations": failed_operations
                }
            )
            
            return error_rate
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate error rate for {operation_name}: {e}")
            return 0.0
    
    def calculate_system_health_score(self) -> float:
        """システムヘルススコア計算"""
        try:
            scores = []
            weights = []
            
            # CPU使用率スコア
            cpu_metric = self._real_time_metrics.get("system_cpu_usage")
            if cpu_metric:
                cpu_score = max(0, 100 - cpu_metric.value)
                scores.append(cpu_score)
                weights.append(0.3)
            
            # メモリ使用率スコア
            memory_metric = self._real_time_metrics.get("system_memory_usage")
            if memory_metric:
                memory_score = max(0, 100 - memory_metric.value)
                scores.append(memory_score)
                weights.append(0.3)
            
            # エラー率スコア（逆数）
            error_rates = [
                metric.value for metric in self._metrics_history[MetricType.ERROR_RATE]
                if (datetime.now() - metric.timestamp).total_seconds() < 1800  # 30分以内
            ]
            if error_rates:
                avg_error_rate = np.mean(error_rates)
                error_score = max(0, 100 - avg_error_rate * 10)
                scores.append(error_score)
                weights.append(0.2)
            
            # パフォーマンススコア
            execution_times = [
                metric.value for metric in self._metrics_history[MetricType.EXECUTION_TIME]
                if (datetime.now() - metric.timestamp).total_seconds() < 1800  # 30分以内
            ]
            if execution_times:
                avg_execution_time = np.mean(execution_times)
                performance_score = max(0, 100 - min(avg_execution_time * 10, 100))
                scores.append(performance_score)
                weights.append(0.2)
            
            # 重み付き平均
            if scores and weights:
                health_score = np.average(scores, weights=weights)
            else:
                health_score = 50.0  # デフォルト
            
            self.record_metric(
                MetricType.SYSTEM_HEALTH,
                "system_health_score",
                health_score,
                "score",
                metadata={"components": len(scores)}
            )
            
            self._performance_stats['last_health_check'] = datetime.now()
            
            return health_score
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate system health score: {e}")
            return 50.0
    
    def _check_threshold(self, metric: MetricData):
        """閾値チェック"""
        try:
            threshold = self._thresholds.get(metric.metric_type)
            if not threshold:
                return
            
            value = metric.value
            operator = threshold.comparison_operator
            
            # 閾値比較
            def compare(val, thresh, op):
                if op == ">":
                    return val > thresh
                elif op == "<":
                    return val < thresh
                elif op == ">=":
                    return val >= thresh
                elif op == "<=":
                    return val <= thresh
                elif op == "==":
                    return val == thresh
                return False
            
            # アラートレベル判定
            alert_level = None
            threshold_value = None
            
            if compare(value, threshold.critical_threshold, operator):
                alert_level = AlertLevel.CRITICAL
                threshold_value = threshold.critical_threshold
            elif compare(value, threshold.error_threshold, operator):
                alert_level = AlertLevel.ERROR
                threshold_value = threshold.error_threshold
            elif compare(value, threshold.warning_threshold, operator):
                alert_level = AlertLevel.WARNING
                threshold_value = threshold.warning_threshold
            
            # アラート生成
            if alert_level:
                self._generate_alert(alert_level, metric, threshold_value)
                
        except Exception as e:
            self.logger.warning(f"Threshold check failed for {metric.name}: {e}")
    
    def _generate_alert(self, level: AlertLevel, metric: MetricData, threshold_value: float):
        """アラート生成"""
        try:
            # 重複アラートチェック
            existing_alert = next(
                (alert for alert in self._active_alerts 
                 if alert.metric_type == metric.metric_type and 
                    alert.level == level and 
                    not alert.resolved and
                    (datetime.now() - alert.timestamp).total_seconds() < 300),  # 5分以内
                None
            )
            
            if existing_alert:
                return  # 重複アラートをスキップ
            
            # アラート作成
            alert = PerformanceAlert(
                level=level,
                metric_type=metric.metric_type,
                message=f"{metric.name}: {metric.value:.2f} {metric.unit} "
                       f"exceeded {level.value} threshold ({threshold_value:.2f})",
                value=metric.value,
                threshold=threshold_value,
                timestamp=datetime.now()
            )
            
            self._active_alerts.append(alert)
            
            # コールバック実行
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.warning(f"Alert callback failed: {e}")
            
            # ログ出力
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(level, logging.INFO)
            
            self.logger.log(log_level, f"Performance Alert: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """アラートコールバック追加"""
        self._alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: int):
        """アラート解決"""
        try:
            if 0 <= alert_id < len(self._active_alerts):
                self._active_alerts[alert_id].resolved = True
                self.logger.info(f"Alert {alert_id} resolved")
        except Exception as e:
            self.logger.warning(f"Failed to resolve alert {alert_id}: {e}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """アクティブアラート取得"""
        return [alert for alert in self._active_alerts if not alert.resolved]
    
    def get_metrics_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """メトリクスサマリ取得"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            summary = {}
            
            for metric_type, metrics in self._metrics_history.items():
                type_metrics = [
                    m for m in metrics 
                    if start_time <= m.timestamp <= end_time
                ]
                
                if type_metrics:
                    values = [m.value for m in type_metrics]
                    summary[metric_type.value] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """パフォーマンスレポート生成"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # サマリメトリクス
            summary_metrics = self.get_metrics_summary(hours)
            
            # 詳細メトリクス
            detailed_metrics = []
            for metrics in self._metrics_history.values():
                detailed_metrics.extend([
                    m for m in metrics 
                    if start_time <= m.timestamp <= end_time
                ])
            
            # アラート
            period_alerts = [
                alert for alert in self._active_alerts
                if start_time <= alert.timestamp <= end_time
            ]
            
            # システムヘルススコア
            health_score = self.calculate_system_health_score()
            
            # 推奨事項
            recommendations = self._generate_recommendations(summary_metrics, period_alerts)
            
            return PerformanceReport(
                period_start=start_time,
                period_end=end_time,
                summary_metrics={k: v.get('mean', 0) for k, v in summary_metrics.items()},
                detailed_metrics=detailed_metrics,
                alerts=period_alerts,
                recommendations=recommendations,
                system_health_score=health_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return PerformanceReport(
                period_start=datetime.now(),
                period_end=datetime.now(),
                summary_metrics={},
                detailed_metrics=[],
                alerts=[],
                recommendations=[],
                system_health_score=0.0
            )
    
    def _generate_recommendations(
        self, 
        summary_metrics: Dict[str, Any], 
        alerts: List[PerformanceAlert]
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        try:
            # CPU使用率が高い場合
            cpu_metrics = summary_metrics.get('cpu_usage', {})
            if cpu_metrics.get('mean', 0) > 70:
                recommendations.append(
                    "CPU使用率が高いです。並列処理の制限や不要な処理の最適化を検討してください。"
                )
            
            # メモリ使用率が高い場合
            memory_metrics = summary_metrics.get('memory_usage', {})
            if memory_metrics.get('mean', 0) > 80:
                recommendations.append(
                    "メモリ使用率が高いです。キャッシュサイズの調整やメモリリークの確認を行ってください。"
                )
            
            # 実行時間が長い場合
            execution_metrics = summary_metrics.get('execution_time', {})
            if execution_metrics.get('mean', 0) > 10:
                recommendations.append(
                    "処理時間が長いです。アルゴリズムの最適化やキャッシュの活用を検討してください。"
                )
            
            # エラー率が高い場合
            error_metrics = summary_metrics.get('error_rate', {})
            if error_metrics.get('mean', 0) > 2:
                recommendations.append(
                    "エラー率が高いです。入力データの検証やエラーハンドリングの改善を行ってください。"
                )
            
            # 多数のアラート
            if len(alerts) > 10:
                recommendations.append(
                    "多数のアラートが発生しています。システム全体の見直しを検討してください。"
                )
            
            # デフォルト推奨事項
            if not recommendations:
                recommendations.append("システムは正常に動作しています。継続的な監視を続けてください。")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {e}")
            recommendations.append("推奨事項の生成に失敗しました。")
        
        return recommendations
    
    def _monitoring_worker(self):
        """監視ワーカー"""
        while self._monitoring_active:
            try:
                # システムメトリクス記録
                self.record_system_metrics()
                
                # ヘルススコア計算
                self.calculate_system_health_score()
                
                # 古いアラートのクリーンアップ
                self._cleanup_old_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.warning(f"Monitoring worker error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _cleanup_old_alerts(self):
        """古いアラートのクリーンアップ"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self._active_alerts = [
                alert for alert in self._active_alerts
                if alert.timestamp > cutoff_time or not alert.resolved
            ]
        except Exception as e:
            self.logger.warning(f"Alert cleanup failed: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'platform': sys.platform
            }
        except Exception:
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """監視状態取得"""
        return {
            'monitoring_active': self._monitoring_active,
            'monitoring_interval': self.monitoring_interval,
            'metrics_count': sum(len(metrics) for metrics in self._metrics_history.values()),
            'active_alerts_count': len(self.get_active_alerts()),
            'performance_stats': self._performance_stats.copy(),
            'system_info': self._system_info,
            'thresholds': {
                metric_type.value: {
                    'warning': threshold.warning_threshold,
                    'error': threshold.error_threshold,
                    'critical': threshold.critical_threshold
                }
                for metric_type, threshold in self._thresholds.items()
            }
        }
    
    def update_threshold(self, metric_type: MetricType, threshold: PerformanceThreshold):
        """閾値更新"""
        self._thresholds[metric_type] = threshold
        self.logger.info(f"Threshold updated for {metric_type.value}")
    
    def export_metrics(self, file_path: str, format_type: str = "csv"):
        """メトリクスエクスポート"""
        try:
            all_metrics = []
            for metrics in self._metrics_history.values():
                all_metrics.extend(metrics)
            
            if not all_metrics:
                self.logger.warning("No metrics to export")
                return
            
            # DataFrameに変換
            data = []
            for metric in all_metrics:
                data.append({
                    'timestamp': metric.timestamp,
                    'metric_type': metric.metric_type.value,
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': str(metric.metadata)
                })
            
            df = pd.DataFrame(data)
            
            if format_type.lower() == "csv":
                df.to_csv(file_path, index=False)
            elif format_type.lower() == "json":
                df.to_json(file_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
