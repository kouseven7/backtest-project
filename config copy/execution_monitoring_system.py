"""
Module: Execution Monitoring System  
File: execution_monitoring_system.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  実行監視・パフォーマンス分析システム

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - リアルタイム実行監視・異常検知
  - パフォーマンス分析・ボトルネック特定
  - アラート・通知システム
  - 監視データ収集・レポート生成
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import queue
from collections import deque, defaultdict
import statistics
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import statistics
import psutil
import queue
from concurrent.futures import Future

# プロジェクトモジュールをインポート
try:
    from resource_allocation_engine import SystemLoad, ResourceAllocation
    from concurrent_execution_scheduler import ExecutionStatus, ExecutionResult
except ImportError:
    # スタンドアロンテスト用フォールバック
    logger = logging.getLogger(__name__)
    logger.warning("Could not import project modules, using fallback definitions")
    
    class ExecutionStatus(Enum):
        PENDING = "pending"
        RUNNING = "running" 
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        TIMEOUT = "timeout"

# ロガー設定
logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """監視レベル"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class AlertSeverity(Enum):
    """アラート重要度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """メトリクス種別"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR = "error"
    THROUGHPUT = "throughput"

@dataclass
class MonitoringMetric:
    """監視メトリクス"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    strategy_name: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['metric_type'] = self.metric_type.value
        return result

@dataclass
class Alert:
    """アラート"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    strategy_name: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['severity'] = self.severity.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    report_id: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    strategies_analyzed: List[str]
    total_executions: int
    success_rate: float
    average_execution_time: float
    throughput_per_minute: float
    resource_efficiency: Dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['generated_at'] = self.generated_at.isoformat()
        result['time_period'] = [tp.isoformat() for tp in self.time_period]
        return result

class MetricCollector:
    """メトリクス収集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: deque = deque(maxlen=config.get('monitoring', {}).get('max_metrics_history', 1000))
        self.real_time_metrics: Dict[str, MonitoringMetric] = {}
        self.collection_lock = threading.Lock()
        
    def collect_metric(self, metric: MonitoringMetric):
        """メトリクス収集"""
        with self.collection_lock:
            self.metrics_history.append(metric)
            self.real_time_metrics[metric.name] = metric
    
    def get_metrics_by_type(self, metric_type: MetricType, minutes: int = 5) -> List[MonitoringMetric]:
        """種別別メトリクス取得"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.collection_lock:
            return [
                metric for metric in self.metrics_history
                if metric.metric_type == metric_type and metric.timestamp > cutoff_time
            ]
    
    def get_strategy_metrics(self, strategy_name: str, minutes: int = 5) -> List[MonitoringMetric]:
        """戦略別メトリクス取得"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.collection_lock:
            return [
                metric for metric in self.metrics_history
                if metric.strategy_name == strategy_name and metric.timestamp > cutoff_time
            ]
    
    def get_current_metrics(self) -> Dict[str, MonitoringMetric]:
        """現在のメトリクス取得"""
        with self.collection_lock:
            return self.real_time_metrics.copy()

class AnomalyDetector:
    """異常検知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_data: Dict[str, List[float]] = defaultdict(list)
        self.alert_thresholds = config.get('monitoring', {}).get('alert_thresholds', {})
        self.detection_lock = threading.Lock()
        
    def update_baseline(self, metric: MonitoringMetric):
        """ベースライン更新"""
        with self.detection_lock:
            key = f"{metric.strategy_name}:{metric.name}" if metric.strategy_name else metric.name
            self.baseline_data[key].append(metric.value)
            
            # ベースライン履歴長制限
            max_baseline_length = self.config.get('monitoring', {}).get('baseline_history_length', 100)
            if len(self.baseline_data[key]) > max_baseline_length:
                self.baseline_data[key].pop(0)
    
    def detect_anomaly(self, metric: MonitoringMetric) -> Optional[Alert]:
        """異常検知"""
        key = f"{metric.strategy_name}:{metric.name}" if metric.strategy_name else metric.name
        
        # 閾値ベース検知
        threshold_alert = self._check_threshold_anomaly(metric)
        if threshold_alert:
            return threshold_alert
        
        # 統計的異常検知
        statistical_alert = self._check_statistical_anomaly(metric, key)
        if statistical_alert:
            return statistical_alert
        
        return None
    
    def _check_threshold_anomaly(self, metric: MonitoringMetric) -> Optional[Alert]:
        """閾値ベース異常検知"""
        thresholds = self.alert_thresholds.get(metric.name, {})
        
        # 上限閾値チェック
        upper_threshold = thresholds.get('upper')
        if upper_threshold and metric.value > upper_threshold:
            severity = AlertSeverity.CRITICAL if metric.value > upper_threshold * 1.5 else AlertSeverity.WARNING
            
            return Alert(
                alert_id=f"threshold_{metric.name}_{int(time.time())}",
                title=f"{metric.name} 上限閾値超過",
                message=f"{metric.name} が閾値 {upper_threshold} を超過しました (実際値: {metric.value})",
                severity=severity,
                timestamp=datetime.now(),
                strategy_name=metric.strategy_name,
                metric_name=metric.name,
                threshold_value=upper_threshold,
                actual_value=metric.value
            )
        
        # 下限閾値チェック
        lower_threshold = thresholds.get('lower')
        if lower_threshold and metric.value < lower_threshold:
            severity = AlertSeverity.WARNING if metric.value > lower_threshold * 0.5 else AlertSeverity.ERROR
            
            return Alert(
                alert_id=f"threshold_{metric.name}_{int(time.time())}",
                title=f"{metric.name} 下限閾値未達",
                message=f"{metric.name} が閾値 {lower_threshold} を下回りました (実際値: {metric.value})",
                severity=severity,
                timestamp=datetime.now(),
                strategy_name=metric.strategy_name,
                metric_name=metric.name,
                threshold_value=lower_threshold,
                actual_value=metric.value
            )
        
        return None
    
    def _check_statistical_anomaly(self, metric: MonitoringMetric, key: str) -> Optional[Alert]:
        """統計的異常検知"""
        with self.detection_lock:
            if key not in self.baseline_data or len(self.baseline_data[key]) < 10:
                return None  # ベースラインデータ不足
            
            baseline_values = self.baseline_data[key]
            mean_value = statistics.mean(baseline_values)
            std_value = statistics.stdev(baseline_values)
            
            # Z-score計算
            if std_value > 0:
                z_score = abs(metric.value - mean_value) / std_value
                
                # 統計的異常検知（Z-score > 2.5で異常と判定）
                if z_score > 2.5:
                    severity = AlertSeverity.WARNING if z_score < 3.5 else AlertSeverity.ERROR
                    
                    return Alert(
                        alert_id=f"statistical_{metric.name}_{int(time.time())}",
                        title=f"{metric.name} 統計的異常検知",
                        message=f"{metric.name} の値が統計的に異常です (Z-score: {z_score:.2f}, 実際値: {metric.value}, 平均値: {mean_value:.2f})",
                        severity=severity,
                        timestamp=datetime.now(),
                        strategy_name=metric.strategy_name,
                        metric_name=metric.name,
                        threshold_value=mean_value + 2.5 * std_value,
                        actual_value=metric.value
                    )
        
        return None

class AlertManager:
    """アラート管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_subscribers: List[Callable[[Alert], None]] = []
        self.alert_lock = threading.Lock()
        
    def add_alert_subscriber(self, callback: Callable[[Alert], None]):
        """アラート通知購読者追加"""
        with self.alert_lock:
            self.alert_subscribers.append(callback)
    
    def raise_alert(self, alert: Alert):
        """アラート発生"""
        with self.alert_lock:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # アラート履歴長制限
            max_history = self.config.get('monitoring', {}).get('max_alert_history', 500)
            if len(self.alert_history) > max_history:
                self.alert_history.pop(0)
        
        # 購読者通知
        for subscriber in self.alert_subscribers:
            try:
                subscriber(alert)
            except Exception as e:
                logger.error(f"Alert subscriber error: {e}")
        
        logger.warning(f"Alert raised: {alert.title} ({alert.severity.value})")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート確認"""
        with self.alert_lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        with self.alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.acknowledged = True
                # アクティブアラートから削除
                del self.active_alerts[alert_id]
                return True
            return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """アクティブアラート取得"""
        with self.alert_lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計"""
        with self.alert_lock:
            total_alerts = len(self.alert_history)
            active_count = len(self.active_alerts)
            
            # 重要度別統計
            severity_counts = defaultdict(int)
            for alert in self.alert_history:
                severity_counts[alert.severity.value] += 1
            
            # 戦略別統計
            strategy_counts = defaultdict(int)
            for alert in self.alert_history:
                if alert.strategy_name:
                    strategy_counts[alert.strategy_name] += 1
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_count,
                'severity_distribution': dict(severity_counts),
                'strategy_distribution': dict(strategy_counts),
                'resolution_rate': (total_alerts - active_count) / total_alerts if total_alerts > 0 else 1.0
            }

class PerformanceAnalyzer:
    """パフォーマンス分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_history: List[ExecutionResult] = []
        self.analysis_lock = threading.Lock()
        
    def record_execution(self, result: ExecutionResult):
        """実行結果記録"""
        with self.analysis_lock:
            self.execution_history.append(result)
            
            # 履歴長制限
            max_history = self.config.get('monitoring', {}).get('max_execution_history', 1000)
            if len(self.execution_history) > max_history:
                self.execution_history.pop(0)
    
    def generate_performance_report(self, time_range_minutes: int = 60) -> PerformanceReport:
        """パフォーマンスレポート生成"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_range_minutes)
        
        with self.analysis_lock:
            # 期間内の実行結果フィルタ
            period_results = [
                result for result in self.execution_history
                if result.start_time and start_time <= result.start_time <= end_time
            ]
            
            if not period_results:
                return self._create_empty_report(start_time, end_time)
            
            # 統計計算
            strategies_analyzed = list(set(result.strategy_name for result in period_results))
            total_executions = len(period_results)
            
            # 成功率計算
            successful_executions = sum(1 for result in period_results if result.status == ExecutionStatus.COMPLETED)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            
            # 平均実行時間
            execution_times = [result.execution_time for result in period_results if result.execution_time > 0]
            average_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            
            # スループット計算（1分あたりの実行数）
            throughput_per_minute = total_executions / time_range_minutes if time_range_minutes > 0 else 0.0
            
            # リソース効率性分析
            resource_efficiency = self._analyze_resource_efficiency(period_results)
            
            # ボトルネック特定
            bottlenecks = self._identify_bottlenecks(period_results)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                period_results, success_rate, average_execution_time, throughput_per_minute
            )
            
            report = PerformanceReport(
                report_id=f"perf_report_{int(time.time())}",
                generated_at=datetime.now(),
                time_period=(start_time, end_time),
                strategies_analyzed=strategies_analyzed,
                total_executions=total_executions,
                success_rate=success_rate,
                average_execution_time=average_execution_time,
                throughput_per_minute=throughput_per_minute,
                resource_efficiency=resource_efficiency,
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
            return report
    
    def _create_empty_report(self, start_time: datetime, end_time: datetime) -> PerformanceReport:
        """空のレポート作成"""
        return PerformanceReport(
            report_id=f"perf_report_empty_{int(time.time())}",
            generated_at=datetime.now(),
            time_period=(start_time, end_time),
            strategies_analyzed=[],
            total_executions=0,
            success_rate=0.0,
            average_execution_time=0.0,
            throughput_per_minute=0.0,
            resource_efficiency={},
            bottlenecks=[],
            recommendations=["実行データが不足しています。より多くの実行を行った後に再分析してください。"]
        )
    
    def _analyze_resource_efficiency(self, results: List[ExecutionResult]) -> Dict[str, float]:
        """リソース効率性分析"""
        efficiency = {}
        
        # 実行時間効率性（短いほど良い）
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        if execution_times:
            efficiency['execution_time_efficiency'] = 1.0 / statistics.mean(execution_times) if statistics.mean(execution_times) > 0 else 0.0
        
        # 成功率効率性
        success_count = sum(1 for r in results if r.status == ExecutionStatus.COMPLETED)
        efficiency['success_rate_efficiency'] = success_count / len(results) if results else 0.0
        
        return efficiency
    
    def _identify_bottlenecks(self, results: List[ExecutionResult]) -> List[str]:
        """ボトルネック特定"""
        bottlenecks = []
        
        # 実行時間が長い戦略
        strategy_times = defaultdict(list)
        for result in results:
            if result.execution_time > 0:
                strategy_times[result.strategy_name].append(result.execution_time)
        
        for strategy, times in strategy_times.items():
            avg_time = statistics.mean(times)
            if avg_time > 60:  # 1分以上
                bottlenecks.append(f"{strategy}: 平均実行時間が長い ({avg_time:.1f}s)")
        
        # 失敗率が高い戦略
        strategy_failures = defaultdict(lambda: {'total': 0, 'failed': 0})
        for result in results:
            strategy_failures[result.strategy_name]['total'] += 1
            if result.status == ExecutionStatus.FAILED:
                strategy_failures[result.strategy_name]['failed'] += 1
        
        for strategy, stats in strategy_failures.items():
            failure_rate = stats['failed'] / stats['total'] if stats['total'] > 0 else 0
            if failure_rate > 0.2:  # 20%以上の失敗率
                bottlenecks.append(f"{strategy}: 失敗率が高い ({failure_rate:.1%})")
        
        return bottlenecks
    
    def _generate_recommendations(
        self, 
        results: List[ExecutionResult], 
        success_rate: float, 
        avg_execution_time: float, 
        throughput: float
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        # 成功率に基づく推奨
        if success_rate < 0.8:
            recommendations.append("成功率が低いため、エラーハンドリングとタイムアウト設定の見直しを推奨")
        
        # 実行時間に基づく推奨
        if avg_execution_time > 120:
            recommendations.append("平均実行時間が長いため、並列化または処理の最適化を推奨")
        
        # スループットに基づく推奨
        if throughput < 1.0:
            recommendations.append("スループットが低いため、並行実行数の増加を検討してください")
        
        # 戦略別分析
        strategy_performance = defaultdict(lambda: {'count': 0, 'success': 0, 'total_time': 0})
        for result in results:
            strategy_performance[result.strategy_name]['count'] += 1
            strategy_performance[result.strategy_name]['total_time'] += result.execution_time
            if result.status == ExecutionStatus.COMPLETED:
                strategy_performance[result.strategy_name]['success'] += 1
        
        # パフォーマンスが悪い戦略の特定
        for strategy, stats in strategy_performance.items():
            if stats['count'] >= 3:  # 十分なサンプル数
                strategy_success_rate = stats['success'] / stats['count']
                avg_strategy_time = stats['total_time'] / stats['count']
                
                if strategy_success_rate < 0.7:
                    recommendations.append(f"{strategy}の成功率改善が必要")
                
                if avg_strategy_time > avg_execution_time * 1.5:
                    recommendations.append(f"{strategy}の実行時間最適化を推奨")
        
        return recommendations

class ExecutionMonitoringSystem:
    """実行監視システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.monitoring_level = MonitoringLevel(
            self.config.get('monitoring', {}).get('level', 'detailed')
        )
        
        # コンポーネント初期化
        self.metric_collector = MetricCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        # 監視スレッド
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metric_queue = queue.Queue()
        
        # デフォルトアラート購読者追加
        self.alert_manager.add_alert_subscriber(self._default_alert_handler)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定読み込み"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'coordination_config.json')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "monitoring": {
                "level": "detailed",
                "collection_interval": 5.0,
                "max_metrics_history": 1000,
                "max_alert_history": 500,
                "max_execution_history": 1000,
                "baseline_history_length": 100,
                "alert_thresholds": {
                    "execution_time": {"upper": 300.0},
                    "memory_usage": {"upper": 0.8},
                    "cpu_usage": {"upper": 0.9},
                    "error_rate": {"upper": 0.2}
                }
            }
        }
    
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Execution monitoring system started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Execution monitoring system stopped")
    
    def _monitoring_loop(self):
        """監視ループ"""
        collection_interval = self.config.get('monitoring', {}).get('collection_interval', 5.0)
        
        while self.monitoring_active:
            try:
                # システムメトリクス収集
                self._collect_system_metrics()
                
                # キューからメトリクス処理
                while not self.metric_queue.empty():
                    try:
                        metric = self.metric_queue.get_nowait()
                        self._process_metric(metric)
                    except queue.Empty:
                        break
                
                time.sleep(collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """システムメトリクス収集"""
        try:
            # CPU使用率
            cpu_metric = MonitoringMetric(
                name="cpu_usage",
                value=psutil.cpu_percent(interval=0.1) / 100.0,  # 0-1の範囲に正規化
                unit="percentage",
                timestamp=datetime.now(),
                metric_type=MetricType.RESOURCE
            )
            self._process_metric(cpu_metric)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_metric = MonitoringMetric(
                name="memory_usage",
                value=memory.percent / 100.0,  # 0-1の範囲に正規化
                unit="percentage",
                timestamp=datetime.now(),
                metric_type=MetricType.RESOURCE
            )
            self._process_metric(memory_metric)
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _process_metric(self, metric: MonitoringMetric):
        """メトリクス処理"""
        # メトリクス収集
        self.metric_collector.collect_metric(metric)
        
        # ベースライン更新
        self.anomaly_detector.update_baseline(metric)
        
        # 異常検知
        alert = self.anomaly_detector.detect_anomaly(metric)
        if alert:
            self.alert_manager.raise_alert(alert)
    
    def _default_alert_handler(self, alert: Alert):
        """デフォルトアラートハンドラ"""
        logger.warning(f"[ALERT] {alert.title}: {alert.message}")
    
    def record_execution_start(self, strategy_name: str, task_id: str) -> str:
        """実行開始記録"""
        start_metric = MonitoringMetric(
            name="execution_start",
            value=1.0,
            unit="count",
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            strategy_name=strategy_name,
            additional_data={'task_id': task_id}
        )
        
        self.metric_queue.put(start_metric)
        return f"execution_{strategy_name}_{int(time.time())}"
    
    def record_execution_completion(self, execution_result: ExecutionResult):
        """実行完了記録"""
        # パフォーマンス分析器に記録
        self.performance_analyzer.record_execution(execution_result)
        
        # 実行時間メトリクス
        if execution_result.execution_time > 0:
            time_metric = MonitoringMetric(
                name="execution_time",
                value=execution_result.execution_time,
                unit="seconds",
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
                strategy_name=execution_result.strategy_name,
                additional_data={'task_id': execution_result.task_id, 'status': execution_result.status.value}
            )
            self.metric_queue.put(time_metric)
        
        # 成功/失敗メトリクス
        success_metric = MonitoringMetric(
            name="execution_success",
            value=1.0 if execution_result.status == ExecutionStatus.COMPLETED else 0.0,
            unit="binary",
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
            strategy_name=execution_result.strategy_name,
            additional_data={'task_id': execution_result.task_id, 'status': execution_result.status.value}
        )
        self.metric_queue.put(success_metric)
        
        # エラーメトリクス（失敗時）
        if execution_result.status == ExecutionStatus.FAILED:
            error_metric = MonitoringMetric(
                name="execution_error",
                value=1.0,
                unit="count",
                timestamp=datetime.now(),
                metric_type=MetricType.ERROR,
                strategy_name=execution_result.strategy_name,
                additional_data={'task_id': execution_result.task_id, 'error': execution_result.error}
            )
            self.metric_queue.put(error_metric)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """リアルタイムメトリクス取得"""
        current_metrics = self.metric_collector.get_current_metrics()
        
        return {
            'system_metrics': {
                name: metric.to_dict() 
                for name, metric in current_metrics.items()
                if metric.metric_type == MetricType.RESOURCE
            },
            'performance_metrics': {
                name: metric.to_dict()
                for name, metric in current_metrics.items()
                if metric.metric_type == MetricType.PERFORMANCE
            },
            'active_alerts_count': len(self.alert_manager.get_active_alerts()),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_monitoring_report(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """監視レポート生成"""
        # パフォーマンスレポート生成
        performance_report = self.performance_analyzer.generate_performance_report(time_range_minutes)
        
        # アラート統計
        alert_stats = self.alert_manager.get_alert_statistics()
        
        # メトリクス統計
        recent_metrics = []
        for metric_type in MetricType:
            type_metrics = self.metric_collector.get_metrics_by_type(metric_type, time_range_minutes)
            recent_metrics.extend(type_metrics)
        
        report = {
            'report_id': f"monitoring_report_{int(time.time())}",
            'generated_at': datetime.now().isoformat(),
            'time_range_minutes': time_range_minutes,
            'performance_analysis': performance_report.to_dict(),
            'alert_statistics': alert_stats,
            'metrics_summary': {
                'total_metrics_collected': len(recent_metrics),
                'metrics_by_type': {
                    metric_type.value: len(self.metric_collector.get_metrics_by_type(metric_type, time_range_minutes))
                    for metric_type in MetricType
                }
            },
            'system_health': self._assess_system_health()
        }
        
        return report
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """システムヘルス評価"""
        current_metrics = self.metric_collector.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # ヘルススコア計算（0-100）
        health_score = 100
        
        # CPU使用率チェック
        if 'cpu_usage' in current_metrics:
            cpu_usage = current_metrics['cpu_usage'].value
            if cpu_usage > 0.8:
                health_score -= 20
            elif cpu_usage > 0.6:
                health_score -= 10
        
        # メモリ使用率チェック
        if 'memory_usage' in current_metrics:
            memory_usage = current_metrics['memory_usage'].value
            if memory_usage > 0.8:
                health_score -= 20
            elif memory_usage > 0.6:
                health_score -= 10
        
        # アクティブアラートチェック
        critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
        error_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.ERROR])
        
        health_score -= critical_alerts * 15
        health_score -= error_alerts * 10
        
        health_score = max(0, health_score)
        
        # ヘルス状態判定
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 70:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 30:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'active_alerts': len(active_alerts),
            'critical_alerts': critical_alerts,
            'error_alerts': error_alerts,
            'assessment_time': datetime.now().isoformat()
        }
    
    def get_strategy_performance_summary(self, strategy_name: str, minutes: int = 30) -> Dict[str, Any]:
        """戦略別パフォーマンスサマリー"""
        strategy_metrics = self.metric_collector.get_strategy_metrics(strategy_name, minutes)
        
        # 実行回数
        execution_count = len([m for m in strategy_metrics if m.name == "execution_start"])
        
        # 成功率
        success_metrics = [m for m in strategy_metrics if m.name == "execution_success"]
        success_rate = sum(m.value for m in success_metrics) / len(success_metrics) if success_metrics else 0.0
        
        # 平均実行時間
        time_metrics = [m for m in strategy_metrics if m.name == "execution_time"]
        avg_execution_time = sum(m.value for m in time_metrics) / len(time_metrics) if time_metrics else 0.0
        
        # エラー回数
        error_count = len([m for m in strategy_metrics if m.name == "execution_error"])
        
        return {
            'strategy_name': strategy_name,
            'time_period_minutes': minutes,
            'execution_count': execution_count,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'error_count': error_count,
            'summary_generated_at': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """シャットダウン"""
        logger.info("Shutting down Execution Monitoring System...")
        self.stop_monitoring()
        logger.info("Execution Monitoring System shutdown complete")

# デモ用ExecutionResult作成関数
def create_demo_execution_result(strategy_name: str, success: bool = True) -> ExecutionResult:
    """デモ用実行結果作成"""
    import random
    
    status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
    execution_time = random.uniform(5.0, 30.0)
    
    if not hasattr(create_demo_execution_result, 'ExecutionResult'):
        # フォールバック用ExecutionResult定義
        @dataclass
        class ExecutionResult:
            task_id: str
            strategy_name: str
            status: ExecutionStatus
            result: Any = None
            error: Optional[str] = None
            execution_time: float = 0.0
            start_time: Optional[datetime] = None
            end_time: Optional[datetime] = None
            retry_count: int = 0
            resource_usage: Dict[str, Any] = field(default_factory=dict)
            
            def to_dict(self) -> Dict[str, Any]:
                result = asdict(self)
                result['status'] = self.status.value
                result['start_time'] = self.start_time.isoformat() if self.start_time else None
                result['end_time'] = self.end_time.isoformat() if self.end_time else None
                return result
        
        create_demo_execution_result.ExecutionResult = ExecutionResult
    
    ExecutionResult = create_demo_execution_result.ExecutionResult
    
    return ExecutionResult(
        task_id=f"demo_{strategy_name}_{int(time.time())}",
        strategy_name=strategy_name,
        status=status,
        result={'pnl': random.uniform(-50, 100)} if success else None,
        error="Demo execution error" if not success else None,
        execution_time=execution_time,
        start_time=datetime.now() - timedelta(seconds=execution_time),
        end_time=datetime.now()
    )

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Execution Monitoring System - Demo")
    print("=" * 60)
    
    try:
        # 監視システム初期化
        monitor = ExecutionMonitoringSystem()
        
        # 監視開始
        monitor.start_monitoring()
        print("[SEARCH] Monitoring system started")
        
        # デモ戦略実行結果を記録
        demo_strategies = ["VWAPBounceStrategy", "GCStrategy", "BreakoutStrategy"]
        
        print(f"\n[CHART] Recording execution results for {len(demo_strategies)} strategies...")
        for i, strategy in enumerate(demo_strategies):
            # 実行開始記録
            execution_id = monitor.record_execution_start(strategy, f"task_{i+1}")
            
            # デモ実行結果作成・記録
            success = (i % 4) != 0  # 25%失敗率
            result = create_demo_execution_result(strategy, success)
            monitor.record_execution_completion(result)
            
            status_emoji = "[OK]" if success else "[ERROR]"
            print(f"  {status_emoji} {strategy}: {result.execution_time:.1f}s")
            
            # 少し待機（監視システムが処理する時間を確保）
            time.sleep(0.5)
        
        # リアルタイムメトリクス取得
        print(f"\n[UP] Real-time Metrics:")
        real_time_metrics = monitor.get_real_time_metrics()
        
        if real_time_metrics['system_metrics']:
            print("  System Resources:")
            for name, metric in real_time_metrics['system_metrics'].items():
                print(f"    {name}: {metric['value']:.3f} {metric['unit']}")
        
        if real_time_metrics['performance_metrics']:
            print("  Performance:")
            for name, metric in real_time_metrics['performance_metrics'].items():
                print(f"    {name}: {metric['value']:.2f} {metric['unit']}")
        
        print(f"  Active Alerts: {real_time_metrics['active_alerts_count']}")
        
        # 戦略別パフォーマンスサマリー
        print(f"\n[TARGET] Strategy Performance Summary:")
        for strategy in demo_strategies:
            summary = monitor.get_strategy_performance_summary(strategy, minutes=10)
            print(f"  {strategy}:")
            print(f"    Executions: {summary['execution_count']}")
            print(f"    Success Rate: {summary['success_rate']:.1%}")
            print(f"    Avg Time: {summary['average_execution_time']:.1f}s")
            print(f"    Errors: {summary['error_count']}")
        
        # 監視レポート生成
        print(f"\n[LIST] Generating monitoring report...")
        time.sleep(2)  # システムが処理を完了するまで待機
        
        report = monitor.generate_monitoring_report(time_range_minutes=10)
        print(f"Report ID: {report['report_id']}")
        print(f"Metrics Collected: {report['metrics_summary']['total_metrics_collected']}")
        
        # システムヘルス
        health = report['system_health']
        print(f"System Health: {health['health_status'].upper()} ({health['health_score']}/100)")
        
        # パフォーマンス分析
        perf = report['performance_analysis']
        print(f"Total Executions: {perf['total_executions']}")
        print(f"Success Rate: {perf['success_rate']:.1%}")
        print(f"Avg Execution Time: {perf['average_execution_time']:.1f}s")
        
        if perf['bottlenecks']:
            print("Bottlenecks Found:")
            for bottleneck in perf['bottlenecks']:
                print(f"  [WARNING] {bottleneck}")
        
        if perf['recommendations']:
            print("Recommendations:")
            for rec in perf['recommendations']:
                print(f"  [IDEA] {rec}")
        
        print("\n[OK] Execution Monitoring System demo completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'monitor' in locals():
            monitor.shutdown()
