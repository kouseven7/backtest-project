"""
監視エージェントシステム
リアルタイムエラー監視、通知、レポート生成を提供
"""

import time
import threading
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import queue

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


@dataclass
class AlertEvent:
    """アラートイベント"""
    timestamp: datetime
    event_type: str
    severity: str
    message: str
    context: Dict[str, Any]
    strategy_name: Optional[str] = None


@dataclass
class NotificationConfig:
    """通知設定"""
    enabled: bool
    email_enabled: bool = False
    email_recipients: List[str] = None
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    webhook_enabled: bool = False
    webhook_url: str = ""
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


class AlertRule:
    """アラートルール"""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                 severity: str = "WARNING", cooldown_minutes: int = 5):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None
        self.trigger_count = 0
    
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """トリガー判定"""
        if not self.condition(context):
            return False
        
        # クールダウン チェック
        if self.last_triggered:
            time_diff = datetime.now() - self.last_triggered
            if time_diff < timedelta(minutes=self.cooldown_minutes):
                return False
        
        return True
    
    def trigger(self):
        """トリガー実行"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1


class MetricsCollector:
    """メトリクス収集"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def add_metric(self, metric_name: str, value: float, 
                   timestamp: Optional[datetime] = None, 
                   tags: Optional[Dict[str, str]] = None):
        """メトリクス追加"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }
        
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append(metric_entry)
            
            # 古いメトリクス削除（1時間以上古い）
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name] 
                if m['timestamp'] > cutoff_time
            ]
    
    def get_metrics(self, metric_name: str, 
                   since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """メトリクス取得"""
        with self.lock:
            metrics = self.metrics.get(metric_name, [])
            
            if since:
                metrics = [m for m in metrics if m['timestamp'] >= since]
            
            return metrics.copy()
    
    def get_metric_summary(self, metric_name: str, 
                          since: Optional[datetime] = None) -> Dict[str, float]:
        """メトリクス要約"""
        metrics = self.get_metrics(metric_name, since)
        
        if not metrics:
            return {'count': 0, 'avg': 0.0, 'min': 0.0, 'max': 0.0, 'sum': 0.0}
        
        values = [m['value'] for m in metrics]
        
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'sum': sum(values)
        }


class NotificationManager:
    """通知管理"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = setup_logger(__name__)
        self.notification_queue = queue.Queue()
        
        # 非同期通知処理
        self.notification_thread = threading.Thread(
            target=self._process_notifications, daemon=True
        )
        self.notification_thread.start()
    
    def send_alert(self, alert: AlertEvent):
        """アラート送信"""
        if not self.config.enabled:
            return
        
        self.notification_queue.put(alert)
    
    def _process_notifications(self):
        """通知処理（非同期）"""
        while True:
            try:
                alert = self.notification_queue.get(timeout=1)
                
                if self.config.email_enabled:
                    self._send_email_notification(alert)
                
                if self.config.webhook_enabled:
                    self._send_webhook_notification(alert)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"通知送信エラー: {e}")
    
    def _send_email_notification(self, alert: AlertEvent):
        """メール通知送信"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity}] バックテストシステムアラート: {alert.event_type}"
            
            body = f"""
            アラート詳細:
            
            時刻: {alert.timestamp.isoformat()}
            種類: {alert.event_type}
            重要度: {alert.severity}
            メッセージ: {alert.message}
            戦略: {alert.strategy_name or 'N/A'}
            
            コンテキスト:
            {json.dumps(alert.context, ensure_ascii=False, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"メール通知送信成功: {alert.event_type}")
            
        except Exception as e:
            self.logger.error(f"メール通知送信失敗: {e}")
    
    def _send_webhook_notification(self, alert: AlertEvent):
        """Webhook通知送信"""
        try:
            import requests
            
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'event_type': alert.event_type,
                'severity': alert.severity,
                'message': alert.message,
                'strategy_name': alert.strategy_name,
                'context': alert.context
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook通知送信成功: {alert.event_type}")
            else:
                self.logger.warning(f"Webhook通知送信失敗: {response.status_code}")
                
        except ImportError:
            self.logger.warning("requestsライブラリが利用できません")
        except Exception as e:
            self.logger.error(f"Webhook通知送信エラー: {e}")


class MonitoringAgent:
    """監視エージェント"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        
        # 設定読み込み
        if config_path is None:
            config_path = project_root / "config" / "error_handling" / "notification_config.json"
        
        self.config_path = Path(config_path)
        self.notification_config = self._load_notification_config()
        
        # コンポーネント初期化
        self.notification_manager = NotificationManager(self.notification_config)
        self.metrics_collector = MetricsCollector()
        
        # アラートルール
        self.alert_rules: Dict[str, AlertRule] = {}
        self._setup_default_alert_rules()
        
        # 監視状態
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 統計
        self.monitoring_stats = {
            'start_time': None,
            'alerts_triggered': 0,
            'rules_evaluated': 0,
            'metrics_collected': 0,
            'uptime': 0.0
        }
    
    def _load_notification_config(self) -> NotificationConfig:
        """通知設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                return NotificationConfig(**config_data.get('notification', {}))
            else:
                self.logger.warning(f"通知設定ファイル未見つけ: {self.config_path}")
                return NotificationConfig(enabled=False)
        except Exception as e:
            self.logger.error(f"通知設定読み込みエラー: {e}")
            return NotificationConfig(enabled=False)
    
    def _setup_default_alert_rules(self):
        """デフォルトアラートルール設定"""
        # エラー率アラート
        self.add_alert_rule(
            "high_error_rate",
            lambda ctx: ctx.get('error_rate', 0) > 10.0,
            "ERROR",
            cooldown_minutes=10
        )
        
        # パフォーマンスアラート
        self.add_alert_rule(
            "slow_execution",
            lambda ctx: ctx.get('execution_time', 0) > 30.0,
            "WARNING",
            cooldown_minutes=5
        )
        
        # メモリ使用量アラート
        self.add_alert_rule(
            "high_memory_usage",
            lambda ctx: ctx.get('memory_usage', 0) > 1000.0,
            "WARNING",
            cooldown_minutes=15
        )
        
        # 戦略失敗アラート
        self.add_alert_rule(
            "strategy_failure",
            lambda ctx: ctx.get('strategy_failed', False),
            "ERROR",
            cooldown_minutes=5
        )
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                      severity: str = "WARNING", cooldown_minutes: int = 5):
        """アラートルール追加"""
        self.alert_rules[name] = AlertRule(name, condition, severity, cooldown_minutes)
        self.logger.info(f"アラートルール追加: {name}")
    
    def remove_alert_rule(self, name: str):
        """アラートルール削除"""
        if name in self.alert_rules:
            del self.alert_rules[name]
            self.logger.info(f"アラートルール削除: {name}")
    
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に開始されています")
            return
        
        self.monitoring_active = True
        self.monitoring_stats['start_time'] = datetime.now()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("監視エージェント開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.monitoring_stats['start_time']:
            self.monitoring_stats['uptime'] = (
                datetime.now() - self.monitoring_stats['start_time']
            ).total_seconds()
        
        self.logger.info("監視エージェント停止")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                # システム状態取得
                system_context = self._collect_system_context()
                
                # メトリクス記録
                self._record_system_metrics(system_context)
                
                # アラートルール評価
                self._evaluate_alert_rules(system_context)
                
                self.monitoring_stats['rules_evaluated'] += len(self.alert_rules)
                self.monitoring_stats['metrics_collected'] += 1
                
                time.sleep(30)  # 30秒間隔
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(60)  # エラー時は長めの間隔
    
    def _collect_system_context(self) -> Dict[str, Any]:
        """システム状態収集"""
        import psutil
        
        try:
            # システムリソース
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_usage_mb': memory.used / 1024 / 1024,
                'disk_percent': disk.percent,
                'timestamp': datetime.now()
            }
        except ImportError:
            self.logger.warning("psutilライブラリが利用できません")
            return {'timestamp': datetime.now()}
        except Exception as e:
            self.logger.error(f"システム状態収集エラー: {e}")
            return {'timestamp': datetime.now()}
    
    def _record_system_metrics(self, context: Dict[str, Any]):
        """システムメトリクス記録"""
        timestamp = context.get('timestamp', datetime.now())
        
        for metric_name, value in context.items():
            if isinstance(value, (int, float)) and metric_name != 'timestamp':
                self.metrics_collector.add_metric(metric_name, value, timestamp)
    
    def _evaluate_alert_rules(self, context: Dict[str, Any]):
        """アラートルール評価"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule.should_trigger(context):
                    rule.trigger()
                    
                    alert = AlertEvent(
                        timestamp=datetime.now(),
                        event_type=rule_name,
                        severity=rule.severity,
                        message=f"アラートルール '{rule_name}' がトリガーされました",
                        context=context
                    )
                    
                    self.notification_manager.send_alert(alert)
                    self.monitoring_stats['alerts_triggered'] += 1
                    
                    self.logger.warning(f"アラートトリガー: {rule_name}")
                    
            except Exception as e:
                self.logger.error(f"アラートルール評価エラー {rule_name}: {e}")
    
    def report_event(self, event_type: str, severity: str, message: str,
                    context: Optional[Dict[str, Any]] = None,
                    strategy_name: Optional[str] = None):
        """イベント報告"""
        alert = AlertEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            message=message,
            context=context or {},
            strategy_name=strategy_name
        )
        
        self.notification_manager.send_alert(alert)
        self.logger.info(f"イベント報告: {event_type} ({severity})")
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """監視統計取得"""
        stats = self.monitoring_stats.copy()
        
        if stats['start_time'] and self.monitoring_active:
            stats['current_uptime'] = (
                datetime.now() - stats['start_time']
            ).total_seconds()
        
        # アラートルール統計
        rule_stats = {}
        for name, rule in self.alert_rules.items():
            rule_stats[name] = {
                'trigger_count': rule.trigger_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            }
        stats['alert_rules'] = rule_stats
        
        return stats
    
    def create_monitoring_report(self, output_path: Optional[str] = None) -> str:
        """監視レポート生成"""
        if output_path is None:
            output_path = project_root / "logs" / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_path = Path(output_path)
        
        # メトリクス要約
        metrics_summary = {}
        for metric_name in self.metrics_collector.metrics.keys():
            metrics_summary[metric_name] = self.metrics_collector.get_metric_summary(
                metric_name, since=datetime.now() - timedelta(hours=24)
            )
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_statistics': self.get_monitoring_statistics(),
            'metrics_summary': metrics_summary,
            'alert_rules_config': {
                name: {
                    'severity': rule.severity,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'trigger_count': rule.trigger_count
                } for name, rule in self.alert_rules.items()
            }
        }
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"監視レポート生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"監視レポート生成失敗: {e}")
            return ""


# グローバルインスタンス
_global_monitoring_agent: Optional[MonitoringAgent] = None


def get_monitoring_agent() -> MonitoringAgent:
    """グローバル監視エージェント取得"""
    global _global_monitoring_agent
    if _global_monitoring_agent is None:
        _global_monitoring_agent = MonitoringAgent()
    return _global_monitoring_agent


def report_error(error: Exception, context: Optional[Dict[str, Any]] = None,
                strategy_name: Optional[str] = None):
    """エラー報告（グローバル関数）"""
    agent = get_monitoring_agent()
    agent.report_event(
        event_type="error_occurred",
        severity="ERROR",
        message=f"{type(error).__name__}: {str(error)}",
        context=context,
        strategy_name=strategy_name
    )


def report_performance_issue(metric_name: str, value: float, threshold: float,
                           context: Optional[Dict[str, Any]] = None):
    """パフォーマンス問題報告（グローバル関数）"""
    agent = get_monitoring_agent()
    agent.report_event(
        event_type="performance_issue",
        severity="WARNING",
        message=f"{metric_name}: {value} (閾値: {threshold})",
        context=context
    )


def start_system_monitoring():
    """システム監視開始（グローバル関数）"""
    get_monitoring_agent().start_monitoring()


def stop_system_monitoring():
    """システム監視停止（グローバル関数）"""
    get_monitoring_agent().stop_monitoring()
