"""
フェーズ3B アラート管理モジュール

このモジュールは、リアルタイムデータシステムのアラート生成、
管理、エスカレーション、通知機能を提供します。
"""

import smtplib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

# プロジェクト内インポート
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logger_config import setup_logger
from src.data.data_feed_integration import DataQualityLevel
from src.error_handling.exception_handler import UnifiedExceptionHandler


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """アラート状態"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCategory(Enum):
    """アラートカテゴリ"""
    DATA_QUALITY = "data_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    NETWORK = "network"
    CACHE = "cache"
    ERROR = "error"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class AlertRule:
    """アラートルール定義"""
    rule_id: str
    name: str
    category: AlertCategory
    level: AlertLevel
    condition: str  # 条件式
    threshold: float
    time_window_minutes: int
    description: str
    enabled: bool = True
    suppression_minutes: int = 60  # 同じアラートの抑制時間
    escalation_minutes: int = 30   # エスカレーション時間
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['category'] = self.category.value
        data['level'] = self.level.value
        return data


@dataclass
class Alert:
    """アラート"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    level: AlertLevel
    category: AlertCategory
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['level'] = self.level.value
        data['category'] = self.category.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.escalated_at:
            data['escalated_at'] = self.escalated_at.isoformat()
        return data


@dataclass
class NotificationChannel:
    """通知チャネル"""
    channel_id: str
    name: str
    type: str  # email, slack, webhook, sms
    config: Dict[str, Any]
    enabled: bool = True
    alert_levels: List[AlertLevel] = None
    
    def __post_init__(self):
        if self.alert_levels is None:
            self.alert_levels = [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]


class AlertEvaluator:
    """アラート条件評価器"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.AlertEvaluator")
        
    def evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """条件評価"""
        try:
            # 安全な評価環境
            safe_globals = {
                "__builtins__": {},
                "abs": abs,
                "min": min,
                "max": max,
                "len": len,
                "sum": sum,
                "avg": lambda x: sum(x) / len(x) if x else 0,
                "and": lambda x, y: x and y,
                "or": lambda x, y: x or y,
                "not": lambda x: not x
            }
            
            # メトリクスを評価環境に追加
            safe_globals.update(metrics)
            
            # 条件評価
            result = eval(condition, safe_globals, {})
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
            
    def evaluate_quality_rules(self, quality_metrics: Dict[str, Any], 
                             rules: List[AlertRule]) -> List[str]:
        """品質ルール評価"""
        triggered_rules = []
        
        for rule in rules:
            if (rule.enabled and 
                rule.category == AlertCategory.DATA_QUALITY and
                self.evaluate_condition(rule.condition, quality_metrics)):
                triggered_rules.append(rule.rule_id)
                
        return triggered_rules
        
    def evaluate_performance_rules(self, performance_metrics: Dict[str, Any],
                                 rules: List[AlertRule]) -> List[str]:
        """パフォーマンスルール評価"""
        triggered_rules = []
        
        for rule in rules:
            if (rule.enabled and 
                rule.category == AlertCategory.SYSTEM_PERFORMANCE and
                self.evaluate_condition(rule.condition, performance_metrics)):
                triggered_rules.append(rule.rule_id)
                
        return triggered_rules


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(f"{__name__}.NotificationManager")
        self.channels: Dict[str, NotificationChannel] = {}
        
        # 通知統計
        self.notification_stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': defaultdict(int),
            'by_level': defaultdict(int)
        }
        
    def add_channel(self, channel: NotificationChannel):
        """通知チャネル追加"""
        self.channels[channel.channel_id] = channel
        self.logger.info(f"Added notification channel: {channel.name}")
        
    def remove_channel(self, channel_id: str):
        """通知チャネル削除"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            self.logger.info(f"Removed notification channel: {channel_id}")
            
    def send_alert_notification(self, alert: Alert) -> bool:
        """アラート通知送信"""
        success = True
        
        for channel_id, channel in self.channels.items():
            if (channel.enabled and 
                alert.level in channel.alert_levels):
                
                try:
                    if channel.type == "email":
                        self._send_email_notification(alert, channel)
                    elif channel.type == "slack":
                        self._send_slack_notification(alert, channel)
                    elif channel.type == "webhook":
                        self._send_webhook_notification(alert, channel)
                    else:
                        self.logger.warning(f"Unknown notification type: {channel.type}")
                        continue
                        
                    self.notification_stats['total_sent'] += 1
                    self.notification_stats['by_channel'][channel_id] += 1
                    self.notification_stats['by_level'][alert.level.value] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to send notification via {channel.name}: {e}")
                    self.notification_stats['total_failed'] += 1
                    success = False
                    
        return success
        
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """メール通知送信"""
        try:
            smtp_config = channel.config
            
            # メール作成
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email']
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # メール本文
            body = f"""
アラートが発生しました:

タイトル: {alert.title}
レベル: {alert.level.value.upper()}
カテゴリ: {alert.category.value}
ソース: {alert.source}
時刻: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

メッセージ:
{alert.message}

詳細情報:
{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}

---
リアルタイムデータ監視システム
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # SMTP送信
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
                
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            raise
            
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Slack通知送信"""
        try:
            import requests
            
            slack_config = channel.config
            webhook_url = slack_config['webhook_url']
            
            # カラーマッピング
            color_map = {
                AlertLevel.INFO: "#36a64f",      # 緑
                AlertLevel.WARNING: "#ffcc00",   # 黄
                AlertLevel.ERROR: "#ff6600",     # オレンジ
                AlertLevel.CRITICAL: "#ff0000"   # 赤
            }
            
            # Slackメッセージ作成
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.level, "#cccccc"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "レベル", "value": alert.level.value.upper(), "short": True},
                        {"title": "カテゴリ", "value": alert.category.value, "short": True},
                        {"title": "ソース", "value": alert.source, "short": True},
                        {"title": "時刻", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "リアルタイムデータ監視システム",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            raise
            
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Webhook通知送信"""
        try:
            import requests
            
            webhook_config = channel.config
            webhook_url = webhook_config['url']
            
            # ペイロード作成
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "source": "realtime_data_monitoring"
            }
            
            # ヘッダー設定
            headers = {'Content-Type': 'application/json'}
            if webhook_config.get('auth_header'):
                headers['Authorization'] = webhook_config['auth_header']
                
            response = requests.post(
                webhook_url, 
                json=payload, 
                headers=headers,
                timeout=webhook_config.get('timeout', 10)
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            raise


class AlertManager:
    """アラート管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = setup_logger(__name__)
        self.exception_handler = UnifiedExceptionHandler()
        
        # アラートルール
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # アクティブアラート
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=self.config.get('max_history', 10000))
        
        # 抑制中のアラート（重複防止）
        self.suppressed_alerts: Dict[str, datetime] = {}
        
        # 評価器と通知管理器
        self.evaluator = AlertEvaluator()
        self.notification_manager = NotificationManager(self.config.get('notifications', {}))
        
        # 評価スレッド
        self.evaluation_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # 統計
        self.alert_stats = {
            'total_generated': 0,
            'total_resolved': 0,
            'by_level': defaultdict(int),
            'by_category': defaultdict(int),
            'avg_resolution_time': 0.0
        }
        
        # デフォルトルール設定
        self._setup_default_rules()
        
        self.logger.info("Alert manager initialized")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'max_history': 10000,
            'evaluation_interval': 30,  # 30秒間隔
            'auto_resolution_minutes': 60,
            'escalation_enabled': True,
            'notifications': {
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'from_email': 'alerts@monitoring.local'
            }
        }
        
    def _setup_default_rules(self):
        """デフォルトアラートルール設定"""
        default_rules = [
            # データ品質アラート
            AlertRule(
                rule_id="quality_score_low",
                name="データ品質スコア低下",
                category=AlertCategory.DATA_QUALITY,
                level=AlertLevel.WARNING,
                condition="overall_score < 0.7",
                threshold=0.7,
                time_window_minutes=5,
                description="データ品質の総合スコアが閾値を下回りました"
            ),
            AlertRule(
                rule_id="quality_score_critical",
                name="データ品質スコア危険レベル",
                category=AlertCategory.DATA_QUALITY,
                level=AlertLevel.CRITICAL,
                condition="overall_score < 0.5",
                threshold=0.5,
                time_window_minutes=5,
                description="データ品質の総合スコアが危険レベルです"
            ),
            
            # パフォーマンスアラート
            AlertRule(
                rule_id="response_time_high",
                name="応答時間遅延",
                category=AlertCategory.SYSTEM_PERFORMANCE,
                level=AlertLevel.WARNING,
                condition="avg_response_time > 1000",
                threshold=1000.0,
                time_window_minutes=5,
                description="平均応答時間が1秒を超えています"
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="エラー率高騰",
                category=AlertCategory.SYSTEM_PERFORMANCE,
                level=AlertLevel.ERROR,
                condition="error_rate > 0.1",
                threshold=0.1,
                time_window_minutes=5,
                description="エラー率が10%を超えています"
            ),
            
            # ネットワークアラート
            AlertRule(
                rule_id="network_timeout_high",
                name="ネットワークタイムアウト多発",
                category=AlertCategory.NETWORK,
                level=AlertLevel.ERROR,
                condition="timeout_rate > 0.05",
                threshold=0.05,
                time_window_minutes=10,
                description="ネットワークタイムアウトが多発しています"
            ),
            
            # キャッシュアラート
            AlertRule(
                rule_id="cache_hit_rate_low",
                name="キャッシュヒット率低下",
                category=AlertCategory.CACHE,
                level=AlertLevel.WARNING,
                condition="cache_hit_rate < 0.8",
                threshold=0.8,
                time_window_minutes=15,
                description="キャッシュヒット率が低下しています"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
            
    def add_alert_rule(self, rule: AlertRule):
        """アラートルール追加"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
        
    def remove_alert_rule(self, rule_id: str):
        """アラートルール削除"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
            
    def enable_alert_rule(self, rule_id: str):
        """アラートルール有効化"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            self.logger.info(f"Enabled alert rule: {rule_id}")
            
    def disable_alert_rule(self, rule_id: str):
        """アラートルール無効化"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            self.logger.info(f"Disabled alert rule: {rule_id}")
            
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """メトリクス評価とアラート生成"""
        try:
            triggered_rules = []
            
            # 品質メトリクス評価
            if 'quality' in metrics:
                quality_rules = [r for r in self.alert_rules.values() 
                               if r.category == AlertCategory.DATA_QUALITY]
                triggered_rules.extend(
                    self.evaluator.evaluate_quality_rules(metrics['quality'], quality_rules)
                )
                
            # パフォーマンスメトリクス評価
            if 'performance' in metrics:
                performance_rules = [r for r in self.alert_rules.values() 
                                   if r.category == AlertCategory.SYSTEM_PERFORMANCE]
                triggered_rules.extend(
                    self.evaluator.evaluate_performance_rules(metrics['performance'], performance_rules)
                )
                
            # ネットワークメトリクス評価
            if 'network' in metrics:
                network_rules = [r for r in self.alert_rules.values()
                               if r.category == AlertCategory.NETWORK]
                for rule in network_rules:
                    if (rule.enabled and 
                        self.evaluator.evaluate_condition(rule.condition, metrics['network'])):
                        triggered_rules.append(rule.rule_id)
                        
            # トリガーされたルールからアラート生成
            for rule_id in triggered_rules:
                self._generate_alert(rule_id, metrics)
                
        except Exception as e:
            self.logger.error(f"Error evaluating metrics: {e}")
            self.exception_handler.handle_system_error(
                e, context={'operation': 'evaluate_metrics'}
            )
            
    def _generate_alert(self, rule_id: str, metrics: Dict[str, Any]):
        """アラート生成"""
        try:
            rule = self.alert_rules.get(rule_id)
            if not rule:
                return
                
            # 抑制チェック
            suppression_key = f"{rule_id}_{rule.category.value}"
            if suppression_key in self.suppressed_alerts:
                suppression_time = self.suppressed_alerts[suppression_key]
                if datetime.now() < suppression_time:
                    return  # まだ抑制中
                else:
                    del self.suppressed_alerts[suppression_key]
                    
            # アラートID生成
            alert_id = f"{rule_id}_{int(datetime.now().timestamp())}"
            
            # アラート作成
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule_id,
                title=rule.name,
                message=rule.description,
                level=rule.level,
                category=rule.category,
                source="realtime_data_monitoring",
                timestamp=datetime.now(),
                metadata=metrics
            )
            
            # アクティブアラートに追加
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # 統計更新
            self.alert_stats['total_generated'] += 1
            self.alert_stats['by_level'][alert.level.value] += 1
            self.alert_stats['by_category'][alert.category.value] += 1
            
            # 通知送信
            if self.notification_manager:
                self.notification_manager.send_alert_notification(alert)
                
            # 抑制設定
            suppression_until = datetime.now() + timedelta(minutes=rule.suppression_minutes)
            self.suppressed_alerts[suppression_key] = suppression_until
            
            self.logger.warning(f"Alert generated: {alert.title} (ID: {alert_id})")
            
        except Exception as e:
            self.logger.error(f"Error generating alert for rule {rule_id}: {e}")
            
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """アラート確認"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
            
    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                # アクティブリストから削除
                del self.active_alerts[alert_id]
                
                # 統計更新
                self.alert_stats['total_resolved'] += 1
                
                # 解決時間計算
                if alert.acknowledged_at:
                    resolution_time = (alert.resolved_at - alert.acknowledged_at).total_seconds()
                    current_avg = self.alert_stats['avg_resolution_time']
                    total_resolved = self.alert_stats['total_resolved']
                    new_avg = ((current_avg * (total_resolved - 1)) + resolution_time) / total_resolved
                    self.alert_stats['avg_resolution_time'] = new_avg
                    
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
            
    def suppress_alert(self, alert_id: str, suppression_minutes: int = 60) -> bool:
        """アラート抑制"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                
                # 抑制期間設定
                rule = self.alert_rules.get(alert.rule_id)
                if rule:
                    suppression_key = f"{alert.rule_id}_{alert.category.value}"
                    suppression_until = datetime.now() + timedelta(minutes=suppression_minutes)
                    self.suppressed_alerts[suppression_key] = suppression_until
                    
                self.logger.info(f"Alert suppressed: {alert_id} for {suppression_minutes} minutes")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error suppressing alert {alert_id}: {e}")
            return False
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
        
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """アラート履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert.to_dict() for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
    def get_alert_stats(self) -> Dict[str, Any]:
        """アラート統計取得"""
        return {
            **self.alert_stats,
            'active_count': len(self.active_alerts),
            'suppressed_count': len(self.suppressed_alerts),
            'notification_stats': self.notification_manager.notification_stats
        }
        
    def start_monitoring(self):
        """監視開始"""
        if self.is_running:
            self.logger.warning("Alert monitoring is already running")
            return
            
        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.evaluation_thread.start()
        self.logger.info("Alert monitoring started")
        
    def stop_monitoring(self):
        """監視停止"""
        self.is_running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
        self.logger.info("Alert monitoring stopped")
        
    def _monitoring_loop(self):
        """監視ループ"""
        while self.is_running:
            try:
                # 自動解決チェック
                self._check_auto_resolution()
                
                # エスカレーションチェック
                if self.config.get('escalation_enabled', True):
                    self._check_escalation()
                    
                # 抑制期間終了チェック
                self._cleanup_suppressions()
                
                time.sleep(self.config.get('evaluation_interval', 30))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def _check_auto_resolution(self):
        """自動解決チェック"""
        try:
            auto_resolution_minutes = self.config.get('auto_resolution_minutes', 60)
            cutoff_time = datetime.now() - timedelta(minutes=auto_resolution_minutes)
            
            alerts_to_resolve = []
            for alert_id, alert in self.active_alerts.items():
                if (alert.status == AlertStatus.ACKNOWLEDGED and 
                    alert.acknowledged_at and 
                    alert.acknowledged_at < cutoff_time):
                    alerts_to_resolve.append(alert_id)
                    
            for alert_id in alerts_to_resolve:
                self.resolve_alert(alert_id)
                self.logger.info(f"Auto-resolved alert: {alert_id}")
                
        except Exception as e:
            self.logger.error(f"Error in auto-resolution check: {e}")
            
    def _check_escalation(self):
        """エスカレーションチェック"""
        try:
            for alert_id, alert in self.active_alerts.items():
                if alert.status == AlertStatus.ACTIVE and not alert.escalated_at:
                    rule = self.alert_rules.get(alert.rule_id)
                    if rule:
                        escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_minutes)
                        if datetime.now() > escalation_time:
                            self._escalate_alert(alert)
                            
        except Exception as e:
            self.logger.error(f"Error in escalation check: {e}")
            
    def _escalate_alert(self, alert: Alert):
        """アラートエスカレーション"""
        try:
            alert.escalated_at = datetime.now()
            
            # エスカレーション通知
            escalation_alert = Alert(
                alert_id=f"{alert.alert_id}_escalated",
                rule_id=alert.rule_id,
                title=f"エスカレーション: {alert.title}",
                message=f"アラートが未解決のままです。\n\n元のアラート: {alert.message}",
                level=AlertLevel.CRITICAL,
                category=alert.category,
                source=alert.source,
                timestamp=datetime.now(),
                metadata=alert.metadata
            )
            
            self.notification_manager.send_alert_notification(escalation_alert)
            self.logger.warning(f"Alert escalated: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.alert_id}: {e}")
            
    def _cleanup_suppressions(self):
        """抑制期間終了のクリーンアップ"""
        try:
            current_time = datetime.now()
            expired_suppressions = [
                key for key, expiry_time in self.suppressed_alerts.items()
                if current_time > expiry_time
            ]
            
            for key in expired_suppressions:
                del self.suppressed_alerts[key]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up suppressions: {e}")


if __name__ == "__main__":
    # テスト用デモ
    alert_manager = AlertManager()
    
    # テスト用通知チャネル設定
    email_channel = NotificationChannel(
        channel_id="email_admin",
        name="管理者メール",
        type="email",
        config={
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'from_email': 'alerts@test.local',
            'to_email': 'admin@test.local'
        },
        alert_levels=[AlertLevel.ERROR, AlertLevel.CRITICAL]
    )
    alert_manager.notification_manager.add_channel(email_channel)
    
    # テスト用メトリクス
    test_metrics = {
        'quality': {
            'overall_score': 0.6,  # 低品質でアラートトリガー
            'completeness_score': 0.8,
            'accuracy_score': 0.5,
            'timeliness_score': 0.7,
            'consistency_score': 0.6
        },
        'performance': {
            'avg_response_time': 1200,  # 高遅延でアラートトリガー
            'error_rate': 0.05,
            'success_rate': 0.95
        }
    }
    
    # メトリクス評価
    alert_manager.evaluate_metrics(test_metrics)
    
    # アクティブアラート表示
    active_alerts = alert_manager.get_active_alerts()
    print(f"Generated {len(active_alerts)} alerts:")
    for alert in active_alerts:
        print(f"- {alert['title']} ({alert['level']})")
        
    # 統計表示
    stats = alert_manager.get_alert_stats()
    print(f"\nAlert stats: {json.dumps(stats, indent=2)}")
