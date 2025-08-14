"""
拡張アラート管理器 - 取引特化版

既存のalert_manager.pyを拡張し、取引システムに特化した
アラート機能とマルチチャネル通知システムを統合

Author: AI Agent
Date: 2025-01-27
"""

import threading
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import queue
from collections import defaultdict
from enum import Enum

# メール関連のインポートを条件付きに変更
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

from .trading_logger import get_trading_logger, TradingLogger

class TradingAlertLevel(Enum):
    """取引アラートレベル"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradingAlertCategory(Enum):
    """取引アラートカテゴリ"""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    MARKET = "market"
    ORDER = "order"
    POSITION = "position"
    CONNECTIVITY = "connectivity"

class NotificationChannel(Enum):
    """通知チャネル"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    FILE = "file"
    CONSOLE = "console"

@dataclass
class TradingAlert:
    """取引アラート"""
    id: str
    level: TradingAlertLevel
    category: TradingAlertCategory
    title: str
    message: str
    timestamp: datetime
    source: str
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    order_id: Optional[str] = None
    position: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    notification_sent: bool = False

@dataclass
class AlertRule:
    """アラートルール"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: TradingAlertLevel
    category: TradingAlertCategory
    cooldown_seconds: int = 300  # 5分
    enabled: bool = True
    last_triggered: Optional[datetime] = None

@dataclass
class NotificationConfig:
    """通知設定"""
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    email_config: Dict[str, str] = field(default_factory=dict)
    webhook_config: Dict[str, str] = field(default_factory=dict)
    file_config: Dict[str, str] = field(default_factory=dict)

@dataclass
class MonitoringMetrics:
    """監視メトリクス"""
    timestamp: datetime
    metric_name: str
    value: float
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedAlertManager:
    """拡張アラート管理器 - 取引特化版"""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        logger: Optional[TradingLogger] = None
    ):
        """
        Args:
            config_file: 設定ファイルパス
            logger: 取引ロガー
        """
        self.logger = logger or get_trading_logger()
        self.config_file = config_file
        
        # アラート管理
        self.alerts: Dict[str, TradingAlert] = {}
        self.alert_history: List[TradingAlert] = []
        self.max_history = 10000
        
        # ルール管理
        self.rules: Dict[str, AlertRule] = {}
        
        # 通知設定
        self.notification_config = NotificationConfig()
        
        # メトリクス管理
        self.metrics_buffer = queue.Queue(maxsize=1000)
        self.metrics_history = defaultdict(list)
        
        # スレッド管理
        self.is_running = False
        self.processor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 統計
        self.stats = {
            'total_alerts': 0,
            'alerts_by_level': defaultdict(int),
            'alerts_by_category': defaultdict(int),
            'notifications_sent': 0,
            'rules_triggered': defaultdict(int)
        }
        
        # 設定読み込み
        if config_file:
            self.load_config(config_file)
        
        # デフォルトルール設定
        self._setup_default_rules()
        
        self.logger.info("EnhancedAlertManager initialized", component="enhanced_alert_manager")
    
    def _setup_default_rules(self):
        """デフォルトアラートルール設定"""
        
        # 高エラー率アラート
        self.add_rule(
            "high_error_rate",
            lambda data: data.get('error_rate', 0) > 0.1,
            TradingAlertLevel.HIGH,
            TradingAlertCategory.SYSTEM,
            cooldown_seconds=600  # 10分
        )
        
        # API制限アラート
        self.add_rule(
            "api_limit_warning",
            lambda data: data.get('api_usage_rate', 0) > 0.8,
            TradingAlertLevel.MEDIUM,
            TradingAlertCategory.SYSTEM,
            cooldown_seconds=3600  # 1時間
        )
        
        # 大きなPnL変動アラート
        self.add_rule(
            "large_pnl_change",
            lambda data: abs(data.get('pnl_change', 0)) > 100000,  # 10万円
            TradingAlertLevel.HIGH,
            TradingAlertCategory.TRADING,
            cooldown_seconds=300  # 5分
        )
        
        # ポジション制限アラート
        self.add_rule(
            "position_limit_warning",
            lambda data: data.get('position_utilization', 0) > 0.9,
            TradingAlertLevel.MEDIUM,
            TradingAlertCategory.RISK,
            cooldown_seconds=600  # 10分
        )
        
        # 接続エラーアラート
        self.add_rule(
            "connection_error",
            lambda data: data.get('connection_errors', 0) > 3,
            TradingAlertLevel.CRITICAL,
            TradingAlertCategory.CONNECTIVITY,
            cooldown_seconds=60  # 1分
        )
        
        # 注文失敗アラート
        self.add_rule(
            "order_failure",
            lambda data: data.get('order_failures', 0) > 2,
            TradingAlertLevel.HIGH,
            TradingAlertCategory.ORDER,
            cooldown_seconds=300  # 5分
        )
        
        self.logger.info("Default alert rules configured", component="enhanced_alert_manager")
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: TradingAlertLevel,
        category: TradingAlertCategory,
        cooldown_seconds: int = 300,
        enabled: bool = True
    ):
        """アラートルールを追加"""
        rule = AlertRule(
            name=name,
            condition=condition,
            level=level,
            category=category,
            cooldown_seconds=cooldown_seconds,
            enabled=enabled
        )
        
        self.rules[name] = rule
        self.logger.info(f"Alert rule added: {name}", component="enhanced_alert_manager")
    
    def remove_rule(self, name: str):
        """アラートルールを削除"""
        if name in self.rules:
            del self.rules[name]
            self.logger.info(f"Alert rule removed: {name}", component="enhanced_alert_manager")
    
    def add_metrics(
        self,
        metric_name: str,
        value: float,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        **metadata
    ):
        """メトリクスを追加"""
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            symbol=symbol,
            strategy=strategy,
            metadata=metadata
        )
        
        try:
            self.metrics_buffer.put_nowait(metrics)
        except queue.Full:
            self.logger.warning("Metrics buffer full, dropping metrics", component="enhanced_alert_manager")
    
    def create_alert(
        self,
        level: TradingAlertLevel,
        category: TradingAlertCategory,
        title: str,
        message: str,
        source: str = "system",
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> TradingAlert:
        """アラートを作成"""
        alert_id = f"{category.value}_{level.value}_{int(time.time())}"
        
        alert = TradingAlert(
            id=alert_id,
            level=level,
            category=category,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            symbol=symbol,
            strategy=strategy,
            order_id=kwargs.get('order_id'),
            position=kwargs.get('position'),
            pnl=kwargs.get('pnl'),
            metadata=kwargs
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 履歴サイズ制限
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history//2:]
        
        # 統計更新
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_level'][level.value] += 1
        self.stats['alerts_by_category'][category.value] += 1
        
        # ログ記録
        self.logger.log_alert(
            category.value,
            f"{title}: {message}",
            priority=level.value.upper(),
            symbol=symbol,
            strategy=strategy,
            component="enhanced_alert_manager"
        )
        
        # 通知送信
        self._send_notification(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """アラートを確認済みにする"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.logger.info(
                f"Alert acknowledged: {alert_id} by {user}",
                component="enhanced_alert_manager"
            )
    
    def resolve_alert(self, alert_id: str, user: str = "system"):
        """アラートを解決済みにする"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.logger.info(
                f"Alert resolved: {alert_id} by {user}",
                component="enhanced_alert_manager"
            )
    
    def start_processing(self):
        """アラート処理開始"""
        if self.is_running:
            self.logger.warning("Alert manager is already running", component="enhanced_alert_manager")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processor_thread.start()
        
        self.logger.info("Alert processing started", component="enhanced_alert_manager")
    
    def stop_processing(self):
        """アラート処理停止"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        
        self.logger.info("Alert processing stopped", component="enhanced_alert_manager")
    
    def _processing_loop(self):
        """処理ループ"""
        self.logger.info("Alert processing loop started", component="enhanced_alert_manager")
        
        while not self.stop_event.is_set():
            try:
                # メトリクス処理
                try:
                    metrics = self.metrics_buffer.get(timeout=1.0)
                    self._process_metrics(metrics)
                except queue.Empty:
                    continue
                
                # ルール評価
                self._evaluate_rules()
                
                # アラートクリーンアップ
                self._cleanup_old_alerts()
                
            except Exception as e:
                self.logger.error(
                    f"Error in alert processing loop: {e}",
                    component="enhanced_alert_manager"
                )
                time.sleep(1.0)
    
    def _process_metrics(self, metrics: MonitoringMetrics):
        """メトリクス処理"""
        # 履歴に追加
        key = f"{metrics.metric_name}_{metrics.symbol}_{metrics.strategy}"
        self.metrics_history[key].append(metrics)
        
        # 履歴サイズ制限
        if len(self.metrics_history[key]) > 1000:
            self.metrics_history[key] = self.metrics_history[key][-500:]
        
        self.logger.trace(
            f"Processed metrics: {metrics.metric_name}={metrics.value}",
            component="enhanced_alert_manager"
        )
    
    def _evaluate_rules(self):
        """ルール評価"""
        current_time = datetime.now()
        
        # 集約メトリクスを準備
        aggregated_data = self._prepare_aggregated_data()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # クールダウンチェック
            if (rule.last_triggered and
                current_time - rule.last_triggered < timedelta(seconds=rule.cooldown_seconds)):
                continue
            
            try:
                # ルール条件評価
                if rule.condition(aggregated_data):
                    # アラート作成
                    self.create_alert(
                        level=rule.level,
                        category=rule.category,
                        title=f"Rule Triggered: {rule_name}",
                        message=f"Alert rule '{rule_name}' has been triggered",
                        source="rule_engine"
                    )
                    
                    rule.last_triggered = current_time
                    self.stats['rules_triggered'][rule_name] += 1
                    
            except Exception as e:
                self.logger.error(
                    f"Error evaluating rule {rule_name}: {e}",
                    component="enhanced_alert_manager"
                )
    
    def _prepare_aggregated_data(self) -> Dict[str, Any]:
        """集約データ準備"""
        data = {}
        current_time = datetime.now()
        
        # 最新メトリクスから集約データを作成
        for key, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue
            
            # 最近5分間のメトリクス
            recent_metrics = [
                m for m in metrics_list
                if current_time - m.timestamp < timedelta(minutes=5)
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                latest = recent_metrics[-1]
                
                metric_name = latest.metric_name
                data[metric_name] = latest.value
                data[f"{metric_name}_avg"] = sum(values) / len(values)
                data[f"{metric_name}_max"] = max(values)
                data[f"{metric_name}_min"] = min(values)
        
        return data
    
    def _cleanup_old_alerts(self):
        """古いアラートクリーンアップ"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        # 24時間以上古い解決済みアラートを削除
        alerts_to_remove = []
        for alert_id, alert in self.alerts.items():
            if (alert.resolved and alert.timestamp < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        if alerts_to_remove:
            self.logger.debug(
                f"Cleaned up {len(alerts_to_remove)} old alerts",
                component="enhanced_alert_manager"
            )
    
    def _send_notification(self, alert: TradingAlert):
        """通知送信"""
        if not self.notification_config.enabled:
            return
        
        try:
            for channel in self.notification_config.channels:
                if channel == NotificationChannel.LOG:
                    self._send_log_notification(alert)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif channel == NotificationChannel.FILE:
                    self._send_file_notification(alert)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console_notification(alert)
            
            alert.notification_sent = True
            self.stats['notifications_sent'] += 1
            
        except Exception as e:
            self.logger.error(
                f"Failed to send notification for alert {alert.id}: {e}",
                component="enhanced_alert_manager"
            )
    
    def _send_log_notification(self, alert: TradingAlert):
        """ログ通知"""
        level_map = {
            TradingAlertLevel.INFO: 'INFO',
            TradingAlertLevel.LOW: 'INFO',
            TradingAlertLevel.MEDIUM: 'WARNING',
            TradingAlertLevel.HIGH: 'ERROR',
            TradingAlertLevel.CRITICAL: 'CRITICAL'
        }
        
        log_level = level_map.get(alert.level, 'INFO')
        self.logger.log(
            log_level,
            f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.message}",
            category="ALERT",
            symbol=alert.symbol,
            strategy=alert.strategy,
            component="enhanced_alert_manager"
        )
    
    def _send_email_notification(self, alert: TradingAlert):
        """メール通知"""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email libraries not available", component="enhanced_alert_manager")
            return
            
        email_config = self.notification_config.email_config
        if not email_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_address')
            msg['To'] = email_config.get('to_address')
            msg['Subject'] = f"Trading Alert [{alert.level.value.upper()}]: {alert.title}"
            
            body = f"""
Trading Alert Notification

Level: {alert.level.value.upper()}
Category: {alert.category.value}
Title: {alert.title}
Message: {alert.message}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

Additional Information:
Symbol: {alert.symbol or 'N/A'}
Strategy: {alert.strategy or 'N/A'}
Order ID: {alert.order_id or 'N/A'}
Position: {alert.position or 'N/A'}
PnL: {alert.pnl or 'N/A'}

Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config.get('smtp_host'), email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username'), email_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.id}", component="enhanced_alert_manager")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}", component="enhanced_alert_manager")
    
    def _send_webhook_notification(self, alert: TradingAlert):
        """Webhook通知"""
        webhook_config = self.notification_config.webhook_config
        if not webhook_config:
            return
        
        try:
            payload = {
                'alert_id': alert.id,
                'level': alert.level.value,
                'category': alert.category.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'symbol': alert.symbol,
                'strategy': alert.strategy,
                'order_id': alert.order_id,
                'position': alert.position,
                'pnl': alert.pnl,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                webhook_config.get('url'),
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.id}", component="enhanced_alert_manager")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}", component="enhanced_alert_manager")
    
    def _send_file_notification(self, alert: TradingAlert):
        """ファイル通知"""
        file_config = self.notification_config.file_config
        if not file_config:
            return
        
        try:
            file_path = file_config.get('path', 'alerts.txt')
            
            alert_line = f"{alert.timestamp.isoformat()},{alert.level.value},{alert.category.value},\"{alert.title}\",\"{alert.message}\",{alert.source},{alert.symbol},{alert.strategy},{alert.id}\n"
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(alert_line)
            
            self.logger.info(f"File notification written for alert {alert.id}", component="enhanced_alert_manager")
            
        except Exception as e:
            self.logger.error(f"Failed to write file notification: {e}", component="enhanced_alert_manager")
    
    def _send_console_notification(self, alert: TradingAlert):
        """コンソール通知"""
        level_colors = {
            TradingAlertLevel.INFO: '\033[36m',      # シアン
            TradingAlertLevel.LOW: '\033[32m',       # 緑
            TradingAlertLevel.MEDIUM: '\033[33m',    # 黄
            TradingAlertLevel.HIGH: '\033[31m',      # 赤
            TradingAlertLevel.CRITICAL: '\033[35m'   # マゼンタ
        }
        
        color = level_colors.get(alert.level, '')
        reset = '\033[0m'
        
        print(f"{color}[ALERT {alert.level.value.upper()}] {alert.title}: {alert.message}{reset}")
    
    def get_active_alerts(self, level: Optional[TradingAlertLevel] = None) -> List[TradingAlert]:
        """アクティブアラート取得"""
        alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計取得"""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': self.stats['total_alerts'],
            'active_alerts': len(active_alerts),
            'acknowledged_alerts': len([a for a in active_alerts if a.acknowledged]),
            'unacknowledged_alerts': len([a for a in active_alerts if not a.acknowledged]),
            'alerts_by_level': dict(self.stats['alerts_by_level']),
            'alerts_by_category': dict(self.stats['alerts_by_category']),
            'notifications_sent': self.stats['notifications_sent'],
            'rules_triggered': dict(self.stats['rules_triggered']),
            'rules_count': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled])
        }
    
    def load_config(self, config_file: str):
        """設定ファイル読み込み"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 通知設定
            if 'notification' in config:
                notification_config = config['notification']
                self.notification_config.enabled = notification_config.get('enabled', True)
                
                channels = notification_config.get('channels', [])
                self.notification_config.channels = [NotificationChannel(ch) for ch in channels]
                
                self.notification_config.email_config = notification_config.get('email', {})
                self.notification_config.webhook_config = notification_config.get('webhook', {})
                self.notification_config.file_config = notification_config.get('file', {})
            
            self.logger.info(f"Configuration loaded from {config_file}", component="enhanced_alert_manager")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}", component="enhanced_alert_manager")
    
    def save_config(self, config_file: str):
        """設定ファイル保存"""
        config = {
            'notification': {
                'enabled': self.notification_config.enabled,
                'channels': [ch.value for ch in self.notification_config.channels],
                'email': self.notification_config.email_config,
                'webhook': self.notification_config.webhook_config,
                'file': self.notification_config.file_config
            }
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {config_file}", component="enhanced_alert_manager")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}", component="enhanced_alert_manager")

# グローバルインスタンス
_enhanced_alert_manager: Optional[EnhancedAlertManager] = None

def get_enhanced_alert_manager() -> EnhancedAlertManager:
    """グローバル拡張アラート管理器取得"""
    global _enhanced_alert_manager
    if _enhanced_alert_manager is None:
        _enhanced_alert_manager = EnhancedAlertManager()
    return _enhanced_alert_manager

def setup_enhanced_alert_manager(
    config_file: Optional[str] = None,
    logger: Optional[TradingLogger] = None
) -> EnhancedAlertManager:
    """拡張アラート管理器設定"""
    global _enhanced_alert_manager
    _enhanced_alert_manager = EnhancedAlertManager(config_file, logger)
    return _enhanced_alert_manager
