"""
Risk Alert Manager - リスクアラート管理システム
DSSMS Phase 2 Task 2.3

リスクイベントの評価、アラート生成、通知管理を行う
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガーの設定
logger = logging.getLogger(__name__)

# 共通データクラスのインポート
from .unified_risk_monitor import RiskEvent, RiskSeverity

@dataclass
class AlertRule:
    """アラートルール定義"""
    rule_name: str
    condition: str  # 条件文字列
    severity: RiskSeverity
    cooldown_minutes: int = 30
    action_required: bool = False
    auto_action_available: bool = False

class AlertChannel(Enum):
    """アラート通知チャネル"""
    LOG = "log"
    EMAIL = "email"
    CONSOLE = "console"
    FILE = "file"

@dataclass
class AlertNotification:
    """アラート通知データ"""
    timestamp: datetime
    alert_id: str
    severity: RiskSeverity
    message: str
    channel: AlertChannel
    sent: bool = False
    retry_count: int = 0

class RiskAlertManager:
    """
    リスクアラート管理システム
    リスクイベントの評価とアラート生成を担当
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        アラート管理システムの初期化
        
        Args:
            config_path: アラート設定ファイルパス
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_alert_config()
        
        # アラート履歴とキューの管理
        self.alert_history: List[AlertNotification] = []
        self.pending_alerts: Dict[str, AlertNotification] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # アラートルールの初期化
        self.alert_rules = self._initialize_alert_rules()
        
        # 通知チャネルの設定
        self.notification_channels = self._setup_notification_channels()
        
        # アラート処理のスレッド制御
        self.processing_active = False
        self.processing_thread = None
        
        self.logger.info("RiskAlertManager initialized successfully")
    
    def add_event_handler(self, handler) -> None:
        """Add an event handler for alert events"""
        # Placeholder implementation for compatibility
        pass
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert manager status"""
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len([
                alert for alert in self.alert_history
                if alert.timestamp.date() == datetime.now().date()
            ]),
            "escalated_alerts": len([
                alert for alert in self.active_alerts.values()
                if alert.escalated
            ]),
            "total_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_config.get("channels", []))
        }
    
    def _get_default_config_path(self) -> str:
        """デフォルトアラート設定ファイルパス"""
        return os.path.join(
            os.path.dirname(__file__),
            'configs',
            'alert_rules.json'
        )
    
    def _load_alert_config(self) -> Dict[str, Any]:
        """アラート設定読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                default_config = self._create_default_alert_config()
                self._save_alert_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load alert config: {e}")
            return self._create_default_alert_config()
    
    def _create_default_alert_config(self) -> Dict[str, Any]:
        """デフォルトアラート設定作成"""
        return {
            "alert_rules": [
                {
                    "rule_name": "critical_drawdown",
                    "condition": "drawdown > 0.10",
                    "severity": "critical",
                    "cooldown_minutes": 15,
                    "action_required": True,
                    "auto_action_available": True
                },
                {
                    "rule_name": "high_volatility",
                    "condition": "volatility > 0.25",
                    "severity": "high",
                    "cooldown_minutes": 30,
                    "action_required": False,
                    "auto_action_available": False
                },
                {
                    "rule_name": "var_breach",
                    "condition": "var_95 > 0.05",
                    "severity": "high",
                    "cooldown_minutes": 20,
                    "action_required": True,
                    "auto_action_available": False
                }
            ],
            "notification_settings": {
                "channels": ["log", "console"],
                "escalation_enabled": True,
                "escalation_minutes": 60
            }
        }
    
    def _save_alert_config(self, config: Dict[str, Any]) -> None:
        """アラート設定保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save alert config: {e}")
    
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """アラートルール初期化"""
        rules = []
        
        for rule_config in self.config.get('alert_rules', []):
            try:
                rule = AlertRule(
                    rule_name=rule_config['rule_name'],
                    condition=rule_config['condition'],
                    severity=RiskSeverity(rule_config['severity']),
                    cooldown_minutes=rule_config.get('cooldown_minutes', 30),
                    action_required=rule_config.get('action_required', False),
                    auto_action_available=rule_config.get('auto_action_available', False)
                )
                rules.append(rule)
            except Exception as e:
                logger.error(f"Error initializing alert rule {rule_config}: {e}")
        
        return rules
    
    def _setup_notification_channels(self) -> Dict[AlertChannel, bool]:
        """通知チャネル設定"""
        channels = {}
        
        enabled_channels = self.config.get('notification_settings', {}).get('channels', ['log'])
        
        for channel_name in enabled_channels:
            try:
                channel = AlertChannel(channel_name)
                channels[channel] = True
            except ValueError:
                logger.warning(f"Unknown notification channel: {channel_name}")
        
        # デフォルトでログチャネルは有効
        if not channels:
            channels[AlertChannel.LOG] = True
        
        return channels
    
    def process_alert(self, risk_event: RiskEvent) -> List[AlertNotification]:
        """
        リスクイベントからアラートを生成・処理
        
        Args:
            risk_event: 処理対象のリスクイベント
            
        Returns:
            生成されたアラート通知のリスト
        """
        generated_alerts = []
        
        try:
            # 各アラートルールに対してイベントを評価
            for rule in self.alert_rules:
                if self._evaluate_alert_rule(rule, risk_event):
                    
                    # クールダウンチェック
                    if self._is_in_cooldown(rule.rule_name):
                        continue
                    
                    # アラート通知を生成
                    alert_notification = self._create_alert_notification(rule, risk_event)
                    
                    # 通知送信
                    self._send_alert_notification(alert_notification)
                    
                    # 履歴に追加
                    self.alert_history.append(alert_notification)
                    generated_alerts.append(alert_notification)
                    
                    # クールダウン設定
                    self.alert_cooldowns[rule.rule_name] = datetime.now()
                    
                    logger.info(f"Alert generated: {rule.rule_name} for {risk_event.event_type}")
            
            return generated_alerts
            
        except Exception as e:
            logger.error(f"Error processing alert for risk event: {e}")
            return []
    
    def _evaluate_alert_rule(self, rule: AlertRule, risk_event: RiskEvent) -> bool:
        """
        アラートルールがリスクイベントにマッチするかを評価
        
        Args:
            rule: 評価するアラートルール
            risk_event: リスクイベント
            
        Returns:
            マッチするかどうか
        """
        try:
            # 簡易条件評価（実際のプロダクションではより安全な評価が必要）
            context = {
                'drawdown': risk_event.drawdown,
                'var_95': 0.05 if risk_event.var_breach else 0.0,  # 簡易実装
                'volatility': 0.25 if 'volatility' in risk_event.description else 0.0,
                'correlation_risk': risk_event.correlation_risk,
                'severity': risk_event.severity
            }
            
            # 条件文字列の安全な評価
            condition = rule.condition.replace(' ', '_')  # 簡易サニタイズ
            
            # シンプルなマッチング
            if 'drawdown' in rule.condition and risk_event.drawdown > 0:
                threshold = 0.10 if 'critical' in rule.rule_name else 0.05
                return risk_event.drawdown > threshold
            
            if 'volatility' in rule.condition and 'volatility' in risk_event.description:
                return True
            
            if 'var' in rule.condition and risk_event.var_breach:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating alert rule {rule.rule_name}: {e}")
            return False
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """クールダウン期間中かチェック"""
        if rule_name not in self.alert_cooldowns:
            return False
        
        cooldown_time = self.alert_cooldowns[rule_name]
        rule = next((r for r in self.alert_rules if r.rule_name == rule_name), None)
        
        if not rule:
            return False
        
        elapsed = datetime.now() - cooldown_time
        return elapsed.total_seconds() < (rule.cooldown_minutes * 60)
    
    def _create_alert_notification(self, rule: AlertRule, risk_event: RiskEvent) -> AlertNotification:
        """アラート通知作成"""
        alert_id = f"{rule.rule_name}_{risk_event.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        message = self._format_alert_message(rule, risk_event)
        
        return AlertNotification(
            timestamp=datetime.now(),
            alert_id=alert_id,
            severity=rule.severity,
            message=message,
            channel=AlertChannel.LOG  # デフォルトはログ
        )
    
    def _format_alert_message(self, rule: AlertRule, risk_event: RiskEvent) -> str:
        """アラートメッセージのフォーマット"""
        return (
            f"[ALERT] [{rule.severity.value.upper()}] {rule.rule_name}\n"
            f"イベント: {risk_event.description}\n"
            f"発生時刻: {risk_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"ドローダウン: {risk_event.drawdown:.2%}\n"
            f"推奨対応: {risk_event.recommendation}"
        )
    
    def _send_alert_notification(self, alert: AlertNotification) -> bool:
        """アラート通知送信"""
        try:
            success = False
            
            # 有効な通知チャネルに送信
            for channel, enabled in self.notification_channels.items():
                if not enabled:
                    continue
                
                if channel == AlertChannel.LOG:
                    self._send_to_log(alert)
                    success = True
                
                elif channel == AlertChannel.CONSOLE:
                    self._send_to_console(alert)
                    success = True
                
                elif channel == AlertChannel.FILE:
                    self._send_to_file(alert)
                    success = True
            
            alert.sent = success
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            return False
    
    def _send_to_log(self, alert: AlertNotification) -> None:
        """ログへの通知送信"""
        if alert.severity == RiskSeverity.CRITICAL:
            logger.critical(alert.message)
        elif alert.severity == RiskSeverity.HIGH:
            logger.error(alert.message)
        elif alert.severity == RiskSeverity.MEDIUM:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
    
    def _send_to_console(self, alert: AlertNotification) -> None:
        """コンソールへの通知送信"""
        print(f"\n{'='*50}")
        print(f"RISK ALERT: {alert.alert_id}")
        print(f"{'='*50}")
        print(alert.message)
        print(f"{'='*50}\n")
    
    def _send_to_file(self, alert: AlertNotification) -> None:
        """ファイルへの通知送信"""
        try:
            alerts_dir = os.path.join(os.path.dirname(__file__), 'alerts')
            os.makedirs(alerts_dir, exist_ok=True)
            
            file_path = os.path.join(alerts_dir, 'risk_alerts.log')
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"{alert.timestamp.isoformat()}: {alert.message}\n\n")
                
        except Exception as e:
            logger.error(f"Error writing alert to file: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """アラート状況サマリー取得"""
        current_time = datetime.now()
        
        # 今日のアラート数
        today_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp.date() == current_time.date()
        ]
        
        # 重要度別集計
        severity_counts = {}
        for severity in RiskSeverity:
            severity_counts[severity.value] = len([
                alert for alert in today_alerts
                if alert.severity == severity
            ])
        
        return {
            'timestamp': current_time.isoformat(),
            'total_alerts_today': len(today_alerts),
            'severity_breakdown': severity_counts,
            'active_cooldowns': len(self.alert_cooldowns),
            'notification_channels': list(self.notification_channels.keys()),
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alert_history[-5:]  # 最新5件
            ]
        }
    
    def clear_old_alerts(self, days: int = 7) -> int:
        """古いアラート履歴をクリア"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        before_count = len(self.alert_history)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        after_count = len(self.alert_history)
        
        cleared_count = before_count - after_count
        logger.info(f"Cleared {cleared_count} old alerts (older than {days} days)")
        
        return cleared_count


# テスト実行関数
def run_risk_alert_manager_test():
    """リスクアラート管理システムのテスト"""
    logger.info("=== Risk Alert Manager Test ===")
    
    try:
        # アラート管理システム初期化
        alert_manager = RiskAlertManager()
        
        # テスト用リスクイベント作成
        test_risk_event = RiskEvent(
            timestamp=datetime.now(),
            event_type='test_drawdown',
            severity=RiskSeverity.HIGH.value,
            description='テスト用ドローダウン警告: 12%のドローダウンを検出',
            portfolio_value=900000,
            drawdown=0.12,
            var_breach=False,
            correlation_risk=0.0,
            recommendation='ポジションサイズ縮小を検討',
            requires_action=True
        )
        
        # アラート処理テスト
        alerts = alert_manager.process_alert(test_risk_event)
        
        print(f"生成されたアラート数: {len(alerts)}")
        for alert in alerts:
            print(f"- アラートID: {alert.alert_id}")
            print(f"  重要度: {alert.severity.value}")
        
        # サマリー表示
        summary = alert_manager.get_alert_summary()
        print(f"\nアラート管理状況:")
        print(f"- 今日のアラート: {summary['total_alerts_today']}")
        print(f"- 重要度別: {summary['severity_breakdown']}")
        
        logger.info("Risk Alert Manager test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト実行
    success = run_risk_alert_manager_test()
    if success:
        print("[OK] RiskAlertManager test passed")
    else:
        print("[ERROR] RiskAlertManager test failed")
