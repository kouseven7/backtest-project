"""
リアルタイムVaR監視システム

VaRのリアルタイム計算、監視、アラート機能
5-3-1 ドローダウン制御システムとの連携
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import pandas as pd

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .advanced_var_engine import AdvancedVaREngine, VaRCalculationConfig, VaRResult

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VaRAlert:
    """VaRアラート"""
    timestamp: datetime
    alert_type: str  # breach, warning, critical
    var_level: str   # var_95, var_99
    current_value: float
    threshold_value: float
    severity_score: float
    portfolio_composition: Dict[str, float]
    recommended_actions: List[str]
    message: str

@dataclass 
class MonitoringConfig:
    """監視設定"""
    monitoring_interval: int = 300  # 5分
    var_95_threshold: float = 0.05  # 5%
    var_99_threshold: float = 0.08  # 8%
    warning_threshold_ratio: float = 0.8  # 閾値の80%で警告
    critical_threshold_ratio: float = 1.2  # 閾値の120%でクリティカル
    
    # アラート設定
    enable_email_alerts: bool = False
    enable_log_alerts: bool = True
    enable_system_integration: bool = True
    
    # 履歴保存設定
    save_monitoring_history: bool = True
    history_retention_days: int = 90

class RealTimeVaRMonitor:
    """リアルタイムVaR監視システム"""
    
    def __init__(self, 
                 var_engine: AdvancedVaREngine,
                 monitoring_config: Optional[MonitoringConfig] = None):
        self.var_engine = var_engine
        self.config = monitoring_config or MonitoringConfig()
        self.logger = self._setup_logger()
        
        # 監視状態
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # アラート履歴
        self.alert_history: List[VaRAlert] = []
        
        # 監視データ履歴
        self.monitoring_history: List[Dict[str, Any]] = []
        
        # ドローダウンコントローラー連携
        self.drawdown_controller_callback: Optional[Callable] = None
        
        self.logger.info("RealTimeVaRMonitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.RealTimeVaRMonitor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def start_monitoring(self,
                        data_provider: Callable[[], pd.DataFrame],
                        weight_provider: Callable[[], Dict[str, float]]) -> bool:
        """リアルタイム監視開始"""
        try:
            if self.is_monitoring:
                self.logger.warning("Monitoring is already running")
                return False
            
            self.logger.info("Starting real-time VaR monitoring")
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(data_provider, weight_provider),
                daemon=True
            )
            self.monitoring_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """監視停止"""
        try:
            if not self.is_monitoring:
                self.logger.warning("Monitoring is not running")
                return False
            
            self.logger.info("Stopping real-time VaR monitoring")
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=30)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _monitoring_loop(self,
                        data_provider: Callable[[], pd.DataFrame],
                        weight_provider: Callable[[], Dict[str, float]]) -> None:
        """監視ループ"""
        self.logger.info("Monitoring loop started")
        
        while self.is_monitoring:
            try:
                # データ取得
                returns_data = data_provider()
                current_weights = weight_provider()
                
                if returns_data.empty or not current_weights:
                    self.logger.warning("No data or weights available for monitoring")
                    time.sleep(self.config.monitoring_interval)
                    continue
                
                # VaR計算
                var_result = self.var_engine.calculate_comprehensive_var(
                    returns_data, current_weights
                )
                
                # 監視データ記録
                self._record_monitoring_data(var_result, current_weights)
                
                # 閾値チェック
                alerts = self._check_var_thresholds(var_result)
                
                # アラート処理
                if alerts:
                    self._process_alerts(alerts, var_result)
                
                # ドローダウンコントローラー連携
                if self.drawdown_controller_callback and alerts:
                    self._notify_drawdown_controller(alerts, var_result)
                
                # 監視間隔待機
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # エラー時は30秒待機
        
        self.logger.info("Monitoring loop stopped")
    
    def _record_monitoring_data(self, var_result: VaRResult, weights: Dict[str, float]) -> None:
        """監視データの記録"""
        try:
            if not self.config.save_monitoring_history:
                return
            
            monitoring_record = {
                'timestamp': datetime.now().isoformat(),
                'var_95': var_result.get_var_95(),
                'var_99': var_result.get_var_99(),
                'portfolio_weights': weights.copy(),
                'market_regime': var_result.market_regime,
                'calculation_method': var_result.calculation_method,
                'diversification_benefit': var_result.diversification_benefit
            }
            
            self.monitoring_history.append(monitoring_record)
            
            # 履歴データのクリーンアップ（保持期間を超えたデータを削除）
            cutoff_date = datetime.now() - timedelta(days=self.config.history_retention_days)
            self.monitoring_history = [
                record for record in self.monitoring_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Monitoring data recording error: {e}")
    
    def _check_var_thresholds(self, var_result: VaRResult) -> List[VaRAlert]:
        """VaR閾値チェック"""
        try:
            alerts = []
            
            # VaR 95%チェック
            var_95 = var_result.get_var_95()
            if var_95 > 0:
                alert = self._check_single_threshold(
                    var_95, self.config.var_95_threshold, 'var_95', var_result
                )
                if alert:
                    alerts.append(alert)
            
            # VaR 99%チェック
            var_99 = var_result.get_var_99()
            if var_99 > 0:
                alert = self._check_single_threshold(
                    var_99, self.config.var_99_threshold, 'var_99', var_result
                )
                if alert:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"VaR threshold check error: {e}")
            return []
    
    def _check_single_threshold(self,
                               current_value: float,
                               threshold: float,
                               var_level: str,
                               var_result: VaRResult) -> Optional[VaRAlert]:
        """単一閾値のチェック"""
        try:
            warning_threshold = threshold * self.config.warning_threshold_ratio
            critical_threshold = threshold * self.config.critical_threshold_ratio
            
            alert = None
            
            if current_value >= critical_threshold:
                # クリティカル
                alert = VaRAlert(
                    timestamp=datetime.now(),
                    alert_type='critical',
                    var_level=var_level,
                    current_value=current_value,
                    threshold_value=threshold,
                    severity_score=min(1.0, current_value / threshold),
                    portfolio_composition=var_result.portfolio_composition.copy(),
                    recommended_actions=[
                        'immediate_position_reduction',
                        'activate_hedging',
                        'emergency_risk_review'
                    ],
                    message=f"CRITICAL: {var_level.upper()} exceeded {critical_threshold:.4f} (current: {current_value:.4f})"
                )
            elif current_value >= threshold:
                # 閾値違反
                alert = VaRAlert(
                    timestamp=datetime.now(),
                    alert_type='breach',
                    var_level=var_level,
                    current_value=current_value,
                    threshold_value=threshold,
                    severity_score=min(1.0, current_value / threshold),
                    portfolio_composition=var_result.portfolio_composition.copy(),
                    recommended_actions=[
                        'reduce_high_risk_positions',
                        'increase_diversification',
                        'review_portfolio_allocation'
                    ],
                    message=f"BREACH: {var_level.upper()} exceeded threshold {threshold:.4f} (current: {current_value:.4f})"
                )
            elif current_value >= warning_threshold:
                # 警告レベル
                alert = VaRAlert(
                    timestamp=datetime.now(),
                    alert_type='warning',
                    var_level=var_level,
                    current_value=current_value,
                    threshold_value=threshold,
                    severity_score=current_value / threshold,
                    portfolio_composition=var_result.portfolio_composition.copy(),
                    recommended_actions=[
                        'monitor_closely',
                        'prepare_risk_mitigation',
                        'review_position_sizes'
                    ],
                    message=f"WARNING: {var_level.upper()} approaching threshold {threshold:.4f} (current: {current_value:.4f})"
                )
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Single threshold check error: {e}")
            return None
    
    def _process_alerts(self, alerts: List[VaRAlert], var_result: VaRResult) -> None:
        """アラート処理"""
        try:
            for alert in alerts:
                # アラート履歴に追加
                self.alert_history.append(alert)
                
                # ログ出力
                if self.config.enable_log_alerts:
                    log_level = logging.CRITICAL if alert.alert_type == 'critical' else \
                               logging.WARNING if alert.alert_type == 'breach' else \
                               logging.INFO
                    
                    self.logger.log(log_level, alert.message)
                    self.logger.info(f"Recommended actions: {', '.join(alert.recommended_actions)}")
                
                # システム統合通知
                if self.config.enable_system_integration:
                    self._send_system_notification(alert, var_result)
            
            # アラート履歴のクリーンアップ
            cutoff_date = datetime.now() - timedelta(days=30)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Alert processing error: {e}")
    
    def _send_system_notification(self, alert: VaRAlert, var_result: VaRResult) -> None:
        """システム通知送信"""
        try:
            # システム通知のロジック
            # 実際の実装では、メッセージング システム、Webhook、
            # または他のシステムコンポーネントに通知を送信
            
            notification_data = {
                'alert': {
                    'type': alert.alert_type,
                    'level': alert.var_level,
                    'severity': alert.severity_score,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                },
                'var_data': {
                    'var_95': var_result.get_var_95(),
                    'var_99': var_result.get_var_99(),
                    'regime': var_result.market_regime,
                    'diversification': var_result.diversification_benefit
                },
                'portfolio': alert.portfolio_composition,
                'recommendations': alert.recommended_actions
            }
            
            self.logger.info(f"System notification sent: {alert.alert_type} alert for {alert.var_level}")
            
        except Exception as e:
            self.logger.error(f"System notification error: {e}")
    
    def _notify_drawdown_controller(self, alerts: List[VaRAlert], var_result: VaRResult) -> None:
        """ドローダウンコントローラー通知"""
        try:
            if not self.drawdown_controller_callback:
                return
            
            # 最も重要なアラートを選択
            critical_alerts = [a for a in alerts if a.alert_type == 'critical']
            breach_alerts = [a for a in alerts if a.alert_type == 'breach']
            
            priority_alert = None
            if critical_alerts:
                priority_alert = max(critical_alerts, key=lambda x: x.severity_score)
            elif breach_alerts:
                priority_alert = max(breach_alerts, key=lambda x: x.severity_score)
            
            if priority_alert:
                # ドローダウンコントローラーへの通知データ
                drawdown_signal = {
                    'signal_type': 'var_breach',
                    'severity_level': priority_alert.severity_score,
                    'risk_source': 'portfolio_var',
                    'var_metrics': {
                        'var_95': var_result.get_var_95(),
                        'var_99': var_result.get_var_99(),
                        'breach_level': priority_alert.var_level,
                        'breach_ratio': priority_alert.current_value / priority_alert.threshold_value
                    },
                    'recommended_actions': priority_alert.recommended_actions,
                    'portfolio_weights': priority_alert.portfolio_composition,
                    'timestamp': datetime.now()
                }
                
                # コールバック実行
                self.drawdown_controller_callback(drawdown_signal)
                self.logger.info("Drawdown controller notified of VaR breach")
            
        except Exception as e:
            self.logger.error(f"Drawdown controller notification error: {e}")
    
    def set_drawdown_controller_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """ドローダウンコントローラーコールバック設定"""
        try:
            self.drawdown_controller_callback = callback
            self.logger.info("Drawdown controller callback set")
        except Exception as e:
            self.logger.error(f"Drawdown controller callback setup error: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視ステータス取得"""
        try:
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= datetime.now() - timedelta(hours=24)
            ]
            
            return {
                'is_monitoring': self.is_monitoring,
                'monitoring_interval': self.config.monitoring_interval,
                'last_calculation': self.var_engine.calculation_history[-1].timestamp.isoformat() \
                                  if self.var_engine.calculation_history else None,
                'total_alerts_24h': len(recent_alerts),
                'critical_alerts_24h': len([a for a in recent_alerts if a.alert_type == 'critical']),
                'breach_alerts_24h': len([a for a in recent_alerts if a.alert_type == 'breach']),
                'monitoring_history_count': len(self.monitoring_history),
                'config': {
                    'var_95_threshold': self.config.var_95_threshold,
                    'var_99_threshold': self.config.var_99_threshold,
                    'warning_ratio': self.config.warning_threshold_ratio,
                    'critical_ratio': self.config.critical_threshold_ratio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring status error: {e}")
            return {'error': str(e)}
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """アラートサマリー取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            summary = {
                'time_period_hours': hours,
                'total_alerts': len(recent_alerts),
                'alert_breakdown': {
                    'critical': len([a for a in recent_alerts if a.alert_type == 'critical']),
                    'breach': len([a for a in recent_alerts if a.alert_type == 'breach']),
                    'warning': len([a for a in recent_alerts if a.alert_type == 'warning'])
                },
                'var_level_breakdown': {
                    'var_95': len([a for a in recent_alerts if a.var_level == 'var_95']),
                    'var_99': len([a for a in recent_alerts if a.var_level == 'var_99'])
                }
            }
            
            if recent_alerts:
                # 最新アラート情報
                latest_alert = max(recent_alerts, key=lambda x: x.timestamp)
                summary['latest_alert'] = {
                    'timestamp': latest_alert.timestamp.isoformat(),
                    'type': latest_alert.alert_type,
                    'level': latest_alert.var_level,
                    'severity': latest_alert.severity_score,
                    'message': latest_alert.message
                }
                
                # 重要度別統計
                severity_scores = [a.severity_score for a in recent_alerts]
                summary['severity_stats'] = {
                    'avg_severity': sum(severity_scores) / len(severity_scores),
                    'max_severity': max(severity_scores),
                    'min_severity': min(severity_scores)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Alert summary error: {e}")
            return {'error': str(e)}
    
    def export_monitoring_data(self, output_path: str, days: int = 7) -> bool:
        """監視データのエクスポート"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 監視データフィルタリング
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'period_days': days,
                    'monitoring_config': {
                        'var_95_threshold': self.config.var_95_threshold,
                        'var_99_threshold': self.config.var_99_threshold,
                        'monitoring_interval': self.config.monitoring_interval
                    }
                },
                'monitoring_history': [
                    record for record in self.monitoring_history
                    if datetime.fromisoformat(record['timestamp']) >= cutoff_date
                ],
                'alert_history': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'type': alert.alert_type,
                        'level': alert.var_level,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold_value,
                        'severity': alert.severity_score,
                        'message': alert.message,
                        'recommendations': alert.recommended_actions
                    }
                    for alert in self.alert_history
                    if alert.timestamp >= cutoff_date
                ]
            }
            
            # JSONファイルに保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Monitoring data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export monitoring data error: {e}")
            return False
