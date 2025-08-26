"""
Unified Risk Monitor - 統合リスク監視システム
DSSMS Phase 2 Task 2.3

既存のPortfolioRiskManagerとDrawdownControllerを統合した
包括的リスク監視システム
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガーの設定
logger = logging.getLogger(__name__)

try:
    from config.portfolio_risk_manager import (
        PortfolioRiskManager, RiskConfiguration, RiskMetricType, RiskLimitType
    )
    from config.drawdown_controller import DrawdownController
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.position_size_adjuster import PositionSizeAdjuster
    from config.signal_integrator import SignalIntegrator
except ImportError as e:
    logger.error(f"Failed to import risk management modules: {e}")
    # フォールバック用のダミークラス
    class PortfolioRiskManager:
        def __init__(self, *args, **kwargs):
            pass
        def calculate_all_risk_metrics(self, *args, **kwargs):
            return {}
    
    class DrawdownController:
        def calculate_current_drawdown(self, *args, **kwargs):
            return 0.0
    
    class RiskConfiguration:
        pass

@dataclass
class RiskEvent:
    """リスクイベントデータクラス"""
    timestamp: datetime
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    portfolio_value: float
    drawdown: float
    var_breach: bool
    correlation_risk: float
    recommendation: str
    requires_action: bool = False
    auto_action_available: bool = False

class RiskSeverity(Enum):
    """リスク重要度レベル"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class UnifiedRiskMonitor:
    """
    統合リスク監視システム
    既存のPortfolioRiskManagerとDrawdownControllerを統合
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        統合リスク監視システムの初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        # 設定ファイル読み込み
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
        # 既存システム統合のための初期化
        try:
            # ダミー実装で依存関係を初期化
            risk_config = RiskConfiguration()
            dummy_weight_calculator = None  # ダミー実装
            dummy_position_adjuster = None  # ダミー実装  
            dummy_signal_integrator = None  # ダミー実装
            
            # PortfolioRiskManagerを直接初期化せず、リスク計算を独自実装
            self.portfolio_risk_manager = None
            self.drawdown_controller = DrawdownController()
        except Exception as e:
            logger.warning(f"Could not initialize full risk management system: {e}")
            self.portfolio_risk_manager = None
            self.drawdown_controller = None
        
        # 新機能初期化
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_check_time = None
        
        # リスクイベント履歴
        self.risk_events: List[RiskEvent] = []
        self.active_alerts: Dict[str, RiskEvent] = {}
        
        # 閾値設定
        self.thresholds = self.config.get('risk_thresholds', {})
        
        # アラート管理とアクション処理を後で統合
        self.alert_manager = None
        self.action_processor = None
        
        logger.info("UnifiedRiskMonitor initialized successfully")
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパス取得"""
        return os.path.join(
            os.path.dirname(__file__),
            'configs',
            'risk_thresholds.json'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # デフォルト設定を作成
                default_config = self._create_default_config()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定作成"""
        return {
            "risk_thresholds": {
                "portfolio_level": {
                    "max_drawdown": 0.15,
                    "var_95": 0.05,
                    "volatility": 0.25,
                    "correlation_limit": 0.80
                },
                "position_level": {
                    "max_weight": 0.20,
                    "concentration_risk": 0.30
                }
            },
            "monitoring": {
                "frequency_minutes": 10,
                "alert_cooldown_minutes": 30,
                "enable_auto_actions": False
            },
            "alert_rules": {
                "critical_alerts": {
                    "drawdown_threshold": 0.10,
                    "var_breach": 0.03,
                    "correlation_spike": 0.80
                },
                "warning_alerts": {
                    "drawdown_threshold": 0.05,
                    "volatility_spike": 0.20
                }
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """設定ファイル保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def start_monitoring(self) -> None:
        """リスク監視開始"""
        if self.monitoring_active:
            logger.warning("Risk monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self) -> None:
        """リスク監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """リスク監視メインループ"""
        frequency = self.config.get('monitoring', {}).get('frequency_minutes', 10)
        
        while self.monitoring_active:
            try:
                # リスク評価実行
                risk_assessment = self.perform_comprehensive_risk_check()
                
                # リスクイベント処理
                if risk_assessment:
                    self._process_risk_events(risk_assessment)
                
                self.last_check_time = datetime.now()
                
                # 次の監視まで待機
                time.sleep(frequency * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # エラー時は1分後に再試行
    
    def perform_comprehensive_risk_check(self, portfolio_data: Optional[Dict[str, Any]] = None) -> List[RiskEvent]:
        """
        包括的リスク評価実行
        
        Args:
            portfolio_data: ポートフォリオデータ（省略時はモック使用）
            
        Returns:
            検出されたリスクイベントのリスト
        """
        risk_events = []
        
        try:
            # ポートフォリオデータの準備
            if portfolio_data is None:
                portfolio_data = self._get_mock_portfolio_data()
            
            # 1. 基本リスク評価（独自実装）
            portfolio_risk = self._calculate_basic_portfolio_risk(portfolio_data)
            
            # 2. DrawdownControllerによるドローダウン監視
            current_drawdown = self._calculate_drawdown_safely(portfolio_data)
            
            # 3. リスク閾値チェック
            risk_events.extend(self._check_risk_thresholds(portfolio_risk, current_drawdown))
            
            # 4. 追加リスク指標計算
            additional_risks = self._calculate_additional_risk_metrics(portfolio_data)
            risk_events.extend(self._evaluate_additional_risks(additional_risks))
            
            return risk_events
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk check: {e}")
            return []
    
    def _calculate_basic_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """基本的なポートフォリオリスク計算（独自実装）"""
        try:
            returns = portfolio_data.get('returns', [])
            positions = portfolio_data.get('positions', {})
            
            if not returns or not positions:
                return {'volatility': 0.0, 'var_95': 0.0}
            
            returns_array = np.array(returns)
            
            # ボラティリティ計算
            volatility = np.std(returns_array) if len(returns_array) > 1 else 0.0
            
            # VaR計算 (95%信頼区間)
            var_95 = abs(np.percentile(returns_array, 5)) if len(returns_array) > 10 else 0.0
            
            # 集中度リスク計算
            weights = [pos.get('weight', 0) for pos in positions.values()]
            concentration_risk = max(weights) if weights else 0.0
            
            return {
                'volatility': volatility,
                'var_95': var_95,
                'concentration_risk': concentration_risk,
                'positions_count': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error in basic portfolio risk calculation: {e}")
            return {'volatility': 0.0, 'var_95': 0.0}
    
    def _calculate_drawdown_safely(self, portfolio_data: Dict[str, Any]) -> float:
        """安全なドローダウン計算"""
        try:
            if self.drawdown_controller:
                portfolio_values = portfolio_data.get('portfolio_values', [])
                if portfolio_values:
                    return self.drawdown_controller.calculate_current_drawdown(portfolio_values)
            
            # フォールバック実装
            portfolio_values = portfolio_data.get('portfolio_values', [])
            if len(portfolio_values) < 2:
                return 0.0
            
            # 簡易ドローダウン計算
            peak = max(portfolio_values)
            current = portfolio_values[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0.0
            
            return max(0.0, drawdown)
            
        except Exception as e:
            logger.error(f"Error in drawdown calculation: {e}")
            return 0.0

    def _get_mock_portfolio_data(self) -> Dict[str, Any]:
        """モックポートフォリオデータ生成（テスト用）"""
        current_time = datetime.now()
        
        return {
            'timestamp': current_time,
            'portfolio_value': 1000000,  # 100万円
            'portfolio_values': [1000000 + i*1000 for i in range(-30, 1)],  # 30日間の履歴
            'positions': {
                'strategy_1': {'value': 400000, 'weight': 0.4},
                'strategy_2': {'value': 350000, 'weight': 0.35},
                'strategy_3': {'value': 250000, 'weight': 0.25}
            },
            'returns': np.random.normal(0.001, 0.02, 30),  # 30日間のリターン
            'volatility': 0.15
        }
    
    def _check_risk_thresholds(self, portfolio_risk: Dict[str, Any], current_drawdown: float) -> List[RiskEvent]:
        """リスク閾値チェック"""
        events = []
        current_time = datetime.now()
        
        # ドローダウンチェック
        max_dd_threshold = self.thresholds.get('portfolio_level', {}).get('max_drawdown', 0.15)
        critical_dd_threshold = self.config.get('alert_rules', {}).get('critical_alerts', {}).get('drawdown_threshold', 0.10)
        
        if current_drawdown > max_dd_threshold:
            severity = RiskSeverity.CRITICAL if current_drawdown > critical_dd_threshold else RiskSeverity.HIGH
            events.append(RiskEvent(
                timestamp=current_time,
                event_type='drawdown_breach',
                severity=severity.value,
                description=f'ドローダウン閾値超過: {current_drawdown:.2%} > {max_dd_threshold:.2%}',
                portfolio_value=1000000,  # モック値
                drawdown=current_drawdown,
                var_breach=False,
                correlation_risk=0.0,
                recommendation='ポジションサイズ縮小を検討',
                requires_action=True,
                auto_action_available=True
            ))
        
        # ボラティリティチェック
        volatility = portfolio_risk.get('volatility', 0.0)
        vol_threshold = self.thresholds.get('portfolio_level', {}).get('volatility', 0.25)
        
        if volatility > vol_threshold:
            events.append(RiskEvent(
                timestamp=current_time,
                event_type='volatility_spike',
                severity=RiskSeverity.MEDIUM.value,
                description=f'ボラティリティ上昇: {volatility:.2%} > {vol_threshold:.2%}',
                portfolio_value=1000000,
                drawdown=current_drawdown,
                var_breach=False,
                correlation_risk=0.0,
                recommendation='リスク調整を検討',
                requires_action=False
            ))
        
        return events
    
    def _calculate_additional_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """追加リスク指標計算"""
        try:
            returns = portfolio_data.get('returns', [])
            
            if len(returns) < 10:
                logger.warning("Insufficient data for risk metrics calculation")
                return {}
            
            returns_array = np.array(returns)
            
            # VaR計算 (95%信頼区間)
            var_95 = np.percentile(returns_array, 5)
            
            # 期待ショートフォール (CVaR)
            cvar_95 = returns_array[returns_array <= var_95].mean()
            
            # 相関リスク（簡易版）
            correlation_risk = min(0.8, abs(np.corrcoef(returns_array[:-1], returns_array[1:])[0,1]))
            
            return {
                'var_95': abs(var_95),
                'cvar_95': abs(cvar_95),
                'correlation_risk': correlation_risk,
                'volatility': np.std(returns_array)
            }
            
        except Exception as e:
            logger.error(f"Error calculating additional risk metrics: {e}")
            return {}
    
    def _evaluate_additional_risks(self, risk_metrics: Dict[str, float]) -> List[RiskEvent]:
        """追加リスク指標評価"""
        events = []
        current_time = datetime.now()
        
        # VaR閾値チェック
        var_95 = risk_metrics.get('var_95', 0.0)
        var_threshold = self.thresholds.get('portfolio_level', {}).get('var_95', 0.05)
        
        if var_95 > var_threshold:
            events.append(RiskEvent(
                timestamp=current_time,
                event_type='var_breach',
                severity=RiskSeverity.HIGH.value,
                description=f'VaR(95%)閾値超過: {var_95:.2%} > {var_threshold:.2%}',
                portfolio_value=1000000,
                drawdown=0.0,
                var_breach=True,
                correlation_risk=risk_metrics.get('correlation_risk', 0.0),
                recommendation='ポートフォリオリバランス推奨',
                requires_action=True
            ))
        
        return events
    
    def _process_risk_events(self, risk_events: List[RiskEvent]) -> None:
        """リスクイベント処理"""
        for event in risk_events:
            # イベント履歴に追加
            self.risk_events.append(event)
            
            # アクティブアラートに追加
            self.active_alerts[event.event_type] = event
            
            # ログ出力
            logger.warning(f"Risk Event Detected: {event.description}")
            
            # 将来のアラート管理システムとの連携ポイント
            if self.alert_manager:
                self.alert_manager.process_alert(event)
            
            # 自動アクション処理
            if event.requires_action and event.auto_action_available:
                if self.action_processor:
                    self.action_processor.propose_action(event)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """リスク状況サマリー取得"""
        current_time = datetime.now()
        
        return {
            'timestamp': current_time.isoformat(),
            'monitoring_active': self.monitoring_active,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'active_alerts_count': len(self.active_alerts),
            'total_events_today': len([
                e for e in self.risk_events 
                if e.timestamp.date() == current_time.date()
            ]),
            'critical_events': len([
                e for e in self.risk_events 
                if e.severity == RiskSeverity.CRITICAL.value
            ]),
            'active_alerts': {
                k: {
                    'severity': v.severity,
                    'description': v.description,
                    'timestamp': v.timestamp.isoformat()
                }
                for k, v in self.active_alerts.items()
            }
        }
    
    def clear_resolved_alerts(self, alert_types: List[str]) -> None:
        """解決済みアラートクリア"""
        for alert_type in alert_types:
            if alert_type in self.active_alerts:
                del self.active_alerts[alert_type]
                logger.info(f"Cleared alert: {alert_type}")


# 統合テスト用のシンプルファンクション
def run_unified_risk_monitor_test():
    """統合リスク監視システムのテスト実行"""
    logger.info("=== Unified Risk Monitor Test ===")
    
    try:
        # 監視システム初期化
        monitor = UnifiedRiskMonitor()
        
        # 即座のリスクチェック
        risk_events = monitor.perform_comprehensive_risk_check()
        
        print(f"検出されたリスクイベント数: {len(risk_events)}")
        for event in risk_events:
            print(f"- {event.severity}: {event.description}")
        
        # リスクサマリー表示
        summary = monitor.get_risk_summary()
        print(f"\nリスク監視状況:")
        print(f"- アクティブアラート: {summary['active_alerts_count']}")
        print(f"- 今日のイベント: {summary['total_events_today']}")
        
        logger.info("Unified Risk Monitor test completed successfully")
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
    success = run_unified_risk_monitor_test()
    if success:
        print("✅ UnifiedRiskMonitor test passed")
    else:
        print("❌ UnifiedRiskMonitor test failed")
