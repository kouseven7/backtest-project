"""
Module: Drawdown Controller
File: drawdown_controller.py
Description: 
  5-3-1「ドローダウン制御機能の追加」
  ポートフォリオドローダウンを動的に監視・制御する専用システム
  既存のPortfolioRiskManagerと連携してリアルタイム制御を提供

Author: imega
Created: 2025-07-20
Modified: 2025-07-20
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

try:
    from config.portfolio_risk_manager import PortfolioRiskManager
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.position_size_adjuster import PositionSizeAdjuster
    HAS_PORTFOLIO_MODULES = True
except ImportError as e:
    logger.warning(f"Import error for portfolio modules: {e}. Using fallback implementations.")
    PortfolioRiskManager = None
    PortfolioWeightCalculator = None
    PositionSizeAdjuster = None
    HAS_PORTFOLIO_MODULES = False

# 一時的にマルチ戦略調整マネージャーのインポートを無効化
# try:
#     from config.multi_strategy_coordination_manager import MultiStrategyCoordinationManager
#     HAS_COORDINATION_MODULE = True
# except Exception as e:
#     logger.warning(f"Import error for coordination module: {e}. Using fallback implementations.")
#     MultiStrategyCoordinationManager = None
#     HAS_COORDINATION_MODULE = False

# 一時的にFalseに設定
MultiStrategyCoordinationManager = None
HAS_COORDINATION_MODULE = False

logger = logging.getLogger(__name__)

class DrawdownSeverity(Enum):
    """ドローダウン深刻度レベル"""
    NORMAL = "normal"           # 通常範囲（0-5%）
    WARNING = "warning"         # 警告レベル（5-10%）
    CRITICAL = "critical"       # 重要レベル（10-15%）
    EMERGENCY = "emergency"     # 緊急レベル（15%以上）

class DrawdownControlAction(Enum):
    """ドローダウン制御アクション"""
    NO_ACTION = "no_action"
    POSITION_REDUCTION_LIGHT = "position_reduction_light"     # 軽度削減（10-20%）
    POSITION_REDUCTION_MODERATE = "position_reduction_moderate" # 中度削減（30-50%）
    POSITION_REDUCTION_HEAVY = "position_reduction_heavy"     # 重度削減（50-70%）
    STRATEGY_SUSPENSION = "strategy_suspension"               # 戦略一時停止
    EMERGENCY_STOP = "emergency_stop"                        # 緊急全停止
    HEDGE_ACTIVATION = "hedge_activation"                    # ヘッジ機能起動

class DrawdownControlMode(Enum):
    """ドローダウン制御モード"""
    CONSERVATIVE = "conservative"   # 保守的（早期介入）
    MODERATE = "moderate"          # 中庸（バランス重視）
    AGGRESSIVE = "aggressive"      # 積極的（リターン重視）

@dataclass
class DrawdownThresholds:
    """ドローダウン閾値設定"""
    # 基本閾値（ポートフォリオ価値に対する割合）
    warning_threshold: float = 0.05      # 5%で警告
    critical_threshold: float = 0.10     # 10%で重要
    emergency_threshold: float = 0.15    # 15%で緊急
    
    # 時系列チェック設定
    consecutive_loss_limit: int = 5      # 連続損失日数制限
    daily_loss_threshold: float = 0.03   # 日次損失制限（3%）
    
    # 戦略別設定
    strategy_dd_threshold: float = 0.08  # 戦略別ドローダウン制限（8%）
    strategy_suspension_threshold: float = 0.12  # 戦略停止閾値（12%）
    
    # 動的調整設定
    volatility_adjustment: bool = True   # ボラティリティ調整有効
    trend_adjustment: bool = True        # トレンド調整有効

@dataclass
class DrawdownEvent:
    """ドローダウンイベント"""
    timestamp: datetime
    portfolio_value: float
    previous_peak: float
    current_drawdown: float
    drawdown_percentage: float
    severity: DrawdownSeverity
    affected_strategies: List[str]
    triggering_factor: str
    
    @property
    def duration_days(self) -> int:
        """ドローダウン継続日数"""
        return (datetime.now() - self.timestamp).days

@dataclass
class DrawdownControlResult:
    """ドローダウン制御実行結果"""
    timestamp: datetime
    event: DrawdownEvent
    action_taken: DrawdownControlAction
    original_positions: Dict[str, float]
    adjusted_positions: Dict[str, float]
    expected_impact: float
    success: bool
    error_message: Optional[str] = None
    
    def get_position_changes(self) -> Dict[str, float]:
        """ポジション変更量を計算"""
        changes = {}
        all_strategies = set(self.original_positions.keys()) | set(self.adjusted_positions.keys())
        
        for strategy in all_strategies:
            original = self.original_positions.get(strategy, 0.0)
            adjusted = self.adjusted_positions.get(strategy, 0.0)
            changes[strategy] = adjusted - original
            
        return changes

class DrawdownController:
    """
    ドローダウンコントローラー - メインクラス
    ポートフォリオのドローダウンを監視し、閾値超過時に自動制御を実行
    """
    
    def __init__(self,
                 config_file: Optional[str] = None,
                 portfolio_risk_manager = None,
                 position_size_adjuster = None,
                 portfolio_weight_calculator = None,
                 coordination_manager = None):
        """
        ドローダウンコントローラーの初期化
        
        Parameters:
            config_file: 設定ファイルパス
            portfolio_risk_manager: ポートフォリオリスク管理器
            position_size_adjuster: ポジションサイズ調整器
            portfolio_weight_calculator: ポートフォリオウェイト計算器
            coordination_manager: マルチ戦略調整管理器
        """
        self.config_file = config_file
        
        # 設定の読み込み
        self.config = self._load_config()
        self.thresholds = DrawdownThresholds(**self.config.get('thresholds', {}))
        self.control_mode = DrawdownControlMode(self.config.get('control_mode', 'moderate'))
        
        # 既存システムとの統合
        self.portfolio_risk_manager = portfolio_risk_manager
        self.position_size_adjuster = position_size_adjuster
        self.portfolio_weight_calculator = portfolio_weight_calculator
        self.coordination_manager = coordination_manager
        
        # 制御状態管理
        self.is_monitoring = False
        self.current_event: Optional[DrawdownEvent] = None
        self.control_history: List[DrawdownControlResult] = []
        self.suspended_strategies: Set[str] = set()
        
        # パフォーマンス追跡
        self.performance_tracker = {
            'portfolio_peak': 0.0,
            'portfolio_history': [],
            'strategy_peaks': {},
            'strategy_histories': {},
            'last_update': datetime.now()
        }
        
        # スレッド制御
        self._control_lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("DrawdownController initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        default_config = {
            "control_mode": "moderate",
            "monitoring_interval": 60,  # 秒
            "thresholds": {
                "warning_threshold": 0.05,
                "critical_threshold": 0.10,
                "emergency_threshold": 0.15,
                "consecutive_loss_limit": 5,
                "daily_loss_threshold": 0.03,
                "strategy_dd_threshold": 0.08,
                "strategy_suspension_threshold": 0.12,
                "volatility_adjustment": True,
                "trend_adjustment": True
            },
            "actions": {
                "warning": "position_reduction_light",
                "critical": "position_reduction_moderate", 
                "emergency": "emergency_stop"
            },
            "notifications": {
                "enabled": True,
                "email_alerts": False,
                "webhook_url": None
            }
        }
        
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # デフォルト設定をユーザー設定で更新
                self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict) -> None:
        """辞書の深いマージ"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def start_monitoring(self, portfolio_value: float = None):
        """ドローダウン監視を開始"""
        if self.is_monitoring:
            logger.warning("Drawdown monitoring already active")
            return
        
        if portfolio_value:
            self.performance_tracker['portfolio_peak'] = portfolio_value
            self.performance_tracker['portfolio_history'] = [(datetime.now(), portfolio_value)]
        
        self.is_monitoring = True
        self._stop_event.clear()
        
        # 監視スレッドの開始
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Drawdown monitoring started")
    
    def stop_monitoring(self):
        """ドローダウン監視を停止"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Drawdown monitoring stopped")
    
    def _monitoring_loop(self):
        """監視ループ（バックグラウンド実行）"""
        monitoring_interval = self.config.get('monitoring_interval', 60)
        
        while not self._stop_event.is_set():
            try:
                # ドローダウンチェック実行
                self._check_drawdown()
                
                # 指定間隔で待機
                self._stop_event.wait(monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(monitoring_interval)
    
    def update_portfolio_value(self, 
                              portfolio_value: float,
                              strategy_values: Optional[Dict[str, float]] = None):
        """ポートフォリオ価値を更新（外部から呼び出し可能）"""
        try:
            with self._control_lock:
                current_time = datetime.now()
                
                # ポートフォリオレベルの更新
                self.performance_tracker['portfolio_history'].append((current_time, portfolio_value))
                
                # ピーク値の更新
                if portfolio_value > self.performance_tracker['portfolio_peak']:
                    self.performance_tracker['portfolio_peak'] = portfolio_value
                
                # 戦略レベルの更新
                if strategy_values:
                    for strategy, value in strategy_values.items():
                        if strategy not in self.performance_tracker['strategy_histories']:
                            self.performance_tracker['strategy_histories'][strategy] = []
                            self.performance_tracker['strategy_peaks'][strategy] = value
                        
                        self.performance_tracker['strategy_histories'][strategy].append((current_time, value))
                        
                        # 戦略別ピーク更新
                        if value > self.performance_tracker['strategy_peaks'][strategy]:
                            self.performance_tracker['strategy_peaks'][strategy] = value
                
                self.performance_tracker['last_update'] = current_time
                
                # 履歴サイズ制限（最新1000件）
                if len(self.performance_tracker['portfolio_history']) > 1000:
                    self.performance_tracker['portfolio_history'] = self.performance_tracker['portfolio_history'][-1000:]
                
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _check_drawdown(self) -> Optional[DrawdownEvent]:
        """ドローダウンチェックを実行"""
        try:
            if not self.performance_tracker['portfolio_history']:
                return None
            
            # 現在の状況を取得
            current_time, current_value = self.performance_tracker['portfolio_history'][-1]
            peak_value = self.performance_tracker['portfolio_peak']
            
            # ドローダウン計算
            if peak_value <= 0:
                return None
            
            current_drawdown = peak_value - current_value
            drawdown_percentage = current_drawdown / peak_value
            
            # 閾値チェック
            severity = self._determine_severity(drawdown_percentage)
            
            if severity != DrawdownSeverity.NORMAL:
                # ドローダウンイベント作成
                event = DrawdownEvent(
                    timestamp=current_time,
                    portfolio_value=current_value,
                    previous_peak=peak_value,
                    current_drawdown=current_drawdown,
                    drawdown_percentage=drawdown_percentage,
                    severity=severity,
                    affected_strategies=self._identify_affected_strategies(),
                    triggering_factor=self._identify_triggering_factor()
                )
                
                # 制御アクション実行
                self._execute_drawdown_control(event)
                
                return event
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return None
    
    def _determine_severity(self, drawdown_percentage: float) -> DrawdownSeverity:
        """ドローダウンの深刻度を判定"""
        # ボラティリティ調整
        adjusted_thresholds = self._adjust_thresholds_for_market_conditions()
        
        if drawdown_percentage >= adjusted_thresholds['emergency_threshold']:
            return DrawdownSeverity.EMERGENCY
        elif drawdown_percentage >= adjusted_thresholds['critical_threshold']:
            return DrawdownSeverity.CRITICAL
        elif drawdown_percentage >= adjusted_thresholds['warning_threshold']:
            return DrawdownSeverity.WARNING
        else:
            return DrawdownSeverity.NORMAL
    
    def _adjust_thresholds_for_market_conditions(self) -> Dict[str, float]:
        """市場環境に応じて閾値を調整"""
        base_thresholds = {
            'warning_threshold': self.thresholds.warning_threshold,
            'critical_threshold': self.thresholds.critical_threshold,
            'emergency_threshold': self.thresholds.emergency_threshold
        }
        
        try:
            # ボラティリティ調整
            if self.thresholds.volatility_adjustment:
                market_volatility = self._estimate_market_volatility()
                volatility_multiplier = 1.0 + (market_volatility - 0.2) * 0.5  # 基準20%
                volatility_multiplier = max(0.5, min(2.0, volatility_multiplier))
                
                for key in base_thresholds:
                    base_thresholds[key] *= volatility_multiplier
            
            # トレンド調整
            if self.thresholds.trend_adjustment:
                trend_factor = self._estimate_trend_factor()
                trend_multiplier = 1.0 + trend_factor * 0.3
                trend_multiplier = max(0.7, min(1.5, trend_multiplier))
                
                for key in base_thresholds:
                    base_thresholds[key] *= trend_multiplier
            
        except Exception as e:
            logger.warning(f"Error adjusting thresholds: {e}")
        
        return base_thresholds
    
    def _estimate_market_volatility(self) -> float:
        """市場ボラティリティの推定"""
        try:
            if len(self.performance_tracker['portfolio_history']) < 20:
                return 0.2  # デフォルト
            
            # 直近の価格変動から計算
            recent_values = [value for _, value in self.performance_tracker['portfolio_history'][-20:]]
            returns = [recent_values[i] / recent_values[i-1] - 1 for i in range(1, len(recent_values))]
            
            volatility = np.std(returns) * np.sqrt(252)  # 年率換算
            return max(0.05, min(1.0, volatility))
            
        except Exception as e:
            logger.warning(f"Error estimating market volatility: {e}")
            return 0.2
    
    def _estimate_trend_factor(self) -> float:
        """トレンド要因の推定（-1: 強い下降, 0: 中立, 1: 強い上昇）"""
        try:
            if len(self.performance_tracker['portfolio_history']) < 10:
                return 0.0
            
            # 直近の価格トレンドを分析
            recent_values = [value for _, value in self.performance_tracker['portfolio_history'][-10:]]
            
            # 線形回帰の傾きを計算
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # 正規化（-1 to 1の範囲）
            mean_value = np.mean(recent_values)
            if mean_value > 0:
                normalized_slope = slope / mean_value * 10  # スケール調整
                return max(-1.0, min(1.0, normalized_slope))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error estimating trend factor: {e}")
            return 0.0
    
    def _identify_affected_strategies(self) -> List[str]:
        """影響を受けている戦略を特定"""
        affected_strategies = []
        
        try:
            for strategy, history in self.performance_tracker['strategy_histories'].items():
                if not history:
                    continue
                
                peak = self.performance_tracker['strategy_peaks'].get(strategy, 0)
                current = history[-1][1] if history else 0
                
                if peak > 0:
                    strategy_drawdown = (peak - current) / peak
                    if strategy_drawdown >= self.thresholds.strategy_dd_threshold:
                        affected_strategies.append(strategy)
            
        except Exception as e:
            logger.warning(f"Error identifying affected strategies: {e}")
        
        return affected_strategies
    
    def _identify_triggering_factor(self) -> str:
        """引き金となった要因を特定"""
        try:
            # 簡易版: 最も損失の大きい戦略を特定
            max_loss_strategy = ""
            max_loss = 0.0
            
            for strategy, history in self.performance_tracker['strategy_histories'].items():
                if not history:
                    continue
                
                peak = self.performance_tracker['strategy_peaks'].get(strategy, 0)
                current = history[-1][1] if history else 0
                
                if peak > 0:
                    loss = (peak - current) / peak
                    if loss > max_loss:
                        max_loss = loss
                        max_loss_strategy = strategy
            
            if max_loss_strategy:
                return f"Primary factor: {max_loss_strategy} (DD: {max_loss:.2%})"
            else:
                return "General portfolio decline"
                
        except Exception as e:
            logger.warning(f"Error identifying triggering factor: {e}")
            return "Unknown factor"
    
    def _execute_drawdown_control(self, event: DrawdownEvent) -> DrawdownControlResult:
        """ドローダウン制御を実行"""
        try:
            with self._control_lock:
                logger.warning(f"Executing drawdown control for {event.severity.value} event")
                
                # アクション決定
                action = self._determine_control_action(event)
                
                # 現在のポジション取得
                original_positions = self._get_current_positions()
                
                # アクション実行
                adjusted_positions, success, error_msg = self._apply_control_action(
                    action, original_positions, event
                )
                
                # 結果記録
                result = DrawdownControlResult(
                    timestamp=datetime.now(),
                    event=event,
                    action_taken=action,
                    original_positions=original_positions,
                    adjusted_positions=adjusted_positions,
                    expected_impact=self._estimate_action_impact(action),
                    success=success,
                    error_message=error_msg
                )
                
                self.control_history.append(result)
                self.current_event = event if not success else None
                
                # 通知送信
                self._send_notification(result)
                
                logger.info(f"Drawdown control executed: {action.value}, Success: {success}")
                return result
                
        except Exception as e:
            logger.error(f"Error executing drawdown control: {e}")
            return DrawdownControlResult(
                timestamp=datetime.now(),
                event=event,
                action_taken=DrawdownControlAction.NO_ACTION,
                original_positions={},
                adjusted_positions={},
                expected_impact=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _determine_control_action(self, event: DrawdownEvent) -> DrawdownControlAction:
        """制御アクション決定"""
        severity = event.severity
        
        # 制御モードに応じたアクション決定
        action_map = {
            DrawdownSeverity.WARNING: {
                DrawdownControlMode.CONSERVATIVE: DrawdownControlAction.POSITION_REDUCTION_LIGHT,
                DrawdownControlMode.MODERATE: DrawdownControlAction.POSITION_REDUCTION_LIGHT,
                DrawdownControlMode.AGGRESSIVE: DrawdownControlAction.NO_ACTION
            },
            DrawdownSeverity.CRITICAL: {
                DrawdownControlMode.CONSERVATIVE: DrawdownControlAction.POSITION_REDUCTION_HEAVY,
                DrawdownControlMode.MODERATE: DrawdownControlAction.POSITION_REDUCTION_MODERATE,
                DrawdownControlMode.AGGRESSIVE: DrawdownControlAction.POSITION_REDUCTION_LIGHT
            },
            DrawdownSeverity.EMERGENCY: {
                DrawdownControlMode.CONSERVATIVE: DrawdownControlAction.EMERGENCY_STOP,
                DrawdownControlMode.MODERATE: DrawdownControlAction.EMERGENCY_STOP,
                DrawdownControlMode.AGGRESSIVE: DrawdownControlAction.POSITION_REDUCTION_HEAVY
            }
        }
        
        return action_map.get(severity, {}).get(self.control_mode, DrawdownControlAction.NO_ACTION)
    
    def _get_current_positions(self) -> Dict[str, float]:
        """現在のポジション取得"""
        try:
            # デフォルトポジション
            default_strategies = ['Momentum', 'Contrarian', 'Pairs_Trading']
            positions = {strategy: 0.33 for strategy in default_strategies}
            
            return positions
                
        except Exception as e:
            logger.warning(f"Error getting current positions: {e}")
            return {'Momentum': 0.33, 'Contrarian': 0.33, 'Pairs_Trading': 0.34}
    
    def _apply_control_action(self, 
                            action: DrawdownControlAction, 
                            original_positions: Dict[str, float],
                            event: DrawdownEvent) -> Tuple[Dict[str, float], bool, Optional[str]]:
        """制御アクション適用"""
        try:
            adjusted_positions = original_positions.copy()
            
            if action == DrawdownControlAction.NO_ACTION:
                return adjusted_positions, True, None
            
            elif action == DrawdownControlAction.POSITION_REDUCTION_LIGHT:
                # 軽度削減（15%）
                reduction_factor = 0.15
                for strategy in adjusted_positions:
                    adjusted_positions[strategy] *= (1 - reduction_factor)
                
            elif action == DrawdownControlAction.POSITION_REDUCTION_MODERATE:
                # 中度削減（40%）
                reduction_factor = 0.40
                for strategy in adjusted_positions:
                    adjusted_positions[strategy] *= (1 - reduction_factor)
                
            elif action == DrawdownControlAction.POSITION_REDUCTION_HEAVY:
                # 重度削減（60%）
                reduction_factor = 0.60
                for strategy in adjusted_positions:
                    adjusted_positions[strategy] *= (1 - reduction_factor)
                
            elif action == DrawdownControlAction.STRATEGY_SUSPENSION:
                # 影響戦略の停止
                for strategy in event.affected_strategies:
                    if strategy in adjusted_positions:
                        adjusted_positions[strategy] = 0.0
                        self.suspended_strategies.add(strategy)
                
                # 残存戦略に再配分
                self._rebalance_remaining_strategies(adjusted_positions)
                
            elif action == DrawdownControlAction.EMERGENCY_STOP:
                # 緊急全停止
                for strategy in adjusted_positions:
                    adjusted_positions[strategy] = 0.0
                
                # 調整マネージャーに停止指示
                if self.coordination_manager:
                    self._request_emergency_coordination_stop()
            
            return adjusted_positions, True, None
            
        except Exception as e:
            logger.error(f"Error applying control action {action}: {e}")
            return original_positions, False, str(e)
    
    def _rebalance_remaining_strategies(self, positions: Dict[str, float]):
        """残存戦略の再配分"""
        try:
            active_strategies = [s for s, pos in positions.items() if pos > 0]
            if not active_strategies:
                return
            
            # 等重量配分
            equal_weight = 1.0 / len(active_strategies)
            for strategy in active_strategies:
                positions[strategy] = equal_weight
                
            logger.info(f"Rebalanced to {len(active_strategies)} active strategies")
            
        except Exception as e:
            logger.warning(f"Error rebalancing strategies: {e}")
    
    def _request_emergency_coordination_stop(self):
        """緊急調整停止要求"""
        try:
            if hasattr(self.coordination_manager, 'shutdown_event'):
                self.coordination_manager.shutdown_event.set()
                logger.critical("Emergency coordination stop requested")
            else:
                logger.warning("Coordination manager shutdown not available")
                
        except Exception as e:
            logger.error(f"Error requesting emergency stop: {e}")
    
    def _estimate_action_impact(self, action: DrawdownControlAction) -> float:
        """アクション影響度推定"""
        impact_estimates = {
            DrawdownControlAction.NO_ACTION: 0.0,
            DrawdownControlAction.POSITION_REDUCTION_LIGHT: 0.15,
            DrawdownControlAction.POSITION_REDUCTION_MODERATE: 0.40,
            DrawdownControlAction.POSITION_REDUCTION_HEAVY: 0.60,
            DrawdownControlAction.STRATEGY_SUSPENSION: 0.30,
            DrawdownControlAction.EMERGENCY_STOP: 1.0,
            DrawdownControlAction.HEDGE_ACTIVATION: 0.20
        }
        return impact_estimates.get(action, 0.0)
    
    def _send_notification(self, result: DrawdownControlResult):
        """通知送信"""
        try:
            notification_config = self.config.get('notifications', {})
            if not notification_config.get('enabled', False):
                return
            
            message = (
                f"[ALERT] Drawdown Control Alert\n"
                f"Severity: {result.event.severity.value.upper()}\n"
                f"Drawdown: {result.event.drawdown_percentage:.2%}\n"
                f"Action: {result.action_taken.value}\n"
                f"Success: {result.success}\n"
                f"Time: {result.timestamp.strftime('%H:%M:%S')}"
            )
            
            logger.warning(f"Notification: {message}")
            
        except Exception as e:
            logger.warning(f"Error sending notification: {e}")
    
    def get_control_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """制御履歴取得"""
        recent_history = self.control_history[-limit:] if limit > 0 else self.control_history
        
        return [
            {
                'timestamp': result.timestamp.isoformat(),
                'severity': result.event.severity.value,
                'drawdown_percentage': result.event.drawdown_percentage,
                'action_taken': result.action_taken.value,
                'success': result.success,
                'expected_impact': result.expected_impact,
                'position_changes': result.get_position_changes()
            }
            for result in recent_history
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        try:
            if not self.performance_tracker['portfolio_history']:
                return {'status': 'no_data'}
            
            current_time, current_value = self.performance_tracker['portfolio_history'][-1]
            peak_value = self.performance_tracker['portfolio_peak']
            
            current_drawdown = 0.0
            if peak_value > 0:
                current_drawdown = (peak_value - current_value) / peak_value
            
            # 制御統計
            total_controls = len(self.control_history)
            successful_controls = sum(1 for r in self.control_history if r.success)
            
            return {
                'current_portfolio_value': current_value,
                'peak_portfolio_value': peak_value,
                'current_drawdown': current_drawdown,
                'drawdown_percentage': current_drawdown,
                'monitoring_status': 'active' if self.is_monitoring else 'inactive',
                'total_control_actions': total_controls,
                'successful_control_actions': successful_controls,
                'suspended_strategies': list(self.suspended_strategies),
                'last_update': self.performance_tracker['last_update'].isoformat(),
                'control_success_rate': successful_controls / total_controls if total_controls > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'status': 'error', 'message': str(e)}

# ユーティリティ関数
def create_default_drawdown_config() -> Dict[str, Any]:
    """デフォルト設定作成"""
    return {
        "control_mode": "moderate",
        "monitoring_interval": 60,
        "thresholds": {
            "warning_threshold": 0.05,
            "critical_threshold": 0.10, 
            "emergency_threshold": 0.15,
            "consecutive_loss_limit": 5,
            "daily_loss_threshold": 0.03,
            "strategy_dd_threshold": 0.08,
            "strategy_suspension_threshold": 0.12,
            "volatility_adjustment": True,
            "trend_adjustment": True
        },
        "actions": {
            "warning": "position_reduction_light",
            "critical": "position_reduction_moderate",
            "emergency": "emergency_stop"
        },
        "notifications": {
            "enabled": True,
            "email_alerts": False,
            "webhook_url": None
        }
    }

if __name__ == "__main__":
    # 基本テスト
    print("=" * 50)
    print("Drawdown Controller - Basic Test")
    print("=" * 50)
    
    try:
        # 設定作成
        config_data = create_default_drawdown_config()
        
        # コントローラー作成
        controller = DrawdownController()
        
        print(f"[OK] Drawdown Controller initialized")
        print(f"Control Mode: {controller.control_mode.value}")
        print(f"Thresholds: Warning={controller.thresholds.warning_threshold:.1%}, "
              f"Critical={controller.thresholds.critical_threshold:.1%}, "
              f"Emergency={controller.thresholds.emergency_threshold:.1%}")
        
        # 監視開始
        initial_value = 1000000.0
        controller.start_monitoring(initial_value)
        print(f"[CHART] Monitoring started with initial value: ${initial_value:,.0f}")
        
        # シミュレーションテスト
        test_values = [
            (990000, "Normal decline"),
            (950000, "Warning level"),
            (900000, "Critical level"), 
            (850000, "Emergency level"),
            (880000, "Recovery")
        ]
        
        for value, description in test_values:
            print(f"\n🔄 Updating portfolio value: ${value:,.0f} ({description})")
            controller.update_portfolio_value(value)
            
            # 短い待機
            time.sleep(2)
            
            # サマリー表示
            summary = controller.get_performance_summary()
            print(f"   Current DD: {summary.get('drawdown_percentage', 0):.2%}")
            print(f"   Control Actions: {summary.get('total_control_actions', 0)}")
        
        # 制御履歴表示
        history = controller.get_control_history()
        if history:
            print(f"\n[LIST] Control History ({len(history)} actions):")
            for i, action in enumerate(history[-3:], 1):  # 最新3件
                print(f"  {i}. {action['timestamp'][:19]} - "
                      f"{action['action_taken']} (DD: {action['drawdown_percentage']:.2%})")
        
        # 監視停止
        controller.stop_monitoring()
        print(f"\n[OK] Drawdown Controller test completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
