"""
統合リスク管理システム
Phase 3: 実行・制御システム構築 - リスク評価・実行制御
DrawdownController + EnhancedRiskManagement の統合

Author: imega
Created: 2025-10-18
Modified: 2025-10-18
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# 既存リスク管理モジュール
from main_system.risk_management.drawdown_controller import (
    DrawdownController, DrawdownSeverity, DrawdownControlAction
)

# EnhancedRiskManagementSystemのインポート試行
try:
    from config.enhanced_risk_management.enhanced_risk_management_system import (
        EnhancedRiskManagementSystem
    )
    HAS_ENHANCED_RISK_SYSTEM = True
except ImportError:
    EnhancedRiskManagementSystem = None
    HAS_ENHANCED_RISK_SYSTEM = False


class RiskAssessmentLevel(Enum):
    """リスク評価レベル"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class UnifiedRiskManager:
    """統合リスク管理クラス - DrawdownController + EnhancedRiskManagement 統合"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: リスク管理設定
        """
        self.config = config or {}
        self.logger = setup_logger(
            "UnifiedRiskManager",
            log_file="logs/unified_risk_manager.log"
        )
        
        # コンポーネント初期化
        try:
            # ドローダウン制御（パラメータなしで初期化）
            self.drawdown_controller = DrawdownController()
            self.logger.info("DrawdownController initialized")
            
            # 強化リスク管理システム
            if HAS_ENHANCED_RISK_SYSTEM and self.config.get('use_enhanced_risk', False):
                self.enhanced_risk_system = EnhancedRiskManagementSystem()
                self.logger.info("EnhancedRiskManagementSystem initialized")
            else:
                self.enhanced_risk_system = None
                self.logger.info("EnhancedRiskManagementSystem not available or disabled")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk management components: {e}")
            raise
        
        # リスク評価履歴
        self.risk_assessment_history = []
    
    def assess_execution_risk(
        self,
        strategy_selection: Dict[str, Any],
        stock_data: pd.DataFrame,
        portfolio_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        実行前リスク評価
        
        Args:
            strategy_selection: 戦略選択結果
            stock_data: 株価データ
            portfolio_value: 現在のポートフォリオ価値
        
        Returns:
            リスク評価結果
        """
        self.logger.info("Executing pre-execution risk assessment")
        
        assessment = {
            'timestamp': datetime.now(),
            'overall_risk_level': RiskAssessmentLevel.SAFE,
            'drawdown_assessment': None,
            'enhanced_risk_assessment': None,
            'execution_approval': True,
            'risk_warnings': [],
            'recommended_actions': []
        }
        
        try:
            # 1. ドローダウンリスク評価
            if portfolio_value is not None:
                drawdown_status = self._assess_drawdown_risk(portfolio_value)
                assessment['drawdown_assessment'] = drawdown_status
                
                # ドローダウンレベルに応じたリスクレベル設定
                if drawdown_status['severity'] == DrawdownSeverity.EMERGENCY:
                    assessment['overall_risk_level'] = RiskAssessmentLevel.CRITICAL
                    assessment['execution_approval'] = False
                    assessment['risk_warnings'].append("EMERGENCY DRAWDOWN LEVEL DETECTED")
                elif drawdown_status['severity'] == DrawdownSeverity.CRITICAL:
                    assessment['overall_risk_level'] = RiskAssessmentLevel.DANGER
                    assessment['recommended_actions'].append("Reduce position sizes significantly")
                elif drawdown_status['severity'] == DrawdownSeverity.WARNING:
                    assessment['overall_risk_level'] = RiskAssessmentLevel.WARNING
                    assessment['recommended_actions'].append("Monitor positions closely")
            
            # 2. 強化リスク管理システム評価
            if self.enhanced_risk_system is not None:
                enhanced_assessment = self._assess_enhanced_risk(
                    strategy_selection, stock_data
                )
                assessment['enhanced_risk_assessment'] = enhanced_assessment
                
                # 強化評価を統合
                if enhanced_assessment.get('high_risk_detected', False):
                    if assessment['overall_risk_level'] == RiskAssessmentLevel.SAFE:
                        assessment['overall_risk_level'] = RiskAssessmentLevel.CAUTION
                    assessment['risk_warnings'].extend(
                        enhanced_assessment.get('warnings', [])
                    )
            
            # 3. 戦略選択リスク評価
            strategy_risk = self._assess_strategy_selection_risk(strategy_selection)
            assessment['strategy_risk_score'] = strategy_risk
            
            if strategy_risk > 0.7:
                assessment['overall_risk_level'] = max(
                    assessment['overall_risk_level'],
                    RiskAssessmentLevel.WARNING,
                    key=lambda x: list(RiskAssessmentLevel).index(x)
                )
                assessment['risk_warnings'].append(
                    f"High strategy risk score: {strategy_risk:.2f}"
                )
            
            # 4. 最終判定
            self._finalize_risk_assessment(assessment)
            
            # 履歴に記録
            self.risk_assessment_history.append(assessment)
            
            self.logger.info(
                f"Risk assessment completed: Level={assessment['overall_risk_level'].value}, "
                f"Approval={assessment['execution_approval']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during risk assessment: {e}")
            # エラー時は安全サイドで実行拒否
            assessment['overall_risk_level'] = RiskAssessmentLevel.CRITICAL
            assessment['execution_approval'] = False
            assessment['risk_warnings'].append(f"Risk assessment error: {str(e)}")
        
        return assessment
    
    def _assess_drawdown_risk(self, portfolio_value: float) -> Dict[str, Any]:
        """ドローダウンリスク評価"""
        try:
            # ドローダウンコントローラーで評価
            current_drawdown = self.drawdown_controller.calculate_current_drawdown(
                portfolio_value
            )
            
            severity = self.drawdown_controller.assess_drawdown_severity(
                current_drawdown
            )
            
            action = self.drawdown_controller.determine_control_action(severity)
            
            return {
                'current_drawdown': current_drawdown,
                'severity': severity,
                'recommended_action': action,
                'max_threshold': self.drawdown_controller.max_drawdown_threshold
            }
        except Exception as e:
            self.logger.error(f"Drawdown assessment error: {e}")
            return {
                'current_drawdown': 0.0,
                'severity': DrawdownSeverity.NORMAL,
                'recommended_action': DrawdownControlAction.NO_ACTION,
                'error': str(e)
            }
    
    def _assess_enhanced_risk(
        self,
        strategy_selection: Dict[str, Any],
        stock_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """強化リスク管理システム評価"""
        try:
            # 強化リスクシステムで評価
            risk_metrics = self.enhanced_risk_system.calculate_risk_metrics(
                strategy_selection, stock_data
            )
            
            high_risk_detected = any(
                metric.get('value', 0) > metric.get('threshold', 1.0)
                for metric in risk_metrics.values()
            )
            
            warnings = []
            if high_risk_detected:
                warnings = [
                    f"{name}: {metric.get('value', 0):.2f} > {metric.get('threshold', 0):.2f}"
                    for name, metric in risk_metrics.items()
                    if metric.get('value', 0) > metric.get('threshold', 1.0)
                ]
            
            return {
                'risk_metrics': risk_metrics,
                'high_risk_detected': high_risk_detected,
                'warnings': warnings
            }
        except Exception as e:
            self.logger.error(f"Enhanced risk assessment error: {e}")
            return {
                'risk_metrics': {},
                'high_risk_detected': False,
                'warnings': [],
                'error': str(e)
            }
    
    def _assess_strategy_selection_risk(
        self,
        strategy_selection: Dict[str, Any]
    ) -> float:
        """戦略選択リスクスコア計算"""
        try:
            selected_strategies = strategy_selection.get('selected_strategies', [])
            strategy_weights = strategy_selection.get('strategy_weights', {})
            confidence_level = strategy_selection.get('confidence_level', 0.5)
            
            # リスク要因
            risk_factors = []
            
            # 1. 戦略数リスク（少なすぎる・多すぎる）
            if len(selected_strategies) < 2:
                risk_factors.append(0.3)  # 分散不足
            elif len(selected_strategies) > 5:
                risk_factors.append(0.2)  # 過度な複雑性
            
            # 2. 重み集中リスク
            if strategy_weights:
                max_weight = max(strategy_weights.values())
                if max_weight > 0.5:
                    risk_factors.append(0.3)  # 重み集中
            
            # 3. 信頼度リスク
            if confidence_level < 0.5:
                risk_factors.append(0.4)  # 低信頼度
            
            # 総合リスクスコア（0.0 = 低リスク、1.0 = 高リスク）
            risk_score = min(1.0, sum(risk_factors))
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Strategy risk assessment error: {e}")
            return 0.5  # デフォルト中程度リスク
    
    def _finalize_risk_assessment(self, assessment: Dict[str, Any]) -> None:
        """リスク評価の最終判定"""
        # クリティカルレベルでは実行拒否
        if assessment['overall_risk_level'] == RiskAssessmentLevel.CRITICAL:
            assessment['execution_approval'] = False
            if not assessment['recommended_actions']:
                assessment['recommended_actions'].append("HALT ALL TRADING IMMEDIATELY")
        
        # デンジャーレベルでは慎重な実行
        elif assessment['overall_risk_level'] == RiskAssessmentLevel.DANGER:
            assessment['recommended_actions'].append("Execute with significantly reduced position sizes")
        
        # 警告レベルでは注意して実行
        elif assessment['overall_risk_level'] == RiskAssessmentLevel.WARNING:
            assessment['recommended_actions'].append("Execute with reduced position sizes")
    
    def check_trade_risk(
        self,
        order_dict: Dict[str, Any],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        個別取引のリスク評価（Phase 5-B-4追加）
        
        Args:
            order_dict: 取引オーダー辞書
            portfolio_value: 現在のポートフォリオ価値
        
        Returns:
            (can_execute: bool, reason: str)
        """
        try:
            # ドローダウンチェック
            current_dd = self.drawdown_controller.calculate_current_drawdown(portfolio_value)
            max_threshold = self.drawdown_controller.max_drawdown_threshold
            
            # [Phase 5-B-4] ログ出力（copilot-instructions.md準拠）
            self.logger.info(
                f"[TRADE_RISK_CHECK] {order_dict.get('symbol', 'UNKNOWN')} {order_dict.get('action', 'UNKNOWN')}, "
                f"Portfolio DD: {current_dd:.2%}, Threshold: {max_threshold:.2%}"
            )
            
            # 緊急停止閾値チェック
            if current_dd >= max_threshold:
                reason = f"Portfolio drawdown {current_dd:.2%} >= {max_threshold:.2%}"
                self.logger.warning(f"[TRADE_RISK_BLOCKED] {reason}")
                return False, reason
            
            # 警告レベルチェック
            warning_threshold = 0.10
            if current_dd >= warning_threshold:
                self.logger.warning(
                    f"[TRADE_RISK_WARNING] Portfolio DD {current_dd:.2%} >= {warning_threshold:.2%}"
                )
            
            return True, "Trade risk check passed"
            
        except Exception as e:
            self.logger.error(f"Trade risk check error: {e}")
            return True, f"Risk check error (allowing trade): {e}"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """リスク評価サマリー取得"""
        if not self.risk_assessment_history:
            return {
                'total_assessments': 0,
                'recent_risk_level': None,
                'approval_rate': 0.0
            }
        
        recent = self.risk_assessment_history[-1]
        approved_count = sum(
            1 for a in self.risk_assessment_history if a['execution_approval']
        )
        
        return {
            'total_assessments': len(self.risk_assessment_history),
            'recent_risk_level': recent['overall_risk_level'].value,
            'approval_rate': approved_count / len(self.risk_assessment_history),
            'recent_warnings': recent.get('risk_warnings', [])
        }


def test_unified_risk_manager():
    """UnifiedRiskManager テスト"""
    print("UnifiedRiskManager テスト開始")
    print("=" * 80)
    
    # テスト用設定
    config = {
        'max_drawdown_threshold': 0.15,
        'warning_threshold': 0.10,
        'critical_threshold': 0.125,
        'use_enhanced_risk': False
    }
    
    # リスクマネージャー作成
    risk_manager = UnifiedRiskManager(config)
    
    # サンプル戦略選択
    strategy_selection = {
        'selected_strategies': ['VWAPBreakoutStrategy', 'MomentumInvestingStrategy'],
        'strategy_weights': {
            'VWAPBreakoutStrategy': 0.6,
            'MomentumInvestingStrategy': 0.4
        },
        'confidence_level': 0.75
    }
    
    # サンプル株価データ
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    }, index=dates)
    
    # リスク評価実行
    assessment = risk_manager.assess_execution_risk(
        strategy_selection=strategy_selection,
        stock_data=sample_data,
        portfolio_value=1000000
    )
    
    # 結果出力
    print("\n=== リスク評価結果 ===")
    print(f"総合リスクレベル: {assessment['overall_risk_level'].value}")
    print(f"実行承認: {assessment['execution_approval']}")
    print(f"戦略リスクスコア: {assessment['strategy_risk_score']:.2f}")
    
    if assessment['risk_warnings']:
        print("\nリスク警告:")
        for warning in assessment['risk_warnings']:
            print(f"  - {warning}")
    
    if assessment['recommended_actions']:
        print("\n推奨アクション:")
        for action in assessment['recommended_actions']:
            print(f"  - {action}")
    
    # サマリー取得
    summary = risk_manager.get_risk_summary()
    print("\n=== リスク評価サマリー ===")
    print(f"総評価回数: {summary['total_assessments']}")
    print(f"最新リスクレベル: {summary['recent_risk_level']}")
    print(f"承認率: {summary['approval_rate']:.2%}")
    
    print("\n=== テスト完了 ===")
    return assessment


if __name__ == "__main__":
    test_unified_risk_manager()
