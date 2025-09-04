"""
Level 3: Emergency Constraints
レベル3：緊急制約 - RiskManagementシステムとの統合による緊急時のみの実行
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from .base_level import BaseSwitchDecisionLevel
from ..decision_context import DecisionContext, HierarchicalDecisionResult


class Level3EmergencyConstraints(BaseSwitchDecisionLevel):
    """
    レベル3：緊急制約
    
    RiskManagementシステムと統合し、緊急時のみに実行される
    リスク制約に基づく戦略切替決定を行う
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Level3固有のデフォルト設定を最初に設定
        self.max_drawdown_threshold = 0.05      # 最大ドローダウン閾値 (5%)
        self.var_threshold = 0.03               # VaR閾値 (3%)
        self.volatility_threshold = 0.25        # ボラティリティ閾値 (25%)
        self.emergency_confidence = 0.9         # 緊急時の最小信頼度
        self.risk_override_threshold = 0.95     # リスクレベルでの強制上書き閾値
        self.emergency_cooldown = 300           # 緊急執行後のクールダウン時間（秒）
        
        # 基底クラスの初期化を呼び出し
        super().__init__(level_number=3, config=config)
        
    def _load_level_config(self) -> None:
        """レベル3固有の設定を読み込み"""
        if 'emergency_constraints' in self.config:
            emergency_config = self.config['emergency_constraints']
            self.max_drawdown_threshold = emergency_config.get('max_drawdown_threshold', self.max_drawdown_threshold)
            self.var_threshold = emergency_config.get('var_threshold', self.var_threshold)
            self.volatility_threshold = emergency_config.get('volatility_threshold', self.volatility_threshold)
            self.emergency_confidence = emergency_config.get('emergency_confidence', self.emergency_confidence)
            self.risk_override_threshold = emergency_config.get('risk_override_threshold', self.risk_override_threshold)
        
        self.logger.info(f"Level3 Config: max_drawdown={self.max_drawdown_threshold}, "
                        f"var_threshold={self.var_threshold}")
    
    def should_activate(self, context: DecisionContext) -> bool:
        """
        Level3がアクティベートするべきかどうかを判定（緊急時のみ）
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            bool: アクティベートするべきかどうか
        """
        # 緊急シグナルの確認
        if context.has_emergency_signal():
            return True
        
        # リスクメトリクスによる緊急判定
        emergency_conditions = [
            self._check_drawdown_emergency(context),
            self._check_var_emergency(context),
            self._check_volatility_emergency(context),
            self._check_correlation_emergency(context),
            self._check_exposure_emergency(context)
        ]
        
        # いずれかの緊急条件が満たされた場合にアクティベート
        return any(emergency_conditions)
    
    def get_priority_score(self, context: DecisionContext) -> float:
        """Level3は最高優先度"""
        if self.should_activate(context):
            return 100.0  # 緊急時は最高優先度
        return 0.0  # 非緊急時は非アクティブ
    
    def evaluate_switch_condition(self, context: DecisionContext) -> HierarchicalDecisionResult:
        """
        緊急制約に基づいて切替条件を評価
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        try:
            # 緊急状況の分析
            emergency_analysis = self._analyze_emergency_situation(context)
            
            # 緊急対応戦略の特定
            emergency_strategy = self._identify_emergency_strategy(context, emergency_analysis)
            
            # 緊急信頼度の計算
            emergency_confidence = self._calculate_emergency_confidence(context, emergency_strategy, emergency_analysis)
            
            # 緊急決定の生成
            decision = self._generate_emergency_decision(context, emergency_strategy, emergency_confidence, emergency_analysis)
            
            # 決定の妥当性検証
            if self.validate_decision(decision, context):
                self.log_decision(decision, context)
                return decision
            else:
                return self._create_emergency_stop_decision("Emergency validation failed")
        
        except Exception as e:
            self.logger.error(f"Level3 evaluation error: {e}")
            return self._create_emergency_stop_decision(f"Error in emergency evaluation: {str(e)}")
    
    def _analyze_emergency_situation(self, context: DecisionContext) -> Dict[str, Any]:
        """
        緊急状況を分析
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            Dict: 緊急状況分析結果
        """
        analysis = {}
        
        try:
            # リスクメトリクスの取得
            current_drawdown = context.get_risk_metric('current_drawdown')
            portfolio_var = context.get_risk_metric('portfolio_var')
            portfolio_volatility = context.get_risk_metric('portfolio_volatility')
            correlation_risk = context.get_risk_metric('correlation_risk')
            exposure_concentration = context.get_risk_metric('exposure_concentration')
            
            # 緊急レベルの評価
            emergency_factors = {
                'drawdown_emergency': self._evaluate_drawdown_severity(current_drawdown),
                'var_emergency': self._evaluate_var_severity(portfolio_var),
                'volatility_emergency': self._evaluate_volatility_severity(portfolio_volatility),
                'correlation_emergency': self._evaluate_correlation_severity(correlation_risk),
                'exposure_emergency': self._evaluate_exposure_severity(exposure_concentration)
            }
            
            # 総合緊急レベル
            emergency_scores = [score for score in emergency_factors.values() if score > 0]
            overall_emergency_level = max(emergency_scores) if emergency_scores else 0.0
            
            # 緊急シグナル情報
            emergency_signals = context.emergency_signals or {}
            
            analysis = {
                'emergency_factors': emergency_factors,
                'overall_emergency_level': overall_emergency_level,
                'emergency_signals': emergency_signals,
                'max_risk_factor': max(emergency_factors.keys(), key=lambda k: emergency_factors[k]),
                'requires_immediate_action': overall_emergency_level >= 0.8,
                'risk_metrics': {
                    'current_drawdown': current_drawdown,
                    'portfolio_var': portfolio_var,
                    'portfolio_volatility': portfolio_volatility,
                    'correlation_risk': correlation_risk,
                    'exposure_concentration': exposure_concentration
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Emergency analysis error: {e}")
            analysis = {
                'emergency_factors': {},
                'overall_emergency_level': 1.0,  # エラー時は緊急とみなす
                'emergency_signals': {},
                'max_risk_factor': 'unknown',
                'requires_immediate_action': True,
                'risk_metrics': {}
            }
        
        return analysis
    
    def _identify_emergency_strategy(self, context: DecisionContext, 
                                   emergency_analysis: Dict[str, Any]) -> Optional[str]:
        """
        緊急対応戦略を特定
        
        Args:
            context: 決定コンテキスト
            emergency_analysis: 緊急状況分析結果
            
        Returns:
            Optional[str]: 緊急対応戦略名
        """
        max_risk_factor = emergency_analysis['max_risk_factor']
        emergency_level = emergency_analysis['overall_emergency_level']
        
        # 極度の緊急時は全ポジション解消
        if emergency_level >= 0.95:
            return 'emergency_liquidation'
        
        # リスク要因別の対応戦略
        emergency_strategies = {
            'drawdown_emergency': self._get_drawdown_recovery_strategy(context),
            'var_emergency': self._get_var_mitigation_strategy(context),
            'volatility_emergency': self._get_volatility_reduction_strategy(context),
            'correlation_emergency': self._get_correlation_hedge_strategy(context),
            'exposure_emergency': self._get_exposure_diversification_strategy(context)
        }
        
        # 最大リスク要因に対応する戦略を選択
        emergency_strategy = emergency_strategies.get(max_risk_factor)
        
        # 戦略が有効かどうかチェック
        if emergency_strategy and emergency_strategy in context.strategies_data:
            return emergency_strategy
        
        # デフォルト：最も安全な戦略を選択
        return self._get_safest_strategy(context)
    
    def _calculate_emergency_confidence(self, context: DecisionContext, 
                                      emergency_strategy: Optional[str],
                                      emergency_analysis: Dict[str, Any]) -> float:
        """
        緊急信頼度を計算
        
        Args:
            context: 決定コンテキスト
            emergency_strategy: 緊急対応戦略
            emergency_analysis: 緊急状況分析結果
            
        Returns:
            float: 信頼度 (0.0-1.0)
        """
        if not emergency_strategy:
            return 0.0
        
        try:
            # 緊急レベルに基づく基本信頼度
            base_confidence = emergency_analysis['overall_emergency_level']
            
            # 戦略の緊急対応能力
            if emergency_strategy == 'emergency_liquidation':
                strategy_confidence = 1.0  # 全解消は確実
            else:
                strategy_data = context.strategies_data.get(emergency_strategy, {})
                strategy_confidence = strategy_data.get('emergency_effectiveness', 0.5)
            
            # リスクメトリクスの深刻度
            risk_severity = self._calculate_risk_severity(emergency_analysis['risk_metrics'])
            
            # 総合信頼度
            total_confidence = (
                base_confidence * 0.4 +
                strategy_confidence * 0.4 +
                risk_severity * 0.2
            )
            
            # 緊急時は高い信頼度を要求
            return min(max(total_confidence, self.emergency_confidence), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Emergency confidence calculation error: {e}")
            return self.emergency_confidence
    
    def _generate_emergency_decision(self, context: DecisionContext, 
                                   emergency_strategy: Optional[str],
                                   emergency_confidence: float,
                                   emergency_analysis: Dict[str, Any]) -> HierarchicalDecisionResult:
        """
        緊急決定を生成
        
        Args:
            context: 決定コンテキスト
            emergency_strategy: 緊急対応戦略
            emergency_confidence: 緊急信頼度
            emergency_analysis: 緊急状況分析結果
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        emergency_level = emergency_analysis['overall_emergency_level']
        max_risk_factor = emergency_analysis['max_risk_factor']
        
        # 極度の緊急時は強制停止
        if emergency_level >= 0.95 or not emergency_strategy:
            return self._create_emergency_stop_decision(
                f"Critical emergency detected: {max_risk_factor} (level: {emergency_level:.3f})"
            )
        
        # 緊急切替決定
        if emergency_strategy and emergency_confidence >= self.emergency_confidence:
            reasoning = (f"Emergency switch to {emergency_strategy} due to {max_risk_factor}. "
                        f"Emergency level: {emergency_level:.3f}")
            
            return HierarchicalDecisionResult(
                decision_level=3,
                decision_type='switch',
                target_strategy=emergency_strategy,
                confidence=emergency_confidence,
                reasoning=reasoning,
                override_conditions=True,  # 緊急時は常に上位レベルを上書き
                metadata={
                    'emergency_analysis': emergency_analysis,
                    'risk_factor': max_risk_factor,
                    'emergency_level': emergency_level
                }
            )
        else:
            return self._create_emergency_stop_decision(
                f"Emergency detected but no suitable strategy (confidence: {emergency_confidence:.3f})"
            )
    
    def _create_emergency_stop_decision(self, reason: str) -> HierarchicalDecisionResult:
        """緊急停止決定を生成"""
        return HierarchicalDecisionResult(
            decision_level=3,
            decision_type='emergency_stop',
            confidence=1.0,
            reasoning=f"Level3 emergency stop: {reason}",
            override_conditions=True
        )
    
    # 緊急条件チェックメソッド群
    def _check_drawdown_emergency(self, context: DecisionContext) -> bool:
        """ドローダウン緊急条件チェック"""
        current_drawdown = context.get_risk_metric('current_drawdown')
        return current_drawdown is not None and current_drawdown > self.max_drawdown_threshold
    
    def _check_var_emergency(self, context: DecisionContext) -> bool:
        """VaR緊急条件チェック"""
        portfolio_var = context.get_risk_metric('portfolio_var')
        return portfolio_var is not None and portfolio_var > self.var_threshold
    
    def _check_volatility_emergency(self, context: DecisionContext) -> bool:
        """ボラティリティ緊急条件チェック"""
        portfolio_volatility = context.get_risk_metric('portfolio_volatility')
        return portfolio_volatility is not None and portfolio_volatility > self.volatility_threshold
    
    def _check_correlation_emergency(self, context: DecisionContext) -> bool:
        """相関緊急条件チェック"""
        correlation_risk = context.get_risk_metric('correlation_risk')
        return correlation_risk is not None and correlation_risk > 0.8
    
    def _check_exposure_emergency(self, context: DecisionContext) -> bool:
        """エクスポージャー緊急条件チェック"""
        exposure_concentration = context.get_risk_metric('exposure_concentration')
        return exposure_concentration is not None and exposure_concentration > 0.7
    
    # 緊急度評価メソッド群
    def _evaluate_drawdown_severity(self, drawdown: Optional[float]) -> float:
        """ドローダウンの深刻度評価"""
        if drawdown is None:
            return 0.0
        if drawdown <= self.max_drawdown_threshold:
            return 0.0
        return min((drawdown - self.max_drawdown_threshold) / self.max_drawdown_threshold, 1.0)
    
    def _evaluate_var_severity(self, var: Optional[float]) -> float:
        """VaRの深刻度評価"""
        if var is None:
            return 0.0
        if var <= self.var_threshold:
            return 0.0
        return min((var - self.var_threshold) / self.var_threshold, 1.0)
    
    def _evaluate_volatility_severity(self, volatility: Optional[float]) -> float:
        """ボラティリティの深刻度評価"""
        if volatility is None:
            return 0.0
        if volatility <= self.volatility_threshold:
            return 0.0
        return min((volatility - self.volatility_threshold) / self.volatility_threshold, 1.0)
    
    def _evaluate_correlation_severity(self, correlation: Optional[float]) -> float:
        """相関の深刻度評価"""
        if correlation is None:
            return 0.0
        return max(correlation - 0.5, 0.0) / 0.5 if correlation > 0.5 else 0.0
    
    def _evaluate_exposure_severity(self, exposure: Optional[float]) -> float:
        """エクスポージャーの深刻度評価"""
        if exposure is None:
            return 0.0
        return max(exposure - 0.5, 0.0) / 0.5 if exposure > 0.5 else 0.0
    
    # 緊急対応戦略選択メソッド群
    def _get_drawdown_recovery_strategy(self, context: DecisionContext) -> Optional[str]:
        """ドローダウン回復戦略"""
        # 安定性の高い戦略を選択
        strategies_with_stability = [
            (name, data.get('stability_score', 0.0))
            for name, data in context.strategies_data.items()
        ]
        if strategies_with_stability:
            return max(strategies_with_stability, key=lambda x: x[1])[0]
        return None
    
    def _get_var_mitigation_strategy(self, context: DecisionContext) -> Optional[str]:
        """VaR軽減戦略"""
        # 低ボラティリティ戦略を選択
        strategies_with_vol = [
            (name, data.get('volatility_score', 1.0))
            for name, data in context.strategies_data.items()
        ]
        if strategies_with_vol:
            return min(strategies_with_vol, key=lambda x: x[1])[0]
        return None
    
    def _get_volatility_reduction_strategy(self, context: DecisionContext) -> Optional[str]:
        """ボラティリティ削減戦略"""
        return self._get_var_mitigation_strategy(context)
    
    def _get_correlation_hedge_strategy(self, context: DecisionContext) -> Optional[str]:
        """相関ヘッジ戦略"""
        # 低相関戦略を選択
        strategies_with_corr = [
            (name, data.get('correlation_score', 1.0))
            for name, data in context.strategies_data.items()
        ]
        if strategies_with_corr:
            return min(strategies_with_corr, key=lambda x: x[1])[0]
        return None
    
    def _get_exposure_diversification_strategy(self, context: DecisionContext) -> Optional[str]:
        """エクスポージャー分散戦略"""
        # 分散効果の高い戦略を選択
        strategies_with_div = [
            (name, data.get('diversification_score', 0.0))
            for name, data in context.strategies_data.items()
        ]
        if strategies_with_div:
            return max(strategies_with_div, key=lambda x: x[1])[0]
        return None
    
    def _get_safest_strategy(self, context: DecisionContext) -> Optional[str]:
        """最も安全な戦略を取得"""
        safety_scores = {}
        for name, data in context.strategies_data.items():
            safety_score = (
                data.get('stability_score', 0.0) * 0.4 +
                (1.0 - data.get('volatility_score', 1.0)) * 0.3 +
                data.get('diversification_score', 0.0) * 0.2 +
                (1.0 - data.get('correlation_score', 1.0)) * 0.1
            )
            safety_scores[name] = safety_score
        
        if safety_scores:
            return max(safety_scores.keys(), key=lambda k: safety_scores[k])
        return None
    
    def _calculate_risk_severity(self, risk_metrics: Dict[str, Any]) -> float:
        """リスクメトリクスの総合深刻度を計算"""
        severities = []
        
        # 各リスクメトリクスの深刻度を計算
        if 'current_drawdown' in risk_metrics:
            severities.append(self._evaluate_drawdown_severity(risk_metrics['current_drawdown']))
        if 'portfolio_var' in risk_metrics:
            severities.append(self._evaluate_var_severity(risk_metrics['portfolio_var']))
        if 'portfolio_volatility' in risk_metrics:
            severities.append(self._evaluate_volatility_severity(risk_metrics['portfolio_volatility']))
        if 'correlation_risk' in risk_metrics:
            severities.append(self._evaluate_correlation_severity(risk_metrics['correlation_risk']))
        if 'exposure_concentration' in risk_metrics:
            severities.append(self._evaluate_exposure_severity(risk_metrics['exposure_concentration']))
        
        # 最大深刻度を返す
        return max(severities) if severities else 0.0
