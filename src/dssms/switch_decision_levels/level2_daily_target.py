"""
Level 2: Daily Target
レベル2：日次ターゲット - DSSMSSwitchCoordinatorV2のインテリジェントターゲットシステムとの統合
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from .base_level import BaseSwitchDecisionLevel
from ..decision_context import DecisionContext, HierarchicalDecisionResult


class Level2DailyTarget(BaseSwitchDecisionLevel):
    """
    レベル2：日次ターゲット
    
    DSSMSSwitchCoordinatorV2のインテリジェントターゲットシステムと統合し、
    日次目標達成に向けた戦略切替決定を行う
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Level2固有のデフォルト設定を最初に設定
        self.daily_target_threshold = 0.02      # 日次目標閾値 (2%)
        self.target_achievement_ratio = 0.8     # 目標達成率閾値
        self.time_pressure_factor = 0.5         # 時間圧力係数
        self.min_confidence_override = 0.8      # Level1を上書きする最小信頼度
        self.market_session_hours = 6.5         # 市場セッション時間
        
        # 基底クラスの初期化を呼び出し
        super().__init__(level_number=2, config=config)
        
    def _load_level_config(self) -> None:
        """レベル2固有の設定を読み込み"""
        if 'daily_target' in self.config:
            target_config = self.config['daily_target']
            self.daily_target_threshold = target_config.get('target_threshold', self.daily_target_threshold)
            self.target_achievement_ratio = target_config.get('achievement_ratio', self.target_achievement_ratio)
            self.time_pressure_factor = target_config.get('time_pressure_factor', self.time_pressure_factor)
            self.min_confidence_override = target_config.get('min_confidence_override', self.min_confidence_override)
        
        self.logger.info(f"Level2 Config: daily_target_threshold={self.daily_target_threshold}, "
                        f"target_achievement_ratio={self.target_achievement_ratio}")
    
    def should_activate(self, context: DecisionContext) -> bool:
        """
        Level2がアクティベートするべきかどうかを判定
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            bool: アクティベートするべきかどうか
        """
        # 日次ターゲット関連の条件をチェック
        current_performance = context.get_portfolio_metric('daily_return')
        daily_target = context.get_portfolio_metric('daily_target')
        
        if current_performance is None or daily_target is None:
            return False
        
        # 目標達成状況に基づいてアクティベーション判定
        achievement_ratio = current_performance / daily_target if daily_target != 0 else 0
        
        # 目標未達成、または大幅未達成の場合にアクティベート
        return achievement_ratio < self.target_achievement_ratio
    
    def evaluate_switch_condition(self, context: DecisionContext) -> HierarchicalDecisionResult:
        """
        日次ターゲットに基づいて切替条件を評価
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        try:
            # 日次ターゲット分析
            target_analysis = self._analyze_daily_target_status(context)
            
            # 時間圧力の評価
            time_pressure = self._calculate_time_pressure(context)
            
            # 最適戦略の特定（ターゲット達成重視）
            target_optimal_strategy = self._identify_target_optimal_strategy(context, target_analysis, time_pressure)
            
            # 切替信頼度の計算
            switch_confidence = self._calculate_target_confidence(context, target_optimal_strategy, target_analysis, time_pressure)
            
            # 決定の生成
            decision = self._generate_target_decision(context, target_optimal_strategy, switch_confidence, target_analysis, time_pressure)
            
            # 決定の妥当性検証
            if self.validate_decision(decision, context):
                self.log_decision(decision, context)
                return decision
            else:
                return self._create_maintain_decision("Target validation failed")
        
        except Exception as e:
            self.logger.error(f"Level2 evaluation error: {e}")
            return self._create_maintain_decision(f"Error in target evaluation: {str(e)}")
    
    def _analyze_daily_target_status(self, context: DecisionContext) -> Dict[str, Any]:
        """
        日次ターゲット状況を分析
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            Dict: ターゲット分析結果
        """
        analysis = {}
        
        try:
            # 基本ターゲット情報
            current_return = context.get_portfolio_metric('daily_return') or 0.0
            daily_target = context.get_portfolio_metric('daily_target') or self.daily_target_threshold
            
            # 達成状況の計算
            achievement_ratio = current_return / daily_target if daily_target != 0 else 0
            gap_to_target = daily_target - current_return
            
            # 戦略別ターゲット寄与度
            strategy_contributions = {}
            for strategy_name, strategy_data in context.strategies_data.items():
                contribution = strategy_data.get('daily_contribution', 0.0)
                target_potential = strategy_data.get('target_potential', 0.0)
                strategy_contributions[strategy_name] = {
                    'current_contribution': contribution,
                    'target_potential': target_potential,
                    'potential_gap': target_potential - contribution
                }
            
            analysis = {
                'current_return': current_return,
                'daily_target': daily_target,
                'achievement_ratio': achievement_ratio,
                'gap_to_target': gap_to_target,
                'is_behind_target': gap_to_target > 0,
                'strategy_contributions': strategy_contributions,
                'urgency_level': self._calculate_urgency_level(achievement_ratio, gap_to_target)
            }
            
        except Exception as e:
            self.logger.warning(f"Target analysis error: {e}")
            analysis = {
                'current_return': 0.0,
                'daily_target': self.daily_target_threshold,
                'achievement_ratio': 0.0,
                'gap_to_target': self.daily_target_threshold,
                'is_behind_target': True,
                'strategy_contributions': {},
                'urgency_level': 'medium'
            }
        
        return analysis
    
    def _calculate_time_pressure(self, context: DecisionContext) -> Dict[str, float]:
        """
        時間圧力を計算
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            Dict: 時間圧力情報
        """
        try:
            current_time = context.timestamp
            market_open = context.get_market_indicator('market_open_time')
            market_close = context.get_market_indicator('market_close_time')
            
            if market_open and market_close:
                # 市場セッション内の経過時間比率
                session_duration = (market_close - market_open).total_seconds() / 3600  # 時間
                elapsed_time = (current_time - market_open).total_seconds() / 3600
                time_ratio = min(elapsed_time / session_duration, 1.0)
            else:
                # デフォルト: 1日の経過時間比率
                time_ratio = current_time.hour / 24.0
            
            # 時間圧力係数の計算
            pressure_factor = min(time_ratio * self.time_pressure_factor, 1.0)
            
            return {
                'time_ratio': time_ratio,
                'pressure_factor': pressure_factor,
                'remaining_time_ratio': 1.0 - time_ratio,
                'is_late_session': time_ratio > 0.7
            }
            
        except Exception as e:
            self.logger.warning(f"Time pressure calculation error: {e}")
            return {
                'time_ratio': 0.5,
                'pressure_factor': 0.25,
                'remaining_time_ratio': 0.5,
                'is_late_session': False
            }
    
    def _identify_target_optimal_strategy(self, context: DecisionContext, 
                                        target_analysis: Dict[str, Any],
                                        time_pressure: Dict[str, float]) -> Optional[str]:
        """
        ターゲット達成に最適な戦略を特定
        
        Args:
            context: 決定コンテキスト
            target_analysis: ターゲット分析結果
            time_pressure: 時間圧力情報
            
        Returns:
            Optional[str]: 最適戦略名
        """
        if not target_analysis['is_behind_target']:
            return None  # ターゲット達成済みの場合は切替不要
        
        strategy_scores = {}
        
        for strategy_name, strategy_data in context.strategies_data.items():
            try:
                # ターゲット寄与ポテンシャル
                contribution_data = target_analysis['strategy_contributions'].get(strategy_name, {})
                potential_gap = contribution_data.get('potential_gap', 0.0)
                target_potential = contribution_data.get('target_potential', 0.0)
                
                # 基本スコア
                base_score = strategy_data.get('score', 0.0)
                
                # ターゲット特化スコア
                target_score = (
                    potential_gap * 0.4 +          # ポテンシャルギャップ
                    target_potential * 0.3 +       # ターゲットポテンシャル
                    base_score * 0.3               # 基本スコア
                )
                
                # 時間圧力による調整
                if time_pressure['is_late_session']:
                    # セッション後半では安定性重視
                    stability_bonus = strategy_data.get('stability_score', 0.0) * 0.2
                    target_score += stability_bonus
                else:
                    # セッション前半では積極性重視
                    aggressiveness_bonus = strategy_data.get('aggressiveness_score', 0.0) * 0.15
                    target_score += aggressiveness_bonus
                
                strategy_scores[strategy_name] = target_score
                
            except Exception as e:
                self.logger.warning(f"Target score calculation error for {strategy_name}: {e}")
                strategy_scores[strategy_name] = 0.0
        
        if not strategy_scores:
            return None
        
        # 最高スコアの戦略を選択
        optimal_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        optimal_score = strategy_scores[optimal_strategy]
        
        # 閾値チェック
        min_score_threshold = target_analysis['gap_to_target'] * 10  # ギャップに比例した閾値
        if optimal_score < min_score_threshold:
            return None
        
        return optimal_strategy
    
    def _calculate_target_confidence(self, context: DecisionContext, 
                                   optimal_candidate: Optional[str],
                                   target_analysis: Dict[str, Any],
                                   time_pressure: Dict[str, float]) -> float:
        """
        ターゲット切替信頼度を計算
        
        Args:
            context: 決定コンテキスト
            optimal_candidate: 最適戦略候補
            target_analysis: ターゲット分析結果
            time_pressure: 時間圧力情報
            
        Returns:
            float: 信頼度 (0.0-1.0)
        """
        if not optimal_candidate:
            return 0.0
        
        try:
            # ターゲットギャップによる緊急度
            gap_urgency = min(target_analysis['gap_to_target'] / target_analysis['daily_target'], 1.0)
            
            # 時間圧力による緊急度
            time_urgency = time_pressure['pressure_factor']
            
            # 戦略の実行可能性
            strategy_data = context.strategies_data.get(optimal_candidate, {})
            execution_confidence = strategy_data.get('execution_confidence', 0.5)
            
            # 総合信頼度
            total_confidence = (
                gap_urgency * 0.4 +
                time_urgency * 0.3 +
                execution_confidence * 0.3
            )
            
            # 緊急度に応じた信頼度ブースト
            if target_analysis['urgency_level'] == 'high':
                total_confidence *= 1.2
            elif target_analysis['urgency_level'] == 'critical':
                total_confidence *= 1.5
            
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Target confidence calculation error: {e}")
            return 0.0
    
    def _generate_target_decision(self, context: DecisionContext, 
                                optimal_candidate: Optional[str],
                                switch_confidence: float,
                                target_analysis: Dict[str, Any],
                                time_pressure: Dict[str, float]) -> HierarchicalDecisionResult:
        """
        ターゲット決定を生成
        
        Args:
            context: 決定コンテキスト
            optimal_candidate: 最適戦略候補
            switch_confidence: 切替信頼度
            target_analysis: ターゲット分析結果
            time_pressure: 時間圧力情報
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        # Level1を上書きするかどうかの判定
        should_override = (switch_confidence >= self.min_confidence_override or 
                          target_analysis['urgency_level'] in ['high', 'critical'])
        
        if optimal_candidate and switch_confidence > 0.5:
            achievement_ratio = target_analysis['achievement_ratio']
            gap_to_target = target_analysis['gap_to_target']
            
            reasoning = (f"Daily target focused switch to {optimal_candidate}. "
                        f"Current achievement: {achievement_ratio:.1%}, "
                        f"Gap: {gap_to_target:.3f}, "
                        f"Time pressure: {time_pressure['pressure_factor']:.3f}")
            
            return HierarchicalDecisionResult(
                decision_level=2,
                decision_type='switch',
                target_strategy=optimal_candidate,
                confidence=switch_confidence,
                reasoning=reasoning,
                override_conditions=should_override,
                metadata={
                    'target_analysis': target_analysis,
                    'time_pressure': time_pressure,
                    'urgency_level': target_analysis['urgency_level']
                }
            )
        else:
            if not target_analysis['is_behind_target']:
                reason = "Daily target already achieved"
            else:
                reason = f"No suitable target strategy found (confidence: {switch_confidence:.3f})"
            
            return self._create_maintain_decision(reason)
    
    def _create_maintain_decision(self, reason: str) -> HierarchicalDecisionResult:
        """現状維持決定を生成"""
        return HierarchicalDecisionResult(
            decision_level=2,
            decision_type='maintain',
            confidence=0.5,
            reasoning=f"Level2 maintain: {reason}"
        )
    
    def _calculate_urgency_level(self, achievement_ratio: float, gap_to_target: float) -> str:
        """緊急度レベルを計算"""
        if achievement_ratio >= 1.0:
            return 'low'  # 目標達成済み
        elif achievement_ratio >= 0.8:
            return 'medium'  # 概ね達成
        elif achievement_ratio >= 0.6:
            return 'high'  # 未達成
        else:
            return 'critical'  # 大幅未達成
