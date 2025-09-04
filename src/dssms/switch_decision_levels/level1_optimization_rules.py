"""
Level 1: Optimization Rules
レベル1：最適化ルール - strategies/, config/, analysis/ モジュールを統合した包括的な決定システム
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from .base_level import BaseSwitchDecisionLevel
from ..decision_context import DecisionContext, HierarchicalDecisionResult


class Level1OptimizationRules(BaseSwitchDecisionLevel):
    """
    レベル1：最適化ルール
    
    strategies/, config/, analysis/ モジュールの情報を統合して
    最適化ベースの戦略切替決定を行う
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Level1固有のデフォルト設定を最初に設定
        self.performance_threshold = 0.05  # パフォーマンス閾値
        self.confidence_threshold = 0.7    # 信頼度閾値
        self.lookback_period = 20          # 振り返り期間
        self.min_data_points = 10          # 最小データポイント数
        
        # 基底クラスの初期化を呼び出し
        super().__init__(level_number=1, config=config)
        
    def _load_level_config(self) -> None:
        """レベル1固有の設定を読み込み"""
        if 'optimization_rules' in self.config:
            opt_config = self.config['optimization_rules']
            self.performance_threshold = opt_config.get('performance_threshold', self.performance_threshold)
            self.confidence_threshold = opt_config.get('confidence_threshold', self.confidence_threshold)
            self.lookback_period = opt_config.get('lookback_period', self.lookback_period)
            self.min_data_points = opt_config.get('min_data_points', self.min_data_points)
        
        self.logger.info(f"Level1 Config: performance_threshold={self.performance_threshold}, "
                        f"confidence_threshold={self.confidence_threshold}")
    
    def evaluate_switch_condition(self, context: DecisionContext) -> HierarchicalDecisionResult:
        """
        最適化ルールに基づいて切替条件を評価
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        try:
            # 戦略パフォーマンス分析
            performance_analysis = self._analyze_strategy_performance(context)
            
            # 最適戦略候補の特定
            optimal_candidate = self._identify_optimal_strategy(context, performance_analysis)
            
            # 切替信頼度の計算
            switch_confidence = self._calculate_switch_confidence(context, optimal_candidate, performance_analysis)
            
            # 決定の生成
            decision = self._generate_decision(context, optimal_candidate, switch_confidence, performance_analysis)
            
            # 決定の妥当性検証
            if self.validate_decision(decision, context):
                self.log_decision(decision, context)
                return decision
            else:
                # 妥当性検証失敗時は現状維持
                return self._create_maintain_decision("Validation failed")
        
        except Exception as e:
            self.logger.error(f"Level1 evaluation error: {e}")
            return self._create_maintain_decision(f"Error in evaluation: {str(e)}")
    
    def _analyze_strategy_performance(self, context: DecisionContext) -> Dict[str, Dict[str, float]]:
        """
        戦略パフォーマンスを分析
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            Dict: 戦略別パフォーマンス分析結果
        """
        performance_analysis = {}
        
        for strategy_name, strategy_data in context.strategies_data.items():
            try:
                # 基本パフォーマンスメトリクス
                score = strategy_data.get('score', 0.0)
                weight = strategy_data.get('weight', 0.0)
                returns = strategy_data.get('returns', [])
                
                # リターン系統の統計分析
                if returns and len(returns) >= self.min_data_points:
                    returns_array = np.array(returns[-self.lookback_period:])
                    
                    performance_metrics = {
                        'current_score': score,
                        'current_weight': weight,
                        'mean_return': np.mean(returns_array),
                        'std_return': np.std(returns_array),
                        'sharpe_ratio': self._calculate_sharpe_ratio(returns_array),
                        'max_drawdown': self._calculate_max_drawdown(returns_array),
                        'win_rate': self._calculate_win_rate(returns_array),
                        'recent_momentum': self._calculate_recent_momentum(returns_array),
                        'consistency_score': self._calculate_consistency_score(returns_array)
                    }
                else:
                    # データ不足時のデフォルト値
                    performance_metrics = {
                        'current_score': score,
                        'current_weight': weight,
                        'mean_return': 0.0,
                        'std_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0,
                        'recent_momentum': 0.0,
                        'consistency_score': 0.0
                    }
                
                performance_analysis[strategy_name] = performance_metrics
                
            except Exception as e:
                self.logger.warning(f"Performance analysis error for {strategy_name}: {e}")
                performance_analysis[strategy_name] = {'current_score': 0.0, 'current_weight': 0.0}
        
        return performance_analysis
    
    def _identify_optimal_strategy(self, context: DecisionContext, 
                                 performance_analysis: Dict[str, Dict[str, float]]) -> Optional[str]:
        """
        最適戦略候補を特定
        
        Args:
            context: 決定コンテキスト
            performance_analysis: パフォーマンス分析結果
            
        Returns:
            Optional[str]: 最適戦略名
        """
        if not performance_analysis:
            return None
        
        # 複合スコアの計算
        strategy_scores = {}
        for strategy_name, metrics in performance_analysis.items():
            try:
                # 重み付き複合スコア
                composite_score = (
                    metrics['current_score'] * 0.3 +
                    metrics['sharpe_ratio'] * 0.25 +
                    metrics['recent_momentum'] * 0.2 +
                    metrics['consistency_score'] * 0.15 +
                    metrics['win_rate'] * 0.1
                )
                
                # リスク調整
                risk_penalty = metrics['max_drawdown'] * 0.5
                adjusted_score = composite_score - risk_penalty
                
                strategy_scores[strategy_name] = adjusted_score
                
            except Exception as e:
                self.logger.warning(f"Score calculation error for {strategy_name}: {e}")
                strategy_scores[strategy_name] = 0.0
        
        if not strategy_scores:
            return None
        
        # 最高スコアの戦略を選択
        optimal_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        optimal_score = strategy_scores[optimal_strategy]
        
        # 閾値チェック
        if optimal_score < self.performance_threshold:
            return None
        
        return optimal_strategy
    
    def _calculate_switch_confidence(self, context: DecisionContext, 
                                   optimal_candidate: Optional[str],
                                   performance_analysis: Dict[str, Dict[str, float]]) -> float:
        """
        切替信頼度を計算
        
        Args:
            context: 決定コンテキスト
            optimal_candidate: 最適戦略候補
            performance_analysis: パフォーマンス分析結果
            
        Returns:
            float: 信頼度 (0.0-1.0)
        """
        if not optimal_candidate or optimal_candidate not in performance_analysis:
            return 0.0
        
        try:
            optimal_metrics = performance_analysis[optimal_candidate]
            
            # 信頼度要素
            score_confidence = min(optimal_metrics['current_score'] / 100.0, 1.0)
            sharpe_confidence = min(max(optimal_metrics['sharpe_ratio'], 0.0) / 2.0, 1.0)
            consistency_confidence = optimal_metrics['consistency_score']
            momentum_confidence = min(max(optimal_metrics['recent_momentum'], 0.0), 1.0)
            
            # 重み付き信頼度
            total_confidence = (
                score_confidence * 0.4 +
                sharpe_confidence * 0.3 +
                consistency_confidence * 0.2 +
                momentum_confidence * 0.1
            )
            
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {e}")
            return 0.0
    
    def _generate_decision(self, context: DecisionContext, optimal_candidate: Optional[str],
                          switch_confidence: float, performance_analysis: Dict[str, Dict[str, float]]) -> HierarchicalDecisionResult:
        """
        決定を生成
        
        Args:
            context: 決定コンテキスト
            optimal_candidate: 最適戦略候補
            switch_confidence: 切替信頼度
            performance_analysis: パフォーマンス分析結果
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        # 切替決定の条件チェック
        if (optimal_candidate and 
            switch_confidence >= self.confidence_threshold and
            optimal_candidate in performance_analysis):
            
            reasoning = (f"Optimization analysis suggests switch to {optimal_candidate} "
                        f"with confidence {switch_confidence:.3f}")
            
            return HierarchicalDecisionResult(
                decision_level=1,
                decision_type='switch',
                target_strategy=optimal_candidate,
                confidence=switch_confidence,
                reasoning=reasoning,
                metadata={
                    'performance_analysis': performance_analysis,
                    'optimal_score': performance_analysis[optimal_candidate]['current_score']
                }
            )
        else:
            # 現状維持決定
            reason = "No optimal strategy found" if not optimal_candidate else f"Low confidence: {switch_confidence:.3f}"
            return self._create_maintain_decision(reason)
    
    def _create_maintain_decision(self, reason: str) -> HierarchicalDecisionResult:
        """現状維持決定を生成"""
        return HierarchicalDecisionResult(
            decision_level=1,
            decision_type='maintain',
            confidence=0.5,
            reasoning=f"Level1 maintain: {reason}"
        )
    
    # ヘルパーメソッド群
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """シャープレシオ計算"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """最大ドローダウン計算"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """勝率計算"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    def _calculate_recent_momentum(self, returns: np.ndarray) -> float:
        """最近のモメンタム計算"""
        if len(returns) < 5:
            return 0.0
        recent_returns = returns[-5:]
        return np.mean(recent_returns)
    
    def _calculate_consistency_score(self, returns: np.ndarray) -> float:
        """一貫性スコア計算"""
        if len(returns) == 0:
            return 0.0
        # 標準偏差の逆数をベースとした一貫性
        std_returns = np.std(returns)
        if std_returns == 0:
            return 1.0
        return min(1.0 / (1.0 + std_returns), 1.0)
