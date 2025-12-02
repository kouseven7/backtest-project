"""
Module: Strategy Selection Rule Engine
File: strategy_selection_rule_engine.py
Description: 
  3-1-3「選択ルールの抽象化（差し替え可能に）」
  戦略選択ルールの抽象化と差し替え可能なルールエンジン
  基本ルール、カスタムルール、JSON設定によるルール定義に対応

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_selector
  - config.strategy_scoring_model
  - indicators.unified_trend_detector
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection, SelectionMethod, TrendType
    )
    from config.strategy_scoring_model import StrategyScore
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    logging.getLogger(__name__).warning(f"Import error: {e}")

# ロガーの設定
logger = logging.getLogger(__name__)

class RuleExecutionStatus(Enum):
    """ルール実行ステータス"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"

@dataclass
class RuleContext:
    """ルール実行コンテキスト"""
    # 入力データ
    strategy_scores: Dict[str, float]
    trend_analysis: Dict[str, Any]
    selection_criteria: SelectionCriteria
    available_strategies: Set[str]
    
    # メタデータ
    ticker: str
    timestamp: datetime
    data_quality: float
    
    # 追加コンテキスト
    historical_selections: List[Dict[str, Any]] = field(default_factory=list)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'strategy_scores': self.strategy_scores,
            'trend_analysis': self.trend_analysis,
            'selection_criteria': self.selection_criteria.__dict__,
            'available_strategies': list(self.available_strategies),
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'data_quality': self.data_quality,
            'historical_selections': self.historical_selections,
            'market_conditions': self.market_conditions,
            'risk_metrics': self.risk_metrics
        }

@dataclass
class RuleExecutionResult:
    """ルール実行結果"""
    selected_strategies: List[str]
    strategy_weights: Dict[str, float]
    rule_name: str
    execution_status: RuleExecutionStatus
    confidence: float
    reasoning: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'selected_strategies': self.selected_strategies,
            'strategy_weights': self.strategy_weights,
            'rule_name': self.rule_name,
            'execution_status': self.execution_status.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'execution_time_ms': self.execution_time_ms,
            'metadata': self.metadata
        }

class BaseSelectionRule(ABC):
    """
    戦略選択ルールの基底クラス
    
    すべての選択ルールはこのクラスを継承して実装する
    """
    
    def __init__(self, name: str, priority: int = 100, enabled: bool = True):
        self.name = name
        self.priority = priority  # 低い値ほど高優先度
        self.enabled = enabled
        self.execution_count = 0
        self.success_count = 0
        self.last_execution_time = None
        
    @abstractmethod
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """
        ルールを実行して戦略選択を行う
        
        Args:
            context: ルール実行コンテキスト
            
        Returns:
            RuleExecutionResult: ルール実行結果
        """
        pass
    
    @abstractmethod
    def validate_context(self, context: RuleContext) -> bool:
        """
        コンテキストの妥当性をチェック
        
        Args:
            context: ルール実行コンテキスト
            
        Returns:
            bool: コンテキストが妥当な場合True
        """
        pass
    
    def can_execute(self, context: RuleContext) -> bool:
        """
        ルールが実行可能かチェック
        
        Args:
            context: ルール実行コンテキスト
            
        Returns:
            bool: 実行可能な場合True
        """
        if not self.enabled:
            return False
        
        return self.validate_context(context)
    
    def get_priority(self) -> int:
        """優先度を取得"""
        return self.priority
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """実行統計を取得"""
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': success_rate,
            'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None
        }

class TrendBasedSelectionRule(BaseSelectionRule):
    """
    トレンドベース選択ルール
    
    現在のトレンド状況に最適な戦略を選択
    """
    
    def __init__(self, name: str = "TrendBased", priority: int = 10):
        super().__init__(name, priority)
        
        # トレンド別戦略マッピング
        self.trend_strategy_mapping = {
            TrendType.UPTREND: ["momentum", "breakout", "trend_following"],
            TrendType.DOWNTREND: ["short_selling", "bear_market", "defensive"],
            TrendType.SIDEWAYS: ["mean_reversion", "range_trading", "pairs"],
            TrendType.UNKNOWN: ["balanced", "adaptive", "multi_strategy"]
        }
        
    def validate_context(self, context: RuleContext) -> bool:
        """コンテキストの妥当性をチェック"""
        return (
            context.trend_analysis and 
            'trend_type' in context.trend_analysis and
            len(context.strategy_scores) > 0
        )
    
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """トレンドベース戦略選択を実行"""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # トレンドタイプの取得
            trend_type_str = context.trend_analysis.get('trend_type', 'unknown')
            trend_type = TrendType(trend_type_str.lower()) if trend_type_str.lower() in [t.value for t in TrendType] else TrendType.UNKNOWN
            
            # トレンドに適した戦略の取得
            preferred_strategies = self.trend_strategy_mapping.get(trend_type, [])
            
            # 利用可能な戦略とのマッチング
            available_preferred = [s for s in preferred_strategies if s in context.available_strategies]
            
            # スコアに基づく最終選択
            selected_strategies = []
            strategy_weights = {}
            
            if available_preferred:
                # 優先戦略から選択
                preferred_scores = {s: context.strategy_scores.get(s, 0) for s in available_preferred}
                sorted_preferred = sorted(preferred_scores.items(), key=lambda x: x[1], reverse=True)
                
                max_strategies = min(context.selection_criteria.max_strategies, len(sorted_preferred))
                threshold = context.selection_criteria.min_score_threshold
                
                for strategy, score in sorted_preferred[:max_strategies]:
                    if score >= threshold:
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score
                        
            # 最小戦略数の確保
            if len(selected_strategies) < context.selection_criteria.min_strategies:
                all_scores = sorted(context.strategy_scores.items(), key=lambda x: x[1], reverse=True)
                for strategy, score in all_scores:
                    if strategy not in selected_strategies and len(selected_strategies) < context.selection_criteria.min_strategies:
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score
            
            # 重みの正規化
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            # 信頼度計算
            trend_confidence = context.trend_analysis.get('confidence', 0.5)
            strategy_match_ratio = len(available_preferred) / len(preferred_strategies) if preferred_strategies else 0
            confidence = (trend_confidence + strategy_match_ratio) / 2
            
            # 推論理由
            reasoning = f"Selected {len(selected_strategies)} strategies based on {trend_type.value} trend"
            if available_preferred:
                reasoning += f" (preferred: {', '.join(available_preferred[:3])})"
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.success_count += 1
            self.last_execution_time = datetime.now()
            
            return RuleExecutionResult(
                selected_strategies=selected_strategies,
                strategy_weights=strategy_weights,
                rule_name=self.name,
                execution_status=RuleExecutionStatus.SUCCESS,
                confidence=confidence,
                reasoning=reasoning,
                execution_time_ms=execution_time_ms,
                metadata={
                    'trend_type': trend_type.value,
                    'preferred_strategies': preferred_strategies,
                    'available_preferred': available_preferred,
                    'trend_confidence': trend_confidence
                }
            )
            
        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"TrendBasedSelectionRule execution failed: {e}")
            
            return RuleExecutionResult(
                selected_strategies=[],
                strategy_weights={},
                rule_name=self.name,
                execution_status=RuleExecutionStatus.FAILED,
                confidence=0.0,
                reasoning=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e)}
            )

class ScoreBasedSelectionRule(BaseSelectionRule):
    """
    スコアベース選択ルール
    
    戦略スコアに基づく選択（既存のStrategySelector.select_strategies_by_scoreと互換）
    """
    
    def __init__(self, name: str = "ScoreBased", priority: int = 20):
        super().__init__(name, priority)
        
    def validate_context(self, context: RuleContext) -> bool:
        """コンテキストの妥当性をチェック"""
        return len(context.strategy_scores) > 0
    
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """スコアベース戦略選択を実行"""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # スコアによるソート
            sorted_strategies = sorted(
                context.strategy_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 選択基準の適用
            selected_strategies = []
            strategy_weights = {}
            threshold = context.selection_criteria.min_score_threshold
            max_strategies = context.selection_criteria.max_strategies
            
            for strategy, score in sorted_strategies:
                if (score >= threshold and 
                    len(selected_strategies) < max_strategies and
                    strategy in context.available_strategies):
                    selected_strategies.append(strategy)
                    strategy_weights[strategy] = score
            
            # 最小戦略数の確保
            if len(selected_strategies) < context.selection_criteria.min_strategies:
                for strategy, score in sorted_strategies:
                    if (strategy not in selected_strategies and
                        strategy in context.available_strategies and
                        len(selected_strategies) < context.selection_criteria.min_strategies):
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score
            
            # 重みの正規化
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            # 信頼度計算（高スコア戦略の比率）
            high_score_count = sum(1 for _, score in sorted_strategies if score >= threshold)
            confidence = min(1.0, high_score_count / max(1, len(sorted_strategies)))
            
            reasoning = f"Selected top {len(selected_strategies)} strategies by score (threshold: {threshold:.2f})"
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.success_count += 1
            self.last_execution_time = datetime.now()
            
            return RuleExecutionResult(
                selected_strategies=selected_strategies,
                strategy_weights=strategy_weights,
                rule_name=self.name,
                execution_status=RuleExecutionStatus.SUCCESS,
                confidence=confidence,
                reasoning=reasoning,
                execution_time_ms=execution_time_ms,
                metadata={
                    'threshold_used': threshold,
                    'max_strategies': max_strategies,
                    'total_candidates': len(sorted_strategies)
                }
            )
            
        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"ScoreBasedSelectionRule execution failed: {e}")
            
            return RuleExecutionResult(
                selected_strategies=[],
                strategy_weights={},
                rule_name=self.name,
                execution_status=RuleExecutionStatus.FAILED,
                confidence=0.0,
                reasoning=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e)}
            )

class RiskAdjustedSelectionRule(BaseSelectionRule):
    """
    リスク調整選択ルール
    
    リスク指標を考慮した戦略選択
    """
    
    def __init__(self, name: str = "RiskAdjusted", priority: int = 30):
        super().__init__(name, priority)
        
    def validate_context(self, context: RuleContext) -> bool:
        """コンテキストの妥当性をチェック"""
        return (
            len(context.strategy_scores) > 0 and
            len(context.risk_metrics) > 0
        )
    
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """リスク調整戦略選択を実行"""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # リスク調整スコアの計算
            risk_adjusted_scores = {}
            
            for strategy, base_score in context.strategy_scores.items():
                if strategy in context.available_strategies:
                    # リスク指標の取得
                    strategy_risk = context.risk_metrics.get(strategy, {})
                    volatility = strategy_risk.get('volatility', 0.2)
                    max_drawdown = strategy_risk.get('max_drawdown', 0.1)
                    sharpe_ratio = strategy_risk.get('sharpe_ratio', 1.0)
                    
                    # リスク調整計算
                    risk_penalty = (volatility * 0.5 + max_drawdown * 0.3) / 2
                    risk_bonus = min(sharpe_ratio / 2, 0.2)  # 最大20%ボーナス
                    
                    adjusted_score = base_score * (1 - risk_penalty + risk_bonus)
                    risk_adjusted_scores[strategy] = max(0, adjusted_score)
            
            # 調整スコアによるソート
            sorted_strategies = sorted(
                risk_adjusted_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 選択基準の適用
            selected_strategies = []
            strategy_weights = {}
            threshold = context.selection_criteria.min_score_threshold * 0.8  # リスク調整で閾値を下げる
            max_strategies = context.selection_criteria.max_strategies
            
            for strategy, score in sorted_strategies:
                if (score >= threshold and 
                    len(selected_strategies) < max_strategies):
                    selected_strategies.append(strategy)
                    strategy_weights[strategy] = score
            
            # 最小戦略数の確保
            if len(selected_strategies) < context.selection_criteria.min_strategies:
                for strategy, score in sorted_strategies:
                    if (strategy not in selected_strategies and
                        len(selected_strategies) < context.selection_criteria.min_strategies):
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score
            
            # 重みの正規化
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            # 信頼度計算
            risk_quality = 1.0 - (sum(context.risk_metrics.get(s, {}).get('volatility', 0.2) 
                                     for s in selected_strategies) / len(selected_strategies) if selected_strategies else 0.5)
            confidence = min(1.0, max(0.1, risk_quality))
            
            reasoning = f"Selected {len(selected_strategies)} strategies with risk adjustment (threshold: {threshold:.2f})"
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.success_count += 1
            self.last_execution_time = datetime.now()
            
            return RuleExecutionResult(
                selected_strategies=selected_strategies,
                strategy_weights=strategy_weights,
                rule_name=self.name,
                execution_status=RuleExecutionStatus.SUCCESS,
                confidence=confidence,
                reasoning=reasoning,
                execution_time_ms=execution_time_ms,
                metadata={
                    'risk_adjusted_scores': risk_adjusted_scores,
                    'threshold_used': threshold,
                    'average_risk': sum(context.risk_metrics.get(s, {}).get('volatility', 0.2) 
                                       for s in selected_strategies) / len(selected_strategies) if selected_strategies else 0
                }
            )
            
        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"RiskAdjustedSelectionRule execution failed: {e}")
            
            return RuleExecutionResult(
                selected_strategies=[],
                strategy_weights={},
                rule_name=self.name,
                execution_status=RuleExecutionStatus.FAILED,
                confidence=0.0,
                reasoning=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e)}
            )

class HybridSelectionRule(BaseSelectionRule):
    """
    ハイブリッド選択ルール
    
    複数の選択手法を組み合わせた高度な戦略選択
    """
    
    def __init__(self, name: str = "Hybrid", priority: int = 15):
        super().__init__(name, priority)
        
        # サブルールの初期化
        self.trend_rule = TrendBasedSelectionRule("HybridTrend", 100)
        self.score_rule = ScoreBasedSelectionRule("HybridScore", 100)
        self.risk_rule = RiskAdjustedSelectionRule("HybridRisk", 100)
        
    def validate_context(self, context: RuleContext) -> bool:
        """コンテキストの妥当性をチェック"""
        return (
            len(context.strategy_scores) > 0 and
            context.trend_analysis
        )
    
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """ハイブリッド戦略選択を実行"""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # サブルールの実行
            sub_results = []
            
            # トレンドベース選択
            if self.trend_rule.can_execute(context):
                trend_result = self.trend_rule.execute(context)
                if trend_result.execution_status == RuleExecutionStatus.SUCCESS:
                    sub_results.append(('trend', trend_result, 0.4))
            
            # スコアベース選択
            if self.score_rule.can_execute(context):
                score_result = self.score_rule.execute(context)
                if score_result.execution_status == RuleExecutionStatus.SUCCESS:
                    sub_results.append(('score', score_result, 0.4))
            
            # リスク調整選択（リスク指標がある場合のみ）
            if context.risk_metrics and self.risk_rule.can_execute(context):
                risk_result = self.risk_rule.execute(context)
                if risk_result.execution_status == RuleExecutionStatus.SUCCESS:
                    sub_results.append(('risk', risk_result, 0.2))
            
            if not sub_results:
                raise ValueError("No sub-rules produced valid results")
            
            # 結果の統合
            strategy_vote_weights = {}
            total_confidence = 0
            
            for rule_type, result, weight in sub_results:
                total_confidence += result.confidence * weight
                
                for strategy in result.selected_strategies:
                    if strategy not in strategy_vote_weights:
                        strategy_vote_weights[strategy] = 0
                    
                    strategy_weight = result.strategy_weights.get(strategy, 0)
                    strategy_vote_weights[strategy] += strategy_weight * weight
            
            # 最終選択
            sorted_strategies = sorted(
                strategy_vote_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            max_strategies = context.selection_criteria.max_strategies
            min_strategies = context.selection_criteria.min_strategies
            
            selected_strategies = []
            strategy_weights = {}
            
            for strategy, vote_weight in sorted_strategies[:max_strategies]:
                if vote_weight > 0:  # 何らかの支持がある戦略
                    selected_strategies.append(strategy)
                    strategy_weights[strategy] = vote_weight
            
            # 最小戦略数の確保
            if len(selected_strategies) < min_strategies:
                remaining_strategies = [s for s in context.available_strategies 
                                      if s not in selected_strategies]
                remaining_scores = [(s, context.strategy_scores.get(s, 0)) 
                                  for s in remaining_strategies]
                remaining_scores.sort(key=lambda x: x[1], reverse=True)
                
                for strategy, score in remaining_scores:
                    if len(selected_strategies) < min_strategies:
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score * 0.5  # 低い重み
            
            # 重みの正規化
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            reasoning = f"Hybrid selection using {len(sub_results)} rules: " + \
                       ", ".join([rule_type for rule_type, _, _ in sub_results])
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.success_count += 1
            self.last_execution_time = datetime.now()
            
            return RuleExecutionResult(
                selected_strategies=selected_strategies,
                strategy_weights=strategy_weights,
                rule_name=self.name,
                execution_status=RuleExecutionStatus.SUCCESS,
                confidence=total_confidence,
                reasoning=reasoning,
                execution_time_ms=execution_time_ms,
                metadata={
                    'sub_rules_used': len(sub_results),
                    'sub_results': [r.to_dict() for _, r, _ in sub_results],
                    'vote_weights': strategy_vote_weights
                }
            )
            
        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"HybridSelectionRule execution failed: {e}")
            
            return RuleExecutionResult(
                selected_strategies=[],
                strategy_weights={},
                rule_name=self.name,
                execution_status=RuleExecutionStatus.FAILED,
                confidence=0.0,
                reasoning=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e)}
            )

class ConfigurableSelectionRule(BaseSelectionRule):
    """
    設定可能選択ルール
    
    JSON設定によりカスタムルールを定義可能
    """
    
    def __init__(self, name: str, config: Dict[str, Any], priority: int = 50):
        super().__init__(name, priority)
        self.config = config
        self.rule_type = config.get('type', 'custom')
        self.conditions = config.get('conditions', [])
        self.actions = config.get('actions', {})
        
    def validate_context(self, context: RuleContext) -> bool:
        """コンテキストの妥当性をチェック"""
        required_fields = self.config.get('required_fields', ['strategy_scores'])
        
        for field in required_fields:
            if field == 'strategy_scores' and not context.strategy_scores:
                return False
            elif field == 'trend_analysis' and not context.trend_analysis:
                return False
            elif field == 'risk_metrics' and not context.risk_metrics:
                return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: RuleContext) -> bool:
        """条件を評価"""
        try:
            condition_type = condition.get('type')
            
            if condition_type == 'trend_confidence':
                threshold = condition.get('threshold', 0.5)
                actual = context.trend_analysis.get('confidence', 0)
                operator = condition.get('operator', '>=')
                
                if operator == '>=':
                    return actual >= threshold
                elif operator == '>':
                    return actual > threshold
                elif operator == '<=':
                    return actual <= threshold
                elif operator == '<':
                    return actual < threshold
                elif operator == '==':
                    return abs(actual - threshold) < 0.01
                    
            elif condition_type == 'trend_type':
                expected = condition.get('value')
                actual = context.trend_analysis.get('trend_type')
                return actual == expected
                
            elif condition_type == 'score_threshold':
                threshold = condition.get('threshold', 0.5)
                strategy = condition.get('strategy')
                
                if strategy == 'any':
                    return any(score >= threshold for score in context.strategy_scores.values())
                elif strategy == 'all':
                    return all(score >= threshold for score in context.strategy_scores.values())
                elif strategy in context.strategy_scores:
                    return context.strategy_scores[strategy] >= threshold
                    
            elif condition_type == 'data_quality':
                threshold = condition.get('threshold', 0.7)
                return context.data_quality >= threshold
                
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
        
        return True
    
    def execute(self, context: RuleContext) -> RuleExecutionResult:
        """設定可能ルールを実行"""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # 条件の評価
            conditions_met = True
            if self.conditions:
                for condition in self.conditions:
                    if not self._evaluate_condition(condition, context):
                        conditions_met = False
                        break
            
            if not conditions_met:
                return RuleExecutionResult(
                    selected_strategies=[],
                    strategy_weights={},
                    rule_name=self.name,
                    execution_status=RuleExecutionStatus.SKIPPED,
                    confidence=0.0,
                    reasoning="Conditions not met",
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={'conditions_met': False}
                )
            
            # アクションの実行
            selected_strategies = []
            strategy_weights = {}
            
            action_type = self.actions.get('type', 'select_top')
            
            if action_type == 'select_top':
                count = self.actions.get('count', context.selection_criteria.max_strategies)
                threshold = self.actions.get('threshold', context.selection_criteria.min_score_threshold)
                
                sorted_strategies = sorted(
                    context.strategy_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for strategy, score in sorted_strategies[:count]:
                    if score >= threshold and strategy in context.available_strategies:
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = score
                        
            elif action_type == 'select_by_trend':
                trend_mappings = self.actions.get('trend_mappings', {})
                current_trend = context.trend_analysis.get('trend_type', 'unknown')
                preferred_strategies = trend_mappings.get(current_trend, [])
                
                for strategy in preferred_strategies:
                    if (strategy in context.available_strategies and 
                        strategy in context.strategy_scores):
                        selected_strategies.append(strategy)
                        strategy_weights[strategy] = context.strategy_scores[strategy]
                        
            elif action_type == 'custom_formula':
                formula = self.actions.get('formula', '')
                # カスタム数式の評価（簡単な例）
                # 実際の実装では、より安全な数式評価が必要
                
            # 重みの正規化
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            # 信頼度計算
            confidence = self.actions.get('base_confidence', 0.7)
            if selected_strategies:
                avg_score = sum(context.strategy_scores[s] for s in selected_strategies) / len(selected_strategies)
                confidence = min(1.0, confidence * avg_score)
            
            reasoning = f"Configurable rule '{self.name}' executed with action '{action_type}'"
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.success_count += 1
            self.last_execution_time = datetime.now()
            
            return RuleExecutionResult(
                selected_strategies=selected_strategies,
                strategy_weights=strategy_weights,
                rule_name=self.name,
                execution_status=RuleExecutionStatus.SUCCESS,
                confidence=confidence,
                reasoning=reasoning,
                execution_time_ms=execution_time_ms,
                metadata={
                    'action_type': action_type,
                    'conditions_checked': len(self.conditions),
                    'config_used': self.config
                }
            )
            
        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"ConfigurableSelectionRule execution failed: {e}")
            
            return RuleExecutionResult(
                selected_strategies=[],
                strategy_weights={},
                rule_name=self.name,
                execution_status=RuleExecutionStatus.FAILED,
                confidence=0.0,
                reasoning=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e)}
            )

class StrategySelectionRuleEngine:
    """
    戦略選択ルールエンジン
    
    複数の選択ルールを管理し、優先度に基づいて実行
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config/rule_engine")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # ルールの管理
        self.rules: List[BaseSelectionRule] = []
        self.rule_registry: Dict[str, Type[BaseSelectionRule]] = {}
        
        # 実行統計
        self.execution_history = []
        self.performance_metrics = {}
        
        # 組み込みルールの登録
        self._register_builtin_rules()
        
        # 設定ファイルからのルール読み込み
        self._load_rules_from_config()
        
        logger.info(f"StrategySelectionRuleEngine initialized with {len(self.rules)} rules")
    
    def _register_builtin_rules(self):
        """組み込みルールの登録"""
        self.rule_registry.update({
            'TrendBased': TrendBasedSelectionRule,
            'ScoreBased': ScoreBasedSelectionRule,
            'RiskAdjusted': RiskAdjustedSelectionRule,
            'Hybrid': HybridSelectionRule,
            'Configurable': ConfigurableSelectionRule
        })
        
        # デフォルトルールの追加
        self.add_rule(TrendBasedSelectionRule())
        self.add_rule(ScoreBasedSelectionRule())
        self.add_rule(HybridSelectionRule())
    
    def _load_rules_from_config(self):
        """設定ファイルからルールを読み込み"""
        config_file = self.config_dir / "rules_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for rule_config in config.get('rules', []):
                    rule_type = rule_config.get('type')
                    if rule_type == 'Configurable':
                        rule = ConfigurableSelectionRule(
                            name=rule_config.get('name', 'CustomRule'),
                            config=rule_config.get('config', {}),
                            priority=rule_config.get('priority', 50)
                        )
                        rule.enabled = rule_config.get('enabled', True)
                        self.add_rule(rule)
                        
            except Exception as e:
                logger.warning(f"Failed to load rules config: {e}")
    
    def add_rule(self, rule: BaseSelectionRule):
        """ルールを追加"""
        # 同名ルールの置き換え
        self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
        
        # 優先度によるソート
        self.rules.sort(key=lambda r: r.get_priority())
        
        logger.info(f"Added rule: {rule.name} (priority: {rule.get_priority()})")
    
    def remove_rule(self, rule_name: str):
        """ルールを削除"""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        
        if len(self.rules) < initial_count:
            logger.info(f"Removed rule: {rule_name}")
        else:
            logger.warning(f"Rule not found: {rule_name}")
    
    def get_rule(self, rule_name: str) -> Optional[BaseSelectionRule]:
        """ルールを取得"""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """ルール一覧を取得"""
        return [
            {
                'name': rule.name,
                'priority': rule.priority,
                'enabled': rule.enabled,
                'execution_stats': rule.get_execution_stats()
            }
            for rule in self.rules
        ]
    
    def execute_rules(self, context: RuleContext, rule_name: Optional[str] = None) -> List[RuleExecutionResult]:
        """ルールを実行"""
        results = []
        
        if rule_name:
            # 特定ルールの実行
            rule = self.get_rule(rule_name)
            if rule and rule.can_execute(context):
                result = rule.execute(context)
                results.append(result)
                self._record_execution(rule, result)
        else:
            # 全ルールの実行（優先度順）
            for rule in self.rules:
                if rule.can_execute(context):
                    result = rule.execute(context)
                    results.append(result)
                    self._record_execution(rule, result)
                    
                    # 成功した場合、高優先度ルールで打ち切り
                    if (result.execution_status == RuleExecutionStatus.SUCCESS and 
                        rule.priority <= 20):  # 高優先度ルールの場合
                        break
        
        return results
    
    def select_best_result(self, results: List[RuleExecutionResult]) -> Optional[RuleExecutionResult]:
        """最適な結果を選択"""
        if not results:
            return None
        
        # 成功した結果のみを対象
        successful_results = [r for r in results if r.execution_status == RuleExecutionStatus.SUCCESS]
        
        if not successful_results:
            return None
        
        # 信頼度とルール優先度による選択
        best_result = None
        best_score = -1
        
        for result in successful_results:
            rule = self.get_rule(result.rule_name)
            if rule:
                # スコア計算：信頼度 × (100 - 優先度) / 100
                priority_factor = (100 - rule.priority) / 100
                score = result.confidence * priority_factor
                
                if score > best_score:
                    best_score = score
                    best_result = result
        
        return best_result
    
    def _record_execution(self, rule: BaseSelectionRule, result: RuleExecutionResult):
        """実行結果を記録"""
        execution_record = {
            'rule_name': rule.name,
            'timestamp': datetime.now(),
            'execution_status': result.execution_status.value,
            'confidence': result.confidence,
            'execution_time_ms': result.execution_time_ms,
            'selected_count': len(result.selected_strategies)
        }
        
        self.execution_history.append(execution_record)
        
        # 履歴サイズの制限
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約を取得"""
        if not self.execution_history:
            return {}
        
        rule_stats = {}
        for record in self.execution_history:
            rule_name = record['rule_name']
            if rule_name not in rule_stats:
                rule_stats[rule_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'total_time_ms': 0,
                    'total_confidence': 0
                }
            
            stats = rule_stats[rule_name]
            stats['total_executions'] += 1
            stats['total_time_ms'] += record['execution_time_ms']
            stats['total_confidence'] += record['confidence']
            
            if record['execution_status'] == 'success':
                stats['successful_executions'] += 1
        
        # 統計計算
        for rule_name, stats in rule_stats.items():
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['average_time_ms'] = stats['total_time_ms'] / stats['total_executions']
            stats['average_confidence'] = stats['total_confidence'] / stats['total_executions']
        
        return {
            'rule_statistics': rule_stats,
            'total_executions': len(self.execution_history),
            'overall_success_rate': sum(1 for r in self.execution_history if r['execution_status'] == 'success') / len(self.execution_history)
        }
    
    def save_config(self):
        """設定をファイルに保存"""
        config_file = self.config_dir / "rules_config.json"
        
        # カスタムルールの設定のみ保存
        custom_rules = []
        for rule in self.rules:
            if isinstance(rule, ConfigurableSelectionRule):
                custom_rules.append({
                    'type': 'Configurable',
                    'name': rule.name,
                    'priority': rule.priority,
                    'enabled': rule.enabled,
                    'config': rule.config
                })
        
        config = {
            'rules': custom_rules,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Rule configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save rule configuration: {e}")

if __name__ == "__main__":
    # テスト用のサンプル実行
    from datetime import datetime
    
    # ルールエンジンの初期化
    engine = StrategySelectionRuleEngine()
    
    # テストコンテキストの作成
    context = RuleContext(
        strategy_scores={
            'momentum': 0.8,
            'mean_reversion': 0.6,
            'breakout': 0.9,
            'pairs': 0.5
        },
        trend_analysis={
            'trend_type': 'uptrend',
            'confidence': 0.85,
            'strength': 0.7
        },
        selection_criteria=SelectionCriteria(),
        available_strategies={'momentum', 'mean_reversion', 'breakout', 'pairs'},
        ticker='AAPL',
        timestamp=datetime.now(),
        data_quality=0.9,
        risk_metrics={
            'momentum': {'volatility': 0.15, 'sharpe_ratio': 1.2},
            'breakout': {'volatility': 0.25, 'sharpe_ratio': 1.0}
        }
    )
    
    # ルール実行
    results = engine.execute_rules(context)
    
    print("Rule Execution Results:")
    for result in results:
        print(f"  {result.rule_name}: {result.execution_status.value}")
        print(f"    Selected: {result.selected_strategies}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Reasoning: {result.reasoning}")
        print()
    
    # 最適結果の選択
    best_result = engine.select_best_result(results)
    if best_result:
        print(f"Best Result: {best_result.rule_name}")
        print(f"  Strategies: {best_result.selected_strategies}")
        print(f"  Weights: {best_result.strategy_weights}")
    
    # パフォーマンス要約
    print("\nPerformance Summary:")
    summary = engine.get_performance_summary()
    for rule_name, stats in summary.get('rule_statistics', {}).items():
        print(f"  {rule_name}: Success Rate {stats['success_rate']:.1%}, Avg Time {stats['average_time_ms']:.1f}ms")
