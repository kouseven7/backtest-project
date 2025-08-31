"""
DSSMS Phase 2 Task 2.1: 戦略DSSMS コーディネーター
階層的切替メカニズムによる統合システムの意思決定調整

主要機能:
1. 階層的切替メカニズム（銘柄 → 戦略優先順位）
2. DSSMSと既存戦略の調整・統合判定
3. 信頼度ベースの意思決定
4. 市場状況に応じた動的調整
5. パフォーマンス追跡と学習機能

設計方針:
- 階層的優先順位による段階的判定
- 信頼度閾値による品質管理
- 市場状況の動的考慮
- エラー処理と安全な フォールバック
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import traceback

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 統合ファイル
try:
    from src.dssms.dssms_strategy_integration_manager import IntegrationResult, IntegrationConfig
except ImportError as e:
    print(f"Integration manager import warning: {e}")
    # 最小限の定義
    @dataclass
    class IntegrationResult:
        selected_system: str
        selected_strategy: Optional[str]
        confidence_score: float
        dssms_score: Optional[float]
        strategy_scores: Dict[str, float]
        position_signal: str
        reason: str
        timestamp: datetime
    
    @dataclass
    class IntegrationConfig:
        use_dssms_priority: bool = True
        dssms_weight: float = 0.7
        strategy_weight: float = 0.3
        confidence_threshold: float = 0.6

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DecisionLevel(Enum):
    """決定レベル"""
    SYMBOL_LEVEL = "symbol_level"  # 銘柄レベル（DSSMS主導）
    STRATEGY_LEVEL = "strategy_level"  # 戦略レベル（個別戦略主導）
    HYBRID_LEVEL = "hybrid_level"  # ハイブリッドレベル（統合判定）
    FALLBACK_LEVEL = "fallback_level"  # フォールバックレベル

class MarketCondition(Enum):
    """市場状況"""
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    STABLE = "stable"
    UNCERTAIN = "uncertain"

@dataclass
class CoordinationDecision:
    """調整決定"""
    decision_level: DecisionLevel
    selected_system: str
    selected_strategy: Optional[str]
    confidence_score: float
    market_condition: MarketCondition
    reasoning: str
    contributing_factors: Dict[str, float]
    timestamp: datetime

@dataclass
class PerformanceTracker:
    """パフォーマンス追跡"""
    decision_history: List[CoordinationDecision] = field(default_factory=list)
    system_performance: Dict[str, List[float]] = field(default_factory=dict)
    confidence_accuracy: Dict[float, List[bool]] = field(default_factory=dict)
    market_condition_performance: Dict[str, List[float]] = field(default_factory=dict)

class StrategyDSSMSCoordinator:
    """
    戦略DSSMSコーディネーター
    
    階層的切替メカニズムを使用してDSSMSと既存戦略の
    最適な統合判定を行います。
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """初期化"""
        self.config = config or IntegrationConfig()
        self.performance_tracker = PerformanceTracker()
        
        # 動的調整パラメータ
        self.dynamic_weights = {
            'dssms_weight': self.config.dssms_weight,
            'strategy_weight': self.config.strategy_weight
        }
        
        # 市場状況判定履歴
        self.market_condition_history = []
        
        # 学習パラメータ
        self.learning_enabled = True
        self.learning_rate = 0.01
        
        logger.info("Strategy DSSMS Coordinator initialized")
    
    def coordinate_decision(self,
                          dssms_score: Optional[float],
                          dssms_signal: Optional[str],
                          strategy_scores: Dict[str, float],
                          strategy_signals: Dict[str, str],
                          symbol: str,
                          date: datetime) -> IntegrationResult:
        """調整判定実行"""
        try:
            logger.debug(f"Coordinating decision for {symbol} on {date}")
            
            # 市場状況判定
            market_condition = self._assess_market_condition(
                dssms_score, strategy_scores
            )
            
            # 階層的判定プロセス
            decision = self._hierarchical_decision_process(
                dssms_score=dssms_score,
                dssms_signal=dssms_signal,
                strategy_scores=strategy_scores,
                strategy_signals=strategy_signals,
                market_condition=market_condition,
                symbol=symbol
            )
            
            # 結果作成
            integration_result = IntegrationResult(
                selected_system=decision.selected_system,
                selected_strategy=decision.selected_strategy,
                confidence_score=decision.confidence_score,
                dssms_score=dssms_score,
                strategy_scores=strategy_scores,
                position_signal=self._determine_position_signal(decision, dssms_signal, strategy_signals),
                reason=decision.reasoning,
                timestamp=datetime.now()
            )
            
            # パフォーマンス追跡記録
            self.performance_tracker.decision_history.append(decision)
            
            # 学習機能
            if self.learning_enabled:
                self._update_learning_parameters(decision, market_condition)
            
            logger.debug(f"Decision: {decision.selected_system} (confidence: {decision.confidence_score:.3f})")
            
            return integration_result
            
        except Exception as e:
            logger.error(f"Coordination decision failed: {e}")
            logger.error(traceback.format_exc())
            
            # エラー時のフォールバック
            return IntegrationResult(
                selected_system="fallback",
                selected_strategy=None,
                confidence_score=0.0,
                dssms_score=dssms_score,
                strategy_scores=strategy_scores,
                position_signal="hold",
                reason=f"Coordination failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _hierarchical_decision_process(self,
                                     dssms_score: Optional[float],
                                     dssms_signal: Optional[str],
                                     strategy_scores: Dict[str, float],
                                     strategy_signals: Dict[str, str],
                                     market_condition: MarketCondition,
                                     symbol: str) -> CoordinationDecision:
        """階層的判定プロセス"""
        
        # レベル1: 銘柄レベル（DSSMS優先）
        if self.config.use_dssms_priority and dssms_score is not None:
            decision = self._evaluate_symbol_level(dssms_score, dssms_signal, market_condition)
            if decision.confidence_score >= self.config.confidence_threshold:
                return decision
        
        # レベル2: 戦略レベル（個別戦略評価）
        if strategy_scores:
            decision = self._evaluate_strategy_level(strategy_scores, strategy_signals, market_condition)
            if decision.confidence_score >= self.config.confidence_threshold:
                return decision
        
        # レベル3: ハイブリッドレベル（統合判定）
        if dssms_score is not None and strategy_scores:
            decision = self._evaluate_hybrid_level(
                dssms_score, dssms_signal, strategy_scores, strategy_signals, market_condition
            )
            if decision.confidence_score >= 0.4:  # より低い閾値
                return decision
        
        # レベル4: フォールバックレベル
        return self._evaluate_fallback_level(market_condition)
    
    def _evaluate_symbol_level(self,
                             dssms_score: float,
                             dssms_signal: Optional[str],
                             market_condition: MarketCondition) -> CoordinationDecision:
        """銘柄レベル評価（DSSMS主導）"""
        
        # DSSMSスコア正規化
        normalized_score = min(1.0, max(0.0, dssms_score))
        
        # 市場状況による調整
        market_adjustment = self._get_market_adjustment(market_condition, "dssms")
        adjusted_score = normalized_score * market_adjustment
        
        # 信頼度計算
        confidence = adjusted_score * self.dynamic_weights['dssms_weight']
        
        # システム選択
        selected_system = "dssms_only"
        
        # 理由付け
        reasoning = f"DSSMS銘柄レベル決定: スコア={dssms_score:.3f}, 市場調整={market_adjustment:.3f}"
        
        return CoordinationDecision(
            decision_level=DecisionLevel.SYMBOL_LEVEL,
            selected_system=selected_system,
            selected_strategy=None,
            confidence_score=confidence,
            market_condition=market_condition,
            reasoning=reasoning,
            contributing_factors={'dssms_score': normalized_score, 'market_adjustment': market_adjustment},
            timestamp=datetime.now()
        )
    
    def _evaluate_strategy_level(self,
                               strategy_scores: Dict[str, float],
                               strategy_signals: Dict[str, str],
                               market_condition: MarketCondition) -> CoordinationDecision:
        """戦略レベル評価（個別戦略主導）"""
        
        if not strategy_scores:
            return self._create_low_confidence_decision("No strategy scores available")
        
        # 最高スコア戦略選択
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        best_strategy_name, best_score = best_strategy
        
        # 市場状況による調整
        market_adjustment = self._get_market_adjustment(market_condition, "strategy")
        adjusted_score = best_score * market_adjustment
        
        # 信頼度計算
        confidence = adjusted_score * self.dynamic_weights['strategy_weight']
        
        # システム選択
        selected_system = "strategy_only"
        
        # 理由付け
        reasoning = f"戦略レベル決定: {best_strategy_name}={best_score:.3f}, 市場調整={market_adjustment:.3f}"
        
        return CoordinationDecision(
            decision_level=DecisionLevel.STRATEGY_LEVEL,
            selected_system=selected_system,
            selected_strategy=best_strategy_name,
            confidence_score=confidence,
            market_condition=market_condition,
            reasoning=reasoning,
            contributing_factors={'best_strategy_score': best_score, 'market_adjustment': market_adjustment},
            timestamp=datetime.now()
        )
    
    def _evaluate_hybrid_level(self,
                             dssms_score: float,
                             dssms_signal: Optional[str],
                             strategy_scores: Dict[str, float],
                             strategy_signals: Dict[str, str],
                             market_condition: MarketCondition) -> CoordinationDecision:
        """ハイブリッドレベル評価（統合判定）"""
        
        # DSSMSスコア正規化
        normalized_dssms = min(1.0, max(0.0, dssms_score))
        
        # 戦略スコア統合
        if strategy_scores:
            avg_strategy_score = np.mean(list(strategy_scores.values()))
            max_strategy_score = max(strategy_scores.values())
            best_strategy_name = max(strategy_scores.items(), key=lambda x: x[1])[0]
        else:
            avg_strategy_score = 0.0
            max_strategy_score = 0.0
            best_strategy_name = None
        
        # ハイブリッドスコア計算
        hybrid_score = (
            normalized_dssms * self.dynamic_weights['dssms_weight'] +
            max_strategy_score * self.dynamic_weights['strategy_weight']
        )
        
        # 市場状況による調整
        market_adjustment = self._get_market_adjustment(market_condition, "hybrid")
        adjusted_score = hybrid_score * market_adjustment
        
        # システム選択（スコアが高い方を選択）
        if normalized_dssms > max_strategy_score:
            selected_system = "dssms_hybrid"
            selected_strategy = None
        else:
            selected_system = "strategy_hybrid"
            selected_strategy = best_strategy_name
        
        # 理由付け
        reasoning = f"ハイブリッド決定: DSSMS={normalized_dssms:.3f}, 戦略={max_strategy_score:.3f}, 選択={selected_system}"
        
        return CoordinationDecision(
            decision_level=DecisionLevel.HYBRID_LEVEL,
            selected_system=selected_system,
            selected_strategy=selected_strategy,
            confidence_score=adjusted_score,
            market_condition=market_condition,
            reasoning=reasoning,
            contributing_factors={
                'dssms_score': normalized_dssms,
                'strategy_score': max_strategy_score,
                'hybrid_score': hybrid_score,
                'market_adjustment': market_adjustment
            },
            timestamp=datetime.now()
        )
    
    def _evaluate_fallback_level(self, market_condition: MarketCondition) -> CoordinationDecision:
        """フォールバックレベル評価"""
        
        # 保守的な判定
        selected_system = "fallback"
        confidence = 0.1  # 非常に低い信頼度
        
        reasoning = f"フォールバック決定: 市場状況={market_condition.value}, 他の判定方法で十分な信頼度を得られず"
        
        return CoordinationDecision(
            decision_level=DecisionLevel.FALLBACK_LEVEL,
            selected_system=selected_system,
            selected_strategy=None,
            confidence_score=confidence,
            market_condition=market_condition,
            reasoning=reasoning,
            contributing_factors={'fallback_reason': 'insufficient_confidence'},
            timestamp=datetime.now()
        )
    
    def _assess_market_condition(self,
                               dssms_score: Optional[float],
                               strategy_scores: Dict[str, float]) -> MarketCondition:
        """市場状況評価"""
        try:
            # スコア分散による判定
            all_scores = []
            if dssms_score is not None:
                all_scores.append(dssms_score)
            all_scores.extend(strategy_scores.values())
            
            if len(all_scores) < 2:
                return MarketCondition.UNCERTAIN
            
            score_std = np.std(all_scores)
            score_mean = np.mean(all_scores)
            
            # 閾値による判定
            if score_std < 0.1:
                if score_mean > 0.7:
                    condition = MarketCondition.STABLE
                else:
                    condition = MarketCondition.SIDEWAYS
            elif score_std > 0.3:
                condition = MarketCondition.VOLATILE
            else:
                if score_mean > 0.6:
                    condition = MarketCondition.TRENDING
                else:
                    condition = MarketCondition.UNCERTAIN
            
            # 履歴に記録
            self.market_condition_history.append({
                'timestamp': datetime.now(),
                'condition': condition,
                'score_std': score_std,
                'score_mean': score_mean
            })
            
            return condition
            
        except Exception as e:
            logger.warning(f"Failed to assess market condition: {e}")
            return MarketCondition.UNCERTAIN
    
    def _get_market_adjustment(self, condition: MarketCondition, system_type: str) -> float:
        """市場状況による調整ファクター"""
        
        # システムタイプ別の市場適応性
        adjustment_matrix = {
            MarketCondition.TRENDING: {
                'dssms': 1.1,     # DSSMSはトレンドに強い
                'strategy': 1.0,
                'hybrid': 1.05
            },
            MarketCondition.SIDEWAYS: {
                'dssms': 0.9,     # DSSMSは横ばいでやや弱い
                'strategy': 1.1,  # 戦略は横ばいに対応
                'hybrid': 1.0
            },
            MarketCondition.VOLATILE: {
                'dssms': 1.2,     # DSSMSは変動性に強い
                'strategy': 0.8,  # 戦略は変動性に弱い
                'hybrid': 1.1
            },
            MarketCondition.STABLE: {
                'dssms': 1.0,
                'strategy': 1.0,
                'hybrid': 1.0
            },
            MarketCondition.UNCERTAIN: {
                'dssms': 0.8,     # 不確実性では保守的
                'strategy': 0.8,
                'hybrid': 0.7
            }
        }
        
        return adjustment_matrix.get(condition, {}).get(system_type, 1.0)
    
    def _determine_position_signal(self,
                                 decision: CoordinationDecision,
                                 dssms_signal: Optional[str],
                                 strategy_signals: Dict[str, str]) -> str:
        """ポジションシグナル決定"""
        
        # システム別シグナル選択
        if "dssms" in decision.selected_system.lower():
            return dssms_signal or "hold"
        elif "strategy" in decision.selected_system.lower() and decision.selected_strategy:
            return strategy_signals.get(decision.selected_strategy, "hold")
        elif "hybrid" in decision.selected_system.lower():
            # ハイブリッドの場合は信頼度で選択
            if decision.confidence_score > 0.7:
                if dssms_signal and dssms_signal != "hold":
                    return dssms_signal
                elif strategy_signals:
                    # 最高スコア戦略のシグナル
                    return next(iter(strategy_signals.values()), "hold")
            return "hold"
        else:
            return "hold"
    
    def _update_learning_parameters(self,
                                  decision: CoordinationDecision,
                                  market_condition: MarketCondition):
        """学習パラメータ更新"""
        try:
            # パフォーマンス追跡による動的重み調整
            if len(self.performance_tracker.decision_history) > 10:
                recent_decisions = self.performance_tracker.decision_history[-10:]
                
                # システム別の成功率計算（仮想的）
                dssms_decisions = [d for d in recent_decisions if "dssms" in d.selected_system]
                strategy_decisions = [d for d in recent_decisions if "strategy" in d.selected_system]
                
                # 重み調整（簡易版）
                if len(dssms_decisions) > len(strategy_decisions):
                    # DSSMSの方が多く選択されている場合、DSSMS重みを微調整
                    self.dynamic_weights['dssms_weight'] = min(0.8, self.dynamic_weights['dssms_weight'] + self.learning_rate)
                    self.dynamic_weights['strategy_weight'] = 1.0 - self.dynamic_weights['dssms_weight']
                else:
                    # 戦略の方が多く選択されている場合、戦略重みを微調整
                    self.dynamic_weights['strategy_weight'] = min(0.8, self.dynamic_weights['strategy_weight'] + self.learning_rate)
                    self.dynamic_weights['dssms_weight'] = 1.0 - self.dynamic_weights['strategy_weight']
                
                logger.debug(f"Updated weights: DSSMS={self.dynamic_weights['dssms_weight']:.3f}, Strategy={self.dynamic_weights['strategy_weight']:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to update learning parameters: {e}")
    
    def _create_low_confidence_decision(self, reason: str) -> CoordinationDecision:
        """低信頼度決定作成"""
        return CoordinationDecision(
            decision_level=DecisionLevel.FALLBACK_LEVEL,
            selected_system="fallback",
            selected_strategy=None,
            confidence_score=0.1,
            market_condition=MarketCondition.UNCERTAIN,
            reasoning=reason,
            contributing_factors={'error': reason},
            timestamp=datetime.now()
        )
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """調整統計取得"""
        try:
            if not self.performance_tracker.decision_history:
                return {}
            
            decisions = self.performance_tracker.decision_history
            
            # システム選択統計
            system_counts = {}
            level_counts = {}
            confidence_scores = []
            
            for decision in decisions:
                system = decision.selected_system
                level = decision.decision_level.value
                
                system_counts[system] = system_counts.get(system, 0) + 1
                level_counts[level] = level_counts.get(level, 0) + 1
                confidence_scores.append(decision.confidence_score)
            
            total_decisions = len(decisions)
            
            return {
                'total_decisions': total_decisions,
                'system_usage': {k: v/total_decisions for k, v in system_counts.items()},
                'decision_level_usage': {k: v/total_decisions for k, v in level_counts.items()},
                'average_confidence': np.mean(confidence_scores),
                'confidence_std': np.std(confidence_scores),
                'high_confidence_ratio': sum(1 for c in confidence_scores if c >= 0.7) / len(confidence_scores),
                'current_weights': self.dynamic_weights,
                'market_condition_distribution': self._get_market_condition_distribution()
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate coordination statistics: {e}")
            return {}
    
    def _get_market_condition_distribution(self) -> Dict[str, float]:
        """市場状況分布取得"""
        if not self.market_condition_history:
            return {}
        
        condition_counts = {}
        for record in self.market_condition_history:
            condition = record['condition'].value
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        total = len(self.market_condition_history)
        return {k: v/total for k, v in condition_counts.items()}

# 使用例とテスト関数
def test_coordinator():
    """コーディネーターのテスト"""
    print("=== Strategy DSSMS Coordinator Test ===")
    
    # コーディネーター初期化
    config = IntegrationConfig(
        use_dssms_priority=True,
        dssms_weight=0.7,
        strategy_weight=0.3,
        confidence_threshold=0.6
    )
    
    coordinator = StrategyDSSMSCoordinator(config)
    
    # テストケース1: DSSMS優位
    result1 = coordinator.coordinate_decision(
        dssms_score=0.85,
        dssms_signal="buy",
        strategy_scores={"VWAP_Breakout": 0.6, "GoldenCross": 0.4},
        strategy_signals={"VWAP_Breakout": "hold", "GoldenCross": "sell"},
        symbol="7203",
        date=datetime.now()
    )
    
    print(f"Test 1 - DSSMS優位:")
    print(f"  Selected: {result1.selected_system}")
    print(f"  Confidence: {result1.confidence_score:.3f}")
    print(f"  Signal: {result1.position_signal}")
    
    # テストケース2: 戦略優位
    result2 = coordinator.coordinate_decision(
        dssms_score=0.3,
        dssms_signal="hold",
        strategy_scores={"VWAP_Breakout": 0.9, "GoldenCross": 0.8},
        strategy_signals={"VWAP_Breakout": "buy", "GoldenCross": "buy"},
        symbol="6758",
        date=datetime.now()
    )
    
    print(f"\nTest 2 - 戦略優位:")
    print(f"  Selected: {result2.selected_system}")
    print(f"  Confidence: {result2.confidence_score:.3f}")
    print(f"  Signal: {result2.position_signal}")
    
    # 統計情報
    stats = coordinator.get_coordination_statistics()
    print(f"\nCoordination Statistics:")
    print(f"  Total decisions: {stats.get('total_decisions', 0)}")
    print(f"  System usage: {stats.get('system_usage', {})}")
    print(f"  Average confidence: {stats.get('average_confidence', 0):.3f}")

if __name__ == "__main__":
    test_coordinator()
