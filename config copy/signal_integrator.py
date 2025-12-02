"""
Module: Signal Integrator
File: signal_integrator.py
Description: 
  3-3-1「シグナル競合時の優先度ルール設計」
  複数戦略からのシグナルを統合し、競合時の優先度制御を行う

Author: imega
Created: 2025-07-16
Modified: 2025-07-16

Dependencies:
  - config.strategy_selector
  - config.portfolio_weight_calculator
  - config.strategy_scoring_model
"""

import os
import sys
import json
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムのインポート
try:
    from config.strategy_selector import StrategySelector
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.strategy_scoring_model import StrategyScoreCalculator
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """シグナルタイプ"""
    ENTRY_LONG = "entry_long"       # ロングエントリー
    ENTRY_SHORT = "entry_short"     # ショートエントリー
    EXIT_LONG = "exit_long"         # ロングエグジット
    EXIT_SHORT = "exit_short"       # ショートエグジット
    HOLD = "hold"                   # ホールド
    NO_SIGNAL = "no_signal"         # シグナルなし

class ConflictType(Enum):
    """競合タイプ"""
    DIRECTION_CONFLICT = "direction_conflict"     # 方向性競合（ロング vs ショート）
    TIMING_CONFLICT = "timing_conflict"           # タイミング競合（同時エントリー）
    RESOURCE_CONFLICT = "resource_conflict"       # リソース競合（資金不足）
    RISK_CONFLICT = "risk_conflict"               # リスク競合（リスク制限超過）

class PriorityMethod(Enum):
    """優先度決定方式"""
    SCORE_BASED = "score_based"           # スコアベース
    RULE_BASED = "rule_based"             # ルールベース
    HYBRID = "hybrid"                     # ハイブリッド
    CONSENSUS = "consensus"               # コンセンサス（多数決）

@dataclass
class StrategySignal:
    """戦略シグナル"""
    strategy_name: str
    signal_type: SignalType
    timestamp: datetime
    confidence: float = 1.0
    position_size: float = 0.0
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 優先度関連
    strategy_score: Optional[float] = None
    rule_priority: Optional[int] = None
    adjusted_priority: Optional[float] = None

@dataclass
class SignalConflict:
    """シグナル競合"""
    conflict_id: str
    conflict_type: ConflictType
    conflicting_signals: List[StrategySignal]
    detected_at: datetime
    resolution_method: Optional[PriorityMethod] = None
    resolved_signal: Optional[StrategySignal] = None
    resolution_reason: str = ""

class ConflictDetector:
    """シグナル競合検出器"""
    
    def __init__(self):
        self.detection_rules = {
            ConflictType.DIRECTION_CONFLICT: self._detect_direction_conflict,
            ConflictType.TIMING_CONFLICT: self._detect_timing_conflict,
            ConflictType.RESOURCE_CONFLICT: self._detect_resource_conflict,
            ConflictType.RISK_CONFLICT: self._detect_risk_conflict
        }
        logger.info("ConflictDetector initialized")
    
    def detect_conflicts(self, signals: List[StrategySignal], 
                        current_portfolio: Dict[str, float],
                        available_capital: float,
                        risk_limits: Dict[str, float]) -> List[SignalConflict]:
        """シグナル競合の検出"""
        conflicts = []
        
        try:
            for conflict_type, detector in self.detection_rules.items():
                detected_conflicts = detector(signals, current_portfolio, available_capital, risk_limits)
                conflicts.extend(detected_conflicts)
            
            logger.info(f"競合検出完了: {len(conflicts)} 個の競合を検出")
            return conflicts
            
        except Exception as e:
            logger.error(f"競合検出エラー: {e}")
            return []
    
    def _detect_direction_conflict(self, signals: List[StrategySignal], 
                                  current_portfolio: Dict[str, float],
                                  available_capital: float,
                                  risk_limits: Dict[str, float]) -> List[SignalConflict]:
        """方向性競合の検出"""
        conflicts = []
        entry_signals = [s for s in signals if s.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]]
        
        if len(entry_signals) <= 1:
            return conflicts
        
        # ロング vs ショート競合の検出
        long_signals = [s for s in entry_signals if s.signal_type == SignalType.ENTRY_LONG]
        short_signals = [s for s in entry_signals if s.signal_type == SignalType.ENTRY_SHORT]
        
        if long_signals and short_signals:
            conflict = SignalConflict(
                conflict_id=f"direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                conflict_type=ConflictType.DIRECTION_CONFLICT,
                conflicting_signals=long_signals + short_signals,
                detected_at=datetime.now()
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_timing_conflict(self, signals: List[StrategySignal], 
                               current_portfolio: Dict[str, float],
                               available_capital: float,
                               risk_limits: Dict[str, float]) -> List[SignalConflict]:
        """タイミング競合の検出"""
        conflicts = []
        
        # 同じタイプのエントリーシグナルが複数ある場合
        entry_signals = [s for s in signals if s.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]]
        
        # シグナルタイプ別にグループ化
        signal_groups = {}
        for signal in entry_signals:
            signal_type = signal.signal_type
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # 各グループで複数シグナルがある場合は競合
        for signal_type, group_signals in signal_groups.items():
            if len(group_signals) > 1:
                conflict = SignalConflict(
                    conflict_id=f"timing_{signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    conflict_type=ConflictType.TIMING_CONFLICT,
                    conflicting_signals=group_signals,
                    detected_at=datetime.now()
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_resource_conflict(self, signals: List[StrategySignal], 
                                 current_portfolio: Dict[str, float],
                                 available_capital: float,
                                 risk_limits: Dict[str, float]) -> List[SignalConflict]:
        """リソース競合の検出"""
        conflicts = []
        
        # エントリーシグナルの総資金需要を計算
        entry_signals = [s for s in signals if s.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]]
        total_required_capital = sum(abs(s.position_size) * available_capital for s in entry_signals)
        
        # 利用可能資金を超える場合は競合
        if total_required_capital > available_capital:
            conflict = SignalConflict(
                conflict_id=f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                conflict_type=ConflictType.RESOURCE_CONFLICT,
                conflicting_signals=entry_signals,
                detected_at=datetime.now()
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_risk_conflict(self, signals: List[StrategySignal], 
                             current_portfolio: Dict[str, float],
                             available_capital: float,
                             risk_limits: Dict[str, float]) -> List[SignalConflict]:
        """リスク競合の検出"""
        conflicts = []
        
        try:
            max_portfolio_risk = risk_limits.get("max_portfolio_risk", 0.02)
            max_single_exposure = risk_limits.get("max_single_exposure", 0.3)
            
            # 単一ポジションサイズチェック
            risky_signals = []
            for signal in signals:
                if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                    if abs(signal.position_size) > max_single_exposure:
                        risky_signals.append(signal)
            
            if risky_signals:
                conflict = SignalConflict(
                    conflict_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    conflict_type=ConflictType.RISK_CONFLICT,
                    conflicting_signals=risky_signals,
                    detected_at=datetime.now()
                )
                conflicts.append(conflict)
        
        except Exception as e:
            logger.warning(f"リスク競合検出エラー: {e}")
        
        return conflicts

class PriorityResolver:
    """優先度解決エンジン"""
    
    def __init__(self, strategy_selector: Optional[StrategySelector] = None, 
                 score_calculator: Optional[StrategyScoreCalculator] = None):
        self.strategy_selector = strategy_selector
        self.score_calculator = score_calculator
        
        # 優先度ルール設定
        self.rule_priorities = {
            "exit_signals": 1,        # エグジットシグナルは最優先
            "risk_management": 2,     # リスク管理シグナル
            "high_confidence": 3,     # 高信頼度シグナル
            "trending_strategy": 4,   # トレンド戦略
            "momentum_strategy": 5,   # モメンタム戦略
            "mean_reversion": 6,      # 平均回帰戦略
            "default": 10             # デフォルト優先度
        }
        
        # 重み設定
        self.priority_weights = {
            "strategy_score": 0.4,
            "signal_confidence": 0.3,
            "rule_priority": 0.2,
            "timing_factor": 0.1
        }
        
        logger.info("PriorityResolver initialized")
    
    def resolve_conflict(self, conflict: SignalConflict) -> Optional[StrategySignal]:
        """シグナル競合の解決"""
        try:
            if conflict.resolution_method == PriorityMethod.HYBRID:
                return self._resolve_hybrid(conflict)
            elif conflict.resolution_method == PriorityMethod.SCORE_BASED:
                return self._resolve_score_based(conflict)
            elif conflict.resolution_method == PriorityMethod.RULE_BASED:
                return self._resolve_rule_based(conflict)
            elif conflict.resolution_method == PriorityMethod.CONSENSUS:
                return self._resolve_consensus(conflict)
            else:
                # デフォルトはハイブリッド方式
                return self._resolve_hybrid(conflict)
                
        except Exception as e:
            logger.error(f"シグナル競合解決エラー: {e}")
            # フォールバック: 最初のシグナルを返す
            return conflict.conflicting_signals[0] if conflict.conflicting_signals else None
    
    def _resolve_hybrid(self, conflict: SignalConflict) -> Optional[StrategySignal]:
        """ハイブリッド方式による競合解決"""
        signals = conflict.conflicting_signals
        
        # 1. エグジットシグナルの優先処理
        exit_signals = [s for s in signals if s.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]]
        if exit_signals:
            return self._select_best_exit_signal(exit_signals)
        
        # 2. 各シグナルの総合優先度計算
        scored_signals = []
        for signal in signals:
            score = self._calculate_hybrid_score(signal)
            signal.adjusted_priority = score
            scored_signals.append((signal, score))
        
        # 3. 最高スコアのシグナルを選択
        if scored_signals:
            best_signal, best_score = max(scored_signals, key=lambda x: x[1])
            
            # 4. 解決理由の記録
            conflict.resolution_reason = f"ハイブリッド方式: スコア={best_score:.3f}"
            
            return best_signal
        
        return None
    
    def _select_best_exit_signal(self, exit_signals: List[StrategySignal]) -> StrategySignal:
        """最適なエグジットシグナルの選択"""
        if len(exit_signals) == 1:
            return exit_signals[0]
        
        # 信頼度の高いエグジットシグナルを選択
        return max(exit_signals, key=lambda s: s.confidence)
    
    def _calculate_hybrid_score(self, signal: StrategySignal) -> float:
        """ハイブリッドスコアの計算"""
        try:
            # 1. 戦略スコア（正規化済み）
            strategy_score = signal.strategy_score or self._get_strategy_score(signal.strategy_name)
            strategy_score_normalized = min(max(strategy_score, 0.0), 1.0)
            
            # 2. シグナル信頼度
            confidence_score = min(max(signal.confidence, 0.0), 1.0)
            
            # 3. ルール優先度（逆数で正規化）
            rule_priority = self._get_rule_priority(signal)
            rule_score = 1.0 / max(rule_priority, 1.0)
            
            # 4. タイミングファクター（新しいシグナルを優先）
            time_diff = (datetime.now() - signal.timestamp).total_seconds()
            timing_score = max(0.0, 1.0 - (time_diff / 3600))  # 1時間で減衰
            
            # 5. 重み付き総合スコア
            hybrid_score = (
                self.priority_weights["strategy_score"] * strategy_score_normalized +
                self.priority_weights["signal_confidence"] * confidence_score +
                self.priority_weights["rule_priority"] * rule_score +
                self.priority_weights["timing_factor"] * timing_score
            )
            
            return hybrid_score
            
        except Exception as e:
            logger.warning(f"ハイブリッドスコア計算エラー: {e}")
            return 0.5  # デフォルトスコア
    
    def _get_strategy_score(self, strategy_name: str) -> float:
        """戦略スコアの取得"""
        try:
            if self.score_calculator:
                # 既存のスコアリングシステムから取得を試行
                if hasattr(self.score_calculator, 'get_current_scores'):
                    current_scores = self.score_calculator.get_current_scores()
                    return current_scores.get(strategy_name, 0.5)
                elif hasattr(self.score_calculator, 'calculate_strategy_score'):
                    # デフォルトティッカーでスコア計算を試行
                    score_obj = self.score_calculator.calculate_strategy_score(
                        strategy_name, ticker="AAPL"  # デフォルトティッカー
                    )
                    if score_obj:
                        return score_obj.total_score
                # フォールバック: 戦略名ベースの簡易スコア
                return self._calculate_fallback_score(strategy_name)
            return 0.5
        except Exception as e:
            logger.warning(f"戦略スコア取得エラー: {e}")
            return 0.5
    
    def _calculate_fallback_score(self, strategy_name: str) -> float:
        """戦略名ベースの簡易スコア計算"""
        # 戦略タイプ別の基本スコア
        strategy_scores = {
            "vwap_bounce": 0.7,
            "momentum": 0.6,
            "momentum_strategy": 0.6,
            "breakout": 0.8,
            "breakout_strategy": 0.8,
            "mean_reversion": 0.5,
            "opening_gap": 0.6,
            "contrarian": 0.5
        }
        
        # 部分マッチングで基本スコアを取得
        for key, score in strategy_scores.items():
            if key.lower() in strategy_name.lower():
                return score
        
        return 0.5  # デフォルトスコア
    
    def _get_rule_priority(self, signal: StrategySignal) -> int:
        """ルール優先度の取得"""
        if signal.rule_priority is not None:
            return signal.rule_priority
        
        # シグナルタイプ別優先度
        if signal.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]:
            return self.rule_priorities["exit_signals"]
        
        # 戦略タイプ別優先度（メタデータから推定）
        strategy_type = signal.metadata.get("strategy_type", "default")
        return self.rule_priorities.get(strategy_type, self.rule_priorities["default"])
    
    def _resolve_score_based(self, conflict: SignalConflict) -> Optional[StrategySignal]:
        """スコアベース解決"""
        signals = conflict.conflicting_signals
        if not signals:
            return None
        
        # 戦略スコアが最も高いシグナルを選択
        best_signal = max(signals, key=lambda s: self._get_strategy_score(s.strategy_name))
        conflict.resolution_reason = f"スコアベース: 戦略スコア={self._get_strategy_score(best_signal.strategy_name):.3f}"
        
        return best_signal
    
    def _resolve_rule_based(self, conflict: SignalConflict) -> Optional[StrategySignal]:
        """ルールベース解決"""
        signals = conflict.conflicting_signals
        if not signals:
            return None
        
        # ルール優先度が最も高い（数値が小さい）シグナルを選択
        best_signal = min(signals, key=lambda s: self._get_rule_priority(s))
        conflict.resolution_reason = f"ルールベース: 優先度={self._get_rule_priority(best_signal)}"
        
        return best_signal
    
    def _resolve_consensus(self, conflict: SignalConflict) -> Optional[StrategySignal]:
        """コンセンサス解決（多数決）"""
        signals = conflict.conflicting_signals
        if not signals:
            return None
        
        # シグナルタイプ別の投票数を計算
        signal_votes = {}
        for signal in signals:
            signal_type = signal.signal_type
            if signal_type not in signal_votes:
                signal_votes[signal_type] = []
            signal_votes[signal_type].append(signal)
        
        # 最も票数の多いシグナルタイプを選択
        best_type = max(signal_votes.keys(), key=lambda t: len(signal_votes[t]))
        best_signals = signal_votes[best_type]
        
        # 同じタイプ内で最も信頼度の高いシグナルを選択
        best_signal = max(best_signals, key=lambda s: s.confidence)
        conflict.resolution_reason = f"コンセンサス: {best_type.value} ({len(best_signals)} 票)"
        
        return best_signal

class ResourceManager:
    """リソース管理（資金・リスク）"""
    
    def __init__(self, portfolio_calculator: Optional[PortfolioWeightCalculator] = None):
        self.portfolio_calculator = portfolio_calculator
        
        # リソース制限
        self.max_total_exposure = 1.0  # 最大エクスポージャー
        self.max_single_position = 0.3  # 単一ポジション最大サイズ
        self.max_concurrent_positions = 5  # 最大同時ポジション数
        
        # リスク制限
        self.max_portfolio_risk = 0.02  # 最大ポートフォリオリスク（日次VaR）
        self.max_correlation_exposure = 0.6  # 相関高戦略への最大エクスポージャー
        
        logger.info("ResourceManager initialized")
    
    def check_resource_availability(self, signals: List[StrategySignal], 
                                  current_portfolio: Dict[str, float],
                                  available_capital: float) -> Dict[str, bool]:
        """リソース利用可能性チェック"""
        results = {}
        
        for signal in signals:
            strategy_name = signal.strategy_name
            position_size = signal.position_size
            
            # 資金チェック
            capital_ok = self._check_capital_availability(
                position_size, available_capital, current_portfolio
            )
            
            # ポジション数チェック
            position_count_ok = self._check_position_count(current_portfolio)
            
            # エクスポージャーチェック
            exposure_ok = self._check_exposure_limits(
                strategy_name, position_size, current_portfolio
            )
            
            # リスクチェック
            risk_ok = self._check_risk_limits(
                strategy_name, position_size, current_portfolio
            )
            
            results[strategy_name] = capital_ok and position_count_ok and exposure_ok and risk_ok
        
        return results
    
    def _check_capital_availability(self, required_capital: float, 
                                   available_capital: float,
                                   current_portfolio: Dict[str, float]) -> bool:
        """資金利用可能性チェック"""
        actual_required = abs(required_capital) * available_capital
        return actual_required <= available_capital
    
    def _check_position_count(self, current_portfolio: Dict[str, float]) -> bool:
        """ポジション数チェック"""
        active_positions = len([pos for pos in current_portfolio.values() if abs(pos) > 0.001])
        return active_positions < self.max_concurrent_positions
    
    def _check_exposure_limits(self, strategy_name: str, position_size: float,
                              current_portfolio: Dict[str, float]) -> bool:
        """エクスポージャー制限チェック"""
        # 単一ポジションサイズチェック
        if abs(position_size) > self.max_single_position:
            return False
        
        # 総エクスポージャーチェック
        current_exposure = sum(abs(pos) for pos in current_portfolio.values())
        new_exposure = current_exposure + abs(position_size)
        
        return new_exposure <= self.max_total_exposure
    
    def _check_risk_limits(self, strategy_name: str, position_size: float,
                          current_portfolio: Dict[str, float]) -> bool:
        """リスク制限チェック"""
        try:
            # ポートフォリオリスク計算（簡易版）
            current_risk = self._calculate_portfolio_risk(current_portfolio)
            new_portfolio = current_portfolio.copy()
            new_portfolio[strategy_name] = new_portfolio.get(strategy_name, 0) + position_size
            new_risk = self._calculate_portfolio_risk(new_portfolio)
            
            return new_risk <= self.max_portfolio_risk
        except Exception as e:
            logger.warning(f"リスク計算エラー: {e}")
            return True  # エラー時は制限なしとする
    
    def _calculate_portfolio_risk(self, portfolio: Dict[str, float]) -> float:
        """ポートフォリオリスク計算（簡易版）"""
        # 簡易リスク計算（実装時により詳細な計算に置き換え）
        total_exposure = sum(abs(pos) for pos in portfolio.values())
        return total_exposure * 0.02  # 仮の計算

class SignalIntegrator:
    """
    3-3-1: シグナル統合エンジン
    複数戦略からのシグナルを統合し、競合を解決する
    """
    
    def __init__(self, strategy_selector: Optional[StrategySelector] = None,
                 portfolio_calculator: Optional[PortfolioWeightCalculator] = None,
                 score_calculator: Optional[StrategyScoreCalculator] = None,
                 config_file: Optional[str] = None):
        
        self.strategy_selector = strategy_selector
        self.portfolio_calculator = portfolio_calculator
        self.score_calculator = score_calculator
        
        # サブコンポーネント初期化
        self.conflict_detector = ConflictDetector()
        self.priority_resolver = PriorityResolver(strategy_selector, score_calculator)
        self.resource_manager = ResourceManager(portfolio_calculator)
        
        # 設定管理
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), "signal_integration_config.json"
        )
        self.config = self._load_config()
        
        # 状態管理
        self.current_signals: List[StrategySignal] = []
        self.active_conflicts: List[SignalConflict] = []
        self.resolved_signals: List[StrategySignal] = []
        
        # 履歴管理
        self.signal_history: List[Tuple[datetime, List[StrategySignal]]] = []
        self.resolution_history: List[Tuple[datetime, SignalConflict]] = []
        
        # パフォーマンス追跡
        self.integration_stats = {
            "total_signals_processed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "integration_failures": 0,
            "average_processing_time": 0.0
        }
        
        logger.info("SignalIntegrator initialized with 3-3-1 functionality")
    
    def integrate_signals(self, strategy_signals: Dict[str, StrategySignal],
                         current_portfolio: Dict[str, float],
                         available_capital: float,
                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        メインの統合処理
        
        Args:
            strategy_signals: 戦略別シグナル辞書
            current_portfolio: 現在のポートフォリオ
            available_capital: 利用可能資金
            timestamp: 処理タイムスタンプ
            
        Returns:
            統合結果辞書
        """
        start_time = datetime.now()
        timestamp = timestamp or start_time
        
        try:
            # 1. 入力検証
            if not self._validate_inputs(strategy_signals, current_portfolio, available_capital):
                return self._create_error_result("入力検証失敗")
            
            # 2. シグナル前処理
            processed_signals = self._preprocess_signals(list(strategy_signals.values()), timestamp)
            
            # 3. 競合検出
            conflicts = self.conflict_detector.detect_conflicts(
                processed_signals, current_portfolio, available_capital, self.config["risk_limits"]
            )
            
            # 4. 競合解決
            resolved_signals = self._resolve_conflicts(conflicts, processed_signals)
            
            # 5. リソースチェック
            resource_check = self.resource_manager.check_resource_availability(
                resolved_signals, current_portfolio, available_capital
            )
            
            # 6. 最終シグナル決定
            final_signals = self._finalize_signals(resolved_signals, resource_check)
            
            # 7. 結果作成
            result = self._create_integration_result(
                final_signals, conflicts, resource_check, timestamp
            )
            
            # 8. 履歴更新
            self._update_history(processed_signals, conflicts, timestamp)
            
            # 9. 統計更新
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, len(conflicts))
            
            logger.info(f"シグナル統合完了: {len(final_signals)} 信号, {len(conflicts)} 競合")
            return result
            
        except Exception as e:
            logger.error(f"シグナル統合エラー: {e}")
            self.integration_stats["integration_failures"] += 1
            return self._create_error_result(f"統合処理エラー: {str(e)}")
    
    def _validate_inputs(self, strategy_signals: Dict[str, StrategySignal],
                        current_portfolio: Dict[str, float],
                        available_capital: float) -> bool:
        """入力データの検証"""
        try:
            # シグナル検証
            if not strategy_signals:
                logger.warning("シグナルが空です")
                return False
            
            for strategy, signal in strategy_signals.items():
                if not isinstance(signal, StrategySignal):
                    logger.error(f"無効なシグナルタイプ: {strategy}")
                    return False
                
                if signal.strategy_name != strategy:
                    logger.error(f"戦略名不一致: {strategy} vs {signal.strategy_name}")
                    return False
            
            # ポートフォリオ検証
            if not isinstance(current_portfolio, dict):
                logger.error("ポートフォリオが辞書ではありません")
                return False
            
            # 資金検証
            if available_capital < 0:
                logger.error("利用可能資金が負の値です")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"入力検証エラー: {e}")
            return False
    
    def _preprocess_signals(self, signals: List[StrategySignal], 
                           timestamp: datetime) -> List[StrategySignal]:
        """シグナル前処理"""
        processed = []
        
        for signal in signals:
            # タイムスタンプ正規化
            if signal.timestamp is None:
                signal.timestamp = timestamp
            
            # 戦略スコア取得
            if signal.strategy_score is None:
                signal.strategy_score = self.priority_resolver._get_strategy_score(signal.strategy_name)
            
            # 信頼度正規化
            signal.confidence = max(0.0, min(1.0, signal.confidence))
            
            # ポジションサイズ検証
            if signal.position_size == 0.0 and signal.signal_type in [
                SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT
            ]:
                # デフォルトポジションサイズ設定
                signal.position_size = self.config.get("default_position_size", 0.1)
            
            processed.append(signal)
        
        return processed
    
    def _resolve_conflicts(self, conflicts: List[SignalConflict], 
                          all_signals: List[StrategySignal]) -> List[StrategySignal]:
        """競合解決処理"""
        resolved_signals = []
        conflicted_strategies = set()
        
        # 競合のあるシグナルを解決
        for conflict in conflicts:
            conflict.resolution_method = PriorityMethod.HYBRID  # デフォルト方式
            resolved_signal = self.priority_resolver.resolve_conflict(conflict)
            
            if resolved_signal:
                resolved_signals.append(resolved_signal)
                conflicted_strategies.update(
                    s.strategy_name for s in conflict.conflicting_signals
                )
                conflict.resolved_signal = resolved_signal
        
        # 競合のないシグナルを追加
        for signal in all_signals:
            if signal.strategy_name not in conflicted_strategies:
                resolved_signals.append(signal)
        
        return resolved_signals
    
    def _finalize_signals(self, signals: List[StrategySignal], 
                         resource_check: Dict[str, bool]) -> List[StrategySignal]:
        """最終シグナル決定"""
        final_signals = []
        
        for signal in signals:
            # リソースチェック
            if not resource_check.get(signal.strategy_name, False):
                logger.warning(f"リソース不足でシグナル除外: {signal.strategy_name}")
                continue
            
            # 除外ルールチェック
            if self._should_exclude_signal(signal):
                continue
            
            final_signals.append(signal)
        
        return final_signals
    
    def _should_exclude_signal(self, signal: StrategySignal) -> bool:
        """シグナル除外判定"""
        # 信頼度フィルター
        min_confidence = self.config.get("min_signal_confidence", 0.3)
        if signal.confidence < min_confidence:
            return True
        
        # 戦略スコアフィルター
        min_strategy_score = self.config.get("min_strategy_score", 0.2)
        if signal.strategy_score and signal.strategy_score < min_strategy_score:
            return True
        
        return False
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        default_config = {
            "priority_method": "hybrid",
            "default_position_size": 0.1,
            "min_signal_confidence": 0.3,
            "min_strategy_score": 0.2,
            "max_concurrent_signals": 10,
            "risk_limits": {
                "max_portfolio_risk": 0.02,
                "max_single_exposure": 0.3,
                "max_total_exposure": 1.0
            },
            "conflict_resolution": {
                "timeout_seconds": 30,
                "fallback_method": "score_based",
                "enable_consensus": True
            },
            "performance_tracking": {
                "enable_history": True,
                "max_history_entries": 1000,
                "enable_stats": True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"設定ファイル読み込み完了: {self.config_file}")
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
        
        return default_config
    
    def _create_integration_result(self, signals: List[StrategySignal],
                                  conflicts: List[SignalConflict],
                                  resource_check: Dict[str, bool],
                                  timestamp: datetime) -> Dict[str, Any]:
        """統合結果の作成"""
        return {
            "timestamp": timestamp.isoformat(),
            "success": True,
            "signals": [self._signal_to_dict(s) for s in signals],
            "conflicts": [self._conflict_to_dict(c) for c in conflicts],
            "resource_status": resource_check,
            "statistics": {
                "total_signals": len(signals),
                "conflicts_count": len(conflicts),
                "resolved_signals": len([s for s in signals if s is not None]),
                "resource_failures": len([v for v in resource_check.values() if not v])
            },
            "metadata": {
                "integration_version": "3.3.1",
                "processing_method": self.config["priority_method"],
                "config_applied": True
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果の作成"""
        return {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": error_message,
            "signals": [],
            "conflicts": [],
            "resource_status": {},
            "statistics": {"error": True},
            "metadata": {"integration_version": "3.3.1"}
        }
    
    def _signal_to_dict(self, signal: StrategySignal) -> Dict[str, Any]:
        """シグナルを辞書に変換"""
        return {
            "strategy_name": signal.strategy_name,
            "signal_type": signal.signal_type.value,
            "timestamp": signal.timestamp.isoformat(),
            "confidence": signal.confidence,
            "position_size": signal.position_size,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "strategy_score": signal.strategy_score,
            "adjusted_priority": signal.adjusted_priority,
            "metadata": signal.metadata
        }
    
    def _conflict_to_dict(self, conflict: SignalConflict) -> Dict[str, Any]:
        """競合を辞書に変換"""
        return {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type.value,
            "detected_at": conflict.detected_at.isoformat(),
            "resolution_method": conflict.resolution_method.value if conflict.resolution_method else None,
            "resolution_reason": conflict.resolution_reason,
            "conflicting_strategies": [s.strategy_name for s in conflict.conflicting_signals],
            "resolved_strategy": conflict.resolved_signal.strategy_name if conflict.resolved_signal else None
        }
    
    def _update_history(self, signals: List[StrategySignal],
                       conflicts: List[SignalConflict],
                       timestamp: datetime):
        """履歴の更新"""
        try:
            # シグナル履歴
            self.signal_history.append((timestamp, signals.copy()))
            
            # 競合履歴
            for conflict in conflicts:
                self.resolution_history.append((timestamp, conflict))
            
            # 履歴サイズ制限
            max_entries = self.config.get("performance_tracking", {}).get("max_history_entries", 1000)
            
            if len(self.signal_history) > max_entries:
                self.signal_history = self.signal_history[-max_entries:]
            
            if len(self.resolution_history) > max_entries:
                self.resolution_history = self.resolution_history[-max_entries:]
                
        except Exception as e:
            logger.warning(f"履歴更新エラー: {e}")
    
    def _update_stats(self, processing_time: float, conflicts_count: int):
        """統計の更新"""
        try:
            self.integration_stats["total_signals_processed"] += 1
            self.integration_stats["conflicts_detected"] += conflicts_count
            if conflicts_count > 0:
                self.integration_stats["conflicts_resolved"] += 1
            
            # 平均処理時間の更新
            current_avg = self.integration_stats["average_processing_time"]
            total_processed = self.integration_stats["total_signals_processed"]
            
            new_avg = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
            self.integration_stats["average_processing_time"] = new_avg
            
        except Exception as e:
            logger.warning(f"統計更新エラー: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """統合統計の取得"""
        return self.integration_stats.copy()
    
    def reset_stats(self):
        """統計のリセット"""
        self.integration_stats = {
            "total_signals_processed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "integration_failures": 0,
            "average_processing_time": 0.0
        }
        logger.info("統合統計をリセットしました")

# ファクトリー関数
def create_signal_integrator(strategy_selector: Optional[StrategySelector] = None,
                           portfolio_calculator: Optional[PortfolioWeightCalculator] = None,
                           score_calculator: Optional[StrategyScoreCalculator] = None,
                           config_file: Optional[str] = None) -> SignalIntegrator:
    """SignalIntegratorのファクトリー関数"""
    return SignalIntegrator(
        strategy_selector=strategy_selector,
        portfolio_calculator=portfolio_calculator,
        score_calculator=score_calculator,
        config_file=config_file
    )

def create_strategy_signal(strategy_name: str,
                          signal_type: SignalType,
                          confidence: float = 1.0,
                          position_size: float = 0.0,
                          metadata: Optional[Dict[str, Any]] = None) -> StrategySignal:
    """StrategySignalのファクトリー関数"""
    return StrategySignal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        timestamp=datetime.now(),
        confidence=confidence,
        position_size=position_size,
        metadata=metadata or {}
    )

# エラーハンドリング用ユーティリティ
def validate_integration_config(config: Dict[str, Any]) -> bool:
    """統合設定の検証"""
    required_keys = [
        "priority_method", "risk_limits", "conflict_resolution",
        "performance_tracking"
    ]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"必須設定キー不足: {key}")
            return False
    
    return True

def monitor_resource_usage():
    """リソース使用量の監視"""
    try:
        process = psutil.Process()
        
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        logger.debug(f"メモリ使用量: {memory_info.rss / 1024 / 1024:.1f} MB")
        logger.debug(f"CPU使用率: {cpu_percent:.1f}%")
        
        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "cpu_percent": cpu_percent
        }
    except ImportError:
        return {"memory_mb": 0, "cpu_percent": 0}
    except Exception as e:
        logger.warning(f"リソース監視エラー: {e}")
        return {"memory_mb": 0, "cpu_percent": 0}
