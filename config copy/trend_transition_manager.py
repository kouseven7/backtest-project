"""
Module: Trend Transition Manager
File: trend_transition_manager.py
Description: 
  トレンド移行期の特別処理ルール管理システム
  エントリー制限、ポジション管理調整、リスク制御を統合管理
  2-2-2「トレンド移行期の特別処理ルール」のメイン管理コンポーネント

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - pandas
  - numpy
  - indicators.trend_transition_detector
  - config.enhanced_strategy_scoring_model
  - config.risk_management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

# プロジェクトパスの追加
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 内部モジュールのインポート
try:
    from indicators.trend_transition_detector import TrendTransitionDetector, TransitionDetectionResult
    from config.enhanced_strategy_scoring_model import TrendConfidenceIntegrator
    from config.risk_management import RiskManagement
except ImportError as e:
    print(f"Import warning: {e}")

# ロガー設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class TransitionRule:
    """移行期特別処理ルール"""
    rule_type: str  # 'entry_restriction', 'position_adjustment', 'risk_modification'
    condition: str  # 適用条件
    action: str     # 実行アクション
    parameters: Dict[str, Any]
    priority: int   # 優先度（1が最高）
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_type': self.rule_type,
            'condition': self.condition,
            'action': self.action,
            'parameters': self.parameters,
            'priority': self.priority,
            'enabled': self.enabled
        }

@dataclass
class PositionAdjustment:
    """ポジション調整指示"""
    strategy_name: str
    current_position_size: float
    recommended_size: float
    adjustment_ratio: float
    reason: str
    urgency: str  # 'low', 'medium', 'high', 'immediate'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'current_position_size': self.current_position_size,
            'recommended_size': self.recommended_size,
            'adjustment_ratio': self.adjustment_ratio,
            'reason': self.reason,
            'urgency': self.urgency
        }

@dataclass
class TransitionManagementResult:
    """移行期管理結果"""
    is_transition_period: bool
    transition_detection: TransitionDetectionResult
    entry_allowed: bool
    entry_restrictions: List[str]
    position_adjustments: List[PositionAdjustment]
    risk_modifications: Dict[str, Any]
    confidence_adjustment: float
    management_timestamp: datetime
    active_rules: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_transition_period': self.is_transition_period,
            'transition_detection': self.transition_detection.to_dict(),
            'entry_allowed': self.entry_allowed,
            'entry_restrictions': self.entry_restrictions,
            'position_adjustments': [adj.to_dict() for adj in self.position_adjustments],
            'risk_modifications': self.risk_modifications,
            'confidence_adjustment': self.confidence_adjustment,
            'management_timestamp': self.management_timestamp.isoformat(),
            'active_rules': self.active_rules
        }

class TrendTransitionManager:
    """
    トレンド移行期特別処理ルール管理システム
    
    機能:
    1. 移行期検出と管理
    2. エントリー制限制御
    3. ポジション管理調整
    4. リスク管理の動的調整
    5. 信頼度スコア調整
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 detection_sensitivity: str = "medium",
                 default_position_reduction: float = 0.3,
                 confidence_penalty_factor: float = 0.2):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルパス
            detection_sensitivity: 検出感度
            default_position_reduction: デフォルトポジション削減率
            confidence_penalty_factor: 信頼度ペナルティ係数
        """
        self.config_path = config_path or self._get_default_config_path()
        self.detection_sensitivity = detection_sensitivity
        self.default_position_reduction = default_position_reduction
        self.confidence_penalty_factor = confidence_penalty_factor
        
        # コンポーネント初期化
        self.detector = TrendTransitionDetector(detection_sensitivity=detection_sensitivity)
        self.confidence_integrator = TrendConfidenceIntegrator()
        
        # ルール管理
        self.rules = self._load_rules()
        self.active_transitions = {}  # strategy_name -> TransitionDetectionResult
        
        # 統計情報
        self.stats = {
            'total_detections': 0,
            'entries_blocked': 0,
            'positions_adjusted': 0,
            'last_update': datetime.now()
        }
        
        logger.info(f"TrendTransitionManager initialized with sensitivity: {detection_sensitivity}")
    
    def manage_transition(self,
                         data: pd.DataFrame,
                         strategy_name: str,
                         current_positions: Optional[Dict[str, float]] = None,
                         price_column: str = "Adj Close") -> TransitionManagementResult:
        """
        移行期管理の実行
        
        Parameters:
            data: 株価データ
            strategy_name: 戦略名
            current_positions: 現在のポジション（ticker -> size）
            price_column: 価格カラム名
            
        Returns:
            TransitionManagementResult: 管理結果
        """
        try:
            # 1. 移行期検出
            detection_result = self.detector.detect_transition(data, strategy_name, price_column)
            
            # 2. 移行期状態の更新
            self._update_transition_state(strategy_name, detection_result)
            
            # 3. 特別処理ルールの適用
            management_result = self._apply_transition_rules(
                detection_result, strategy_name, current_positions
            )
            
            # 4. 統計情報更新
            self._update_statistics(management_result)
            
            logger.debug(f"Transition management completed for {strategy_name}")
            return management_result
            
        except Exception as e:
            logger.error(f"Error in transition management: {e}")
            return self._create_error_result(strategy_name)
    
    def _apply_transition_rules(self,
                               detection: TransitionDetectionResult,
                               strategy_name: str,
                               current_positions: Optional[Dict[str, float]]) -> TransitionManagementResult:
        """特別処理ルールの適用"""
        
        # エントリー制限判定
        entry_result = self._evaluate_entry_restrictions(detection, strategy_name)
        
        # ポジション調整判定
        position_adjustments = self._evaluate_position_adjustments(
            detection, strategy_name, current_positions
        )
        
        # リスク管理調整
        risk_modifications = self._evaluate_risk_modifications(detection, strategy_name)
        
        # 信頼度調整
        confidence_adjustment = self._calculate_confidence_adjustment(detection)
        
        # アクティブルール一覧
        active_rules = self._get_active_rules(detection)
        
        return TransitionManagementResult(
            is_transition_period=detection.is_transition_period,
            transition_detection=detection,
            entry_allowed=entry_result['allowed'],
            entry_restrictions=entry_result['restrictions'],
            position_adjustments=position_adjustments,
            risk_modifications=risk_modifications,
            confidence_adjustment=confidence_adjustment,
            management_timestamp=datetime.now(),
            active_rules=active_rules
        )
    
    def _evaluate_entry_restrictions(self,
                                   detection: TransitionDetectionResult,
                                   strategy_name: str) -> Dict[str, Any]:
        """エントリー制限評価"""
        restrictions = []
        allowed = True
        
        if not detection.is_transition_period:
            return {'allowed': True, 'restrictions': []}
        
        # リスクレベル別制限
        if detection.risk_level == 'high':
            allowed = False
            restrictions.extend([
                "high_risk_transition_period",
                "volatility_spike_detected",
                "multiple_indicators_triggered"
            ])
        elif detection.risk_level == 'medium':
            # 条件付き許可
            if detection.confidence_score > 0.7:
                allowed = False
                restrictions.append("medium_risk_high_confidence")
            else:
                restrictions.append("reduced_size_only")
        
        # 移行タイプ別制限
        if detection.transition_type == 'trend_to_range':
            restrictions.append("trend_breakdown_caution")
        elif detection.transition_type == 'unknown_transition':
            allowed = False
            restrictions.append("unknown_market_state")
        
        # 戦略特化制限
        strategy_rules = self._get_strategy_specific_rules(strategy_name, detection)
        restrictions.extend(strategy_rules)
        
        return {'allowed': allowed, 'restrictions': restrictions}
    
    def _evaluate_position_adjustments(self,
                                     detection: TransitionDetectionResult,
                                     strategy_name: str,
                                     current_positions: Optional[Dict[str, float]]) -> List[PositionAdjustment]:
        """ポジション調整評価"""
        adjustments = []
        
        if not detection.is_transition_period or not current_positions:
            return adjustments
        
        for ticker, current_size in current_positions.items():
            if current_size == 0:
                continue
            
            # 調整率計算
            adjustment_ratio = self._calculate_position_adjustment_ratio(detection, strategy_name)
            recommended_size = current_size * (1.0 - adjustment_ratio)
            
            # 緊急度判定
            urgency = self._determine_adjustment_urgency(detection, adjustment_ratio)
            
            # 調整理由
            reason = self._generate_adjustment_reason(detection, adjustment_ratio)
            
            if adjustment_ratio > 0.1:  # 10%以上の調整が必要な場合
                adjustments.append(PositionAdjustment(
                    strategy_name=f"{strategy_name}_{ticker}",
                    current_position_size=current_size,
                    recommended_size=recommended_size,
                    adjustment_ratio=adjustment_ratio,
                    reason=reason,
                    urgency=urgency
                ))
        
        return adjustments
    
    def _calculate_position_adjustment_ratio(self,
                                           detection: TransitionDetectionResult,
                                           strategy_name: str) -> float:
        """ポジション調整率計算"""
        base_reduction = 0.0
        
        # リスクレベル別調整
        risk_multipliers = {
            'high': 0.5,     # 50%削減
            'medium': 0.3,   # 30%削減
            'low': 0.1       # 10%削減
        }
        base_reduction = risk_multipliers.get(detection.risk_level, 0.2)
        
        # 信頼度による調整
        confidence_factor = detection.confidence_score * 0.3
        
        # ボラティリティによる調整
        volatility_factor = min(0.3, (detection.volatility_factor - 1.0) * 0.2)
        
        # 総合調整率
        total_reduction = min(0.8, base_reduction + confidence_factor + volatility_factor)
        
        return max(0.0, total_reduction)
    
    def _determine_adjustment_urgency(self,
                                    detection: TransitionDetectionResult,
                                    adjustment_ratio: float) -> str:
        """調整緊急度判定"""
        if detection.risk_level == 'high' and adjustment_ratio > 0.4:
            return 'immediate'
        elif detection.risk_level == 'high' or adjustment_ratio > 0.3:
            return 'high'
        elif adjustment_ratio > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_adjustment_reason(self,
                                  detection: TransitionDetectionResult,
                                  adjustment_ratio: float) -> str:
        """調整理由生成"""
        reasons = []
        
        if detection.risk_level == 'high':
            reasons.append("高リスク移行期")
        
        if detection.volatility_factor > 2.0:
            reasons.append("ボラティリティ急上昇")
        
        if detection.confidence_score > 0.7:
            reasons.append("高信頼度移行検出")
        
        if len(detection.indicators_used) >= 3:
            reasons.append("複数指標同時警告")
        
        return ", ".join(reasons) if reasons else "移行期リスク管理"
    
    def _evaluate_risk_modifications(self,
                                   detection: TransitionDetectionResult,
                                   strategy_name: str) -> Dict[str, Any]:
        """リスク管理調整評価"""
        modifications = {}
        
        if not detection.is_transition_period:
            return modifications
        
        # ストップロス調整
        if detection.risk_level in ['high', 'medium']:
            modifications['tighten_stop_loss'] = {
                'factor': 0.7 if detection.risk_level == 'high' else 0.85,
                'reason': 'transition_period_risk'
            }
        
        # ポジションサイズ制限
        if detection.risk_level == 'high':
            modifications['max_position_size'] = {
                'limit': 0.5,  # 通常の50%に制限
                'reason': 'high_risk_transition'
            }
        
        # 監視頻度調整
        if detection.confidence_score > 0.6:
            modifications['monitoring_frequency'] = {
                'multiplier': 2.0,  # 2倍の頻度
                'reason': 'active_transition_monitoring'
            }
        
        return modifications
    
    def _calculate_confidence_adjustment(self, detection: TransitionDetectionResult) -> float:
        """信頼度調整計算"""
        if not detection.is_transition_period:
            return 0.0
        
        # 基本ペナルティ
        base_penalty = self.confidence_penalty_factor
        
        # リスクレベル別調整
        risk_multipliers = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }
        
        risk_factor = risk_multipliers.get(detection.risk_level, 1.0)
        
        # 最終調整値（負の値として返す）
        adjustment = -min(0.5, base_penalty * risk_factor * detection.confidence_score)
        
        return adjustment
    
    def _get_active_rules(self, detection: TransitionDetectionResult) -> List[str]:
        """アクティブルール取得"""
        active_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if self._rule_condition_met(rule, detection):
                active_rules.append(f"{rule.rule_type}:{rule.action}")
        
        return active_rules
    
    def _rule_condition_met(self, rule: TransitionRule, detection: TransitionDetectionResult) -> bool:
        """ルール条件判定"""
        condition = rule.condition
        
        # 基本条件
        if condition == 'is_transition':
            return detection.is_transition_period
        elif condition == 'high_risk':
            return detection.risk_level == 'high'
        elif condition == 'medium_risk':
            return detection.risk_level == 'medium'
        elif condition == 'high_confidence':
            return detection.confidence_score > 0.7
        elif condition == 'high_volatility':
            return detection.volatility_factor > 2.0
        
        return False
    
    def _get_strategy_specific_rules(self,
                                   strategy_name: str,
                                   detection: TransitionDetectionResult) -> List[str]:
        """戦略特化ルール取得"""
        rules = []
        
        # VWAP戦略特化
        if strategy_name.startswith("VWAP_"):
            if detection.volatility_factor > 1.8:
                rules.append("vwap_high_volatility_restriction")
        
        # ブレイクアウト戦略特化
        if "Breakout" in strategy_name:
            if detection.transition_type == 'trend_to_range':
                rules.append("breakout_range_transition_caution")
        
        # モメンタム戦略特化
        if "Momentum" in strategy_name:
            if detection.trend_strength_change > 0.5:
                rules.append("momentum_strength_change_warning")
        
        return rules
    
    def _update_transition_state(self,
                               strategy_name: str,
                               detection: TransitionDetectionResult):
        """移行期状態更新"""
        self.active_transitions[strategy_name] = detection
        
        # 古い状態のクリーンアップ（24時間以上前）
        cutoff_time = datetime.now() - timedelta(hours=24)
        to_remove = []
        
        for name, det in self.active_transitions.items():
            if det.detection_timestamp < cutoff_time:
                to_remove.append(name)
        
        for name in to_remove:
            del self.active_transitions[name]
    
    def _update_statistics(self, result: TransitionManagementResult):
        """統計情報更新"""
        if result.is_transition_period:
            self.stats['total_detections'] += 1
        
        if not result.entry_allowed:
            self.stats['entries_blocked'] += 1
        
        if result.position_adjustments:
            self.stats['positions_adjusted'] += len(result.position_adjustments)
        
        self.stats['last_update'] = datetime.now()
    
    def _load_rules(self) -> List[TransitionRule]:
        """ルール設定読み込み"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                rules = []
                for rule_data in rules_data.get('rules', []):
                    rules.append(TransitionRule(**rule_data))
                
                logger.info(f"Loaded {len(rules)} transition rules from {self.config_path}")
                return rules
            else:
                logger.info("Config file not found, using default rules")
                return self._get_default_rules()
                
        except Exception as e:
            logger.warning(f"Failed to load rules: {e}, using defaults")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> List[TransitionRule]:
        """デフォルトルール定義"""
        return [
            TransitionRule(
                rule_type='entry_restriction',
                condition='high_risk',
                action='block_all_entries',
                parameters={'exceptions': []},
                priority=1
            ),
            TransitionRule(
                rule_type='position_adjustment',
                condition='is_transition',
                action='reduce_positions',
                parameters={'reduction_factor': 0.3},
                priority=2
            ),
            TransitionRule(
                rule_type='risk_modification',
                condition='high_volatility',
                action='tighten_stops',
                parameters={'factor': 0.7},
                priority=3
            )
        ]
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパス"""
        return os.path.join(
            os.path.dirname(__file__), 
            "..", "config", "transition_rules.json"
        )
    
    def _create_error_result(self, strategy_name: str) -> TransitionManagementResult:
        """エラー時のデフォルト結果"""
        error_detection = TransitionDetectionResult(
            is_transition_period=False,
            transition_type='error',
            confidence_score=0.0,
            volatility_factor=1.0,
            trend_strength_change=0.0,
            detection_timestamp=datetime.now(),
            indicators_used=[],
            risk_level='unknown',
            recommended_actions=['error_state']
        )
        
        return TransitionManagementResult(
            is_transition_period=False,
            transition_detection=error_detection,
            entry_allowed=True,  # エラー時は制限しない
            entry_restrictions=[],
            position_adjustments=[],
            risk_modifications={},
            confidence_adjustment=0.0,
            management_timestamp=datetime.now(),
            active_rules=[]
        )
    
    # 公開インターフェース
    def get_transition_status(self, strategy_name: str) -> Optional[TransitionDetectionResult]:
        """戦略の移行期状態取得"""
        return self.active_transitions.get(strategy_name)
    
    def is_entry_allowed(self, strategy_name: str) -> bool:
        """エントリー許可状態確認"""
        status = self.get_transition_status(strategy_name)
        if not status or not status.is_transition_period:
            return True
        
        return status.risk_level != 'high'
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return self.stats.copy()
    
    def save_rules(self, filepath: Optional[str] = None):
        """ルール設定保存"""
        path = filepath or self.config_path
        
        try:
            rules_data = {
                'rules': [rule.to_dict() for rule in self.rules],
                'last_updated': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Rules saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")

# 便利関数
def manage_trend_transition(data: pd.DataFrame,
                          strategy_name: str,
                          current_positions: Optional[Dict[str, float]] = None,
                          **kwargs) -> TransitionManagementResult:
    """
    トレンド移行期管理の便利関数
    
    Parameters:
        data: 株価データ
        strategy_name: 戦略名
        current_positions: 現在のポジション
        **kwargs: TrendTransitionManagerの初期化パラメータ
        
    Returns:
        TransitionManagementResult: 管理結果
    """
    manager = TrendTransitionManager(**kwargs)
    return manager.manage_transition(data, strategy_name, current_positions)

if __name__ == "__main__":
    # テスト用のサンプル実行
    print("TrendTransitionManager - 開発版テスト")
    
    # サンプルデータ作成
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.03)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Adj Close': prices
    })
    
    # サンプルポジション
    positions = {'AAPL': 100.0, 'GOOGL': 50.0}
    
    # 管理テスト
    result = manage_trend_transition(sample_data, "VWAP_Breakout", positions)
    print(f"管理結果: 移行期={result.is_transition_period}, エントリー許可={result.entry_allowed}")
    print(f"ポジション調整数: {len(result.position_adjustments)}")
