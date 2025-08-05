"""
Module: Constraint Manager
File: constraint_manager.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  最適化制約条件の管理システム

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """制約タイプ"""
    WEIGHT_CONSTRAINT = "weight_constraint"
    VOLATILITY_CONSTRAINT = "volatility_constraint"
    DRAWDOWN_CONSTRAINT = "drawdown_constraint"
    CORRELATION_CONSTRAINT = "correlation_constraint"
    CONCENTRATION_CONSTRAINT = "concentration_constraint"
    TURNOVER_CONSTRAINT = "turnover_constraint"
    LEVERAGE_CONSTRAINT = "leverage_constraint"

class ConstraintSeverity(Enum):
    """制約違反の深刻度"""
    HARD = "hard"          # 厳格な制約（違反は許可されない）
    SOFT = "soft"          # 柔軟な制約（違反時にペナルティ）
    ADAPTIVE = "adaptive"  # 適応的制約（市場環境に応じて調整）

@dataclass
class ConstraintViolation:
    """制約違反情報"""
    constraint_name: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    violation_amount: float
    limit_value: float
    current_value: float
    penalty_score: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConstraintResult:
    """制約チェック結果"""
    is_satisfied: bool
    total_penalty: float
    violations: List[ConstraintViolation]
    checked_constraints: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizationConstraint(ABC):
    """最適化制約の抽象基底クラス"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.severity = ConstraintSeverity(config.get('severity', 'soft'))
        self.enabled = config.get('enabled', True)
        self.penalty_multiplier = config.get('penalty_multiplier', 1.0)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> ConstraintViolation:
        """制約をチェック"""
        pass
    
    @abstractmethod
    def get_constraint_type(self) -> ConstraintType:
        """制約タイプを取得"""
        pass
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """ペナルティを計算"""
        if self.severity == ConstraintSeverity.HARD:
            return float('inf') if violation_amount > 0 else 0.0
        elif self.severity == ConstraintSeverity.SOFT:
            return violation_amount * self.penalty_multiplier
        else:  # ADAPTIVE
            return violation_amount * self.penalty_multiplier * 0.5

class WeightConstraint(OptimizationConstraint):
    """重み制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.min_weight = config.get('min_weight', 0.0)
        self.max_weight = config.get('max_weight', 1.0)
        self.max_single_weight = config.get('max_single_weight', 0.4)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """重み制約をチェック"""
        violations = []
        max_violation = 0.0
        
        for strategy, weight in weights.items():
            # 最小重み制約
            if weight < self.min_weight:
                violation = self.min_weight - weight
                max_violation = max(max_violation, violation)
                
            # 最大重み制約
            if weight > self.max_weight:
                violation = weight - self.max_weight
                max_violation = max(max_violation, violation)
                
            # 単一戦略最大重み制約
            if weight > self.max_single_weight:
                violation = weight - self.max_single_weight
                max_violation = max(max_violation, violation)
        
        if max_violation > 0:
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=max_violation,
                limit_value=self.max_single_weight,
                current_value=max(weights.values()) if weights else 0.0,
                penalty_score=self.calculate_penalty(max_violation),
                description=f"Weight constraint violated by {max_violation:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.WEIGHT_CONSTRAINT

class VolatilityConstraint(OptimizationConstraint):
    """ボラティリティ制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_portfolio_volatility = config.get('max_portfolio_volatility', 0.25)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """ボラティリティ制約をチェック"""
        portfolio_volatility = metrics.get('portfolio_volatility', 0.0)
        
        if portfolio_volatility > self.max_portfolio_volatility:
            violation = portfolio_volatility - self.max_portfolio_volatility
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=violation,
                limit_value=self.max_portfolio_volatility,
                current_value=portfolio_volatility,
                penalty_score=self.calculate_penalty(violation),
                description=f"Portfolio volatility {portfolio_volatility:.4f} exceeds limit {self.max_portfolio_volatility:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.VOLATILITY_CONSTRAINT

class DrawdownConstraint(OptimizationConstraint):
    """ドローダウン制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """ドローダウン制約をチェック"""
        max_drawdown = metrics.get('max_drawdown', 0.0)
        
        if max_drawdown > self.max_drawdown:
            violation = max_drawdown - self.max_drawdown
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=violation,
                limit_value=self.max_drawdown,
                current_value=max_drawdown,
                penalty_score=self.calculate_penalty(violation),
                description=f"Maximum drawdown {max_drawdown:.4f} exceeds limit {self.max_drawdown:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.DRAWDOWN_CONSTRAINT

class CorrelationConstraint(OptimizationConstraint):
    """相関制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.7)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """相関制約をチェック"""
        correlation_exposure = metrics.get('correlation_exposure', 0.0)
        
        if correlation_exposure > self.max_correlation_exposure:
            violation = correlation_exposure - self.max_correlation_exposure
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=violation,
                limit_value=self.max_correlation_exposure,
                current_value=correlation_exposure,
                penalty_score=self.calculate_penalty(violation),
                description=f"Correlation exposure {correlation_exposure:.4f} exceeds limit {self.max_correlation_exposure:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.CORRELATION_CONSTRAINT

class ConcentrationConstraint(OptimizationConstraint):
    """集中度制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_concentration_ratio = config.get('max_concentration_ratio', 0.6)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """集中度制約をチェック"""
        # ハーフィンダール指数（HHI）を計算
        weight_values = list(weights.values())
        hhi = sum(w**2 for w in weight_values) if weight_values else 0.0
        
        if hhi > self.max_concentration_ratio:
            violation = hhi - self.max_concentration_ratio
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=violation,
                limit_value=self.max_concentration_ratio,
                current_value=hhi,
                penalty_score=self.calculate_penalty(violation),
                description=f"Concentration ratio {hhi:.4f} exceeds limit {self.max_concentration_ratio:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.CONCENTRATION_CONSTRAINT

class TurnoverConstraint(OptimizationConstraint):
    """回転率制約"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_turnover = config.get('max_turnover', 0.5)
        
    def check_constraint(self, weights: Dict[str, float], metrics: Dict[str, float], **kwargs) -> Optional[ConstraintViolation]:
        """回転率制約をチェック"""
        current_weights = weights
        previous_weights = kwargs.get('previous_weights', {})
        
        if not previous_weights:
            return None  # 前回重みがない場合はスキップ
        
        # 回転率計算
        turnover = 0.0
        for strategy in set(current_weights.keys()) | set(previous_weights.keys()):
            current_w = current_weights.get(strategy, 0.0)
            previous_w = previous_weights.get(strategy, 0.0)
            turnover += abs(current_w - previous_w)
        
        turnover = turnover / 2  # 両方向の変化の合計を2で割る
        
        if turnover > self.max_turnover:
            violation = turnover - self.max_turnover
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.get_constraint_type(),
                severity=self.severity,
                violation_amount=violation,
                limit_value=self.max_turnover,
                current_value=turnover,
                penalty_score=self.calculate_penalty(violation),
                description=f"Portfolio turnover {turnover:.4f} exceeds limit {self.max_turnover:.4f}"
            )
        
        return None
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.TURNOVER_CONSTRAINT

class RiskConstraintManager:
    """リスク制約管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.constraints = self._initialize_constraints()
        
    def _initialize_constraints(self) -> List[OptimizationConstraint]:
        """制約条件を初期化"""
        constraints = []
        
        # 制約設定から制約オブジェクトを作成
        constraint_configs = self.config.get('constraints', {})
        
        for constraint_name, constraint_config in constraint_configs.items():
            constraint_type = constraint_config.get('type')
            
            if constraint_type == 'weight':
                constraints.append(WeightConstraint(constraint_name, constraint_config))
            elif constraint_type == 'volatility':
                constraints.append(VolatilityConstraint(constraint_name, constraint_config))
            elif constraint_type == 'drawdown':
                constraints.append(DrawdownConstraint(constraint_name, constraint_config))
            elif constraint_type == 'correlation':
                constraints.append(CorrelationConstraint(constraint_name, constraint_config))
            elif constraint_type == 'concentration':
                constraints.append(ConcentrationConstraint(constraint_name, constraint_config))
            elif constraint_type == 'turnover':
                constraints.append(TurnoverConstraint(constraint_name, constraint_config))
            else:
                self.logger.warning(f"Unknown constraint type: {constraint_type}")
        
        return constraints
    
    def check_all_constraints(
        self, 
        weights: Dict[str, float], 
        metrics: Dict[str, float], 
        **kwargs
    ) -> ConstraintResult:
        """すべての制約をチェック"""
        violations = []
        total_penalty = 0.0
        checked_constraints = []
        
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
                
            try:
                violation = constraint.check_constraint(weights, metrics, **kwargs)
                checked_constraints.append(constraint.name)
                
                if violation is not None:
                    violations.append(violation)
                    total_penalty += violation.penalty_score
                    
            except Exception as e:
                self.logger.error(f"Error checking constraint {constraint.name}: {e}")
        
        is_satisfied = len(violations) == 0
        
        return ConstraintResult(
            is_satisfied=is_satisfied,
            total_penalty=total_penalty,
            violations=violations,
            checked_constraints=checked_constraints
        )
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """制約の概要を取得"""
        summary = {
            'total_constraints': len(self.constraints),
            'enabled_constraints': sum(1 for c in self.constraints if c.enabled),
            'constraint_types': {},
            'severity_distribution': {}
        }
        
        for constraint in self.constraints:
            # 制約タイプ別の集計
            constraint_type = constraint.get_constraint_type().value
            if constraint_type not in summary['constraint_types']:
                summary['constraint_types'][constraint_type] = 0
            summary['constraint_types'][constraint_type] += 1
            
            # 深刻度別の集計
            severity = constraint.severity.value
            if severity not in summary['severity_distribution']:
                summary['severity_distribution'][severity] = 0
            summary['severity_distribution'][severity] += 1
        
        return summary
    
    def update_constraint_config(self, constraint_name: str, new_config: Dict[str, Any]):
        """制約設定を更新"""
        for constraint in self.constraints:
            if constraint.name == constraint_name:
                constraint.config.update(new_config)
                # 必要に応じて属性を更新
                if 'enabled' in new_config:
                    constraint.enabled = new_config['enabled']
                if 'penalty_multiplier' in new_config:
                    constraint.penalty_multiplier = new_config['penalty_multiplier']
                break
        else:
            self.logger.warning(f"Constraint {constraint_name} not found for update")
    
    def enable_constraint(self, constraint_name: str):
        """制約を有効化"""
        self.update_constraint_config(constraint_name, {'enabled': True})
    
    def disable_constraint(self, constraint_name: str):
        """制約を無効化"""
        self.update_constraint_config(constraint_name, {'enabled': False})

class AdaptiveConstraintAdjuster:
    """適応的制約調整器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def adjust_constraints_for_market_conditions(
        self,
        constraint_manager: RiskConstraintManager,
        market_volatility: float,
        trend_strength: float,
        market_regime: str = "normal"
    ) -> Dict[str, Any]:
        """市場環境に応じた制約調整"""
        adjustments = {}
        
        try:
            # ボラティリティ調整
            vol_multiplier = self._calculate_volatility_adjustment(market_volatility)
            
            # トレンド強度調整
            trend_multiplier = self._calculate_trend_adjustment(trend_strength)
            
            # 市場レジーム調整
            regime_multiplier = self._calculate_regime_adjustment(market_regime)
            
            # 総合調整
            total_multiplier = vol_multiplier * trend_multiplier * regime_multiplier
            
            # 制約別の調整
            for constraint in constraint_manager.constraints:
                if constraint.severity == ConstraintSeverity.ADAPTIVE:
                    if isinstance(constraint, VolatilityConstraint):
                        new_limit = constraint.max_portfolio_volatility * total_multiplier
                        adjustments[constraint.name] = {
                            'max_portfolio_volatility': min(new_limit, 0.4)  # 上限設定
                        }
                    elif isinstance(constraint, DrawdownConstraint):
                        new_limit = constraint.max_drawdown / total_multiplier
                        adjustments[constraint.name] = {
                            'max_drawdown': max(new_limit, 0.05)  # 下限設定
                        }
                        
        except Exception as e:
            self.logger.error(f"Error adjusting constraints: {e}")
        
        return adjustments
    
    def _calculate_volatility_adjustment(self, market_volatility: float) -> float:
        """ボラティリティに基づく調整係数"""
        baseline_vol = 0.2
        return 1.0 + (market_volatility - baseline_vol) * 0.5
    
    def _calculate_trend_adjustment(self, trend_strength: float) -> float:
        """トレンド強度に基づく調整係数"""
        return 1.0 + abs(trend_strength) * 0.3
    
    def _calculate_regime_adjustment(self, market_regime: str) -> float:
        """市場レジームに基づく調整係数"""
        regime_multipliers = {
            'normal': 1.0,
            'volatile': 1.3,
            'crisis': 1.5,
            'trending': 0.9,
            'sideways': 1.1
        }
        return regime_multipliers.get(market_regime, 1.0)


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Constraint Manager...")
    
    # テスト設定
    config = {
        'constraints': {
            'weight_constraint': {
                'type': 'weight',
                'severity': 'soft',
                'min_weight': 0.05,
                'max_weight': 0.6,
                'max_single_weight': 0.4,
                'penalty_multiplier': 10.0
            },
            'volatility_constraint': {
                'type': 'volatility',
                'severity': 'hard',
                'max_portfolio_volatility': 0.25,
                'penalty_multiplier': 50.0
            },
            'drawdown_constraint': {
                'type': 'drawdown',
                'severity': 'soft',
                'max_drawdown': 0.15,
                'penalty_multiplier': 20.0
            }
        }
    }
    
    # 制約マネージャーのテスト
    constraint_manager = RiskConstraintManager(config)
    
    # テスト重み
    test_weights = {
        'strategy1': 0.5,  # 制約違反
        'strategy2': 0.3,
        'strategy3': 0.2
    }
    
    # テスト指標
    test_metrics = {
        'portfolio_volatility': 0.30,  # 制約違反
        'max_drawdown': 0.10
    }
    
    # 制約チェック
    result = constraint_manager.check_all_constraints(test_weights, test_metrics)
    
    logger.info("Constraint Check Results:")
    logger.info(f"Is Satisfied: {result.is_satisfied}")
    logger.info(f"Total Penalty: {result.total_penalty:.4f}")
    logger.info(f"Number of Violations: {len(result.violations)}")
    
    for violation in result.violations:
        logger.info(f"Violation: {violation.constraint_name} - {violation.description}")
    
    # 制約概要
    summary = constraint_manager.get_constraint_summary()
    logger.info(f"Constraint Summary: {summary}")
    
    logger.info("Constraint Manager test completed successfully!")
