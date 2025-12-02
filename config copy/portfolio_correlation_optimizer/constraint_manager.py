"""
5-3-3 戦略間相関を考慮した配分最適化 - 制約管理システム

多様な制約条件を管理・適用するシステム

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

class ConstraintType(Enum):
    """制約種類"""
    WEIGHT_BOUNDS = "weight_bounds"
    SUM_TO_ONE = "sum_to_one"
    TURNOVER_LIMIT = "turnover_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    CORRELATION_PENALTY = "correlation_penalty"
    SECTOR_EXPOSURE = "sector_exposure"
    RISK_BUDGET = "risk_budget"
    TRACKING_ERROR = "tracking_error"
    DRAWDOWN_LIMIT = "drawdown_limit"

class ConstraintSeverity(Enum):
    """制約深刻度"""
    CRITICAL = "critical"    # 必須制約
    WARNING = "warning"      # 警告レベル
    SOFT = "soft"           # ソフト制約

@dataclass
class ConstraintViolation:
    """制約違反情報"""
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    current_value: float
    limit_value: float
    violation_amount: float
    strategy_names: List[str] = field(default_factory=list)
    description: str = ""
    
    @property
    def violation_ratio(self) -> float:
        """違反率"""
        if self.limit_value != 0:
            return abs(self.violation_amount / self.limit_value)
        return 0.0

@dataclass
class ConstraintResult:
    """制約検証結果"""
    is_feasible: bool
    violations: List[ConstraintViolation]
    adjusted_weights: Optional[Dict[str, float]] = None
    constraint_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_violations(self) -> List[ConstraintViolation]:
        """重大違反のみ"""
        return [v for v in self.violations if v.severity == ConstraintSeverity.CRITICAL]
    
    @property
    def has_critical_violations(self) -> bool:
        """重大違反があるか"""
        return len(self.critical_violations) > 0

@dataclass
class ConstraintConfig:
    """制約設定"""
    # 重み制約
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # ターンオーバー制約
    max_turnover: Optional[float] = None
    turnover_penalty_factor: float = 1.0
    
    # 集中度制約
    max_single_weight: Optional[float] = None
    max_top_n_concentration: Optional[Dict[int, float]] = None  # {n: max_weight}
    
    # 相関制約
    max_correlation_exposure: Optional[float] = None
    correlation_penalty_threshold: float = 0.8
    correlation_penalty_factor: float = 2.0
    
    # リスク制約
    max_portfolio_volatility: Optional[float] = None
    max_var: Optional[float] = None
    max_expected_drawdown: Optional[float] = None
    
    # セクター制約
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # {sector: (min, max)}
    
    # ソフト制約設定
    soft_constraint_penalty: float = 1000.0
    constraint_relaxation_factor: float = 1.1

class CorrelationConstraintManager:
    """相関制約管理システム"""
    
    def __init__(self, config: ConstraintConfig, logger: Optional[logging.Logger] = None):
        """
        初期化
        
        Args:
            config: 制約設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # 制約チェッカーの登録
        self.constraint_checkers = {
            ConstraintType.WEIGHT_BOUNDS: self._check_weight_bounds,
            ConstraintType.SUM_TO_ONE: self._check_sum_to_one,
            ConstraintType.TURNOVER_LIMIT: self._check_turnover_limit,
            ConstraintType.CONCENTRATION_LIMIT: self._check_concentration_limit,
            ConstraintType.CORRELATION_PENALTY: self._check_correlation_penalty,
            ConstraintType.RISK_BUDGET: self._check_risk_budget
        }
        
        # 制約調整器の登録
        self.constraint_adjusters = {
            ConstraintType.WEIGHT_BOUNDS: self._adjust_weight_bounds,
            ConstraintType.SUM_TO_ONE: self._adjust_sum_to_one,
            ConstraintType.TURNOVER_LIMIT: self._adjust_turnover_limit,
            ConstraintType.CONCENTRATION_LIMIT: self._adjust_concentration_limit
        }
        
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_constraints(
        self,
        weights: Dict[str, float],
        strategy_returns: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> ConstraintResult:
        """
        制約検証
        
        Args:
            weights: 検証対象重み
            strategy_returns: 戦略リターンデータ
            current_weights: 現在の重み
            correlation_matrix: 相関行列
            covariance_matrix: 共分散行列
            
        Returns:
            制約検証結果
        """
        
        try:
            violations = []
            
            # 各制約チェック実行
            for constraint_type, checker in self.constraint_checkers.items():
                try:
                    constraint_violations = checker(
                        weights=weights,
                        strategy_returns=strategy_returns,
                        current_weights=current_weights,
                        correlation_matrix=correlation_matrix,
                        covariance_matrix=covariance_matrix
                    )
                    violations.extend(constraint_violations)
                    
                except Exception as e:
                    self.logger.warning(f"Constraint check failed for {constraint_type}: {e}")
            
            # 重大違反判定
            is_feasible = not any(v.severity == ConstraintSeverity.CRITICAL for v in violations)
            
            # メタデータ作成
            metadata = {
                'total_violations': len(violations),
                'critical_violations': len([v for v in violations if v.severity == ConstraintSeverity.CRITICAL]),
                'warning_violations': len([v for v in violations if v.severity == ConstraintSeverity.WARNING]),
                'soft_violations': len([v for v in violations if v.severity == ConstraintSeverity.SOFT])
            }
            
            return ConstraintResult(
                is_feasible=is_feasible,
                violations=violations,
                constraint_metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {e}")
            return ConstraintResult(
                is_feasible=False,
                violations=[],
                constraint_metadata={'error': str(e)}
            )
    
    def adjust_weights_for_constraints(
        self,
        weights: Dict[str, float],
        constraint_result: ConstraintResult,
        strategy_returns: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        制約違反を解消するための重み調整
        
        Args:
            weights: 調整対象重み
            constraint_result: 制約検証結果
            strategy_returns: 戦略リターンデータ
            current_weights: 現在の重み
            
        Returns:
            調整後重み
        """
        
        if constraint_result.is_feasible:
            return weights.copy()
        
        try:
            adjusted_weights = weights.copy()
            
            # 重大違反から優先的に対処
            critical_violations = constraint_result.critical_violations
            
            for violation in critical_violations:
                constraint_type = violation.constraint_type
                
                if constraint_type in self.constraint_adjusters:
                    try:
                        adjusted_weights = self.constraint_adjusters[constraint_type](
                            weights=adjusted_weights,
                            violation=violation,
                            strategy_returns=strategy_returns,
                            current_weights=current_weights
                        )
                        
                        self.logger.info(f"Adjusted weights for constraint: {constraint_type}")
                        
                    except Exception as e:
                        self.logger.warning(f"Weight adjustment failed for {constraint_type}: {e}")
            
            # 最終正規化
            adjusted_weights = self._normalize_weights(adjusted_weights)
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Weight adjustment failed: {e}")
            return weights.copy()
    
    def _check_weight_bounds(
        self,
        weights: Dict[str, float],
        **kwargs
    ) -> List[ConstraintViolation]:
        """重み境界制約チェック"""
        
        violations = []
        
        for strategy, weight in weights.items():
            # 最小重み制約
            if weight < self.config.min_weight:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.WEIGHT_BOUNDS,
                    severity=ConstraintSeverity.CRITICAL,
                    current_value=weight,
                    limit_value=self.config.min_weight,
                    violation_amount=self.config.min_weight - weight,
                    strategy_names=[strategy],
                    description=f"Weight below minimum: {strategy}"
                ))
            
            # 最大重み制約
            if weight > self.config.max_weight:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.WEIGHT_BOUNDS,
                    severity=ConstraintSeverity.CRITICAL,
                    current_value=weight,
                    limit_value=self.config.max_weight,
                    violation_amount=weight - self.config.max_weight,
                    strategy_names=[strategy],
                    description=f"Weight above maximum: {strategy}"
                ))
        
        return violations
    
    def _check_sum_to_one(
        self,
        weights: Dict[str, float],
        **kwargs
    ) -> List[ConstraintViolation]:
        """重み合計制約チェック"""
        
        violations = []
        total_weight = sum(weights.values())
        
        # 許容誤差
        tolerance = 1e-4
        
        if abs(total_weight - 1.0) > tolerance:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.SUM_TO_ONE,
                severity=ConstraintSeverity.CRITICAL,
                current_value=total_weight,
                limit_value=1.0,
                violation_amount=abs(total_weight - 1.0),
                strategy_names=list(weights.keys()),
                description=f"Weight sum violation: {total_weight:.6f}"
            ))
        
        return violations
    
    def _check_turnover_limit(
        self,
        weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[ConstraintViolation]:
        """ターンオーバー制約チェック"""
        
        violations = []
        
        if self.config.max_turnover is None or current_weights is None:
            return violations
        
        # ターンオーバー計算
        turnover = 0.0
        for strategy in set(weights.keys()) | set(current_weights.keys()):
            current_w = current_weights.get(strategy, 0.0)
            new_w = weights.get(strategy, 0.0)
            turnover += abs(new_w - current_w)
        
        turnover /= 2.0  # 片道ターンオーバー
        
        if turnover > self.config.max_turnover:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.TURNOVER_LIMIT,
                severity=ConstraintSeverity.WARNING,  # 通常は警告レベル
                current_value=turnover,
                limit_value=self.config.max_turnover,
                violation_amount=turnover - self.config.max_turnover,
                strategy_names=list(weights.keys()),
                description=f"Turnover limit exceeded: {turnover:.4f}"
            ))
        
        return violations
    
    def _check_concentration_limit(
        self,
        weights: Dict[str, float],
        **kwargs
    ) -> List[ConstraintViolation]:
        """集中度制約チェック"""
        
        violations = []
        
        # 単一戦略集中度制約
        if self.config.max_single_weight is not None:
            max_weight = max(weights.values())
            if max_weight > self.config.max_single_weight:
                max_strategy = max(weights.items(), key=lambda x: x[1])[0]
                
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CONCENTRATION_LIMIT,
                    severity=ConstraintSeverity.WARNING,
                    current_value=max_weight,
                    limit_value=self.config.max_single_weight,
                    violation_amount=max_weight - self.config.max_single_weight,
                    strategy_names=[max_strategy],
                    description=f"Single strategy concentration: {max_strategy}"
                ))
        
        # 上位N戦略集中度制約
        if self.config.max_top_n_concentration is not None:
            sorted_weights = sorted(weights.values(), reverse=True)
            
            for n, max_concentration in self.config.max_top_n_concentration.items():
                if len(sorted_weights) >= n:
                    top_n_concentration = sum(sorted_weights[:n])
                    
                    if top_n_concentration > max_concentration:
                        # 上位N戦略を特定
                        sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        top_strategies = [s[0] for s in sorted_strategies[:n]]
                        
                        violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.CONCENTRATION_LIMIT,
                            severity=ConstraintSeverity.WARNING,
                            current_value=top_n_concentration,
                            limit_value=max_concentration,
                            violation_amount=top_n_concentration - max_concentration,
                            strategy_names=top_strategies,
                            description=f"Top {n} concentration: {top_n_concentration:.4f}"
                        ))
        
        return violations
    
    def _check_correlation_penalty(
        self,
        weights: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[ConstraintViolation]:
        """相関ペナルティ制約チェック"""
        
        violations = []
        
        if correlation_matrix is None or self.config.max_correlation_exposure is None:
            return violations
        
        try:
            # 重み配列作成
            strategy_names = list(weights.keys())
            weight_array = np.array([weights[name] for name in strategy_names])
            
            # ポートフォリオ相関露出計算
            # 定義：重み付き平均相関
            if len(weight_array) > 1:
                weight_matrix = np.outer(weight_array, weight_array)
                correlation_exposure = np.sum(
                    weight_matrix * correlation_matrix
                ) - np.sum(weight_array ** 2)  # 対角要素除去
                
                correlation_exposure /= (1.0 - np.sum(weight_array ** 2))  # 正規化
                
                if correlation_exposure > self.config.max_correlation_exposure:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.CORRELATION_PENALTY,
                        severity=ConstraintSeverity.SOFT,
                        current_value=correlation_exposure,
                        limit_value=self.config.max_correlation_exposure,
                        violation_amount=correlation_exposure - self.config.max_correlation_exposure,
                        strategy_names=strategy_names,
                        description=f"High correlation exposure: {correlation_exposure:.4f}"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Correlation penalty check failed: {e}")
        
        return violations
    
    def _check_risk_budget(
        self,
        weights: Dict[str, float],
        covariance_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[ConstraintViolation]:
        """リスク予算制約チェック"""
        
        violations = []
        
        if covariance_matrix is None:
            return violations
        
        try:
            # 重み配列作成
            strategy_names = list(weights.keys())
            weight_array = np.array([weights[name] for name in strategy_names])
            
            # ポートフォリオリスク計算
            portfolio_variance = np.dot(weight_array, np.dot(covariance_matrix, weight_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # ボラティリティ制約
            if self.config.max_portfolio_volatility is not None:
                if portfolio_volatility > self.config.max_portfolio_volatility:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.RISK_BUDGET,
                        severity=ConstraintSeverity.WARNING,
                        current_value=portfolio_volatility,
                        limit_value=self.config.max_portfolio_volatility,
                        violation_amount=portfolio_volatility - self.config.max_portfolio_volatility,
                        strategy_names=strategy_names,
                        description=f"Portfolio volatility: {portfolio_volatility:.4f}"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Risk budget check failed: {e}")
        
        return violations
    
    def _adjust_weight_bounds(
        self,
        weights: Dict[str, float],
        violation: ConstraintViolation,
        **kwargs
    ) -> Dict[str, float]:
        """重み境界制約調整"""
        
        adjusted_weights = weights.copy()
        
        for strategy in violation.strategy_names:
            current_weight = adjusted_weights.get(strategy, 0.0)
            
            # 制約適用
            adjusted_weight = max(self.config.min_weight, 
                                min(self.config.max_weight, current_weight))
            adjusted_weights[strategy] = adjusted_weight
        
        return self._normalize_weights(adjusted_weights)
    
    def _adjust_sum_to_one(
        self,
        weights: Dict[str, float],
        violation: ConstraintViolation,
        **kwargs
    ) -> Dict[str, float]:
        """重み合計制約調整"""
        
        return self._normalize_weights(weights)
    
    def _adjust_turnover_limit(
        self,
        weights: Dict[str, float],
        violation: ConstraintViolation,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """ターンオーバー制約調整"""
        
        if current_weights is None:
            return weights.copy()
        
        adjusted_weights = weights.copy()
        
        try:
            # 現在のターンオーバー計算
            current_turnover = 0.0
            for strategy in set(weights.keys()) | set(current_weights.keys()):
                current_w = current_weights.get(strategy, 0.0)
                new_w = weights.get(strategy, 0.0)
                current_turnover += abs(new_w - current_w)
            current_turnover /= 2.0
            
            # 調整係数計算
            if current_turnover > self.config.max_turnover:
                adjustment_factor = self.config.max_turnover / current_turnover
                
                # 現在重みに向けて調整
                for strategy in adjusted_weights.keys():
                    current_w = current_weights.get(strategy, 0.0)
                    target_w = weights[strategy]
                    
                    adjusted_w = current_w + (target_w - current_w) * adjustment_factor
                    adjusted_weights[strategy] = adjusted_w
        
        except Exception as e:
            self.logger.warning(f"Turnover adjustment failed: {e}")
        
        return self._normalize_weights(adjusted_weights)
    
    def _adjust_concentration_limit(
        self,
        weights: Dict[str, float],
        violation: ConstraintViolation,
        **kwargs
    ) -> Dict[str, float]:
        """集中度制約調整"""
        
        adjusted_weights = weights.copy()
        
        try:
            if violation.constraint_type == ConstraintType.CONCENTRATION_LIMIT:
                # 上位戦略の重みを制限
                for strategy in violation.strategy_names:
                    if strategy in adjusted_weights:
                        current_weight = adjusted_weights[strategy]
                        
                        # 単一戦略制約
                        if self.config.max_single_weight is not None:
                            adjusted_weights[strategy] = min(
                                current_weight, self.config.max_single_weight
                            )
        
        except Exception as e:
            self.logger.warning(f"Concentration adjustment failed: {e}")
        
        return self._normalize_weights(adjusted_weights)
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重み正規化"""
        
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            # 等重み
            n = len(weights)
            return {strategy: 1.0 / n for strategy in weights.keys()}
        
        return {strategy: weight / total_weight 
                for strategy, weight in weights.items()}
    
    def get_constraint_summary(self, constraint_result: ConstraintResult) -> str:
        """制約検証結果サマリー"""
        
        lines = []
        lines.append("=== Constraint Validation Summary ===")
        lines.append(f"Feasible: {'Yes' if constraint_result.is_feasible else 'No'}")
        lines.append(f"Total Violations: {len(constraint_result.violations)}")
        
        if constraint_result.violations:
            lines.append("")
            lines.append("Violations by Severity:")
            
            for severity in ConstraintSeverity:
                violations = [v for v in constraint_result.violations if v.severity == severity]
                if violations:
                    lines.append(f"  {severity.value}: {len(violations)}")
                    
                    for violation in violations[:3]:  # 最大3つまで表示
                        lines.append(f"    - {violation.description}")
                        lines.append(f"      Current: {violation.current_value:.4f}, " +
                                    f"Limit: {violation.limit_value:.4f}")
                    
                    if len(violations) > 3:
                        lines.append(f"    ... and {len(violations) - 3} more")
        
        # メタデータ
        if constraint_result.constraint_metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in constraint_result.constraint_metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def calculate_constraint_penalty(
        self,
        weights: Dict[str, float],
        constraint_result: ConstraintResult
    ) -> float:
        """制約違反ペナルティ計算"""
        
        total_penalty = 0.0
        
        for violation in constraint_result.violations:
            # 深刻度別ペナルティ重み
            severity_weights = {
                ConstraintSeverity.CRITICAL: 1000.0,
                ConstraintSeverity.WARNING: 100.0,
                ConstraintSeverity.SOFT: 10.0
            }
            
            severity_weight = severity_weights.get(violation.severity, 1.0)
            violation_penalty = severity_weight * violation.violation_ratio
            
            total_penalty += violation_penalty
        
        return total_penalty
    
    def update_config(self, new_config: ConstraintConfig):
        """設定更新"""
        self.config = new_config
        self.logger.info("Constraint configuration updated")
