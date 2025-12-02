"""
重み制約管理システム

階層的重みとメタパラメータに対する制約の定義と管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class WeightConstraint:
    """重み制約の定義"""
    name: str
    min_value: float
    max_value: float
    constraint_type: str  # 'hard', 'soft', 'penalty'
    penalty_weight: float = 1.0
    description: str = ""
    
@dataclass
class ConstraintViolation:
    """制約違反の記録"""
    constraint_name: str
    parameter_name: str
    current_value: float
    allowed_range: Tuple[float, float]
    violation_severity: float
    timestamp: datetime

class WeightConstraintManager:
    """
    重み制約管理システム
    
    階層的重み構造に対する制約を管理し、
    最適化プロセスでの制約違反を検出・修正する。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 制約設定ファイルのパス
        """
        self.logger = self._setup_logger()
        
        # 制約定義
        self.strategy_constraints = {}
        self.portfolio_constraints = {}
        self.meta_constraints = {}
        
        # 違反履歴
        self.violation_history = []
        
        # 制約の初期化
        self._initialize_default_constraints()
        
        # 設定ファイルからの読み込み
        if config_path:
            self._load_constraints_from_config(config_path)
            
        self.logger.info("WeightConstraintManager initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.WeightConstraintManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _initialize_default_constraints(self) -> None:
        """デフォルト制約の初期化"""
        # ストラテジー重み制約
        self.strategy_constraints = {
            'trend_following': WeightConstraint(
                name='trend_following',
                min_value=0.1,
                max_value=0.4,
                constraint_type='hard',
                description='Trend following strategy weight'
            ),
            'mean_reversion': WeightConstraint(
                name='mean_reversion',
                min_value=0.1,
                max_value=0.4,
                constraint_type='hard',
                description='Mean reversion strategy weight'
            ),
            'momentum': WeightConstraint(
                name='momentum',
                min_value=0.05,
                max_value=0.3,
                constraint_type='hard',
                description='Momentum strategy weight'
            ),
            'volatility_breakout': WeightConstraint(
                name='volatility_breakout',
                min_value=0.05,
                max_value=0.3,
                constraint_type='hard',
                description='Volatility breakout strategy weight'
            )
        }
        
        # ポートフォリオ重み制約
        self.portfolio_constraints = {
            'max_single_asset': WeightConstraint(
                name='max_single_asset',
                min_value=0.0,
                max_value=0.3,  # 最大30%
                constraint_type='hard',
                description='Maximum weight for single asset'
            ),
            'min_diversification': WeightConstraint(
                name='min_diversification',
                min_value=0.01,  # 最小1%
                max_value=1.0,
                constraint_type='hard',
                description='Minimum weight for diversification'
            ),
            'sector_concentration': WeightConstraint(
                name='sector_concentration',
                min_value=0.0,
                max_value=0.5,  # セクター最大50%
                constraint_type='soft',
                penalty_weight=2.0,
                description='Maximum sector concentration'
            )
        }
        
        # メタパラメータ制約
        self.meta_constraints = {
            'learning_rate': WeightConstraint(
                name='learning_rate',
                min_value=0.1,
                max_value=3.0,
                constraint_type='hard',
                description='Learning rate multiplier'
            ),
            'volatility_scaling': WeightConstraint(
                name='volatility_scaling',
                min_value=0.5,
                max_value=2.5,
                constraint_type='hard',
                description='Volatility scaling factor'
            ),
            'risk_aversion': WeightConstraint(
                name='risk_aversion',
                min_value=0.5,
                max_value=3.0,
                constraint_type='hard',
                description='Risk aversion parameter'
            ),
            'rebalancing_threshold': WeightConstraint(
                name='rebalancing_threshold',
                min_value=0.01,
                max_value=0.1,
                constraint_type='soft',
                penalty_weight=1.5,
                description='Rebalancing threshold'
            )
        }
        
    def _load_constraints_from_config(self, config_path: str) -> None:
        """設定ファイルから制約を読み込み"""
        # 実装省略（JSONまたはYAMLファイルからの読み込み）
        self.logger.info(f"Constraint configuration loaded from {config_path}")
        
    def validate_weights(
        self,
        weights: Dict[str, float],
        weight_type: str = "all"
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        重みの制約検証
        
        Args:
            weights: 検証する重み
            weight_type: 検証対象 ('strategy', 'portfolio', 'meta', 'all')
            
        Returns:
            (制約満足フラグ, 制約違反リスト)
        """
        violations = []
        
        if weight_type in ["strategy", "all"]:
            strategy_violations = self._validate_strategy_weights(weights)
            violations.extend(strategy_violations)
            
        if weight_type in ["portfolio", "all"]:
            portfolio_violations = self._validate_portfolio_weights(weights)
            violations.extend(portfolio_violations)
            
        if weight_type in ["meta", "all"]:
            meta_violations = self._validate_meta_parameters(weights)
            violations.extend(meta_violations)
            
        # 階層的制約の検証
        hierarchical_violations = self._validate_hierarchical_constraints(weights)
        violations.extend(hierarchical_violations)
        
        # 違反履歴に記録
        self.violation_history.extend(violations)
        
        is_valid = len(violations) == 0
        
        if violations:
            self.logger.warning(f"Found {len(violations)} constraint violations")
        
        return is_valid, violations
        
    def _validate_strategy_weights(
        self,
        weights: Dict[str, float]
    ) -> List[ConstraintViolation]:
        """ストラテジー重み制約の検証"""
        violations = []
        
        strategy_weights = {
            k.replace('strategy_', ''): v 
            for k, v in weights.items() 
            if k.startswith('strategy_')
        }
        
        for param_name, value in strategy_weights.items():
            if param_name in self.strategy_constraints:
                constraint = self.strategy_constraints[param_name]
                
                if value < constraint.min_value or value > constraint.max_value:
                    violation_severity = self._calculate_violation_severity(
                        value, constraint.min_value, constraint.max_value
                    )
                    
                    violations.append(ConstraintViolation(
                        constraint_name=constraint.name,
                        parameter_name=f'strategy_{param_name}',
                        current_value=value,
                        allowed_range=(constraint.min_value, constraint.max_value),
                        violation_severity=violation_severity,
                        timestamp=datetime.now()
                    ))
                    
        # ストラテジー重みの合計制約
        strategy_sum = sum(strategy_weights.values())
        if abs(strategy_sum - 1.0) > 0.01:  # 1%の許容誤差
            violations.append(ConstraintViolation(
                constraint_name="strategy_sum",
                parameter_name="strategy_total",
                current_value=strategy_sum,
                allowed_range=(0.99, 1.01),
                violation_severity=abs(strategy_sum - 1.0),
                timestamp=datetime.now()
            ))
            
        return violations
        
    def _validate_portfolio_weights(
        self,
        weights: Dict[str, float]
    ) -> List[ConstraintViolation]:
        """ポートフォリオ重み制約の検証"""
        violations = []
        
        portfolio_weights = {
            k.replace('portfolio_', ''): v 
            for k, v in weights.items() 
            if k.startswith('portfolio_')
        }
        
        if not portfolio_weights:
            return violations
            
        # 個別資産制約の検証
        for asset_name, weight in portfolio_weights.items():
            # 最大単一資産制約
            max_constraint = self.portfolio_constraints['max_single_asset']
            if weight > max_constraint.max_value:
                violations.append(ConstraintViolation(
                    constraint_name=max_constraint.name,
                    parameter_name=f'portfolio_{asset_name}',
                    current_value=weight,
                    allowed_range=(max_constraint.min_value, max_constraint.max_value),
                    violation_severity=weight - max_constraint.max_value,
                    timestamp=datetime.now()
                ))
                
            # 最小分散制約
            min_constraint = self.portfolio_constraints['min_diversification']
            if weight < min_constraint.min_value and weight > 0:
                violations.append(ConstraintViolation(
                    constraint_name=min_constraint.name,
                    parameter_name=f'portfolio_{asset_name}',
                    current_value=weight,
                    allowed_range=(min_constraint.min_value, min_constraint.max_value),
                    violation_severity=min_constraint.min_value - weight,
                    timestamp=datetime.now()
                ))
                
        # ポートフォリオ重みの合計制約
        portfolio_sum = sum(portfolio_weights.values())
        if abs(portfolio_sum - 1.0) > 0.01:
            violations.append(ConstraintViolation(
                constraint_name="portfolio_sum",
                parameter_name="portfolio_total",
                current_value=portfolio_sum,
                allowed_range=(0.99, 1.01),
                violation_severity=abs(portfolio_sum - 1.0),
                timestamp=datetime.now()
            ))
            
        return violations
        
    def _validate_meta_parameters(
        self,
        weights: Dict[str, float]
    ) -> List[ConstraintViolation]:
        """メタパラメータ制約の検証"""
        violations = []
        
        meta_params = {
            k.replace('meta_', ''): v 
            for k, v in weights.items() 
            if k.startswith('meta_')
        }
        
        for param_name, value in meta_params.items():
            if param_name in self.meta_constraints:
                constraint = self.meta_constraints[param_name]
                
                if value < constraint.min_value or value > constraint.max_value:
                    violation_severity = self._calculate_violation_severity(
                        value, constraint.min_value, constraint.max_value
                    )
                    
                    violations.append(ConstraintViolation(
                        constraint_name=constraint.name,
                        parameter_name=f'meta_{param_name}',
                        current_value=value,
                        allowed_range=(constraint.min_value, constraint.max_value),
                        violation_severity=violation_severity,
                        timestamp=datetime.now()
                    ))
                    
        return violations
        
    def _validate_hierarchical_constraints(
        self,
        weights: Dict[str, float]
    ) -> List[ConstraintViolation]:
        """階層的制約の検証"""
        violations = []
        
        # ストラテジーとポートフォリオの整合性チェック
        strategy_weights = {
            k.replace('strategy_', ''): v 
            for k, v in weights.items() 
            if k.startswith('strategy_')
        }
        
        portfolio_weights = {
            k.replace('portfolio_', ''): v 
            for k, v in weights.items() 
            if k.startswith('portfolio_')
        }
        
        # リスクレベルの整合性
        if 'trend_following' in strategy_weights and len(portfolio_weights) > 0:
            tf_weight = strategy_weights['trend_following']
            max_portfolio_weight = max(portfolio_weights.values()) if portfolio_weights else 0
            
            # トレンドフォロー戦略が高い場合、ポートフォリオ集中度も制限
            if tf_weight > 0.3 and max_portfolio_weight > 0.4:
                violations.append(ConstraintViolation(
                    constraint_name="risk_consistency",
                    parameter_name="trend_portfolio_consistency",
                    current_value=tf_weight * max_portfolio_weight,
                    allowed_range=(0.0, 0.12),
                    violation_severity=(tf_weight * max_portfolio_weight) - 0.12,
                    timestamp=datetime.now()
                ))
                
        return violations
        
    def _calculate_violation_severity(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """制約違反の重要度計算"""
        if value < min_val:
            return (min_val - value) / (max_val - min_val)
        elif value > max_val:
            return (value - max_val) / (max_val - min_val)
        else:
            return 0.0
            
    def apply_constraint_corrections(
        self,
        weights: Dict[str, float],
        correction_method: str = "projection"
    ) -> Dict[str, float]:
        """
        制約違反の修正
        
        Args:
            weights: 修正対象の重み
            correction_method: 修正方法 ('projection', 'penalty', 'soft')
            
        Returns:
            修正された重み
        """
        corrected_weights = weights.copy()
        
        if correction_method == "projection":
            corrected_weights = self._apply_projection_correction(corrected_weights)
        elif correction_method == "penalty":
            corrected_weights = self._apply_penalty_correction(corrected_weights)
        elif correction_method == "soft":
            corrected_weights = self._apply_soft_correction(corrected_weights)
            
        # 最終検証
        is_valid, remaining_violations = self.validate_weights(corrected_weights)
        
        if not is_valid:
            self.logger.warning(f"Still {len(remaining_violations)} violations after correction")
            
        return corrected_weights
        
    def _apply_projection_correction(self, weights: Dict[str, float]) -> Dict[str, float]:
        """投影による制約修正"""
        corrected = weights.copy()
        
        # ハード制約の適用
        for key, value in corrected.items():
            if key.startswith('strategy_'):
                param_name = key.replace('strategy_', '')
                if param_name in self.strategy_constraints:
                    constraint = self.strategy_constraints[param_name]
                    corrected[key] = np.clip(value, constraint.min_value, constraint.max_value)
                    
            elif key.startswith('portfolio_'):
                param_name = key.replace('portfolio_', '')
                # 個別制約の適用
                max_constraint = self.portfolio_constraints['max_single_asset']
                min_constraint = self.portfolio_constraints['min_diversification']
                
                corrected[key] = np.clip(
                    value,
                    min_constraint.min_value if value > 0 else 0,
                    max_constraint.max_value
                )
                
            elif key.startswith('meta_'):
                param_name = key.replace('meta_', '')
                if param_name in self.meta_constraints:
                    constraint = self.meta_constraints[param_name]
                    corrected[key] = np.clip(value, constraint.min_value, constraint.max_value)
                    
        # 正規化制約の適用
        corrected = self._apply_normalization_constraints(corrected)
        
        return corrected
        
    def _apply_normalization_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """正規化制約の適用"""
        corrected = weights.copy()
        
        # ストラテジー重みの正規化
        strategy_keys = [k for k in weights if k.startswith('strategy_')]
        if strategy_keys:
            strategy_sum = sum(corrected[k] for k in strategy_keys)
            if strategy_sum > 0:
                for k in strategy_keys:
                    corrected[k] /= strategy_sum
                    
        # ポートフォリオ重みの正規化
        portfolio_keys = [k for k in weights if k.startswith('portfolio_')]
        if portfolio_keys:
            portfolio_sum = sum(corrected[k] for k in portfolio_keys)
            if portfolio_sum > 0:
                for k in portfolio_keys:
                    corrected[k] /= portfolio_sum
                    
        return corrected
        
    def _apply_penalty_correction(self, weights: Dict[str, float]) -> Dict[str, float]:
        """ペナルティによる制約修正"""
        # ペナルティ項を含む最適化による修正
        # 実装省略（より複雑な最適化が必要）
        return self._apply_projection_correction(weights)
        
    def _apply_soft_correction(self, weights: Dict[str, float]) -> Dict[str, float]:
        """ソフト制約による修正"""
        # ソフト制約を考慮した緩やかな修正
        corrected = weights.copy()
        
        for key, value in corrected.items():
            if key.startswith('strategy_'):
                param_name = key.replace('strategy_', '')
                if param_name in self.strategy_constraints:
                    constraint = self.strategy_constraints[param_name]
                    if constraint.constraint_type == 'soft':
                        # ソフト制約の場合は段階的修正
                        if value < constraint.min_value:
                            corrected[key] = value + (constraint.min_value - value) * 0.5
                        elif value > constraint.max_value:
                            corrected[key] = value - (value - constraint.max_value) * 0.5
                            
        return corrected
        
    def calculate_constraint_penalty(
        self,
        weights: Dict[str, float]
    ) -> float:
        """制約ペナルティの計算"""
        is_valid, violations = self.validate_weights(weights)
        
        if is_valid:
            return 0.0
            
        total_penalty = 0.0
        
        for violation in violations:
            # 制約タイプに応じたペナルティ
            constraint_type = self._get_constraint_type(violation.constraint_name)
            
            if constraint_type == 'hard':
                penalty = violation.violation_severity * 100  # 重いペナルティ
            elif constraint_type == 'soft':
                penalty = violation.violation_severity * 10   # 軽いペナルティ
            else:
                penalty = violation.violation_severity * 50   # 中程度のペナルティ
                
            total_penalty += penalty
            
        return total_penalty
        
    def _get_constraint_type(self, constraint_name: str) -> str:
        """制約タイプの取得"""
        # 各制約辞書から制約タイプを検索
        for constraints in [self.strategy_constraints, self.portfolio_constraints, self.meta_constraints]:
            if constraint_name in constraints:
                return constraints[constraint_name].constraint_type
        return 'penalty'
        
    def get_constraint_summary(self) -> Dict[str, Any]:
        """制約サマリーの取得"""
        return {
            'strategy_constraints': {
                name: {
                    'min_value': constraint.min_value,
                    'max_value': constraint.max_value,
                    'constraint_type': constraint.constraint_type,
                    'description': constraint.description
                }
                for name, constraint in self.strategy_constraints.items()
            },
            'portfolio_constraints': {
                name: {
                    'min_value': constraint.min_value,
                    'max_value': constraint.max_value,
                    'constraint_type': constraint.constraint_type,
                    'description': constraint.description
                }
                for name, constraint in self.portfolio_constraints.items()
            },
            'meta_constraints': {
                name: {
                    'min_value': constraint.min_value,
                    'max_value': constraint.max_value,
                    'constraint_type': constraint.constraint_type,
                    'description': constraint.description
                }
                for name, constraint in self.meta_constraints.items()
            },
            'total_violations': len(self.violation_history),
            'recent_violations': len([
                v for v in self.violation_history
                if (datetime.now() - v.timestamp).days <= 7
            ])
        }
        
    def export_violation_history(self, filepath: str) -> None:
        """制約違反履歴のエクスポート"""
        if not self.violation_history:
            self.logger.warning("No violation history to export")
            return
            
        records = []
        for violation in self.violation_history:
            records.append({
                'timestamp': violation.timestamp,
                'constraint_name': violation.constraint_name,
                'parameter_name': violation.parameter_name,
                'current_value': violation.current_value,
                'min_allowed': violation.allowed_range[0],
                'max_allowed': violation.allowed_range[1],
                'violation_severity': violation.violation_severity
            })
            
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Violation history exported to {filepath}")
