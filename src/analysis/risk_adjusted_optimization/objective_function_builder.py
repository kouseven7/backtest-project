"""
Module: Objective Function Builder
File: objective_function_builder.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  複合最適化目的関数の構築と管理システム

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
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 既存モジュールのインポート
try:
    from metrics.performance_metrics import (
        calculate_sharpe_ratio, calculate_sortino_ratio, 
        calculate_expectancy, calculate_max_drawdown_during_losses
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error in objective_function_builder: {e}")
    # フォールバック実装
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / trading_days)
        return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / trading_days)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        downside_std = np.sqrt((downside_returns ** 2).mean())
        return np.sqrt(trading_days) * excess_returns.mean() / downside_std
    
    def calculate_expectancy(trade_results):
        if len(trade_results) == 0:
            return 0.0
        return trade_results['pnl'].mean()
    
    def calculate_max_drawdown_during_losses(trade_results):
        if len(trade_results) == 0:
            return 0.0
        cumulative_pnl = trade_results['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdowns = (cumulative_pnl - running_max) / running_max.abs()
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """最適化目的"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_SORTINO = "maximize_sortino"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MAXIMIZE_INFORMATION_RATIO = "maximize_information_ratio"
    MINIMIZE_VAR = "minimize_var"
    MAXIMIZE_EXPECTED_RETURN = "maximize_expected_return"
    COMPOSITE_OPTIMIZATION = "composite_optimization"  # 複合最適化

@dataclass
class ObjectiveScore:
    """目的関数スコア"""
    objective_type: OptimizationObjective
    raw_score: float
    normalized_score: float
    weight: float
    weighted_score: float
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CompositeScoreResult:
    """複合スコア結果"""
    composite_score: float
    individual_scores: Dict[str, ObjectiveScore]
    optimization_direction: str  # "maximize" or "minimize"
    total_weight: float
    score_breakdown: Dict[str, float]
    confidence_level: float
    calculation_timestamp: datetime = field(default_factory=datetime.now)

class BaseObjectiveFunction(ABC):
    """目的関数基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        """目的関数値を計算"""
        pass
    
    @abstractmethod
    def get_optimization_direction(self) -> str:
        """最適化方向を取得 ('maximize' or 'minimize')"""
        pass

class SharpeRatioObjective(BaseObjectiveFunction):
    """シャープレシオ最適化目的関数"""
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """シャープレシオを計算"""
        return calculate_sharpe_ratio(returns, risk_free_rate)
    
    def get_optimization_direction(self) -> str:
        return "maximize"

class SortinoRatioObjective(BaseObjectiveFunction):
    """ソルティノレシオ最適化目的関数"""
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """ソルティノレシオを計算"""
        return calculate_sortino_ratio(returns, risk_free_rate)
    
    def get_optimization_direction(self) -> str:
        return "maximize"

class CalmarRatioObjective(BaseObjectiveFunction):
    """カルマーレシオ最適化目的関数"""
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        """カルマーレシオを計算"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_drawdown = self._calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """最大ドローダウンを計算"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    def get_optimization_direction(self) -> str:
        return "maximize"

class DrawdownMinimizationObjective(BaseObjectiveFunction):
    """ドローダウン最小化目的関数"""
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        """最大ドローダウンを計算"""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    def get_optimization_direction(self) -> str:
        return "minimize"

class ValueAtRiskObjective(BaseObjectiveFunction):
    """Value at Risk最小化目的関数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.confidence_level = config.get('var_confidence_level', 0.05)
        
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        """VaRを計算"""
        if len(returns) == 0:
            return 0.0
        
        var_value = np.percentile(returns, self.confidence_level * 100)
        return abs(var_value)
    
    def get_optimization_direction(self) -> str:
        return "minimize"

class CompositeObjectiveFunction:
    """複合目的関数"""
    
    def __init__(self, weights: Dict[str, float], config: Optional[Dict[str, Any]] = None):
        self.weights = weights
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 目的関数のインスタンス化
        self.objective_functions = self._initialize_objective_functions()
        
        # 正規化のためのスコア範囲
        self.score_ranges = self.config.get('score_ranges', {
            'sharpe': {'min': -2.0, 'max': 3.0},
            'sortino': {'min': -2.0, 'max': 4.0},
            'calmar': {'min': -1.0, 'max': 2.0},
            'drawdown': {'min': 0.0, 'max': 0.5},
            'var': {'min': 0.0, 'max': 0.1}
        })
        
    def _initialize_objective_functions(self) -> Dict[str, BaseObjectiveFunction]:
        """目的関数を初期化"""
        functions = {}
        
        if 'sharpe' in self.weights:
            functions['sharpe'] = SharpeRatioObjective(self.config)
        if 'sortino' in self.weights:
            functions['sortino'] = SortinoRatioObjective(self.config)
        if 'calmar' in self.weights:
            functions['calmar'] = CalmarRatioObjective(self.config)
        if 'drawdown' in self.weights:
            functions['drawdown'] = DrawdownMinimizationObjective(self.config)
        if 'var' in self.weights:
            functions['var'] = ValueAtRiskObjective(self.config)
        
        return functions
    
    def calculate(self, returns: pd.Series, benchmark_return: float = 0.02, **kwargs) -> CompositeScoreResult:
        """複合スコアを計算"""
        try:
            individual_scores = {}
            score_breakdown = {}
            total_weight = sum(self.weights.values())
            
            # 各目的関数のスコアを計算
            for name, objective_function in self.objective_functions.items():
                if name in self.weights:
                    raw_score = objective_function.calculate(returns, risk_free_rate=benchmark_return, **kwargs)
                    normalized_score = self._normalize_score(raw_score, name, objective_function.get_optimization_direction())
                    weight = self.weights[name]
                    weighted_score = normalized_score * weight
                    
                    individual_scores[name] = ObjectiveScore(
                        objective_type=OptimizationObjective(f"maximize_{name}"),
                        raw_score=raw_score,
                        normalized_score=normalized_score,
                        weight=weight,
                        weighted_score=weighted_score,
                        confidence=self._calculate_confidence(raw_score, name)
                    )
                    
                    score_breakdown[name] = weighted_score
            
            # 複合スコアの計算
            composite_score = sum(score_breakdown.values()) / total_weight if total_weight > 0 else 0.0
            
            # 信頼度レベルの計算
            confidence_level = np.mean([score.confidence for score in individual_scores.values()])
            
            return CompositeScoreResult(
                composite_score=composite_score,
                individual_scores=individual_scores,
                optimization_direction="maximize",
                total_weight=total_weight,
                score_breakdown=score_breakdown,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return CompositeScoreResult(
                composite_score=0.0,
                individual_scores={},
                optimization_direction="maximize",
                total_weight=0.0,
                score_breakdown={},
                confidence_level=0.0
            )
    
    def _normalize_score(self, raw_score: float, metric_name: str, direction: str) -> float:
        """スコアを正規化"""
        if metric_name not in self.score_ranges:
            return raw_score
        
        min_val = self.score_ranges[metric_name]['min']
        max_val = self.score_ranges[metric_name]['max']
        
        # スコアをクリップ
        clipped_score = np.clip(raw_score, min_val, max_val)
        
        # 0-1範囲に正規化
        if max_val != min_val:
            normalized = (clipped_score - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        
        # 最小化目的関数の場合は反転
        if direction == "minimize":
            normalized = 1.0 - normalized
        
        return normalized
    
    def _calculate_confidence(self, raw_score: float, metric_name: str) -> float:
        """スコアの信頼度を計算"""
        if metric_name not in self.score_ranges:
            return 1.0
        
        min_val = self.score_ranges[metric_name]['min']
        max_val = self.score_ranges[metric_name]['max']
        mid_val = (min_val + max_val) / 2
        
        # 中央値からの距離に基づく信頼度
        distance_from_center = abs(raw_score - mid_val) / ((max_val - min_val) / 2)
        confidence = min(1.0, max(0.1, 1.0 - distance_from_center * 0.5))
        
        return confidence

    def calculate_portfolio_returns(self, weights: Dict[str, float], strategy_returns: pd.DataFrame) -> pd.Series:
        """ポートフォリオリターンを計算"""
        try:
            # 重み配列の作成
            weight_vector = []
            strategy_columns = []
            
            for strategy in strategy_returns.columns:
                if strategy in weights:
                    weight_vector.append(weights[strategy])
                    strategy_columns.append(strategy)
                else:
                    weight_vector.append(0.0)
                    strategy_columns.append(strategy)
            
            weight_vector = np.array(weight_vector)
            weight_vector = weight_vector / weight_vector.sum() if weight_vector.sum() > 0 else weight_vector
            
            # ポートフォリオリターンの計算
            portfolio_returns = (strategy_returns[strategy_columns] * weight_vector).sum(axis=1)
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)

class ObjectiveFunctionBuilder:
    """目的関数構築器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _load_config(self) -> Dict[str, Any]:
        """設定をロード"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # デフォルト設定
        return {
            'default_weights': {
                'sharpe': 0.4,
                'sortino': 0.3,
                'calmar': 0.2,
                'drawdown': 0.1
            },
            'score_ranges': {
                'sharpe': {'min': -2.0, 'max': 3.0},
                'sortino': {'min': -2.0, 'max': 4.0},
                'calmar': {'min': -1.0, 'max': 2.0},
                'drawdown': {'min': 0.0, 'max': 0.5}
            }
        }
    
    def build_composite_objective(self, weights: Optional[Dict[str, float]] = None) -> CompositeObjectiveFunction:
        """複合目的関数を構築"""
        if weights is None:
            weights = self.config.get('default_weights', {
                'sharpe': 0.4,
                'sortino': 0.3,
                'calmar': 0.2,
                'drawdown': 0.1
            })
        
        return CompositeObjectiveFunction(weights, self.config)
    
    def build_single_objective(self, objective_type: OptimizationObjective) -> BaseObjectiveFunction:
        """単一目的関数を構築"""
        if objective_type == OptimizationObjective.MAXIMIZE_SHARPE:
            return SharpeRatioObjective(self.config)
        elif objective_type == OptimizationObjective.MAXIMIZE_SORTINO:
            return SortinoRatioObjective(self.config)
        elif objective_type == OptimizationObjective.MAXIMIZE_CALMAR:
            return CalmarRatioObjective(self.config)
        elif objective_type == OptimizationObjective.MINIMIZE_DRAWDOWN:
            return DrawdownMinimizationObjective(self.config)
        elif objective_type == OptimizationObjective.MINIMIZE_VAR:
            return ValueAtRiskObjective(self.config)
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Objective Function Builder...")
    
    # テストデータの生成
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    test_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    # 目的関数構築器のテスト
    builder = ObjectiveFunctionBuilder()
    
    # 複合目的関数のテスト
    composite_obj = builder.build_composite_objective()
    result = composite_obj.calculate(test_returns)
    
    logger.info("Composite Objective Function Test Results:")
    logger.info(f"Composite Score: {result.composite_score:.4f}")
    logger.info(f"Confidence Level: {result.confidence_level:.4f}")
    
    for name, score in result.individual_scores.items():
        logger.info(f"{name}: raw={score.raw_score:.4f}, normalized={score.normalized_score:.4f}, weighted={score.weighted_score:.4f}")
    
    logger.info("Objective Function Builder test completed successfully!")
