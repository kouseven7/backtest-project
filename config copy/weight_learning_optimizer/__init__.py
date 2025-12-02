"""
5-2-3 最適な重み付け比率の学習アルゴリズム (Optimal Weight Ratio Learning Algorithm)

ベイジアン最適化を使用した階層的重み学習システム
- Strategy score weights
- Portfolio weights  
- Meta parameters

統合された学習アルゴリズムによる期待値最大化とドローダウン最小化
"""

from .bayesian_weight_optimizer import BayesianWeightOptimizer
from .performance_evaluator import PerformanceEvaluator
from .adaptive_learning_scheduler import AdaptiveLearningScheduler
from .weight_constraint_manager import WeightConstraintManager
from .integration_bridge import IntegrationBridge
from .optimization_history_manager import OptimizationHistoryManager
from .meta_parameter_controller import MetaParameterController

__version__ = "1.0.0"
__author__ = "Weight Learning System"

__all__ = [
    'BayesianWeightOptimizer',
    'PerformanceEvaluator', 
    'AdaptiveLearningScheduler',
    'WeightConstraintManager',
    'IntegrationBridge',
    'OptimizationHistoryManager',
    'MetaParameterController'
]
