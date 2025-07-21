"""
5-1-3「リスク調整後リターンの最適化」パッケージ

このパッケージは、複数戦略ポートフォリオのリスク調整後リターンを
最大化するための包括的な最適化システムを提供します。

Author: imega
Created: 2025-07-21
"""

# メインクラスのインポート
try:
    from .objective_function_builder import (
        CompositeObjectiveFunction,
        ObjectiveFunctionBuilder,
        OptimizationObjective,
        CompositeScoreResult
    )
    
    from .constraint_manager import (
        RiskConstraintManager,
        ConstraintResult,
        ConstraintViolation,
        ConstraintType,
        ConstraintSeverity,
        AdaptiveConstraintAdjuster
    )
    
    from .optimization_algorithms import (
        OptimizationEngine,
        OptimizationMethod,
        OptimizationConfig,
        OptimizationResult,
        DifferentialEvolutionOptimizer,
        ScipyMinimizeOptimizer,
        GradientDescentOptimizer
    )
    
    from .performance_evaluator import (
        EnhancedPerformanceEvaluator,
        ComprehensivePerformanceReport,
        PerformanceMetric,
        MetricCategory
    )
    
    from .risk_return_optimizer import (
        RiskAdjustedOptimizationEngine,
        OptimizationContext,
        RiskAdjustedOptimizationResult
    )
    
    from .portfolio_optimizer import (
        AdvancedPortfolioOptimizer,
        PortfolioOptimizationProfile,
        MultiPeriodOptimizationRequest,
        PortfolioOptimizationResult
    )
    
    from .optimization_validator import (
        OptimizationValidator,
        ValidationTest,
        ValidationReport,
        BacktestValidationConfig
    )
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import some modules in risk_adjusted_optimization: {e}")
    
    # 最小限のダミークラスを提供
    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass
    
    # 失敗したインポートの場合はダミーで置換
    globals().update({
        'CompositeObjectiveFunction': DummyClass,
        'ObjectiveFunctionBuilder': DummyClass,
        'OptimizationObjective': DummyClass,
        'CompositeScoreResult': DummyClass,
        'RiskConstraintManager': DummyClass,
        'ConstraintResult': DummyClass,
        'ConstraintViolation': DummyClass,
        'ConstraintType': DummyClass,
        'ConstraintSeverity': DummyClass,
        'AdaptiveConstraintAdjuster': DummyClass,
        'OptimizationEngine': DummyClass,
        'OptimizationMethod': DummyClass,
        'OptimizationConfig': DummyClass,
        'OptimizationResult': DummyClass,
        'DifferentialEvolutionOptimizer': DummyClass,
        'ScipyMinimizeOptimizer': DummyClass,
        'GradientDescentOptimizer': DummyClass,
        'EnhancedPerformanceEvaluator': DummyClass,
        'ComprehensivePerformanceReport': DummyClass,
        'PerformanceMetric': DummyClass,
        'MetricCategory': DummyClass,
        'RiskAdjustedOptimizationEngine': DummyClass,
        'OptimizationContext': DummyClass,
        'RiskAdjustedOptimizationResult': DummyClass,
        'AdvancedPortfolioOptimizer': DummyClass,
        'PortfolioOptimizationProfile': DummyClass,
        'MultiPeriodOptimizationRequest': DummyClass,
        'PortfolioOptimizationResult': DummyClass,
        'OptimizationValidator': DummyClass,
        'ValidationTest': DummyClass,
        'ValidationReport': DummyClass,
        'BacktestValidationConfig': DummyClass
    })

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "imega"
__description__ = "5-1-3「リスク調整後リターンの最適化」システム"

# パブリックAPI
__all__ = [
    # 目的関数関連
    'CompositeObjectiveFunction',
    'ObjectiveFunctionBuilder', 
    'OptimizationObjective',
    'CompositeScoreResult',
    
    # 制約管理関連
    'RiskConstraintManager',
    'ConstraintResult',
    'ConstraintViolation',
    'ConstraintType',
    'ConstraintSeverity',
    'AdaptiveConstraintAdjuster',
    
    # 最適化アルゴリズム関連
    'OptimizationEngine',
    'OptimizationMethod',
    'OptimizationConfig',
    'OptimizationResult',
    'DifferentialEvolutionOptimizer',
    'ScipyMinimizeOptimizer',
    'GradientDescentOptimizer',
    
    # パフォーマンス評価関連
    'EnhancedPerformanceEvaluator',
    'ComprehensivePerformanceReport',
    'PerformanceMetric',
    'MetricCategory',
    
    # メイン最適化エンジン
    'RiskAdjustedOptimizationEngine',
    'OptimizationContext',
    'RiskAdjustedOptimizationResult',
    
    # 高度ポートフォリオ最適化
    'AdvancedPortfolioOptimizer',
    'PortfolioOptimizationProfile',
    'MultiPeriodOptimizationRequest', 
    'PortfolioOptimizationResult',
    
    # 結果検証
    'OptimizationValidator',
    'ValidationTest',
    'ValidationReport',
    'BacktestValidationConfig'
]
