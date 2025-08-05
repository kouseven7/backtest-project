# Market Classification System
"""
A→B市場分類システムの統合モジュール
市場状況を7段階で詳細分類し、既存システムとの統合を提供

主要コンポーネント:
- 市場状況検出・分析基盤
- 技術指標統合分析
- 市場レジーム分類
- ボラティリティ分析
- トレンド強度評価
- 市場相関分析
- 統合状態管理
- キャッシュ管理
- エラーハンドリング
- 統合デモンストレーション
"""

# 基本コンポーネント
from .market_conditions import SimpleMarketCondition, DetailedMarketCondition, MarketMetrics, ClassificationResult, MarketConditions
from .market_classifier import MarketClassifier
from .classification_analyzer import ClassificationAnalyzer

# 拡張分析コンポーネント
from .market_condition_detector import MarketConditionDetector, DetectionResult
from .technical_indicator_analyzer import TechnicalIndicatorAnalyzer, TechnicalAnalysisResult
from .market_regime_classifier import MarketRegimeClassifier, RegimeClassificationResult
from .volatility_analyzer import VolatilityAnalyzer, VolatilityAnalysisResult
from .trend_strength_evaluator import TrendStrengthEvaluator, TrendStrengthResult
from .market_correlation_analyzer import MarketCorrelationAnalyzer, MarketCorrelationAnalysis

# 統合管理システム
from .integrated_market_state_manager import (
    IntegratedMarketStateManager, 
    IntegratedMarketState,
    StateIntegrationMethod,
    ComponentWeight
)

# サポートシステム
from .cache_manager import (
    MultiLevelCacheManager,
    MarketDataCacheManager,
    AnalysisResultCacheManager,
    CacheType,
    CacheLevel
)

from .error_handling import (
    RobustAnalysisSystem,
    RecoveryAction,
    RecoveryStrategy,
    ErrorCategory,
    ErrorSeverity,
    robust_analysis
)

# デモンストレーション
from .integrated_demo import IntegratedMarketClassificationDemo

__all__ = [
    # 基本コンポーネント
    'SimpleMarketCondition',
    'DetailedMarketCondition', 
    'MarketMetrics',
    'ClassificationResult',
    'MarketConditions',
    'MarketClassifier',
    'ClassificationAnalyzer',
    
    # 検出・分析コンポーネント
    'MarketConditionDetector',
    'DetectionResult',
    'TechnicalIndicatorAnalyzer',
    'TechnicalAnalysisResult',
    'MarketRegimeClassifier',
    'RegimeClassificationResult',
    'VolatilityAnalyzer',
    'VolatilityAnalysisResult',
    'TrendStrengthEvaluator',
    'TrendStrengthResult',
    'MarketCorrelationAnalyzer',
    'MarketCorrelationAnalysis',
    
    # 統合管理
    'IntegratedMarketStateManager',
    'IntegratedMarketState',
    'StateIntegrationMethod',
    'ComponentWeight',
    
    # サポートシステム
    'MultiLevelCacheManager',
    'MarketDataCacheManager',
    'AnalysisResultCacheManager',
    'CacheType',
    'CacheLevel',
    'RobustAnalysisSystem',
    'RecoveryAction',
    'RecoveryStrategy',
    'ErrorCategory',
    'ErrorSeverity',
    'robust_analysis',
    
    # デモンストレーション
    'IntegratedMarketClassificationDemo'
]

__version__ = "2.0.0"
__author__ = "A→B Market Classification System"
__description__ = "Advanced market classification system with 7-level detailed analysis"

# 便利な初期化関数
def create_basic_classification_system():
    """基本的な市場分類システムを作成"""
    return IntegratedMarketStateManager(
        integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE,
        auto_update=False
    )

def create_cached_classification_system(cache_dir="market_cache"):
    """キャッシュ付き市場分類システムを作成"""
    cache_manager = MultiLevelCacheManager(cache_dir=cache_dir)
    state_manager = IntegratedMarketStateManager(
        integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE,
        auto_update=False
    )
    return state_manager, cache_manager

def create_robust_classification_system(cache_dir="market_cache", error_log="market_errors.log"):
    """堅牢性を持つ市場分類システムを作成"""
    cache_manager = MultiLevelCacheManager(cache_dir=cache_dir)
    error_handler = RobustAnalysisSystem(error_log_file=error_log)
    state_manager = IntegratedMarketStateManager(
        integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE,
        auto_update=False
    )
    return state_manager, cache_manager, error_handler

def run_system_demo():
    """システムデモンストレーション実行"""
    demo = IntegratedMarketClassificationDemo()
    demo.setup_test_environment()
    return demo.run_all_tests()

from .market_classifier import MarketClassifier
from .market_conditions import MarketConditions, SimpleMarketCondition, DetailedMarketCondition, ClassificationResult, MarketMetrics
from .classification_analyzer import ClassificationAnalyzer

__all__ = [
    'MarketClassifier', 
    'MarketConditions', 
    'SimpleMarketCondition', 
    'DetailedMarketCondition', 
    'ClassificationResult', 
    'MarketMetrics',
    'ClassificationAnalyzer'
]
