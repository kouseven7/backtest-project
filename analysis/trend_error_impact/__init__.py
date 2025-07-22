"""
Module: Trend Error Impact Integration
File: __init__.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  統合パッケージ初期化

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

from .error_classification_engine import (
    TrendErrorClassificationEngine,
    TrendErrorType,
    ErrorSeverity,
    TrendErrorInstance,
    ErrorClassificationResult
)

from .error_impact_calculator import (
    ErrorImpactCalculator,
    BatchErrorImpactAnalyzer,
    ImpactMetrics,
    ErrorImpactResult
)

from .trend_error_detector import (
    TrendErrorDetector,
    ErrorDetectionResult
)

from .trend_error_analyzer import (
    TrendErrorAnalyzer,
    ComprehensiveAnalysisResult
)

__all__ = [
    'TrendErrorClassificationEngine',
    'TrendErrorType',
    'ErrorSeverity', 
    'TrendErrorInstance',
    'ErrorClassificationResult',
    'ErrorImpactCalculator',
    'BatchErrorImpactAnalyzer',
    'ImpactMetrics',
    'ErrorImpactResult',
    'TrendErrorDetector',
    'ErrorDetectionResult',
    'TrendErrorAnalyzer',
    'ComprehensiveAnalysisResult'
]

__version__ = '1.0.0'
__author__ = 'imega'
