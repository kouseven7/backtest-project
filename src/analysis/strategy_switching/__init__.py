"""
Strategy Switching Analysis Package
5-1-1「戦略切替のタイミング分析ツール」

Author: imega
Created: 2025-01-21
Modified: 2025-01-21
"""

from .strategy_switching_analyzer import StrategySwitchingAnalyzer, SwitchingAnalysisResult
from .switching_timing_evaluator import SwitchingTimingEvaluator, TimingEvaluationResult
from .switching_pattern_detector import SwitchingPatternDetector, PatternAnalysisResult, PatternType
from .switching_performance_calculator import SwitchingPerformanceCalculator, SwitchingPerformanceResult
from .switching_analysis_dashboard import SwitchingAnalysisDashboard
from .switching_integration_system import SwitchingIntegrationSystem

__version__ = "1.0.0"
__author__ = "imega"

__all__ = [
    "StrategySwitchingAnalyzer",
    "SwitchingAnalysisResult", 
    "SwitchingTimingEvaluator",
    "TimingEvaluationResult",
    "SwitchingPatternDetector", 
    "PatternAnalysisResult",
    "PatternType",
    "SwitchingPerformanceCalculator",
    "SwitchingPerformanceResult",
    "SwitchingAnalysisDashboard",
    "SwitchingIntegrationSystem"
]
