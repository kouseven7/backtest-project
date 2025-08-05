"""
5-2-1「戦略実績に基づくスコア補正機能」パッケージ

このパッケージは、戦略の実績を基にスコア補正を行う包括的なシステムを提供します。
指数移動平均と適応的学習を組み合わせた補正アルゴリズムにより、
予測精度の向上と戦略選択の最適化を実現します。

Author: imega
Created: 2025-07-22
"""

# メインクラスのインポート
try:
    from .performance_tracker import PerformanceTracker, StrategyPerformanceRecord
    from .score_corrector import PerformanceBasedScoreCorrector, CorrectionResult
    from .enhanced_score_calculator import EnhancedStrategyScoreCalculator, CorrectedStrategyScore
    from .batch_processor import ScoreCorrectionBatchProcessor, BatchUpdateResult
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    # 基本的なエクスポート
    from .performance_tracker import PerformanceTracker
    from .score_corrector import PerformanceBasedScoreCorrector

# バージョン情報
__version__ = "1.0.0"
__author__ = "imega"
__description__ = "5-2-1 戦略実績に基づくスコア補正機能"

# パッケージレベルのエクスポート
__all__ = [
    # 実績追跡
    "PerformanceTracker",
    "StrategyPerformanceRecord",
    
    # スコア補正
    "PerformanceBasedScoreCorrector", 
    "CorrectionResult",
    
    # 統合計算器
    "EnhancedStrategyScoreCalculator",
    "CorrectedStrategyScore",
    
    # バッチ処理
    "ScoreCorrectionBatchProcessor",
    "BatchUpdateResult"
]
