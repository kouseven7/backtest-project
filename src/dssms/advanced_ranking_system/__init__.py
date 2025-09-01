"""
DSSMS Phase 3 Task 3.1: Advanced Ranking System
高度なランキングシステム統合モジュール

このパッケージは、DSSMSの既存システムを拡張し、
高度な多次元分析と動的最適化機能を提供します。

主要コンポーネント:
- AdvancedRankingEngine: メインエンジン
- MultiDimensionalAnalyzer: 多次元分析器
- DynamicWeightOptimizer: 動的重み最適化器
- IntegrationBridge: 既存システム統合ブリッジ
- RankingCacheManager: ランキングキャッシュ管理
- PerformanceMonitor: パフォーマンス監視
- RealtimeUpdater: リアルタイム更新機能
"""

# バージョン情報
__version__ = "3.1.0"
__author__ = "DSSMS Development Team"
__description__ = "Advanced Ranking System for DSSMS Phase 3"

# メインクラスのインポート
from .advanced_ranking_engine import AdvancedRankingEngine
from .multi_dimensional_analyzer import MultiDimensionalAnalyzer
from .dynamic_weight_optimizer import DynamicWeightOptimizer
from .integration_bridge import IntegrationBridge
from .ranking_cache_manager import RankingCacheManager
from .performance_monitor import PerformanceMonitor
from .realtime_updater import RealtimeUpdater

# 公開API
__all__ = [
    "AdvancedRankingEngine",
    "MultiDimensionalAnalyzer", 
    "DynamicWeightOptimizer",
    "IntegrationBridge",
    "RankingCacheManager",
    "PerformanceMonitor",
    "RealtimeUpdater"
]

# システム情報
SYSTEM_INFO = {
    "phase": "3.1",
    "task": "Advanced Ranking System",
    "integration_level": "full",
    "compatibility": "DSSMS 2.x+",
    "dependencies": [
        "hierarchical_ranking_system",
        "hybrid_ranking_engine",
        "comprehensive_scoring_engine"
    ]
}

# 設定ファイルパス
CONFIG_PATHS = {
    "main_config": "config/dssms/advanced_ranking_config.json",
    "weights_config": "config/dssms/ranking_weights_config.json",
    "cache_config": "config/dssms/cache_config.json"
}
