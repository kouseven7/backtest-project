"""
包括的レポートシステム

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムパッケージ

このパッケージは以下のコンポーネントを提供します:
- ComprehensiveReportEngine: メインエンジン
- DataAggregator: データ集約
- VisualizationGenerator: 可視化生成
- ReportTemplateManager: テンプレート管理
- ExportManager: エクスポート管理
"""

from .comprehensive_report_engine import ComprehensiveReportEngine
from .data_aggregator import DataAggregator
from .visualization_generator import VisualizationGenerator
from .report_template_manager import ReportTemplateManager
from .export_manager import ExportManager

__version__ = "1.0.0"
__author__ = "DSSMS Team"

__all__ = [
    "ComprehensiveReportEngine",
    "DataAggregator", 
    "VisualizationGenerator",
    "ReportTemplateManager",
    "ExportManager"
]
