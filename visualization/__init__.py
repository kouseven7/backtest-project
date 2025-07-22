"""
4-3-1: トレンド変化と戦略切替の時系列グラフ可視化システム

このモジュールは、トレンド変化と戦略切替のタイミングを時系列グラフで可視化するシステムを提供します。
- 価格・トレンド・戦略・信頼度・ボリューム情報を統合したマルチパネルチャート
- Matplotlibベースの軽量な可視化エンジン
- 既存システムとの独立した動作
- PNG形式での効率的な出力

主要コンポーネント:
- trend_strategy_time_series.py: メイン可視化エンジン
- chart_config.py: チャート設定管理
- data_aggregator.py: データ集約・前処理
"""

__version__ = "1.0.0"
__author__ = "Backtest System"

from .trend_strategy_time_series import TrendStrategyTimeSeriesVisualizer
from .chart_config import ChartConfigManager
from .data_aggregator import VisualizationDataAggregator

__all__ = [
    'TrendStrategyTimeSeriesVisualizer',
    'ChartConfigManager', 
    'VisualizationDataAggregator'
]
