"""
フェーズ3B リアルタイムデータ監視モジュール

このモジュールは、リアルタイムデータシステムの監視とダッシュボード機能を提供します。

主要コンポーネント:
- MonitoringDashboard: Webベースのリアルタイム監視ダッシュボード
- DashboardAgent: ダッシュボードの自動更新とアラート管理
- MetricsCollector: システムメトリクス収集
- AlertManager: アラート管理とエスカレーション
"""

from .dashboard import MonitoringDashboard, DashboardAgent
from .metrics_collector import MetricsCollector
from .alert_manager import AlertManager

__all__ = [
    'MonitoringDashboard',
    'DashboardAgent', 
    'MetricsCollector',
    'AlertManager'
]

__version__ = '1.0.0'
