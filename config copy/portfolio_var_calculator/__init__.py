"""
5-3-2「ポートフォリオVaR（バリューアットリスク）計算」

高度なVaR計算システム - 既存システムとの統合による包括的リスク管理
- ハイブリッドVaR計算手法
- 動的リスクモデリング
- リアルタイム監視・アラート
- 既存システム統合ブリッジ

Author: imega
Created: 2025-07-22
"""

from .advanced_var_engine import AdvancedVaREngine, VaRCalculationConfig
from .hybrid_var_calculator import HybridVaRCalculator
from .real_time_var_monitor import RealTimeVaRMonitor
from .var_integration_bridge import VaRIntegrationBridge
from .var_backtesting_engine import VaRBacktestingEngine

__version__ = "1.0.0"
__author__ = "Portfolio VaR System"

__all__ = [
    'AdvancedVaREngine',
    'VaRCalculationConfig', 
    'HybridVaRCalculator',
    'RealTimeVaRMonitor',
    'VaRIntegrationBridge',
    'VaRBacktestingEngine'
]
