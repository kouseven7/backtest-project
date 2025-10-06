

# TODO-PERF-001: Phase 2 Stage 2 - SystemFallbackPolicy Integration
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType
    _fallback_policy = SystemFallbackPolicy()
except ImportError:
    # フォールバック用のダミークラス
    class _DummyFallbackPolicy:
        def handle_component_failure(self, **kwargs):
            return None
    _fallback_policy = _DummyFallbackPolicy()

def _handle_lazy_import_failure(component_name: str, error: Exception, fallback_func=None):
    """遅延インポート失敗時のフォールバック処理"""
    try:
        return _fallback_policy.handle_component_failure(
            component_type="STRATEGY_ENGINE",
            component_name=component_name,
            error=error,
            fallback_func=fallback_func
        )
    except:
        # 最終フォールバック
        if fallback_func:
            return fallback_func()
        return None


# TODO-PERF-001: Phase 2 Stage 2 - Correlation Lazy Import System
import importlib
from typing import Any, Optional

class LazyCorrelationImporter:
    """Correlation系モジュール遅延インポートクラス"""
    
    def __init__(self):
        self._correlation_modules = {}
    
    def get_correlation_module(self, module_name: str) -> Optional[Any]:
        """correlation系モジュール遅延ロード"""
        if module_name not in self._correlation_modules:
            try:
                full_module_name = f'config.correlation.{module_name}'
                self._correlation_modules[module_name] = importlib.import_module(full_module_name)
            except ImportError:
                self._correlation_modules[module_name] = None
        return self._correlation_modules[module_name]
    
    def __getattr__(self, name: str) -> Any:
        """属性アクセス時の動的ロード"""
        module = self.get_correlation_module(name)
        if module is not None:
            return module
        raise AttributeError(f"module 'config.correlation' has no attribute '{name}'")

# 遅延インポートシステム初期化
_lazy_correlation = LazyCorrelationImporter()


"""
Correlation Analysis Package

戦略間相関分析システム
"""

# from .strategy_correlation_analyzer import (  # TODO-PERF-001: Converted to lazy import
#     CorrelationConfig,
#     CorrelationMatrix,
#     StrategyPerformanceData,
#     StrategyCorrelationAnalyzer
# )

# from .correlation_matrix_visualizer import (  # TODO-PERF-001: Converted to lazy import
#     CorrelationMatrixVisualizer
# )

from .strategy_correlation_dashboard import (
    StrategyCorrelationDashboard
)

__all__ = [
    # 'CorrelationConfig',  # TODO-PERF-001: Lazy import - not directly available
    # 'CorrelationMatrix',  # TODO-PERF-001: Lazy import - not directly available
    # 'StrategyPerformanceData',  # TODO-PERF-001: Lazy import - not directly available
    # 'StrategyCorrelationAnalyzer',  # TODO-PERF-001: Lazy import - not directly available
    # 'CorrelationMatrixVisualizer',  # TODO-PERF-001: Lazy import - not directly available
    'StrategyCorrelationDashboard'
]
