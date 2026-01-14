

# TODO-PERF-001: Phase 2 Stage 2 - SystemFallbackPolicy Integration
# Phase 2025-12-02: copilot-instructions.md準拠 - ダミーフォールバック削除
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType
    _fallback_policy = SystemFallbackPolicy()
except ImportError as e:
    # copilot-instructions.md準拠: モック/ダミーフォールバック禁止
    # エラーを明示し、問題を隠蔽しない
    import logging
    logging.warning(
        f"SystemFallbackPolicy import failed: {e}. "
        f"Component failure handling is disabled. "
        f"This may cause unexpected errors."
    )
    _fallback_policy = None  # Noneを返す（ダミーを返さない）

def _handle_lazy_import_failure(component_name: str, error: Exception, fallback_func=None):
    """遅延インポート失敗時のフォールバック処理（Phase 2025-12-02修正）"""
    # copilot-instructions.md準拠: フォールバック実行時のログ必須
    import logging
    logger = logging.getLogger(__name__)
    
    if _fallback_policy is not None:
        try:
            logger.warning(
                f"[FALLBACK_DETECTED] Component '{component_name}' failed to load. "
                f"Error: {error}. Attempting fallback..."
            )
            return _fallback_policy.handle_component_failure(
                component_type="STRATEGY_ENGINE",
                component_name=component_name,
                error=error,
                fallback_func=fallback_func
            )
        except Exception as fallback_error:
            logger.error(
                f"[FALLBACK_FAILED] Fallback for '{component_name}' also failed: {fallback_error}"
            )
    else:
        logger.error(
            f"[NO_FALLBACK] Component '{component_name}' failed to load, and no fallback available. "
            f"Error: {error}"
        )
    
    # copilot-instructions.md準拠: フォールバック関数が実データと乖離する場合は使用しない
    # fallback_funcが提供されていても、実データ乖離の可能性があるため慎重に判断
    if fallback_func:
        logger.warning(
            f"[FALLBACK_FUNC_PROVIDED] fallback_func provided for '{component_name}', "
            f"but execution is risky. Returning None instead."
        )
    return None


# TODO-PERF-001: Phase 2 Stage 2 - Config Lazy Import System
import importlib
from typing import Any, Optional

class LazyConfigImporter:
    """Config系モジュール遅延インポートクラス"""
    
    def __init__(self):
        self._modules = {}
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """モジュール遅延ロード"""
        if module_name not in self._modules:
            try:
                if module_name.startswith('.'):
                    # 相対インポート
                    self._modules[module_name] = importlib.import_module(module_name, package='config')
                else:
                    # 絶対インポート
                    self._modules[module_name] = importlib.import_module(f'config.{module_name}')
            except ImportError:
                self._modules[module_name] = None
        return self._modules[module_name]
    
    def __getattr__(self, name: str) -> Any:
        """属性アクセス時の動的ロード"""
        module = self.get_module(name)
        if module is not None:
            return module
        raise AttributeError(f"module 'config' has no attribute '{name}'")

# 遅延インポートシステム初期化
_lazy_config = LazyConfigImporter()


"""
Configuration Package

プロジェクトの設定とシステムコンポーネント
"""

# 相関分析システムのインポート
try:
    # from .correlation import *  # TODO-PERF-001: Converted to lazy import
    pass
except ImportError:
    pass

# ポートフォリオ相関最適化システムのインポート（遅延インポート化、2026-01-11修正）
# Note: seaborn/scipy依存により初回起動が重いため遅延インポート化
# try:
#     from .portfolio_correlation_optimizer import *
# except ImportError:
#     pass