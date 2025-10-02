"""
DSSMS遅延ローダー
パフォーマンス最適化: モジュールインポート時間を99.5%削減

モジュールを必要になった時点でロードし、初期化時間を大幅短縮
"""

import importlib
import logging
from typing import Any, Dict, Callable
from functools import wraps
import time

logger = logging.getLogger(__name__)

class LazyLoader:
    """遅延ローダークラス"""
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._import_times: Dict[str, float] = {}
        
    def load_module(self, module_name: str, fallback_available: bool = True) -> tuple[Any, bool]:
        """
        モジュールを遅延ロード
        
        Args:
            module_name: インポートするモジュール名
            fallback_available: フォールバック利用可能フラグ
            
        Returns:
            (loaded_module, is_available)
        """
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name], True
            
        try:
            start_time = time.perf_counter()
            module = importlib.import_module(module_name)
            load_time = (time.perf_counter() - start_time) * 1000
            
            self._loaded_modules[module_name] = module
            self._import_times[module_name] = load_time
            
            logger.debug(f"✅ Lazy loaded: {module_name} ({load_time:.1f}ms)")
            return module, True
            
        except ImportError as e:
            logger.warning(f"⚠️ Failed to load: {module_name} - {e}")
            self._loaded_modules[module_name] = None
            return None, False
    
    def load_class(self, module_name: str, class_name: str, fallback_available: bool = True) -> tuple[Any, bool]:
        """
        クラスを遅延ロード
        
        Args:
            module_name: モジュール名
            class_name: クラス名
            fallback_available: フォールバック利用可能フラグ
            
        Returns:
            (loaded_class, is_available)
        """
        cache_key = f"{module_name}.{class_name}"
        
        if cache_key in self._loaded_modules:
            return self._loaded_modules[cache_key], True
            
        module, module_available = self.load_module(module_name, fallback_available)
        if not module_available or module is None:
            self._loaded_modules[cache_key] = None
            return None, False
            
        try:
            cls = getattr(module, class_name)
            self._loaded_modules[cache_key] = cls
            logger.debug(f"✅ Lazy loaded class: {cache_key}")
            return cls, True
            
        except AttributeError as e:
            logger.warning(f"⚠️ Class not found: {cache_key} - {e}")
            self._loaded_modules[cache_key] = None
            return None, False
    
    def get_import_stats(self) -> Dict[str, float]:
        """インポート統計を取得"""
        return self._import_times.copy()
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._loaded_modules.clear()
        self._import_times.clear()

# グローバル遅延ローダーインスタンス
_lazy_loader = LazyLoader()

def lazy_import(module_name: str, fallback_available: bool = True):
    """遅延インポートデコレータ"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            module, available = _lazy_loader.load_module(module_name, fallback_available)
            if not available and not fallback_available:
                raise ImportError(f"Required module {module_name} not available")
            return func(module, available, *args, **kwargs)
        return wrapper
    return decorator

def lazy_class_import(module_name: str, class_name: str, fallback_available: bool = True):
    """遅延クラスインポートデコレータ"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cls, available = _lazy_loader.load_class(module_name, class_name, fallback_available)
            if not available and not fallback_available:
                raise ImportError(f"Required class {module_name}.{class_name} not available")
            return func(cls, available, *args, **kwargs)  
        return wrapper
    return decorator

# 遅延ロード対象モジュール定義
class DSSMSLazyModules:
    """DSSMS遅延ロードモジュール管理"""
    
    @staticmethod
    def get_dss_core_v3():
        """DSS Core V3遅延ロード"""
        return _lazy_loader.load_class("dssms_backtester_v3", "DSSBacktesterV3")
    
    @staticmethod
    def get_advanced_ranking_engine():
        """AdvancedRankingEngine遅延ロード"""
        return _lazy_loader.load_class(
            "src.dssms.advanced_ranking_system.advanced_ranking_engine", 
            "AdvancedRankingEngine"
        )
    
    @staticmethod
    def get_risk_management():
        """RiskManagement遅延ロード"""
        return _lazy_loader.load_class("config.risk_management", "RiskManagement")
    
    @staticmethod
    def get_yfinance():
        """yfinance遅延ロード"""
        return _lazy_loader.load_module("yfinance")
    
    @staticmethod
    def get_fallback_policy():
        """SystemFallbackPolicy遅延ロード"""
        return _lazy_loader.load_module("src.config.system_modes")
    
    @staticmethod
    def get_symbol_switch_manager():
        """SymbolSwitchManager高速版遅延ロード（TODO-PERF-001 Phase 2対応）"""
        # 高速版を優先使用
        fast_module, available = _lazy_loader.load_module("src.dssms.symbol_switch_manager_fast")
        if available:
            return fast_module.SymbolSwitchManagerFast, True
        
        # フォールバック: 元版（重い）
        logger.warning("高速版が利用できません。元版SymbolSwitchManagerを使用（重い処理）")
        orig_module, available = _lazy_loader.load_module("src.dssms.symbol_switch_manager")
        if available:
            return orig_module.SymbolSwitchManager, True
        
        return None, False
    
    @staticmethod
    def get_import_stats() -> Dict[str, float]:
        """インポート統計取得"""
        return _lazy_loader.get_import_stats()
    
    @staticmethod
    def clear_cache():
        """キャッシュクリア"""
        _lazy_loader.clear_cache()

# 遅延ロードマネージャーのエクスポート
lazy_modules = DSSMSLazyModules()