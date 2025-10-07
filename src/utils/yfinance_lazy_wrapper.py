#!/usr/bin/env python3
"""
yfinance遅延インポートラッパー
TODO-PERF-001: Phase 1 Stage 2実装

yfinanceの初回インポート時のみ遅延を発生させ、
2回目以降は高速アクセスを提供する。
"""

import importlib.util
import sys
import time
from typing import Any, Optional, Dict


# SystemFallbackPolicy統合 (TODO-PERF-001: Phase 1 Stage 2)
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType
    _fallback_policy = SystemFallbackPolicy()
except ImportError:
    _fallback_policy = None
    print("[WARNING] SystemFallbackPolicy not available")

def _handle_yfinance_error(error: Exception, operation: str):
    """yfinanceエラーハンドリング"""
    if _fallback_policy:
        return _fallback_policy.handle_component_failure(
            component_type=ComponentType.DATA_FETCHER,
            component_name="yfinance_lazy_wrapper",
            error=error,
            fallback_func=lambda: None
        )
    else:
        print(f"[ERROR] yfinance error in {operation}: {error}")
        raise error


class YfinanceLazyWrapper:
    """yfinance遅延インポートラッパー"""
    
    def __init__(self):
        self._yfinance = None
        self._import_time = None
        self._first_access = True
        
    def _import_yfinance(self) -> Any:
        """yfinance実際のインポート（初回のみ）"""
        if self._yfinance is None:
            start_time = time.perf_counter()
            
            try:
                import yfinance as yf
                self._yfinance = yf
                self._import_time = (time.perf_counter() - start_time) * 1000
                
                if self._first_access:
                    print(f"[INFO] yfinance lazy import: {self._import_time:.1f}ms")
                    self._first_access = False
                    
            except ImportError as e:
                _handle_yfinance_error(e, "_import_yfinance")
                
        return self._yfinance
    
    def __getattr__(self, name: str) -> Any:
        """yfinanceの属性・メソッドに透明アクセス"""
        yf = self._import_yfinance()
        return getattr(yf, name)
    
    # よく使用されるメソッドの直接実装
    def download(self, *args, **kwargs):
        """yf.download()の遅延ラッパー"""
        yf = self._import_yfinance()
        return yf.download(*args, **kwargs)
    
    def Ticker(self, *args, **kwargs):
        """yf.Ticker()の遅延ラッパー"""
        yf = self._import_yfinance()
        return yf.Ticker(*args, **kwargs)
    
    def get_import_stats(self) -> Dict[str, Any]:
        """インポート統計取得"""
        return {
            'imported': self._yfinance is not None,
            'import_time_ms': self._import_time,
            'first_access_completed': not self._first_access
        }

# グローバルインスタンス
_lazy_yfinance = YfinanceLazyWrapper()

# yfinanceのAPIをエクスポート
def download(*args, **kwargs):
    return _lazy_yfinance.download(*args, **kwargs)

def Ticker(*args, **kwargs):
    return _lazy_yfinance.Ticker(*args, **kwargs)

# 統計情報エクスポート
def get_yfinance_import_stats():
    return _lazy_yfinance.get_import_stats()

# 属性アクセス用
def __getattr__(name: str):
    return getattr(_lazy_yfinance, name)
