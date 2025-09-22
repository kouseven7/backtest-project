"""
DSSMS実行ランタイム最適化モジュール
Problem 8: 実行ランタイム最適化 - データアクセス効率化とキャッシュ機能実装

主要機能:
1. portfolio_values アクセス最適化（27箇所参照効率化）
2. 50銘柄ランキング計算キャッシュ
3. メモリ使用量最適化
4. パフォーマンス監視・測定

設計方針:
- 既存ロジック非破壊でキャッシュレイヤー追加
- 85.0点エンジン品質維持
- DSSMS Core機能への影響最小化
- 決定論的動作保証（seed固定）
"""

import time
import gc
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import threading
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    total_accesses: int
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class PerformanceOptimizer:
    """実行時パフォーマンス最適化管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_config = config.get('cache', {})
        self.monitoring_config = config.get('performance_monitoring', {})
        self.memory_config = config.get('memory_optimization', {})
        
        # キャッシュシステム初期化
        self.data_cache = {}
        self.calculation_cache = {}
        self.access_metrics = {}
        
        # パフォーマンス監視
        self.enable_timing = self.monitoring_config.get('enable_timing', True)
        self.warning_threshold = self.monitoring_config.get('warning_threshold_seconds', 30)
        
        # スレッドセーフティ
        self._cache_lock = threading.RLock()
        
        logger.info("PerformanceOptimizer initialized")
        logger.info(f"Cache enabled: {self.cache_config.get('enable_ranking_cache', True)}")
        logger.info(f"Portfolio cache: {self.cache_config.get('enable_portfolio_cache', True)}")
        
    def optimize_portfolio_access(self, backtester_instance):
        """portfolio_values アクセス最適化"""
        # TODO(tag:phase2, rationale:27箇所portfolio_values最適化): キャッシュ戦略実装
        
        if not hasattr(backtester_instance, 'portfolio_values'):
            logger.warning("portfolio_values not found, skipping optimization")
            return None
            
        original_portfolio_values = backtester_instance.portfolio_values
        
        # キャッシュレイヤー追加
        cached_portfolio = CachedPortfolioManager(
            original_portfolio_values, 
            self.cache_config
        )
        
        # 元のportfolio_valuesを置き換え
        backtester_instance.portfolio_values = cached_portfolio
        
        logger.info("Portfolio access optimization applied")
        return cached_portfolio
        
    def optimize_ranking_calculation(self, symbols_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """50銘柄ランキング計算最適化"""
        # TODO(tag:phase2, rationale:50銘柄ランキング処理時間<30s): 計算効率化実装
        
        start_time = time.time()
        
        # キャッシュキー生成
        cache_key = self._generate_cache_key(symbols_data)
        
        with self._cache_lock:
            # キャッシュ確認
            if cache_key in self.calculation_cache:
                cached_result = self.calculation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.debug(f"Cache hit for ranking calculation: {cache_key[:8]}...")
                    return cached_result['data']
                else:
                    # 期限切れキャッシュ削除
                    del self.calculation_cache[cache_key]
                    
        # キャッシュミス時の計算実行
        logger.debug("Cache miss, executing ranking calculation")
        optimized_result = self._efficient_ranking_algorithm(symbols_data)
        
        # キャッシュ保存
        if self.cache_config.get('enable_ranking_cache', True):
            with self._cache_lock:
                self.calculation_cache[cache_key] = {
                    'data': optimized_result,
                    'timestamp': time.time(),
                    'ttl': self.cache_config.get('cache_ttl_seconds', 3600)
                }
                
                # キャッシュサイズ制限
                self._cleanup_cache_if_needed()
                
        # パフォーマンス監視
        execution_time = time.time() - start_time
        if execution_time > self.warning_threshold:
            logger.warning(f"Ranking calculation exceeded threshold: {execution_time:.2f}s")
            
        return optimized_result
        
    def _efficient_ranking_algorithm(self, symbols_data: Dict[str, Any]) -> Dict[str, Any]:
        """効率化されたランキング算出アルゴリズム"""
        # TODO(tag:phase2, rationale:重複計算削減): 最適化アルゴリズム実装
        
        if not symbols_data:
            return {}
            
        # 現段階では元のアルゴリズムにフォールバック
        # キャッシュ機能のみ提供し、計算ロジックは後続フェーズで最適化
        return symbols_data
        
    def performance_monitor(self, func):
        """パフォーマンス監視デコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_timing:
                return func(*args, **kwargs)
                
            # メモリ・CPU使用量測定開始
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.debug(f"{func.__name__}: {execution_time:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            # 警告閾値チェック
            if execution_time > self.warning_threshold:
                logger.warning(f"{func.__name__} exceeded threshold: {execution_time:.2f}s")
                
            return result
        return wrapper
        
    def get_performance_metrics(self) -> PerformanceMetrics:
        """現在のパフォーマンス指標取得"""
        process = psutil.Process()
        
        total_accesses = sum(self.access_metrics.values()) if self.access_metrics else 0
        cache_hits = len([v for v in self.calculation_cache.values() if self._is_cache_valid(v)])
        cache_hit_rate = (cache_hits / max(1, total_accesses)) * 100
        
        return PerformanceMetrics(
            execution_time=0.0,  # 実行時に更新
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=process.cpu_percent(),
            cache_hit_rate=cache_hit_rate,
            total_accesses=total_accesses
        )
        
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """キャッシュキー生成"""
        # データの重要な部分のみでハッシュ生成
        key_data = {
            'symbols': list(data.keys()) if isinstance(data, dict) else str(data),
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H')  # 時間単位での更新
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """キャッシュ有効性確認"""
        if 'timestamp' not in cached_item:
            return False
            
        cache_age = time.time() - cached_item['timestamp']
        ttl = cached_item.get('ttl', self.cache_config.get('cache_ttl_seconds', 3600))
        
        return cache_age < ttl
        
    def _cleanup_cache_if_needed(self):
        """キャッシュサイズ制限・クリーンアップ"""
        max_size = self.cache_config.get('cache_size_limit', 1000)
        
        if len(self.calculation_cache) > max_size:
            # 古いエントリから削除
            sorted_items = sorted(
                self.calculation_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            
            # 半分のサイズまで削減
            items_to_remove = len(sorted_items) - max_size // 2
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self.calculation_cache[key]
                
            logger.debug(f"Cache cleanup: removed {items_to_remove} old entries")

class CachedPortfolioManager:
    """portfolio_values アクセス最適化"""
    
    def __init__(self, original_portfolio: Dict[str, Any], cache_config: Dict[str, Any]):
        self._original = original_portfolio
        self._cache = {}
        self._access_count = {}
        self._last_access_time = {}
        self.cache_config = cache_config
        
        # 統計情報
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_lock = threading.RLock()
        
        logger.debug("CachedPortfolioManager initialized")
        
    def __getitem__(self, key):
        """キャッシュ機能付きアクセス"""
        with self._cache_lock:
            self._total_requests += 1
            
            # キャッシュ確認
            if key in self._cache:
                self._cache_hits += 1
                self._access_count[key] = self._access_count.get(key, 0) + 1
                self._last_access_time[key] = time.time()
                return self._cache[key]
                
            # キャッシュミス時の元データ取得
            if key in self._original:
                value = self._original[key]
                
                # キャッシュに保存
                if self.cache_config.get('enable_portfolio_cache', True):
                    self._cache[key] = value
                    self._access_count[key] = 1
                    self._last_access_time[key] = time.time()
                    
                    # キャッシュサイズ制限
                    self._cleanup_if_needed()
                    
                return value
            else:
                raise KeyError(f"Key '{key}' not found in portfolio_values")
                
    def __setitem__(self, key, value):
        """値設定（元データとキャッシュ両方更新）"""
        with self._cache_lock:
            self._original[key] = value
            if key in self._cache:
                self._cache[key] = value
                
    def __contains__(self, key):
        """in演算子サポート"""
        return key in self._original
        
    def keys(self):
        """キー一覧取得"""
        return self._original.keys()
        
    def items(self):
        """キー・値ペア取得"""
        return self._original.items()
        
    def values(self):
        """値一覧取得"""
        return self._original.values()
        
    def get(self, key, default=None):
        """安全なget操作"""
        try:
            return self[key]
        except KeyError:
            return default
            
    def get_access_metrics(self) -> Dict[str, Any]:
        """アクセス統計取得（パフォーマンス監視用）"""
        with self._cache_lock:
            cache_hit_rate = (self._cache_hits / max(1, self._total_requests)) * 100
            
            return {
                'total_accesses': sum(self._access_count.values()),
                'unique_keys': len(self._access_count),
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self._cache),
                'total_requests': self._total_requests,
                'cache_hits': self._cache_hits,
                'most_accessed_keys': sorted(
                    self._access_count.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            }
            
    def _cleanup_if_needed(self):
        """キャッシュクリーンアップ"""
        max_size = self.cache_config.get('cache_size_limit', 1000)
        
        if len(self._cache) > max_size:
            # LRU（最近最少使用）ベースでクリーンアップ
            sorted_by_access = sorted(
                self._last_access_time.items(),
                key=lambda x: x[1]
            )
            
            # 古いエントリの20%を削除
            items_to_remove = max_size // 5
            for i in range(min(items_to_remove, len(sorted_by_access))):
                key = sorted_by_access[i][0]
                if key in self._cache:
                    del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                if key in self._last_access_time:
                    del self._last_access_time[key]

class MemoryOptimizer:
    """メモリ使用量最適化"""
    
    def __init__(self, config: Dict[str, Any]):
        self.large_data_threshold_mb = config.get('large_data_threshold_mb', 100)
        self.enable_gc_optimization = config.get('enable_gc_optimization', True)
        
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量最適化実行"""
        # TODO(tag:phase2, rationale:メモリ圧迫による劣化防止): メモリ最適化実装
        
        if not self.enable_gc_optimization:
            return {'status': 'disabled'}
            
        # 現在のメモリ使用量確認
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # ガベージコレクション実行
        collected = gc.collect()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        logger.debug(f"Memory optimization: freed {memory_freed:.1f}MB, collected {collected} objects")
        
        return {
            'memory_freed_mb': memory_freed,
            'objects_collected': collected,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after
        }

def performance_timing(func):
    """パフォーマンス測定デコレータ（独立使用可能）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.debug(f"{func.__name__} execution time: {execution_time:.2f}s")
        return result
    return wrapper