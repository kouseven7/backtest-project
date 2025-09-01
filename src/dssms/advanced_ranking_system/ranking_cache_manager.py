"""
DSSMS Phase 3 Task 3.1: Ranking Cache Manager
ランキングキャッシュ管理クラス

高度ランキングシステムの計算結果をキャッシュし、
パフォーマンスを最適化します。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
import json
import hashlib
import threading
import time
from collections import OrderedDict

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class CacheType(Enum):
    """キャッシュタイプ定義"""
    RANKING_RESULTS = "ranking_results"
    ANALYSIS_DATA = "analysis_data"
    WEIGHT_OPTIMIZATION = "weight_optimization"
    MARKET_DATA = "market_data"
    PERFORMANCE_METRICS = "performance_metrics"

class CacheStrategy(Enum):
    """キャッシュ戦略定義"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # 適応的
    SIZE_BASED = "size_based"      # サイズベース

@dataclass
class CacheConfig:
    """キャッシュ設定"""
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_size_mb: int = 500
    max_entries: int = 10000
    default_ttl_seconds: int = 3600
    enable_persistence: bool = True
    persistence_path: str = "cache/dssms/ranking_cache"
    cleanup_interval_seconds: int = 300
    enable_compression: bool = True
    enable_encryption: bool = False

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    value: Any
    cache_type: CacheType
    created_time: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """キャッシュ統計"""
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float
    avg_access_time_ms: float
    cache_efficiency: float
    type_distribution: Dict[str, int]

class RankingCacheManager:
    """
    ランキングキャッシュ管理クラス
    
    機能:
    - 多階層キャッシュ管理
    - 複数のキャッシュ戦略サポート
    - 永続化とリストア
    - パフォーマンス監視
    - 自動クリーンアップ
    - 圧縮と暗号化
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初期化
        
        Args:
            config: キャッシュ設定
        """
        self.config = config or CacheConfig()
        self.logger = logger
        
        # キャッシュストレージ
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._type_caches: Dict[CacheType, OrderedDict[str, CacheEntry]] = {
            cache_type: OrderedDict() for cache_type in CacheType
        }
        
        # 統計情報
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_access_time': 0.0,
            'access_count': 0
        }
        
        # スレッドロック
        self._lock = threading.RLock()
        
        # クリーンアップスレッド
        self._cleanup_thread = None
        self._cleanup_running = False
        
        # 初期化実行
        self._initialize_cache()
        
        self.logger.info(f"Ranking Cache Manager initialized with strategy: {self.config.strategy}")
    
    def _initialize_cache(self):
        """キャッシュ初期化"""
        try:
            # 永続化ディレクトリ作成
            if self.config.enable_persistence:
                cache_dir = Path(self.config.persistence_path)
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 既存キャッシュの復元
                self._restore_cache()
            
            # クリーンアップスレッド開始
            self._start_cleanup_thread()
            
        except Exception as e:
            self.logger.error(f"Cache initialization failed: {e}")
            raise
    
    def put(
        self, 
        key: str, 
        value: Any, 
        cache_type: CacheType,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        キャッシュエントリ追加
        
        Args:
            key: キャッシュキー
            value: キャッシュ値
            cache_type: キャッシュタイプ
            ttl_seconds: 有効期限（秒）
            metadata: メタデータ
            
        Returns:
            成功フラグ
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # TTL設定
                ttl = ttl_seconds or self.config.default_ttl_seconds
                
                # サイズ計算
                size_bytes = self._calculate_size(value)
                
                # 容量チェック
                if not self._check_capacity(size_bytes):
                    self._evict_entries(size_bytes)
                
                # エントリ作成
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=cache_type,
                    created_time=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    ttl_seconds=ttl,
                    size_bytes=size_bytes,
                    metadata=metadata or {}
                )
                
                # キャッシュに追加
                self._cache[key] = entry
                self._type_caches[cache_type][key] = entry
                
                # 永続化
                if self.config.enable_persistence:
                    self._persist_entry(entry)
                
                # 統計更新
                access_time = (time.time() - start_time) * 1000
                self._update_stats('put', access_time)
                
                self.logger.debug(f"Cache entry added: {key} ({cache_type.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"Cache put failed for key {key}: {e}")
            return False
    
    def get(self, key: str, cache_type: Optional[CacheType] = None) -> Optional[Any]:
        """
        キャッシュエントリ取得
        
        Args:
            key: キャッシュキー
            cache_type: キャッシュタイプ（フィルタ用）
            
        Returns:
            キャッシュ値（存在しない場合はNone）
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # エントリ取得
                entry = self._cache.get(key)
                
                if entry is None:
                    self._stats['misses'] += 1
                    return None
                
                # タイプフィルタ
                if cache_type and entry.cache_type != cache_type:
                    self._stats['misses'] += 1
                    return None
                
                # TTLチェック
                if self._is_expired(entry):
                    self._remove_entry(key)
                    self._stats['misses'] += 1
                    return None
                
                # アクセス情報更新
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # LRU順序更新
                if self.config.strategy == CacheStrategy.LRU:
                    self._cache.move_to_end(key)
                    self._type_caches[entry.cache_type].move_to_end(key)
                
                # 統計更新
                access_time = (time.time() - start_time) * 1000
                self._update_stats('get', access_time)
                self._stats['hits'] += 1
                
                self.logger.debug(f"Cache hit: {key}")
                return entry.value
                
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            self._stats['misses'] += 1
            return None
    
    def remove(self, key: str) -> bool:
        """
        キャッシュエントリ削除
        
        Args:
            key: キャッシュキー
            
        Returns:
            成功フラグ
        """
        try:
            with self._lock:
                return self._remove_entry(key)
                
        except Exception as e:
            self.logger.error(f"Cache remove failed for key {key}: {e}")
            return False
    
    def clear(self, cache_type: Optional[CacheType] = None):
        """
        キャッシュクリア
        
        Args:
            cache_type: 特定タイプのみクリア（Noneの場合は全体）
        """
        try:
            with self._lock:
                if cache_type:
                    # 特定タイプのみクリア
                    type_cache = self._type_caches[cache_type]
                    keys_to_remove = list(type_cache.keys())
                    
                    for key in keys_to_remove:
                        self._remove_entry(key)
                    
                    self.logger.info(f"Cache cleared for type: {cache_type.value}")
                else:
                    # 全体クリア
                    self._cache.clear()
                    for type_cache in self._type_caches.values():
                        type_cache.clear()
                    
                    # 永続化ファイルも削除
                    if self.config.enable_persistence:
                        self._clear_persistence()
                    
                    self.logger.info("All cache cleared")
                    
        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")
    
    def get_by_pattern(self, pattern: str, cache_type: Optional[CacheType] = None) -> Dict[str, Any]:
        """
        パターンマッチによる複数エントリ取得
        
        Args:
            pattern: キーパターン（ワイルドカード対応）
            cache_type: キャッシュタイプフィルタ
            
        Returns:
            マッチしたエントリ辞書
        """
        try:
            import fnmatch
            
            with self._lock:
                results = {}
                
                target_cache = self._type_caches[cache_type] if cache_type else self._cache
                
                for key, entry in target_cache.items():
                    if fnmatch.fnmatch(key, pattern):
                        if not self._is_expired(entry):
                            results[key] = entry.value
                        else:
                            self._remove_entry(key)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Pattern get failed for pattern {pattern}: {e}")
            return {}
    
    def get_stats(self) -> CacheStats:
        """キャッシュ統計取得"""
        try:
            with self._lock:
                total_hits = self._stats['hits']
                total_misses = self._stats['misses']
                total_accesses = total_hits + total_misses
                
                hit_rate = (total_hits / total_accesses) if total_accesses > 0 else 0.0
                miss_rate = (total_misses / total_accesses) if total_accesses > 0 else 0.0
                
                avg_access_time = (self._stats['total_access_time'] / self._stats['access_count']) \
                    if self._stats['access_count'] > 0 else 0.0
                
                total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
                total_size_mb = total_size_bytes / (1024 * 1024)
                
                # タイプ別分布
                type_distribution = {}
                for cache_type, type_cache in self._type_caches.items():
                    type_distribution[cache_type.value] = len(type_cache)
                
                # 効率性計算
                cache_efficiency = hit_rate * (1 - total_size_mb / self.config.max_size_mb)
                
                return CacheStats(
                    total_entries=len(self._cache),
                    total_size_mb=total_size_mb,
                    hit_rate=hit_rate,
                    miss_rate=miss_rate,
                    avg_access_time_ms=avg_access_time,
                    cache_efficiency=cache_efficiency,
                    type_distribution=type_distribution
                )
                
        except Exception as e:
            self.logger.error(f"Stats calculation failed: {e}")
            return CacheStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    
    def optimize_cache(self):
        """キャッシュ最適化"""
        try:
            with self._lock:
                self.logger.info("Starting cache optimization")
                
                # 期限切れエントリ削除
                expired_count = self._cleanup_expired()
                
                # 戦略別最適化
                if self.config.strategy == CacheStrategy.ADAPTIVE:
                    self._adaptive_optimization()
                elif self.config.strategy == CacheStrategy.SIZE_BASED:
                    self._size_based_optimization()
                
                # 統計リセット
                self._reset_stats()
                
                self.logger.info(f"Cache optimization completed: {expired_count} expired entries removed")
                
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
    
    def _remove_entry(self, key: str) -> bool:
        """エントリ削除（内部用）"""
        try:
            entry = self._cache.get(key)
            if entry:
                del self._cache[key]
                del self._type_caches[entry.cache_type][key]
                
                # 永続化ファイル削除
                if self.config.enable_persistence:
                    self._remove_persisted_entry(key)
                
                return True
            return False
            
        except Exception as e:
            self.logger.warning(f"Entry removal failed for key {key}: {e}")
            return False
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """期限切れチェック"""
        now = datetime.now()
        expiry_time = entry.created_time + timedelta(seconds=entry.ttl_seconds)
        return now > expiry_time
    
    def _calculate_size(self, value: Any) -> int:
        """値のサイズ計算"""
        try:
            if self.config.enable_compression:
                # 圧縮サイズで計算
                import zlib
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                return len(compressed)
            else:
                # 通常のサイズ
                return len(pickle.dumps(value))
                
        except Exception:
            # フォールバック：推定サイズ
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return len(value) * 100  # 推定
            elif isinstance(value, dict):
                return len(value) * 200  # 推定
            else:
                return 1000  # デフォルト推定
    
    def _check_capacity(self, new_size_bytes: int) -> bool:
        """容量チェック"""
        current_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
        current_size_mb = current_size_bytes / (1024 * 1024)
        new_size_mb = new_size_bytes / (1024 * 1024)
        
        # サイズチェック
        if current_size_mb + new_size_mb > self.config.max_size_mb:
            return False
        
        # エントリ数チェック
        if len(self._cache) >= self.config.max_entries:
            return False
        
        return True
    
    def _evict_entries(self, required_size_bytes: int):
        """エントリ退避"""
        if self.config.strategy == CacheStrategy.LRU:
            self._evict_lru(required_size_bytes)
        elif self.config.strategy == CacheStrategy.LFU:
            self._evict_lfu(required_size_bytes)
        elif self.config.strategy == CacheStrategy.TTL:
            self._evict_ttl(required_size_bytes)
        else:
            self._evict_adaptive(required_size_bytes)
    
    def _evict_lru(self, required_size_bytes: int):
        """LRU退避"""
        freed_bytes = 0
        keys_to_remove = []
        
        # 古いアクセス順に削除
        for key, entry in self._cache.items():
            keys_to_remove.append(key)
            freed_bytes += entry.size_bytes
            
            if freed_bytes >= required_size_bytes:
                break
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats['evictions'] += 1
    
    def _evict_lfu(self, required_size_bytes: int):
        """LFU退避"""
        # アクセス回数でソート
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        
        freed_bytes = 0
        keys_to_remove = []
        
        for key, entry in sorted_entries:
            keys_to_remove.append(key)
            freed_bytes += entry.size_bytes
            
            if freed_bytes >= required_size_bytes:
                break
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats['evictions'] += 1
    
    def _evict_ttl(self, required_size_bytes: int):
        """TTL退避"""
        # 期限切れが近い順に削除
        now = datetime.now()
        
        def time_to_expire(entry):
            expiry_time = entry.created_time + timedelta(seconds=entry.ttl_seconds)
            return (expiry_time - now).total_seconds()
        
        sorted_entries = sorted(self._cache.items(), key=lambda x: time_to_expire(x[1]))
        
        freed_bytes = 0
        keys_to_remove = []
        
        for key, entry in sorted_entries:
            keys_to_remove.append(key)
            freed_bytes += entry.size_bytes
            
            if freed_bytes >= required_size_bytes:
                break
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats['evictions'] += 1
    
    def _evict_adaptive(self, required_size_bytes: int):
        """適応的退避"""
        # 複数の要素を考慮したスコアベース退避
        now = datetime.now()
        
        def calculate_eviction_score(entry):
            # アクセス頻度（低いほど高スコア）
            frequency_score = 1.0 / (entry.access_count + 1)
            
            # 最終アクセス時間（古いほど高スコア）
            time_since_access = (now - entry.last_accessed).total_seconds()
            recency_score = time_since_access / 3600  # 時間単位
            
            # サイズ（大きいほど高スコア）
            size_score = entry.size_bytes / (1024 * 1024)  # MB単位
            
            # TTL残時間（短いほど高スコア）
            expiry_time = entry.created_time + timedelta(seconds=entry.ttl_seconds)
            ttl_remaining = (expiry_time - now).total_seconds()
            ttl_score = 1.0 / (ttl_remaining / 3600 + 1)  # 時間単位
            
            # 重み付き合計
            return frequency_score * 0.3 + recency_score * 0.3 + size_score * 0.2 + ttl_score * 0.2
        
        # スコア順にソート
        sorted_entries = sorted(self._cache.items(), key=lambda x: calculate_eviction_score(x[1]), reverse=True)
        
        freed_bytes = 0
        keys_to_remove = []
        
        for key, entry in sorted_entries:
            keys_to_remove.append(key)
            freed_bytes += entry.size_bytes
            
            if freed_bytes >= required_size_bytes:
                break
        
        for key in keys_to_remove:
            self._remove_entry(key)
            self._stats['evictions'] += 1
    
    def _cleanup_expired(self) -> int:
        """期限切れエントリクリーンアップ"""
        expired_count = 0
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if self._remove_entry(key):
                expired_count += 1
        
        return expired_count
    
    def _adaptive_optimization(self):
        """適応的最適化"""
        # ヒット率に基づく戦略調整
        stats = self.get_stats()
        
        if stats.hit_rate < 0.5:
            # ヒット率が低い場合：TTLを延長
            for entry in self._cache.values():
                entry.ttl_seconds = int(entry.ttl_seconds * 1.2)
        elif stats.hit_rate > 0.9:
            # ヒット率が高い場合：TTLを短縮
            for entry in self._cache.values():
                entry.ttl_seconds = int(entry.ttl_seconds * 0.8)
    
    def _size_based_optimization(self):
        """サイズベース最適化"""
        # 大きなエントリを優先的に圧縮
        large_entries = [
            entry for entry in self._cache.values() 
            if entry.size_bytes > 1024 * 1024  # 1MB以上
        ]
        
        for entry in large_entries:
            if not entry.metadata.get('compressed', False):
                entry.value = self._compress_value(entry.value)
                entry.size_bytes = self._calculate_size(entry.value)
                entry.metadata['compressed'] = True
    
    def _compress_value(self, value: Any) -> Any:
        """値の圧縮"""
        try:
            if self.config.enable_compression:
                import zlib
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                return {'compressed_data': compressed, 'is_compressed': True}
            return value
        except Exception:
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        """値の展開"""
        try:
            if isinstance(value, dict) and value.get('is_compressed'):
                import zlib
                compressed_data = value['compressed_data']
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            return value
        except Exception:
            return value
    
    def _update_stats(self, operation: str, access_time_ms: float):
        """統計更新"""
        self._stats['total_access_time'] += access_time_ms
        self._stats['access_count'] += 1
    
    def _reset_stats(self):
        """統計リセット"""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_access_time': 0.0,
            'access_count': 0
        }
    
    def _start_cleanup_thread(self):
        """クリーンアップスレッド開始"""
        if self.config.cleanup_interval_seconds > 0:
            self._cleanup_running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        """クリーンアップワーカー"""
        while self._cleanup_running:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                if self._cleanup_running:
                    self._cleanup_expired()
            except Exception as e:
                self.logger.warning(f"Cleanup worker error: {e}")
    
    def _persist_entry(self, entry: CacheEntry):
        """エントリ永続化"""
        try:
            if not self.config.enable_persistence:
                return
            
            cache_dir = Path(self.config.persistence_path)
            entry_file = cache_dir / f"{self._hash_key(entry.key)}.pkl"
            
            with open(entry_file, 'wb') as f:
                pickle.dump(entry, f)
                
        except Exception as e:
            self.logger.warning(f"Entry persistence failed for {entry.key}: {e}")
    
    def _restore_cache(self):
        """キャッシュ復元"""
        try:
            cache_dir = Path(self.config.persistence_path)
            if not cache_dir.exists():
                return
            
            restored_count = 0
            
            for entry_file in cache_dir.glob("*.pkl"):
                try:
                    with open(entry_file, 'rb') as f:
                        entry = pickle.load(f)
                    
                    # 期限切れチェック
                    if not self._is_expired(entry):
                        self._cache[entry.key] = entry
                        self._type_caches[entry.cache_type][entry.key] = entry
                        restored_count += 1
                    else:
                        entry_file.unlink()  # 期限切れファイル削除
                        
                except Exception as e:
                    self.logger.warning(f"Entry restore failed for {entry_file}: {e}")
                    continue
            
            self.logger.info(f"Cache restored: {restored_count} entries")
            
        except Exception as e:
            self.logger.warning(f"Cache restore failed: {e}")
    
    def _remove_persisted_entry(self, key: str):
        """永続化エントリ削除"""
        try:
            cache_dir = Path(self.config.persistence_path)
            entry_file = cache_dir / f"{self._hash_key(key)}.pkl"
            
            if entry_file.exists():
                entry_file.unlink()
                
        except Exception as e:
            self.logger.warning(f"Persisted entry removal failed for {key}: {e}")
    
    def _clear_persistence(self):
        """永続化クリア"""
        try:
            cache_dir = Path(self.config.persistence_path)
            if cache_dir.exists():
                for file in cache_dir.glob("*.pkl"):
                    file.unlink()
                    
        except Exception as e:
            self.logger.warning(f"Persistence clear failed: {e}")
    
    def _hash_key(self, key: str) -> str:
        """キーハッシュ"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def generate_cache_key(
        self, 
        operation: str, 
        symbol: str, 
        params: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """キャッシュキー生成"""
        try:
            # パラメータをソート済み文字列に変換
            params_str = ""
            if params:
                sorted_params = sorted(params.items())
                params_str = json.dumps(sorted_params, sort_keys=True, default=str)
            
            # タイムスタンプ（分単位で丸める）
            if timestamp:
                rounded_time = timestamp.replace(second=0, microsecond=0)
                time_str = rounded_time.isoformat()
            else:
                time_str = ""
            
            # キー構築
            key_components = [operation, symbol, params_str, time_str]
            key_string = "|".join(filter(None, key_components))
            
            # ハッシュ化
            return hashlib.sha256(key_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Cache key generation failed: {e}")
            return f"{operation}_{symbol}_{int(time.time())}"
    
    def stop(self):
        """キャッシュマネージャー停止"""
        try:
            # クリーンアップスレッド停止
            self._cleanup_running = False
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)
            
            # 最終永続化
            if self.config.enable_persistence:
                for entry in self._cache.values():
                    self._persist_entry(entry)
            
            self.logger.info("Cache manager stopped")
            
        except Exception as e:
            self.logger.error(f"Cache manager stop failed: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報取得"""
        stats = self.get_stats()
        
        return {
            'config': {
                'strategy': self.config.strategy.value,
                'max_size_mb': self.config.max_size_mb,
                'max_entries': self.config.max_entries,
                'default_ttl_seconds': self.config.default_ttl_seconds,
                'enable_persistence': self.config.enable_persistence
            },
            'statistics': {
                'total_entries': stats.total_entries,
                'total_size_mb': stats.total_size_mb,
                'hit_rate': stats.hit_rate,
                'miss_rate': stats.miss_rate,
                'avg_access_time_ms': stats.avg_access_time_ms,
                'cache_efficiency': stats.cache_efficiency,
                'type_distribution': stats.type_distribution
            },
            'internal_stats': self._stats.copy()
        }
