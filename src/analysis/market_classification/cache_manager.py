"""
キャッシュ管理システム - A→B市場分類システム基盤
分析結果の効率的キャッシュ管理とパフォーマンス最適化を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
import hashlib
import pickle
import json
import os
import sqlite3
from pathlib import Path
import warnings
import gc

class CacheType(Enum):
    """キャッシュタイプ"""
    MEMORY = "memory"
    DISK = "disk"
    DATABASE = "database"
    HYBRID = "hybrid"

class CacheEvictionPolicy(Enum):
    """キャッシュ退避ポリシー"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"
    PRIORITY = "priority"

class CacheLevel(Enum):
    """キャッシュレベル"""
    L1_MEMORY = "l1_memory"     # 高速メモリキャッシュ
    L2_DISK = "l2_disk"         # ディスクキャッシュ
    L3_DATABASE = "l3_database" # データベースキャッシュ

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    value: Any
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0
    expiry_time: Optional[datetime] = None
    size_bytes: int = 0
    priority: int = 0
    level: CacheLevel = CacheLevel.L1_MEMORY
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.last_accessed = self.created_time
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """オブジェクトサイズ計算"""
        try:
            return len(pickle.dumps(self.value))
        except:
            return 1024  # デフォルトサイズ

    def is_expired(self) -> bool:
        """有効期限チェック"""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time

    def access(self):
        """アクセス記録"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """キャッシュ統計"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """ヒット率更新"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests

    def record_hit(self):
        """ヒット記録"""
        self.total_requests += 1
        self.cache_hits += 1
        self.update_hit_rate()

    def record_miss(self):
        """ミス記録"""
        self.total_requests += 1
        self.cache_misses += 1
        self.update_hit_rate()

    def record_eviction(self):
        """退避記録"""
        self.evictions += 1

class MultiLevelCacheManager:
    """
    多層キャッシュ管理システム
    L1(メモリ) -> L2(ディスク) -> L3(データベース) の3層構造
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 l1_max_size: int = 100 * 1024 * 1024,  # 100MB
                 l2_max_size: int = 1024 * 1024 * 1024,  # 1GB
                 l3_max_entries: int = 10000,
                 default_ttl: int = 3600,  # 1時間
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU):
        """
        多層キャッシュ管理器の初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
            l1_max_size: L1キャッシュ最大サイズ（バイト）
            l2_max_size: L2キャッシュ最大サイズ（バイト）
            l3_max_entries: L3キャッシュ最大エントリ数
            default_ttl: デフォルトTTL（秒）
            eviction_policy: 退避ポリシー
        """
        self.cache_dir = Path(cache_dir)
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.l3_max_entries = l3_max_entries
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # キャッシュストレージ
        self.l1_cache: Dict[str, CacheEntry] = {}  # メモリキャッシュ
        self.l2_cache_dir = self.cache_dir / "l2_disk"
        self.l3_db_path = self.cache_dir / "l3_cache.db"
        
        # ディレクトリ作成
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # データベース初期化
        self._init_database()
        
        # ロック
        self._cache_lock = threading.RLock()
        
        # 統計
        self.stats = CacheStats()
        
        # クリーンアップタイマー
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()
        
        self.logger.info("MultiLevelCacheManager初期化完了")

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(str(self.l3_db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        created_time TEXT,
                        last_accessed TEXT,
                        access_count INTEGER,
                        expiry_time TEXT,
                        size_bytes INTEGER,
                        priority INTEGER,
                        metadata TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed 
                    ON cache_entries(last_accessed)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expiry_time 
                    ON cache_entries(expiry_time)
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    def put(self, 
            key: str, 
            value: Any, 
            ttl: Optional[int] = None,
            priority: int = 0,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        キャッシュに値を保存
        
        Args:
            key: キャッシュキー
            value: 保存する値
            ttl: TTL（秒）
            priority: 優先度
            metadata: メタデータ
            
        Returns:
            bool: 保存成功フラグ
        """
        try:
            with self._cache_lock:
                # TTL設定
                if ttl is None:
                    ttl = self.default_ttl
                
                expiry_time = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
                
                # キャッシュエントリ作成
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_time=datetime.now(),
                    last_accessed=datetime.now(),
                    expiry_time=expiry_time,
                    priority=priority,
                    metadata=metadata or {}
                )
                
                # サイズチェックとレベル決定
                if entry.size_bytes <= self.l1_max_size // 10:  # L1に適したサイズ
                    return self._put_l1(entry)
                elif entry.size_bytes <= self.l2_max_size // 10:  # L2に適したサイズ
                    return self._put_l2(entry)
                else:  # L3に保存
                    return self._put_l3(entry)
                
        except Exception as e:
            self.logger.error(f"キャッシュ保存エラー ({key}): {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        キャッシュから値を取得
        
        Args:
            key: キャッシュキー
            
        Returns:
            Optional[Any]: 取得した値（None=キャッシュミス）
        """
        try:
            with self._cache_lock:
                # L1キャッシュから検索
                entry = self._get_l1(key)
                if entry:
                    self.stats.record_hit()
                    return entry.value
                
                # L2キャッシュから検索
                entry = self._get_l2(key)
                if entry:
                    # L1に昇格
                    self._promote_to_l1(entry)
                    self.stats.record_hit()
                    return entry.value
                
                # L3キャッシュから検索
                entry = self._get_l3(key)
                if entry:
                    # L2に昇格
                    self._promote_to_l2(entry)
                    self.stats.record_hit()
                    return entry.value
                
                # キャッシュミス
                self.stats.record_miss()
                return None
                
        except Exception as e:
            self.logger.error(f"キャッシュ取得エラー ({key}): {e}")
            self.stats.record_miss()
            return None

    def _put_l1(self, entry: CacheEntry) -> bool:
        """L1キャッシュに保存"""
        try:
            # 容量チェック
            current_size = sum(e.size_bytes for e in self.l1_cache.values())
            if current_size + entry.size_bytes > self.l1_max_size:
                self._evict_l1()
            
            entry.level = CacheLevel.L1_MEMORY
            self.l1_cache[entry.key] = entry
            return True
            
        except Exception as e:
            self.logger.error(f"L1キャッシュ保存エラー: {e}")
            return False

    def _put_l2(self, entry: CacheEntry) -> bool:
        """L2キャッシュに保存"""
        try:
            # ファイルパス
            file_path = self.l2_cache_dir / f"{self._hash_key(entry.key)}.pkl"
            
            # 容量チェック
            self._check_l2_capacity()
            
            # ファイル保存
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"L2キャッシュ保存エラー: {e}")
            return False

    def _put_l3(self, entry: CacheEntry) -> bool:
        """L3キャッシュに保存"""
        try:
            with sqlite3.connect(str(self.l3_db_path)) as conn:
                # エントリ数チェック
                count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                if count >= self.l3_max_entries:
                    self._evict_l3(conn)
                
                # データ保存
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_time, last_accessed, access_count, 
                     expiry_time, size_bytes, priority, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    pickle.dumps(entry.value),
                    entry.created_time.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry.expiry_time.isoformat() if entry.expiry_time else None,
                    entry.size_bytes,
                    entry.priority,
                    json.dumps(entry.metadata)
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"L3キャッシュ保存エラー: {e}")
            return False

    def _get_l1(self, key: str) -> Optional[CacheEntry]:
        """L1キャッシュから取得"""
        entry = self.l1_cache.get(key)
        if entry and not entry.is_expired():
            entry.access()
            return entry
        elif entry and entry.is_expired():
            # 期限切れエントリを削除
            del self.l1_cache[key]
        return None

    def _get_l2(self, key: str) -> Optional[CacheEntry]:
        """L2キャッシュから取得"""
        try:
            file_path = self.l2_cache_dir / f"{self._hash_key(key)}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                if not entry.is_expired():
                    entry.access()
                    return entry
                else:
                    # 期限切れファイルを削除
                    file_path.unlink()
            
            return None
            
        except Exception as e:
            self.logger.error(f"L2キャッシュ取得エラー: {e}")
            return None

    def _get_l3(self, key: str) -> Optional[CacheEntry]:
        """L3キャッシュから取得"""
        try:
            with sqlite3.connect(str(self.l3_db_path)) as conn:
                row = conn.execute("""
                    SELECT value, created_time, last_accessed, access_count,
                           expiry_time, size_bytes, priority, metadata
                    FROM cache_entries WHERE key = ?
                """, (key,)).fetchone()
                
                if row:
                    value, created_time, last_accessed, access_count, expiry_time, size_bytes, priority, metadata = row
                    
                    entry = CacheEntry(
                        key=key,
                        value=pickle.loads(value),
                        created_time=datetime.fromisoformat(created_time),
                        last_accessed=datetime.fromisoformat(last_accessed),
                        access_count=access_count,
                        expiry_time=datetime.fromisoformat(expiry_time) if expiry_time else None,
                        size_bytes=size_bytes,
                        priority=priority,
                        level=CacheLevel.L3_DATABASE,
                        metadata=json.loads(metadata) if metadata else {}
                    )
                    
                    if not entry.is_expired():
                        entry.access()
                        # アクセス統計更新
                        conn.execute("""
                            UPDATE cache_entries 
                            SET last_accessed = ?, access_count = ?
                            WHERE key = ?
                        """, (entry.last_accessed.isoformat(), entry.access_count, key))
                        conn.commit()
                        return entry
                    else:
                        # 期限切れエントリを削除
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                
                return None
                
        except Exception as e:
            self.logger.error(f"L3キャッシュ取得エラー: {e}")
            return None

    def _promote_to_l1(self, entry: CacheEntry):
        """エントリをL1に昇格"""
        try:
            if entry.size_bytes <= self.l1_max_size // 10:
                self._put_l1(entry)
        except Exception as e:
            self.logger.error(f"L1昇格エラー: {e}")

    def _promote_to_l2(self, entry: CacheEntry):
        """エントリをL2に昇格"""
        try:
            if entry.size_bytes <= self.l2_max_size // 10:
                self._put_l2(entry)
        except Exception as e:
            self.logger.error(f"L2昇格エラー: {e}")

    def _evict_l1(self):
        """L1キャッシュ退避"""
        try:
            if not self.l1_cache:
                return
            
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                # 最も古いアクセスのエントリを削除
                oldest_key = min(self.l1_cache.keys(), 
                               key=lambda k: self.l1_cache[k].last_accessed)
                del self.l1_cache[oldest_key]
            
            elif self.eviction_policy == CacheEvictionPolicy.LFU:
                # 最も使用頻度の低いエントリを削除
                least_used_key = min(self.l1_cache.keys(),
                                   key=lambda k: self.l1_cache[k].access_count)
                del self.l1_cache[least_used_key]
            
            elif self.eviction_policy == CacheEvictionPolicy.PRIORITY:
                # 最も優先度の低いエントリを削除
                lowest_priority_key = min(self.l1_cache.keys(),
                                        key=lambda k: self.l1_cache[k].priority)
                del self.l1_cache[lowest_priority_key]
            
            self.stats.record_eviction()
            
        except Exception as e:
            self.logger.error(f"L1退避エラー: {e}")

    def _check_l2_capacity(self):
        """L2キャッシュ容量チェック"""
        try:
            total_size = sum(f.stat().st_size for f in self.l2_cache_dir.glob("*.pkl"))
            if total_size > self.l2_max_size:
                self._evict_l2()
        except Exception as e:
            self.logger.error(f"L2容量チェックエラー: {e}")

    def _evict_l2(self):
        """L2キャッシュ退避"""
        try:
            files = list(self.l2_cache_dir.glob("*.pkl"))
            if not files:
                return
            
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                # 最も古いアクセス時刻のファイルを削除
                oldest_file = min(files, key=lambda f: f.stat().st_atime)
                oldest_file.unlink()
            else:
                # ファイルサイズで削除（大きいファイルから）
                largest_file = max(files, key=lambda f: f.stat().st_size)
                largest_file.unlink()
            
            self.stats.record_eviction()
            
        except Exception as e:
            self.logger.error(f"L2退避エラー: {e}")

    def _evict_l3(self, conn: sqlite3.Connection):
        """L3キャッシュ退避"""
        try:
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                # 最も古いアクセスのエントリを削除
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE key = (
                        SELECT key FROM cache_entries 
                        ORDER BY last_accessed ASC LIMIT 1
                    )
                """)
            elif self.eviction_policy == CacheEvictionPolicy.LFU:
                # 最も使用頻度の低いエントリを削除
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE key = (
                        SELECT key FROM cache_entries 
                        ORDER BY access_count ASC LIMIT 1
                    )
                """)
            else:
                # 最も古い作成日のエントリを削除
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE key = (
                        SELECT key FROM cache_entries 
                        ORDER BY created_time ASC LIMIT 1
                    )
                """)
            
            self.stats.record_eviction()
            
        except Exception as e:
            self.logger.error(f"L3退避エラー: {e}")

    def _hash_key(self, key: str) -> str:
        """キーハッシュ化"""
        return hashlib.md5(key.encode()).hexdigest()

    def _start_cleanup_timer(self):
        """クリーンアップタイマー開始"""
        self._cleanup_timer = threading.Timer(300, self._cleanup_expired)  # 5分間隔
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _cleanup_expired(self):
        """期限切れエントリクリーンアップ"""
        try:
            with self._cache_lock:
                # L1期限切れエントリ削除
                expired_l1_keys = [k for k, v in self.l1_cache.items() if v.is_expired()]
                for key in expired_l1_keys:
                    del self.l1_cache[key]
                
                # L2期限切れファイル削除
                for file_path in self.l2_cache_dir.glob("*.pkl"):
                    try:
                        with open(file_path, 'rb') as f:
                            entry = pickle.load(f)
                        if entry.is_expired():
                            file_path.unlink()
                    except:
                        continue
                
                # L3期限切れエントリ削除
                with sqlite3.connect(str(self.l3_db_path)) as conn:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE expiry_time < ?
                    """, (datetime.now().isoformat(),))
                    conn.commit()
            
            # 次回タイマー設定
            self._start_cleanup_timer()
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            self._start_cleanup_timer()

    def contains(self, key: str) -> bool:
        """キー存在チェック"""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """キー削除"""
        try:
            with self._cache_lock:
                deleted = False
                
                # L1から削除
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    deleted = True
                
                # L2から削除
                file_path = self.l2_cache_dir / f"{self._hash_key(key)}.pkl"
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
                
                # L3から削除
                with sqlite3.connect(str(self.l3_db_path)) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    if cursor.rowcount > 0:
                        deleted = True
                    conn.commit()
                
                return deleted
                
        except Exception as e:
            self.logger.error(f"キー削除エラー ({key}): {e}")
            return False

    def clear(self, level: Optional[CacheLevel] = None):
        """キャッシュクリア"""
        try:
            with self._cache_lock:
                if level is None or level == CacheLevel.L1_MEMORY:
                    self.l1_cache.clear()
                
                if level is None or level == CacheLevel.L2_DISK:
                    for file_path in self.l2_cache_dir.glob("*.pkl"):
                        file_path.unlink()
                
                if level is None or level == CacheLevel.L3_DATABASE:
                    with sqlite3.connect(str(self.l3_db_path)) as conn:
                        conn.execute("DELETE FROM cache_entries")
                        conn.commit()
                
                if level is None:
                    self.stats = CacheStats()
                
        except Exception as e:
            self.logger.error(f"キャッシュクリアエラー: {e}")

    def get_stats(self) -> CacheStats:
        """キャッシュ統計取得"""
        with self._cache_lock:
            # 現在のサイズ・エントリ数更新
            self.stats.entry_count = len(self.l1_cache)
            self.stats.total_size_bytes = sum(e.size_bytes for e in self.l1_cache.values())
            
            # L2のエントリ数・サイズ追加
            l2_files = list(self.l2_cache_dir.glob("*.pkl"))
            self.stats.entry_count += len(l2_files)
            self.stats.total_size_bytes += sum(f.stat().st_size for f in l2_files)
            
            # L3のエントリ数追加
            try:
                with sqlite3.connect(str(self.l3_db_path)) as conn:
                    l3_count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                    self.stats.entry_count += l3_count
            except:
                pass
            
            return self.stats

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報取得"""
        stats = self.get_stats()
        return {
            'l1_entries': len(self.l1_cache),
            'l1_size_mb': sum(e.size_bytes for e in self.l1_cache.values()) / 1024 / 1024,
            'l2_files': len(list(self.l2_cache_dir.glob("*.pkl"))),
            'l2_size_mb': sum(f.stat().st_size for f in self.l2_cache_dir.glob("*.pkl")) / 1024 / 1024,
            'hit_rate': stats.hit_rate,
            'total_requests': stats.total_requests,
            'evictions': stats.evictions,
            'eviction_policy': self.eviction_policy.value
        }

    def optimize(self):
        """キャッシュ最適化"""
        try:
            with self._cache_lock:
                # メモリ使用量最適化
                gc.collect()
                
                # L2ファイル整理
                self._defragment_l2()
                
                # L3データベース最適化
                with sqlite3.connect(str(self.l3_db_path)) as conn:
                    conn.execute("VACUUM")
                    conn.commit()
                
                self.logger.info("キャッシュ最適化完了")
                
        except Exception as e:
            self.logger.error(f"キャッシュ最適化エラー: {e}")

    def _defragment_l2(self):
        """L2キャッシュデフラグ"""
        try:
            # 使用頻度の低いファイルを特定して削除
            files = list(self.l2_cache_dir.glob("*.pkl"))
            if len(files) > 100:  # ファイル数が多い場合
                # アクセス時刻でソートして古いものを削除
                files.sort(key=lambda f: f.stat().st_atime)
                for file_path in files[:len(files)//4]:  # 25%削除
                    file_path.unlink()
        except Exception as e:
            self.logger.error(f"L2デフラグエラー: {e}")

    def __del__(self):
        """デストラクター"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

# 特殊化されたキャッシュマネージャー
class MarketDataCacheManager(MultiLevelCacheManager):
    """市場データ特化キャッシュマネージャー"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_freshness_threshold = timedelta(minutes=5)
    
    def cache_market_data(self, 
                         symbol: str, 
                         data: pd.DataFrame, 
                         data_type: str = "ohlcv") -> bool:
        """市場データキャッシュ"""
        key = f"market_data_{symbol}_{data_type}"
        metadata = {
            'symbol': symbol,
            'data_type': data_type,
            'data_shape': data.shape,
            'data_start': str(data.index[0]) if not data.empty else None,
            'data_end': str(data.index[-1]) if not data.empty else None
        }
        return self.put(key, data, ttl=300, metadata=metadata)  # 5分キャッシュ
    
    def get_market_data(self, symbol: str, data_type: str = "ohlcv") -> Optional[pd.DataFrame]:
        """市場データ取得"""
        key = f"market_data_{symbol}_{data_type}"
        return self.get(key)
    
    def is_data_fresh(self, symbol: str, data_type: str = "ohlcv") -> bool:
        """データ鮮度チェック"""
        key = f"market_data_{symbol}_{data_type}"
        entry = self._get_l1(key) or self._get_l2(key) or self._get_l3(key)
        if entry:
            age = datetime.now() - entry.created_time
            return age < self.data_freshness_threshold
        return False

class AnalysisResultCacheManager(MultiLevelCacheManager):
    """分析結果特化キャッシュマネージャー"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def cache_analysis_result(self, 
                            analysis_type: str, 
                            parameters: Dict[str, Any], 
                            result: Any,
                            symbol: Optional[str] = None) -> bool:
        """分析結果キャッシュ"""
        param_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()[:8]
        key = f"analysis_{analysis_type}_{symbol or 'global'}_{param_hash}"
        
        metadata = {
            'analysis_type': analysis_type,
            'symbol': symbol,
            'parameters': parameters,
            'result_type': type(result).__name__
        }
        
        # 分析結果は比較的長時間キャッシュ
        ttl = 1800 if symbol else 3600  # シンボル固有30分、グローバル1時間
        return self.put(key, result, ttl=ttl, priority=1, metadata=metadata)
    
    def get_analysis_result(self, 
                          analysis_type: str, 
                          parameters: Dict[str, Any],
                          symbol: Optional[str] = None) -> Optional[Any]:
        """分析結果取得"""
        param_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()[:8]
        key = f"analysis_{analysis_type}_{symbol or 'global'}_{param_hash}"
        return self.get(key)

# ファクトリー関数
def create_cache_manager(cache_type: CacheType = CacheType.HYBRID, **kwargs) -> MultiLevelCacheManager:
    """キャッシュマネージャー作成"""
    if cache_type == CacheType.MEMORY:
        return MultiLevelCacheManager(l2_max_size=0, l3_max_entries=0, **kwargs)
    elif cache_type == CacheType.DISK:
        return MultiLevelCacheManager(l1_max_size=0, l3_max_entries=0, **kwargs)
    elif cache_type == CacheType.DATABASE:
        return MultiLevelCacheManager(l1_max_size=0, l2_max_size=0, **kwargs)
    else:  # HYBRID
        return MultiLevelCacheManager(**kwargs)

if __name__ == "__main__":
    # テスト用コード
    import tempfile
    import shutil
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== キャッシュ管理システム テスト ===")
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = MultiLevelCacheManager(
            cache_dir=temp_dir,
            l1_max_size=1024,  # 1KB
            l2_max_size=10240,  # 10KB
            l3_max_entries=10
        )
        
        print("\n1. 基本キャッシュ操作テスト")
        # 小さなデータ（L1キャッシュ）
        cache_manager.put("small_data", {"value": 123}, ttl=60)
        result = cache_manager.get("small_data")
        print(f"小データ取得: {result}")
        
        # 中サイズデータ（L2キャッシュ）
        large_data = {"data": list(range(1000))}
        cache_manager.put("medium_data", large_data, ttl=60)
        result = cache_manager.get("medium_data")
        print(f"中データ取得成功: {result is not None}")
        
        print("\n2. キャッシュ統計")
        stats = cache_manager.get_stats()
        print(f"ヒット率: {stats.hit_rate:.3f}")
        print(f"総リクエスト: {stats.total_requests}")
        print(f"エントリ数: {stats.entry_count}")
        
        print("\n3. キャッシュ情報")
        info = cache_manager.get_cache_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n4. 特化キャッシュマネージャーテスト")
        market_cache = MarketDataCacheManager(cache_dir=f"{temp_dir}/market")
        
        # サンプル市場データ
        dates = pd.date_range('2024-01-01', periods=10)
        sample_data = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)
        
        market_cache.cache_market_data("AAPL", sample_data)
        retrieved_data = market_cache.get_market_data("AAPL")
        print(f"市場データキャッシュ成功: {retrieved_data is not None}")
        
        print("\n5. 分析結果キャッシュテスト")
        analysis_cache = AnalysisResultCacheManager(cache_dir=f"{temp_dir}/analysis")
        
        analysis_params = {"window": 20, "method": "sma"}
        analysis_result = {"sma_value": 105.5, "signal": "BUY"}
        
        analysis_cache.cache_analysis_result("moving_average", analysis_params, analysis_result, "AAPL")
        retrieved_result = analysis_cache.get_analysis_result("moving_average", analysis_params, "AAPL")
        print(f"分析結果キャッシュ成功: {retrieved_result == analysis_result}")
        
        print("\n=== テスト完了 ===")
