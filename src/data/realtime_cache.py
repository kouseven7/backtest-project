"""
リアルタイムキャッシュシステム
フェーズ3B: ハイブリッドメモリ・ディスクキャッシュによるパフォーマンス最適化
"""

import sys
import json
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict
import pandas as pd
import hashlib

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.exception_handler import UnifiedExceptionHandler, DataError
from src.utils.error_recovery import ErrorRecoveryManager
from config.logger_config import setup_logger


@dataclass
class CacheEntry:
    """キャッシュエントリー"""
    key: str
    data: Any
    timestamp: datetime
    expiry_time: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    data_size: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp
        if self.metadata is None:
            self.metadata = {}
        if self.data_size == 0:
            self.data_size = self._calculate_size()
            
    def _calculate_size(self) -> int:
        """データサイズ計算"""
        try:
            if isinstance(self.data, pd.DataFrame):
                return self.data.memory_usage(deep=True).sum()
            elif isinstance(self.data, (dict, list)):
                return len(str(self.data).encode('utf-8'))
            else:
                return len(pickle.dumps(self.data))
        except:
            return 0
            
    def is_expired(self) -> bool:
        """期限切れチェック"""
        return datetime.now() > self.expiry_time
        
    def touch(self):
        """アクセス記録更新"""
        self.access_count += 1
        self.last_access = datetime.now()


class CacheStorageBackend(ABC):
    """キャッシュストレージバックエンドの基底クラス"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """データ取得"""
        pass
        
    @abstractmethod
    def put(self, entry: CacheEntry) -> bool:
        """データ保存"""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """データ削除"""
        pass
        
    @abstractmethod
    def clear(self) -> bool:
        """全データクリア"""
        pass
        
    @abstractmethod
    def keys(self) -> List[str]:
        """キー一覧取得"""
        pass
        
    @abstractmethod
    def size(self) -> int:
        """ストレージサイズ取得"""
        pass


class MemoryCacheBackend(CacheStorageBackend):
    """メモリキャッシュバックエンド"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.logger = setup_logger(f"{__name__}.MemoryCache")
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """データ取得"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.touch()
                    # 最近使用したアイテムを最後に移動（LRU）
                    self.cache.move_to_end(key)
                    return entry
                else:
                    # 期限切れアイテム削除
                    del self.cache[key]
            return None
            
    def put(self, entry: CacheEntry) -> bool:
        """データ保存"""
        with self.lock:
            try:
                # 容量チェック
                self._ensure_capacity(entry.data_size)
                
                # データ保存
                self.cache[entry.key] = entry
                self.cache.move_to_end(entry.key)
                
                self.logger.debug(f"Cached {entry.key} ({entry.data_size} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache {entry.key}: {e}")
                return False
                
    def delete(self, key: str) -> bool:
        """データ削除"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
            
    def clear(self) -> bool:
        """全データクリア"""
        with self.lock:
            self.cache.clear()
            return True
            
    def keys(self) -> List[str]:
        """キー一覧取得"""
        with self.lock:
            return list(self.cache.keys())
            
    def size(self) -> int:
        """メモリ使用量取得"""
        with self.lock:
            return sum(entry.data_size for entry in self.cache.values())
            
    def _ensure_capacity(self, new_item_size: int):
        """容量確保"""
        current_size = self.size()
        
        # サイズ制限チェック
        while (len(self.cache) >= self.max_size or 
               current_size + new_item_size > self.max_memory_bytes):
            if not self.cache:
                break
                
            # LRU削除
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            current_size -= oldest_entry.data_size
            self.logger.debug(f"Evicted {oldest_key} from memory cache")


class DiskCacheBackend(CacheStorageBackend):
    """ディスクキャッシュバックエンド"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.lock = threading.RLock()
        self.logger = setup_logger(f"{__name__}.DiskCache")
        
        self._init_database()
        
    def _init_database(self):
        """データベース初期化"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    timestamp TEXT,
                    expiry_time TEXT,
                    access_count INTEGER,
                    last_access TEXT,
                    data_size INTEGER,
                    metadata TEXT
                )
            """)
            conn.commit()
            conn.close()
            
    def _get_file_path(self, key: str) -> Path:
        """ファイルパス生成"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """データ取得"""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                    
                # メタデータ復元
                (db_key, file_path, timestamp_str, expiry_str, 
                 access_count, last_access_str, data_size, metadata_str) = row
                
                entry = CacheEntry(
                    key=db_key,
                    data=None,
                    timestamp=datetime.fromisoformat(timestamp_str),
                    expiry_time=datetime.fromisoformat(expiry_str),
                    access_count=access_count,
                    last_access=datetime.fromisoformat(last_access_str) if last_access_str else None,
                    data_size=data_size,
                    metadata=json.loads(metadata_str) if metadata_str else {}
                )
                
                # 期限切れチェック
                if entry.is_expired():
                    self.delete(key)
                    return None
                    
                # データファイル読み込み
                data_file = Path(file_path)
                if data_file.exists():
                    with open(data_file, 'rb') as f:
                        entry.data = pickle.load(f)
                        
                    # アクセス記録更新
                    entry.touch()
                    self._update_access_info(key, entry)
                    
                    return entry
                else:
                    # ファイルが存在しない場合はメタデータも削除
                    self.delete(key)
                    return None
                    
            except Exception as e:
                self.logger.error(f"Failed to get {key}: {e}")
                return None
                
    def put(self, entry: CacheEntry) -> bool:
        """データ保存"""
        with self.lock:
            try:
                # 容量チェック
                self._ensure_capacity(entry.data_size)
                
                # データファイル保存
                file_path = self._get_file_path(entry.key)
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.data, f)
                    
                # メタデータ保存
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, timestamp, expiry_time, access_count, 
                     last_access, data_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    str(file_path),
                    entry.timestamp.isoformat(),
                    entry.expiry_time.isoformat(),
                    entry.access_count,
                    entry.last_access.isoformat() if entry.last_access else None,
                    entry.data_size,
                    json.dumps(entry.metadata) if entry.metadata else None
                ))
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Cached {entry.key} to disk ({entry.data_size} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache {entry.key}: {e}")
                return False
                
    def delete(self, key: str) -> bool:
        """データ削除"""
        with self.lock:
            try:
                # データファイル削除
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
                    
                # メタデータ削除
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                conn.close()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete {key}: {e}")
                return False
                
    def clear(self) -> bool:
        """全データクリア"""
        with self.lock:
            try:
                # 全ファイル削除
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                    
                # メタデータクリア
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
                conn.close()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to clear cache: {e}")
                return False
                
    def keys(self) -> List[str]:
        """キー一覧取得"""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute("SELECT key FROM cache_entries")
                keys = [row[0] for row in cursor.fetchall()]
                conn.close()
                return keys
            except Exception as e:
                self.logger.error(f"Failed to get keys: {e}")
                return []
                
    def size(self) -> int:
        """ディスク使用量取得"""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute("SELECT SUM(data_size) FROM cache_entries")
                result = cursor.fetchone()[0]
                conn.close()
                return result or 0
            except Exception as e:
                self.logger.error(f"Failed to get size: {e}")
                return 0
                
    def _update_access_info(self, key: str, entry: CacheEntry):
        """アクセス情報更新"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                UPDATE cache_entries 
                SET access_count = ?, last_access = ?
                WHERE key = ?
            """, (entry.access_count, entry.last_access.isoformat(), key))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to update access info for {key}: {e}")
            
    def _ensure_capacity(self, new_item_size: int):
        """容量確保"""
        current_size = self.size()
        
        if current_size + new_item_size <= self.max_size_bytes:
            return
            
        # LRU削除
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("""
            SELECT key, data_size FROM cache_entries 
            ORDER BY last_access ASC
        """)
        
        for key, size in cursor.fetchall():
            if current_size + new_item_size <= self.max_size_bytes:
                break
                
            self.delete(key)
            current_size -= size
            self.logger.debug(f"Evicted {key} from disk cache")
            
        conn.close()


class HybridRealtimeCache:
    """ハイブリッドリアルタイムキャッシュ"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.exception_handler = UnifiedExceptionHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # キャッシュバックエンド初期化
        self.memory_cache = MemoryCacheBackend(
            max_size=config.get('memory_max_items', 100),
            max_memory_mb=config.get('memory_max_mb', 512)
        )
        
        cache_dir = config.get('disk_cache_dir', 'cache/realtime')
        self.disk_cache = DiskCacheBackend(
            cache_dir=cache_dir,
            max_size_mb=config.get('disk_max_mb', 1024)
        )
        
        # キャッシュポリシー
        self.memory_ttl = timedelta(seconds=config.get('memory_ttl_seconds', 300))  # 5分
        self.disk_ttl = timedelta(seconds=config.get('disk_ttl_seconds', 3600))     # 1時間
        self.enable_write_through = config.get('enable_write_through', True)
        
        # 統計情報
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0,
            'errors': 0
        }
        
        # クリーンアップスレッド
        self.cleanup_interval = config.get('cleanup_interval_seconds', 600)  # 10分
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        
        self._start_cleanup_thread()
        
    def get(self, key: str) -> Optional[Any]:
        """データ取得（L1→L2の順で検索）"""
        try:
            # L1キャッシュ（メモリ）から検索
            entry = self.memory_cache.get(key)
            if entry:
                self.stats['hits'] += 1
                self.stats['memory_hits'] += 1
                self.logger.debug(f"Memory cache hit: {key}")
                return entry.data
                
            # L2キャッシュ（ディスク）から検索
            entry = self.disk_cache.get(key)
            if entry:
                self.stats['hits'] += 1
                self.stats['disk_hits'] += 1
                self.logger.debug(f"Disk cache hit: {key}")
                
                # メモリキャッシュに昇格
                memory_entry = CacheEntry(
                    key=key,
                    data=entry.data,
                    timestamp=datetime.now(),
                    expiry_time=datetime.now() + self.memory_ttl,
                    access_count=entry.access_count,
                    last_access=entry.last_access,
                    metadata=entry.metadata
                )
                self.memory_cache.put(memory_entry)
                
                return entry.data
                
            # キャッシュミス
            self.stats['misses'] += 1
            self.logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'cache_get', 'key': key}
            )
            return None
            
    def put(self, key: str, data: Any, ttl: Optional[timedelta] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """データ保存"""
        try:
            now = datetime.now()
            
            # TTL設定
            memory_expiry = now + (ttl or self.memory_ttl)
            disk_expiry = now + (ttl or self.disk_ttl)
            
            # メモリキャッシュエントリー作成
            memory_entry = CacheEntry(
                key=key,
                data=data,
                timestamp=now,
                expiry_time=memory_expiry,
                metadata=metadata or {}
            )
            
            # メモリキャッシュに保存
            memory_success = self.memory_cache.put(memory_entry)
            
            # Write-throughが有効な場合はディスクにも保存
            disk_success = True
            if self.enable_write_through:
                disk_entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=now,
                    expiry_time=disk_expiry,
                    metadata=metadata or {}
                )
                disk_success = self.disk_cache.put(disk_entry)
                
            self.logger.debug(f"Cached {key} (memory: {memory_success}, disk: {disk_success})")
            return memory_success or disk_success
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'cache_put', 'key': key}
            )
            return False
            
    def delete(self, key: str) -> bool:
        """データ削除"""
        try:
            memory_deleted = self.memory_cache.delete(key)
            disk_deleted = self.disk_cache.delete(key)
            
            self.logger.debug(f"Deleted {key} (memory: {memory_deleted}, disk: {disk_deleted})")
            return memory_deleted or disk_deleted
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'cache_delete', 'key': key}
            )
            return False
            
    def clear(self) -> bool:
        """全データクリア"""
        try:
            memory_cleared = self.memory_cache.clear()
            disk_cleared = self.disk_cache.clear()
            
            # 統計リセット
            self.stats = {key: 0 for key in self.stats}
            
            self.logger.info("Cache cleared")
            return memory_cleared and disk_cleared
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'cache_clear'}
            )
            return False
            
    def keys(self) -> List[str]:
        """キー一覧取得"""
        try:
            memory_keys = set(self.memory_cache.keys())
            disk_keys = set(self.disk_cache.keys())
            return list(memory_keys.union(disk_keys))
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'cache_keys'}
            )
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_size_bytes': self.memory_cache.size(),
            'disk_size_bytes': self.disk_cache.size(),
            'memory_items': len(self.memory_cache.keys()),
            'disk_items': len(self.disk_cache.keys())
        }
        
    def _start_cleanup_thread(self):
        """クリーンアップスレッド開始"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self.cleanup_thread.start()
            self.logger.info("Cache cleanup thread started")
            
    def _cleanup_worker(self):
        """クリーンアップワーカー"""
        while not self.shutdown_event.wait(self.cleanup_interval):
            try:
                self._cleanup_expired_entries()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                
    def _cleanup_expired_entries(self):
        """期限切れエントリーのクリーンアップ"""
        try:
            # メモリキャッシュクリーンアップ
            memory_keys = self.memory_cache.keys().copy()
            for key in memory_keys:
                entry = self.memory_cache.get(key)
                if entry and entry.is_expired():
                    self.memory_cache.delete(key)
                    self.stats['evictions'] += 1
                    
            # ディスクキャッシュクリーンアップ
            disk_keys = self.disk_cache.keys().copy()
            for key in disk_keys:
                entry = self.disk_cache.get(key)
                if entry and entry.is_expired():
                    self.disk_cache.delete(key)
                    self.stats['evictions'] += 1
                    
            self.logger.debug("Completed cache cleanup")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            
    def shutdown(self):
        """キャッシュシャットダウン"""
        self.shutdown_event.set()
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        self.logger.info("Cache shutdown complete")


if __name__ == "__main__":
    # デモ実行
    import numpy as np
    
    # テスト用設定
    test_config = {
        'memory_max_items': 50,
        'memory_max_mb': 64,
        'disk_cache_dir': 'cache/test_realtime',
        'disk_max_mb': 128,
        'memory_ttl_seconds': 60,
        'disk_ttl_seconds': 300,
        'enable_write_through': True,
        'cleanup_interval_seconds': 30
    }
    
    # キャッシュ初期化
    cache = HybridRealtimeCache(test_config)
    
    try:
        # テストデータ
        test_data = {
            'AAPL_price': 150.50,
            'AAPL_data': pd.DataFrame({
                'price': np.random.randn(100),
                'volume': np.random.randint(1000, 10000, 100)
            }),
            'market_status': {'open': True, 'last_update': datetime.now().isoformat()}
        }
        
        print("=== Cache Demo ===")
        
        # データ保存テスト
        for key, data in test_data.items():
            success = cache.put(key, data, metadata={'source': 'test'})
            print(f"Cached {key}: {success}")
            
        print()
        
        # データ取得テスト
        for key in test_data.keys():
            data = cache.get(key)
            print(f"Retrieved {key}: {'✓' if data is not None else '✗'}")
            
        print()
        
        # 統計情報
        stats = cache.get_stats()
        print("Cache Statistics:")
        for stat_key, value in stats.items():
            print(f"  {stat_key}: {value}")
            
        print()
        
        # キー一覧
        keys = cache.keys()
        print(f"Cached keys: {keys}")
        
        # 削除テスト
        cache.delete('AAPL_price')
        print(f"After deletion, keys: {cache.keys()}")
        
    finally:
        # クリーンアップ
        cache.shutdown()
        print("Cache demo completed")
