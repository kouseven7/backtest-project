"""
DSSMS Phase 2 Task 2.2: ランキングパフォーマンス最適化器
ハイブリッドキャッシュ管理とパフォーマンス最適化

機能:
- ハイブリッドキャッシュ管理
- メモリ最適化
- パフォーマンス監視
- 自動調整システム
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
import asyncio
import threading
import time
import psutil
import gc
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from config.logger_config import setup_logger

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    data: Any
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    size_bytes: int = 0
    priority: float = 1.0

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_requests, 1)

class RankingPerformanceOptimizer:
    """
    ランキングパフォーマンス最適化器
    
    機能:
    - ハイブリッドキャッシュ（LRU + Priority + TTL）
    - メモリ使用量監視と自動調整
    - パフォーマンス指標追跡
    - 自動最適化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初期化"""
        self.logger = setup_logger('dssms.performance_optimizer')
        self.config = config
        
        # キャッシュ設定
        self.max_cache_size = config.get('max_cache_size', 1000)
        self.max_memory_mb = config.get('max_memory_mb', 512)
        self.ttl_minutes = config.get('default_ttl_minutes', 15)
        self.cleanup_interval_minutes = config.get('cleanup_interval_minutes', 5)
        
        # ハイブリッドキャッシュ
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # パフォーマンス監視
        self.metrics = PerformanceMetrics()
        self._performance_history: List[Dict[str, Any]] = []
        
        # 自動最適化設定
        self.auto_optimization_enabled = config.get('auto_optimization', True)
        self.optimization_threshold = config.get('optimization_threshold', {
            'memory_usage_percent': 80,
            'cache_hit_rate_min': 0.7,
            'response_time_max_ms': 1000
        })
        
        # バックグラウンドタスク
        self._cleanup_task = None
        self._monitoring_task = None
        self._shutdown_event = asyncio.Event()
        
        # 最適化状態
        self._last_optimization = datetime.now()
        self._optimization_history: List[Dict[str, Any]] = []
        
        self.logger.info("RankingPerformanceOptimizer initialized")
        
        # バックグラウンドタスク開始
        asyncio.create_task(self._start_background_tasks())
    
    async def _start_background_tasks(self):
        """バックグラウンドタスク開始"""
        try:
            self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
            self.logger.info("バックグラウンドタスク開始")
        except Exception as e:
            self.logger.error(f"バックグラウンドタスク開始エラー: {e}")
    
    async def get_cached_result(self, key: str) -> Optional[Any]:
        """キャッシュから結果取得"""
        start_time = time.time()
        
        try:
            with self._cache_lock:
                self.metrics.total_requests += 1
                
                if key in self._cache:
                    entry = self._cache[key]
                    
                    # TTLチェック
                    if self._is_entry_valid(entry):
                        # アクセス情報更新
                        entry.access_count += 1
                        entry.last_access = datetime.now()
                        
                        # LRUのため最新に移動
                        self._cache.move_to_end(key)
                        
                        self.metrics.cache_hits += 1
                        
                        # レスポンス時間更新
                        response_time = (time.time() - start_time) * 1000
                        self._update_response_time(response_time)
                        
                        return entry.data
                    else:
                        # 期限切れエントリ削除
                        del self._cache[key]
                
                self.metrics.cache_misses += 1
                return None
                
        except Exception as e:
            self.logger.error(f"キャッシュ取得エラー ({key}): {e}")
            return None
    
    async def cache_result(self, key: str, data: Any, priority: float = 1.0, 
                          ttl_minutes: Optional[int] = None) -> bool:
        """結果をキャッシュ"""
        try:
            if ttl_minutes is None:
                ttl_minutes = self.ttl_minutes
            
            # データサイズ計算
            data_size = self._calculate_data_size(data)
            
            entry = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                size_bytes=data_size,
                priority=priority,
                last_access=datetime.now()
            )
            
            with self._cache_lock:
                # キャッシュサイズチェック
                if len(self._cache) >= self.max_cache_size:
                    await self._evict_entries()
                
                # メモリ使用量チェック
                if self._get_memory_usage_mb() > self.max_memory_mb:
                    await self._memory_pressure_cleanup()
                
                self._cache[key] = entry
                self._cache.move_to_end(key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"キャッシュ保存エラー ({key}): {e}")
            return False
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """エントリ有効性チェック"""
        age = datetime.now() - entry.timestamp
        return age.total_seconds() < (self.ttl_minutes * 60)
    
    def _calculate_data_size(self, data: Any) -> int:
        """データサイズ計算（概算）"""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, dict):
                return sum(self._calculate_data_size(k) + self._calculate_data_size(v) 
                          for k, v in data.items())
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_data_size(item) for item in data)
            elif hasattr(data, '__sizeof__'):
                return data.__sizeof__()
            else:
                return 100  # デフォルト推定サイズ
        except:
            return 100
    
    async def _evict_entries(self):
        """エントリ退避（ハイブリッド戦略）"""
        try:
            if not self._cache:
                return
            
            # 退避候補選定（優先度 + アクセス頻度 + 年齢）
            eviction_candidates = []
            
            for key, entry in self._cache.items():
                age_minutes = (datetime.now() - entry.timestamp).total_seconds() / 60
                access_frequency = entry.access_count / max(age_minutes, 1)
                
                # 退避スコア計算（低いほど退避候補）
                eviction_score = (
                    entry.priority * 0.4 +
                    access_frequency * 0.4 +
                    (1 / max(age_minutes, 1)) * 0.2
                )
                
                eviction_candidates.append((key, eviction_score))
            
            # スコア順にソート
            eviction_candidates.sort(key=lambda x: x[1])
            
            # 下位25%を退避
            num_to_evict = max(1, len(eviction_candidates) // 4)
            
            for i in range(num_to_evict):
                key_to_evict = eviction_candidates[i][0]
                if key_to_evict in self._cache:
                    del self._cache[key_to_evict]
            
            self.logger.info(f"キャッシュエントリ退避: {num_to_evict}件")
            
        except Exception as e:
            self.logger.error(f"エントリ退避エラー: {e}")
    
    async def _memory_pressure_cleanup(self):
        """メモリ圧迫時クリーンアップ"""
        try:
            self.logger.warning("メモリ圧迫によるクリーンアップ実行")
            
            # 期限切れエントリ削除
            expired_keys = []
            for key, entry in self._cache.items():
                if not self._is_entry_valid(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            # 大きなエントリから削除
            if self._get_memory_usage_mb() > self.max_memory_mb:
                large_entries = sorted(self._cache.items(), 
                                     key=lambda x: x[1].size_bytes, reverse=True)
                
                removed_count = 0
                for key, entry in large_entries:
                    if self._get_memory_usage_mb() <= self.max_memory_mb * 0.8:
                        break
                    del self._cache[key]
                    removed_count += 1
                
                self.logger.info(f"大容量エントリ削除: {removed_count}件")
            
            # ガベージコレクション
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"メモリ圧迫クリーンアップエラー: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """メモリ使用量取得（MB）"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except:
            return 0.0
    
    def _update_response_time(self, response_time_ms: float):
        """レスポンス時間更新"""
        if self.metrics.average_response_time_ms == 0:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            # 移動平均
            self.metrics.average_response_time_ms = (
                self.metrics.average_response_time_ms * 0.9 + 
                response_time_ms * 0.1
            )
    
    async def _cache_cleanup_loop(self):
        """キャッシュクリーンアップループ"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self._periodic_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"キャッシュクリーンアップエラー: {e}")
    
    async def _periodic_cleanup(self):
        """定期クリーンアップ"""
        try:
            with self._cache_lock:
                # 期限切れエントリ削除
                expired_keys = []
                for key, entry in self._cache.items():
                    if not self._is_entry_valid(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    self.logger.info(f"期限切れエントリ削除: {len(expired_keys)}件")
            
            # 自動最適化チェック
            if self.auto_optimization_enabled:
                await self._check_auto_optimization()
                
        except Exception as e:
            self.logger.error(f"定期クリーンアップエラー: {e}")
    
    async def _performance_monitoring_loop(self):
        """パフォーマンス監視ループ"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # 1分間隔
                await self._collect_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"パフォーマンス監視エラー: {e}")
    
    async def _collect_performance_metrics(self):
        """パフォーマンス指標収集"""
        try:
            # システム指標更新
            self.metrics.memory_usage_mb = self._get_memory_usage_mb()
            
            try:
                self.metrics.cpu_usage_percent = psutil.cpu_percent()
            except:
                self.metrics.cpu_usage_percent = 0.0
            
            # エラー率計算
            total_requests = max(self.metrics.total_requests, 1)
            self.metrics.error_rate = (total_requests - self.metrics.cache_hits - self.metrics.cache_misses) / total_requests
            
            # 履歴記録
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'average_response_time_ms': self.metrics.average_response_time_ms,
                'cache_size': len(self._cache),
                'error_rate': self.metrics.error_rate
            }
            
            self._performance_history.append(performance_record)
            
            # 履歴サイズ制限
            max_history = self.config.get('max_performance_history', 1440)  # 24時間分
            if len(self._performance_history) > max_history:
                self._performance_history = self._performance_history[-max_history:]
            
        except Exception as e:
            self.logger.error(f"パフォーマンス指標収集エラー: {e}")
    
    async def _check_auto_optimization(self):
        """自動最適化チェック"""
        try:
            thresholds = self.optimization_threshold
            optimization_needed = False
            reasons = []
            
            # メモリ使用量チェック
            memory_percent = (self.metrics.memory_usage_mb / self.max_memory_mb) * 100
            if memory_percent > thresholds.get('memory_usage_percent', 80):
                optimization_needed = True
                reasons.append(f"メモリ使用率高: {memory_percent:.1f}%")
            
            # キャッシュヒット率チェック
            if self.metrics.cache_hit_rate < thresholds.get('cache_hit_rate_min', 0.7):
                optimization_needed = True
                reasons.append(f"キャッシュヒット率低: {self.metrics.cache_hit_rate:.3f}")
            
            # レスポンス時間チェック
            if self.metrics.average_response_time_ms > thresholds.get('response_time_max_ms', 1000):
                optimization_needed = True
                reasons.append(f"レスポンス時間長: {self.metrics.average_response_time_ms:.1f}ms")
            
            if optimization_needed:
                await self._execute_auto_optimization(reasons)
                
        except Exception as e:
            self.logger.error(f"自動最適化チェックエラー: {e}")
    
    async def _execute_auto_optimization(self, reasons: List[str]):
        """自動最適化実行"""
        try:
            self.logger.info(f"自動最適化実行: {', '.join(reasons)}")
            
            optimization_actions = []
            
            # メモリ最適化
            if "メモリ使用率高" in ' '.join(reasons):
                await self._memory_pressure_cleanup()
                optimization_actions.append("メモリクリーンアップ")
            
            # キャッシュサイズ調整
            if "キャッシュヒット率低" in ' '.join(reasons):
                new_cache_size = min(self.max_cache_size * 1.2, 2000)
                self.max_cache_size = int(new_cache_size)
                optimization_actions.append(f"キャッシュサイズ拡張: {new_cache_size}")
            
            # TTL調整
            if "レスポンス時間長" in ' '.join(reasons):
                self.ttl_minutes = min(self.ttl_minutes * 1.5, 60)
                optimization_actions.append(f"TTL延長: {self.ttl_minutes}分")
            
            # 最適化履歴記録
            optimization_record = {
                'timestamp': datetime.now().isoformat(),
                'reasons': reasons,
                'actions': optimization_actions,
                'metrics_before': {
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'response_time_ms': self.metrics.average_response_time_ms
                }
            }
            
            self._optimization_history.append(optimization_record)
            self._last_optimization = datetime.now()
            
            self.logger.info(f"自動最適化完了: {optimization_actions}")
            
        except Exception as e:
            self.logger.error(f"自動最適化実行エラー: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標取得"""
        return {
            'total_requests': self.metrics.total_requests,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'average_response_time_ms': self.metrics.average_response_time_ms,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'cpu_usage_percent': self.metrics.cpu_usage_percent,
            'cache_size': len(self._cache),
            'error_rate': self.metrics.error_rate,
            'last_optimization': self._last_optimization.isoformat() if self._last_optimization else None
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計情報"""
        try:
            with self._cache_lock:
                if not self._cache:
                    return {'cache_size': 0, 'total_size_mb': 0}
                
                total_size = sum(entry.size_bytes for entry in self._cache.values())
                total_accesses = sum(entry.access_count for entry in self._cache.values())
                avg_age_minutes = sum(
                    (datetime.now() - entry.timestamp).total_seconds() / 60 
                    for entry in self._cache.values()
                ) / len(self._cache)
                
                return {
                    'cache_size': len(self._cache),
                    'total_size_mb': total_size / 1024 / 1024,
                    'average_entry_size_kb': (total_size / len(self._cache)) / 1024,
                    'total_accesses': total_accesses,
                    'average_accesses_per_entry': total_accesses / len(self._cache),
                    'average_age_minutes': avg_age_minutes,
                    'max_cache_size': self.max_cache_size,
                    'ttl_minutes': self.ttl_minutes
                }
        except Exception as e:
            self.logger.error(f"キャッシュ統計取得エラー: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """キャッシュクリア"""
        try:
            with self._cache_lock:
                self._cache.clear()
            self.logger.info("キャッシュをクリアしました")
        except Exception as e:
            self.logger.error(f"キャッシュクリアエラー: {e}")
    
    def reset_metrics(self):
        """メトリクスリセット"""
        self.metrics = PerformanceMetrics()
        self._performance_history.clear()
        self._optimization_history.clear()
        self.logger.info("パフォーマンスメトリクスをリセットしました")
    
    async def shutdown(self):
        """シャットダウン"""
        try:
            self.logger.info("パフォーマンス最適化器シャットダウン開始")
            
            # バックグラウンドタスク停止
            self._shutdown_event.set()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # キャッシュクリア
            self.clear_cache()
            
            self.logger.info("パフォーマンス最適化器シャットダウン完了")
            
        except Exception as e:
            self.logger.error(f"シャットダウンエラー: {e}")
    
    def export_performance_data(self, file_path: str):
        """パフォーマンスデータエクスポート"""
        try:
            export_data = {
                'metrics': self.get_performance_metrics(),
                'cache_statistics': self.get_cache_statistics(),
                'performance_history': self._performance_history,
                'optimization_history': self._optimization_history,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"パフォーマンスデータエクスポート完了: {file_path}")
            
        except Exception as e:
            self.logger.error(f"パフォーマンスデータエクスポートエラー: {e}")
