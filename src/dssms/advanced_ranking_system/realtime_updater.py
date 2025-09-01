"""
DSSMS Phase 3 Task 3.1: Realtime Updater
リアルタイム更新クラス

高度ランキングシステムのリアルタイム更新機能を提供します。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import time
import json
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

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

class UpdateType(Enum):
    """更新タイプ定義"""
    MARKET_DATA = "market_data"
    RANKING_SCORES = "ranking_scores"
    WEIGHTS = "weights"
    CONFIGURATION = "configuration"
    ALERT = "alert"

class UpdatePriority(Enum):
    """更新優先度定義"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class UpdateEvent:
    """更新イベント"""
    event_id: str
    update_type: UpdateType
    priority: UpdatePriority
    data: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    retry_count: int = 0

@dataclass
class UpdateConfig:
    """更新設定"""
    enable_realtime: bool = True
    update_interval_seconds: int = 30
    batch_size: int = 100
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_async_processing: bool = True
    max_workers: int = 4
    queue_max_size: int = 10000

@dataclass
class UpdateStatus:
    """更新状態"""
    is_running: bool
    last_update_time: datetime
    total_updates_processed: int
    failed_updates: int
    queue_size: int
    average_processing_time: float

class RealtimeUpdater:
    """
    リアルタイム更新クラス
    
    機能:
    - 非同期更新処理
    - 優先度ベースの更新キュー
    - バッチ処理とリアルタイム処理の統合
    - エラー処理とリトライ機能
    - パフォーマンス監視
    - 更新イベントの通知
    """
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        """
        初期化
        
        Args:
            config: 更新設定
        """
        self.config = config or UpdateConfig()
        self.logger = logger
        
        # 更新キュー（優先度別）
        self._update_queues: Dict[UpdatePriority, Queue] = {
            priority: Queue(maxsize=self.config.queue_max_size)
            for priority in UpdatePriority
        }
        
        # 更新状態
        self._is_running = False
        self._last_update_time = None
        self._update_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'total_processing_time': 0.0,
            'type_stats': {update_type: 0 for update_type in UpdateType}
        }
        
        # ワーカースレッド
        self._worker_threads = []
        self._update_thread = None
        
        # 非同期実行器
        self._executor = None
        if self.config.enable_async_processing:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # イベントハンドラ
        self._event_handlers: Dict[UpdateType, List[Callable[[UpdateEvent], None]]] = {
            update_type: [] for update_type in UpdateType
        }
        
        # 統計ロック
        self._stats_lock = threading.Lock()
        
        self.logger.info("Realtime Updater initialized")
    
    def start(self):
        """更新処理開始"""
        try:
            if not self._is_running:
                self._is_running = True
                
                # メイン更新スレッド開始
                self._update_thread = threading.Thread(
                    target=self._update_worker,
                    daemon=True
                )
                self._update_thread.start()
                
                # 優先度別ワーカースレッド開始
                for priority in UpdatePriority:
                    worker_thread = threading.Thread(
                        target=self._priority_worker,
                        args=(priority,),
                        daemon=True
                    )
                    worker_thread.start()
                    self._worker_threads.append(worker_thread)
                
                self.logger.info("Realtime updater started")
            
        except Exception as e:
            self.logger.error(f"Failed to start realtime updater: {e}")
            raise
    
    def stop(self):
        """更新処理停止"""
        try:
            self._is_running = False
            
            # スレッド終了待機
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=5)
            
            for worker_thread in self._worker_threads:
                if worker_thread.is_alive():
                    worker_thread.join(timeout=5)
            
            # 実行器停止
            if self._executor:
                self._executor.shutdown(wait=True)
            
            self.logger.info("Realtime updater stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop realtime updater: {e}")
    
    def schedule_update(
        self,
        update_type: UpdateType,
        data: Any,
        priority: UpdatePriority = UpdatePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        更新スケジュール
        
        Args:
            update_type: 更新タイプ
            data: 更新データ
            priority: 優先度
            metadata: メタデータ
            
        Returns:
            イベントID
        """
        try:
            # イベント作成
            event_id = f"{update_type.value}_{int(time.time() * 1000)}"
            event = UpdateEvent(
                event_id=event_id,
                update_type=update_type,
                priority=priority,
                data=data,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # キューに追加
            queue = self._update_queues[priority]
            
            try:
                queue.put_nowait(event)
                self.logger.debug(f"Update scheduled: {event_id} ({update_type.value}, {priority.value})")
                return event_id
            except:
                # キューが満杯の場合、低優先度のイベントを削除
                if priority in [UpdatePriority.HIGH, UpdatePriority.CRITICAL]:
                    self._make_room_in_queue(priority)
                    queue.put_nowait(event)
                    return event_id
                else:
                    self.logger.warning(f"Update queue full, dropping event: {event_id}")
                    return ""
            
        except Exception as e:
            self.logger.error(f"Failed to schedule update: {e}")
            return ""
    
    def schedule_batch_update(
        self,
        updates: List[Tuple[UpdateType, Any, UpdatePriority]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        バッチ更新スケジュール
        
        Args:
            updates: 更新リスト（タイプ、データ、優先度）
            metadata: メタデータ
            
        Returns:
            イベントIDリスト
        """
        event_ids = []
        
        for update_type, data, priority in updates:
            event_id = self.schedule_update(update_type, data, priority, metadata)
            if event_id:
                event_ids.append(event_id)
        
        self.logger.info(f"Batch update scheduled: {len(event_ids)} events")
        return event_ids
    
    def schedule_periodic_update(
        self,
        update_type: UpdateType,
        data_provider: Callable[[], Any],
        interval_seconds: int,
        priority: UpdatePriority = UpdatePriority.NORMAL
    ):
        """
        定期更新スケジュール
        
        Args:
            update_type: 更新タイプ
            data_provider: データプロバイダ関数
            interval_seconds: 更新間隔（秒）
            priority: 優先度
        """
        def periodic_worker():
            while self._is_running:
                try:
                    data = data_provider()
                    self.schedule_update(
                        update_type,
                        data,
                        priority,
                        {"periodic": True, "interval": interval_seconds}
                    )
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.warning(f"Periodic update failed: {e}")
                    time.sleep(interval_seconds)
        
        periodic_thread = threading.Thread(target=periodic_worker, daemon=True)
        periodic_thread.start()
        
        self.logger.info(f"Periodic update scheduled: {update_type.value} every {interval_seconds}s")
    
    def add_event_handler(self, update_type: UpdateType, handler: Callable[[UpdateEvent], None]):
        """
        イベントハンドラ追加
        
        Args:
            update_type: 更新タイプ
            handler: ハンドラ関数
        """
        self._event_handlers[update_type].append(handler)
        self.logger.info(f"Event handler added for {update_type.value}")
    
    def remove_event_handler(self, update_type: UpdateType, handler: Callable[[UpdateEvent], None]):
        """
        イベントハンドラ削除
        
        Args:
            update_type: 更新タイプ
            handler: ハンドラ関数
        """
        try:
            self._event_handlers[update_type].remove(handler)
            self.logger.info(f"Event handler removed for {update_type.value}")
        except ValueError:
            self.logger.warning(f"Handler not found for {update_type.value}")
    
    async def process_update_async(self, event: UpdateEvent) -> bool:
        """
        非同期更新処理
        
        Args:
            event: 更新イベント
            
        Returns:
            成功フラグ
        """
        try:
            start_time = time.time()
            
            # イベントハンドラ実行
            handlers = self._event_handlers.get(event.update_type, [])
            
            if self.config.enable_async_processing and handlers:
                # 非同期実行
                loop = asyncio.get_event_loop()
                tasks = []
                
                for handler in handlers:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        task = loop.run_in_executor(self._executor, handler, event)
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # 同期実行
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.warning(f"Handler failed for {event.event_id}: {e}")
            
            # 統計更新
            processing_time = time.time() - start_time
            self._update_processing_stats(event, processing_time, success=True)
            
            event.processed = True
            return True
            
        except Exception as e:
            self.logger.error(f"Update processing failed for {event.event_id}: {e}")
            self._update_processing_stats(event, 0, success=False)
            return False
    
    def _update_worker(self):
        """メイン更新ワーカー"""
        while self._is_running:
            try:
                # 定期的な統計更新
                self._update_periodic_stats()
                
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                self.logger.warning(f"Update worker error: {e}")
                time.sleep(self.config.update_interval_seconds)
    
    def _priority_worker(self, priority: UpdatePriority):
        """優先度別ワーカー"""
        queue = self._update_queues[priority]
        
        while self._is_running:
            try:
                # タイムアウト付きでイベント取得
                timeout = 1.0 if priority == UpdatePriority.CRITICAL else 5.0
                event = queue.get(timeout=timeout)
                
                # 処理実行
                success = self._process_update_sync(event)
                
                # リトライ処理
                if not success and event.retry_count < self.config.max_retry_attempts:
                    event.retry_count += 1
                    time.sleep(self.config.retry_delay_seconds)
                    queue.put(event)
                    self.logger.info(f"Retrying event {event.event_id} (attempt {event.retry_count})")
                
                queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.warning(f"Priority worker error ({priority.value}): {e}")
                continue
    
    def _process_update_sync(self, event: UpdateEvent) -> bool:
        """同期更新処理"""
        try:
            start_time = time.time()
            
            # イベントハンドラ実行
            handlers = self._event_handlers.get(event.update_type, [])
            
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.warning(f"Handler failed for {event.event_id}: {e}")
                    return False
            
            # 統計更新
            processing_time = time.time() - start_time
            self._update_processing_stats(event, processing_time, success=True)
            
            event.processed = True
            return True
            
        except Exception as e:
            self.logger.error(f"Sync update processing failed for {event.event_id}: {e}")
            self._update_processing_stats(event, 0, success=False)
            return False
    
    def _update_processing_stats(self, event: UpdateEvent, processing_time: float, success: bool):
        """処理統計更新"""
        with self._stats_lock:
            if success:
                self._update_stats['total_processed'] += 1
                self._update_stats['total_processing_time'] += processing_time
            else:
                self._update_stats['total_failed'] += 1
            
            self._update_stats['type_stats'][event.update_type] += 1
            self._last_update_time = datetime.now()
    
    def _update_periodic_stats(self):
        """定期統計更新"""
        try:
            # キューサイズ統計
            total_queue_size = sum(queue.qsize() for queue in self._update_queues.values())
            
            # ログ出力
            if total_queue_size > self.config.queue_max_size * 0.8:
                self.logger.warning(f"Update queue size high: {total_queue_size}")
            
        except Exception as e:
            self.logger.warning(f"Periodic stats update failed: {e}")
    
    def _make_room_in_queue(self, target_priority: UpdatePriority):
        """キューに空きを作る"""
        try:
            # 低優先度のキューから古いイベントを削除
            priorities_to_clean = [p for p in UpdatePriority if p.value < target_priority.value]
            
            for priority in sorted(priorities_to_clean, key=lambda x: x.value):
                queue = self._update_queues[priority]
                if not queue.empty():
                    try:
                        dropped_event = queue.get_nowait()
                        self.logger.warning(f"Dropped low priority event: {dropped_event.event_id}")
                        return
                    except Empty:
                        continue
                        
        except Exception as e:
            self.logger.warning(f"Failed to make room in queue: {e}")
    
    def get_status(self) -> UpdateStatus:
        """更新状態取得"""
        with self._stats_lock:
            total_processed = self._update_stats['total_processed']
            total_processing_time = self._update_stats['total_processing_time']
            
            avg_processing_time = (
                total_processing_time / total_processed 
                if total_processed > 0 else 0.0
            )
            
            total_queue_size = sum(queue.qsize() for queue in self._update_queues.values())
            
            return UpdateStatus(
                is_running=self._is_running,
                last_update_time=self._last_update_time or datetime.now(),
                total_updates_processed=total_processed,
                failed_updates=self._update_stats['total_failed'],
                queue_size=total_queue_size,
                average_processing_time=avg_processing_time
            )
    
    def get_queue_status(self) -> Dict[str, int]:
        """キュー状態取得"""
        return {
            priority.name: queue.qsize()
            for priority, queue in self._update_queues.items()
        }
    
    def get_type_statistics(self) -> Dict[str, int]:
        """タイプ別統計取得"""
        with self._stats_lock:
            return {
                update_type.value: count
                for update_type, count in self._update_stats['type_stats'].items()
            }
    
    def clear_queues(self, priority: Optional[UpdatePriority] = None):
        """キュークリア"""
        try:
            if priority:
                # 特定優先度のキューをクリア
                queue = self._update_queues[priority]
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
                self.logger.info(f"Cleared queue for priority: {priority.name}")
            else:
                # 全キュークリア
                for queue in self._update_queues.values():
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except Empty:
                            break
                self.logger.info("Cleared all update queues")
                
        except Exception as e:
            self.logger.error(f"Failed to clear queues: {e}")
    
    def reset_statistics(self):
        """統計リセット"""
        with self._stats_lock:
            self._update_stats = {
                'total_processed': 0,
                'total_failed': 0,
                'total_processing_time': 0.0,
                'type_stats': {update_type: 0 for update_type in UpdateType}
            }
            self._last_update_time = None
            
        self.logger.info("Update statistics reset")
    
    def export_statistics(self, file_path: str):
        """統計エクスポート"""
        try:
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'status': self.get_status().__dict__,
                'queue_status': self.get_queue_status(),
                'type_statistics': self.get_type_statistics(),
                'config': {
                    'update_interval_seconds': self.config.update_interval_seconds,
                    'batch_size': self.config.batch_size,
                    'max_workers': self.config.max_workers,
                    'queue_max_size': self.config.queue_max_size
                }
            }
            
            # JSONファイルとして保存
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, default=str)
            
            self.logger.info(f"Statistics exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export statistics: {e}")
            raise
    
    def force_process_queue(self, priority: UpdatePriority, max_items: int = 100):
        """キューの強制処理"""
        try:
            queue = self._update_queues[priority]
            processed_count = 0
            
            while not queue.empty() and processed_count < max_items:
                try:
                    event = queue.get_nowait()
                    self._process_update_sync(event)
                    processed_count += 1
                except Empty:
                    break
                except Exception as e:
                    self.logger.warning(f"Force processing failed: {e}")
                    continue
            
            self.logger.info(f"Force processed {processed_count} items from {priority.name} queue")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Failed to force process queue: {e}")
            return 0
    
    def update_config(self, new_config: UpdateConfig):
        """設定更新"""
        try:
            old_config = self.config
            self.config = new_config
            
            # 実行器の再設定
            if (old_config.enable_async_processing != new_config.enable_async_processing or
                old_config.max_workers != new_config.max_workers):
                
                if self._executor:
                    self._executor.shutdown(wait=True)
                
                if new_config.enable_async_processing:
                    self._executor = ThreadPoolExecutor(max_workers=new_config.max_workers)
                else:
                    self._executor = None
            
            self.logger.info("Realtime updater config updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            raise
