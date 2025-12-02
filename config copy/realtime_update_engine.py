"""
Module: Realtime Update Engine
File: realtime_update_engine.py
Description:
  リアルタイムスコア更新エンジン
  ScoreUpdateTriggerSystemと連携して高頻度更新を処理

Author: GitHub Copilot  
Created: 2025-07-13
Modified: 2025-07-13
"""

import asyncio
import logging
import threading
import time
import json
import queue
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# 既存モジュールのインポート
try:
    from .score_update_trigger_system import (
        ScoreUpdateTriggerSystem, TriggerEvent, UpdateRequest, UpdateResult,
        TriggerType, TriggerPriority
    )
    from .enhanced_score_history_manager import EnhancedScoreHistoryManager
    from .strategy_scoring_model import StrategyScore, StrategyScoreCalculator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerEvent, UpdateRequest, UpdateResult,
            TriggerType, TriggerPriority
        )
        from enhanced_score_history_manager import EnhancedScoreHistoryManager
        from strategy_scoring_model import StrategyScore, StrategyScoreCalculator
    except ImportError:
        # さらに上の階層から
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerEvent, UpdateRequest, UpdateResult,
            TriggerType, TriggerPriority
        )
        from config.enhanced_score_history_manager import EnhancedScoreHistoryManager
        from config.strategy_scoring_model import StrategyScore, StrategyScoreCalculator

logger = logging.getLogger(__name__)

class UpdatePriority(Enum):
    """更新優先度"""
    REALTIME = 1     # リアルタイム（即座）
    HIGH = 2         # 高優先度（秒単位）
    NORMAL = 3       # 通常（分単位）
    LOW = 4          # 低優先度（時間単位）
    BATCH = 5        # バッチ処理（日次）

class EngineStatus(Enum):
    """エンジン状態"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class UpdateTask:
    """更新タスク"""
    task_id: str
    request: UpdateRequest
    priority: UpdatePriority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """優先度比較（低い値が高優先度）"""
        return self.priority.value < other.priority.value

@dataclass
class BatchUpdateJob:
    """バッチ更新ジョブ"""
    job_id: str
    strategy_names: List[str]
    tickers: List[str]
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

@dataclass
class EngineMetrics:
    """エンジンメトリクス"""
    total_requests: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    average_processing_time: float = 0.0
    peak_queue_size: int = 0
    active_workers: int = 0
    last_update_time: Optional[datetime] = None
    uptime_seconds: float = 0.0

class RealtimeUpdateEngine:
    """
    リアルタイム更新エンジン
    
    ScoreUpdateTriggerSystemからのトリガーを受けて
    高頻度・低遅延でスコア更新を実行するエンジン
    """
    
    def __init__(self,
                 trigger_system: Optional[ScoreUpdateTriggerSystem] = None,
                 enhanced_manager: Optional[EnhancedScoreHistoryManager] = None,
                 score_calculator: Optional[StrategyScoreCalculator] = None,
                 max_workers: int = 5,
                 batch_size: int = 50):
        """
        初期化
        
        Parameters:
            trigger_system: トリガーシステム
            enhanced_manager: 拡張スコア履歴管理
            score_calculator: スコア計算器
            max_workers: 最大ワーカー数
            batch_size: バッチサイズ
        """
        # コアコンポーネント
        self.trigger_system = trigger_system
        self.enhanced_manager = enhanced_manager
        self.score_calculator = score_calculator or StrategyScoreCalculator()
        
        # 設定
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 状態管理
        self.status = EngineStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        # 非同期処理
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # タスク管理
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, UpdateTask] = {}
        self.completed_tasks: List[UpdateTask] = []
        
        # バッチ処理
        self.batch_jobs: Dict[str, BatchUpdateJob] = {}
        self.batch_scheduler_task: Optional[asyncio.Task] = None
        
        # メトリクス
        self.metrics = EngineMetrics()
        
        # コールバック
        self.update_callbacks: List[Callable[[UpdateResult], None]] = []
        
        # 設定
        self.config = {
            "realtime_latency_ms": 100,     # リアルタイム処理目標遅延
            "batch_interval_minutes": 30,   # バッチ処理間隔
            "queue_size_limit": 1000,      # キューサイズ制限
            "retry_delay_seconds": 5,      # リトライ遅延
            "health_check_interval": 60,   # ヘルスチェック間隔
        }
        
        logger.info("Realtime Update Engine initialized")
    
    async def start(self):
        """エンジン開始"""
        if self.status != EngineStatus.STOPPED:
            logger.warning(f"Engine already running or in transition: {self.status}")
            return
        
        self.status = EngineStatus.STARTING
        self.start_time = datetime.now()
        
        try:
            # イベントループ取得
            self.loop = asyncio.get_running_loop()
            
            # ワーカータスク開始
            worker_tasks = []
            for i in range(self.max_workers):
                task = asyncio.create_task(
                    self._worker_loop(f"Worker-{i}"),
                    name=f"UpdateWorker-{i}"
                )
                worker_tasks.append(task)
            
            # バッチスケジューラ開始
            self.batch_scheduler_task = asyncio.create_task(
                self._batch_scheduler_loop(),
                name="BatchScheduler"
            )
            
            # ヘルスチェックタスク開始
            health_task = asyncio.create_task(
                self._health_check_loop(),
                name="HealthCheck"
            )
            
            # トリガーシステムとの連携設定
            if self.trigger_system:
                self.trigger_system.add_trigger_callback(self._handle_trigger_event)
            
            self.status = EngineStatus.RUNNING
            self.metrics.active_workers = len(worker_tasks)
            
            logger.info(f"Realtime Update Engine started with {len(worker_tasks)} workers")
            
            # メインタスクとして実行
            await asyncio.gather(*worker_tasks, self.batch_scheduler_task, health_task)
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Failed to start engine: {e}")
            raise
    
    async def stop(self):
        """エンジン停止"""
        if self.status == EngineStatus.STOPPED:
            return
        
        self.status = EngineStatus.STOPPING
        logger.info("Stopping Realtime Update Engine...")
        
        try:
            # タスクキャンセル
            if self.batch_scheduler_task:
                self.batch_scheduler_task.cancel()
            
            # 実行中タスクの完了待機
            if self.active_tasks:
                await asyncio.sleep(2.0)  # 2秒待機
            
            # エグゼキューター停止
            self.executor.shutdown(wait=True)
            
            self.status = EngineStatus.STOPPED
            logger.info("Realtime Update Engine stopped")
            
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
            self.status = EngineStatus.ERROR
    
    def pause(self):
        """エンジン一時停止"""
        if self.status == EngineStatus.RUNNING:
            self.status = EngineStatus.PAUSED
            logger.info("Realtime Update Engine paused")
    
    def resume(self):
        """エンジン再開"""
        if self.status == EngineStatus.PAUSED:
            self.status = EngineStatus.RUNNING
            logger.info("Realtime Update Engine resumed")
    
    async def _worker_loop(self, worker_name: str):
        """ワーカーループ"""
        logger.debug(f"Worker {worker_name} started")
        
        while self.status not in [EngineStatus.STOPPED, EngineStatus.STOPPING]:
            try:
                if self.status == EngineStatus.PAUSED:
                    await asyncio.sleep(1.0)
                    continue
                
                # タスク取得（タイムアウト付き）
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # タスク実行
                await self._execute_update_task(task, worker_name)
                
                # キュー完了通知
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1.0)
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _execute_update_task(self, task: UpdateTask, worker_name: str):
        """更新タスク実行"""
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        try:
            # 実際のスコア更新実行
            result = await self._perform_score_update(task.request)
            
            task.completed_at = datetime.now()
            
            # 結果処理
            if result.success:
                self.metrics.successful_updates += 1
                logger.debug(f"Task {task.task_id} completed successfully by {worker_name}")
            else:
                # リトライ処理
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    await asyncio.sleep(self.config["retry_delay_seconds"])
                    await self._queue_update_task(task)
                    logger.debug(f"Task {task.task_id} queued for retry ({task.retry_count}/{task.max_retries})")
                    return
                else:
                    self.metrics.failed_updates += 1
                    logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
            
            # コールバック実行
            for callback in self.update_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Update callback error: {e}")
            
            # メトリクス更新
            self._update_metrics(task, result)
            
        except Exception as e:
            task.completed_at = datetime.now()
            self.metrics.failed_updates += 1
            logger.error(f"Task {task.task_id} execution error: {e}")
        
        finally:
            # アクティブタスクから削除
            self.active_tasks.pop(task.task_id, None)
            
            # 完了履歴に追加
            self.completed_tasks.append(task)
            
            # 履歴サイズ制限
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-500:]
    
    async def _perform_score_update(self, request: UpdateRequest) -> UpdateResult:
        """スコア更新実行"""
        start_time = time.time()
        
        try:
            # 現在のスコア取得
            old_score = await self._get_current_score_async(request.strategy_name, request.ticker)
            
            # 新しいスコア計算
            new_score = await self._calculate_score_async(request)
            
            if new_score is not None:
                # スコア保存
                if self.enhanced_manager:
                    await self._save_score_async(request.strategy_name, new_score, request.metadata)
                
                execution_time = time.time() - start_time
                
                return UpdateResult(
                    request_id=request.request_id,
                    strategy_name=request.strategy_name,
                    ticker=request.ticker,
                    success=True,
                    old_score=old_score,
                    new_score=new_score.total_score if hasattr(new_score, 'total_score') else None,
                    execution_time=execution_time
                )
            else:
                raise ValueError("Failed to calculate new score")
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return UpdateResult(
                request_id=request.request_id,
                strategy_name=request.strategy_name,
                ticker=request.ticker,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _get_current_score_async(self, strategy_name: str, ticker: str) -> Optional[float]:
        """非同期現在スコア取得"""
        try:
            # メインスレッドで実行
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self._get_current_score_sync,
                strategy_name,
                ticker
            )
        except Exception as e:
            logger.debug(f"Failed to get current score async: {e}")
            return None
    
    def _get_current_score_sync(self, strategy_name: str, ticker: str) -> Optional[float]:
        """同期現在スコア取得"""
        try:
            if self.enhanced_manager:
                entries = self.enhanced_manager.get_entries(strategy_name, limit=1)
                if entries and hasattr(entries[0].strategy_score, 'total_score'):
                    return entries[0].strategy_score.total_score
        except Exception as e:
            logger.debug(f"Failed to get current score sync: {e}")
        
        return None
    
    async def _calculate_score_async(self, request: UpdateRequest) -> Optional[StrategyScore]:
        """非同期スコア計算"""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self._calculate_score_sync,
                request
            )
        except Exception as e:
            logger.error(f"Failed to calculate score async: {e}")
            return None
    
    def _calculate_score_sync(self, request: UpdateRequest) -> Optional[StrategyScore]:
        """同期スコア計算"""
        try:
            # 実際のスコア計算（プレースホルダー）
            # 実際の実装では市場データやパフォーマンスデータを使用
            base_score = 0.75 + (hash(request.strategy_name) % 100) / 1000
            
            score = StrategyScore(
                strategy_name=request.strategy_name,
                ticker=request.ticker,
                total_score=base_score,
                component_scores={
                    "performance": base_score + 0.1,
                    "stability": base_score - 0.05,
                    "risk_adjusted": base_score,
                    "reliability": base_score + 0.05
                },
                trend_fitness=base_score,
                confidence=0.85,
                metadata={
                    **request.metadata,
                    "update_source": "realtime_engine",
                    "priority": request.priority
                },
                calculated_at=datetime.now()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate score sync: {e}")
            return None
    
    async def _save_score_async(self, 
                              strategy_name: str, 
                              score: StrategyScore, 
                              metadata: Dict[str, Any]):
        """非同期スコア保存"""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.executor,
                self._save_score_sync,
                strategy_name,
                score,
                metadata
            )
        except Exception as e:
            logger.error(f"Failed to save score async: {e}")
    
    def _save_score_sync(self, 
                        strategy_name: str, 
                        score: StrategyScore, 
                        metadata: Dict[str, Any]):
        """同期スコア保存"""
        try:
            if self.enhanced_manager:
                self.enhanced_manager.add_enhanced_entry(
                    strategy_name=strategy_name,
                    strategy_score=score,
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Failed to save score sync: {e}")
    
    async def _batch_scheduler_loop(self):
        """バッチスケジューラループ"""
        logger.debug("Batch scheduler started")
        
        while self.status not in [EngineStatus.STOPPED, EngineStatus.STOPPING]:
            try:
                if self.status == EngineStatus.PAUSED:
                    await asyncio.sleep(60)
                    continue
                
                # バッチジョブスケジューリング
                await self._schedule_batch_jobs()
                
                # バッチ処理間隔待機
                await asyncio.sleep(self.config["batch_interval_minutes"] * 60)
                
            except asyncio.CancelledError:
                logger.debug("Batch scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Batch scheduler error: {e}")
                await asyncio.sleep(300)  # エラー時は5分待機
        
        logger.debug("Batch scheduler stopped")
    
    async def _schedule_batch_jobs(self):
        """バッチジョブスケジューリング"""
        try:
            # 低優先度の更新をバッチ処理として実行
            strategies = self._get_all_strategies()
            tickers = self._get_all_tickers()
            
            if strategies and tickers:
                job_id = f"batch_{int(time.time())}"
                job = BatchUpdateJob(
                    job_id=job_id,
                    strategy_names=strategies,
                    tickers=tickers,
                    scheduled_at=datetime.now(),
                    total_tasks=len(strategies) * len(tickers)
                )
                
                self.batch_jobs[job_id] = job
                
                # バッチタスクをキューに追加
                for strategy_name in strategies:
                    for ticker in tickers:
                        request = UpdateRequest(
                            request_id=f"{job_id}_{strategy_name}_{ticker}",
                            strategy_name=strategy_name,
                            ticker=ticker,
                            trigger_type=TriggerType.TIME_BASED,
                            priority=5,  # バッチ優先度
                            metadata={"batch_job_id": job_id}
                        )
                        
                        task = UpdateTask(
                            task_id=request.request_id,
                            request=request,
                            priority=UpdatePriority.BATCH
                        )
                        
                        await self._queue_update_task(task)
                
                job.started_at = datetime.now()
                logger.info(f"Scheduled batch job {job_id} with {job.total_tasks} tasks")
        
        except Exception as e:
            logger.error(f"Failed to schedule batch jobs: {e}")
    
    async def _health_check_loop(self):
        """ヘルスチェックループ"""
        logger.debug("Health check started")
        
        while self.status not in [EngineStatus.STOPPED, EngineStatus.STOPPING]:
            try:
                # エンジン状態チェック
                await self._perform_health_check()
                
                # メトリクス更新
                if self.start_time:
                    self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
                # ヘルスチェック間隔
                await asyncio.sleep(self.config["health_check_interval"])
                
            except asyncio.CancelledError:
                logger.debug("Health check cancelled")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
        
        logger.debug("Health check stopped")
    
    async def _perform_health_check(self):
        """ヘルスチェック実行"""
        # キューサイズチェック
        queue_size = self.task_queue.qsize()
        self.metrics.peak_queue_size = max(self.metrics.peak_queue_size, queue_size)
        
        if queue_size > self.config["queue_size_limit"]:
            logger.warning(f"Queue size limit exceeded: {queue_size}")
        
        # アクティブタスク数チェック
        active_count = len(self.active_tasks)
        if active_count > self.max_workers * 2:
            logger.warning(f"High number of active tasks: {active_count}")
        
        # メモリ使用量チェック（簡易）
        completed_count = len(self.completed_tasks)
        if completed_count > 1000:
            logger.debug("Cleaning up completed tasks history")
    
    async def _queue_update_task(self, task: UpdateTask):
        """更新タスクをキューに追加"""
        try:
            await self.task_queue.put((task.priority.value, task))
            self.metrics.total_requests += 1
            logger.debug(f"Queued task {task.task_id} with priority {task.priority.name}")
        except Exception as e:
            logger.error(f"Failed to queue task {task.task_id}: {e}")
    
    def _handle_trigger_event(self, event: TriggerEvent):
        """トリガーイベント処理"""
        try:
            # トリガーイベントを更新リクエストに変換
            request = UpdateRequest(
                request_id=event.event_id,
                strategy_name=event.strategy_name,
                ticker=event.ticker,
                trigger_type=event.trigger_type,
                priority=event.priority.value,
                metadata=event.event_data
            )
            
            # 優先度マッピング
            priority_map = {
                TriggerPriority.CRITICAL: UpdatePriority.REALTIME,
                TriggerPriority.HIGH: UpdatePriority.HIGH,
                TriggerPriority.MEDIUM: UpdatePriority.NORMAL,
                TriggerPriority.LOW: UpdatePriority.LOW
            }
            
            priority = priority_map.get(event.priority, UpdatePriority.NORMAL)
            
            task = UpdateTask(
                task_id=request.request_id,
                request=request,
                priority=priority
            )
            
            # 非同期キューに追加
            if self.loop and self.status == EngineStatus.RUNNING:
                asyncio.create_task(self._queue_update_task(task))
            
        except Exception as e:
            logger.error(f"Failed to handle trigger event {event.event_id}: {e}")
    
    def _update_metrics(self, task: UpdateTask, result: UpdateResult):
        """メトリクス更新"""
        self.metrics.last_update_time = datetime.now()
        
        # 平均処理時間更新
        if result.execution_time > 0:
            total = self.metrics.total_requests
            current_avg = self.metrics.average_processing_time
            
            self.metrics.average_processing_time = (
                (current_avg * (total - 1) + result.execution_time) / total
            )
    
    def _get_all_strategies(self) -> List[str]:
        """全戦略取得"""
        # プレースホルダー実装
        return ["vwap_bounce_strategy", "momentum_strategy", "mean_reversion_strategy"]
    
    def _get_all_tickers(self) -> List[str]:
        """全ティッカー取得"""
        # プレースホルダー実装
        return ["AAPL", "MSFT", "GOOGL"]
    
    # =========================================================================
    # 公開API
    # =========================================================================
    
    async def submit_update_request(self, request: UpdateRequest, priority: UpdatePriority = UpdatePriority.NORMAL) -> str:
        """更新リクエスト送信"""
        task = UpdateTask(
            task_id=request.request_id,
            request=request,
            priority=priority
        )
        
        await self._queue_update_task(task)
        return task.task_id
    
    def get_engine_status(self) -> Dict[str, Any]:
        """エンジン状態取得"""
        return {
            "status": self.status.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "total_requests": self.metrics.total_requests,
            "successful_updates": self.metrics.successful_updates,
            "failed_updates": self.metrics.failed_updates,
            "average_processing_time": self.metrics.average_processing_time,
            "peak_queue_size": self.metrics.peak_queue_size,
            "last_update_time": self.metrics.last_update_time.isoformat() if self.metrics.last_update_time else None
        }
    
    def get_batch_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """バッチジョブ状態取得"""
        if job_id in self.batch_jobs:
            job = self.batch_jobs[job_id]
            return {
                "job_id": job.job_id,
                "total_tasks": job.total_tasks,
                "completed_tasks": job.completed_tasks,
                "failed_tasks": job.failed_tasks,
                "progress": job.completed_tasks / job.total_tasks if job.total_tasks > 0 else 0,
                "scheduled_at": job.scheduled_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
        return None
    
    def add_update_callback(self, callback: Callable[[UpdateResult], None]):
        """更新コールバック追加"""
        self.update_callbacks.append(callback)


# =============================================================================
# ユーティリティ関数
# =============================================================================

async def create_realtime_engine(trigger_system: Optional[ScoreUpdateTriggerSystem] = None,
                                enhanced_manager: Optional[EnhancedScoreHistoryManager] = None,
                                score_calculator: Optional[StrategyScoreCalculator] = None) -> RealtimeUpdateEngine:
    """リアルタイム更新エンジン作成"""
    return RealtimeUpdateEngine(trigger_system, enhanced_manager, score_calculator)


# =============================================================================
# エクスポート  
# =============================================================================

__all__ = [
    "RealtimeUpdateEngine",
    "UpdatePriority",
    "EngineStatus",
    "UpdateTask",
    "BatchUpdateJob",
    "EngineMetrics",
    "create_realtime_engine"
]


if __name__ == "__main__":
    # デバッグ用テスト
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        engine = await create_realtime_engine()
        
        try:
            # エンジン開始
            start_task = asyncio.create_task(engine.start())
            
            # テスト用更新リクエスト
            await asyncio.sleep(1)
            
            request = UpdateRequest(
                request_id="test_request_001",
                strategy_name="test_strategy",
                ticker="TEST",
                trigger_type=TriggerType.MANUAL,
                metadata={"test": True}
            )
            
            task_id = await engine.submit_update_request(request, UpdatePriority.HIGH)
            print(f"Submitted update request: {task_id}")
            
            # 状態確認
            await asyncio.sleep(2)
            status = engine.get_engine_status()
            print(f"Engine Status: {status}")
            
            # 停止
            await engine.stop()
            start_task.cancel()
            
        except KeyboardInterrupt:
            await engine.stop()
    
    asyncio.run(main())
