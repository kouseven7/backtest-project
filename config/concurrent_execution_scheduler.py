"""
Module: Concurrent Execution Scheduler
File: concurrent_execution_scheduler.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  ハイブリッド並行実行スケジューラー

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - ハイブリッド実行モード管理 (Thread/Process/Async)
  - 動的負荷分散・リアルタイム実行制御
  - 実行状態監視・パフォーマンス最適化
  - 例外処理・フォールバック実行
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
import threading
import multiprocessing
from functools import wraps
import traceback
import signal

# プロジェクトモジュールをインポート
try:
    from resource_allocation_engine import ResourceAllocation, ExecutionMode, Priority, ResourceAllocationEngine
    from strategy_dependency_resolver import DependencyResolution, ExecutionGraph, StrategyDependencyResolver
except ImportError:
    # スタンドアロンテスト用フォールバック
    logger = logging.getLogger(__name__)
    logger.warning("Could not import project modules, using fallback definitions")
    
    class ExecutionMode(Enum):
        THREAD = "thread"
        PROCESS = "process"
        ASYNC = "async"
        AUTO = "auto"

# ロガー設定
logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """実行状態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class SchedulerState(Enum):
    """スケジューラー状態"""
    IDLE = "idle"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class ExecutionTask:
    """実行タスク"""
    task_id: str
    strategy_name: str
    execution_mode: ExecutionMode
    function: Callable
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 300.0
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['execution_mode'] = self.execution_mode.value
        result['created_at'] = self.created_at.isoformat()
        # 関数は文字列表現に変換
        result['function'] = str(self.function)
        return result

@dataclass
class ExecutionResult:
    """実行結果"""
    task_id: str
    strategy_name: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['start_time'] = self.start_time.isoformat() if self.start_time else None
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result

@dataclass
class SchedulingPlan:
    """スケジューリング計画"""
    execution_batches: List[List[ExecutionTask]] = field(default_factory=list)
    total_estimated_time: float = 0.0
    parallel_factor: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    optimization_notes: List[str] = field(default_factory=list)

class TaskExecutor:
    """タスク実行器基底クラス"""
    
    def __init__(self, execution_mode: ExecutionMode, config: Dict[str, Any]):
        self.execution_mode = execution_mode
        self.config = config
        self.active_tasks: Dict[str, Future] = {}
        self.task_lock = threading.Lock()
    
    def submit_task(self, task: ExecutionTask) -> Future:
        """タスク投入"""
        raise NotImplementedError
    
    def cancel_task(self, task_id: str) -> bool:
        """タスクキャンセル"""
        with self.task_lock:
            if task_id in self.active_tasks:
                future = self.active_tasks[task_id]
                success = future.cancel()
                if success:
                    del self.active_tasks[task_id]
                return success
            return False
    
    def get_active_task_count(self) -> int:
        """アクティブタスク数取得"""
        with self.task_lock:
            return len(self.active_tasks)
    
    def shutdown(self):
        """シャットダウン"""
        with self.task_lock:
            for future in self.active_tasks.values():
                future.cancel()
            self.active_tasks.clear()

class ThreadTaskExecutor(TaskExecutor):
    """スレッドタスク実行器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExecutionMode.THREAD, config)
        max_workers = config.get('execution_modes', {}).get('thread_pool_size', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ConcurrentScheduler")
        logger.info(f"Initialized ThreadTaskExecutor with {max_workers} workers")
    
    def submit_task(self, task: ExecutionTask) -> Future:
        """スレッドタスク投入"""
        wrapped_function = self._wrap_function_with_monitoring(task)
        future = self.executor.submit(wrapped_function, *task.args, **task.kwargs)
        
        with self.task_lock:
            self.active_tasks[task.task_id] = future
        
        # タスク完了時のクリーンアップコールバック
        future.add_done_callback(lambda f: self._cleanup_task(task.task_id))
        
        return future
    
    def _wrap_function_with_monitoring(self, task: ExecutionTask) -> Callable:
        """関数をモニタリング付きでラップ"""
        @wraps(task.function)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                logger.debug(f"Thread execution started for {task.strategy_name} ({task.task_id})")
                result = task.function(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Thread execution completed for {task.strategy_name} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Thread execution failed for {task.strategy_name} after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    
    def _cleanup_task(self, task_id: str):
        """タスククリーンアップ"""
        with self.task_lock:
            self.active_tasks.pop(task_id, None)
    
    def shutdown(self):
        """シャットダウン"""
        super().shutdown()
        self.executor.shutdown(wait=True)
        logger.info("ThreadTaskExecutor shutdown complete")

class ProcessTaskExecutor(TaskExecutor):
    """プロセスタスク実行器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExecutionMode.PROCESS, config)
        max_workers = config.get('execution_modes', {}).get('process_pool_size', 2)
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized ProcessTaskExecutor with {max_workers} workers")
    
    def submit_task(self, task: ExecutionTask) -> Future:
        """プロセスタスク投入"""
        # プロセス実行用に関数をラップ
        wrapped_function = self._create_process_wrapper(task)
        future = self.executor.submit(wrapped_function, task.strategy_name, *task.args, **task.kwargs)
        
        with self.task_lock:
            self.active_tasks[task.task_id] = future
        
        # タスク完了時のクリーンアップコールバック
        future.add_done_callback(lambda f: self._cleanup_task(task.task_id))
        
        return future
    
    def _create_process_wrapper(self, task: ExecutionTask) -> Callable:
        """プロセス実行用ラッパー作成"""
        def process_wrapper(strategy_name: str, *args, **kwargs):
            """プロセス内で実行される関数ラッパー"""
            import time
            start_time = time.time()
            try:
                print(f"Process execution started for {strategy_name}")
                result = task.function(*args, **kwargs)
                execution_time = time.time() - start_time
                print(f"Process execution completed for {strategy_name} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"Process execution failed for {strategy_name} after {execution_time:.2f}s: {e}")
                raise
        
        return process_wrapper
    
    def _cleanup_task(self, task_id: str):
        """タスククリーンアップ"""
        with self.task_lock:
            self.active_tasks.pop(task_id, None)
    
    def shutdown(self):
        """シャットダウン"""
        super().shutdown()
        self.executor.shutdown(wait=True)
        logger.info("ProcessTaskExecutor shutdown complete")

class AsyncTaskExecutor(TaskExecutor):
    """非同期タスク実行器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ExecutionMode.ASYNC, config)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        logger.info("Initialized AsyncTaskExecutor")
    
    def submit_task(self, task: ExecutionTask) -> Future:
        """非同期タスク投入"""
        if not self.loop or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # 同期関数を非同期でラップ
        async_wrapper = self._create_async_wrapper(task)
        
        # Future作成（ThreadPoolExecutorのFutureと互換性を持たせる）
        future = Future()
        
        # 非同期タスク作成と実行
        async_task = self.loop.create_task(async_wrapper())
        
        with self.task_lock:
            self.active_tasks[task.task_id] = future
            self.async_tasks[task.task_id] = async_task
        
        # 結果をFutureに設定するコールバック
        def set_future_result(async_task: asyncio.Task):
            try:
                if async_task.cancelled():
                    future.cancel()
                elif async_task.exception():
                    future.set_exception(async_task.exception())
                else:
                    future.set_result(async_task.result())
            except Exception as e:
                future.set_exception(e)
            
            # クリーンアップ
            self._cleanup_task(task.task_id)
        
        async_task.add_done_callback(set_future_result)
        
        return future
    
    def _create_async_wrapper(self, task: ExecutionTask) -> Callable:
        """非同期実行用ラッパー作成"""
        async def async_wrapper():
            start_time = time.time()
            try:
                logger.debug(f"Async execution started for {task.strategy_name} ({task.task_id})")
                
                # 同期関数を非同期で実行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.function, *task.args)
                
                execution_time = time.time() - start_time
                logger.debug(f"Async execution completed for {task.strategy_name} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Async execution failed for {task.strategy_name} after {execution_time:.2f}s: {e}")
                raise
        
        return async_wrapper
    
    def _cleanup_task(self, task_id: str):
        """タスククリーンアップ"""
        with self.task_lock:
            self.active_tasks.pop(task_id, None)
            self.async_tasks.pop(task_id, None)
    
    def cancel_task(self, task_id: str) -> bool:
        """非同期タスクキャンセル"""
        with self.task_lock:
            if task_id in self.async_tasks:
                async_task = self.async_tasks[task_id]
                success = async_task.cancel()
                if success:
                    del self.async_tasks[task_id]
                    self.active_tasks.pop(task_id, None)
                return success
            return super().cancel_task(task_id)
    
    def shutdown(self):
        """シャットダウン"""
        with self.task_lock:
            for async_task in self.async_tasks.values():
                async_task.cancel()
            self.async_tasks.clear()
        
        super().shutdown()
        
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        
        logger.info("AsyncTaskExecutor shutdown complete")

class ConcurrentExecutionScheduler:
    """並行実行スケジューラー"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.state = SchedulerState.IDLE
        self.executors: Dict[ExecutionMode, TaskExecutor] = {}
        self.pending_tasks: List[ExecutionTask] = []
        self.running_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.task_counter = 0
        self.scheduler_lock = threading.Lock()
        
        # 実行器初期化
        self._initialize_executors()
        
        # パフォーマンス監視用
        self.performance_metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'average_execution_time': 0.0,
            'peak_concurrent_tasks': 0
        }
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定読み込み"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'coordination_config.json')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "execution_modes": {
                "thread_pool_size": 4,
                "process_pool_size": 2,
                "execution_timeout": 300,
                "default_retry_count": 3
            },
            "scheduling": {
                "max_concurrent_tasks": 10,
                "task_submission_interval": 0.1,
                "load_balancing_interval": 5.0
            }
        }
    
    def _initialize_executors(self):
        """実行器初期化"""
        try:
            self.executors[ExecutionMode.THREAD] = ThreadTaskExecutor(self.config)
            self.executors[ExecutionMode.PROCESS] = ProcessTaskExecutor(self.config)
            self.executors[ExecutionMode.ASYNC] = AsyncTaskExecutor(self.config)
            logger.info("All task executors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize executors: {e}")
            raise
    
    def create_scheduling_plan(self, tasks: List[ExecutionTask]) -> SchedulingPlan:
        """スケジューリング計画作成"""
        logger.info(f"Creating scheduling plan for {len(tasks)} tasks")
        
        # 依存関係による実行順序決定
        execution_batches = self._group_tasks_by_dependencies(tasks)
        
        # リソース要求量計算
        resource_requirements = self._calculate_resource_requirements(tasks)
        
        # 推定実行時間計算
        total_estimated_time = self._calculate_total_execution_time(execution_batches)
        
        # 並列度計算
        parallel_factor = self._calculate_parallel_factor(execution_batches)
        
        # 最適化提案
        optimization_notes = self._generate_optimization_notes(execution_batches, resource_requirements)
        
        plan = SchedulingPlan(
            execution_batches=execution_batches,
            total_estimated_time=total_estimated_time,
            parallel_factor=parallel_factor,
            resource_requirements=resource_requirements,
            optimization_notes=optimization_notes
        )
        
        logger.info(f"Scheduling plan created: {len(execution_batches)} batches, {total_estimated_time:.1f}s estimated time")
        return plan
    
    def _group_tasks_by_dependencies(self, tasks: List[ExecutionTask]) -> List[List[ExecutionTask]]:
        """依存関係によるタスクグループ化"""
        batches = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # 現在実行可能なタスク（依存関係が解決済み）を見つける
            ready_tasks = []
            for task in remaining_tasks:
                if not task.dependencies or all(
                    dep_id in [t.task_id for batch in batches for t in batch] 
                    for dep_id in task.dependencies
                ):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # 依存関係が解決できない場合、残りのタスクを強制実行
                logger.warning("Unresolved dependencies found, forcing execution of remaining tasks")
                ready_tasks = remaining_tasks.copy()
            
            # 優先度でソート
            ready_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            batches.append(ready_tasks)
            
            # 処理済みタスクを除去
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return batches
    
    def _calculate_resource_requirements(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """リソース要求量計算"""
        total_cpu = 0.0
        total_memory = 0.0
        execution_mode_count = {mode: 0 for mode in ExecutionMode}
        
        for task in tasks:
            # 実行モード別カウント
            execution_mode_count[task.execution_mode] += 1
            
            # 簡単なリソース推定
            if task.execution_mode == ExecutionMode.PROCESS:
                total_cpu += 0.5  # プロセスは多くのCPUを使用
                total_memory += 128  # MB
            elif task.execution_mode == ExecutionMode.THREAD:
                total_cpu += 0.2
                total_memory += 64
            else:  # ASYNC
                total_cpu += 0.1
                total_memory += 32
        
        return {
            'total_cpu_requirement': total_cpu,
            'total_memory_requirement_mb': total_memory,
            'execution_mode_distribution': {
                mode.value: count for mode, count in execution_mode_count.items()
            }
        }
    
    def _calculate_total_execution_time(self, execution_batches: List[List[ExecutionTask]]) -> float:
        """総実行時間計算"""
        total_time = 0.0
        
        for batch in execution_batches:
            if not batch:
                continue
            
            # バッチ内の最大実行時間（並列実行のため）
            batch_max_time = max(task.timeout for task in batch)
            total_time += batch_max_time
        
        return total_time
    
    def _calculate_parallel_factor(self, execution_batches: List[List[ExecutionTask]]) -> float:
        """並列度計算"""
        if not execution_batches:
            return 1.0
        
        total_tasks = sum(len(batch) for batch in execution_batches)
        total_batches = len(execution_batches)
        
        # 平均バッチサイズ = 並列度の指標
        return total_tasks / total_batches if total_batches > 0 else 1.0
    
    def _generate_optimization_notes(
        self, 
        execution_batches: List[List[ExecutionTask]], 
        resource_requirements: Dict[str, Any]
    ) -> List[str]:
        """最適化提案生成"""
        notes = []
        
        # バッチ数チェック
        if len(execution_batches) > 5:
            notes.append("実行バッチが多いため、依存関係の簡素化を検討してください")
        
        # リソース使用量チェック
        if resource_requirements.get('total_cpu_requirement', 0) > 2.0:
            notes.append("CPU要求量が高いため、プロセス並列数の調整を検討してください")
        
        if resource_requirements.get('total_memory_requirement_mb', 0) > 1024:
            notes.append("メモリ要求量が高いため、メモリ使用量の最適化を検討してください")
        
        # 実行モード分布チェック
        mode_dist = resource_requirements.get('execution_mode_distribution', {})
        if mode_dist.get('process', 0) > 3:
            notes.append("プロセス並列タスクが多いため、スレッド並列への変更を検討してください")
        
        return notes
    
    def submit_task(self, task: ExecutionTask) -> str:
        """タスク投入"""
        with self.scheduler_lock:
            self.task_counter += 1
            if not task.task_id:
                task.task_id = f"task_{self.task_counter}_{task.strategy_name}"
            
            self.pending_tasks.append(task)
            self.performance_metrics['total_tasks_submitted'] += 1
            
            logger.info(f"Task submitted: {task.task_id} ({task.strategy_name}, {task.execution_mode.value})")
            return task.task_id
    
    def execute_tasks(self, tasks: List[ExecutionTask]) -> Dict[str, ExecutionResult]:
        """タスク群実行"""
        logger.info(f"Executing {len(tasks)} tasks")
        
        with self.scheduler_lock:
            self.state = SchedulerState.SCHEDULING
        
        try:
            # スケジューリング計画作成
            plan = self.create_scheduling_plan(tasks)
            
            with self.scheduler_lock:
                self.state = SchedulerState.EXECUTING
            
            # バッチごとに実行
            results = {}
            for batch_index, batch in enumerate(plan.execution_batches):
                logger.info(f"Executing batch {batch_index + 1}/{len(plan.execution_batches)} with {len(batch)} tasks")
                
                batch_results = self._execute_batch(batch)
                results.update(batch_results)
                
                # バッチ間の待機時間
                if batch_index < len(plan.execution_batches) - 1:
                    time.sleep(self.config.get('scheduling', {}).get('task_submission_interval', 0.1))
            
            # 実行履歴記録
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'task_count': len(tasks),
                'execution_time': sum(r.execution_time for r in results.values()),
                'success_count': sum(1 for r in results.values() if r.status == ExecutionStatus.COMPLETED),
                'failure_count': sum(1 for r in results.values() if r.status == ExecutionStatus.FAILED)
            })
            
            with self.scheduler_lock:
                self.state = SchedulerState.IDLE
            
            logger.info(f"Task execution completed: {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            with self.scheduler_lock:
                self.state = SchedulerState.IDLE
            raise
    
    def _execute_batch(self, batch: List[ExecutionTask]) -> Dict[str, ExecutionResult]:
        """バッチ実行"""
        futures: Dict[str, Future] = {}
        results: Dict[str, ExecutionResult] = {}
        
        # バッチ内タスクを並列投入
        for task in batch:
            try:
                executor = self.executors[task.execution_mode]
                future = executor.submit_task(task)
                futures[task.task_id] = future
                
                with self.scheduler_lock:
                    self.running_tasks[task.task_id] = task
                
            except Exception as e:
                logger.error(f"Failed to submit task {task.task_id}: {e}")
                results[task.task_id] = ExecutionResult(
                    task_id=task.task_id,
                    strategy_name=task.strategy_name,
                    status=ExecutionStatus.FAILED,
                    error=str(e)
                )
        
        # バッチ完了待機
        completed_futures = as_completed(futures.values(), 
                                       timeout=max(task.timeout for task in batch) if batch else 300)
        
        for future in completed_futures:
            # Future→TaskIDマッピングを探す
            task_id = None
            for tid, f in futures.items():
                if f is future:
                    task_id = tid
                    break
            
            if not task_id:
                continue
            
            task = self.running_tasks.get(task_id)
            if not task:
                continue
            
            # 結果処理
            start_time = datetime.now()
            try:
                result_value = future.result()
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                result = ExecutionResult(
                    task_id=task_id,
                    strategy_name=task.strategy_name,
                    status=ExecutionStatus.COMPLETED,
                    result=result_value,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time
                )
                
                self.performance_metrics['total_tasks_completed'] += 1
                
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                result = ExecutionResult(
                    task_id=task_id,
                    strategy_name=task.strategy_name,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time
                )
                
                self.performance_metrics['total_tasks_failed'] += 1
                logger.error(f"Task {task_id} failed: {e}")
            
            results[task_id] = result
            
            with self.scheduler_lock:
                self.running_tasks.pop(task_id, None)
                self.completed_tasks[task_id] = result
        
        # 平均実行時間更新
        if results:
            avg_time = sum(r.execution_time for r in results.values()) / len(results)
            current_avg = self.performance_metrics['average_execution_time']
            total_completed = self.performance_metrics['total_tasks_completed']
            
            # 移動平均計算
            self.performance_metrics['average_execution_time'] = (
                (current_avg * (total_completed - len(results)) + avg_time * len(results)) / total_completed
            )
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """タスクキャンセル"""
        # 実行中タスクキャンセル
        with self.scheduler_lock:
            task = self.running_tasks.get(task_id)
            if task:
                executor = self.executors[task.execution_mode]
                success = executor.cancel_task(task_id)
                if success:
                    self.running_tasks.pop(task_id, None)
                    self.completed_tasks[task_id] = ExecutionResult(
                        task_id=task_id,
                        strategy_name=task.strategy_name,
                        status=ExecutionStatus.CANCELLED
                    )
                    logger.info(f"Task {task_id} cancelled successfully")
                return success
            
            # 待機中タスクキャンセル
            for i, pending_task in enumerate(self.pending_tasks):
                if pending_task.task_id == task_id:
                    self.pending_tasks.pop(i)
                    self.completed_tasks[task_id] = ExecutionResult(
                        task_id=task_id,
                        strategy_name=pending_task.strategy_name,
                        status=ExecutionStatus.CANCELLED
                    )
                    logger.info(f"Pending task {task_id} cancelled successfully")
                    return True
        
        return False
    
    def get_execution_status(self) -> Dict[str, Any]:
        """実行状態取得"""
        with self.scheduler_lock:
            status = {
                'scheduler_state': self.state.value,
                'pending_tasks': len(self.pending_tasks),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'performance_metrics': self.performance_metrics.copy()
            }
            
            # アクティブタスク数統計
            active_by_mode = {}
            for mode, executor in self.executors.items():
                active_by_mode[mode.value] = executor.get_active_task_count()
            
            status['active_tasks_by_mode'] = active_by_mode
            
            return status
    
    def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """タスク結果取得"""
        with self.scheduler_lock:
            return self.completed_tasks.get(task_id)
    
    def shutdown(self):
        """シャットダウン"""
        logger.info("Shutting down Concurrent Execution Scheduler...")
        
        with self.scheduler_lock:
            self.state = SchedulerState.STOPPED
        
        # 全実行器をシャットダウン
        for mode, executor in self.executors.items():
            try:
                executor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {mode.value} executor: {e}")
        
        logger.info("Concurrent Execution Scheduler shutdown complete")

# デモ用関数
def demo_strategy_function(strategy_name: str, duration: float = 1.0) -> Dict[str, Any]:
    """デモ用戦略関数"""
    import time
    import random
    
    print(f"Executing {strategy_name} for {duration}s...")
    time.sleep(duration)
    
    # ランダムな結果を生成
    result = {
        'strategy_name': strategy_name,
        'execution_duration': duration,
        'trades_executed': random.randint(1, 10),
        'pnl': round(random.uniform(-100, 200), 2),
        'success': random.choice([True, True, True, False])  # 75% success rate
    }
    
    if not result['success']:
        raise Exception(f"Strategy {strategy_name} failed with random error")
    
    return result

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Concurrent Execution Scheduler - Demo")
    print("=" * 60)
    
    try:
        # スケジューラー初期化
        scheduler = ConcurrentExecutionScheduler()
        
        # デモタスク作成
        demo_tasks = [
            ExecutionTask(
                task_id=f"demo_task_{i}",
                strategy_name=f"Strategy{i}",
                execution_mode=ExecutionMode.THREAD if i % 3 == 0 
                              else ExecutionMode.PROCESS if i % 3 == 1 
                              else ExecutionMode.ASYNC,
                function=demo_strategy_function,
                args=(f"Strategy{i}", 2.0 + i * 0.5),
                priority=3 - (i % 3),
                timeout=30.0
            )
            for i in range(1, 6)
        ]
        
        print(f"\n🎯 Testing concurrent execution with {len(demo_tasks)} tasks")
        for task in demo_tasks:
            print(f"  - {task.strategy_name}: {task.execution_mode.value} mode, priority {task.priority}")
        
        # スケジューリング計画作成
        print(f"\n📋 Creating scheduling plan...")
        plan = scheduler.create_scheduling_plan(demo_tasks)
        
        print(f"Execution Batches: {len(plan.execution_batches)}")
        print(f"Total Estimated Time: {plan.total_estimated_time:.1f}s")
        print(f"Parallel Factor: {plan.parallel_factor:.2f}")
        
        if plan.optimization_notes:
            print(f"Optimization Notes:")
            for note in plan.optimization_notes:
                print(f"  💡 {note}")
        
        # タスク実行
        print(f"\n🚀 Executing tasks...")
        start_time = time.time()
        results = scheduler.execute_tasks(demo_tasks)
        execution_time = time.time() - start_time
        
        print(f"\n📊 Execution Results (completed in {execution_time:.2f}s):")
        print("-" * 50)
        
        for task_id, result in results.items():
            status_emoji = "✅" if result.status == ExecutionStatus.COMPLETED else "❌"
            print(f"{status_emoji} {result.strategy_name} ({task_id})")
            print(f"    Status: {result.status.value}")
            print(f"    Execution Time: {result.execution_time:.2f}s")
            if result.error:
                print(f"    Error: {result.error}")
            elif result.result:
                print(f"    Result: {result.result}")
        
        # 実行状態統計
        status = scheduler.get_execution_status()
        print(f"\n📈 Scheduler Statistics:")
        print(f"  Completed Tasks: {status['completed_tasks']}")
        print(f"  Performance Metrics:")
        metrics = status['performance_metrics']
        print(f"    Total Submitted: {metrics['total_tasks_submitted']}")
        print(f"    Total Completed: {metrics['total_tasks_completed']}")
        print(f"    Total Failed: {metrics['total_tasks_failed']}")
        print(f"    Average Execution Time: {metrics['average_execution_time']:.2f}s")
        
        print("\n✅ Concurrent Execution Scheduler demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'scheduler' in locals():
            scheduler.shutdown()
