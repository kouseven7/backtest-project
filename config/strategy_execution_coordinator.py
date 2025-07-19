"""
Module: Strategy Execution Coordinator
File: strategy_execution_coordinator.py
Description: 
  4-1-2「複合戦略実行フロー設計・実装」- Coordination Component
  マルチストラテジー実行の調整・スケジューリング
  動的順序付け、負荷分散、リソース管理

Author: imega
Created: 2025-01-28
Modified: 2025-01-28

Dependencies:
  - config.strategy_execution_pipeline
  - config.strategy_selector
"""

import os
import sys
import json
import logging
import time
import psutil
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
import threading
import queue
import multiprocessing

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムのインポート
try:
    from config.strategy_execution_pipeline import StrategyExecutionPipeline, PipelineContext
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """実行モード"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"

class CoordinationStrategy(Enum):
    """調整戦略"""
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    DYNAMIC_ORDERING = "dynamic_ordering"

@dataclass
class ExecutionTask:
    """実行タスク"""
    task_id: str
    strategy_name: str
    market_data: pd.DataFrame
    parameters: Dict[str, Any]
    priority: int = 0
    timeout: int = 180
    retry_count: int = 0
    max_retries: int = 2
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass 
class ExecutionResult:
    """実行結果"""
    task_id: str
    strategy_name: str
    context: Optional[PipelineContext] = None
    success: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    completed_at: datetime = field(default_factory=datetime.now)

@dataclass
class SystemResources:
    """システムリソース情報"""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    active_processes: int
    load_average: float
    timestamp: datetime = field(default_factory=datetime.now)

class ResourceMonitor:
    """リソース監視"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")
        
    def get_current_resources(self) -> SystemResources:
        """現在のリソース状況を取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # プロセス数
            active_processes = len(psutil.pids())
            
            # ロードアベレージ（Windows では近似値）
            load_average = cpu_percent / 100.0
            
            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available,
                active_processes=active_processes,
                load_average=load_average
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return SystemResources(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available=0,
                active_processes=0,
                load_average=0.0
            )
            
    def is_resources_available(self, cpu_limit: float = 80.0, 
                             memory_limit: float = 85.0) -> bool:
        """リソース使用可能性チェック"""
        resources = self.get_current_resources()
        return (resources.cpu_percent < cpu_limit and 
                resources.memory_percent < memory_limit)

class LoadBalancer:
    """負荷分散器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.resource_monitor = ResourceMonitor()
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: List[ExecutionResult] = []
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
        self._lock = threading.Lock()
        
    def should_use_parallel(self, task_count: int, 
                          resources: SystemResources) -> bool:
        """並列実行の判定"""
        # リソース制約チェック
        if resources.cpu_percent > 70 or resources.memory_percent > 80:
            return False
            
        # タスク数による判定
        return task_count >= 2 and task_count <= self.max_workers
        
    def get_optimal_workers(self, task_count: int, 
                          resources: SystemResources) -> int:
        """最適ワーカー数の決定"""
        available_cpu = max(1, int((100 - resources.cpu_percent) / 15))
        available_memory = max(1, int((100 - resources.memory_percent) / 20))
        
        optimal = min(
            task_count,
            self.max_workers,
            available_cpu,
            available_memory
        )
        
        return max(1, optimal)
        
    def prioritize_tasks(self, tasks: List[ExecutionTask]) -> List[ExecutionTask]:
        """タスクの優先度付け"""
        # 優先度、作成時刻でソート
        return sorted(tasks, key=lambda t: (-t.priority, t.created_at))

class DynamicOrderingManager:
    """動的順序付け管理"""
    
    def __init__(self):
        self.execution_history: List[ExecutionResult] = []
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger(f"{__name__}.DynamicOrderingManager")
        self._lock = threading.Lock()
        
    def update_performance(self, result: ExecutionResult):
        """パフォーマンス情報の更新"""
        with self._lock:
            strategy = result.strategy_name
            
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "execution_count": 0,
                    "last_updated": datetime.now().timestamp()
                }
                
            perf = self.strategy_performance[strategy]
            count = perf["execution_count"]
            
            # 成功率更新
            current_success = perf["success_rate"] * count
            new_success = current_success + (1.0 if result.success else 0.0)
            perf["success_rate"] = new_success / (count + 1)
            
            # 平均実行時間更新
            current_time = perf["avg_execution_time"] * count
            new_time = current_time + result.execution_time
            perf["avg_execution_time"] = new_time / (count + 1)
            
            perf["execution_count"] = count + 1
            perf["last_updated"] = datetime.now().timestamp()
            
            # 履歴に追加
            self.execution_history.append(result)
            if len(self.execution_history) > 1000:
                self.execution_history.pop(0)
                
    def get_strategy_score(self, strategy: str) -> float:
        """戦略のスコア計算"""
        if strategy not in self.strategy_performance:
            return 0.5  # デフォルトスコア
            
        perf = self.strategy_performance[strategy]
        success_weight = 0.6
        time_weight = 0.4
        
        # 成功率スコア
        success_score = perf["success_rate"]
        
        # 実行時間スコア（高速ほど高スコア）
        max_time = 180.0  # 最大実行時間の想定
        time_score = max(0.0, 1.0 - (perf["avg_execution_time"] / max_time))
        
        total_score = success_weight * success_score + time_weight * time_score
        return min(1.0, max(0.0, total_score))
        
    def order_strategies(self, strategies: List[str]) -> List[str]:
        """戦略の動的順序付け"""
        strategy_scores = [
            (strategy, self.get_strategy_score(strategy)) 
            for strategy in strategies
        ]
        
        # スコア順でソート
        ordered = sorted(strategy_scores, key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in ordered]

class StrategyExecutionCoordinator:
    """戦略実行調整器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.pipeline = StrategyExecutionPipeline()
        self.resource_monitor = ResourceMonitor()
        self.load_balancer = LoadBalancer(self.config.get("coordination", {}).get("parallel_strategies", 4))
        self.ordering_manager = DynamicOrderingManager()
        self.logger = logging.getLogger(__name__)
        
        # 統計情報
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "start_time": datetime.now()
        }
        self._stats_lock = threading.Lock()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "composite_execution_config.json"
            )
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}. Using default config.")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "coordination": {
                "execution_mode": "adaptive",
                "parallel_strategies": 4,
                "sequential_threshold": 8,
                "dynamic_ordering": True,
                "load_balancing": True
            },
            "thresholds": {
                "max_execution_time": 180,
                "cpu_usage_limit": 80,
                "memory_limit_mb": 512
            }
        }
        
    def execute_strategies(self, strategies: List[str], 
                         market_data: pd.DataFrame,
                         parameters: Optional[Dict[str, Any]] = None) -> List[ExecutionResult]:
        """複数戦略の協調実行"""
        start_time = time.time()
        parameters = parameters or {}
        
        # 実行タスクの作成
        tasks = []
        for i, strategy in enumerate(strategies):
            task = ExecutionTask(
                task_id=f"task_{strategy}_{int(time.time() * 1000)}_{i}",
                strategy_name=strategy,
                market_data=market_data.copy(),
                parameters={**parameters, "strategy_name": strategy},
                priority=i
            )
            tasks.append(task)
            
        # 動的順序付け
        if self.config.get("coordination", {}).get("dynamic_ordering", True):
            ordered_strategies = self.ordering_manager.order_strategies(strategies)
            tasks = sorted(tasks, key=lambda t: ordered_strategies.index(t.strategy_name))
            
        # リソース状況の確認
        resources = self.resource_monitor.get_current_resources()
        
        # 実行モードの決定
        execution_mode = self._determine_execution_mode(len(tasks), resources)
        
        # 実行
        results = []
        if execution_mode in [ExecutionMode.PARALLEL, ExecutionMode.HYBRID]:
            results = self._execute_parallel(tasks, resources)
        else:
            results = self._execute_sequential(tasks)
            
        # 統計更新
        total_time = time.time() - start_time
        self._update_execution_stats(results, total_time)
        
        # パフォーマンス更新
        for result in results:
            self.ordering_manager.update_performance(result)
            
        return results
        
    def _determine_execution_mode(self, task_count: int, 
                                resources: SystemResources) -> ExecutionMode:
        """実行モードの決定"""
        config_mode = self.config.get("coordination", {}).get("execution_mode", "adaptive")
        
        if config_mode == "sequential":
            return ExecutionMode.SEQUENTIAL
        elif config_mode == "parallel":
            return ExecutionMode.PARALLEL
        elif config_mode == "adaptive":
            # アダプティブ判定
            threshold = self.config.get("coordination", {}).get("sequential_threshold", 8)
            
            if task_count >= threshold:
                return ExecutionMode.SEQUENTIAL
            elif self.load_balancer.should_use_parallel(task_count, resources):
                return ExecutionMode.PARALLEL
            else:
                return ExecutionMode.SEQUENTIAL
        else:  # hybrid
            return ExecutionMode.HYBRID
            
    def _execute_sequential(self, tasks: List[ExecutionTask]) -> List[ExecutionResult]:
        """逐次実行"""
        results = []
        
        for task in tasks:
            self.logger.info(f"Executing task {task.task_id} sequentially")
            result = self._execute_single_task(task)
            results.append(result)
            
            # 失敗したタスクのリトライ
            if not result.success and task.retry_count < task.max_retries:
                task.retry_count += 1
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                retry_result = self._execute_single_task(task)
                if retry_result.success:
                    results[-1] = retry_result
                    
        return results
        
    def _execute_parallel(self, tasks: List[ExecutionTask], 
                        resources: SystemResources) -> List[ExecutionResult]:
        """並列実行"""
        max_workers = self.load_balancer.get_optimal_workers(len(tasks), resources)
        results = []
        
        self.logger.info(f"Executing {len(tasks)} tasks in parallel with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # タスクの投入
            future_to_task = {
                executor.submit(self._execute_single_task, task): task 
                for task in tasks
            }
            
            # 結果の収集
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=task.timeout)
                    results.append(result)
                    
                    # リトライ処理
                    if not result.success and task.retry_count < task.max_retries:
                        task.retry_count += 1
                        retry_future = executor.submit(self._execute_single_task, task)
                        retry_result = retry_future.result(timeout=task.timeout)
                        if retry_result.success:
                            # 元の結果を置換
                            for i, r in enumerate(results):
                                if r.task_id == result.task_id:
                                    results[i] = retry_result
                                    break
                                    
                except TimeoutError:
                    self.logger.error(f"Task {task.task_id} timed out")
                    results.append(ExecutionResult(
                        task_id=task.task_id,
                        strategy_name=task.strategy_name,
                        success=False,
                        error_message="Task execution timed out"
                    ))
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {str(e)}")
                    results.append(ExecutionResult(
                        task_id=task.task_id,
                        strategy_name=task.strategy_name,
                        success=False,
                        error_message=str(e)
                    ))
                    
        return results
        
    def _execute_single_task(self, task: ExecutionTask) -> ExecutionResult:
        """単一タスクの実行"""
        start_time = time.time()
        start_resources = self.resource_monitor.get_current_resources()
        
        try:
            # パイプライン実行
            context = self.pipeline.execute(
                market_data=task.market_data,
                parameters=task.parameters
            )
            
            execution_time = time.time() - start_time
            end_resources = self.resource_monitor.get_current_resources()
            
            # リソース使用量の計算
            resource_usage = {
                "cpu_usage": end_resources.cpu_percent - start_resources.cpu_percent,
                "memory_delta": end_resources.memory_percent - start_resources.memory_percent,
                "execution_time": execution_time
            }
            
            # 成功判定
            success = self._evaluate_execution_success(context)
            
            return ExecutionResult(
                task_id=task.task_id,
                strategy_name=task.strategy_name,
                context=context,
                success=success,
                execution_time=execution_time,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                task_id=task.task_id,
                strategy_name=task.strategy_name,
                success=False,
                execution_time=execution_time,
                error_message=error_msg
            )
            
    def _evaluate_execution_success(self, context: PipelineContext) -> bool:
        """実行成功の評価"""
        if not context.stage_results:
            return False
            
        # クリティカルステージの成功チェック
        critical_stages = ["strategy_selection", "weight_calculation", "signal_integration", "execution"]
        
        for stage_id in critical_stages:
            if stage_id in context.stage_results:
                if not context.stage_results[stage_id].is_success():
                    return False
                    
        return True
        
    def _update_execution_stats(self, results: List[ExecutionResult], 
                              total_time: float):
        """実行統計の更新"""
        with self._stats_lock:
            self.execution_stats["total_executions"] += len(results)
            successful = sum(1 for r in results if r.success)
            self.execution_stats["successful_executions"] += successful
            self.execution_stats["failed_executions"] += len(results) - successful
            self.execution_stats["total_execution_time"] += total_time
            
    def get_coordination_summary(self) -> Dict[str, Any]:
        """調整サマリーの取得"""
        with self._stats_lock:
            uptime = (datetime.now() - self.execution_stats["start_time"]).total_seconds()
            
            summary = {
                "uptime_seconds": uptime,
                "total_executions": self.execution_stats["total_executions"],
                "success_rate": (
                    self.execution_stats["successful_executions"] / 
                    max(self.execution_stats["total_executions"], 1)
                ),
                "average_execution_time": (
                    self.execution_stats["total_execution_time"] / 
                    max(self.execution_stats["total_executions"], 1)
                ),
                "current_resources": self.resource_monitor.get_current_resources().__dict__,
                "strategy_performance": self.ordering_manager.strategy_performance,
                "config": self.config
            }
            
        return summary
        
    def optimize_coordination(self) -> Dict[str, Any]:
        """調整の最適化"""
        current_resources = self.resource_monitor.get_current_resources()
        
        recommendations = []
        
        # CPU使用率チェック
        if current_resources.cpu_percent > 80:
            recommendations.append("Consider reducing parallel_strategies due to high CPU usage")
            
        # メモリ使用率チェック
        if current_resources.memory_percent > 85:
            recommendations.append("Consider sequential execution due to high memory usage")
            
        # 実行履歴から推奨事項を生成
        if self.execution_stats["total_executions"] > 10:
            success_rate = (
                self.execution_stats["successful_executions"] / 
                self.execution_stats["total_executions"]
            )
            
            if success_rate < 0.8:
                recommendations.append("Consider increasing timeout or retry attempts")
                
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "current_performance": self.get_coordination_summary(),
            "recommendations": recommendations,
            "suggested_config_changes": self._generate_config_recommendations(current_resources)
        }
        
        return optimization_result
        
    def _generate_config_recommendations(self, resources: SystemResources) -> Dict[str, Any]:
        """設定推奨事項の生成"""
        suggestions = {}
        
        # リソースベースの推奨事項
        if resources.cpu_percent > 70:
            suggestions["parallel_strategies"] = max(2, self.config.get("coordination", {}).get("parallel_strategies", 4) - 1)
            
        if resources.memory_percent > 80:
            suggestions["execution_mode"] = "sequential"
            
        # パフォーマンスベースの推奨事項
        if self.execution_stats["total_executions"] > 5:
            avg_time = (
                self.execution_stats["total_execution_time"] / 
                self.execution_stats["total_executions"]
            )
            
            if avg_time > 120:
                suggestions["max_execution_time"] = int(avg_time * 1.5)
                
        return suggestions

if __name__ == "__main__":
    # テスト用のサンプル実行
    coordinator = StrategyExecutionCoordinator()
    
    # サンプルデータ生成
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='1H'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 複数戦略実行
    strategies = ["strategy_a", "strategy_b", "strategy_c"]
    results = coordinator.execute_strategies(strategies, sample_data)
    
    # 結果表示
    print("Coordination Results:")
    for result in results:
        print(f"  {result.strategy_name}: {'Success' if result.success else 'Failed'} ({result.execution_time:.2f}s)")
        
    # サマリー表示
    summary = coordinator.get_coordination_summary()
    print(f"\nSuccess Rate: {summary['success_rate']:.2%}")
    print(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
