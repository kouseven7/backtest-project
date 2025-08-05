"""
Module: Resource Allocation Engine
File: resource_allocation_engine.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  動的リソース配分エンジン

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 動的負荷分散による最適リソース配分
  - システムリソース監視と調整
  - 実行方式の動的選択 (Thread/Process/Async)
  - パフォーマンス履歴による学習機能
"""

import os
import sys
import json
import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# ロガー設定
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """実行モード"""
    THREAD = "thread"
    PROCESS = "process" 
    ASYNC = "async"
    AUTO = "auto"

class Priority(Enum):
    """優先度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ResourceType(Enum):
    """リソース種別"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class SystemLoad:
    """システム負荷情報"""
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    timestamp: datetime
    
    def is_high_load(self, thresholds: Dict[str, float]) -> bool:
        """高負荷状態かチェック"""
        return (
            self.cpu_percent > thresholds.get('cpu', 0.8) or
            self.memory_percent > thresholds.get('memory', 0.8)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class ResourceRequirement:
    """リソース要求"""
    cpu: float = 0.1
    memory_mb: int = 64
    disk_mb: int = 0
    network_mbps: float = 0.0
    priority: Priority = Priority.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['priority'] = self.priority.value
        return result

@dataclass
class ResourceAllocation:
    """リソース配分結果"""
    strategy_name: str
    execution_mode: ExecutionMode
    allocated_cpu: float
    allocated_memory_mb: int
    priority: Priority
    estimated_duration: float
    allocation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['execution_mode'] = self.execution_mode.value
        result['priority'] = self.priority.value
        result['allocation_timestamp'] = self.allocation_timestamp.isoformat()
        return result

@dataclass
class OptimizationResult:
    """最適化結果"""
    optimized_allocations: List[ResourceAllocation]
    total_cpu_usage: float
    total_memory_usage: float
    estimated_completion_time: float
    optimization_score: float
    recommendations: List[str]

class SystemResourceMonitor:
    """システムリソース監視"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.load_history: List[SystemLoad] = []
        self.max_history_length = 100
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System resource monitoring started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("System resource monitoring stopped")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                load = self._collect_system_load()
                self.load_history.append(load)
                
                # 履歴長制限
                if len(self.load_history) > self.max_history_length:
                    self.load_history.pop(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(5)  # エラー時は少し待機
    
    def _collect_system_load(self) -> SystemLoad:
        """システム負荷収集"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # ディスクIO
            disk_io = psutil.disk_io_counters()
            disk_stats = {
                'read_bytes': float(disk_io.read_bytes) if disk_io else 0.0,
                'write_bytes': float(disk_io.write_bytes) if disk_io else 0.0
            }
            
            # ネットワークIO
            network_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': float(network_io.bytes_sent) if network_io else 0.0,
                'bytes_recv': float(network_io.bytes_recv) if network_io else 0.0
            }
            
            return SystemLoad(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io=disk_stats,
                network_io=network_stats,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system load: {e}")
            return SystemLoad(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_io={},
                network_io={},
                timestamp=datetime.now()
            )
    
    def get_current_load(self) -> Optional[SystemLoad]:
        """現在の負荷取得"""
        if self.load_history:
            return self.load_history[-1]
        return self._collect_system_load()
    
    def get_average_load(self, minutes: int = 5) -> Optional[SystemLoad]:
        """平均負荷取得"""
        if not self.load_history:
            return None
            
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_loads = [
            load for load in self.load_history 
            if load.timestamp > cutoff_time
        ]
        
        if not recent_loads:
            return None
        
        avg_cpu = sum(load.cpu_percent for load in recent_loads) / len(recent_loads)
        avg_memory = sum(load.memory_percent for load in recent_loads) / len(recent_loads)
        
        return SystemLoad(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_io={},
            network_io={},
            timestamp=datetime.now()
        )

class ResourceAllocationEngine:
    """動的リソース配分エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.resource_monitor = SystemResourceMonitor(
            monitoring_interval=self.config.get('monitoring', {}).get('resource_monitoring_interval', 1.0)
        )
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.allocation_history: List[ResourceAllocation] = []
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # 設定に基づいてプール初期化
        self._initialize_executor_pools()
        
        # リソース監視開始
        self.resource_monitor.start_monitoring()
        
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
                "mode_selection": {
                    "cpu_intensive_threshold": 0.7,
                    "io_bound_threshold": 0.3,
                    "memory_threshold": 0.8
                }
            },
            "load_balancing": {
                "cpu_threshold": 0.8,
                "memory_threshold": 0.7,
                "adjustment_interval": 5.0
            },
            "strategy_profiles": {}
        }
    
    def _initialize_executor_pools(self):
        """実行プール初期化"""
        execution_config = self.config.get('execution_modes', {})
        
        # スレッドプール
        thread_pool_size = execution_config.get('thread_pool_size', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # プロセスプール
        process_pool_size = execution_config.get('process_pool_size', 2)
        self.process_pool = ProcessPoolExecutor(max_workers=process_pool_size)
        
        logger.info(f"Initialized thread pool ({thread_pool_size}) and process pool ({process_pool_size})")
    
    def determine_execution_mode(self, strategy: str, current_load: SystemLoad) -> ExecutionMode:
        """実行方式決定"""
        strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
        preferred_mode = strategy_profile.get('execution_type', 'auto')
        
        # プロファイルで明示指定されている場合
        if preferred_mode != 'auto':
            try:
                return ExecutionMode(preferred_mode)
            except ValueError:
                logger.warning(f"Invalid execution mode {preferred_mode} for {strategy}, using auto")
        
        # 自動選択ロジック
        mode_config = self.config.get('execution_modes', {}).get('mode_selection', {})
        cpu_threshold = mode_config.get('cpu_intensive_threshold', 0.7)
        memory_threshold = mode_config.get('memory_threshold', 0.8)
        
        # システム負荷に基づく判定
        if current_load.cpu_percent > cpu_threshold * 100:
            # CPU負荷が高い場合はプロセス並列
            return ExecutionMode.PROCESS
        elif current_load.memory_percent > memory_threshold * 100:
            # メモリ負荷が高い場合は軽量なasync
            return ExecutionMode.ASYNC
        else:
            # 通常はスレッド並列
            return ExecutionMode.THREAD
    
    def allocate_resources(self, strategies: List[str]) -> List[ResourceAllocation]:
        """リソース配分"""
        current_load = self.resource_monitor.get_current_load()
        if not current_load:
            current_load = SystemLoad(0, 0, {}, {}, datetime.now())
        
        allocations = []
        total_cpu_requested = 0.0
        total_memory_requested = 0
        
        # 各戦略のリソース要求を計算
        for strategy in strategies:
            strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
            resource_req = strategy_profile.get('resource_requirements', {})
            
            # 実行方式決定
            execution_mode = self.determine_execution_mode(strategy, current_load)
            
            # リソース要求
            cpu_req = resource_req.get('cpu', 0.1)
            memory_req = int(str(resource_req.get('memory', '64MB')).replace('MB', ''))
            
            # 優先度
            priority_str = strategy_profile.get('priority', 'medium')
            priority = Priority.MEDIUM
            try:
                priority_map = {'low': Priority.LOW, 'medium': Priority.MEDIUM, 
                              'high': Priority.HIGH, 'critical': Priority.CRITICAL}
                priority = priority_map.get(priority_str.lower(), Priority.MEDIUM)
            except:
                priority = Priority.MEDIUM
            
            # 予想実行時間
            estimated_duration = strategy_profile.get('expected_duration', 30.0)
            
            allocation = ResourceAllocation(
                strategy_name=strategy,
                execution_mode=execution_mode,
                allocated_cpu=cpu_req,
                allocated_memory_mb=memory_req,
                priority=priority,
                estimated_duration=estimated_duration,
                allocation_timestamp=datetime.now()
            )
            
            allocations.append(allocation)
            total_cpu_requested += cpu_req
            total_memory_requested += memory_req
        
        # リソース制約チェックと調整
        allocations = self._adjust_for_resource_constraints(
            allocations, current_load, total_cpu_requested, total_memory_requested
        )
        
        # 配分履歴に記録
        self.allocation_history.extend(allocations)
        
        return allocations
    
    def _adjust_for_resource_constraints(
        self, 
        allocations: List[ResourceAllocation],
        current_load: SystemLoad,
        total_cpu_requested: float,
        total_memory_requested: int
    ) -> List[ResourceAllocation]:
        """リソース制約による調整"""
        
        # CPU使用率が高い場合の調整
        if current_load.cpu_percent > 80:
            cpu_scale_factor = 0.7  # CPU要求を30%削減
            for allocation in allocations:
                allocation.allocated_cpu *= cpu_scale_factor
            logger.info("CPU usage is high, reduced CPU allocation by 30%")
        
        # メモリ使用率が高い場合の調整
        if current_load.memory_percent > 80:
            memory_scale_factor = 0.8  # メモリ要求を20%削減
            for allocation in allocations:
                allocation.allocated_memory_mb = int(allocation.allocated_memory_mb * memory_scale_factor)
            logger.info("Memory usage is high, reduced memory allocation by 20%")
        
        # 優先度による調整
        allocations.sort(key=lambda x: x.priority.value, reverse=True)
        
        return allocations
    
    def optimize_load_balancing(self, execution_stats: List[Dict[str, Any]]) -> OptimizationResult:
        """負荷分散最適化"""
        
        # パフォーマンス履歴更新
        for stat in execution_stats:
            strategy_name = stat.get('strategy_name')
            if strategy_name:
                if strategy_name not in self.performance_history:
                    self.performance_history[strategy_name] = []
                self.performance_history[strategy_name].append(stat)
                
                # 履歴長制限
                max_history = self.config.get('load_balancing', {}).get('performance_history_length', 100)
                if len(self.performance_history[strategy_name]) > max_history:
                    self.performance_history[strategy_name].pop(0)
        
        # 最適化実行
        current_load = self.resource_monitor.get_current_load()
        if not current_load:
            current_load = SystemLoad(0, 0, {}, {}, datetime.now())
        
        # 最適化戦略の決定
        strategies = list(self.performance_history.keys())
        optimized_allocations = self.allocate_resources(strategies)
        
        # 総リソース使用量計算
        total_cpu = sum(alloc.allocated_cpu for alloc in optimized_allocations)
        total_memory = sum(alloc.allocated_memory_mb for alloc in optimized_allocations)
        
        # 推定完了時間（最大の推定実行時間）
        estimated_completion = max(
            (alloc.estimated_duration for alloc in optimized_allocations),
            default=0.0
        )
        
        # 最適化スコア計算（シンプルな効率指標）
        efficiency_score = 0.0
        if strategies:
            avg_cpu_efficiency = total_cpu / len(strategies)
            avg_memory_efficiency = (total_memory / 1024) / len(strategies)  # GB換算
            efficiency_score = min(avg_cpu_efficiency * 10, avg_memory_efficiency)
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(
            current_load, optimized_allocations, total_cpu, total_memory
        )
        
        result = OptimizationResult(
            optimized_allocations=optimized_allocations,
            total_cpu_usage=total_cpu,
            total_memory_usage=total_memory,
            estimated_completion_time=estimated_completion,
            optimization_score=efficiency_score,
            recommendations=recommendations
        )
        
        return result
    
    def _generate_recommendations(
        self, 
        current_load: SystemLoad,
        allocations: List[ResourceAllocation],
        total_cpu: float,
        total_memory: int
    ) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []
        
        # CPU使用率チェック
        if current_load.cpu_percent > 90:
            recommendations.append("CPU使用率が非常に高いため、戦略数を削減することを推奨")
        elif total_cpu > 1.0:
            recommendations.append("CPU要求総量がシステム能力を超過、並列度を下げることを推奨")
        
        # メモリ使用率チェック
        if current_load.memory_percent > 90:
            recommendations.append("メモリ使用率が非常に高いため、メモリ使用量の多い戦略を制限")
        elif total_memory > 2048:  # 2GB
            recommendations.append("メモリ要求が大きいため、軽量な戦略を優先することを推奨")
        
        # 実行方式推奨
        process_count = sum(1 for alloc in allocations if alloc.execution_mode == ExecutionMode.PROCESS)
        if process_count > 3:
            recommendations.append("プロセス並列が多すぎるため、一部をスレッド並列に変更を推奨")
        
        # 優先度バランス
        high_priority_count = sum(1 for alloc in allocations if alloc.priority == Priority.HIGH)
        if high_priority_count > len(allocations) * 0.7:
            recommendations.append("高優先度戦略が多すぎるため、優先度の見直しを推奨")
        
        return recommendations
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """リソース使用統計取得"""
        current_load = self.resource_monitor.get_current_load()
        average_load = self.resource_monitor.get_average_load(5)
        
        stats = {
            'current_load': current_load.to_dict() if current_load else None,
            'average_load_5min': average_load.to_dict() if average_load else None,
            'allocation_count': len(self.allocation_history),
            'performance_history_count': {
                strategy: len(history) 
                for strategy, history in self.performance_history.items()
            }
        }
        
        return stats
    
    def shutdown(self):
        """シャットダウン"""
        logger.info("Shutting down Resource Allocation Engine...")
        
        # リソース監視停止
        self.resource_monitor.stop_monitoring()
        
        # 実行プール終了
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Resource Allocation Engine shutdown complete")

def create_system_load_for_demo() -> SystemLoad:
    """デモ用システム負荷作成"""
    return SystemLoad(
        cpu_percent=45.0,
        memory_percent=60.0,
        disk_io={'read_bytes': 1000000, 'write_bytes': 500000},
        network_io={'bytes_sent': 2000000, 'bytes_recv': 1500000},
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Resource Allocation Engine - Demo")
    print("=" * 60)
    
    try:
        # エンジン初期化
        engine = ResourceAllocationEngine()
        
        # デモ戦略リスト
        demo_strategies = ["VWAPBounceStrategy", "GCStrategy", "BreakoutStrategy", "OpeningGapStrategy"]
        
        print(f"\n🎯 Testing resource allocation for strategies: {demo_strategies}")
        
        # リソース配分実行
        allocations = engine.allocate_resources(demo_strategies)
        
        print(f"\n📊 Resource Allocation Results:")
        print("-" * 50)
        for allocation in allocations:
            print(f"Strategy: {allocation.strategy_name}")
            print(f"  Execution Mode: {allocation.execution_mode.value}")
            print(f"  CPU: {allocation.allocated_cpu:.2f}")
            print(f"  Memory: {allocation.allocated_memory_mb}MB")
            print(f"  Priority: {allocation.priority.name}")
            print(f"  Est. Duration: {allocation.estimated_duration}s")
            print()
        
        # 最適化実行
        demo_stats: List[Dict[str, Any]] = [
            {'strategy_name': 'VWAPBounceStrategy', 'execution_time': 12.5, 'cpu_usage': 0.1},
            {'strategy_name': 'GCStrategy', 'execution_time': 28.3, 'cpu_usage': 0.3},
        ]
        
        print("🔄 Running load balancing optimization...")
        optimization = engine.optimize_load_balancing(demo_stats)
        
        print(f"\n📈 Optimization Results:")
        print(f"  Total CPU Usage: {optimization.total_cpu_usage:.2f}")
        print(f"  Total Memory Usage: {optimization.total_memory_usage}MB")
        print(f"  Est. Completion Time: {optimization.estimated_completion_time:.1f}s")
        print(f"  Optimization Score: {optimization.optimization_score:.2f}")
        
        if optimization.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(optimization.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # リソース使用統計
        stats = engine.get_resource_usage_stats()
        print(f"\n📊 Resource Usage Statistics:")
        if stats['current_load']:
            load = stats['current_load']
            print(f"  Current CPU: {load['cpu_percent']:.1f}%")
            print(f"  Current Memory: {load['memory_percent']:.1f}%")
        
        print(f"  Allocation History: {stats['allocation_count']} entries")
        
        print("\n✅ Resource Allocation Engine demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'engine' in locals():
            engine.shutdown()
