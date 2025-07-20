"""
Module: Multi-Strategy Coordination Manager
File: multi_strategy_coordination_manager.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  マルチ戦略調整マネージャー（中心制御）

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 全コンポーネント統合管理・制御
  - 段階的フォールバック実行制御
  - 高可用性・信頼性保証機能
  - 統合ダッシュボード・状態管理
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import traceback
import hashlib

# プロジェクトモジュールをインポート
try:
    # from resource_allocation_engine import ResourceAllocationEngine, ResourceAllocation, ExecutionMode
    # from strategy_dependency_resolver import StrategyDependencyResolver, DependencyResolution
    # from concurrent_execution_scheduler import ConcurrentExecutionScheduler, ExecutionTask, ExecutionResult, ExecutionStatus
    # from execution_monitoring_system import ExecutionMonitoringSystem, Alert, AlertSeverity
    pass
except ImportError as e:
    # スタンドアロンテスト用フォールバック
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import project modules: {e}, using fallback definitions")
    
    class ExecutionMode(Enum):
        SINGLE = "single"
        PARALLEL = "parallel"
        SEQUENTIAL = "sequential"
    
    class ExecutionStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class AlertSeverity(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
    
    @dataclass
    class ExecutionResult:
        result_id: str
        status: ExecutionStatus
        message: str = ""
    
    @dataclass
    class Alert:
        alert_id: str
        severity: AlertSeverity
        message: str
        strategy_name: Optional[str] = None
    
    @dataclass 
    class ExecutionTask:
        task_id: str
        strategy_name: str
        parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Mock classes for unavailable imports
    class ResourceAllocationEngine:
        def allocate_resources(self, strategies): return []
        def shutdown(self): pass
    
    class StrategyDependencyResolver:
        def resolve_dependencies(self, strategies): return DependencyResolution({}, [], {})
    
    class ConcurrentExecutionScheduler:
        def __init__(self, max_workers=4): self.max_workers = max_workers
        def schedule_execution(self, tasks): return {}
        def wait_for_completion(self): return []
        def shutdown(self): pass
    
    class ExecutionMonitoringSystem:
        def __init__(self): pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def get_alerts(self): return []
        THREAD = "thread"
        PROCESS = "process"
        ASYNC = "async"
        AUTO = "auto"
    
    class ExecutionStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        TIMEOUT = "timeout"
    
    class AlertSeverity(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

# ロガー設定
logger = logging.getLogger(__name__)

class CoordinationState(Enum):
    """調整状態"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    FALLBACK = "fallback"
    EMERGENCY = "emergency"
    SHUTTING_DOWN = "shutting_down"

class FallbackLevel(Enum):
    """フォールバックレベル"""
    NONE = "none"
    INDIVIDUAL = "individual"    # 個別戦略レベル
    GROUP = "group"              # グループレベル
    SYSTEM = "system"            # システム全体レベル
    EMERGENCY = "emergency"      # 緊急停止レベル

class CoordinationMode(Enum):
    """調整モード"""
    AUTONOMOUS = "autonomous"    # 自律実行モード
    SUPERVISED = "supervised"   # 監視付きモード
    MANUAL = "manual"           # 手動制御モード

@dataclass
class ResourceAllocation:
    """リソース配分"""
    strategy_id: str
    cpu_allocation: float  # CPU使用率 (0.0-1.0)
    memory_allocation: float  # メモリ使用量 (MB)
    priority: int  # 優先度 (1-10)

@dataclass
class DependencyResolution:
    """依存関係解決"""
    dependencies: Dict[str, List[str]]  # 戦略ID -> 依存戦略IDs
    execution_order: List[str]  # 実行順序
    conflict_resolution: Dict[str, str]  # 競合解決方法

@dataclass
class CoordinationPlan:
    """調整計画"""
    plan_id: str
    strategies: List[str]
    resource_allocations: List[ResourceAllocation]
    dependency_resolution: DependencyResolution
    execution_timeline: List[Tuple[datetime, str, str]]  # (時刻, 戦略, アクション)
    fallback_scenarios: Dict[FallbackLevel, Dict[str, Any]]
    estimated_completion_time: datetime
    risk_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['estimated_completion_time'] = self.estimated_completion_time.isoformat()
        result['execution_timeline'] = [
            (dt.isoformat(), strategy, action) 
            for dt, strategy, action in self.execution_timeline
        ]
        return result

@dataclass
class CoordinationStatus:
    """調整状況"""
    state: CoordinationState
    current_plan: Optional[str] = None
    active_strategies: List[str] = field(default_factory=list)
    completed_strategies: List[str] = field(default_factory=list)
    failed_strategies: List[str] = field(default_factory=list)
    fallback_level: FallbackLevel = FallbackLevel.NONE
    last_update: datetime = field(default_factory=datetime.now)
    alerts_count: int = 0
    system_health_score: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['state'] = self.state.value
        result['fallback_level'] = self.fallback_level.value
        result['last_update'] = self.last_update.isoformat()
        return result

@dataclass 
class ExecutionContext:
    """実行コンテキスト"""
    execution_id: str
    plan: CoordinationPlan
    start_time: datetime
    expected_end_time: datetime
    mode: CoordinationMode = CoordinationMode.AUTONOMOUS
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FallbackManager:
    """フォールバック管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fallback_config = config.get('fallback_system', {})
        self.active_fallbacks: Dict[FallbackLevel, bool] = {
            level: False for level in FallbackLevel
        }
        self.fallback_history: List[Dict[str, Any]] = []
        
    def should_trigger_fallback(
        self, 
        current_status: CoordinationStatus,
        execution_results: Dict[str, 'ExecutionResult'],
        alerts: List['Alert']
    ) -> Optional[FallbackLevel]:
        """フォールバック発動判定"""
        
        # 緊急レベル: システム致命的エラー
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) >= self.fallback_config.get('emergency_alert_threshold', 3):
            return FallbackLevel.EMERGENCY
        
        # システムレベル: 全体成功率低下
        if execution_results:
            success_count = sum(1 for r in execution_results.values() if r.status == ExecutionStatus.COMPLETED)
            total_count = len(execution_results)
            success_rate = success_count / total_count if total_count > 0 else 1.0
            
            if success_rate < self.fallback_config.get('system_failure_threshold', 0.3):
                return FallbackLevel.SYSTEM
        
        # グループレベル: 多数の戦略失敗
        failed_count = len(current_status.failed_strategies)
        total_count = len(current_status.active_strategies) + len(current_status.completed_strategies) + failed_count
        
        if total_count > 0:
            failure_rate = failed_count / total_count
            if failure_rate > self.fallback_config.get('group_failure_threshold', 0.4):
                return FallbackLevel.GROUP
        
        # 個別レベル: 個別戦略の問題
        error_alerts = [a for a in alerts if a.severity == AlertSeverity.ERROR and a.strategy_name]
        strategy_errors = {}
        for alert in error_alerts:
            if alert.strategy_name:
                strategy_errors[alert.strategy_name] = strategy_errors.get(alert.strategy_name, 0) + 1
        
        for strategy, error_count in strategy_errors.items():
            if error_count >= self.fallback_config.get('individual_error_threshold', 3):
                return FallbackLevel.INDIVIDUAL
        
        return None
    
    def execute_fallback(
        self, 
        level: FallbackLevel, 
        context: ExecutionContext,
        failed_strategies: List[str]
    ) -> Dict[str, Any]:
        """フォールバック実行"""
        
        fallback_action = {
            'level': level.value,
            'timestamp': datetime.now().isoformat(),
            'failed_strategies': failed_strategies,
            'actions_taken': [],
            'recovery_plan': None
        }
        
        try:
            if level == FallbackLevel.INDIVIDUAL:
                # 個別戦略フォールバック: 失敗戦略のみ停止・リトライ
                fallback_action['actions_taken'].extend([
                    f"戦略 {strategy} を停止" for strategy in failed_strategies
                ])
                
                # リトライ可能戦略の特定
                retryable_strategies = [
                    s for s in failed_strategies 
                    if self._is_retryable_strategy(s, context)
                ]
                
                if retryable_strategies:
                    fallback_action['recovery_plan'] = {
                        'type': 'retry',
                        'strategies': retryable_strategies,
                        'delay_seconds': self.fallback_config.get('retry_delay', 30)
                    }
                    fallback_action['actions_taken'].append(f"リトライ対象: {retryable_strategies}")
            
            elif level == FallbackLevel.GROUP:
                # グループレベルフォールバック: 関連戦略群を停止
                related_groups = self._identify_strategy_groups(context.plan.strategies)
                affected_groups = []
                
                for group in related_groups:
                    if any(strategy in failed_strategies for strategy in group):
                        affected_groups.extend(group)
                
                fallback_action['actions_taken'].append(f"戦略グループ停止: {affected_groups}")
                fallback_action['recovery_plan'] = {
                    'type': 'reduced_execution',
                    'excluded_strategies': affected_groups,
                    'continue_with': [s for s in context.plan.strategies if s not in affected_groups]
                }
            
            elif level == FallbackLevel.SYSTEM:
                # システムレベルフォールバック: 基本機能のみ継続
                essential_strategies = self._identify_essential_strategies(context.plan.strategies)
                
                fallback_action['actions_taken'].append("非必須戦略全停止")
                fallback_action['recovery_plan'] = {
                    'type': 'essential_only',
                    'essential_strategies': essential_strategies,
                    'degraded_mode': True
                }
            
            elif level == FallbackLevel.EMERGENCY:
                # 緊急フォールバック: 全実行停止
                fallback_action['actions_taken'].append("全戦略緊急停止")
                fallback_action['recovery_plan'] = {
                    'type': 'emergency_stop',
                    'manual_restart_required': True,
                    'safe_mode': True
                }
            
            self.active_fallbacks[level] = True
            self.fallback_history.append(fallback_action)
            
            logger.warning(f"Fallback executed: {level.value} level")
            
        except Exception as e:
            fallback_action['actions_taken'].append(f"フォールバック実行エラー: {str(e)}")
            logger.error(f"Fallback execution failed: {e}")
        
        return fallback_action
    
    def _is_retryable_strategy(self, strategy: str, context: ExecutionContext) -> bool:
        """戦略リトライ可能性判定"""
        # 設定からリトライ可能戦略を確認
        retryable_strategies = self.fallback_config.get('retryable_strategies', [])
        if retryable_strategies and strategy not in retryable_strategies:
            return False
        
        # リトライ回数制限チェック
        max_retries = self.fallback_config.get('max_retries_per_strategy', 3)
        current_retries = context.metadata.get('retry_counts', {}).get(strategy, 0)
        
        return current_retries < max_retries
    
    def _identify_strategy_groups(self, strategies: List[str]) -> List[List[str]]:
        """戦略グループ特定"""
        # 設定から戦略グループ情報を取得
        predefined_groups = self.fallback_config.get('strategy_groups', {})
        
        groups = []
        for group_name, group_strategies in predefined_groups.items():
            group_in_execution = [s for s in group_strategies if s in strategies]
            if group_in_execution:
                groups.append(group_in_execution)
        
        # グループ未定義の戦略は個別グループとして扱う
        grouped_strategies = set()
        for group in groups:
            grouped_strategies.update(group)
        
        ungrouped = [s for s in strategies if s not in grouped_strategies]
        for strategy in ungrouped:
            groups.append([strategy])
        
        return groups
    
    def _identify_essential_strategies(self, strategies: List[str]) -> List[str]:
        """必須戦略特定"""
        essential_strategies = self.fallback_config.get('essential_strategies', [])
        return [s for s in strategies if s in essential_strategies]

class MultiStrategyCoordinationManager:
    """マルチ戦略調整マネージャー"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.state = CoordinationState.IDLE
        self.coordination_mode = CoordinationMode.AUTONOMOUS
        
        # コンポーネント初期化
        self.resource_engine: Optional[ResourceAllocationEngine] = None
        self.dependency_resolver: Optional[StrategyDependencyResolver] = None
        self.execution_scheduler: Optional[ConcurrentExecutionScheduler] = None
        self.monitoring_system: Optional[ExecutionMonitoringSystem] = None
        
        # フォールバック管理器
        self.fallback_manager = FallbackManager(self.config)
        
        # 実行管理
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.current_status = CoordinationStatus(state=CoordinationState.IDLE)
        
        # 制御用
        self.coordination_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # 統計情報
        self.performance_stats = {
            'total_coordinations': 0,
            'successful_coordinations': 0,
            'failed_coordinations': 0,
            'fallback_activations': 0,
            'average_coordination_time': 0.0
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
            "coordination": {
                "mode": "autonomous",
                "max_concurrent_coordinations": 3,
                "coordination_timeout": 1800,
                "health_check_interval": 30
            },
            "fallback_system": {
                "enabled": True,
                "individual_error_threshold": 3,
                "group_failure_threshold": 0.4,
                "system_failure_threshold": 0.3,
                "emergency_alert_threshold": 3,
                "retry_delay": 30,
                "max_retries_per_strategy": 3,
                "essential_strategies": [],
                "retryable_strategies": [],
                "strategy_groups": {}
            }
        }
    
    def initialize_components(self):
        """コンポーネント初期化"""
        try:
            with self.coordination_lock:
                self.state = CoordinationState.INITIALIZING
            
            logger.info("Initializing coordination components...")
            
            # リソース配分エンジン
            if 'ResourceAllocationEngine' in globals():
                self.resource_engine = ResourceAllocationEngine(
                    os.path.join(os.path.dirname(__file__), 'coordination_config.json')
                )
                logger.info("Resource Allocation Engine initialized")
            
            # 依存関係リゾルバー
            if 'StrategyDependencyResolver' in globals():
                self.dependency_resolver = StrategyDependencyResolver(
                    os.path.join(os.path.dirname(__file__), 'coordination_config.json')
                )
                logger.info("Strategy Dependency Resolver initialized")
            
            # 実行スケジューラー
            if 'ConcurrentExecutionScheduler' in globals():
                self.execution_scheduler = ConcurrentExecutionScheduler(
                    os.path.join(os.path.dirname(__file__), 'coordination_config.json')
                )
                logger.info("Concurrent Execution Scheduler initialized")
            
            # 監視システム
            if 'ExecutionMonitoringSystem' in globals():
                self.monitoring_system = ExecutionMonitoringSystem(
                    os.path.join(os.path.dirname(__file__), 'coordination_config.json')
                )
                self.monitoring_system.start_monitoring()
                logger.info("Execution Monitoring System initialized")
            
            with self.coordination_lock:
                self.state = CoordinationState.IDLE
            
            logger.info("All coordination components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            with self.coordination_lock:
                self.state = CoordinationState.IDLE
            raise
    
    def create_coordination_plan(self, strategies: List[str]) -> CoordinationPlan:
        """調整計画作成"""
        logger.info(f"Creating coordination plan for {len(strategies)} strategies")
        
        with self.coordination_lock:
            self.state = CoordinationState.PLANNING
        
        try:
            # リソース配分計画
            resource_allocations = []
            if self.resource_engine:
                resource_allocations = self.resource_engine.allocate_resources(strategies)
            
            # 依存関係解決
            dependency_resolution = None
            if self.dependency_resolver:
                dependency_resolution = self.dependency_resolver.resolve_dependencies(strategies)
            
            # 実行タイムライン構築
            execution_timeline = self._build_execution_timeline(strategies, dependency_resolution)
            
            # 推定完了時間計算
            estimated_completion = self._calculate_completion_time(resource_allocations, dependency_resolution)
            
            # フォールバックシナリオ準備
            fallback_scenarios = self._prepare_fallback_scenarios(strategies)
            
            # リスク評価
            risk_assessment = self._assess_execution_risks(strategies, resource_allocations)
            
            plan = CoordinationPlan(
                plan_id=f"coord_plan_{int(time.time())}_{hashlib.md5('|'.join(strategies).encode()).hexdigest()[:8]}",
                strategies=strategies,
                resource_allocations=resource_allocations,
                dependency_resolution=dependency_resolution,
                execution_timeline=execution_timeline,
                fallback_scenarios=fallback_scenarios,
                estimated_completion_time=estimated_completion,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"Coordination plan created: {plan.plan_id}")
            
        except Exception as e:
            logger.error(f"Coordination plan creation failed: {e}")
            raise
        finally:
            with self.coordination_lock:
                self.state = CoordinationState.IDLE
        
        return plan
    
    def execute_coordination_plan(self, plan: CoordinationPlan) -> str:
        """調整計画実行"""
        logger.info(f"Executing coordination plan: {plan.plan_id}")
        
        # 実行コンテキスト作成
        execution_id = f"exec_{plan.plan_id}_{int(time.time())}"
        context = ExecutionContext(
            execution_id=execution_id,
            plan=plan,
            start_time=datetime.now(),
            expected_end_time=plan.estimated_completion_time,
            mode=self.coordination_mode,
            metadata={'retry_counts': {}}
        )
        
        with self.coordination_lock:
            self.active_executions[execution_id] = context
            self.state = CoordinationState.EXECUTING
            self.current_status = CoordinationStatus(
                state=CoordinationState.EXECUTING,
                current_plan=plan.plan_id,
                active_strategies=plan.strategies.copy()
            )
        
        # バックグラウンドで実行
        execution_thread = threading.Thread(
            target=self._execute_coordination_async,
            args=(context,),
            daemon=True
        )
        execution_thread.start()
        
        self.performance_stats['total_coordinations'] += 1
        
        return execution_id
    
    def _execute_coordination_async(self, context: ExecutionContext):
        """非同期調整実行"""
        try:
            logger.info(f"Starting coordination execution: {context.execution_id}")
            
            # 実行タスク作成
            execution_tasks = self._create_execution_tasks(context.plan)
            
            # 実行監視開始
            if self.monitoring_system:
                for strategy in context.plan.strategies:
                    self.monitoring_system.record_execution_start(strategy, f"coord_{context.execution_id}")
            
            # タスク実行
            if self.execution_scheduler:
                execution_results = self.execution_scheduler.execute_tasks(execution_tasks)
            else:
                execution_results = {}
            
            # 結果処理・監視
            self._process_execution_results(context, execution_results)
            
            # フォールバック制御チェック
            self._check_and_handle_fallbacks(context, execution_results)
            
            # 実行完了
            self._finalize_coordination(context, execution_results)
            
        except Exception as e:
            logger.error(f"Coordination execution failed: {e}")
            self._handle_coordination_failure(context, str(e))
    
    def _create_execution_tasks(self, plan: CoordinationPlan) -> List[ExecutionTask]:
        """実行タスク作成"""
        if 'ExecutionTask' not in globals():
            logger.warning("ExecutionTask not available, creating mock tasks")
            return []
        
        tasks = []
        for i, strategy in enumerate(plan.strategies):
            # リソース配分情報から実行モード決定
            execution_mode = ExecutionMode.THREAD
            for allocation in plan.resource_allocations:
                if allocation.strategy_name == strategy:
                    execution_mode = allocation.execution_mode
                    break
            
            # デモ用実行関数
            def strategy_function(strategy_name=strategy):
                time.sleep(5 + i)  # 実行時間シミュレーション
                return {'strategy': strategy_name, 'result': 'success', 'pnl': 100.0}
            
            task = ExecutionTask(
                task_id=f"task_{strategy}_{i}",
                strategy_name=strategy,
                execution_mode=execution_mode,
                function=strategy_function,
                args=(),
                kwargs={},
                timeout=300.0
            )
            tasks.append(task)
        
        return tasks
    
    def _process_execution_results(self, context: ExecutionContext, results: Dict[str, 'ExecutionResult']):
        """実行結果処理"""
        completed_strategies = []
        failed_strategies = []
        
        for task_id, result in results.items():
            strategy_name = result.strategy_name
            
            # 監視システムに結果記録
            if self.monitoring_system:
                self.monitoring_system.record_execution_completion(result)
            
            if result.status == ExecutionStatus.COMPLETED:
                completed_strategies.append(strategy_name)
            else:
                failed_strategies.append(strategy_name)
        
        # 状態更新
        with self.coordination_lock:
            self.current_status.completed_strategies = completed_strategies
            self.current_status.failed_strategies = failed_strategies
            self.current_status.active_strategies = [
                s for s in context.plan.strategies 
                if s not in completed_strategies and s not in failed_strategies
            ]
            self.current_status.last_update = datetime.now()
    
    def _check_and_handle_fallbacks(self, context: ExecutionContext, results: Dict[str, 'ExecutionResult']):
        """フォールバック制御確認・処理"""
        # アラート取得
        alerts = []
        if self.monitoring_system:
            alerts = self.monitoring_system.alert_manager.get_active_alerts()
        
        # フォールバック発動判定
        fallback_level = self.fallback_manager.should_trigger_fallback(
            self.current_status, results, alerts
        )
        
        if fallback_level and fallback_level != FallbackLevel.NONE:
            logger.warning(f"Triggering fallback: {fallback_level.value}")
            
            with self.coordination_lock:
                self.state = CoordinationState.FALLBACK
                self.current_status.state = CoordinationState.FALLBACK
                self.current_status.fallback_level = fallback_level
            
            # フォールバック実行
            fallback_action = self.fallback_manager.execute_fallback(
                fallback_level, context, self.current_status.failed_strategies
            )
            
            # 緊急停止の場合
            if fallback_level == FallbackLevel.EMERGENCY:
                with self.coordination_lock:
                    self.state = CoordinationState.EMERGENCY
                    self.current_status.state = CoordinationState.EMERGENCY
                
                logger.critical("Emergency fallback activated - stopping all operations")
                self._emergency_shutdown()
            
            self.performance_stats['fallback_activations'] += 1
    
    def _finalize_coordination(self, context: ExecutionContext, results: Dict[str, 'ExecutionResult']):
        """調整終了処理"""
        end_time = datetime.now()
        execution_duration = (end_time - context.start_time).total_seconds()
        
        # 成功判定
        success_count = sum(1 for r in results.values() if r.status == ExecutionStatus.COMPLETED)
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        coordination_success = success_rate >= 0.7  # 70%以上成功で全体成功とみなす
        
        # 履歴記録
        coordination_record = {
            'execution_id': context.execution_id,
            'plan_id': context.plan.plan_id,
            'strategies': context.plan.strategies,
            'start_time': context.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': execution_duration,
            'success_rate': success_rate,
            'coordination_success': coordination_success,
            'fallback_level': self.current_status.fallback_level.value,
            'results_summary': {
                'total': total_count,
                'completed': success_count,
                'failed': total_count - success_count
            }
        }
        
        self.coordination_history.append(coordination_record)
        
        # 統計更新
        if coordination_success:
            self.performance_stats['successful_coordinations'] += 1
        else:
            self.performance_stats['failed_coordinations'] += 1
        
        # 平均実行時間更新
        current_avg = self.performance_stats['average_coordination_time']
        total_coordinations = self.performance_stats['total_coordinations']
        self.performance_stats['average_coordination_time'] = (
            (current_avg * (total_coordinations - 1) + execution_duration) / total_coordinations
        )
        
        # 状態クリーンアップ
        with self.coordination_lock:
            self.state = CoordinationState.IDLE
            self.current_status = CoordinationStatus(state=CoordinationState.IDLE)
            self.active_executions.pop(context.execution_id, None)
        
        logger.info(f"Coordination completed: {context.execution_id} ({'SUCCESS' if coordination_success else 'FAILED'}) in {execution_duration:.1f}s")
    
    def _handle_coordination_failure(self, context: ExecutionContext, error_message: str):
        """調整失敗処理"""
        logger.error(f"Coordination failed: {context.execution_id} - {error_message}")
        
        # 失敗記録
        failure_record = {
            'execution_id': context.execution_id,
            'plan_id': context.plan.plan_id,
            'failure_time': datetime.now().isoformat(),
            'error_message': error_message,
            'coordination_success': False
        }
        
        self.coordination_history.append(failure_record)
        self.performance_stats['failed_coordinations'] += 1
        
        # 状態クリーンアップ
        with self.coordination_lock:
            self.state = CoordinationState.IDLE
            self.current_status = CoordinationStatus(state=CoordinationState.IDLE)
            self.active_executions.pop(context.execution_id, None)
    
    def _build_execution_timeline(
        self, 
        strategies: List[str], 
        dependency_resolution: Optional[DependencyResolution]
    ) -> List[Tuple[datetime, str, str]]:
        """実行タイムライン構築"""
        timeline = []
        current_time = datetime.now()
        
        if dependency_resolution:
            # 依存関係に基づいたタイムライン
            for group in dependency_resolution.parallel_groups:
                for strategy in group:
                    timeline.append((current_time, strategy, "start"))
                    timeline.append((current_time + timedelta(seconds=30), strategy, "complete"))
                current_time += timedelta(seconds=35)  # グループ間の余裕時間
        else:
            # シンプルな順次実行タイムライン
            for strategy in strategies:
                timeline.append((current_time, strategy, "start"))
                timeline.append((current_time + timedelta(seconds=30), strategy, "complete"))
                current_time += timedelta(seconds=35)
        
        return timeline
    
    def _calculate_completion_time(
        self, 
        allocations: List[ResourceAllocation], 
        dependency_resolution: Optional[DependencyResolution]
    ) -> datetime:
        """完了時間計算"""
        if dependency_resolution:
            return datetime.now() + timedelta(seconds=dependency_resolution.critical_path_duration + 60)
        else:
            total_duration = sum(alloc.estimated_duration for alloc in allocations) if allocations else 300
            return datetime.now() + timedelta(seconds=total_duration + 60)
    
    def _prepare_fallback_scenarios(self, strategies: List[str]) -> Dict[FallbackLevel, Dict[str, Any]]:
        """フォールバックシナリオ準備"""
        scenarios = {}
        
        for level in FallbackLevel:
            if level == FallbackLevel.NONE:
                continue
            
            scenario = {
                'level': level.value,
                'trigger_conditions': self._get_fallback_triggers(level),
                'actions': self._get_fallback_actions(level, strategies),
                'recovery_options': self._get_recovery_options(level, strategies)
            }
            scenarios[level] = scenario
        
        return scenarios
    
    def _get_fallback_triggers(self, level: FallbackLevel) -> List[str]:
        """フォールバック発動条件取得"""
        triggers = {
            FallbackLevel.INDIVIDUAL: ["個別戦略エラー閾値超過", "戦略タイムアウト"],
            FallbackLevel.GROUP: ["戦略グループ失敗率40%超過", "関連戦略連鎖失敗"],
            FallbackLevel.SYSTEM: ["システム全体成功率30%未満", "リソース枯渇"],
            FallbackLevel.EMERGENCY: ["クリティカルアラート3件以上", "システム不安定"]
        }
        return triggers.get(level, [])
    
    def _get_fallback_actions(self, level: FallbackLevel, strategies: List[str]) -> List[str]:
        """フォールバックアクション取得"""
        actions = {
            FallbackLevel.INDIVIDUAL: ["失敗戦略停止", "リトライ実行"],
            FallbackLevel.GROUP: ["関連戦略群停止", "縮小実行継続"],
            FallbackLevel.SYSTEM: ["非必須戦略停止", "基本機能のみ継続"],
            FallbackLevel.EMERGENCY: ["全戦略緊急停止", "セーフモード移行"]
        }
        return actions.get(level, [])
    
    def _get_recovery_options(self, level: FallbackLevel, strategies: List[str]) -> List[str]:
        """復旧オプション取得"""
        options = {
            FallbackLevel.INDIVIDUAL: ["自動リトライ", "パラメータ調整後再実行"],
            FallbackLevel.GROUP: ["健全戦略のみ継続", "段階的復旧"],
            FallbackLevel.SYSTEM: ["システム再初期化", "基本機能検証後復旧"],
            FallbackLevel.EMERGENCY: ["手動再起動必須", "全面的システム点検"]
        }
        return options.get(level, [])
    
    def _assess_execution_risks(
        self, 
        strategies: List[str], 
        allocations: List[ResourceAllocation]
    ) -> Dict[str, Any]:
        """実行リスク評価"""
        risk_assessment = {
            'overall_risk_level': 'medium',
            'risk_factors': [],
            'mitigation_strategies': [],
            'confidence_score': 0.7
        }
        
        # リソース使用量リスク
        total_cpu = sum(alloc.allocated_cpu for alloc in allocations)
        total_memory = sum(alloc.allocated_memory_mb for alloc in allocations)
        
        if total_cpu > 2.0:
            risk_assessment['risk_factors'].append("高CPU使用率")
            risk_assessment['mitigation_strategies'].append("並列度調整")
        
        if total_memory > 2048:  # 2GB
            risk_assessment['risk_factors'].append("高メモリ使用量")
            risk_assessment['mitigation_strategies'].append("メモリ使用量監視")
        
        # 戦略数リスク
        if len(strategies) > 10:
            risk_assessment['risk_factors'].append("多数戦略同時実行")
            risk_assessment['mitigation_strategies'].append("段階的実行")
        
        # 信頼性スコア計算
        risk_factor_count = len(risk_assessment['risk_factors'])
        if risk_factor_count == 0:
            risk_assessment['overall_risk_level'] = 'low'
            risk_assessment['confidence_score'] = 0.9
        elif risk_factor_count <= 2:
            risk_assessment['overall_risk_level'] = 'medium'
            risk_assessment['confidence_score'] = 0.7
        else:
            risk_assessment['overall_risk_level'] = 'high'
            risk_assessment['confidence_score'] = 0.4
        
        return risk_assessment
    
    def _emergency_shutdown(self):
        """緊急シャットダウン"""
        logger.critical("Initiating emergency shutdown")
        
        # 全実行停止
        if self.execution_scheduler:
            for execution_id in list(self.active_executions.keys()):
                context = self.active_executions[execution_id]
                for strategy in context.plan.strategies:
                    # タスクキャンセル試行
                    try:
                        self.execution_scheduler.cancel_task(f"task_{strategy}")
                    except:
                        pass
        
        # 全アクティブ実行をクリア
        with self.coordination_lock:
            self.active_executions.clear()
            self.state = CoordinationState.EMERGENCY
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """調整状況取得"""
        with self.coordination_lock:
            status_dict = self.current_status.to_dict()
        
        # システムヘルス評価
        if self.monitoring_system:
            try:
                health_report = self.monitoring_system._assess_system_health()
                status_dict['system_health_score'] = health_report.get('health_score', 100)
            except:
                status_dict['system_health_score'] = 100
        
        # パフォーマンス統計追加
        status_dict['performance_statistics'] = self.performance_stats.copy()
        
        # アクティブ実行情報
        status_dict['active_executions'] = {
            exec_id: {
                'plan_id': context.plan.plan_id,
                'strategies': context.plan.strategies,
                'start_time': context.start_time.isoformat(),
                'expected_end_time': context.expected_end_time.isoformat()
            }
            for exec_id, context in self.active_executions.items()
        }
        
        return status_dict
    
    def cancel_coordination(self, execution_id: str) -> bool:
        """調整キャンセル"""
        with self.coordination_lock:
            if execution_id not in self.active_executions:
                return False
            
            context = self.active_executions[execution_id]
            
            # 実行中タスクのキャンセル
            if self.execution_scheduler:
                for strategy in context.plan.strategies:
                    self.execution_scheduler.cancel_task(f"task_{strategy}")
            
            # 実行コンテキスト削除
            del self.active_executions[execution_id]
            
            logger.info(f"Coordination cancelled: {execution_id}")
            return True
    
    def shutdown(self):
        """シャットダウン"""
        logger.info("Shutting down Multi-Strategy Coordination Manager...")
        
        self.shutdown_event.set()
        
        with self.coordination_lock:
            self.state = CoordinationState.SHUTTING_DOWN
        
        # 全アクティブ実行をキャンセル
        for execution_id in list(self.active_executions.keys()):
            self.cancel_coordination(execution_id)
        
        # コンポーネントシャットダウン
        if self.monitoring_system:
            self.monitoring_system.shutdown()
        
        if self.execution_scheduler:
            self.execution_scheduler.shutdown()
        
        if self.resource_engine:
            self.resource_engine.shutdown()
        
        logger.info("Multi-Strategy Coordination Manager shutdown complete")

def create_demo_strategies() -> List[str]:
    """デモ戦略作成"""
    return ["VWAPBounceStrategy", "GCStrategy", "BreakoutStrategy", "OpeningGapStrategy", "MomentumStrategy"]

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Multi-Strategy Coordination Manager - Demo")
    print("=" * 60)
    
    try:
        # 調整マネージャー初期化
        manager = MultiStrategyCoordinationManager()
        
        # コンポーネント初期化（利用可能なもののみ）
        print("🔧 Initializing coordination components...")
        try:
            manager.initialize_components()
            print("✅ Components initialized successfully")
        except Exception as e:
            print(f"⚠️ Component initialization partial: {e}")
        
        # デモ戦略
        demo_strategies = create_demo_strategies()
        
        print(f"\n🎯 Creating coordination plan for {len(demo_strategies)} strategies:")
        for strategy in demo_strategies:
            print(f"  - {strategy}")
        
        # 調整計画作成
        plan = manager.create_coordination_plan(demo_strategies)
        
        print(f"\n📋 Coordination Plan Created:")
        print(f"  Plan ID: {plan.plan_id}")
        print(f"  Strategies: {len(plan.strategies)}")
        print(f"  Resource Allocations: {len(plan.resource_allocations)}")
        print(f"  Estimated Completion: {plan.estimated_completion_time.strftime('%H:%M:%S')}")
        
        # リスク評価表示
        risk = plan.risk_assessment
        print(f"  Risk Level: {risk['overall_risk_level'].upper()}")
        print(f"  Confidence Score: {risk['confidence_score']:.1%}")
        
        if risk['risk_factors']:
            print(f"  Risk Factors:")
            for factor in risk['risk_factors']:
                print(f"    ⚠️ {factor}")
        
        # フォールバックシナリオ表示
        print(f"  Fallback Scenarios: {len(plan.fallback_scenarios)} levels prepared")
        
        # 調整実行開始
        print(f"\n🚀 Starting coordination execution...")
        execution_id = manager.execute_coordination_plan(plan)
        print(f"Execution ID: {execution_id}")
        
        # 実行状況監視
        monitoring_duration = 30  # 30秒間監視
        start_monitor_time = time.time()
        
        while time.time() - start_monitor_time < monitoring_duration:
            status = manager.get_coordination_status()
            
            print(f"\n📊 Coordination Status (t+{int(time.time() - start_monitor_time)}s):")
            print(f"  State: {status['state']}")
            print(f"  Active Strategies: {len(status.get('active_strategies', []))}")
            print(f"  Completed Strategies: {len(status.get('completed_strategies', []))}")
            print(f"  Failed Strategies: {len(status.get('failed_strategies', []))}")
            print(f"  Fallback Level: {status.get('fallback_level', 'none')}")
            print(f"  System Health: {status.get('system_health_score', 100)}/100")
            
            # 実行完了チェック
            if status['state'] in ['idle', 'emergency']:
                break
            
            time.sleep(5)
        
        # 最終統計
        print(f"\n📈 Final Statistics:")
        stats = manager.performance_stats
        print(f"  Total Coordinations: {stats['total_coordinations']}")
        print(f"  Successful: {stats['successful_coordinations']}")
        print(f"  Failed: {stats['failed_coordinations']}")
        print(f"  Fallback Activations: {stats['fallback_activations']}")
        print(f"  Average Duration: {stats['average_coordination_time']:.1f}s")
        
        if manager.coordination_history:
            print(f"  Coordination History: {len(manager.coordination_history)} entries")
        
        print("\n✅ Multi-Strategy Coordination Manager demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'manager' in locals():
            manager.shutdown()
