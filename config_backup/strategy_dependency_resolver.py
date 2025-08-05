"""
Module: Strategy Dependency Resolver
File: strategy_dependency_resolver.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  戦略依存関係解決・実行順序最適化

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 依存関係グラフ構築・循環参照検出
  - 実行順序最適化・並列実行可能性分析
  - データ共有最適化・依存関係管理
  - 動的依存関係解決・実行時調整
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
from concurrent.futures import Future
import networkx as nx
import hashlib

# ロガー設定
logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """依存関係種別"""
    DATA = "data"
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    LOGICAL = "logical"

class ExecutionPhase(Enum):
    """実行フェーズ"""
    PREPARATION = "preparation"
    ANALYSIS = "analysis" 
    EXECUTION = "execution"
    POST_PROCESSING = "post_processing"

class DependencyStatus(Enum):
    """依存関係状態"""
    PENDING = "pending"
    SATISFIED = "satisfied"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class StrategyDependency:
    """戦略依存関係"""
    source_strategy: str
    target_strategy: str
    dependency_type: DependencyType
    data_key: Optional[str] = None
    is_critical: bool = True
    timeout_seconds: float = 300.0
    retry_count: int = 3
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['dependency_type'] = self.dependency_type.value
        return result
    
    def get_dependency_id(self) -> str:
        """依存関係ID生成"""
        content = f"{self.source_strategy}->{self.target_strategy}:{self.dependency_type.value}:{self.data_key or 'none'}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class ExecutionNode:
    """実行ノード"""
    strategy_name: str
    phase: ExecutionPhase
    dependencies: List[StrategyDependency] = field(default_factory=list)
    execution_priority: int = 1
    estimated_duration: float = 30.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    parallel_capable: bool = True
    data_outputs: List[str] = field(default_factory=list)
    data_inputs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['phase'] = self.phase.value
        result['dependencies'] = [dep.to_dict() for dep in self.dependencies]
        return result
    
    def get_node_id(self) -> str:
        """ノードID生成"""
        return f"{self.strategy_name}#{self.phase.value}"

@dataclass
class ExecutionGraph:
    """実行グラフ"""
    nodes: List[ExecutionNode] = field(default_factory=list)
    edges: List[StrategyDependency] = field(default_factory=list)
    execution_levels: List[List[str]] = field(default_factory=list)
    shared_data_keys: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'execution_levels': self.execution_levels,
            'shared_data_keys': list(self.shared_data_keys)
        }
        return result

@dataclass
class DependencyResolution:
    """依存関係解決結果"""
    resolved_dependencies: List[StrategyDependency]
    execution_order: List[str]
    parallel_groups: List[List[str]]
    data_flow_optimization: Dict[str, Any]
    critical_path_duration: float
    optimization_suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['resolved_dependencies'] = [dep.to_dict() for dep in self.resolved_dependencies]
        return result

class DataFlowTracker:
    """データフロー追跡器"""
    
    def __init__(self):
        self.data_registry: Dict[str, Any] = {}
        self.data_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.data_consumers: Dict[str, Set[str]] = defaultdict(set)
        self.access_lock = threading.Lock()
        
    def register_data_output(self, strategy_name: str, data_key: str, data_value: Any = None):
        """データ出力登録"""
        with self.access_lock:
            self.data_registry[data_key] = {
                'value': data_value,
                'producer': strategy_name,
                'timestamp': datetime.now(),
                'access_count': 0
            }
            logger.debug(f"Registered data output: {data_key} from {strategy_name}")
    
    def register_data_input(self, strategy_name: str, data_key: str):
        """データ入力登録"""
        with self.access_lock:
            self.data_dependencies[strategy_name].add(data_key)
            self.data_consumers[data_key].add(strategy_name)
            logger.debug(f"Registered data input: {data_key} for {strategy_name}")
    
    def check_data_availability(self, data_keys: List[str]) -> Dict[str, bool]:
        """データ利用可能性確認"""
        with self.access_lock:
            return {
                key: key in self.data_registry 
                for key in data_keys
            }
    
    def get_data(self, data_key: str) -> Optional[Any]:
        """データ取得"""
        with self.access_lock:
            if data_key in self.data_registry:
                self.data_registry[data_key]['access_count'] += 1
                return self.data_registry[data_key]['value']
            return None
    
    def get_data_flow_stats(self) -> Dict[str, Any]:
        """データフロー統計"""
        with self.access_lock:
            return {
                'total_data_keys': len(self.data_registry),
                'dependency_count': sum(len(deps) for deps in self.data_dependencies.values()),
                'consumer_count': sum(len(consumers) for consumers in self.data_consumers.values()),
                'data_utilization': {
                    key: info['access_count'] 
                    for key, info in self.data_registry.items()
                }
            }

class StrategyDependencyResolver:
    """戦略依存関係リゾルバー"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config = self._load_config(config_path)
        self.data_flow_tracker = DataFlowTracker()
        self.dependency_graph = nx.DiGraph()
        self.execution_history: List[Dict[str, Any]] = []
        self.dependency_cache: Dict[str, ExecutionGraph] = {}
        
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
            "dependency_management": {
                "max_dependency_depth": 10,
                "parallel_execution_threshold": 2,
                "data_sharing_optimization": True,
                "dependency_timeout": 300,
                "cache_dependency_resolution": True
            },
            "strategy_profiles": {},
            "execution_phases": {
                "preparation": {"weight": 0.1, "parallel_factor": 0.8},
                "analysis": {"weight": 0.3, "parallel_factor": 0.6},
                "execution": {"weight": 0.5, "parallel_factor": 0.4},
                "post_processing": {"weight": 0.1, "parallel_factor": 0.9}
            }
        }
    
    def build_dependency_graph(self, strategies: List[str]) -> ExecutionGraph:
        """依存関係グラフ構築"""
        
        # キャッシュチェック
        cache_key = self._generate_cache_key(strategies)
        if (self.config.get('dependency_management', {}).get('cache_dependency_resolution', True) and
            cache_key in self.dependency_cache):
            logger.info(f"Using cached dependency graph for {len(strategies)} strategies")
            return self.dependency_cache[cache_key]
        
        logger.info(f"Building dependency graph for strategies: {strategies}")
        
        # ノード作成
        nodes = []
        edges = []
        
        for strategy in strategies:
            strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
            
            # 各フェーズのノード作成
            for phase in ExecutionPhase:
                node = ExecutionNode(
                    strategy_name=strategy,
                    phase=phase,
                    execution_priority=strategy_profile.get('priority', 1),
                    estimated_duration=strategy_profile.get('expected_duration', 30.0) * 
                                     self.config.get('execution_phases', {}).get(phase.value, {}).get('weight', 0.25),
                    resource_requirements=strategy_profile.get('resource_requirements', {}),
                    parallel_capable=strategy_profile.get('parallel_capable', True),
                    data_outputs=strategy_profile.get('data_outputs', []),
                    data_inputs=strategy_profile.get('data_inputs', [])
                )
                nodes.append(node)
        
        # 依存関係構築
        edges = self._build_strategy_dependencies(strategies, nodes)
        
        # NetworkXグラフ構築
        self.dependency_graph.clear()
        for node in nodes:
            self.dependency_graph.add_node(node.get_node_id(), **node.to_dict())
        
        for edge in edges:
            source_id = f"{edge.source_strategy}#{ExecutionPhase.EXECUTION.value}"
            target_id = f"{edge.target_strategy}#{ExecutionPhase.PREPARATION.value}"
            self.dependency_graph.add_edge(source_id, target_id, **edge.to_dict())
        
        # 循環参照検出
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycles = list(nx.simple_cycles(self.dependency_graph))
            logger.error(f"Circular dependencies detected: {cycles}")
            raise ValueError(f"Circular dependencies found: {cycles}")
        
        # 実行レベル計算
        execution_levels = self._calculate_execution_levels()
        
        # 共有データキー抽出
        shared_data_keys = set()
        for node in nodes:
            shared_data_keys.update(node.data_outputs)
            shared_data_keys.update(node.data_inputs)
        
        execution_graph = ExecutionGraph(
            nodes=nodes,
            edges=edges,
            execution_levels=execution_levels,
            shared_data_keys=shared_data_keys
        )
        
        # キャッシュ保存
        if self.config.get('dependency_management', {}).get('cache_dependency_resolution', True):
            self.dependency_cache[cache_key] = execution_graph
        
        logger.info(f"Built dependency graph with {len(nodes)} nodes, {len(edges)} edges, {len(execution_levels)} levels")
        
        return execution_graph
    
    def _generate_cache_key(self, strategies: List[str]) -> str:
        """キャッシュキー生成"""
        content = "|".join(sorted(strategies))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _build_strategy_dependencies(self, strategies: List[str], nodes: List[ExecutionNode]) -> List[StrategyDependency]:
        """戦略依存関係構築"""
        dependencies = []
        
        # プロファイルベース依存関係
        for strategy in strategies:
            strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
            strategy_deps = strategy_profile.get('dependencies', [])
            
            for dep_config in strategy_deps:
                if isinstance(dep_config, str):
                    # シンプル依存関係
                    target_strategy = dep_config
                    if target_strategy in strategies:
                        dependency = StrategyDependency(
                            source_strategy=target_strategy,
                            target_strategy=strategy,
                            dependency_type=DependencyType.LOGICAL,
                            description=f"{strategy} depends on {target_strategy}"
                        )
                        dependencies.append(dependency)
                
                elif isinstance(dep_config, dict):
                    # 詳細依存関係設定
                    target_strategy = dep_config.get('strategy')
                    if target_strategy and target_strategy in strategies:
                        dependency = StrategyDependency(
                            source_strategy=target_strategy,
                            target_strategy=strategy,
                            dependency_type=DependencyType(dep_config.get('type', 'logical')),
                            data_key=dep_config.get('data_key'),
                            is_critical=dep_config.get('critical', True),
                            timeout_seconds=dep_config.get('timeout', 300.0),
                            description=dep_config.get('description', '')
                        )
                        dependencies.append(dependency)
        
        # データフロー依存関係
        data_dependencies = self._analyze_data_flow_dependencies(nodes)
        dependencies.extend(data_dependencies)
        
        return dependencies
    
    def _analyze_data_flow_dependencies(self, nodes: List[ExecutionNode]) -> List[StrategyDependency]:
        """データフロー依存関係分析"""
        dependencies = []
        
        # データ出力→入力マッピング
        data_producers = {}
        data_consumers = defaultdict(list)
        
        for node in nodes:
            # データ出力者記録
            for data_key in node.data_outputs:
                data_producers[data_key] = node.strategy_name
            
            # データ消費者記録
            for data_key in node.data_inputs:
                data_consumers[data_key].append(node.strategy_name)
        
        # 依存関係構築
        for data_key, consumers in data_consumers.items():
            if data_key in data_producers:
                producer = data_producers[data_key]
                for consumer in consumers:
                    if producer != consumer:  # 自己参照は除外
                        dependency = StrategyDependency(
                            source_strategy=producer,
                            target_strategy=consumer,
                            dependency_type=DependencyType.DATA,
                            data_key=data_key,
                            description=f"{consumer} needs data '{data_key}' from {producer}"
                        )
                        dependencies.append(dependency)
        
        return dependencies
    
    def _calculate_execution_levels(self) -> List[List[str]]:
        """実行レベル計算（トポロジカルソート）"""
        if not self.dependency_graph.nodes():
            return []
        
        levels = []
        graph_copy = self.dependency_graph.copy()
        
        while graph_copy.nodes():
            # 入力エッジがないノードを見つける
            current_level = [
                node for node in graph_copy.nodes()
                if graph_copy.in_degree(node) == 0
            ]
            
            if not current_level:
                # すべてのノードに入力エッジがある = 循環参照
                remaining = list(graph_copy.nodes())
                logger.error(f"Cannot resolve dependencies, remaining nodes: {remaining}")
                break
            
            levels.append(current_level)
            graph_copy.remove_nodes_from(current_level)
        
        return levels
    
    def resolve_dependencies(self, strategies: List[str]) -> DependencyResolution:
        """依存関係解決"""
        logger.info(f"Resolving dependencies for {len(strategies)} strategies")
        
        # 依存関係グラフ構築
        execution_graph = self.build_dependency_graph(strategies)
        
        # 実行順序計算
        execution_order = []
        for level in execution_graph.execution_levels:
            # レベル内での優先度ソート
            level_strategies = []
            for node_id in level:
                strategy_name = node_id.split('#')[0]
                if strategy_name not in level_strategies:
                    level_strategies.append(strategy_name)
            
            # 優先度とリソース要求量によるソート
            level_strategies.sort(key=lambda s: (
                -self.config.get('strategy_profiles', {}).get(s, {}).get('priority', 1),
                -self.config.get('strategy_profiles', {}).get(s, {}).get('expected_duration', 30.0)
            ))
            
            execution_order.extend(level_strategies)
        
        # 並列グループ計算
        parallel_groups = self._calculate_parallel_groups(execution_graph)
        
        # データフロー最適化
        data_flow_optimization = self._optimize_data_flow(execution_graph)
        
        # クリティカルパス計算
        critical_path_duration = self._calculate_critical_path_duration(execution_graph)
        
        # 最適化提案生成
        optimization_suggestions = self._generate_optimization_suggestions(
            execution_graph, critical_path_duration
        )
        
        resolution = DependencyResolution(
            resolved_dependencies=execution_graph.edges,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            data_flow_optimization=data_flow_optimization,
            critical_path_duration=critical_path_duration,
            optimization_suggestions=optimization_suggestions
        )
        
        # 履歴保存
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategies': strategies,
            'resolution': resolution.to_dict()
        })
        
        logger.info(f"Dependency resolution complete: {len(execution_order)} strategies, {len(parallel_groups)} parallel groups")
        
        return resolution
    
    def _calculate_parallel_groups(self, execution_graph: ExecutionGraph) -> List[List[str]]:
        """並列実行グループ計算"""
        parallel_groups = []
        
        for level in execution_graph.execution_levels:
            level_strategies = set()
            for node_id in level:
                strategy_name = node_id.split('#')[0]
                level_strategies.add(strategy_name)
            
            # レベル内で並列実行可能なグループを作成
            if len(level_strategies) > 1:
                # 並列実行可能性チェック
                parallel_capable_strategies = []
                for strategy in level_strategies:
                    strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
                    if strategy_profile.get('parallel_capable', True):
                        parallel_capable_strategies.append(strategy)
                
                if len(parallel_capable_strategies) >= self.config.get('dependency_management', {}).get('parallel_execution_threshold', 2):
                    parallel_groups.append(parallel_capable_strategies)
                else:
                    # 並列実行に適さない場合は個別実行
                    for strategy in level_strategies:
                        parallel_groups.append([strategy])
            else:
                parallel_groups.append(list(level_strategies))
        
        return parallel_groups
    
    def _optimize_data_flow(self, execution_graph: ExecutionGraph) -> Dict[str, Any]:
        """データフロー最適化"""
        optimization = {
            'shared_data_optimization': {},
            'data_caching_recommendations': [],
            'memory_usage_estimation': {}
        }
        
        if not self.config.get('dependency_management', {}).get('data_sharing_optimization', True):
            return optimization
        
        # 共有データ最適化
        data_usage_count = defaultdict(int)
        for node in execution_graph.nodes:
            for data_key in node.data_inputs:
                data_usage_count[data_key] += 1
        
        # 多く使用されるデータのキャッシュ推奨
        for data_key, usage_count in data_usage_count.items():
            if usage_count > 2:
                optimization['data_caching_recommendations'].append({
                    'data_key': data_key,
                    'usage_count': usage_count,
                    'recommendation': 'データをメモリにキャッシュして再利用を最適化'
                })
        
        # メモリ使用量推定
        for data_key in execution_graph.shared_data_keys:
            # 簡単な推定（実際にはデータ型やサイズを考慮）
            estimated_size_mb = 10  # デフォルト推定値
            optimization['memory_usage_estimation'][data_key] = estimated_size_mb
        
        optimization['shared_data_optimization'] = {
            'total_shared_data_keys': len(execution_graph.shared_data_keys),
            'high_usage_data_keys': len(optimization['data_caching_recommendations']),
            'estimated_total_memory_mb': sum(optimization['memory_usage_estimation'].values())
        }
        
        return optimization
    
    def _calculate_critical_path_duration(self, execution_graph: ExecutionGraph) -> float:
        """クリティカルパス実行時間計算"""
        if not execution_graph.nodes:
            return 0.0
        
        # 各ノードの最早開始時間と最遅終了時間を計算
        node_durations = {}
        for node in execution_graph.nodes:
            node_durations[node.get_node_id()] = node.estimated_duration
        
        # 最長パス計算（DAGの場合）
        try:
            longest_path = nx.dag_longest_path(self.dependency_graph, weight='estimated_duration')
            critical_path_duration = sum(node_durations.get(node_id, 0) for node_id in longest_path)
        except:
            # フォールバック: すべてのノードの合計実行時間
            critical_path_duration = sum(node.estimated_duration for node in execution_graph.nodes)
        
        return critical_path_duration
    
    def _generate_optimization_suggestions(
        self, 
        execution_graph: ExecutionGraph, 
        critical_path_duration: float
    ) -> List[str]:
        """最適化提案生成"""
        suggestions = []
        
        # 並列化提案
        if len(execution_graph.execution_levels) > 3:
            suggestions.append("複数の実行レベルがあります。並列実行でパフォーマンス向上が期待できます")
        
        # クリティカルパス最適化
        if critical_path_duration > 120:  # 2分以上
            suggestions.append("クリティカルパスが長いため、ボトルネック戦略の最適化を検討してください")
        
        # データフロー最適化
        if len(execution_graph.shared_data_keys) > 10:
            suggestions.append("共有データキーが多いため、データ構造の統合を検討してください")
        
        # 依存関係簡素化
        if len(execution_graph.edges) > len(execution_graph.nodes) * 1.5:
            suggestions.append("依存関係が複雑です。不要な依存関係の削減を検討してください")
        
        return suggestions
    
    def validate_execution_readiness(self, strategies: List[str]) -> Dict[str, Any]:
        """実行準備状況検証"""
        validation_result = {
            'ready': True,
            'issues': [],
            'warnings': [],
            'dependency_status': {},
            'data_availability': {}
        }
        
        try:
            # 依存関係解決
            resolution = self.resolve_dependencies(strategies)
            
            # 依存関係状態チェック
            for dependency in resolution.resolved_dependencies:
                dep_id = dependency.get_dependency_id()
                
                if dependency.dependency_type == DependencyType.DATA:
                    # データ依存関係の場合、データ利用可能性チェック
                    if dependency.data_key:
                        available = self.data_flow_tracker.check_data_availability([dependency.data_key])
                        validation_result['data_availability'][dependency.data_key] = available.get(dependency.data_key, False)
                        
                        if not available.get(dependency.data_key, False):
                            if dependency.is_critical:
                                validation_result['issues'].append(
                                    f"Critical data '{dependency.data_key}' not available for {dependency.target_strategy}"
                                )
                                validation_result['ready'] = False
                            else:
                                validation_result['warnings'].append(
                                    f"Optional data '{dependency.data_key}' not available for {dependency.target_strategy}"
                                )
                
                validation_result['dependency_status'][dep_id] = DependencyStatus.SATISFIED.value
            
            # リソース要求検証
            total_cpu_requirement = 0
            total_memory_requirement = 0
            
            for strategy in strategies:
                strategy_profile = self.config.get('strategy_profiles', {}).get(strategy, {})
                resource_req = strategy_profile.get('resource_requirements', {})
                
                total_cpu_requirement += resource_req.get('cpu', 0.1)
                memory_str = str(resource_req.get('memory', '64MB')).replace('MB', '')
                total_memory_requirement += int(memory_str)
            
            # 簡単なリソースチェック
            if total_cpu_requirement > 2.0:  # CPU使用率200%以上
                validation_result['warnings'].append(
                    f"高CPU要求 ({total_cpu_requirement:.1f}) - 実行時間が長くなる可能性があります"
                )
            
            if total_memory_requirement > 4096:  # 4GB以上
                validation_result['warnings'].append(
                    f"高メモリ要求 ({total_memory_requirement}MB) - メモリ不足の可能性があります"
                )
        
        except Exception as e:
            validation_result['ready'] = False
            validation_result['issues'].append(f"Dependency resolution failed: {str(e)}")
        
        return validation_result
    
    def get_dependency_stats(self) -> Dict[str, Any]:
        """依存関係統計取得"""
        stats = {
            'total_resolutions': len(self.execution_history),
            'cache_entries': len(self.dependency_cache),
            'data_flow_stats': self.data_flow_tracker.get_data_flow_stats(),
            'graph_stats': {
                'nodes': len(self.dependency_graph.nodes),
                'edges': len(self.dependency_graph.edges),
                'is_acyclic': nx.is_directed_acyclic_graph(self.dependency_graph) if self.dependency_graph.nodes else True
            }
        }
        
        return stats

def create_demo_strategies() -> List[str]:
    """デモ戦略作成"""
    return ["VWAPBounceStrategy", "GCStrategy", "BreakoutStrategy", "OpeningGapStrategy"]

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Strategy Dependency Resolver - Demo")
    print("=" * 60)
    
    try:
        # リゾルバー初期化
        resolver = StrategyDependencyResolver()
        
        # デモ戦略
        demo_strategies = create_demo_strategies()
        
        print(f"\n🎯 Testing dependency resolution for strategies: {demo_strategies}")
        
        # 依存関係解決
        resolution = resolver.resolve_dependencies(demo_strategies)
        
        print(f"\n📊 Dependency Resolution Results:")
        print("-" * 50)
        print(f"Execution Order: {resolution.execution_order}")
        print(f"Parallel Groups: {resolution.parallel_groups}")
        print(f"Critical Path Duration: {resolution.critical_path_duration:.1f}s")
        print(f"Dependencies Resolved: {len(resolution.resolved_dependencies)}")
        
        # 最適化提案
        if resolution.optimization_suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(resolution.optimization_suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        # データフロー最適化
        data_flow = resolution.data_flow_optimization
        if data_flow.get('shared_data_optimization'):
            shared_opt = data_flow['shared_data_optimization']
            print(f"\n📈 Data Flow Optimization:")
            print(f"  Shared Data Keys: {shared_opt.get('total_shared_data_keys', 0)}")
            print(f"  High Usage Keys: {shared_opt.get('high_usage_data_keys', 0)}")
            print(f"  Est. Memory Usage: {shared_opt.get('estimated_total_memory_mb', 0)}MB")
        
        # 実行準備検証
        print(f"\n🔍 Testing execution readiness validation...")
        validation = resolver.validate_execution_readiness(demo_strategies)
        
        print(f"Execution Ready: {'✅ Yes' if validation['ready'] else '❌ No'}")
        if validation['issues']:
            print(f"Issues: {len(validation['issues'])}")
            for issue in validation['issues']:
                print(f"  ⚠️ {issue}")
        
        if validation['warnings']:
            print(f"Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"  ⚡ {warning}")
        
        # 統計情報
        stats = resolver.get_dependency_stats()
        print(f"\n📊 Dependency Statistics:")
        print(f"  Total Resolutions: {stats['total_resolutions']}")
        print(f"  Cache Entries: {stats['cache_entries']}")
        print(f"  Graph Nodes: {stats['graph_stats']['nodes']}")
        print(f"  Graph Edges: {stats['graph_stats']['edges']}")
        print(f"  Graph is Acyclic: {'✅' if stats['graph_stats']['is_acyclic'] else '❌'}")
        
        print("\n✅ Strategy Dependency Resolver demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
