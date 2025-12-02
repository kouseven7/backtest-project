"""
Module: Composite Strategy Execution Engine
File: composite_strategy_execution_engine.py
Description: 
  4-1-2「複合戦略実行フロー設計・実装」- Main Engine
  複合戦略実行システムのメインエンジン
  パイプライン、調整、集約の統合制御

Author: imega
Created: 2025-01-28
Modified: 2025-01-28

Dependencies:
  - config.strategy_execution_pipeline
  - config.strategy_execution_coordinator
  - config.execution_result_aggregator
  - config.main_integration_config
"""

import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムのインポート
try:
    from config.strategy_execution_pipeline import StrategyExecutionPipeline, PipelineContext
    from config.strategy_execution_coordinator import StrategyExecutionCoordinator, ExecutionResult
    from config.execution_result_aggregator import ExecutionResultAggregator, AggregatedResult
    from config.strategy_selector import StrategySelector
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """実行モード"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    COMPOSITE = "composite"
    HYBRID = "hybrid"

class ExecutionStatus(Enum):
    """実行ステータス"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExecutionRequest:
    """実行リクエスト"""
    request_id: str
    market_data: pd.DataFrame
    strategies: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_mode: ExecutionMode = ExecutionMode.COMPOSITE
    timeout: int = 300
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionResponse:
    """実行レスポンス"""
    request_id: str
    status: ExecutionStatus
    aggregated_result: Optional[AggregatedResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    strategy_count: int = 0
    successful_strategies: int = 0
    metadata: Optional[Dict[str, Any]] = None
    completed_at: datetime = field(default_factory=datetime.now)

class CompositeStrategyExecutionEngine:
    """複合戦略実行エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._initialize_components()
        
        # 実行状態管理
        self.current_status = ExecutionStatus.IDLE
        self.execution_history: List[ExecutionResponse] = []
        self.active_requests: Dict[str, ExecutionRequest] = {}
        
        # パフォーマンス監視
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "engine_start_time": datetime.now()
        }
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        configs = {}
        
        # メイン統合設定の読み込み
        main_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", "main_integration_config.json"
        )
        
        try:
            with open(main_config_path, 'r', encoding='utf-8') as f:
                configs["main_integration"] = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load main integration config: {e}")
            configs["main_integration"] = {}
            
        # 複合実行設定の読み込み
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "composite_execution_config.json"
            )
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs["composite_execution"] = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load composite execution config: {e}")
            configs["composite_execution"] = self._get_default_composite_config()
            
        return configs
        
    def _get_default_composite_config(self) -> Dict[str, Any]:
        """デフォルト複合実行設定"""
        return {
            "execution_pipeline": {
                "stages": [
                    {"id": "strategy_selection", "enabled": True, "timeout_seconds": 30, "critical": True},
                    {"id": "weight_calculation", "enabled": True, "timeout_seconds": 20, "critical": True},
                    {"id": "signal_integration", "enabled": True, "timeout_seconds": 15, "critical": True},
                    {"id": "risk_adjustment", "enabled": True, "timeout_seconds": 25, "critical": False},
                    {"id": "execution", "enabled": True, "timeout_seconds": 60, "critical": True}
                ]
            },
            "coordination": {
                "execution_mode": "adaptive",
                "parallel_strategies": 4,
                "dynamic_ordering": True
            },
            "aggregation": {
                "method": "weighted",
                "confidence_weighting": True,
                "outlier_handling": "cap"
            }
        }
        
    def _initialize_components(self):
        """コンポーネントの初期化"""
        try:
            # 戦略選択器
            self.strategy_selector = StrategySelector()
            
            # パイプライン実行器
            self.pipeline = StrategyExecutionPipeline()
            
            # 実行調整器
            self.coordinator = StrategyExecutionCoordinator()
            
            # 結果集約器
            self.aggregator = ExecutionResultAggregator(
                self.config.get("composite_execution", {})
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize components: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        """複合戦略実行のメイン処理"""
        self.current_status = ExecutionStatus.INITIALIZING
        start_time = time.time()
        
        try:
            # リクエストの登録
            self.active_requests[request.request_id] = request
            self.logger.info(f"Starting execution for request {request.request_id}")
            
            # 戦略選択
            strategies = self._select_strategies(request)
            if not strategies:
                raise ValueError("No strategies selected for execution")
                
            self.current_status = ExecutionStatus.RUNNING
            
            # 実行モードに応じた処理
            if request.execution_mode == ExecutionMode.SINGLE_STRATEGY:
                execution_results = self._execute_single_strategy(request, strategies[0])
            elif request.execution_mode == ExecutionMode.MULTI_STRATEGY:
                execution_results = self._execute_multi_strategy(request, strategies)
            elif request.execution_mode == ExecutionMode.COMPOSITE:
                execution_results = self._execute_composite_strategy(request, strategies)
            else:  # HYBRID
                execution_results = self._execute_hybrid_strategy(request, strategies)
                
            self.current_status = ExecutionStatus.AGGREGATING
            
            # 結果の集約
            aggregated_result = self.aggregator.aggregate_results(execution_results)
            
            execution_time = time.time() - start_time
            
            # レスポンスの作成
            response = ExecutionResponse(
                request_id=request.request_id,
                status=ExecutionStatus.COMPLETED,
                aggregated_result=aggregated_result,
                execution_time=execution_time,
                strategy_count=len(strategies),
                successful_strategies=sum(1 for r in execution_results if r.success),
                metadata={
                    "execution_mode": request.execution_mode.value,
                    "selected_strategies": strategies,
                    "individual_results": len(execution_results)
                }
            )
            
            # パフォーマンス更新
            self._update_performance_metrics(response)
            
            self.current_status = ExecutionStatus.COMPLETED
            self.logger.info(f"Execution completed successfully: {request.request_id}")
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            response = ExecutionResponse(
                request_id=request.request_id,
                status=ExecutionStatus.FAILED,
                execution_time=execution_time,
                error_message=error_msg
            )
            
            self.current_status = ExecutionStatus.FAILED
            return response
            
        finally:
            # リクエストのクリーンアップ
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
                
            # 実行履歴への追加
            if len(self.execution_history) >= 100:
                self.execution_history.pop(0)
            # response を履歴に追加（上記で定義されている）
            
    def _select_strategies(self, request: ExecutionRequest) -> List[str]:
        """戦略選択"""
        if request.strategies:
            # 明示的に指定された戦略を使用
            return request.strategies
            
        # 戦略選択器を使用した自動選択
        try:
            selection_result = self.strategy_selector.select_strategies(
                ticker="EXAMPLE",  # デフォルトティッカー  
                market_data=request.market_data
            )
            
            strategies = selection_result.selected_strategies
            self.logger.info(f"Auto-selected strategies: {strategies}")
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}")
            # フォールバック戦略
            default_strategies = self.config.get("main_integration", {}).get(
                "fallback_strategies", ["default_strategy"]
            )
            self.logger.warning(f"Using fallback strategies: {default_strategies}")
            return default_strategies
            
    def _execute_single_strategy(self, request: ExecutionRequest, 
                                strategy: str) -> List[ExecutionResult]:
        """単一戦略実行"""
        self.logger.info(f"Executing single strategy: {strategy}")
        
        context = self.pipeline.execute(
            market_data=request.market_data,
            parameters={**(request.parameters or {}), "strategy_name": strategy}
        )
        
        # ExecutionResultに変換
        success = self._evaluate_pipeline_success(context)
        
        result = ExecutionResult(
            task_id=f"single_{strategy}_{int(time.time() * 1000)}",
            strategy_name=strategy,
            context=context,
            success=success,
            execution_time=sum(r.execution_time for r in context.stage_results.values())
        )
        
        return [result]
        
    def _execute_multi_strategy(self, request: ExecutionRequest, 
                               strategies: List[str]) -> List[ExecutionResult]:
        """マルチ戦略実行"""
        self.logger.info(f"Executing multi strategies: {strategies}")
        
        return self.coordinator.execute_strategies(
            strategies=strategies,
            market_data=request.market_data,
            parameters=request.parameters
        )
        
    def _execute_composite_strategy(self, request: ExecutionRequest, 
                                  strategies: List[str]) -> List[ExecutionResult]:
        """複合戦略実行"""
        self.logger.info(f"Executing composite strategies: {strategies}")
        
        # より高度な複合戦略実行
        # 段階的実行、依存関係考慮、動的調整など
        
        # 第1段階: 基本戦略実行
        primary_results = self.coordinator.execute_strategies(
            strategies=strategies[:2] if len(strategies) > 2 else strategies,
            market_data=request.market_data,
            parameters=request.parameters
        )
        
        # 第2段階: 補完戦略実行（必要に応じて）
        if len(strategies) > 2 and len([r for r in primary_results if r.success]) < 2:
            self.logger.info("Running supplementary strategies")
            supplementary_results = self.coordinator.execute_strategies(
                strategies=strategies[2:],
                market_data=request.market_data,
                parameters=request.parameters
            )
            primary_results.extend(supplementary_results)
            
        return primary_results
        
    def _execute_hybrid_strategy(self, request: ExecutionRequest, 
                                strategies: List[str]) -> List[ExecutionResult]:
        """ハイブリッド戦略実行"""
        self.logger.info(f"Executing hybrid strategies: {strategies}")
        
        # メイン統合設定に基づくハイブリッド実行
        main_config = self.config.get("main_integration", {})
        execution_mode = main_config.get("execution_mode", "hybrid")
        
        if execution_mode == "parallel":
            return self.coordinator.execute_strategies(
                strategies=strategies,
                market_data=request.market_data,
                parameters=request.parameters
            )
        elif execution_mode == "sequential":
            # 逐次実行
            results = []
            for strategy in strategies:
                strategy_results = self.coordinator.execute_strategies(
                    strategies=[strategy],
                    market_data=request.market_data,
                    parameters=request.parameters
                )
                results.extend(strategy_results)
                
                # 成功した戦略があれば後続をスキップする選択肢
                if main_config.get("early_exit", False) and any(r.success for r in strategy_results):
                    break
                    
            return results
        else:
            # adaptiveまたはhybrid: 状況に応じて切り替え
            if len(strategies) <= 2:
                return self.coordinator.execute_strategies(
                    strategies=strategies,
                    market_data=request.market_data,
                    parameters=request.parameters
                )
            else:
                return self._execute_composite_strategy(request, strategies)
                
    def _evaluate_pipeline_success(self, context: PipelineContext) -> bool:
        """パイプライン成功評価"""
        if not context.stage_results:
            return False
            
        critical_stages = ["strategy_selection", "weight_calculation", "execution"]
        
        for stage_id in critical_stages:
            if stage_id in context.stage_results:
                if not context.stage_results[stage_id].is_success():
                    return False
                    
        return True
        
    def _update_performance_metrics(self, response: ExecutionResponse):
        """パフォーマンス指標の更新"""
        self.performance_metrics["total_executions"] += 1
        
        if response.status == ExecutionStatus.COMPLETED:
            self.performance_metrics["successful_executions"] += 1
            
        self.performance_metrics["total_execution_time"] += response.execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / 
            max(self.performance_metrics["total_executions"], 1)
        )
        
        # 実行履歴への追加
        self.execution_history.append(response)
        
    def get_engine_status(self) -> Dict[str, Any]:
        """エンジン状態取得"""
        uptime = (datetime.now() - self.performance_metrics["engine_start_time"]).total_seconds()
        
        return {
            "current_status": self.current_status.value,
            "uptime_seconds": uptime,
            "performance_metrics": self.performance_metrics.copy(),
            "active_requests": len(self.active_requests),
            "execution_history_count": len(self.execution_history),
            "component_status": {
                "strategy_selector": hasattr(self, 'strategy_selector'),
                "pipeline": hasattr(self, 'pipeline'),
                "coordinator": hasattr(self, 'coordinator'),
                "aggregator": hasattr(self, 'aggregator')
            },
            "config_loaded": {
                "main_integration": "main_integration" in self.config,
                "composite_execution": "composite_execution" in self.config
            }
        }
        
    def get_execution_report(self, request_id: str = None) -> str:
        """実行レポートの生成"""
        if request_id:
            # 特定リクエストのレポート
            history_item = next(
                (h for h in self.execution_history if h.request_id == request_id), 
                None
            )
            if not history_item:
                return f"Request {request_id} not found in execution history"
                
            return self._generate_single_execution_report(history_item)
        else:
            # 全体サマリーレポート
            return self._generate_summary_report()
            
    def _generate_single_execution_report(self, response: ExecutionResponse) -> str:
        """単一実行レポート生成"""
        report = []
        report.append(f"=== 実行レポート (ID: {response.request_id}) ===")
        report.append(f"実行時刻: {response.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"ステータス: {response.status.value}")
        report.append(f"実行時間: {response.execution_time:.2f}秒")
        report.append(f"戦略数: {response.strategy_count}")
        report.append(f"成功戦略数: {response.successful_strategies}")
        report.append("")
        
        if response.aggregated_result:
            report.append("--- 集約結果 ---")
            report.append(f"全体信頼度: {response.aggregated_result.overall_confidence:.3f}")
            report.append(f"シグナル数: {len(response.aggregated_result.final_signals)}")
            report.append(f"重み設定数: {len(response.aggregated_result.final_weights)}")
            
            # 詳細レポート
            detail_report = self.aggregator.get_aggregation_report(response.aggregated_result)
            report.append("")
            report.append(detail_report)
            
        if response.error_message:
            report.append("--- エラー情報 ---")
            report.append(response.error_message)
            
        return "\n".join(report)
        
    def _generate_summary_report(self) -> str:
        """サマリーレポート生成"""
        metrics = self.performance_metrics
        status = self.get_engine_status()
        
        report = []
        report.append("=== 複合戦略実行エンジン サマリー ===")
        report.append(f"稼働時間: {status['uptime_seconds']:.0f}秒")
        report.append(f"現在ステータス: {status['current_status']}")
        report.append("")
        
        report.append("--- パフォーマンス指標 ---")
        report.append(f"総実行回数: {metrics['total_executions']}")
        report.append(f"成功実行回数: {metrics['successful_executions']}")
        
        if metrics['total_executions'] > 0:
            success_rate = metrics['successful_executions'] / metrics['total_executions']
            report.append(f"成功率: {success_rate:.2%}")
            
        report.append(f"平均実行時間: {metrics['average_execution_time']:.2f}秒")
        report.append("")
        
        # 最近の実行履歴
        if self.execution_history:
            report.append("--- 最近の実行履歴 (最新5件) ---")
            for response in self.execution_history[-5:]:
                status_symbol = "✓" if response.status == ExecutionStatus.COMPLETED else "✗"
                report.append(
                    f"{status_symbol} {response.request_id}: "
                    f"{response.execution_time:.1f}s ({response.successful_strategies}/{response.strategy_count})"
                )
                
        return "\n".join(report)
        
    def shutdown(self):
        """エンジンの停止"""
        self.logger.info("Shutting down composite strategy execution engine")
        self.current_status = ExecutionStatus.CANCELLED
        
        # アクティブリクエストのクリーンアップ
        self.active_requests.clear()
        
        # コンポーネントのクリーンアップ（必要に応じて）
        # 今回は特別なクリーンアップは不要
        
        self.logger.info("Engine shutdown completed")

def create_execution_request(market_data: pd.DataFrame,
                           strategies: Optional[List[str]] = None,
                           parameters: Optional[Dict[str, Any]] = None,
                           execution_mode: ExecutionMode = ExecutionMode.COMPOSITE,
                           timeout: int = 300) -> ExecutionRequest:
    """実行リクエストファクトリ"""
    request_id = f"req_{int(time.time() * 1000)}"
    
    return ExecutionRequest(
        request_id=request_id,
        market_data=market_data,
        strategies=strategies,
        parameters=parameters,
        execution_mode=execution_mode,
        timeout=timeout
    )

if __name__ == "__main__":
    # テスト用のサンプル実行
    engine = CompositeStrategyExecutionEngine()
    
    # サンプルデータ生成
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='1H'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 実行リクエスト作成
    request = create_execution_request(
        market_data=sample_data,
        strategies=["strategy_a", "strategy_b"],
        execution_mode=ExecutionMode.COMPOSITE
    )
    
    # 実行
    response = engine.execute(request)
    
    # 結果表示
    print("Engine Status:")
    print(json.dumps(engine.get_engine_status(), indent=2, default=str, ensure_ascii=False))
    
    print("\nExecution Report:")
    print(engine.get_execution_report(response.request_id))
