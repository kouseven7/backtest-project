"""
Module: Strategy Execution Pipeline
File: strategy_execution_pipeline.py
Description: 
  4-1-2「複合戦略実行フロー設計・実装」- Pipeline Component
  ステージベースの戦略実行パイプライン
  既存システムとの完全統合

Author: imega
Created: 2025-01-28
Modified: 2025-01-28

Dependencies:
  - config.strategy_selector
  - config.portfolio_weight_calculator
  - config.signal_integrator
  - config.portfolio_risk_manager
"""

import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムのインポート
try:
    from config.strategy_selector import StrategySelector
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.signal_integrator import SignalIntegrator
    from config.portfolio_risk_manager import PortfolioRiskManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class StageStatus(Enum):
    """ステージ実行ステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

@dataclass
class StageResult:
    """ステージ実行結果"""
    stage_id: str
    status: StageStatus
    data: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_success(self) -> bool:
        """成功判定"""
        return self.status == StageStatus.COMPLETED

@dataclass
class PipelineContext:
    """パイプライン実行コンテキスト"""
    execution_id: str
    market_data: Optional[pd.DataFrame] = None
    strategies: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
class StageExecutor(ABC):
    """ステージ実行器の抽象基底クラス"""
    
    def __init__(self, stage_id: str, config: Dict[str, Any]):
        self.stage_id = stage_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{stage_id}")
        
    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """ステージを実行"""
        pass
        
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力データの検証"""
        return True
        
    def handle_error(self, error: Exception, context: PipelineContext) -> StageResult:
        """エラーハンドリング"""
        error_msg = f"Stage {self.stage_id} failed: {str(error)}"
        self.logger.error(error_msg)
        return StageResult(
            stage_id=self.stage_id,
            status=StageStatus.FAILED,
            error_message=error_msg
        )

class StrategySelectionStage(StageExecutor):
    """戦略選択ステージ"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("strategy_selection", config)
        self.strategy_selector = None
        
    def execute(self, context: PipelineContext) -> StageResult:
        """戦略選択を実行"""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(context):
                raise ValueError("Invalid input data for strategy selection")
                
            # StrategySelector の初期化
            if self.strategy_selector is None:
                self.strategy_selector = StrategySelector()
                
            # 戦略選択実行
            selection_result = self.strategy_selector.select_strategies(
                ticker="EXAMPLE",  # デフォルトティッカー
                market_data=context.market_data
            )
            
            # 結果の整理
            result_data = {
                "selected_strategies": selection_result.selected_strategies,
                "strategy_scores": selection_result.strategy_scores,
                "selection_confidence": selection_result.confidence_level,
                "metadata": selection_result.metadata
            }
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_id=self.stage_id,
                status=StageStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self.handle_error(e, context)
            
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力検証"""
        return context.market_data is not None

class WeightCalculationStage(StageExecutor):
    """重み計算ステージ"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("weight_calculation", config)
        self.weight_calculator = None
        
    def execute(self, context: PipelineContext) -> StageResult:
        """重み計算を実行"""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(context):
                raise ValueError("Invalid input data for weight calculation")
                
            # PortfolioWeightCalculator の初期化
            if self.weight_calculator is None:
                self.weight_calculator = PortfolioWeightCalculator()
                
            # 前ステージの結果取得
            strategy_result = context.stage_results.get("strategy_selection")
            if not strategy_result or not strategy_result.is_success():
                raise ValueError("Strategy selection stage did not complete successfully")
                
            strategies = strategy_result.data["selected_strategies"]
            scores = strategy_result.data["strategy_scores"]
            
            # 重み計算実行
            weight_result = self.weight_calculator.calculate_weights(
                strategies=strategies,
                scores=scores,
                market_data=context.market_data,
                parameters=context.parameters or {}
            )
            
            # 結果の整理
            result_data = {
                "strategy_weights": weight_result.get("weights", {}),
                "total_weight": weight_result.get("total_weight", 0.0),
                "weight_confidence": weight_result.get("confidence", 0.0),
                "allocation_details": weight_result.get("allocation_details", {}),
                "metadata": weight_result.get("metadata", {})
            }
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_id=self.stage_id,
                status=StageStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self.handle_error(e, context)
            
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力検証"""
        strategy_result = context.stage_results.get("strategy_selection")
        return (strategy_result is not None and 
                strategy_result.is_success() and
                strategy_result.data and
                "selected_strategies" in strategy_result.data)

class SignalIntegrationStage(StageExecutor):
    """シグナル統合ステージ"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("signal_integration", config)
        self.signal_integrator = None
        
    def execute(self, context: PipelineContext) -> StageResult:
        """シグナル統合を実行"""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(context):
                raise ValueError("Invalid input data for signal integration")
                
            # SignalIntegrator の初期化
            if self.signal_integrator is None:
                self.signal_integrator = SignalIntegrator()
                
            # 前ステージの結果取得
            strategy_result = context.stage_results.get("strategy_selection")
            weight_result = context.stage_results.get("weight_calculation")
            
            strategies = strategy_result.data["selected_strategies"]
            weights = weight_result.data["strategy_weights"]
            
            # シグナル統合実行
            integration_result = self.signal_integrator.integrate_signals(
                strategies=strategies,
                weights=weights,
                market_data=context.market_data,
                parameters=context.parameters or {}
            )
            
            # 結果の整理
            result_data = {
                "integrated_signals": integration_result.get("signals", {}),
                "signal_strength": integration_result.get("strength", 0.0),
                "conflict_resolution": integration_result.get("conflicts", {}),
                "signal_confidence": integration_result.get("confidence", 0.0),
                "metadata": integration_result.get("metadata", {})
            }
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_id=self.stage_id,
                status=StageStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self.handle_error(e, context)
            
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力検証"""
        strategy_result = context.stage_results.get("strategy_selection")
        weight_result = context.stage_results.get("weight_calculation")
        return (strategy_result is not None and strategy_result.is_success() and
                weight_result is not None and weight_result.is_success())

class RiskAdjustmentStage(StageExecutor):
    """リスク調整ステージ"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("risk_adjustment", config)
        self.risk_manager = None
        
    def execute(self, context: PipelineContext) -> StageResult:
        """リスク調整を実行"""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(context):
                raise ValueError("Invalid input data for risk adjustment")
                
            # PortfolioRiskManager の初期化
            if self.risk_manager is None:
                self.risk_manager = PortfolioRiskManager()
                
            # 前ステージの結果取得
            signal_result = context.stage_results.get("signal_integration")
            weight_result = context.stage_results.get("weight_calculation")
            
            signals = signal_result.data["integrated_signals"]
            weights = weight_result.data["strategy_weights"]
            
            # リスク調整実行
            risk_result = self.risk_manager.adjust_portfolio_risk(
                signals=signals,
                weights=weights,
                market_data=context.market_data,
                parameters=context.parameters or {}
            )
            
            # 結果の整理
            result_data = {
                "adjusted_weights": risk_result.get("adjusted_weights", {}),
                "risk_metrics": risk_result.get("risk_metrics", {}),
                "adjustments": risk_result.get("adjustments", {}),
                "risk_confidence": risk_result.get("confidence", 0.0),
                "metadata": risk_result.get("metadata", {})
            }
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_id=self.stage_id,
                status=StageStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self.handle_error(e, context)
            
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力検証"""
        signal_result = context.stage_results.get("signal_integration")
        weight_result = context.stage_results.get("weight_calculation")
        return (signal_result is not None and signal_result.is_success() and
                weight_result is not None and weight_result.is_success())

class ExecutionStage(StageExecutor):
    """実行ステージ"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("execution", config)
        
    def execute(self, context: PipelineContext) -> StageResult:
        """最終実行を行う"""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(context):
                raise ValueError("Invalid input data for execution")
                
            # 前ステージの結果取得
            signal_result = context.stage_results.get("signal_integration")
            risk_result = context.stage_results.get("risk_adjustment")
            
            # リスク調整が無効/失敗の場合は元の重みを使用
            if risk_result and risk_result.is_success():
                weights = risk_result.data["adjusted_weights"]
                risk_metrics = risk_result.data["risk_metrics"]
            else:
                weight_result = context.stage_results.get("weight_calculation")
                weights = weight_result.data["strategy_weights"]
                risk_metrics = {}
                
            signals = signal_result.data["integrated_signals"]
            
            # 実行決定の生成
            execution_decisions = self._generate_execution_decisions(
                signals, weights, risk_metrics
            )
            
            # 結果の整理
            result_data = {
                "execution_decisions": execution_decisions,
                "final_weights": weights,
                "final_signals": signals,
                "risk_metrics": risk_metrics,
                "execution_confidence": self._calculate_execution_confidence(context),
                "metadata": {
                    "execution_timestamp": datetime.now().isoformat(),
                    "context_id": context.execution_id
                }
            }
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage_id=self.stage_id,
                status=StageStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return self.handle_error(e, context)
            
    def _generate_execution_decisions(self, signals: Dict, weights: Dict, 
                                    risk_metrics: Dict) -> Dict[str, Any]:
        """実行決定を生成"""
        decisions = {}
        
        for strategy, signal_data in signals.items():
            if strategy in weights and weights[strategy] > 0:
                decisions[strategy] = {
                    "action": signal_data.get("action", "hold"),
                    "weight": weights[strategy],
                    "confidence": signal_data.get("confidence", 0.0),
                    "risk_adjustment": risk_metrics.get(strategy, {})
                }
                
        return decisions
        
    def _calculate_execution_confidence(self, context: PipelineContext) -> float:
        """実行信頼度を計算"""
        total_confidence = 0.0
        stage_count = 0
        
        for stage_id, result in context.stage_results.items():
            if result.is_success() and result.data:
                confidence_key = f"{stage_id.split('_')[0]}_confidence"
                if confidence_key in result.data:
                    total_confidence += result.data[confidence_key]
                    stage_count += 1
                    
        return total_confidence / max(stage_count, 1)
        
    def validate_inputs(self, context: PipelineContext) -> bool:
        """入力検証"""
        signal_result = context.stage_results.get("signal_integration")
        weight_result = context.stage_results.get("weight_calculation")
        return (signal_result is not None and signal_result.is_success() and
                weight_result is not None and weight_result.is_success())

class StrategyExecutionPipeline:
    """戦略実行パイプライン"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.stages = self._initialize_stages()
        self.logger = logging.getLogger(__name__)
        self.execution_history: List[PipelineContext] = []
        self._lock = threading.Lock()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
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
            "execution_pipeline": {
                "stages": [
                    {"id": "strategy_selection", "enabled": True, "timeout_seconds": 30, "critical": True},
                    {"id": "weight_calculation", "enabled": True, "timeout_seconds": 20, "critical": True},
                    {"id": "signal_integration", "enabled": True, "timeout_seconds": 15, "critical": True},
                    {"id": "risk_adjustment", "enabled": True, "timeout_seconds": 25, "critical": False},
                    {"id": "execution", "enabled": True, "timeout_seconds": 60, "critical": True}
                ]
            },
            "thresholds": {
                "max_execution_time": 180
            }
        }
        
    def _initialize_stages(self) -> Dict[str, StageExecutor]:
        """ステージ実行器の初期化"""
        stages = {}
        
        stage_classes = {
            "strategy_selection": StrategySelectionStage,
            "weight_calculation": WeightCalculationStage,
            "signal_integration": SignalIntegrationStage,
            "risk_adjustment": RiskAdjustmentStage,
            "execution": ExecutionStage
        }
        
        for stage_config in self.config["execution_pipeline"]["stages"]:
            if stage_config["enabled"]:
                stage_id = stage_config["id"]
                if stage_id in stage_classes:
                    stages[stage_id] = stage_classes[stage_id](stage_config)
                    
        return stages
        
    def execute(self, market_data: pd.DataFrame, 
                parameters: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """パイプライン実行"""
        execution_id = f"exec_{int(time.time() * 1000)}"
        context = PipelineContext(
            execution_id=execution_id,
            market_data=market_data,
            parameters=parameters or {}
        )
        
        try:
            # ステージ順次実行
            for stage_config in self.config["execution_pipeline"]["stages"]:
                if not stage_config["enabled"]:
                    continue
                    
                stage_id = stage_config["id"]
                if stage_id not in self.stages:
                    continue
                    
                # ステージ実行
                result = self._execute_stage_with_timeout(
                    self.stages[stage_id], 
                    context, 
                    stage_config
                )
                
                context.stage_results[stage_id] = result
                
                # クリティカルステージの失敗チェック
                if stage_config.get("critical", False) and not result.is_success():
                    self.logger.error(f"Critical stage {stage_id} failed: {result.error_message}")
                    break
                    
            # 実行履歴に追加
            with self._lock:
                self.execution_history.append(context)
                if len(self.execution_history) > 100:
                    self.execution_history.pop(0)
                    
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            
        return context
        
    def _execute_stage_with_timeout(self, stage: StageExecutor, 
                                  context: PipelineContext,
                                  stage_config: Dict[str, Any]) -> StageResult:
        """タイムアウト付きでステージを実行"""
        timeout = stage_config.get("timeout_seconds", 30)
        retry_attempts = stage_config.get("retry_attempts", 1)
        
        for attempt in range(retry_attempts + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(stage.execute, context)
                    result = future.result(timeout=timeout)
                    
                if result.is_success():
                    return result
                elif attempt < retry_attempts:
                    self.logger.warning(f"Stage {stage.stage_id} attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.1 * (attempt + 1))  # バックオフ
                    
            except TimeoutError:
                self.logger.error(f"Stage {stage.stage_id} timed out after {timeout} seconds")
                return StageResult(
                    stage_id=stage.stage_id,
                    status=StageStatus.TIMEOUT,
                    error_message=f"Execution timed out after {timeout} seconds"
                )
            except Exception as e:
                self.logger.error(f"Stage {stage.stage_id} execution error: {str(e)}")
                if attempt >= retry_attempts:
                    return stage.handle_error(e, context)
                    
        # すべての再試行が失敗した場合
        return StageResult(
            stage_id=stage.stage_id,
            status=StageStatus.FAILED,
            error_message=f"Failed after {retry_attempts + 1} attempts"
        )
        
    def get_execution_summary(self, context: PipelineContext) -> Dict[str, Any]:
        """実行サマリーを取得"""
        total_time = sum(result.execution_time for result in context.stage_results.values())
        successful_stages = sum(1 for result in context.stage_results.values() if result.is_success())
        
        summary = {
            "execution_id": context.execution_id,
            "total_execution_time": total_time,
            "successful_stages": successful_stages,
            "total_stages": len(context.stage_results),
            "success_rate": successful_stages / max(len(context.stage_results), 1),
            "stage_details": {}
        }
        
        for stage_id, result in context.stage_results.items():
            summary["stage_details"][stage_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "retry_count": result.retry_count,
                "has_data": result.data is not None
            }
            
        return summary
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標を取得"""
        if not self.execution_history:
            return {"message": "No execution history available"}
            
        with self._lock:
            recent_executions = self.execution_history[-10:]  # 直近10件
            
        total_executions = len(recent_executions)
        successful_executions = 0
        avg_execution_time = 0.0
        stage_success_rates = {}
        
        for context in recent_executions:
            stage_times = sum(result.execution_time for result in context.stage_results.values())
            avg_execution_time += stage_times
            
            execution_success = True
            for stage_id, result in context.stage_results.items():
                if stage_id not in stage_success_rates:
                    stage_success_rates[stage_id] = []
                stage_success_rates[stage_id].append(result.is_success())
                
                if not result.is_success():
                    execution_success = False
                    
            if execution_success:
                successful_executions += 1
                
        metrics = {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": avg_execution_time / total_executions,
            "stage_success_rates": {
                stage_id: sum(successes) / len(successes)
                for stage_id, successes in stage_success_rates.items()
            }
        }
        
        return metrics

if __name__ == "__main__":
    # テスト用のサンプル実行
    pipeline = StrategyExecutionPipeline()
    
    # サンプルデータ生成
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='1H'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # パイプライン実行
    result = pipeline.execute(sample_data)
    
    # 結果表示
    summary = pipeline.get_summary(result)
    print("Pipeline Execution Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
