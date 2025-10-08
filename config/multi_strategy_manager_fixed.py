"""
Module: Multi Strategy Manager (Fixed)
File: multi_strategy_manager_fixed.py
Description: 
  Phase 4-A実装: バックテスト基本理念準拠のMultiStrategyManager

Author: imega
Created: 2025-10-07
Modified: 2025-10-07
"""

import os
import sys
import json
import logging
import warnings
import pandas as pd
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

# 基本システムのインポート  
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """実行モード"""
    LEGACY_ONLY = "legacy_only"           # 従来システムのみ
    MULTI_STRATEGY = "multi_strategy"     # マルチ戦略システム
    HYBRID = "hybrid"                     # ハイブリッド（自動切替）

class IntegrationStatus(Enum):
    """統合ステータス"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing" 
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    FALLBACK = "fallback"

@dataclass
class MultiStrategyResult:
    """マルチ戦略実行結果"""
    execution_mode: ExecutionMode
    selected_strategies: List[str]
    portfolio_weights: Dict[str, float]
    final_positions: Dict[str, float]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    execution_time: float
    status: IntegrationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backtest_data: Optional[Dict[str, Any]] = None  # Excel出力用データ

class MultiStrategyManager:
    """マルチ戦略管理メインクラス (Phase 4-A実装)"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config_path = config_path
        self.execution_mode = ExecutionMode.MULTI_STRATEGY
        self.status = IntegrationStatus.READY
        self.execution_history: List[MultiStrategyResult] = []
        
        logger.info("MultiStrategyManager initialized with backtest principle compliance")
    
    def initialize_systems(self) -> bool:
        """システム初期化"""
        try:
            self.status = IntegrationStatus.INITIALIZING
            # 基本的な初期化のみ実行
            self.status = IntegrationStatus.READY
            logger.info("MultiStrategyManager systems initialized successfully")
            return True
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def execute_multi_strategy_flow(self, market_data: Dict[str, Any], available_strategies: List[str]) -> MultiStrategyResult:
        """
        メインエントリーポイント: マルチ戦略フロー実行
        """
        start_time = datetime.now()
        self.status = IntegrationStatus.RUNNING
        
        try:
            # Phase 4-A実装: 実際のbacktest()実行
            return self._execute_multi_strategy_flow(market_data, available_strategies, start_time)
            
        except Exception as e:
            logger.error(f"Multi-strategy execution failed: {e}")
            # TODO(tag:backtest_execution, rationale:implement proper error handling)
            return MultiStrategyResult(
                execution_mode=ExecutionMode.MULTI_STRATEGY,
                selected_strategies=[],
                portfolio_weights={},
                final_positions={},
                performance_metrics={},
                risk_metrics={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=IntegrationStatus.ERROR,
                errors=[str(e)]
            )
    
    def _execute_multi_strategy_flow(self, market_data: Dict[str, Any], available_strategies: List[str], start_time: datetime) -> MultiStrategyResult:
        """
        バックテスト基本理念遵守: 統合システムでも実際のbacktest()実行必須
        apply_strategies_with_optimized_params()パターン移植
        """
        try:
            logger.info("Executing multi-strategy flow with actual backtest execution")
            
            # マーケットデータからstock_dataとindex_dataを抽出
            stock_data = market_data.get("data", None)
            index_data = market_data.get("index", None)
            
            if stock_data is None:
                raise ValueError("Stock data is required for backtest execution")
            
            # [OK] 必須: 最適化パラメータの取得
            try:
                from config.optimized_parameters import OptimizedParameterManager
                param_manager = OptimizedParameterManager()
                optimized_params = param_manager.get_approved_parameters()
            except Exception as e:
                logger.error(f"Failed to load optimized parameters: {e}")
                optimized_params = {}
            
            # [OK] 必須: 戦略の実際のbacktest()実行
            integrated_results = stock_data.copy()
            strategy_performances = {}
            combined_signals = {}
            
            # 戦略優先順位 (apply_strategies_with_optimized_params()パターン)
            strategy_priority = [
                ('VWAPBreakoutStrategy', 'VWAP_Breakout'),
                ('MomentumInvestingStrategy', 'Momentum_Investing'), 
                ('BreakoutStrategy', 'Breakout'),
                ('VWAPBounceStrategy', 'VWAP_Bounce'),
                ('OpeningGapStrategy', 'Opening_Gap'),
                ('ContrarianStrategy', 'contrarian_strategy'),
                ('GCStrategy', 'gc_strategy_signal')
            ]
            
            # 統合されたシグナル列を初期化
            integrated_results['Entry_Signal'] = 0
            integrated_results['Exit_Signal'] = 0
            integrated_results['Strategy'] = ''
            integrated_results['Position'] = 0
            
            successful_strategies = []
            
            for strategy_name, module_name in strategy_priority:
                if strategy_name not in available_strategies:
                    continue
                    
                try:
                    # [OK] 基本理念遵守: 実際の戦略クラス取得・インスタンス化
                    strategy_class = self._get_strategy_class(strategy_name, module_name)
                    params = optimized_params.get(strategy_name, {})
                    
                    # 戦略インスタンス化 (main.pyパターンに準拠)
                    if strategy_name == 'VWAPBreakoutStrategy':
                        strategy_instance = strategy_class(
                            data=stock_data.copy(),
                            index_data=index_data,
                            params=params,
                            price_column="Adj Close"
                        )
                    elif strategy_name == 'OpeningGapStrategy':
                        strategy_instance = strategy_class(
                            data=stock_data.copy(),
                            params=params,
                            price_column="Adj Close",
                            dow_data=index_data
                        )
                    else:
                        strategy_instance = strategy_class(
                            data=stock_data.copy(),
                            params=params,
                            price_column="Adj Close"
                        )
                    
                    # [OK] 基本理念遵守: 実際のbacktest()実行
                    logger.info(f"Executing backtest for strategy: {strategy_name}")
                    strategy_result = strategy_instance.backtest()
                    
                    # [OK] 基本理念違反検出
                    self._validate_backtest_output(strategy_result, strategy_name)
                    
                    # シグナル統合処理
                    entry_count = (strategy_result.get('Entry_Signal', pd.Series()) == 1).sum()
                    exit_count = (strategy_result.get('Exit_Signal', pd.Series()) == -1).sum()
                    
                    logger.info(f"Strategy {strategy_name}: {entry_count} entries, {exit_count} exits")
                    
                    # パフォーマンス記録
                    strategy_performances[strategy_name] = {
                        'entries': int(entry_count),
                        'exits': int(exit_count),
                        'backtest_completed': True
                    }
                    
                    combined_signals[strategy_name] = strategy_result
                    successful_strategies.append(strategy_name)
                    
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} execution failed: {e}")
                    # TODO(tag:backtest_execution, rationale:strategy execution failure handling)
                    strategy_performances[strategy_name] = {
                        'entries': 0,
                        'exits': 0,  
                        'backtest_completed': False,
                        'error': str(e)
                    }
            
            # 結果作成
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Excel出力対応: 完全データ返却
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: result_data = self._format_for_excel_output(integrated_results, strategy_performances, combined_signals)
            
            result = MultiStrategyResult(
                execution_mode=ExecutionMode.MULTI_STRATEGY,
                selected_strategies=successful_strategies,
                portfolio_weights={strategy: 1.0/len(successful_strategies) for strategy in successful_strategies} if successful_strategies else {},
                final_positions={strategy: 1.0/len(successful_strategies) for strategy in successful_strategies} if successful_strategies else {},
                performance_metrics=strategy_performances,
                risk_metrics={"backtest_principle_compliant": True},
                execution_time=execution_time,
                status=IntegrationStatus.READY,
                # Excel出力用データ追加
                backtest_data=result_data
            )
            
            self.execution_history.append(result)
            logger.info(f"Multi-strategy flow completed. Successful strategies: {len(successful_strategies)}")
            return result
            
        except Exception as e:
            logger.error(f"Multi-strategy flow failed: {e}")
            # TODO(tag:backtest_execution, rationale:implement proper error handling)
            raise

    # Phase 4-A-2: 支援メソッド実装 (バックテスト基本理念準拠)
    
    def _get_strategy_class(self, strategy_name: str, module_name: str):
        """戦略クラスの動的取得"""
        try:
            module = importlib.import_module(f'strategies.{module_name}')
            
            # 戦略名に基づくクラス取得
            strategy_classes = {
                'VWAPBreakoutStrategy': 'VWAPBreakoutStrategy',
                'MomentumInvestingStrategy': 'MomentumInvestingStrategy',
                'BreakoutStrategy': 'BreakoutStrategy',
                'VWAPBounceStrategy': 'VWAPBounceStrategy',
                'OpeningGapStrategy': 'OpeningGapStrategy', 
                'ContrarianStrategy': 'ContrarianStrategy',
                'GCStrategy': 'GCStrategy'
            }
            
            class_name = strategy_classes.get(strategy_name, strategy_name)
            strategy_class = getattr(module, class_name)
            
            logger.debug(f"Successfully loaded strategy class: {strategy_name}")
            return strategy_class
            
        except Exception as e:
            logger.error(f"Failed to load strategy class {strategy_name}: {e}")
            # TODO(tag:backtest_execution, rationale:implement fallback strategy loading)
            raise ImportError(f"Cannot load strategy: {strategy_name}")
    
    def _validate_backtest_output(self, strategy_result, strategy_name: str):
        """バックテスト基本理念違反検出"""
        try:
            # 基本理念遵守チェック: Entry_Signal/Exit_Signal存在確認
            required_columns = ['Entry_Signal', 'Exit_Signal']
            missing_columns = [col for col in required_columns if col not in strategy_result.columns]
            
            if missing_columns:
                violation_msg = f"Strategy {strategy_name} violates backtest principle: missing {missing_columns}"
                logger.error(violation_msg)
                # TODO(tag:backtest_execution, rationale:fix principle violations)
                raise ValueError(violation_msg)
            
            # 取引数チェック (基本理念違反検出)
            if 'Entry_Signal' in strategy_result.columns and 'Exit_Signal' in strategy_result.columns:
                total_trades = (strategy_result['Entry_Signal'] == 1).sum() + (strategy_result['Exit_Signal'] == -1).sum()  
                if total_trades == 0:
                    logger.warning(f"Strategy {strategy_name}: Zero trades - potential backtest principle violation")
            
            # データ完整性チェック
            if len(strategy_result) == 0:
                violation_msg = f"Strategy {strategy_name}: Empty result data violates backtest principle"
                logger.error(violation_msg)
                # TODO(tag:backtest_execution, rationale:implement proper data validation)
                raise ValueError(violation_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Backtest validation failed for {strategy_name}: {e}")
            # TODO(tag:backtest_execution, rationale:improve validation robustness)
            raise
    
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: def _format_for_excel_output(self, integrated_results, strategy_performances, combined_signals):
        """Excel出力対応: 完全データフォーマット"""
        try:
            # DSSMSレベルの品質を目標とした出力データ準備
            output_data = {
                'stock_data': integrated_results,
                'strategy_performances': strategy_performances,
                'combined_signals': combined_signals,
                'execution_metadata': {
                    'execution_time': datetime.now().isoformat(),
                    'backtest_period': f"{integrated_results.index.min()} -> {integrated_results.index.max()}",
                    'total_strategies': len(strategy_performances),
                    'successful_strategies': len([s for s, perf in strategy_performances.items() if perf.get('backtest_completed', False)]),
                    'total_trades': sum([perf.get('entries', 0) + perf.get('exits', 0) for perf in strategy_performances.values()])
                }
            }
            
            logger.info(f"Excel output data formatted: {output_data['execution_metadata']['total_trades']} total trades")
            return output_data
            
        except Exception as e:
            logger.error(f"Excel output formatting failed: {e}")
            # TODO(tag:backtest_execution, rationale:implement robust excel formatting)
            return {
                'stock_data': integrated_results,
                'error': str(e)
            }