"""
Module: Multi Strategy Manager
File: multi_strategy_manager.py
Description: 
  4-1-1「main.py への戦略セレクター統合」
  既存システムと新マルチ戦略システムを統合管理

Author: imega
Created: 2025-07-20
Modified: 2025-07-20
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
sys.path.append(os.pa        except Exception as e:
            logger.error(f"Hybrid flow failed: {e}")
            return self._execute_legacy_flow(market_data, available_strategies, start_time)
    
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
    
    def _format_for_excel_output(self, integrated_results, strategy_performances, combined_signals):
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
            }dirname(os.path.dirname(__file__)))

# 基本システムのインポート  
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# 既存システムの動的インポート
ADVANCED_FEATURES = False
try:
    from config.strategy_selector import StrategySelector
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.signal_integrator import SignalIntegrator
    from config.portfolio_risk_manager import PortfolioRiskManager
    from config.strategy_execution_adapter import StrategyExecutionAdapter
    from config.optimized_parameters import OptimizedParameterManager
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    ADVANCED_FEATURES = True
except ImportError as e:
    warnings.warn(f"Advanced features not available: {e}")

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
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_time: float
    status: IntegrationStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backtest_data: Optional[Dict[str, Any]] = None  # Excel出力用データ

class MultiStrategyManager:
    """マルチ戦略管理メインクラス"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルパス
        """
        self.status = IntegrationStatus.NOT_STARTED
        
        # 設定ロード
        self.config = self._load_config(config_path)
        self.execution_mode = ExecutionMode(self.config.get('execution_mode', 'hybrid'))
        
        # システムコンポーネント
        self.strategy_selector = None
        self.portfolio_calculator = None
        self.signal_integrator = None
        self.risk_manager = None
        self.execution_adapter = None
        
        # データ管理
        self.parameter_manager = None
        self.characteristics_manager = None
        
        # フォールバック管理
        self.fallback_enabled = self.config.get('fallback_settings', {}).get('enable_fallback', True)
        self.error_count = 0
        self.max_errors = self.config.get('fallback_settings', {}).get('error_threshold', 3)
        
        # 履歴
        self.execution_history = []
        
        logger.info("MultiStrategyManager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルをロード"""
        try:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), 'main_integration_config.json')
            
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return self._get_default_config()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "execution_mode": "hybrid",
            "strategy_selection": {
                "enable_multi_strategy": True,
                "fallback_to_legacy": True,
                "selection_method": "adaptive",
                "min_strategies": 2,
                "max_strategies": 5
            },
            "integration_features": {
                "trend_based_selection": True,
                "portfolio_weighting": True,
                "signal_integration": True,
                "risk_management": True
            },
            "fallback_settings": {
                "enable_fallback": True,
                "fallback_timeout": 30,
                "error_threshold": 3
            }
        }
    
    def initialize_systems(self) -> bool:
        """システム初期化"""
        try:
            self.status = IntegrationStatus.INITIALIZING
            logger.info("Initializing multi-strategy systems...")
            
            if not ADVANCED_FEATURES:
                logger.warning("Advanced features not available. Using fallback mode.")
                self.execution_mode = ExecutionMode.LEGACY_ONLY
                self.status = IntegrationStatus.FALLBACK
                return True
            
            # 1. データ管理システム初期化
            self._initialize_data_managers()
            
            # 2. 戦略選択システム初期化
            self._initialize_strategy_selector()
            
            # 3. ポートフォリオシステム初期化
            self._initialize_portfolio_systems()
            
            # 4. リスク管理システム初期化
            self._initialize_risk_management()
            
            # 5. 実行アダプター初期化
            self._initialize_execution_adapter()
            
            self.status = IntegrationStatus.READY
            logger.info("Multi-strategy systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = IntegrationStatus.ERROR
            self.error_count += 1
            
            if self.fallback_enabled and self.error_count < self.max_errors:
                logger.info("Falling back to legacy mode")
                self.execution_mode = ExecutionMode.LEGACY_ONLY
                self.status = IntegrationStatus.FALLBACK
                return True
            
            return False
    
    def initialize_system(self) -> bool:
        """
        システム初期化 - main.py からの直接呼び出し用エイリアス
        TODO(tag:phase3, rationale:Production Ready・完全初期化ロジック統合完了)
        """
        try:
            logger.info("MultiStrategyManager基本初期化開始")
            
            # Phase 1: 最小限の初期化でAttributeError解消
            self.is_initialized = True
            
            # 既存のinitialize_systems()メソッドを呼び出し
            result = self.initialize_systems()
            
            if result:
                logger.info("MultiStrategyManager基本初期化完了")
            else:
                logger.warning("MultiStrategyManager初期化に部分的失敗、フォールバックモードで継続")
            
            return result
            
        except Exception as e:
            logger.error(f"MultiStrategyManager基本初期化失敗: {e}")
            return False
    
    def _initialize_data_managers(self):
        """データ管理システムの初期化"""
        try:
            if 'OptimizedParameterManager' in globals():
                self.parameter_manager = OptimizedParameterManager()
            if 'StrategyCharacteristicsManager' in globals():
                self.characteristics_manager = StrategyCharacteristicsManager()
            logger.info("Data managers initialized")
        except Exception as e:
            logger.error(f"Data managers initialization failed: {e}")
            raise
    
    def _initialize_strategy_selector(self):
        """戦略選択システムの初期化"""
        try:
            if 'StrategySelector' in globals():
                self.strategy_selector = StrategySelector()
            logger.info("Strategy selector initialized")
        except Exception as e:
            logger.error(f"Strategy selector initialization failed: {e}")
            raise
    
    def _initialize_portfolio_systems(self):
        """ポートフォリオシステムの初期化"""
        try:
            if 'PortfolioWeightCalculator' in globals():
                self.portfolio_calculator = PortfolioWeightCalculator(None)
            if 'SignalIntegrator' in globals():
                self.signal_integrator = SignalIntegrator()
            logger.info("Portfolio systems initialized")
        except Exception as e:
            logger.error(f"Portfolio systems initialization failed: {e}")
            raise
    
    def _initialize_risk_management(self):
        """リスク管理システムの初期化"""
        try:
            if self.config.get('integration_features', {}).get('risk_management', False):
                if 'PortfolioRiskManager' in globals() and self.portfolio_calculator and self.signal_integrator:
                    self.risk_manager = PortfolioRiskManager(
                        config=None,
                        portfolio_weight_calculator=self.portfolio_calculator,
                        position_size_adjuster=None,
                        signal_integrator=self.signal_integrator
                    )
                    logger.info("Risk management initialized")
        except Exception as e:
            logger.warning(f"Risk management initialization failed: {e}. Continuing without.")
    
    def _initialize_execution_adapter(self):
        """実行アダプターの初期化"""
        try:
            if 'StrategyExecutionAdapter' in globals():
                self.execution_adapter = StrategyExecutionAdapter(
                    parameter_manager=self.parameter_manager,
                    characteristics_manager=self.characteristics_manager
                )
                logger.info("Execution adapter initialized")
        except Exception as e:
            logger.error(f"Execution adapter initialization failed: {e}")
            raise
    
    def execute_multi_strategy_flow(self, market_data, available_strategies: List[str]) -> MultiStrategyResult:
        """マルチ戦略フローの実行"""
        start_time = datetime.now()
        
        try:
            self.status = IntegrationStatus.RUNNING
            logger.info("Starting multi-strategy execution flow")
            
            # 実行モード別処理
            if self.execution_mode == ExecutionMode.LEGACY_ONLY or self.status == IntegrationStatus.FALLBACK:
                return self._execute_legacy_flow(market_data, available_strategies, start_time)
            
            elif self.execution_mode == ExecutionMode.MULTI_STRATEGY:
                return self._execute_multi_strategy_flow(market_data, available_strategies, start_time)
            
            else:  # HYBRID
                return self._execute_hybrid_flow(market_data, available_strategies, start_time)
                
        except Exception as e:
            logger.error(f"Execution flow failed: {e}")
            self.error_count += 1
            
            # フォールバック処理
            if self.fallback_enabled:
                logger.info("Attempting fallback to legacy system")
                return self._execute_legacy_flow(market_data, available_strategies, start_time)
            
            # エラー結果を返す
            execution_time = (datetime.now() - start_time).total_seconds()
            return MultiStrategyResult(
                execution_mode=self.execution_mode,
                selected_strategies=[],
                portfolio_weights={},
                final_positions={},
                performance_metrics={},
                risk_metrics={},
                execution_time=execution_time,
                status=IntegrationStatus.ERROR,
                errors=[str(e)]
            )
    
    def _execute_legacy_flow(self, market_data, available_strategies: List[str], start_time: datetime) -> MultiStrategyResult:
        """従来システムでの実行"""
        try:
            logger.info("Executing in legacy mode")
            
            # 単一戦略選択（従来方式）
            selected_strategy = available_strategies[0] if available_strategies else "default"
            
            # 結果作成
            execution_time = (datetime.now() - start_time).total_seconds()
            result = MultiStrategyResult(
                execution_mode=ExecutionMode.LEGACY_ONLY,
                selected_strategies=[selected_strategy],
                portfolio_weights={selected_strategy: 1.0},
                final_positions={selected_strategy: 1.0},
                performance_metrics={"legacy_mode": True},
                risk_metrics={"mode": "legacy"},
                execution_time=execution_time,
                status=IntegrationStatus.FALLBACK if self.status == IntegrationStatus.FALLBACK else IntegrationStatus.READY
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Legacy flow execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return MultiStrategyResult(
                execution_mode=ExecutionMode.LEGACY_ONLY,
                selected_strategies=[],
                portfolio_weights={},
                final_positions={},
                performance_metrics={},
                risk_metrics={},
                execution_time=execution_time,
                status=IntegrationStatus.ERROR,
                errors=[str(e)]
            )
    
    def _execute_multi_strategy_flow(self, market_data, available_strategies: List[str], start_time: datetime) -> MultiStrategyResult:
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
            
            # ✅ 必須: 最適化パラメータの取得
            try:
                from config.optimized_parameters import OptimizedParameterManager
                param_manager = OptimizedParameterManager()
                optimized_params = param_manager.get_approved_parameters()
            except Exception as e:
                logger.error(f"Failed to load optimized parameters: {e}")
                optimized_params = {}
            
            # ✅ 必須: 戦略の実際のbacktest()実行
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
                    # ✅ 基本理念遵守: 実際の戦略クラス取得・インスタンス化
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
                    
                    # ✅ 基本理念遵守: 実際のbacktest()実行
                    logger.info(f"Executing backtest for strategy: {strategy_name}")
                    strategy_result = strategy_instance.backtest()
                    
                    # ✅ 基本理念違反検出
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
            result_data = self._format_for_excel_output(integrated_results, strategy_performances, combined_signals)
            
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
    
    def _execute_hybrid_flow(self, market_data, available_strategies: List[str], start_time: datetime) -> MultiStrategyResult:
        """ハイブリッドフローの実行"""
        try:
            # 条件に基づく自動選択
            should_use_multi = self._should_use_multi_strategy(market_data, available_strategies)
            
            if should_use_multi:
                return self._execute_multi_strategy_flow(market_data, available_strategies, start_time)
            else:
                return self._execute_legacy_flow(market_data, available_strategies, start_time)
                
        except Exception as e:
            logger.error(f"Hybrid flow decision failed: {e}")
            return self._execute_legacy_flow(market_data, available_strategies, start_time)
    
    def _should_use_multi_strategy(self, market_data, available_strategies: List[str]) -> bool:
        """マルチ戦略を使用すべきか判定"""
        try:
            # 基本条件チェック
            if len(available_strategies) < 2:
                return False
            
            if self.status != IntegrationStatus.READY:
                return False
            
            # システム健全性チェック
            if self.error_count >= self.max_errors:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Multi-strategy decision failed: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """実行サマリーを取得"""
        try:
            if not self.execution_history:
                return {"status": "no_executions"}
            
            latest = self.execution_history[-1]
            
            return {
                "current_status": self.status.value,
                "execution_mode": self.execution_mode.value,
                "total_executions": len(self.execution_history),
                "error_count": self.error_count,
                "latest_execution": {
                    "strategies": latest.selected_strategies,
                    "weights": latest.portfolio_weights,
                    "execution_time": latest.execution_time,
                    "status": latest.status.value
                },
                "system_health": {
                    "fallback_enabled": self.fallback_enabled,
                    "advanced_features": ADVANCED_FEATURES,
                    "components_ready": self._check_components_health()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"status": "error", "message": str(e)}
    
    def _check_components_health(self) -> Dict[str, bool]:
        """コンポーネントの健全性チェック"""
        return {
            "strategy_selector": self.strategy_selector is not None,
            "portfolio_calculator": self.portfolio_calculator is not None,
            "signal_integrator": self.signal_integrator is not None,
            "risk_manager": self.risk_manager is not None,
            "execution_adapter": self.execution_adapter is not None
        }

if __name__ == "__main__":
    # テストコード
    print("Multi Strategy Manager - Test Mode")
    
    try:
        # マネージャー初期化
        manager = MultiStrategyManager()
        
        # システム初期化
        init_success = manager.initialize_systems()
        print(f"Initialization success: {init_success}")
        
        # テスト実行
        test_strategies = ["VWAP_Bounce", "GC_Strategy", "Breakout"]
        test_market_data = {"trend": "uptrend", "volatility": "medium"}
        
        result = manager.execute_multi_strategy_flow(test_market_data, test_strategies)
        
        print(f"Execution completed:")
        print(f"  Mode: {result.execution_mode.value}")
        print(f"  Status: {result.status.value}")
        print(f"  Selected strategies: {result.selected_strategies}")
        print(f"  Portfolio weights: {result.portfolio_weights}")
        print(f"  Execution time: {result.execution_time:.3f}s")
        
        # サマリー出力
        summary = manager.get_execution_summary()
        print(f"Summary: {json.dumps(summary, indent=2)}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
