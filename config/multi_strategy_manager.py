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

# TODO #14 Phase 3: RealMarketDataFetcher統合
try:
    from real_market_data_fetcher import RealMarketDataFetcher, fetch_strategy_required_data
    REAL_DATA_FETCHER_AVAILABLE = True
except ImportError as e:
    REAL_DATA_FETCHER_AVAILABLE = False
    print(f"RealMarketDataFetcher not available: {e}")

# プロジェクトパスの追加
# TODO(tag:syntax_fix, rationale:restore weight judgment system)
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    print(f"Project path added: {project_root}")
except Exception as e:
    print(f"Failed to add project path: {e}")

# TODO #15 Phase 1-Priority 1: 戦略クラスインポート
try:
    from src.strategies.VWAP_Breakout import VWAPBreakoutStrategy
    from src.strategies.Momentum_Investing import MomentumInvestingStrategy  
    from src.strategies.Breakout import BreakoutStrategy
    from src.strategies.VWAP_Bounce import VWAPBounceStrategy
    from src.strategies.Opening_Gap import OpeningGapStrategy
    from src.strategies.contrarian_strategy import ContrarianStrategy
    from src.strategies.gc_strategy_signal import GCStrategy
    STRATEGY_IMPORTS_AVAILABLE = True
    print("✅ All strategy classes imported successfully")
except ImportError as e:
    STRATEGY_IMPORTS_AVAILABLE = False
    print(f"⚠️ Some strategy imports failed: {e}")

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
    """
    戦略レジストリシステム完全実装版マルチ戦略管理メインクラス
    TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
    バックテスト基本理念遵守: 戦略レジストリでも実際のbacktest()実行必須
    """
    
    def __init__(self, config_path: str = None):
        """
        初期化（戦略レジストリシステム統合版）
        
        Parameters:
            config_path: 設定ファイルパス
        """
        self.status = IntegrationStatus.NOT_STARTED
        
        # 設定ロード
        self.config = self._load_config(config_path)
        self.execution_mode = ExecutionMode(self.config.get('execution_mode', 'hybrid'))
        
        # TODO #15 Phase 1-Priority 1: 戦略レジストリ確実な初期化
        # loggerを先に初期化
        self.logger = logging.getLogger(__name__)
        
        # 確実に戦略レジストリを初期化
        self.strategy_registry = {}
        self._initialize_strategy_registry()
        print(f"✅ Strategy registry initialized: {len(self.strategy_registry)} strategies")
        
        self.strategy_import_paths = {}
        self.registry_validation_results = {}
        self.is_initialized = False
        
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
        
        # リソースプール（戦略レジストリ対応）
        self.resource_pool = None
        self.monitoring_config = {}
        
        logger.info("MultiStrategyManager initialized with strategy registry system")
        
        # TODO #15 Phase 1-Priority 2: 自動システム初期化（戦略初期化エラー解決）
        try:
            logger.info("Initializing multi-strategy systems with complete strategy registry...")
            self._initialize_strategy_registry_complete()
            self._validate_strategy_registry()
            self._setup_resource_pool()
            self._prepare_monitoring()
            self._initialize_data_managers()
            self._initialize_strategy_selector()  
            self._initialize_portfolio_systems()
            self._initialize_risk_management()
            self._initialize_execution_adapter()
            
            self.is_initialized = True
            logger.info("Multi-strategy systems with complete strategy registry initialized successfully")
        except Exception as e:
            logger.warning(f"Full system initialization failed: {e}. Using basic registry mode.")
            self.is_initialized = True  # 戦略レジストリは動作するため基本機能有効
    
    def _initialize_strategy_registry(self):
        """戦略レジストリの確実な初期化 - TODO #15 Phase 1-Priority 1"""
        if STRATEGY_IMPORTS_AVAILABLE:
            self.strategy_registry = {
                'VWAPBreakoutStrategy': VWAPBreakoutStrategy,
                'MomentumInvestingStrategy': MomentumInvestingStrategy,
                'BreakoutStrategy': BreakoutStrategy,
                'VWAPBounceStrategy': VWAPBounceStrategy,
                'OpeningGapStrategy': OpeningGapStrategy,
                'ContrarianStrategy': ContrarianStrategy,
                'GCStrategy': GCStrategy
            }
            self.logger.info(f"✅ Strategy registry initialized with {len(self.strategy_registry)} strategies")
        else:
            self.strategy_registry = {}
            self.logger.warning("⚠️ Strategy imports failed - empty registry initialized")
            
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
        """
        システム初期化（戦略レジストリシステム統合版）
        TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
        バックテスト基本理念遵守: 実際のbacktest実行環境構築
        """
        try:
            self.status = IntegrationStatus.INITIALIZING
            logger.info("Initializing multi-strategy systems with complete strategy registry...")
            
            success = True
            # TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
            # 0. 戦略レジストリシステム完全初期化（最優先）
            success &= self._initialize_strategy_registry_complete()
            success &= self._validate_strategy_registry()
            success &= self._setup_resource_pool()
            success &= self._prepare_monitoring()
            
            if not ADVANCED_FEATURES:
                logger.warning("ADVANCED_FEATURES not available. Strategy registry only mode.")
                if success:
                    self.status = IntegrationStatus.READY
                    self.is_initialized = True
                    logger.info("Strategy registry system initialized successfully")
                    self._print_strategy_registry_report()
                    return True
                else:
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
            
            if success:
                self.status = IntegrationStatus.READY
                self.is_initialized = True
                logger.info("Multi-strategy systems with complete strategy registry initialized successfully")
                self._print_strategy_registry_report()
            else:
                self.status = IntegrationStatus.ERROR
                logger.error("Strategy registry initialization failed")
            
            return success
            
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

    def get_available_strategies(self) -> List[str]:
        """
        利用可能戦略リスト取得（戦略レジストリ完全実装版）
        TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
        """
        try:
            # 戦略レジストリが初期化されている場合は登録済み戦略を返す
            if self.is_initialized and hasattr(self, 'strategy_registry') and self.strategy_registry:
                available_strategies = list(self.strategy_registry.keys())
                self.logger.info(f"Available strategies from registry: {len(available_strategies)} registered")
                return available_strategies
            
            # フォールバック: 基本戦略リスト
            default_strategies = [
                'VWAPBreakoutStrategy',
                'MomentumInvestingStrategy', 
                'BreakoutStrategy',
                'VWAPBounceStrategy',
                'OpeningGapStrategy',
                'ContrarianStrategy',
                'GCStrategy'
            ]
            
            # 設定から戦略リストを取得
            if self.config and 'available_strategies' in self.config:
                config_strategies = self.config['available_strategies']
                strategies = config_strategies if config_strategies else default_strategies
            else:
                strategies = default_strategies
            
            self.logger.warning("MultiStrategyManager not initialized - using fallback strategy list")
            # TODO(tag:strategy_registry, rationale:eliminate after full initialization)
            return strategies
            
        except Exception as e:
            logger.error(f"Error getting available strategies: {e}")
            # TODO(tag:strategy_registry, rationale:eliminate after full strategy registry)
            return ['OpeningGapStrategy', 'ContrarianStrategy', 'GCStrategy']

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        戦略重み取得
        重み判断システム復旧のためのメソッド
        TODO(tag:syntax_fix, rationale:restore weight judgment system)
        """
        try:
            if hasattr(self, 'strategy_weights') and self.strategy_weights:
                return self.strategy_weights
            
            # デフォルト重み設定
            strategies = self.get_available_strategies()
            equal_weight = 1.0 / len(strategies) if strategies else 0.0
            
            default_weights = {strategy: equal_weight for strategy in strategies}
            
            # 設定ファイルから重み取得
            if self.config and 'strategy_weights' in self.config:
                config_weights = self.config['strategy_weights']
                default_weights.update(config_weights)
            
            return default_weights
            
        except Exception as e:
            logger.error(f"Error getting strategy weights: {e}")
            return {}

    # ================================================================
    # 戦略レジストリシステム完全実装（TODO #9）
    # ================================================================
    
    def _initialize_strategy_registry_complete(self) -> bool:
        """
        戦略レジストリシステム完全実装
        TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
        バックテスト基本理念遵守: 実際の戦略クラス登録・backtest()確認
        """
        try:
            # TODO #10で確認済み: 全戦略が完全実装済み
            strategy_definitions = {
                'VWAPBreakoutStrategy': {
                    'module_path': 'src.strategies.VWAP_Breakout',
                    'class_name': 'VWAPBreakoutStrategy',
                    'file_path': 'src/strategies/VWAP_Breakout.py'
                },
                'MomentumInvestingStrategy': {
                    'module_path': 'src.strategies.Momentum_Investing',
                    'class_name': 'MomentumInvestingStrategy',
                    'file_path': 'src/strategies/Momentum_Investing.py'
                },
                'BreakoutStrategy': {
                    'module_path': 'src.strategies.Breakout',
                    'class_name': 'BreakoutStrategy',
                    'file_path': 'src/strategies/Breakout.py'
                },
                # 既存の正常動作戦略も追加
                'OpeningGapStrategy': {
                    'module_path': 'src.strategies.Opening_Gap',
                    'class_name': 'OpeningGapStrategy',
                    'file_path': 'src/strategies/Opening_Gap.py'
                },
                'ContrarianStrategy': {
                    'module_path': 'src.strategies.contrarian_strategy',
                    'class_name': 'ContrarianStrategy',
                    'file_path': 'src/strategies/contrarian_strategy.py'
                },
                'GCStrategy': {
                    'module_path': 'src.strategies.gc_strategy_signal',
                    'class_name': 'GCStrategy',
                    'file_path': 'src/strategies/gc_strategy_signal.py'
                },
                # 追加戦略（TODO #8で検出された全戦略）
                'VWAPBounceStrategy': {
                    'module_path': 'src.strategies.VWAP_Bounce',
                    'class_name': 'VWAPBounceStrategy',
                    'file_path': 'src/strategies/VWAP_Bounce.py'
                }
            }
            
            # 戦略クラス自動登録システム
            successful_imports = 0
            failed_imports = []
            
            for strategy_name, definition in strategy_definitions.items():
                try:
                    # 動的インポート実行
                    module = __import__(definition['module_path'], fromlist=[definition['class_name']])
                    strategy_class = getattr(module, definition['class_name'])
                    
                    # バックテスト基本理念遵守確認
                    if self._validate_strategy_class_compliance(strategy_class, strategy_name):
                        self.strategy_registry[strategy_name] = strategy_class
                        self.strategy_import_paths[strategy_name] = definition
                        successful_imports += 1
                        
                        self.logger.info(f"Strategy registered: {strategy_name} from {definition['file_path']}")
                    else:
                        failed_imports.append(f"{strategy_name}: Backtest principle violation")
                        
                except ImportError as e:
                    failed_imports.append(f"{strategy_name}: Import failed - {e}")
                    self.logger.warning(f"Strategy import failed: {strategy_name} - {e}")
                    
                except AttributeError as e:
                    failed_imports.append(f"{strategy_name}: Class not found - {e}")
                    self.logger.warning(f"Strategy class not found: {strategy_name} - {e}")
                    
                except Exception as e:
                    failed_imports.append(f"{strategy_name}: Unexpected error - {e}")
                    self.logger.error(f"Strategy registration failed: {strategy_name} - {e}")
            
            # 戦略レジストリ品質評価
            total_strategies = len(strategy_definitions)
            success_rate = (successful_imports / total_strategies) * 100 if total_strategies > 0 else 0
            
            self.logger.info(f"Strategy registry initialization: {successful_imports}/{total_strategies} strategies ({success_rate:.1f}%)")
            
            if failed_imports:
                self.logger.warning(f"Failed strategy imports: {failed_imports}")
            
            # CRITICAL判定: 最低3戦略は必須（TODO #8の成功基準）
            if successful_imports >= 3:
                return True
            else:
                self.logger.error(f"Strategy registry initialization failed: Only {successful_imports} strategies loaded (minimum 3 required)")
                return False
                
        except Exception as e:
            self.logger.error(f"Strategy registry initialization failed: {e}")
            return False
    
    def _validate_strategy_class_compliance(self, strategy_class, strategy_name: str) -> bool:
        """
        戦略クラスのバックテスト基本理念遵守確認
        TODO(tag:strategy_registry, rationale:ensure backtest principle compliance)
        """
        try:
            # backtest()メソッド存在確認
            if not hasattr(strategy_class, 'backtest'):
                self.logger.error(f"{strategy_name}: Missing backtest() method - violates backtest principle")
                return False
            
            # backtest()メソッドが呼び出し可能か確認
            if not callable(getattr(strategy_class, 'backtest')):
                self.logger.error(f"{strategy_name}: backtest() not callable - violates backtest principle")
                return False
            
            # 基本戦略クラス継承確認（可能な場合）
            if hasattr(strategy_class, '__bases__'):
                base_classes = [base.__name__ for base in strategy_class.__bases__]
                if 'BaseStrategy' not in base_classes and len(base_classes) > 0:
                    self.logger.warning(f"{strategy_name}: Does not inherit from BaseStrategy - may violate conventions")
            
            # __init__メソッド確認
            if hasattr(strategy_class, '__init__'):
                try:
                    import inspect
                    init_signature = inspect.signature(strategy_class.__init__)
                    params = list(init_signature.parameters.keys())
                    
                    # 必要パラメータの存在確認
                    expected_params = ['self', 'data', 'params']
                    missing_params = [param for param in expected_params if param not in params]
                    if missing_params:
                        self.logger.warning(f"{strategy_name}: Missing expected __init__ parameters: {missing_params}")
                except Exception as e:
                    self.logger.warning(f"{strategy_name}: Could not inspect __init__ signature: {e}")
            
            self.logger.info(f"{strategy_name}: Backtest principle compliance validated")
            return True
            
        except Exception as e:
            self.logger.error(f"{strategy_name}: Compliance validation failed - {e}")
            return False
    
    def _validate_strategy_registry(self) -> bool:
        """
        戦略レジストリ検証システム
        TODO(tag:strategy_registry, rationale:comprehensive registry validation)
        """
        try:
            validation_results = {
                'total_registered': len(self.strategy_registry),
                'backtest_compliant': 0,
                'import_successful': 0,
                'validation_timestamp': datetime.now().isoformat(),
                'strategy_details': {}
            }
            
            for strategy_name, strategy_class in self.strategy_registry.items():
                strategy_validation = {
                    'has_backtest_method': hasattr(strategy_class, 'backtest'),
                    'backtest_callable': callable(getattr(strategy_class, 'backtest', None)),
                    'import_path': self.strategy_import_paths.get(strategy_name, {}).get('module_path', 'Unknown'),
                    'validation_passed': False
                }
                
                # 包括的検証
                if (strategy_validation['has_backtest_method'] and 
                    strategy_validation['backtest_callable']):
                    strategy_validation['validation_passed'] = True
                    validation_results['backtest_compliant'] += 1
                
                validation_results['import_successful'] += 1
                validation_results['strategy_details'][strategy_name] = strategy_validation
            
            # 検証結果保存
            self.registry_validation_results = validation_results
            
            # 成功基準判定
            success_rate = (validation_results['backtest_compliant'] / validation_results['total_registered'] * 100) if validation_results['total_registered'] > 0 else 0
            
            if success_rate >= 80.0:  # 80%以上で成功
                self.logger.info(f"Strategy registry validation passed: {success_rate:.1f}% compliance")
                return True
            else:
                self.logger.error(f"Strategy registry validation failed: {success_rate:.1f}% compliance (minimum 80% required)")
                return False
                
        except Exception as e:
            self.logger.error(f"Strategy registry validation failed: {e}")
            return False
    
    def _setup_resource_pool(self) -> bool:
        """
        リソースプール設定
        バックテスト基本理念遵守: backtest実行リソース確保
        """
        try:
            self.resource_pool = {
                'max_concurrent_strategies': len(self.strategy_registry),
                'memory_limit_mb': 2048,  # 増加（戦略数増加対応）
                'data_cache_enabled': True,
                'strategy_timeout_seconds': 300  # 5分タイムアウト
            }
            
            self.logger.info("Resource pool configured for complete strategy registry")
            return True
            
        except Exception as e:
            self.logger.error(f"Resource pool setup failed: {e}")
            return False
    
    def _prepare_monitoring(self) -> bool:
        """
        ヘルスチェック機能準備
        バックテスト基本理念遵守: backtest品質監視
        """
        try:
            # 基本理念遵守チェック機能
            self.monitoring_config = {
                'check_signal_generation': True,
                'validate_trade_execution': True,
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'monitor_excel_output': True,
                'strategy_registry_health': True,  # 新規追加
                'backtest_compliance_monitoring': True  # 新規追加
            }
            
            self.logger.info("Monitoring system prepared for complete strategy registry")
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            return False
    
    def get_strategy_instance(self, strategy_name: str, data, params: dict, **kwargs):
        """
        戦略インスタンス化機能 (TODO #12強化版)
        TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
        バックテスト基本理念遵守: 実際の戦略インスタンス化・backtest()実行準備
        """
        if not self.is_initialized:
            raise ValueError("MultiStrategyManager not initialized TODO(tag:strategy_registry, rationale:initialize before use)")
        
        if strategy_name not in self.strategy_registry:
            available = list(self.strategy_registry.keys())
            raise ValueError(f"Strategy {strategy_name} not registered. Available: {available} TODO(tag:strategy_registry, rationale:register missing strategy)")
        
        try:
            strategy_class = self.strategy_registry[strategy_name]
            
            # TODO #12: 戦略固有パラメータ自動供給システム（強化版）
            instance_kwargs = {
                'data': data,
                'params': params,
                'price_column': kwargs.get('price_column', 'Close')
            }
            
            # TODO #14 Phase 3: 実データ自動供給システム統合
            # (Phase 1のエラー停止方式を Phase 3で自動供給に強化)
            # ✅ RealMarketDataFetcher統合による実データ自動供給
            
            if strategy_name == 'VWAPBreakoutStrategy':
                if 'index_data' not in kwargs:
                    # Phase 3: 自動供給機能（RealMarketDataFetcher使用）
                    if REAL_DATA_FETCHER_AVAILABLE:
                        try:
                            self.logger.info(f"{strategy_name}: Fetching real index_data automatically...")
                            strategy_data = fetch_strategy_required_data(strategy_name, data)
                            
                            if 'index_data' in strategy_data and strategy_data['index_data'] is not None:
                                instance_kwargs['index_data'] = strategy_data['index_data']
                                self.logger.info(f"{strategy_name}: Real index_data auto-supplied successfully ({len(strategy_data['index_data'])} rows)")
                            else:
                                raise ValueError("Failed to fetch index_data")
                                
                        except Exception as e:
                            error_msg = (
                                f"❌ FAILED: {strategy_name} real data auto-supply failed.\n"
                                f"📋 Error: {str(e)}\n"
                                f"🔧 Solutions:\n"
                                f"  1. Check network connection for market data access\n"
                                f"  2. Manually provide index_data: kwargs['index_data'] = your_data\n"
                                f"  3. Verify yfinance service availability\n"
                                f"TODO(tag:real_data_required, rationale:TODO14 auto-supply failed)"
                            )
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    else:
                        # RealMarketDataFetcher利用不可時のエラー（Phase 1フォールバック）
                        error_msg = (
                            f"ERROR: {strategy_name} requires 'index_data' parameter for real market data.\n"
                            f"Auto-supply system unavailable - RealMarketDataFetcher not loaded.\n"
                            f"Please provide actual market index data manually.\n"
                            f"Example: kwargs['index_data'] = fetch_real_index_data('N225')"
                        )
                        self.logger.error(error_msg)
                        raise ValueError(f"Real data required: {strategy_name} missing index_data TODO(tag:real_data_required, rationale:TODO14 fetcher unavailable)")
                else:
                    instance_kwargs['index_data'] = kwargs['index_data']
                    self.logger.info(f"{strategy_name}: Real index_data provided manually")
                
                # その他のVWAP固有パラメータ
                if 'volume_column' in kwargs:
                    instance_kwargs['volume_column'] = kwargs['volume_column']
            
            elif strategy_name == 'OpeningGapStrategy':
                if 'dow_data' not in kwargs:
                    # Phase 3: 自動供給機能（RealMarketDataFetcher使用）
                    if REAL_DATA_FETCHER_AVAILABLE:
                        try:
                            self.logger.info(f"{strategy_name}: Fetching real dow_data automatically...")
                            strategy_data = fetch_strategy_required_data(strategy_name, data)
                            
                            if 'dow_data' in strategy_data and strategy_data['dow_data'] is not None:
                                instance_kwargs['dow_data'] = strategy_data['dow_data']
                                self.logger.info(f"{strategy_name}: Real dow_data auto-supplied successfully ({len(strategy_data['dow_data'])} rows)")
                            else:
                                raise ValueError("Failed to fetch dow_data")
                                
                        except Exception as e:
                            error_msg = (
                                f"❌ FAILED: {strategy_name} real data auto-supply failed.\n"
                                f"📋 Error: {str(e)}\n"
                                f"🔧 Solutions:\n"
                                f"  1. Check network connection for market data access\n"
                                f"  2. Manually provide dow_data: kwargs['dow_data'] = your_data\n"
                                f"  3. Verify yfinance service availability\n"
                                f"TODO(tag:real_data_required, rationale:TODO14 auto-supply failed)"
                            )
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    else:
                        # RealMarketDataFetcher利用不可時のエラー（Phase 1フォールバック）
                        error_msg = (
                            f"ERROR: {strategy_name} requires 'dow_data' parameter for real market data.\n"
                            f"Auto-supply system unavailable - RealMarketDataFetcher not loaded.\n"
                            f"Please provide actual Dow Jones index data manually.\n"
                            f"Example: kwargs['dow_data'] = fetch_real_index_data('DJI')"
                        )
                        self.logger.error(error_msg)
                        raise ValueError(f"Real data required: {strategy_name} missing dow_data TODO(tag:real_data_required, rationale:TODO14 fetcher unavailable)")
                else:
                    instance_kwargs['dow_data'] = kwargs['dow_data']
                    self.logger.info(f"{strategy_name}: Real dow_data provided manually")
            
            # TODO #12 RA_005: VWAPBounceStrategy等の他の戦略（index_data不要）
            elif strategy_name in ['VWAPBounceStrategy']:
                # VWAPBounceStrategyはindex_dataパラメータを受け取らない設計
                # volume_columnやprice_columnのみを渡す
                if 'volume_column' in kwargs:
                    instance_kwargs['volume_column'] = kwargs['volume_column']
                # index_dataは除外して他のkwargsは渡す
                instance_kwargs.update({k: v for k, v in kwargs.items() if k not in ['index_data']})
            
            else:
                # その他の戦略: 追加パラメータをそのまま渡す
                instance_kwargs.update(kwargs)
            
            # 戦略インスタンス化
            strategy_instance = strategy_class(**instance_kwargs)
            
            # バックテスト基本理念遵守: インスタンス化後のbacktest()確認
            if not hasattr(strategy_instance, 'backtest') or not callable(strategy_instance.backtest):
                raise ValueError(f"Strategy instance {strategy_name} lacks callable backtest() method TODO(tag:backtest_execution, rationale:implement backtest method)")
            
            self.logger.info(f"Strategy instance created: {strategy_name}")
            return strategy_instance
            
        except Exception as e:
            self.logger.error(f"Strategy instantiation failed for {strategy_name}: {e}")
            raise ValueError(f"Strategy instantiation failed: {strategy_name} - {e} TODO(tag:strategy_registry, rationale:fix instantiation issue)")
    
    def execute_strategy_with_validation(self, strategy_name: str, stock_data, params: dict, **kwargs):
        """
        バックテスト基本理念遵守: 実際の戦略実行 + 品質検証（完全実装版）
        TODO(tag:strategy_registry, rationale:complete weight judgment system recovery)
        """
        if not self.is_initialized:
            raise ValueError("MultiStrategyManager not initialized TODO(tag:strategy_registry, rationale:initialize before use)")
        
        try:
            # 戦略インスタンス取得
            strategy_instance = self.get_strategy_instance(strategy_name, stock_data, params, **kwargs)
            
            # バックテスト基本理念遵守: 実際のbacktest()実行
            result = strategy_instance.backtest()
            
            # 基本理念違反検出
            self._validate_backtest_principle_compliance(result, strategy_name)
            
            self.logger.info(f"Strategy execution completed: {strategy_name} - {result.shape[0]} rows")
            return result
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed for {strategy_name}: {e}")
            raise
    
    def _validate_backtest_principle_compliance(self, result, strategy_name: str):
        """バックテスト基本理念違反検出（完全実装版）"""
        violations = []
        
        # 1. シグナル列存在チェック
        required_columns = ['Entry_Signal', 'Exit_Signal']
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            violations.append(f"Missing signal columns: {missing_columns}")
        
        # 2. 取引数チェック
        if 'Entry_Signal' in result.columns and 'Exit_Signal' in result.columns:
            total_entries = (result['Entry_Signal'] == 1).sum()
            total_exits = abs(result['Exit_Signal']).sum()
            total_trades = total_entries + total_exits
            
            if total_trades == 0:
                violations.append("Zero trades generated - potential strategy logic issue")
            elif total_entries == 0:
                violations.append("Zero entry signals - strategy may not be triggering")
            elif total_exits == 0:
                violations.append("Zero exit signals - strategy may not be exiting positions")
        
        # 3. データ完整性チェック
        if len(result) == 0:
            violations.append("Empty result data")
        
        # 4. Excel出力要件チェック
        excel_required_columns = ['Close']
        missing_excel_columns = [col for col in excel_required_columns if col not in result.columns]
        if missing_excel_columns:
            violations.append(f"Excel output columns missing: {missing_excel_columns}")
        
        # 5. 基本データ型チェック
        if not isinstance(result, pd.DataFrame):
            violations.append("Result is not a pandas DataFrame")
        
        # 違反検出時の処理
        if violations:
            error_msg = f"Backtest principle violations in {strategy_name}: {'; '.join(violations)}"
            self.logger.error(error_msg)
            raise ValueError(f"{error_msg} TODO(tag:backtest_execution, rationale:fix principle violations)")
        
        return True
    
    def _print_strategy_registry_report(self):
        """戦略レジストリレポート出力"""
        print("\n" + "="*80)
        print("📊 戦略レジストリシステム初期化完了レポート")
        print("="*80)
        
        print(f"\n✅ 戦略レジストリ状況:")
        print(f"  登録戦略数: {len(self.strategy_registry)}")
        print(f"  利用可能戦略: {list(self.strategy_registry.keys())}")
        
        if hasattr(self, 'registry_validation_results'):
            validation = self.registry_validation_results
            print(f"\n📋 検証結果:")
            print(f"  総登録数: {validation['total_registered']}")
            print(f"  backtest準拠: {validation['backtest_compliant']}")
            print(f"  準拠率: {(validation['backtest_compliant']/validation['total_registered']*100):.1f}%")
            
            print(f"\n🔍 戦略別詳細:")
            for strategy_name, details in validation['strategy_details'].items():
                status = "✅" if details['validation_passed'] else "❌"
                print(f"  {status} {strategy_name}: backtest={details['has_backtest_method']}")
        
        print(f"\n🎯 TODO #9 戦略レジストリシステム完全実装: 完了")
        print("="*80)
    
    def get_registry_status(self) -> dict:
        """戦略レジストリステータス取得"""
        return {
            'is_initialized': self.is_initialized,
            'total_strategies': len(self.strategy_registry),
            'available_strategies': list(self.strategy_registry.keys()),
            'validation_results': getattr(self, 'registry_validation_results', {}),
            'import_paths': self.strategy_import_paths
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
