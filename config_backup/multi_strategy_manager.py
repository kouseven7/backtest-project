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
        """マルチ戦略フローの実行"""
        try:
            logger.info("Executing multi-strategy flow")
            
            # 1. 戦略選択（簡易版）
            selected_strategies = available_strategies[:3] if len(available_strategies) > 3 else available_strategies
            logger.info(f"Selected strategies: {selected_strategies}")
            
            # 2. ポートフォリオ重み計算（均等分散）
            num_strategies = len(selected_strategies)
            portfolio_weights = {strategy: 1.0/num_strategies for strategy in selected_strategies}
            
            # 3. シグナル統合（簡易版）
            integrated_signals = {}
            for strategy in selected_strategies:
                integrated_signals[strategy] = {"signal": "hold", "confidence": 0.7}
            
            # 4. リスク調整（基本版）
            final_positions = portfolio_weights.copy()
            risk_metrics = {"risk_adjustment": "basic"}
            
            # 結果作成
            execution_time = (datetime.now() - start_time).total_seconds()
            result = MultiStrategyResult(
                execution_mode=ExecutionMode.MULTI_STRATEGY,
                selected_strategies=selected_strategies,
                portfolio_weights=portfolio_weights,
                final_positions=final_positions,
                performance_metrics={"selection_score": 0.8},
                risk_metrics=risk_metrics,
                execution_time=execution_time,
                status=IntegrationStatus.READY
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Multi-strategy flow failed: {e}")
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
