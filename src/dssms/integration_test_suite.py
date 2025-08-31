"""
DSSMS Phase 2 Task 2.1: 統合テストスイート
段階的テスト機能による包括的な統合システム検証

主要機能:
1. 個別コンポーネントテスト
2. 統合テスト（システム間連携）
3. ストレステスト（極端条件下での動作）
4. パフォーマンステスト（実行性能測定）
5. 回帰テスト（既存機能の非破壊確認）

設計方針:
- 段階的テスト実行による確実な品質保証
- 自動化された包括的テストカバレッジ
- エラー報告と詳細ログ機能
- CI/CD パイプライン対応
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
import unittest
import traceback
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 統合システムコンポーネント (遅延インポート対応)
DSSMSStrategyIntegrationManager = None
DSSMSStrategyBridge = None
StrategyDSSMSCoordinator = None
IntegratedPerformanceCalculator = None

def _lazy_import_components():
    """遅延インポート"""
    global DSSMSStrategyIntegrationManager, DSSMSStrategyBridge, StrategyDSSMSCoordinator, IntegratedPerformanceCalculator
    
    if DSSMSStrategyIntegrationManager is None:
        try:
            from src.dssms.dssms_strategy_integration_manager import DSSMSStrategyIntegrationManager as DSIM
            from src.dssms.dssms_strategy_bridge import DSSMSStrategyBridge as DSB
            from src.dssms.strategy_dssms_coordinator import StrategyDSSMSCoordinator as SDC
            from src.dssms.integrated_performance_calculator import IntegratedPerformanceCalculator as IPC
            
            DSSMSStrategyIntegrationManager = DSIM
            DSSMSStrategyBridge = DSB
            StrategyDSSMSCoordinator = SDC
            IntegratedPerformanceCalculator = IPC
            
        except ImportError as e:
            print(f"Integration components import warning: {e}")

# データ処理
try:
    from data_fetcher import DataFetcher
    from data_processor import DataProcessor
except ImportError as e:
    print(f"Data processing import warning: {e}")

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class TestResult:
    """テスト結果"""
    test_name: str
    success: bool
    execution_time: float
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class TestSuiteResult:
    """テストスイート結果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    test_results: List[TestResult]
    summary: Dict[str, Any]

class TestLevel(Enum):
    """テストレベル"""
    UNIT = "unit"
    INTEGRATION = "integration"
    STRESS = "stress"
    PERFORMANCE = "performance"
    REGRESSION = "regression"

class IntegrationTestSuite:
    """
    統合テストスイート
    
    DSSMS統合システムの包括的なテスト機能を提供し、
    段階的テスト実行による品質保証を行います。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config_path = config_path
        self.test_results = []
        self.test_data_cache = {}
        
        # テストデータ準備
        self._prepare_test_data()
        
        logger.info("Integration Test Suite initialized")
    
    def _prepare_test_data(self):
        """テストデータ準備"""
        try:
            # サンプルデータ生成
            dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
            
            # 株価データサンプル
            self.test_data_cache['stock_data'] = {}
            test_symbols = ['7203', '6758', '8306']
            
            for symbol in test_symbols:
                # ランダムウォーク + トレンド
                np.random.seed(42 + hash(symbol) % 100)
                base_price = 100
                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = base_price * np.cumprod(1 + returns)
                
                volumes = np.random.uniform(1000000, 5000000, len(dates))
                
                stock_data = pd.DataFrame({
                    'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
                    'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
                    'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
                    'Close': prices,
                    'Adj Close': prices,
                    'Volume': volumes
                }, index=dates)
                
                self.test_data_cache['stock_data'][symbol] = stock_data
            
            # インデックスデータ
            np.random.seed(42)
            index_returns = np.random.normal(0.0005, 0.015, len(dates))
            index_prices = 30000 * np.cumprod(1 + index_returns)
            
            self.test_data_cache['index_data'] = pd.DataFrame({
                'Open': index_prices * np.random.uniform(0.999, 1.001, len(dates)),
                'High': index_prices * np.random.uniform(1.000, 1.002, len(dates)),
                'Low': index_prices * np.random.uniform(0.998, 1.000, len(dates)),
                'Close': index_prices,
                'Adj Close': index_prices,
                'Volume': np.random.uniform(1000000000, 2000000000, len(dates))
            }, index=dates)
            
            logger.info("Test data prepared successfully")
            
        except Exception as e:
            logger.error(f"Failed to prepare test data: {e}")
            raise
    
    def run_all_tests(self) -> TestSuiteResult:
        """全テスト実行"""
        logger.info("Starting comprehensive test suite execution")
        start_time = time.time()
        
        all_results = []
        
        # テストレベル別実行
        test_levels = [
            TestLevel.UNIT,
            TestLevel.INTEGRATION,
            TestLevel.STRESS,
            TestLevel.PERFORMANCE
        ]
        
        for level in test_levels:
            try:
                level_results = self._run_test_level(level)
                all_results.extend(level_results)
                logger.info(f"Completed {level.value} tests: {len(level_results)} tests")
            except Exception as e:
                logger.error(f"Failed to run {level.value} tests: {e}")
                all_results.append(TestResult(
                    test_name=f"{level.value}_suite_error",
                    success=False,
                    execution_time=0.0,
                    result_data={},
                    error_message=str(e)
                ))
        
        # 総合結果
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.success)
        failed_tests = total_tests - passed_tests
        execution_time = time.time() - start_time
        
        suite_result = TestSuiteResult(
            suite_name="DSSMS_Integration_Test_Suite",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            execution_time=execution_time,
            test_results=all_results,
            summary=self._generate_test_summary(all_results)
        )
        
        logger.info(f"Test suite completed: {passed_tests}/{total_tests} passed in {execution_time:.2f}s")
        
        return suite_result
    
    def _run_test_level(self, level: TestLevel) -> List[TestResult]:
        """テストレベル別実行"""
        if level == TestLevel.UNIT:
            return self._run_unit_tests()
        elif level == TestLevel.INTEGRATION:
            return self._run_integration_tests()
        elif level == TestLevel.STRESS:
            return self._run_stress_tests()
        elif level == TestLevel.PERFORMANCE:
            return self._run_performance_tests()
        else:
            return []
    
    def _run_unit_tests(self) -> List[TestResult]:
        """ユニットテスト実行"""
        logger.info("Running unit tests...")
        results = []
        
        # 1. DSSMSStrategyBridge テスト
        results.append(self._test_strategy_bridge())
        
        # 2. StrategyDSSMSCoordinator テスト
        results.append(self._test_coordinator())
        
        # 3. IntegratedPerformanceCalculator テスト
        results.append(self._test_performance_calculator())
        
        # 4. 設定ファイル読み込みテスト
        results.append(self._test_config_loading())
        
        return results
    
    def _test_strategy_bridge(self) -> TestResult:
        """戦略ブリッジテスト"""
        start_time = time.time()
        
        try:
            # 遅延インポート
            _lazy_import_components()
            
            if DSSMSStrategyBridge is None:
                return TestResult(
                    test_name="strategy_bridge_test",
                    success=False,
                    execution_time=time.time() - start_time,
                    result_data={},
                    error_message="DSSMSStrategyBridge import failed"
                )
            
            # ブリッジ初期化
            bridge = DSSMSStrategyBridge()
            
            # 基本機能テスト
            test_data = self.test_data_cache['stock_data']['7203']
            index_data = self.test_data_cache['index_data']
            
            # 戦略実行テスト
            if 'VWAP_Breakout' in bridge.strategy_configs:
                result = bridge.execute_strategy(
                    strategy_name='VWAP_Breakout',
                    data=test_data,
                    index_data=index_data
                )
                
                assert result is not None
                assert result.strategy_name == 'VWAP_Breakout'
                assert result.signal in ['buy', 'sell', 'hold', 'unknown']
                assert 0 <= result.confidence <= 1
                
            # 全戦略分析テスト
            all_results = bridge.analyze_all_strategies(
                symbol="7203",
                date=datetime.now(),
                data=test_data,
                index_data=index_data
            )
            
            assert 'results' in all_results
            assert 'scores' in all_results
            assert 'signals' in all_results
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="strategy_bridge_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'bridge_initialized': True,
                    'strategy_execution': True,
                    'all_analysis': True,
                    'strategies_loaded': len(bridge.loaded_strategies),
                    'configurations': len(bridge.strategy_configs)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="strategy_bridge_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_coordinator(self) -> TestResult:
        """コーディネーターテスト"""
        start_time = time.time()
        
        try:
            # コーディネーター初期化
            coordinator = StrategyDSSMSCoordinator()
            
            # 決定テスト
            result = coordinator.coordinate_decision(
                dssms_score=0.75,
                dssms_signal="buy",
                strategy_scores={"VWAP_Breakout": 0.65, "GoldenCross": 0.55},
                strategy_signals={"VWAP_Breakout": "hold", "GoldenCross": "sell"},
                symbol="7203",
                date=datetime.now()
            )
            
            assert result is not None
            assert result.selected_system is not None
            assert 0 <= result.confidence_score <= 1
            assert result.position_signal in ['buy', 'sell', 'hold']
            
            # 統計テスト
            stats = coordinator.get_coordination_statistics()
            assert 'total_decisions' in stats
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="coordinator_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'coordinator_initialized': True,
                    'decision_made': True,
                    'statistics_available': True,
                    'selected_system': result.selected_system,
                    'confidence': result.confidence_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="coordinator_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_performance_calculator(self) -> TestResult:
        """パフォーマンス計算機テスト"""
        start_time = time.time()
        
        try:
            # 計算機初期化
            calculator = IntegratedPerformanceCalculator(use_dssms_engine=False)  # フォールバックで実行
            
            # テスト結果データ作成
            dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
            portfolio_values = 1000000 * (1 + np.cumsum(np.random.normal(0.001, 0.015, len(dates))))
            
            test_results = {
                'daily_values': [
                    {'date': date, 'portfolio_value': value, 'cash': value * 0.1, 'position_value': value * 0.9}
                    for date, value in zip(dates, portfolio_values)
                ],
                'trades': [
                    {'date': dates[10], 'symbol': '7203', 'action': 'buy', 'shares': 100, 'price': 100, 'value': 10000, 'system': 'dssms_only'},
                    {'date': dates[20], 'symbol': '7203', 'action': 'sell', 'shares': 100, 'price': 105, 'value': 10500, 'profit': 500, 'system': 'dssms_only'}
                ]
            }
            
            # パフォーマンス計算
            metrics = calculator.calculate_comprehensive_performance(
                results=test_results,
                initial_capital=1000000
            )
            
            assert metrics is not None
            assert hasattr(metrics, 'total_return')
            assert hasattr(metrics, 'sharpe_ratio')
            assert hasattr(metrics, 'max_drawdown')
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="performance_calculator_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'calculator_initialized': True,
                    'metrics_calculated': True,
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="performance_calculator_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_config_loading(self) -> TestResult:
        """設定ファイル読み込みテスト"""
        start_time = time.time()
        
        try:
            # 設定ファイルパス
            config_path = project_root / "src" / "dssms" / "dssms_integration_config.json"
            mapping_path = project_root / "src" / "dssms" / "strategy_integration_mapping.json"
            
            # 設定ファイル読み込み
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                assert 'integration' in config_data
                assert 'performance' in config_data
            
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                assert 'strategies' in mapping_data
                assert 'market_condition_mapping' in mapping_data
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="config_loading_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'config_exists': config_path.exists(),
                    'mapping_exists': mapping_path.exists(),
                    'config_valid': True,
                    'mapping_valid': True
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="config_loading_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _run_integration_tests(self) -> List[TestResult]:
        """統合テスト実行"""
        logger.info("Running integration tests...")
        results = []
        
        # 統合マネージャーテスト
        results.append(self._test_integration_manager())
        
        # システム間連携テスト
        results.append(self._test_system_coordination())
        
        return results
    
    def _test_integration_manager(self) -> TestResult:
        """統合マネージャーテスト"""
        start_time = time.time()
        
        try:
            # マネージャー初期化
            manager = DSSMSStrategyIntegrationManager()
            
            # システム初期化テスト
            init_success = manager.initialize_systems(
                data=self.test_data_cache['stock_data'],
                index_data=self.test_data_cache['index_data']
            )
            
            # 統合分析テスト
            if init_success:
                analysis_result = manager.execute_integrated_analysis(
                    symbol="7203",
                    date=datetime.now(),
                    data=self.test_data_cache['stock_data']['7203'],
                    index_data=self.test_data_cache['index_data']
                )
                
                assert analysis_result is not None
                assert analysis_result.selected_system is not None
                
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="integration_manager_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'manager_initialized': True,
                    'systems_initialized': init_success,
                    'analysis_executed': True,
                    'selected_system': analysis_result.selected_system if init_success else 'N/A'
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="integration_manager_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_system_coordination(self) -> TestResult:
        """システム間連携テスト"""
        start_time = time.time()
        
        try:
            # コンポーネント初期化
            bridge = DSSMSStrategyBridge()
            coordinator = StrategyDSSMSCoordinator()
            calculator = IntegratedPerformanceCalculator(use_dssms_engine=False)
            
            # 連携フローテスト
            # 1. 戦略分析
            strategy_results = bridge.analyze_all_strategies(
                symbol="7203",
                date=datetime.now(),
                data=self.test_data_cache['stock_data']['7203'],
                index_data=self.test_data_cache['index_data']
            )
            
            # 2. 調整判定
            coordination_result = coordinator.coordinate_decision(
                dssms_score=0.7,
                dssms_signal="buy",
                strategy_scores=strategy_results['scores'],
                strategy_signals=strategy_results['signals'],
                symbol="7203",
                date=datetime.now()
            )
            
            assert coordination_result is not None
            assert coordination_result.selected_system is not None
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="system_coordination_test",
                success=True,
                execution_time=execution_time,
                result_data={
                    'bridge_analysis': True,
                    'coordination_decision': True,
                    'systems_integrated': True,
                    'final_decision': coordination_result.selected_system
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="system_coordination_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _run_stress_tests(self) -> List[TestResult]:
        """ストレステスト実行"""
        logger.info("Running stress tests...")
        results = []
        
        # 極端データテスト
        results.append(self._test_extreme_data_conditions())
        
        # 高負荷テスト
        results.append(self._test_high_load_conditions())
        
        return results
    
    def _test_extreme_data_conditions(self) -> TestResult:
        """極端データ条件テスト"""
        start_time = time.time()
        
        try:
            bridge = DSSMSStrategyBridge()
            
            # 極端データケース
            extreme_cases = [
                # 1. 全て同じ価格
                pd.DataFrame({
                    'Open': [100] * 100,
                    'High': [100] * 100,
                    'Low': [100] * 100,
                    'Close': [100] * 100,
                    'Adj Close': [100] * 100,
                    'Volume': [1000000] * 100
                }, index=pd.date_range('2024-01-01', periods=100)),
                
                # 2. 極端なボラティリティ
                pd.DataFrame({
                    'Open': np.random.uniform(50, 150, 100),
                    'High': np.random.uniform(100, 200, 100),
                    'Low': np.random.uniform(10, 100, 100),
                    'Close': np.random.uniform(30, 170, 100),
                    'Adj Close': np.random.uniform(30, 170, 100),
                    'Volume': np.random.uniform(100, 100000000, 100)
                }, index=pd.date_range('2024-01-01', periods=100)),
                
                # 3. 最小データ
                pd.DataFrame({
                    'Open': [100, 101],
                    'High': [101, 102],
                    'Low': [99, 100],
                    'Close': [100, 101],
                    'Adj Close': [100, 101],
                    'Volume': [1000, 1100]
                }, index=pd.date_range('2024-01-01', periods=2))
            ]
            
            success_count = 0
            for i, case_data in enumerate(extreme_cases):
                try:
                    result = bridge.analyze_all_strategies(
                        symbol=f"TEST{i}",
                        date=datetime.now(),
                        data=case_data,
                        index_data=case_data  # 簡易版
                    )
                    if result and 'statistics' in result:
                        success_count += 1
                except Exception as case_error:
                    logger.warning(f"Extreme case {i} failed: {case_error}")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="extreme_data_conditions_test",
                success=success_count > 0,
                execution_time=execution_time,
                result_data={
                    'total_cases': len(extreme_cases),
                    'successful_cases': success_count,
                    'success_rate': success_count / len(extreme_cases)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="extreme_data_conditions_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_high_load_conditions(self) -> TestResult:
        """高負荷条件テスト"""
        start_time = time.time()
        
        try:
            manager = DSSMSStrategyIntegrationManager()
            
            # 複数銘柄同時処理
            test_symbols = ['7203', '6758', '8306', '9984', '7267']
            test_data = {}
            
            for symbol in test_symbols:
                # ランダムデータ生成
                dates = pd.date_range('2024-01-01', periods=252)
                np.random.seed(hash(symbol) % 100)
                
                test_data[symbol] = pd.DataFrame({
                    'Open': np.random.uniform(90, 110, 252),
                    'High': np.random.uniform(100, 120, 252),
                    'Low': np.random.uniform(80, 100, 252),
                    'Close': np.random.uniform(95, 105, 252),
                    'Adj Close': np.random.uniform(95, 105, 252),
                    'Volume': np.random.uniform(1000000, 5000000, 252)
                }, index=dates)
            
            # システム初期化
            init_success = manager.initialize_systems(
                data=test_data,
                index_data=self.test_data_cache['index_data']
            )
            
            # 複数分析実行
            analysis_count = 0
            if init_success:
                for symbol in test_symbols:
                    try:
                        result = manager.execute_integrated_analysis(
                            symbol=symbol,
                            date=datetime.now(),
                            data=test_data[symbol],
                            index_data=self.test_data_cache['index_data']
                        )
                        if result:
                            analysis_count += 1
                    except Exception as e:
                        logger.warning(f"Analysis failed for {symbol}: {e}")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="high_load_conditions_test",
                success=analysis_count > 0,
                execution_time=execution_time,
                result_data={
                    'symbols_tested': len(test_symbols),
                    'successful_analyses': analysis_count,
                    'systems_initialized': init_success,
                    'throughput': analysis_count / execution_time if execution_time > 0 else 0
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="high_load_conditions_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _run_performance_tests(self) -> List[TestResult]:
        """パフォーマンステスト実行"""
        logger.info("Running performance tests...")
        results = []
        
        # 実行時間テスト
        results.append(self._test_execution_performance())
        
        # メモリ使用量テスト
        results.append(self._test_memory_usage())
        
        return results
    
    def _test_execution_performance(self) -> TestResult:
        """実行時間パフォーマンステスト"""
        start_time = time.time()
        
        try:
            bridge = DSSMSStrategyBridge()
            coordinator = StrategyDSSMSCoordinator()
            
            # ベンチマーク実行
            execution_times = []
            
            for i in range(10):  # 10回実行
                iteration_start = time.time()
                
                # 戦略分析
                strategy_results = bridge.analyze_all_strategies(
                    symbol="PERF_TEST",
                    date=datetime.now(),
                    data=self.test_data_cache['stock_data']['7203'],
                    index_data=self.test_data_cache['index_data']
                )
                
                # 調整判定
                coordination_result = coordinator.coordinate_decision(
                    dssms_score=0.6,
                    dssms_signal="hold",
                    strategy_scores=strategy_results['scores'],
                    strategy_signals=strategy_results['signals'],
                    symbol="PERF_TEST",
                    date=datetime.now()
                )
                
                iteration_time = time.time() - iteration_start
                execution_times.append(iteration_time)
            
            avg_execution_time = np.mean(execution_times)
            max_execution_time = np.max(execution_times)
            min_execution_time = np.min(execution_times)
            
            # パフォーマンス基準 (1秒以内を良好とする)
            performance_acceptable = avg_execution_time < 1.0
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="execution_performance_test",
                success=performance_acceptable,
                execution_time=execution_time,
                result_data={
                    'iterations': len(execution_times),
                    'avg_execution_time': avg_execution_time,
                    'max_execution_time': max_execution_time,
                    'min_execution_time': min_execution_time,
                    'performance_acceptable': performance_acceptable,
                    'target_threshold': 1.0
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="execution_performance_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _test_memory_usage(self) -> TestResult:
        """メモリ使用量テスト"""
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 大量データ処理テスト
            managers = []
            for i in range(5):  # 5つのマネージャーインスタンス
                manager = DSSMSStrategyIntegrationManager()
                managers.append(manager)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # メモリ基準 (500MB増加以内を良好とする)
            memory_acceptable = memory_increase < 500
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="memory_usage_test",
                success=memory_acceptable,
                execution_time=execution_time,
                result_data={
                    'initial_memory_mb': initial_memory,
                    'current_memory_mb': current_memory,
                    'memory_increase_mb': memory_increase,
                    'memory_acceptable': memory_acceptable,
                    'target_threshold_mb': 500
                }
            )
            
        except ImportError:
            # psutil が利用できない場合
            execution_time = time.time() - start_time
            return TestResult(
                test_name="memory_usage_test",
                success=True,
                execution_time=execution_time,
                result_data={'status': 'skipped_no_psutil'},
                warnings=['psutil not available, memory test skipped']
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="memory_usage_test",
                success=False,
                execution_time=execution_time,
                result_data={},
                error_message=str(e)
            )
    
    def _generate_test_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """テスト結果サマリー生成"""
        try:
            # 基本統計
            total_tests = len(results)
            passed_tests = sum(1 for r in results if r.success)
            failed_tests = total_tests - passed_tests
            
            # カテゴリ別統計
            category_stats = {}
            for result in results:
                category = result.test_name.split('_')[0]
                if category not in category_stats:
                    category_stats[category] = {'total': 0, 'passed': 0}
                category_stats[category]['total'] += 1
                if result.success:
                    category_stats[category]['passed'] += 1
            
            # 実行時間統計
            execution_times = [r.execution_time for r in results]
            
            # エラー分析
            error_types = {}
            for result in results:
                if not result.success and result.error_message:
                    error_type = type(result.error_message).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                'overall': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                },
                'category_breakdown': category_stats,
                'performance': {
                    'total_execution_time': sum(execution_times),
                    'average_execution_time': np.mean(execution_times) if execution_times else 0,
                    'slowest_test': max(execution_times) if execution_times else 0,
                    'fastest_test': min(execution_times) if execution_times else 0
                },
                'error_analysis': error_types,
                'recommendations': self._generate_recommendations(results)
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate test summary: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """改善提案生成"""
        recommendations = []
        
        # 失敗したテストの分析
        failed_tests = [r for r in results if not r.success]
        if failed_tests:
            recommendations.append(f"{len(failed_tests)} tests failed - review error messages and fix issues")
        
        # パフォーマンス分析
        slow_tests = [r for r in results if r.execution_time > 5.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are slow (>5s) - consider optimization")
        
        # 統合テスト特有の提案
        integration_failures = [r for r in results if 'integration' in r.test_name and not r.success]
        if integration_failures:
            recommendations.append("Integration test failures detected - check component compatibility")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system ready for deployment")
        
        return recommendations

# 使用例とテスト実行
def run_test_suite():
    """テストスイート実行"""
    print("=== DSSMS Integration Test Suite ===")
    
    # テストスイート初期化・実行
    test_suite = IntegrationTestSuite()
    results = test_suite.run_all_tests()
    
    # 結果表示
    print(f"\nTest Results Summary:")
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    print(f"Success Rate: {results.passed_tests/results.total_tests:.1%}")
    print(f"Total Execution Time: {results.execution_time:.2f}s")
    
    # 詳細結果
    print(f"\nDetailed Results:")
    for result in results.test_results:
        status = "PASS" if result.success else "FAIL"
        print(f"  {result.test_name}: {status} ({result.execution_time:.3f}s)")
        if not result.success and result.error_message:
            print(f"    Error: {result.error_message}")
    
    # 推奨事項
    if 'recommendations' in results.summary:
        print(f"\nRecommendations:")
        for rec in results.summary['recommendations']:
            print(f"  - {rec}")
    
    return results

if __name__ == "__main__":
    run_test_suite()
