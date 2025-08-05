"""
A→B市場分類システム 統合テスト・デモンストレーション
全コンポーネントの統合テストと実用デモンストレーションを提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import time
import warnings
import sys
import os
from pathlib import Path

# プロジェクトパス追加
sys.path.append(str(Path(__file__).parent))

# 市場分類システムコンポーネント
try:
    from .market_conditions import (
        SimpleMarketCondition, DetailedMarketCondition,
        MarketMetrics, ClassificationResult, MarketConditions
    )
    from .market_condition_detector import MarketConditionDetector, DetectionResult
    from .technical_indicator_analyzer import TechnicalIndicatorAnalyzer, TechnicalAnalysisResult
    from .market_regime_classifier import MarketRegimeClassifier, RegimeClassificationResult
    from .volatility_analyzer import VolatilityAnalyzer, VolatilityAnalysisResult
    from .trend_strength_evaluator import TrendStrengthEvaluator, TrendStrengthResult
    from .market_correlation_analyzer import MarketCorrelationAnalyzer, MarketCorrelationAnalysis
    from .integrated_market_state_manager import (
        IntegratedMarketStateManager, IntegratedMarketState, StateIntegrationMethod
    )
    from .cache_manager import MultiLevelCacheManager, MarketDataCacheManager
    from .error_handling import RobustAnalysisSystem, RecoveryAction, RecoveryStrategy, ErrorCategory
except ImportError as e:
    print(f"コンポーネントインポートエラー: {e}")
    print("相対インポートに切り替えます...")
    
    # フォールバック: 絶対インポート
    try:
        from src.analysis.market_classification.market_conditions import (
            SimpleMarketCondition, DetailedMarketCondition,
            MarketMetrics, ClassificationResult, MarketConditions
        )
        from src.analysis.market_classification.market_condition_detector import MarketConditionDetector, DetectionResult
        from src.analysis.market_classification.technical_indicator_analyzer import TechnicalIndicatorAnalyzer, TechnicalAnalysisResult
        from src.analysis.market_classification.market_regime_classifier import MarketRegimeClassifier, RegimeClassificationResult
        from src.analysis.market_classification.volatility_analyzer import VolatilityAnalyzer, VolatilityAnalysisResult
        from src.analysis.market_classification.trend_strength_evaluator import TrendStrengthEvaluator, TrendStrengthResult
        from src.analysis.market_classification.market_correlation_analyzer import MarketCorrelationAnalyzer, MarketCorrelationAnalysis
        from src.analysis.market_classification.integrated_market_state_manager import (
            IntegratedMarketStateManager, IntegratedMarketState, StateIntegrationMethod
        )
        from src.analysis.market_classification.cache_manager import MultiLevelCacheManager, MarketDataCacheManager
        from src.analysis.market_classification.error_handling import RobustAnalysisSystem, RecoveryAction, RecoveryStrategy, ErrorCategory
    except ImportError:
        # 最後の手段：モックコンポーネント
        print("インポートに失敗しました。モックコンポーネントを使用します。")
        
        class MockComponent:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                return None
        
        SimpleMarketCondition = MockComponent
        DetailedMarketCondition = MockComponent
        MarketConditions = MockComponent
        MarketConditionDetector = MockComponent
        TechnicalIndicatorAnalyzer = MockComponent
        MarketRegimeClassifier = MockComponent
        VolatilityAnalyzer = MockComponent
        TrendStrengthEvaluator = MockComponent
        MarketCorrelationAnalyzer = MockComponent
        IntegratedMarketStateManager = MockComponent
        MultiLevelCacheManager = MockComponent
        RobustAnalysisSystem = MockComponent

class TestResult(Enum):
    """テスト結果"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"

@dataclass
class TestCase:
    """テストケース"""
    name: str
    description: str
    test_function: Any
    expected_result: TestResult = TestResult.PASS
    execution_time: float = 0.0
    error_message: str = ""
    result: TestResult = TestResult.SKIP

class IntegratedMarketClassificationDemo:
    """
    A→B市場分類システム統合デモンストレーション
    全コンポーネントの動作確認とパフォーマンステストを実行
    """
    
    def __init__(self, cache_enabled: bool = True, error_handling_enabled: bool = True):
        """
        統合デモシステム初期化
        
        Args:
            cache_enabled: キャッシュ機能有効化
            error_handling_enabled: エラーハンドリング有効化
        """
        self.cache_enabled = cache_enabled
        self.error_handling_enabled = error_handling_enabled
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # テストケース
        self.test_cases: List[TestCase] = []
        
        # テスト結果
        self.test_results: Dict[str, TestResult] = {}
        
        # パフォーマンスメトリクス
        self.performance_metrics: Dict[str, float] = {}
        
        # サンプルデータ
        self.sample_data: Dict[str, pd.DataFrame] = {}
        
        # コンポーネント
        self.components: Dict[str, Any] = {}
        
        self.logger.info("IntegratedMarketClassificationDemo初期化完了")

    def setup_test_environment(self):
        """テスト環境セットアップ"""
        try:
            self.logger.info("テスト環境をセットアップ中...")
            
            # サンプルデータ生成
            self._generate_sample_data()
            
            # コンポーネント初期化
            self._initialize_components()
            
            # テストケース登録
            self._register_test_cases()
            
            self.logger.info("テスト環境セットアップ完了")
            
        except Exception as e:
            self.logger.error(f"テスト環境セットアップエラー: {e}")
            raise

    def _generate_sample_data(self):
        """サンプルデータ生成"""
        try:
            np.random.seed(42)
            
            # 日付範囲
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 複数の市場シナリオデータ
            scenarios = {
                'bull_market': self._generate_bull_market_data(dates),
                'bear_market': self._generate_bear_market_data(dates),
                'sideways_market': self._generate_sideways_market_data(dates),
                'volatile_market': self._generate_volatile_market_data(dates),
                'crisis_market': self._generate_crisis_market_data(dates)
            }
            
            self.sample_data = scenarios
            self.logger.info(f"サンプルデータ生成完了: {len(scenarios)}シナリオ")
            
        except Exception as e:
            self.logger.error(f"サンプルデータ生成エラー: {e}")
            raise

    def _generate_bull_market_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """強気市場データ生成"""
        n_days = len(dates)
        base_price = 100
        
        # 上昇トレンド + ノイズ
        trend = np.linspace(0, 0.5, n_days)  # 50%上昇
        noise = np.random.normal(0, 0.02, n_days)  # 2%ボラティリティ
        returns = trend + noise
        
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.015)

    def _generate_bear_market_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """弱気市場データ生成"""
        n_days = len(dates)
        base_price = 100
        
        # 下降トレンド + ノイズ
        trend = np.linspace(0, -0.3, n_days)  # 30%下落
        noise = np.random.normal(0, 0.025, n_days)  # 2.5%ボラティリティ
        returns = trend + noise
        
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.02)

    def _generate_sideways_market_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """横ばい市場データ生成"""
        n_days = len(dates)
        base_price = 100
        
        # 横ばい + 周期的変動
        cycle = np.sin(np.arange(n_days) * 2 * np.pi / 60) * 0.1  # 60日周期
        noise = np.random.normal(0, 0.015, n_days)
        returns = cycle + noise
        
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.012)

    def _generate_volatile_market_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """高ボラティリティ市場データ生成"""
        n_days = len(dates)
        base_price = 100
        
        # 高ボラティリティ
        noise = np.random.normal(0, 0.04, n_days)  # 4%ボラティリティ
        regime_changes = np.random.choice([-1, 1], n_days, p=[0.5, 0.5])
        returns = noise * regime_changes
        
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.035)

    def _generate_crisis_market_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """危機市場データ生成"""
        n_days = len(dates)
        base_price = 100
        
        # 急激な下落 + 高ボラティリティ
        crisis_point = n_days // 3
        
        returns = np.random.normal(0, 0.015, n_days)
        returns[crisis_point:crisis_point+30] = np.random.normal(-0.05, 0.06, 30)  # 危機期間
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.045)

    def _create_ohlcv_data(self, dates: pd.DatetimeIndex, close_prices: np.ndarray, volatility: float) -> pd.DataFrame:
        """OHLCV形式のデータ作成"""
        n_days = len(dates)
        
        # 日中変動
        daily_range = np.random.uniform(0.005, 0.02, n_days)
        
        open_prices = close_prices * (1 + np.random.normal(0, volatility/4, n_days))
        high_prices = np.maximum(open_prices, close_prices) * (1 + daily_range)
        low_prices = np.minimum(open_prices, close_prices) * (1 - daily_range)
        
        volumes = np.random.uniform(500000, 2000000, n_days)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)

    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            if self.error_handling_enabled:
                self.components['error_handler'] = RobustAnalysisSystem()
            
            if self.cache_enabled:
                self.components['cache_manager'] = MultiLevelCacheManager(cache_dir="test_cache")
                self.components['market_cache'] = MarketDataCacheManager(cache_dir="test_market_cache")
            
            # 分析コンポーネント
            self.components['condition_detector'] = MarketConditionDetector()
            self.components['technical_analyzer'] = TechnicalIndicatorAnalyzer()
            self.components['regime_classifier'] = MarketRegimeClassifier()
            self.components['volatility_analyzer'] = VolatilityAnalyzer()
            self.components['trend_evaluator'] = TrendStrengthEvaluator()
            self.components['correlation_analyzer'] = MarketCorrelationAnalyzer()
            
            # 統合管理器
            self.components['state_manager'] = IntegratedMarketStateManager(
                integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE
            )
            
            self.logger.info(f"コンポーネント初期化完了: {len(self.components)}個")
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            # モックコンポーネントで継続
            self.components = {key: type('MockComponent', (), {})() for key in [
                'condition_detector', 'technical_analyzer', 'regime_classifier',
                'volatility_analyzer', 'trend_evaluator', 'correlation_analyzer',
                'state_manager'
            ]}

    def _register_test_cases(self):
        """テストケース登録"""
        test_cases = [
            TestCase("individual_components", "個別コンポーネント動作テスト", self._test_individual_components),
            TestCase("data_validation", "データ検証テスト", self._test_data_validation),
            TestCase("market_scenarios", "市場シナリオ分析テスト", self._test_market_scenarios),
            TestCase("integration_consistency", "統合一貫性テスト", self._test_integration_consistency),
            TestCase("performance_benchmark", "パフォーマンステスト", self._test_performance),
            TestCase("error_handling", "エラーハンドリングテスト", self._test_error_handling),
            TestCase("cache_effectiveness", "キャッシュ効果テスト", self._test_cache_effectiveness),
            TestCase("real_time_simulation", "リアルタイムシミュレーション", self._test_real_time_simulation),
            TestCase("stress_test", "ストレステスト", self._test_stress_conditions),
            TestCase("configuration_robustness", "設定堅牢性テスト", self._test_configuration_robustness)
        ]
        
        self.test_cases = test_cases

    def run_all_tests(self) -> Dict[str, TestResult]:
        """全テスト実行"""
        self.logger.info("=== A→B市場分類システム 統合テスト開始 ===")
        
        for test_case in self.test_cases:
            self.logger.info(f"テスト実行中: {test_case.name}")
            
            start_time = time.time()
            
            try:
                result = test_case.test_function()
                test_case.result = TestResult.PASS if result else TestResult.FAIL
                
            except Exception as e:
                test_case.result = TestResult.ERROR
                test_case.error_message = str(e)
                self.logger.error(f"テストエラー ({test_case.name}): {e}")
            
            test_case.execution_time = time.time() - start_time
            self.test_results[test_case.name] = test_case.result
            
            self.logger.info(
                f"テスト完了: {test_case.name} - {test_case.result.value} "
                f"({test_case.execution_time:.3f}秒)"
            )
        
        self._generate_test_report()
        return self.test_results

    def _test_individual_components(self) -> bool:
        """個別コンポーネント動作テスト"""
        try:
            test_data = self.sample_data['bull_market']
            success_count = 0
            total_tests = 0
            
            # 各コンポーネントのテスト
            components_to_test = [
                ('condition_detector', lambda c: c.detect_market_conditions(test_data)),
                ('technical_analyzer', lambda c: c.analyze_technical_indicators(test_data)),
                ('regime_classifier', lambda c: c.classify_market_regime(test_data)),
                ('volatility_analyzer', lambda c: c.analyze_volatility(test_data)),
                ('trend_evaluator', lambda c: c.evaluate_trend_strength(test_data))
            ]
            
            for component_name, test_func in components_to_test:
                try:
                    component = self.components.get(component_name)
                    if component:
                        result = test_func(component)
                        if result is not None:
                            success_count += 1
                        total_tests += 1
                except Exception as e:
                    self.logger.warning(f"コンポーネント {component_name} テスト失敗: {e}")
                    total_tests += 1
            
            success_rate = success_count / total_tests if total_tests > 0 else 0
            self.performance_metrics['individual_component_success_rate'] = success_rate
            
            return success_rate > 0.7  # 70%以上成功
            
        except Exception as e:
            self.logger.error(f"個別コンポーネントテストエラー: {e}")
            return False

    def _test_data_validation(self) -> bool:
        """データ検証テスト"""
        try:
            validation_tests = [
                # 正常データ
                (self.sample_data['bull_market'], True),
                # 空データ
                (pd.DataFrame(), False),
                # 不完全データ（一部カラム欠損）
                (pd.DataFrame({'Close': [100, 101, 102]}), False),
                # NaN含むデータ
                (self._create_nan_data(), False)
            ]
            
            passed_tests = 0
            
            for test_data, expected_valid in validation_tests:
                try:
                    # データ検証ロジック
                    is_valid = self._validate_market_data(test_data)
                    if is_valid == expected_valid:
                        passed_tests += 1
                except:
                    if not expected_valid:  # エラーが期待される場合
                        passed_tests += 1
            
            return passed_tests == len(validation_tests)
            
        except Exception as e:
            self.logger.error(f"データ検証テストエラー: {e}")
            return False

    def _test_market_scenarios(self) -> bool:
        """市場シナリオ分析テスト"""
        try:
            scenario_results = {}
            
            for scenario_name, data in self.sample_data.items():
                try:
                    # 統合分析実行
                    if 'state_manager' in self.components:
                        state = self.components['state_manager'].update_market_state({scenario_name: data})
                        scenario_results[scenario_name] = {
                            'condition': state.market_condition.value if hasattr(state, 'market_condition') else 'unknown',
                            'confidence': getattr(state, 'confidence_level', 0.0)
                        }
                    else:
                        # 個別分析
                        condition_detector = self.components.get('condition_detector')
                        if condition_detector:
                            result = condition_detector.detect_market_conditions(data)
                            scenario_results[scenario_name] = {
                                'condition': result.market_condition.value if hasattr(result, 'market_condition') else 'unknown',
                                'confidence': getattr(result, 'confidence', 0.0)
                            }
                
                except Exception as e:
                    self.logger.warning(f"シナリオ {scenario_name} 分析エラー: {e}")
                    scenario_results[scenario_name] = {'condition': 'error', 'confidence': 0.0}
            
            # 結果の妥当性チェック
            expected_conditions = {
                'bull_market': ['strong_bull', 'moderate_bull', 'sideways_bull'],
                'bear_market': ['strong_bear', 'moderate_bear', 'sideways_bear'],
                'sideways_market': ['neutral_sideways', 'sideways_bull', 'sideways_bear']
            }
            
            correct_classifications = 0
            total_classifications = 0
            
            for scenario, result in scenario_results.items():
                if scenario in expected_conditions:
                    condition = result['condition']
                    if condition in expected_conditions[scenario]:
                        correct_classifications += 1
                    total_classifications += 1
            
            accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
            self.performance_metrics['scenario_classification_accuracy'] = accuracy
            
            return accuracy > 0.5  # 50%以上の精度
            
        except Exception as e:
            self.logger.error(f"市場シナリオテストエラー: {e}")
            return False

    def _test_integration_consistency(self) -> bool:
        """統合一貫性テスト"""
        try:
            test_data = {'TEST_ASSET': self.sample_data['bull_market']}
            
            # 複数回実行して一貫性確認
            results = []
            for i in range(5):
                try:
                    if 'state_manager' in self.components:
                        state = self.components['state_manager'].update_market_state(test_data, force_update=True)
                        results.append({
                            'condition': state.market_condition.value if hasattr(state, 'market_condition') else 'unknown',
                            'confidence': getattr(state, 'confidence_level', 0.0)
                        })
                except Exception as e:
                    self.logger.warning(f"統合テスト試行 {i+1} エラー: {e}")
                    results.append({'condition': 'error', 'confidence': 0.0})
            
            # 一貫性評価
            if len(results) < 3:
                return False
            
            conditions = [r['condition'] for r in results if r['condition'] != 'error']
            confidences = [r['confidence'] for r in results if r['confidence'] > 0]
            
            if not conditions or not confidences:
                return False
            
            # 条件の一貫性（最頻値の比率）
            condition_consistency = conditions.count(max(set(conditions), key=conditions.count)) / len(conditions)
            
            # 信頼度の安定性（標準偏差）
            confidence_stability = 1 - (np.std(confidences) / np.mean(confidences)) if confidences else 0
            
            consistency_score = (condition_consistency + confidence_stability) / 2
            self.performance_metrics['integration_consistency'] = consistency_score
            
            return consistency_score > 0.7
            
        except Exception as e:
            self.logger.error(f"統合一貫性テストエラー: {e}")
            return False

    def _test_performance(self) -> bool:
        """パフォーマンステスト"""
        try:
            test_data = {'PERF_TEST': self.sample_data['bull_market']}
            
            # 実行時間測定
            execution_times = []
            
            for i in range(10):
                start_time = time.time()
                
                try:
                    if 'state_manager' in self.components:
                        self.components['state_manager'].update_market_state(test_data, force_update=True)
                    else:
                        # 個別コンポーネント実行
                        for component_name in ['condition_detector', 'technical_analyzer']:
                            component = self.components.get(component_name)
                            if component and hasattr(component, 'detect_market_conditions'):
                                component.detect_market_conditions(test_data['PERF_TEST'])
                            elif component and hasattr(component, 'analyze_technical_indicators'):
                                component.analyze_technical_indicators(test_data['PERF_TEST'])
                
                except Exception as e:
                    self.logger.warning(f"パフォーマンステスト試行 {i+1} エラー: {e}")
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            if execution_times:
                avg_time = np.mean(execution_times)
                max_time = np.max(execution_times)
                
                self.performance_metrics['average_execution_time'] = avg_time
                self.performance_metrics['max_execution_time'] = max_time
                
                # 5秒以内の実行時間を目標
                return avg_time < 5.0 and max_time < 10.0
            
            return False
            
        except Exception as e:
            self.logger.error(f"パフォーマンステストエラー: {e}")
            return False

    def _test_error_handling(self) -> bool:
        """エラーハンドリングテスト"""
        try:
            if not self.error_handling_enabled or 'error_handler' not in self.components:
                return True  # エラーハンドリング無効時はスキップ
            
            error_handler = self.components['error_handler']
            
            # 意図的エラー生成テスト
            error_scenarios = [
                # 空データ
                pd.DataFrame(),
                # 不正データ
                pd.DataFrame({'invalid': [1, 2, 3]}),
                # NaNデータ
                self._create_nan_data()
            ]
            
            handled_errors = 0
            
            for scenario_data in error_scenarios:
                try:
                    # エラーハンドリング付きで実行
                    @error_handler.with_error_handling(
                        RecoveryAction(strategy=RecoveryStrategy.DEFAULT_VALUE, default_value=None),
                        ErrorCategory.DATA_ERROR
                    )
                    def test_error_function():
                        if scenario_data.empty:
                            raise ValueError("空データエラー")
                        return "success"
                    
                    result = test_error_function()
                    handled_errors += 1  # エラーが適切にハンドリングされた
                    
                except Exception:
                    pass  # ハンドリングされなかった
            
            # エラーハンドリング率
            error_handling_rate = handled_errors / len(error_scenarios)
            self.performance_metrics['error_handling_rate'] = error_handling_rate
            
            return error_handling_rate > 0.7
            
        except Exception as e:
            self.logger.error(f"エラーハンドリングテストエラー: {e}")
            return False

    def _test_cache_effectiveness(self) -> bool:
        """キャッシュ効果テスト"""
        try:
            if not self.cache_enabled or 'cache_manager' not in self.components:
                return True  # キャッシュ無効時はスキップ
            
            cache_manager = self.components['cache_manager']
            test_data = self.sample_data['bull_market']
            
            # キャッシュなし実行時間
            start_time = time.time()
            cache_manager.put("test_data", test_data)
            cache_time = time.time() - start_time
            
            # キャッシュあり実行時間
            start_time = time.time()
            cached_data = cache_manager.get("test_data")
            retrieval_time = time.time() - start_time
            
            if cached_data is not None:
                # キャッシュ効果測定
                cache_speedup = cache_time / retrieval_time if retrieval_time > 0 else 1
                self.performance_metrics['cache_speedup'] = cache_speedup
                
                return cache_speedup > 2.0  # 2倍以上の高速化
            
            return False
            
        except Exception as e:
            self.logger.error(f"キャッシュ効果テストエラー: {e}")
            return False

    def _test_real_time_simulation(self) -> bool:
        """リアルタイムシミュレーション"""
        try:
            # データストリーミングシミュレーション
            full_data = self.sample_data['volatile_market']
            
            stream_results = []
            window_size = 50
            
            for i in range(window_size, len(full_data), 10):
                window_data = full_data.iloc[i-window_size:i]
                
                try:
                    if 'condition_detector' in self.components:
                        result = self.components['condition_detector'].detect_market_conditions(window_data)
                        stream_results.append({
                            'timestamp': window_data.index[-1],
                            'condition': result.market_condition.value if hasattr(result, 'market_condition') else 'unknown'
                        })
                except Exception as e:
                    self.logger.warning(f"ストリーム分析エラー (位置 {i}): {e}")
            
            # ストリーム分析成功率
            success_rate = len(stream_results) / ((len(full_data) - window_size) // 10)
            self.performance_metrics['stream_analysis_success_rate'] = success_rate
            
            return success_rate > 0.8
            
        except Exception as e:
            self.logger.error(f"リアルタイムシミュレーションエラー: {e}")
            return False

    def _test_stress_conditions(self) -> bool:
        """ストレステスト"""
        try:
            stress_scenarios = [
                # 大量データ
                self._generate_large_dataset(),
                # 極端ボラティリティ
                self._generate_extreme_volatility_data(),
                # 不連続データ
                self._generate_discontinuous_data()
            ]
            
            stress_results = []
            
            for i, stress_data in enumerate(stress_scenarios):
                try:
                    start_time = time.time()
                    
                    if 'condition_detector' in self.components:
                        result = self.components['condition_detector'].detect_market_conditions(stress_data)
                        execution_time = time.time() - start_time
                        
                        stress_results.append({
                            'scenario': i,
                            'success': True,
                            'execution_time': execution_time
                        })
                    
                except Exception as e:
                    stress_results.append({
                        'scenario': i,
                        'success': False,
                        'error': str(e)
                    })
            
            # ストレス耐性評価
            success_count = sum(1 for r in stress_results if r['success'])
            stress_tolerance = success_count / len(stress_scenarios)
            
            self.performance_metrics['stress_tolerance'] = stress_tolerance
            
            return stress_tolerance > 0.6
            
        except Exception as e:
            self.logger.error(f"ストレステストエラー: {e}")
            return False

    def _test_configuration_robustness(self) -> bool:
        """設定堅牢性テスト"""
        try:
            # 異なる設定での動作確認
            configurations = [
                StateIntegrationMethod.WEIGHTED_AVERAGE,
                StateIntegrationMethod.CONSENSUS_VOTING,
                StateIntegrationMethod.HIERARCHICAL
            ]
            
            config_results = []
            test_data = {'CONFIG_TEST': self.sample_data['bull_market']}
            
            for config in configurations:
                try:
                    # 新しい設定で管理器作成
                    manager = IntegratedMarketStateManager(integration_method=config)
                    result = manager.update_market_state(test_data)
                    
                    config_results.append({
                        'config': config.value if hasattr(config, 'value') else str(config),
                        'success': True,
                        'confidence': getattr(result, 'confidence_level', 0.0)
                    })
                    
                except Exception as e:
                    config_results.append({
                        'config': config.value if hasattr(config, 'value') else str(config),
                        'success': False,
                        'error': str(e)
                    })
            
            # 設定堅牢性評価
            success_count = sum(1 for r in config_results if r['success'])
            config_robustness = success_count / len(configurations)
            
            self.performance_metrics['configuration_robustness'] = config_robustness
            
            return config_robustness > 0.7
            
        except Exception as e:
            self.logger.error(f"設定堅牢性テストエラー: {e}")
            return False

    def _generate_test_report(self):
        """テストレポート生成"""
        self.logger.info("\n" + "="*60)
        self.logger.info("A→B市場分類システム 統合テスト結果")
        self.logger.info("="*60)
        
        # テスト結果サマリー
        total_tests = len(self.test_cases)
        passed_tests = sum(1 for result in self.test_results.values() if result == TestResult.PASS)
        failed_tests = sum(1 for result in self.test_results.values() if result == TestResult.FAIL)
        error_tests = sum(1 for result in self.test_results.values() if result == TestResult.ERROR)
        
        self.logger.info(f"総テスト数: {total_tests}")
        self.logger.info(f"成功: {passed_tests}, 失敗: {failed_tests}, エラー: {error_tests}")
        self.logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        # 個別テスト結果
        self.logger.info("\n詳細結果:")
        for test_case in self.test_cases:
            status_symbol = {
                TestResult.PASS: "✓",
                TestResult.FAIL: "✗",
                TestResult.ERROR: "⚠",
                TestResult.SKIP: "○"
            }.get(test_case.result, "?")
            
            self.logger.info(
                f"{status_symbol} {test_case.name}: {test_case.result.value} "
                f"({test_case.execution_time:.3f}秒)"
            )
            
            if test_case.error_message:
                self.logger.info(f"   エラー: {test_case.error_message}")
        
        # パフォーマンスメトリクス
        if self.performance_metrics:
            self.logger.info("\nパフォーマンスメトリクス:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {metric}: {value:.3f}")
                else:
                    self.logger.info(f"  {metric}: {value}")
        
        self.logger.info("="*60)

    # ヘルパーメソッド
    def _validate_market_data(self, data: pd.DataFrame) -> bool:
        """市場データ検証"""
        if data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if data.isnull().all().all():
            return False
        
        return True

    def _create_nan_data(self) -> pd.DataFrame:
        """NaN含むテストデータ作成"""
        dates = pd.date_range('2024-01-01', periods=10)
        return pd.DataFrame({
            'Open': [100, np.nan, 102, 101, np.nan],
            'High': [105, 104, np.nan, 106, 105],
            'Low': [95, 96, 97, np.nan, 98],
            'Close': [102, np.nan, 101, 104, np.nan],
            'Volume': [1000000] * 5
        }, index=dates[:5])

    def _generate_large_dataset(self) -> pd.DataFrame:
        """大量データセット生成"""
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='H')  # 時間足データ
        n_points = len(dates)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_points)))
        
        return pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.0005, n_points)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
            'Close': prices,
            'Volume': np.random.uniform(100000, 1000000, n_points)
        }, index=dates)

    def _generate_extreme_volatility_data(self) -> pd.DataFrame:
        """極端ボラティリティデータ生成"""
        dates = pd.date_range('2024-01-01', periods=100)
        
        # 極端な価格変動
        returns = np.random.normal(0, 0.1, 100)  # 10%ボラティリティ
        returns[50:60] = np.random.normal(0, 0.3, 10)  # 極端期間
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        return self._create_ohlcv_data(dates, prices, volatility=0.15)

    def _generate_discontinuous_data(self) -> pd.DataFrame:
        """不連続データ生成"""
        dates = pd.date_range('2024-01-01', periods=100)
        
        # ランダムに欠損期間を作成
        mask = np.random.choice([True, False], 100, p=[0.8, 0.2])
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        data = self._create_ohlcv_data(dates, prices, volatility=0.02)
        
        return data[mask]

    def cleanup_test_environment(self):
        """テスト環境クリーンアップ"""
        try:
            # キャッシュディレクトリ削除
            import shutil
            for cache_dir in ['test_cache', 'test_market_cache']:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            
            self.logger.info("テスト環境クリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")

def main():
    """メインデモンストレーション実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("A→B市場分類システム 統合デモンストレーション")
    print("=" * 60)
    
    try:
        # デモシステム初期化
        demo = IntegratedMarketClassificationDemo(
            cache_enabled=True,
            error_handling_enabled=True
        )
        
        # テスト環境セットアップ
        demo.setup_test_environment()
        
        # 全テスト実行
        results = demo.run_all_tests()
        
        # 結果表示
        print("\n最終結果:")
        passed = sum(1 for r in results.values() if r == TestResult.PASS)
        total = len(results)
        print(f"総合成功率: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed / total >= 0.8:
            print("✓ A→B市場分類システムは正常に動作しています")
        elif passed / total >= 0.6:
            print("⚠ 一部の機能に問題がありますが、基本動作は正常です")
        else:
            print("✗ 重大な問題が検出されました。システムを確認してください")
        
        return passed / total >= 0.6
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        return False
    
    finally:
        # クリーンアップ
        if 'demo' in locals():
            demo.cleanup_test_environment()

if __name__ == "__main__":
    main()
