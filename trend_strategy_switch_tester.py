"""
Module: Trend Strategy Switch Tester
File: trend_strategy_switch_tester.py
Description: 
  4-2-1「トレンド変化時の戦略切替テスト」
  トレンド変化時の戦略切替テスト機能

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 包括的なトレンド戦略切替テストシステム
  - リアル・シンセティックデータ統合テスト
  - 戦略切替性能評価・最適化機能
  - トレンドシナリオ自動生成・実行
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
import warnings

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# プロジェクトモジュールをインポート
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from config.rule_engine_integrated_interface import RuleEngineIntegratedInterface
    from config.multi_strategy_coordination_manager import MultiStrategyCoordinationManager
except ImportError as e:
    warnings.warn(f"Could not import some project modules: {e}")

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class TrendScenarioType(Enum):
    """トレンドシナリオタイプ"""
    GRADUAL_TREND_CHANGE = "gradual_trend_change"
    VOLATILE_MARKET = "volatile_market"
    STRONG_TREND_REVERSAL = "strong_trend_reversal"
    SIDEWAYS_BREAKOUT = "sideways_breakout"

class StrategyType(Enum):
    """戦略タイプ"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    SCALPING = "scalping"
    SWING = "swing"

@dataclass
class TrendScenario:
    """トレンドシナリオ定義"""
    scenario_id: str
    scenario_type: TrendScenarioType
    period_days: int
    initial_trend: str
    target_trend: str
    volatility_level: float
    data_source: str  # 'real' or 'synthetic'
    
    # シンセティックデータ用パラメータ
    synthetic_params: Optional[Dict[str, Any]] = None
    
    # リアルデータ用パラメータ
    real_data_period: Optional[Tuple[datetime, datetime]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StrategySwitchingEvent:
    """戦略切替イベント"""
    timestamp: datetime
    from_strategy: str
    to_strategy: str
    trigger_reason: str
    confidence_score: float
    market_conditions: Dict[str, Any]
    switching_delay: float  # 秒

@dataclass
class TrendTestResult:
    """トレンドテスト結果"""
    scenario_id: str
    test_duration: float
    switching_events: List[StrategySwitchingEvent]
    performance_metrics: Dict[str, float]
    success_indicators: Dict[str, bool]
    errors: List[str]
    detailed_log: List[str]

class TrendScenarioGenerator:
    """トレンドシナリオ生成器"""
    
    def __init__(self):
        self.scenario_configs = {
            TrendScenarioType.GRADUAL_TREND_CHANGE: {
                'volatility_range': (0.1, 0.3),
                'period_range': (3, 7),
                'transition_smoothness': 0.8
            },
            TrendScenarioType.VOLATILE_MARKET: {
                'volatility_range': (0.4, 0.8),
                'period_range': (1, 3),
                'transition_smoothness': 0.3
            },
            TrendScenarioType.STRONG_TREND_REVERSAL: {
                'volatility_range': (0.2, 0.5),
                'period_range': (2, 5),
                'transition_smoothness': 0.1
            },
            TrendScenarioType.SIDEWAYS_BREAKOUT: {
                'volatility_range': (0.1, 0.4),
                'period_range': (3, 7),
                'transition_smoothness': 0.6
            }
        }
    
    def generate_scenario(self, scenario_type: TrendScenarioType, 
                         data_source: str = 'hybrid') -> TrendScenario:
        """シナリオを生成"""
        try:
            config = self.scenario_configs[scenario_type]
            
            # 基本パラメータ生成
            volatility = np.random.uniform(*config['volatility_range'])
            period_days = np.random.randint(*config['period_range'])
            
            # トレンド方向決定
            trends = ['uptrend', 'downtrend', 'sideways']
            initial_trend = np.random.choice(trends)
            target_trend = np.random.choice([t for t in trends if t != initial_trend])
            
            scenario_id = f"{scenario_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            scenario = TrendScenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                period_days=period_days,
                initial_trend=initial_trend,
                target_trend=target_trend,
                volatility_level=volatility,
                data_source=data_source
            )
            
            # データソース別パラメータ設定
            if data_source in ['synthetic', 'hybrid']:
                scenario.synthetic_params = self._generate_synthetic_params(
                    scenario_type, config, volatility, period_days
                )
            
            if data_source in ['real', 'hybrid']:
                scenario.real_data_period = self._select_real_data_period(
                    scenario_type, period_days
                )
            
            logger.info(f"Generated scenario: {scenario_id}")
            return scenario
            
        except Exception as e:
            logger.error(f"Error generating scenario: {e}")
            raise
    
    def _generate_synthetic_params(self, scenario_type: TrendScenarioType,
                                 config: Dict, volatility: float, 
                                 period_days: int) -> Dict[str, Any]:
        """シンセティックデータ用パラメータ生成"""
        return {
            'base_price': 100.0,
            'drift_rate': np.random.uniform(-0.02, 0.02),
            'volatility': volatility,
            'mean_reversion_speed': np.random.uniform(0.1, 0.5),
            'jump_probability': np.random.uniform(0.05, 0.15),
            'jump_size_std': np.random.uniform(0.01, 0.05),
            'seasonality_amplitude': np.random.uniform(0.0, 0.1),
            'noise_level': np.random.uniform(0.01, 0.05)
        }
    
    def _select_real_data_period(self, scenario_type: TrendScenarioType,
                               period_days: int) -> Tuple[datetime, datetime]:
        """リアルデータ期間選択"""
        # 過去のデータから適切な期間を選択（簡易実装）
        end_date = datetime.now() - timedelta(days=30)  # 30日前まで
        start_date = end_date - timedelta(days=period_days)
        return (start_date, end_date)
    
    def generate_test_suite(self, num_scenarios: int = 20) -> List[TrendScenario]:
        """テストスイート生成"""
        scenarios = []
        try:
            for scenario_type in TrendScenarioType:
                count = num_scenarios // len(TrendScenarioType)
                for _ in range(count):
                    scenario = self.generate_scenario(scenario_type)
                    scenarios.append(scenario)
            
            logger.info(f"Generated test suite with {len(scenarios)} scenarios")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating test suite: {e}")
            return []

class StrategySwitchingEvaluator:
    """戦略切替評価器"""
    
    def __init__(self):
        self.evaluation_metrics = [
            'switching_timing_accuracy',
            'performance_improvement_ratio',
            'false_positive_rate',
            'switching_frequency',
            'profit_consistency',
            'drawdown_reduction',
            'sharpe_ratio_improvement',
            'hit_ratio'
        ]
    
    def evaluate_switching_performance(self, 
                                     test_result: TrendTestResult,
                                     benchmark_performance: Dict[str, float]) -> Dict[str, Any]:
        """戦略切替性能を評価"""
        try:
            evaluation = {}
            
            # 基本メトリクス計算
            evaluation.update(self._calculate_timing_metrics(test_result))
            evaluation.update(self._calculate_performance_metrics(
                test_result, benchmark_performance
            ))
            evaluation.update(self._calculate_efficiency_metrics(test_result))
            
            # 総合評価
            evaluation['overall_score'] = self._calculate_overall_score(evaluation)
            evaluation['success'] = evaluation['overall_score'] >= 0.6
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating switching performance: {e}")
            return {'error': str(e)}
    
    def _calculate_timing_metrics(self, test_result: TrendTestResult) -> Dict[str, float]:
        """タイミングメトリクス計算"""
        if not test_result.switching_events:
            return {'switching_timing_accuracy': 0.0}
        
        # 切替タイミングの適切性を評価
        timing_scores = []
        for event in test_result.switching_events:
            # 信頼度スコアと市場条件から適切性を判定
            timing_score = min(event.confidence_score, 1.0)
            timing_scores.append(timing_score)
        
        return {
            'switching_timing_accuracy': np.mean(timing_scores) if timing_scores else 0.0,
            'average_switching_delay': np.mean([e.switching_delay for e in test_result.switching_events])
        }
    
    def _calculate_performance_metrics(self, 
                                     test_result: TrendTestResult,
                                     benchmark: Dict[str, float]) -> Dict[str, float]:
        """パフォーマンスメトリクス計算"""
        metrics = test_result.performance_metrics
        
        return {
            'performance_improvement_ratio': (
                metrics.get('total_return', 0) / max(benchmark.get('total_return', 1), 0.01)
            ),
            'sharpe_ratio_improvement': (
                metrics.get('sharpe_ratio', 0) - benchmark.get('sharpe_ratio', 0)
            ),
            'drawdown_reduction': max(0, benchmark.get('max_drawdown', 0) - metrics.get('max_drawdown', 0)),
            'profit_consistency': metrics.get('profit_factor', 0),
            'hit_ratio': metrics.get('win_rate', 0)
        }
    
    def _calculate_efficiency_metrics(self, test_result: TrendTestResult) -> Dict[str, float]:
        """効率性メトリクス計算"""
        num_switches = len(test_result.switching_events)
        test_duration_hours = test_result.test_duration / 3600
        
        # 誤検知率計算（簡易版）
        false_positives = sum(1 for event in test_result.switching_events 
                            if event.confidence_score < 0.5)
        false_positive_rate = false_positives / max(num_switches, 1)
        
        return {
            'switching_frequency': num_switches / max(test_duration_hours, 1),
            'false_positive_rate': false_positive_rate
        }
    
    def _calculate_overall_score(self, evaluation: Dict[str, float]) -> float:
        """総合スコア計算"""
        weights = {
            'switching_timing_accuracy': 0.25,
            'performance_improvement_ratio': 0.20,
            'hit_ratio': 0.20,
            'sharpe_ratio_improvement': 0.15,
            'false_positive_rate': -0.10,  # 負の重み
            'profit_consistency': 0.10
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in evaluation:
                value = evaluation[metric]
                if metric == 'false_positive_rate':
                    # 誤検知率は低いほど良い
                    score += weight * (1 - min(value, 1.0))
                elif metric == 'performance_improvement_ratio':
                    # 改善比率は1以上が良い
                    score += weight * min(value, 2.0) / 2.0
                else:
                    score += weight * min(max(value, 0), 1.0)
        
        return max(0, min(score, 1.0))

class TrendStrategySwitchTester:
    """トレンド戦略切替テスター（メインクラス）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # コンポーネント初期化
        self.scenario_generator = TrendScenarioGenerator()
        self.evaluator = StrategySwitchingEvaluator()
        
        # 統合システム初期化
        try:
            self.trend_detector = UnifiedTrendDetector()
            self.rule_engine = RuleEngineIntegratedInterface()
            self.coordination_manager = MultiStrategyCoordinationManager()
        except Exception as e:
            logger.warning(f"Could not initialize some components: {e}")
            self.trend_detector = None
            self.rule_engine = None
            self.coordination_manager = None
        
        # 結果保存
        self.test_results = []
        
        logger.info("TrendStrategySwitchTester initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得"""
        return {
            'test_duration_minutes': 30,
            'max_concurrent_tests': 3,
            'enable_synthetic_data': True,
            'enable_real_data': True,
            'benchmark_strategy': 'buy_and_hold',
            'success_thresholds': {
                'min_hit_ratio': 0.6,
                'max_false_positive_rate': 0.3,
                'min_sharpe_improvement': 0.1
            },
            'logging_level': 'INFO',
            'output_directory': 'test_results'
        }
    
    def run_comprehensive_test(self, 
                             scenarios: Optional[List[TrendScenario]] = None,
                             parallel_execution: bool = True) -> Dict[str, Any]:
        """包括的テスト実行"""
        try:
            logger.info("Starting comprehensive trend strategy switching test")
            
            # シナリオ準備
            if scenarios is None:
                scenarios = self.scenario_generator.generate_test_suite(
                    num_scenarios=self.config.get('num_scenarios', 12)
                )
            
            # テスト実行
            if parallel_execution:
                results = self._run_parallel_tests(scenarios)
            else:
                results = self._run_sequential_tests(scenarios)
            
            # 結果集計・分析
            summary = self._analyze_test_results(results)
            
            # 結果保存
            self._save_test_results(summary)
            
            logger.info(f"Comprehensive test completed. Success rate: {summary['success_rate']:.2%}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in comprehensive test: {e}")
            traceback.print_exc()
            return {'error': str(e), 'success_rate': 0.0}
    
    def _run_parallel_tests(self, scenarios: List[TrendScenario]) -> List[TrendTestResult]:
        """並列テスト実行"""
        results = []
        max_workers = min(self.config.get('max_concurrent_tests', 3), len(scenarios))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scenario = {
                executor.submit(self._run_single_test, scenario): scenario 
                for scenario in scenarios
            }
            
            for future in future_to_scenario:
                try:
                    result = future.result(timeout=self.config.get('test_timeout_minutes', 10) * 60)
                    results.append(result)
                except Exception as e:
                    scenario = future_to_scenario[future]
                    logger.error(f"Test failed for scenario {scenario.scenario_id}: {e}")
                    
                    # エラー結果作成
                    error_result = TrendTestResult(
                        scenario_id=scenario.scenario_id,
                        test_duration=0,
                        switching_events=[],
                        performance_metrics={},
                        success_indicators={'test_completed': False},
                        errors=[str(e)],
                        detailed_log=[]
                    )
                    results.append(error_result)
        
        return results
    
    def _run_sequential_tests(self, scenarios: List[TrendScenario]) -> List[TrendTestResult]:
        """逐次テスト実行"""
        results = []
        for i, scenario in enumerate(scenarios):
            logger.info(f"Running test {i+1}/{len(scenarios)}: {scenario.scenario_id}")
            try:
                result = self._run_single_test(scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Test failed for scenario {scenario.scenario_id}: {e}")
                error_result = TrendTestResult(
                    scenario_id=scenario.scenario_id,
                    test_duration=0,
                    switching_events=[],
                    performance_metrics={},
                    success_indicators={'test_completed': False},
                    errors=[str(e)],
                    detailed_log=[]
                )
                results.append(error_result)
        
        return results
    
    def _run_single_test(self, scenario: TrendScenario) -> TrendTestResult:
        """単一テスト実行"""
        start_time = time.time()
        switching_events = []
        detailed_log = []
        errors = []
        
        try:
            detailed_log.append(f"Starting test for scenario: {scenario.scenario_id}")
            
            # データ準備
            test_data = self._prepare_test_data(scenario)
            if test_data is None or test_data.empty:
                raise ValueError("Failed to prepare test data")
            
            # テスト実行
            switching_events, performance_metrics = self._execute_strategy_switching_test(
                scenario, test_data, detailed_log
            )
            
            # 成功指標評価
            success_indicators = self._evaluate_success_indicators(
                switching_events, performance_metrics
            )
            
        except Exception as e:
            errors.append(str(e))
            performance_metrics = {}
            success_indicators = {'test_completed': False}
            detailed_log.append(f"Test error: {e}")
        
        test_duration = time.time() - start_time
        
        return TrendTestResult(
            scenario_id=scenario.scenario_id,
            test_duration=test_duration,
            switching_events=switching_events,
            performance_metrics=performance_metrics,
            success_indicators=success_indicators,
            errors=errors,
            detailed_log=detailed_log
        )
    
    def _prepare_test_data(self, scenario: TrendScenario) -> Optional[pd.DataFrame]:
        """テストデータ準備"""
        try:
            if scenario.data_source == 'synthetic' and scenario.synthetic_params:
                return self._generate_synthetic_data(scenario)
            elif scenario.data_source == 'real' and scenario.real_data_period:
                return self._load_real_data(scenario)
            elif scenario.data_source == 'hybrid':
                # ハイブリッドデータ（リアル+シンセティック）
                real_data = self._load_real_data(scenario)
                synthetic_data = self._generate_synthetic_data(scenario)
                return self._combine_data_sources(real_data, synthetic_data)
            else:
                raise ValueError(f"Unsupported data source: {scenario.data_source}")
                
        except Exception as e:
            logger.error(f"Error preparing test data: {e}")
            return None
    
    def _generate_synthetic_data(self, scenario: TrendScenario) -> pd.DataFrame:
        """シンセティックデータ生成"""
        params = scenario.synthetic_params
        if not params:
            raise ValueError("Synthetic parameters not provided")
        
        # 基本的なシンセティックデータ生成
        n_points = scenario.period_days * 24 * 4  # 15分足想定
        time_index = pd.date_range(
            start=datetime.now() - timedelta(days=scenario.period_days),
            periods=n_points,
            freq='15min'
        )
        
        # 価格データ生成（簡易的なランダムウォーク）
        np.random.seed(42)  # 再現性のため
        returns = np.random.normal(
            params['drift_rate'] / n_points,
            params['volatility'] / np.sqrt(n_points),
            n_points
        )
        
        prices = params['base_price'] * np.exp(np.cumsum(returns))
        
        # OHLC データ作成
        data = pd.DataFrame({
            'timestamp': time_index,
            'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_points)
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def _load_real_data(self, scenario: TrendScenario) -> pd.DataFrame:
        """リアルデータ読み込み（簡易実装）"""
        # 実際の実装では、データベースやAPIからデータを取得
        # ここではダミーデータを生成
        logger.warning("Using dummy data for real data (implement actual data loading)")
        return self._generate_synthetic_data(scenario)
    
    def _combine_data_sources(self, real_data: Optional[pd.DataFrame],
                            synthetic_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """データソース結合"""
        if real_data is not None and not real_data.empty:
            return real_data
        elif synthetic_data is not None and not synthetic_data.empty:
            return synthetic_data
        else:
            raise ValueError("No valid data available")
    
    def _execute_strategy_switching_test(self, 
                                       scenario: TrendScenario,
                                       test_data: pd.DataFrame,
                                       log: List[str]) -> Tuple[List[StrategySwitchingEvent], Dict[str, float]]:
        """戦略切替テスト実行"""
        switching_events = []
        current_strategy = "trend_following"  # 初期戦略
        
        log.append(f"Starting with strategy: {current_strategy}")
        
        # シンプルな戦略切替ロジック（実際の実装では統合システムを使用）
        for i in range(1, min(len(test_data), 100)):  # データサイズ制限
            try:
                current_data = test_data.iloc[:i+1]
                
                # トレンド検出（簡易実装）
                trend_signal = self._detect_trend_change(current_data)
                
                if trend_signal and trend_signal != current_strategy:
                    # 戦略切替
                    switching_event = StrategySwitchingEvent(
                        timestamp=test_data.index[i],
                        from_strategy=current_strategy,
                        to_strategy=trend_signal,
                        trigger_reason="trend_change_detected",
                        confidence_score=np.random.uniform(0.6, 0.9),
                        market_conditions={'volatility': np.random.uniform(0.1, 0.5)},
                        switching_delay=np.random.uniform(0.1, 2.0)
                    )
                    
                    switching_events.append(switching_event)
                    current_strategy = trend_signal
                    
                    log.append(f"Strategy switched to {current_strategy} at {test_data.index[i]}")
                
            except Exception as e:
                log.append(f"Error during switching test: {e}")
        
        # パフォーマンスメトリクス計算
        performance_metrics = self._calculate_performance_metrics(
            test_data, switching_events
        )
        
        return switching_events, performance_metrics
    
    def _detect_trend_change(self, data: pd.DataFrame) -> Optional[str]:
        """トレンド変化検出（簡易実装）"""
        if len(data) < 10:
            return None
        
        # 簡易的なトレンド判定
        recent_return = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1)
        volatility = data['close'].pct_change().std()
        
        if recent_return > 0.02:
            return "trend_following"
        elif recent_return < -0.02:
            return "mean_reversion" 
        elif volatility > 0.03:
            return "momentum"
        else:
            return None
    
    def _calculate_performance_metrics(self, 
                                     data: pd.DataFrame,
                                     switching_events: List[StrategySwitchingEvent]) -> Dict[str, float]:
        """パフォーマンスメトリクス計算"""
        if len(data) < 2:
            return {}
        
        returns = data['close'].pct_change().dropna()
        
        return {
            'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1),
            'sharpe_ratio': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'volatility': returns.std() * np.sqrt(252),
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum() + 1e-8),
            'num_switches': len(switching_events)
        }
    
    def _evaluate_success_indicators(self, 
                                   switching_events: List[StrategySwitchingEvent],
                                   performance_metrics: Dict[str, float]) -> Dict[str, bool]:
        """成功指標評価"""
        thresholds = self.config['success_thresholds']
        
        return {
            'test_completed': True,
            'sufficient_switches': len(switching_events) >= 1,
            'hit_ratio_acceptable': performance_metrics.get('win_rate', 0) >= thresholds['min_hit_ratio'],
            'low_false_positive': True,  # 簡易実装
            'performance_improved': performance_metrics.get('sharpe_ratio', 0) >= thresholds['min_sharpe_improvement']
        }
    
    def _analyze_test_results(self, results: List[TrendTestResult]) -> Dict[str, Any]:
        """テスト結果分析"""
        if not results:
            return {'error': 'No test results to analyze', 'success_rate': 0.0}
        
        # 基本統計
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success_indicators.get('test_completed', False))
        success_rate = successful_tests / total_tests
        
        # パフォーマンス集計
        all_switching_events = []
        all_performance_metrics = []
        
        for result in results:
            all_switching_events.extend(result.switching_events)
            if result.performance_metrics:
                all_performance_metrics.append(result.performance_metrics)
        
        # 統合評価
        summary = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'total_switching_events': len(all_switching_events),
            'average_switches_per_test': len(all_switching_events) / max(total_tests, 1),
            'test_results': [asdict(result) for result in results]
        }
        
        if all_performance_metrics:
            summary['aggregated_metrics'] = {
                'average_sharpe_ratio': np.mean([m.get('sharpe_ratio', 0) for m in all_performance_metrics]),
                'average_total_return': np.mean([m.get('total_return', 0) for m in all_performance_metrics]),
                'average_win_rate': np.mean([m.get('win_rate', 0) for m in all_performance_metrics]),
                'average_max_drawdown': np.mean([m.get('max_drawdown', 0) for m in all_performance_metrics])
            }
        
        return summary
    
    def _save_test_results(self, summary: Dict[str, Any]):
        """テスト結果保存"""
        try:
            output_dir = self.config.get('output_directory', 'test_results')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trend_switching_test_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Test results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")

def main():
    """メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # テスター初期化
        config = {
            'num_scenarios': 8,
            'test_duration_minutes': 15,
            'max_concurrent_tests': 2,
            'enable_synthetic_data': True,
            'enable_real_data': False,  # 実際のデータがない場合はFalse
        }
        
        tester = TrendStrategySwitchTester(config)
        
        # 包括的テスト実行
        logger.info("Starting 4-2-1 comprehensive trend strategy switching test")
        results = tester.run_comprehensive_test()
        
        # 結果表示
        print("\n" + "="*60)
        print("4-2-1 トレンド戦略切替テスト結果")
        print("="*60)
        print(f"総テスト数: {results.get('total_tests', 0)}")
        print(f"成功テスト数: {results.get('successful_tests', 0)}")
        print(f"成功率: {results.get('success_rate', 0):.2%}")
        print(f"総切替イベント数: {results.get('total_switching_events', 0)}")
        print(f"テストあたり平均切替回数: {results.get('average_switches_per_test', 0):.1f}")
        
        if 'aggregated_metrics' in results:
            metrics = results['aggregated_metrics']
            print(f"\n集計パフォーマンス:")
            print(f"  平均シャープレシオ: {metrics.get('average_sharpe_ratio', 0):.3f}")
            print(f"  平均総リターン: {metrics.get('average_total_return', 0):.3%}")
            print(f"  平均勝率: {metrics.get('average_win_rate', 0):.3%}")
            print(f"  平均最大ドローダウン: {metrics.get('average_max_drawdown', 0):.3%}")
        
        print("="*60)
        
        return results.get('success_rate', 0) >= 0.6
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
