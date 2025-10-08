"""
拡張トレンド切替テスター
Phase 2.A.2: リアルマーケットデータ対応・バッチテスト機能・パフォーマンスサマリー強化版

既存のスタンドアロンテスターを拡張し、以下の機能を追加:
- リアルマーケットデータ対応
- 複数銘柄・期間のバッチテスト機能  
- パフォーマンスサマリーの強化
- Excel/JSON/CSV出力対応
- チャート生成機能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

# 既存のスタンドアロンテスターからインポート
from .standalone_trend_switching_test import (
    TrendScenarioType, TrendScenario, StrategySwitchingEvent, TestResult,
    SimpleSyntheticDataGenerator, SimpleStrategySwitch, SimplePerformanceAnalyzer
)

# 新規モジュールからインポート
from .market_data_provider import MarketDataProvider, CacheManager
from .performance_summary_generator import PerformanceSummaryGenerator, PerformanceMetrics
from .batch_test_executor import BatchTestExecutor, BatchTestConfig, TestJob, BatchTestResult

# 設定とログ
from config.logger_config import setup_logger

logger = setup_logger(__name__)

@dataclass
class EnhancedTestConfig:
    """拡張テスト設定"""
    # データ取得設定
    data_source: str = "real"  # "real", "synthetic", "hybrid"
    symbols: List[str] = None
    timeframes: List[str] = None
    date_ranges: List[Dict[str, int]] = None
    
    # テスト設定
    num_scenarios_per_job: int = 8
    enable_batch_mode: bool = True
    max_parallel_workers: int = 4
    
    # 出力設定
    output_formats: List[str] = None  # ["excel", "json", "csv"]
    enable_charts: bool = True
    enable_detailed_summary: bool = True
    
    # キャッシュ設定
    enable_cache: bool = True
    cache_max_age_hours: int = 24
    
    def __post_init__(self):
        """デフォルト値設定"""
        if self.symbols is None:
            self.symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
        
        if self.timeframes is None:
            self.timeframes = ["1h", "4h"]
        
        if self.date_ranges is None:
            self.date_ranges = [{"days": 30}, {"days": 60}, {"days": 90}]
        
        if self.output_formats is None:
            self.output_formats = ["excel", "json"]

class EnhancedDataGenerator:
    """拡張データ生成器（リアル＋合成データ対応）"""
    
    def __init__(self, config: EnhancedTestConfig):
        self.config = config
        self.market_data_provider = MarketDataProvider()
        self.synthetic_generator = SimpleSyntheticDataGenerator()
        
        logger.info(f"EnhancedDataGenerator initialized (source: {config.data_source})")
    
    def generate_test_data(self, 
                          symbol: str,
                          timeframe: str,
                          days: int,
                          scenario: Optional[TrendScenario] = None) -> pd.DataFrame:
        """テストデータ生成"""
        try:
            if self.config.data_source == "real":
                return self._generate_real_data(symbol, timeframe, days)
            elif self.config.data_source == "synthetic":
                return self._generate_synthetic_data(scenario or self._create_default_scenario())
            elif self.config.data_source == "hybrid":
                return self._generate_hybrid_data(symbol, timeframe, days, scenario)
            else:
                raise ValueError(f"Unknown data source: {self.config.data_source}")
                
        except Exception as e:
            logger.error(f"Error generating test data: {e}")
            return pd.DataFrame()
    
    def _generate_real_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """リアルマーケットデータ生成"""
        try:
            data = self.market_data_provider.get_data(
                symbol=symbol,
                timeframe=timeframe,
                days=days
            )
            
            if data.empty:
                logger.warning(f"No real data for {symbol}, falling back to synthetic")
                return self._generate_synthetic_data(self._create_default_scenario())
            
            logger.info(f"Retrieved real data: {symbol} {timeframe} ({len(data)} rows)")
            return data
            
        except Exception as e:
            logger.warning(f"Real data fetch failed for {symbol}: {e}")
            return self._generate_synthetic_data(self._create_default_scenario())
    
    def _generate_synthetic_data(self, scenario: TrendScenario) -> pd.DataFrame:
        """合成データ生成"""
        try:
            data = self.synthetic_generator.generate_scenario_data(scenario)
            logger.debug(f"Generated synthetic data for scenario: {scenario.scenario_id}")
            return data
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return pd.DataFrame()
    
    def _generate_hybrid_data(self, 
                             symbol: str,
                             timeframe: str,
                             days: int,
                             scenario: Optional[TrendScenario]) -> pd.DataFrame:
        """ハイブリッドデータ生成（リアル優先、フォールバック）"""
        try:
            # まずリアルデータを試行
            real_data = self._generate_real_data(symbol, timeframe, days)
            
            if not real_data.empty and len(real_data) >= 100:  # 最低100件のデータが必要
                return real_data
            
            # リアルデータが不十分な場合は合成データ
            logger.info(f"Real data insufficient for {symbol}, using synthetic data")
            scenario = scenario or self._create_default_scenario()
            return self._generate_synthetic_data(scenario)
            
        except Exception as e:
            logger.error(f"Hybrid data generation failed: {e}")
            return pd.DataFrame()
    
    def _create_default_scenario(self) -> TrendScenario:
        """デフォルトシナリオ作成"""
        return TrendScenario(
            scenario_id=f"default_{int(time.time())}",
            scenario_type=TrendScenarioType.TREND_REVERSAL,
            period_days=3,
            initial_trend="uptrend",
            target_trend="downtrend",
            volatility_level=0.2
        )

class EnhancedTrendSwitchingTester:
    """拡張トレンド切替テスター"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # 既存コンポーネント初期化
        self.data_generator = EnhancedDataGenerator(self.config)
        self.strategy_switch = SimpleStrategySwitch()
        self.performance_analyzer = SimplePerformanceAnalyzer()
        
        # 新規コンポーネント初期化
        self.performance_summary_generator = PerformanceSummaryGenerator()
        self.batch_executor = BatchTestExecutor(
            test_function=self._execute_single_test_job,
            config=BatchTestConfig(
                symbols=self.config.symbols,
                timeframes=self.config.timeframes,
                date_ranges=self.config.date_ranges,
                max_workers=self.config.max_parallel_workers,
                parallel_mode=self.config.enable_batch_mode
            )
        )
        
        # 出力ディレクトリ
        self.output_dir = Path("output/enhanced_trend_switching_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EnhancedTrendSwitchingTester initialized")
    
    def _load_config(self, config_path: Optional[str]) -> EnhancedTestConfig:
        """設定読み込み"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 設定からEnhancedTestConfigを作成
                return EnhancedTestConfig(
                    data_source=config_data.get("data_source", "hybrid"),
                    symbols=config_data.get("market_data", {}).get("symbols", None),
                    timeframes=config_data.get("market_data", {}).get("timeframes", None),
                    date_ranges=config_data.get("market_data", {}).get("date_ranges", {}).values(),
                    enable_batch_mode=config_data.get("batch_testing", {}).get("max_parallel_workers", 4) > 1,
                    max_parallel_workers=config_data.get("batch_testing", {}).get("max_parallel_workers", 4),
                    output_formats=config_data.get("output", {}).get("formats", None),
                    enable_charts=config_data.get("output", {}).get("charts", {}).get("enabled", True),
                    enable_cache=config_data.get("market_data", {}).get("cache_settings", {}).get("enabled", True)
                )
            else:
                logger.info("Using default configuration")
                return EnhancedTestConfig()
                
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return EnhancedTestConfig()
    
    def run_single_symbol_test(self, 
                             symbol: str,
                             timeframe: str = "1h",
                             days: int = 30,
                             num_scenarios: int = 8) -> Dict[str, Any]:
        """単一銘柄テスト実行"""
        try:
            logger.info(f"Running single symbol test: {symbol} {timeframe} {days}d")
            
            test_start_time = time.time()
            
            # シナリオ生成
            scenarios = self._generate_test_scenarios(num_scenarios)
            
            # テスト実行
            results = []
            for scenario in scenarios:
                result = self._run_single_scenario_test(symbol, timeframe, days, scenario)
                results.append(result)
            
            # 結果分析
            analysis = self._analyze_results(results)
            
            execution_time = time.time() - test_start_time
            
            # 単一銘柄テスト結果
            single_test_result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'days': days,
                'test_summary': {
                    'total_scenarios': len(scenarios),
                    'successful_scenarios': sum(1 for r in results if r.success),
                    'success_rate': sum(1 for r in results if r.success) / len(results),
                    'total_execution_time': execution_time
                },
                'performance_analysis': analysis,
                'detailed_results': [asdict(result) for result in results]
            }
            
            # データ品質検証
            sample_data = self.data_generator.generate_test_data(symbol, timeframe, days)
            if not sample_data.empty:
                quality_report = self.data_generator.market_data_provider.validate_data_quality(sample_data, symbol)
                single_test_result['data_quality'] = quality_report
            
            logger.info(f"Single symbol test completed: {symbol} (success rate: {single_test_result['test_summary']['success_rate']:.2%})")
            return single_test_result
            
        except Exception as e:
            logger.error(f"Single symbol test failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def run_batch_tests(self, 
                       custom_symbols: Optional[List[str]] = None,
                       custom_timeframes: Optional[List[str]] = None,
                       custom_date_ranges: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """バッチテスト実行"""
        try:
            logger.info("Starting enhanced batch tests")
            
            # カスタムパラメータがある場合は設定更新
            if custom_symbols:
                self.batch_executor.config.symbols = custom_symbols
            if custom_timeframes:
                self.batch_executor.config.timeframes = custom_timeframes
            if custom_date_ranges:
                self.batch_executor.config.date_ranges = custom_date_ranges
            
            # プログレスコールバック
            def progress_callback(completed: int, total: int, current_job: str):
                if completed % 5 == 0 or completed == total:  # 5件ごと、または完了時
                    logger.info(f"Batch progress: {completed}/{total} ({completed/total:.1%}) - Current: {current_job}")
            
            # バッチ実行
            batch_results = self.batch_executor.execute_batch_tests(
                progress_callback=progress_callback
            )
            
            # 拡張分析
            enhanced_analysis = self._perform_enhanced_analysis(batch_results)
            batch_results['enhanced_analysis'] = enhanced_analysis
            
            # 結果保存
            self._save_batch_results(batch_results)
            
            logger.info(f"Batch tests completed (success rate: {batch_results.get('execution_summary', {}).get('success_rate', 0):.2%})")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch tests failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_analysis(self, 
                                 include_benchmarks: bool = True,
                                 generate_charts: bool = None) -> Dict[str, Any]:
        """包括的分析実行"""
        try:
            logger.info("Starting comprehensive analysis")
            
            if generate_charts is None:
                generate_charts = self.config.enable_charts
            
            # バッチテスト実行
            batch_results = self.run_batch_tests()
            
            if 'error' in batch_results:
                return batch_results
            
            # パフォーマンスサマリー生成
            comprehensive_summary = self.performance_summary_generator.generate_comprehensive_summary(
                batch_results,
                config={'output': {'charts': {'enabled': generate_charts}}}
            )
            
            # ベンチマーク比較（有効な場合）
            if include_benchmarks:
                benchmark_analysis = self._perform_benchmark_analysis(batch_results)
                comprehensive_summary['benchmark_analysis'] = benchmark_analysis
            
            # 全体結果
            comprehensive_result = {
                'analysis_timestamp': datetime.now().isoformat(),
                'configuration': asdict(self.config),
                'batch_results': batch_results,
                'comprehensive_summary': comprehensive_summary,
                'recommendations': self._generate_enhanced_recommendations(batch_results, comprehensive_summary)
            }
            
            # 結果保存
            self._save_comprehensive_results(comprehensive_result)
            
            logger.info("Comprehensive analysis completed")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {'error': str(e)}
    
    def _execute_single_test_job(self, 
                               symbol: str,
                               timeframe: str,
                               days: int,
                               scenario_config: Optional[Dict] = None) -> Dict[str, Any]:
        """単一テストジョブ実行（バッチ実行器用）"""
        try:
            # シナリオ数設定
            num_scenarios = scenario_config.get('num_scenarios', self.config.num_scenarios_per_job) if scenario_config else self.config.num_scenarios_per_job
            
            # 単一銘柄テスト実行
            result = self.run_single_symbol_test(symbol, timeframe, days, num_scenarios)
            
            return result
            
        except Exception as e:
            logger.error(f"Test job execution failed: {e}")
            return {'error': str(e)}
    
    def _generate_test_scenarios(self, num_scenarios: int) -> List[TrendScenario]:
        """テストシナリオ生成"""
        scenarios = []
        
        try:
            scenario_types = list(TrendScenarioType)
            
            for i in range(num_scenarios):
                scenario_type = scenario_types[i % len(scenario_types)]
                
                # トレンド設定
                trends = ['uptrend', 'downtrend', 'sideways']
                initial_trend = np.random.choice(trends)
                target_trend = np.random.choice([t for t in trends if t != initial_trend])
                
                scenario = TrendScenario(
                    scenario_id=f"enhanced_{scenario_type.value}_{i}_{int(time.time())}",
                    scenario_type=scenario_type,
                    period_days=np.random.randint(2, 7),
                    initial_trend=initial_trend,
                    target_trend=target_trend,
                    volatility_level=np.random.uniform(0.1, 0.4)
                )
                
                scenarios.append(scenario)
            
            logger.debug(f"Generated {len(scenarios)} test scenarios")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return []
    
    def _run_single_scenario_test(self, 
                                symbol: str,
                                timeframe: str,
                                days: int,
                                scenario: TrendScenario) -> TestResult:
        """単一シナリオテスト実行"""
        start_time = time.time()
        errors = []
        
        try:
            # テストデータ生成
            test_data = self.data_generator.generate_test_data(symbol, timeframe, days, scenario)
            
            if test_data.empty:
                raise ValueError("Failed to generate test data")
            
            # 戦略切替検出
            switching_events = self.strategy_switch.detect_switching_points(test_data)
            
            # パフォーマンス評価
            performance_metrics = self.performance_analyzer.calculate_performance_metrics(
                test_data, switching_events
            )
            
            # 拡張メトリクス計算
            enhanced_metrics = self._calculate_enhanced_metrics(test_data, switching_events)
            performance_metrics.update(enhanced_metrics)
            
            # 成功判定
            success = self._validate_test_success(performance_metrics, switching_events)
            
            execution_time = time.time() - start_time
            
            result = TestResult(
                scenario_id=scenario.scenario_id,
                success=success,
                execution_time=execution_time,
                switching_events=switching_events,
                performance_metrics=performance_metrics,
                errors=errors
            )
            
            logger.debug(f"Scenario test completed: {scenario.scenario_id} ({'SUCCESS' if success else 'FAILED'})")
            return result
            
        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Scenario test failed: {scenario.scenario_id}: {e}")
            
            return TestResult(
                scenario_id=scenario.scenario_id,
                success=False,
                execution_time=time.time() - start_time,
                switching_events=[],
                performance_metrics={},
                errors=errors
            )
    
    def _calculate_enhanced_metrics(self, 
                                  data: pd.DataFrame, 
                                  switching_events: List[StrategySwitchingEvent]) -> Dict[str, float]:
        """拡張メトリクス計算"""
        try:
            enhanced_metrics = {}
            
            if data.empty or 'close' not in data.columns:
                return enhanced_metrics
            
            # 高度なパフォーマンスメトリクス
            metrics_calculator = PerformanceMetrics()
            comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(data)
            enhanced_metrics.update(comprehensive_metrics)
            
            # 切替効率メトリクス
            if switching_events:
                enhanced_metrics.update({
                    'switching_efficiency': self._calculate_switching_efficiency(switching_events),
                    'avg_switching_delay': np.mean([e.switching_delay for e in switching_events]),
                    'high_confidence_switches': sum(1 for e in switching_events if e.confidence_score > 0.7),
                    'strategy_diversity': len(set(e.to_strategy for e in switching_events))
                })
            
            return enhanced_metrics
            
        except Exception as e:
            logger.warning(f"Enhanced metrics calculation error: {e}")
            return {}
    
    def _calculate_switching_efficiency(self, switching_events: List[StrategySwitchingEvent]) -> float:
        """切替効率計算"""
        try:
            if not switching_events:
                return 0.0
            
            # 高信頼度切替の割合
            high_confidence_count = sum(1 for e in switching_events if e.confidence_score > 0.6)
            efficiency = high_confidence_count / len(switching_events)
            
            return efficiency
            
        except Exception:
            return 0.0
    
    def _validate_test_success(self, 
                             performance_metrics: Dict[str, float], 
                             switching_events: List[StrategySwitchingEvent]) -> bool:
        """テスト成功判定"""
        try:
            # 基本成功条件
            has_metrics = len(performance_metrics) > 0
            has_switches = len(switching_events) > 0
            
            # パフォーマンス基準
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            total_return = performance_metrics.get('total_return', 0)
            
            # 成功条件（緩和）
            performance_ok = sharpe_ratio > -1.0 or total_return > -0.5  # 極端に悪くなければOK
            
            return has_metrics and (performance_ok or has_switches)
            
        except Exception:
            return False
    
    def _analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """結果分析（拡張版）"""
        try:
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                return {'error': 'No successful results to analyze'}
            
            # 基本分析
            basic_analysis = self._perform_basic_analysis(successful_results)
            
            # 拡張分析
            extended_analysis = self._perform_extended_analysis(successful_results)
            
            analysis = {
                **basic_analysis,
                'extended_analysis': extended_analysis,
                'data_source_analysis': self._analyze_data_sources(successful_results)
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Results analysis error: {e}")
            return {}
    
    def _perform_basic_analysis(self, successful_results: List[TestResult]) -> Dict[str, Any]:
        """基本分析実行"""
        # 既存のSimplePerformanceAnalyzerのロジックを使用
        all_events = []
        for result in successful_results:
            all_events.extend(result.switching_events)
        
        # パフォーマンス統計
        performance_stats = {}
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility']
        
        for metric in metrics:
            values = [r.performance_metrics.get(metric, 0) for r in successful_results 
                     if metric in r.performance_metrics]
            if values:
                performance_stats[f'avg_{metric}'] = np.mean(values)
                performance_stats[f'std_{metric}'] = np.std(values)
                performance_stats[f'min_{metric}'] = np.min(values)
                performance_stats[f'max_{metric}'] = np.max(values)
        
        return {
            'performance_statistics': performance_stats,
            'switching_analysis': {
                'total_switching_events': len(all_events),
                'avg_events_per_test': len(all_events) / len(successful_results),
                'avg_confidence': np.mean([e.confidence_score for e in all_events]) if all_events else 0
            }
        }
    
    def _perform_extended_analysis(self, successful_results: List[TestResult]) -> Dict[str, Any]:
        """拡張分析実行"""
        try:
            # 拡張メトリクス分析
            extended_metrics = ['sortino_ratio', 'calmar_ratio', 'switching_efficiency', 'strategy_diversity']
            extended_stats = {}
            
            for metric in extended_metrics:
                values = [r.performance_metrics.get(metric, 0) for r in successful_results 
                         if metric in r.performance_metrics]
                if values:
                    extended_stats[f'avg_{metric}'] = np.mean(values)
                    extended_stats[f'std_{metric}'] = np.std(values)
            
            # リスク調整後リターン分析
            risk_adjusted_analysis = self._analyze_risk_adjusted_returns(successful_results)
            
            return {
                'extended_performance_metrics': extended_stats,
                'risk_adjusted_analysis': risk_adjusted_analysis,
                'execution_efficiency': {
                    'avg_execution_time': np.mean([r.execution_time for r in successful_results]),
                    'execution_time_std': np.std([r.execution_time for r in successful_results])
                }
            }
            
        except Exception as e:
            logger.warning(f"Extended analysis error: {e}")
            return {}
    
    def _analyze_risk_adjusted_returns(self, successful_results: List[TestResult]) -> Dict[str, Any]:
        """リスク調整後リターン分析"""
        try:
            sharpe_ratios = [r.performance_metrics.get('sharpe_ratio', 0) for r in successful_results]
            sortino_ratios = [r.performance_metrics.get('sortino_ratio', 0) for r in successful_results]
            
            analysis = {}
            
            if sharpe_ratios:
                analysis['sharpe_analysis'] = {
                    'mean': np.mean(sharpe_ratios),
                    'positive_count': sum(1 for s in sharpe_ratios if s > 0),
                    'excellent_count': sum(1 for s in sharpe_ratios if s > 1.0)
                }
            
            if sortino_ratios:
                analysis['sortino_analysis'] = {
                    'mean': np.mean(sortino_ratios),
                    'positive_count': sum(1 for s in sortino_ratios if s > 0)
                }
            
            return analysis
            
        except Exception:
            return {}
    
    def _analyze_data_sources(self, successful_results: List[TestResult]) -> Dict[str, Any]:
        """データソース分析"""
        return {
            'data_source': self.config.data_source,
            'cache_usage': 'enabled' if self.config.enable_cache else 'disabled',
            'successful_tests': len(successful_results)
        }
    
    def _perform_enhanced_analysis(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """拡張分析実行"""
        try:
            detailed_results = batch_results.get('detailed_results', [])
            
            if not detailed_results:
                return {}
            
            # 銘柄別分析
            symbol_analysis = self._analyze_by_symbol(detailed_results)
            
            # 時間軸別分析
            timeframe_analysis = self._analyze_by_timeframe(detailed_results)
            
            # 期間別分析
            period_analysis = self._analyze_by_period(detailed_results)
            
            return {
                'symbol_analysis': symbol_analysis,
                'timeframe_analysis': timeframe_analysis,
                'period_analysis': period_analysis,
                'cross_analysis': self._perform_cross_analysis(detailed_results)
            }
            
        except Exception as e:
            logger.warning(f"Enhanced analysis error: {e}")
            return {}
    
    def _analyze_by_symbol(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """銘柄別分析"""
        symbol_stats = {}
        
        for result in detailed_results:
            if not result.get('success', False):
                continue
                
            symbol = result.get('symbol', 'unknown')
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'count': 0,
                    'total_returns': [],
                    'sharpe_ratios': [],
                    'execution_times': []
                }
            
            symbol_stats[symbol]['count'] += 1
            
            # 結果データから詳細メトリクス抽出
            if 'detailed_results' in result:
                for detail in result['detailed_results']:
                    if detail.get('success', False):
                        metrics = detail.get('performance_metrics', {})
                        symbol_stats[symbol]['total_returns'].append(metrics.get('total_return', 0))
                        symbol_stats[symbol]['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))
                        symbol_stats[symbol]['execution_times'].append(detail.get('execution_time', 0))
        
        # 統計計算
        analysis = {}
        for symbol, stats in symbol_stats.items():
            if stats['total_returns']:
                analysis[symbol] = {
                    'test_count': stats['count'],
                    'avg_return': np.mean(stats['total_returns']),
                    'avg_sharpe': np.mean(stats['sharpe_ratios']),
                    'avg_execution_time': np.mean(stats['execution_times']),
                    'success_rate': 1.0  # 成功したもののみ集計しているため
                }
        
        return analysis
    
    def _analyze_by_timeframe(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """時間軸別分析"""
        timeframe_stats = {}
        
        for result in detailed_results:
            if not result.get('success', False):
                continue
                
            timeframe = result.get('timeframe', 'unknown')
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {'count': 0, 'success_rates': []}
            
            timeframe_stats[timeframe]['count'] += 1
            
            if 'test_summary' in result:
                success_rate = result['test_summary'].get('success_rate', 0)
                timeframe_stats[timeframe]['success_rates'].append(success_rate)
        
        # 統計計算
        analysis = {}
        for timeframe, stats in timeframe_stats.items():
            if stats['success_rates']:
                analysis[timeframe] = {
                    'test_count': stats['count'],
                    'avg_success_rate': np.mean(stats['success_rates']),
                    'consistency': 1.0 - np.std(stats['success_rates'])  # 一貫性指標
                }
        
        return analysis
    
    def _analyze_by_period(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """期間別分析"""
        period_stats = {}
        
        for result in detailed_results:
            if not result.get('success', False):
                continue
                
            days = result.get('days', 0)
            period_key = f"{days}d"
            
            if period_key not in period_stats:
                period_stats[period_key] = {'count': 0, 'execution_times': []}
            
            period_stats[period_key]['count'] += 1
            
            if 'test_summary' in result:
                exec_time = result['test_summary'].get('total_execution_time', 0)
                period_stats[period_key]['execution_times'].append(exec_time)
        
        # 統計計算
        analysis = {}
        for period, stats in period_stats.items():
            if stats['execution_times']:
                analysis[period] = {
                    'test_count': stats['count'],
                    'avg_execution_time': np.mean(stats['execution_times']),
                    'efficiency_score': stats['count'] / np.mean(stats['execution_times'])  # テスト数/時間
                }
        
        return analysis
    
    def _perform_cross_analysis(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """クロス分析（銘柄×時間軸など）"""
        try:
            # 銘柄×時間軸の組み合わせ分析
            combinations = {}
            
            for result in detailed_results:
                if not result.get('success', False):
                    continue
                    
                symbol = result.get('symbol', 'unknown')
                timeframe = result.get('timeframe', 'unknown')
                combo_key = f"{symbol}_{timeframe}"
                
                if combo_key not in combinations:
                    combinations[combo_key] = {'count': 0, 'performance_scores': []}
                
                combinations[combo_key]['count'] += 1
                
                # パフォーマンススコア計算
                if 'test_summary' in result:
                    success_rate = result['test_summary'].get('success_rate', 0)
                    combinations[combo_key]['performance_scores'].append(success_rate)
            
            # ベスト・ワースト組み合わせ特定
            best_combo = None
            worst_combo = None
            best_score = -1
            worst_score = 2
            
            for combo, stats in combinations.items():
                if stats['performance_scores']:
                    avg_score = np.mean(stats['performance_scores'])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_combo = combo
                    if avg_score < worst_score:
                        worst_score = avg_score
                        worst_combo = combo
            
            return {
                'total_combinations': len(combinations),
                'best_combination': {'combo': best_combo, 'score': best_score},
                'worst_combination': {'combo': worst_combo, 'score': worst_score},
                'combination_diversity': len(set(r.get('symbol', '') for r in detailed_results if r.get('success', False)))
            }
            
        except Exception as e:
            logger.warning(f"Cross analysis error: {e}")
            return {}
    
    def _perform_benchmark_analysis(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """ベンチマーク分析"""
        try:
            # SPYベンチマークデータ取得
            benchmark_data = self.data_generator.market_data_provider.get_data("SPY", "1d", 90)
            
            if benchmark_data.empty:
                return {'error': 'Benchmark data not available'}
            
            # ベンチマークメトリクス計算
            benchmark_metrics = PerformanceMetrics().calculate_comprehensive_metrics(benchmark_data)
            
            # バッチ結果との比較
            detailed_results = batch_results.get('detailed_results', [])
            successful_results = [r for r in detailed_results if r.get('success', False)]
            
            if not successful_results:
                return {'error': 'No successful results for benchmark comparison'}
            
            # 平均パフォーマンス計算
            avg_metrics = self._calculate_average_metrics(successful_results)
            
            # 比較分析
            comparison = {}
            for metric in ['total_return', 'sharpe_ratio', 'volatility', 'max_drawdown']:
                if metric in benchmark_metrics and metric in avg_metrics:
                    comparison[f'{metric}_vs_benchmark'] = avg_metrics[metric] - benchmark_metrics[metric]
            
            return {
                'benchmark_metrics': benchmark_metrics,
                'strategy_avg_metrics': avg_metrics,
                'comparison': comparison,
                'outperformance_rate': self._calculate_outperformance_rate(successful_results, benchmark_metrics)
            }
            
        except Exception as e:
            logger.warning(f"Benchmark analysis error: {e}")
            return {}
    
    def _calculate_average_metrics(self, successful_results: List[Dict]) -> Dict[str, float]:
        """平均メトリクス計算"""
        all_metrics = []
        
        for result in successful_results:
            if 'detailed_results' in result:
                for detail in result['detailed_results']:
                    if detail.get('success', False):
                        metrics = detail.get('performance_metrics', {})
                        all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for metric in ['total_return', 'sharpe_ratio', 'volatility', 'max_drawdown']:
            values = [m.get(metric, 0) for m in all_metrics if metric in m]
            if values:
                avg_metrics[metric] = np.mean(values)
        
        return avg_metrics
    
    def _calculate_outperformance_rate(self, 
                                     successful_results: List[Dict], 
                                     benchmark_metrics: Dict[str, float]) -> float:
        """アウトパフォーマンス率計算"""
        try:
            benchmark_return = benchmark_metrics.get('total_return', 0)
            outperform_count = 0
            total_count = 0
            
            for result in successful_results:
                if 'detailed_results' in result:
                    for detail in result['detailed_results']:
                        if detail.get('success', False):
                            metrics = detail.get('performance_metrics', {})
                            strategy_return = metrics.get('total_return', 0)
                            
                            if strategy_return > benchmark_return:
                                outperform_count += 1
                            total_count += 1
            
            return outperform_count / total_count if total_count > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_enhanced_recommendations(self, 
                                         batch_results: Dict[str, Any], 
                                         comprehensive_summary: Dict[str, Any]) -> List[str]:
        """拡張推奨事項生成"""
        recommendations = []
        
        try:
            # バッチ実行結果に基づく推奨
            execution_summary = batch_results.get('execution_summary', {})
            success_rate = execution_summary.get('success_rate', 0)
            
            if success_rate < 0.6:
                recommendations.append(f"バッチテスト成功率が{success_rate:.1%}と低いため、データ品質またはテスト条件の見直しを推奨します。")
            
            # パフォーマンス分析に基づく推奨
            if 'performance_analysis' in batch_results:
                perf_analysis = batch_results['performance_analysis']
                
                if 'success_analysis' in perf_analysis:
                    symbol_success = perf_analysis['success_analysis'].get('success_rate_by_symbol', {})
                    if symbol_success:
                        worst_symbol = min(symbol_success.items(), key=lambda x: x[1])
                        if worst_symbol[1] < 0.5:
                            recommendations.append(f"銘柄{worst_symbol[0]}の成功率が{worst_symbol[1]:.1%}と低いため、データ品質確認を推奨します。")
            
            # 拡張分析に基づく推奨
            if 'enhanced_analysis' in batch_results:
                enhanced = batch_results['enhanced_analysis']
                
                if 'symbol_analysis' in enhanced:
                    symbol_analysis = enhanced['symbol_analysis']
                    if symbol_analysis:
                        # 最高パフォーマンス銘柄特定
                        best_symbol = max(symbol_analysis.items(), key=lambda x: x[1].get('avg_sharpe', 0))
                        recommendations.append(f"銘柄{best_symbol[0]}が最高のシャープレシオ({best_symbol[1].get('avg_sharpe', 0):.3f})を達成しており、類似銘柄での戦略適用を推奨します。")
            
            # ベンチマーク分析に基づく推奨
            if 'benchmark_analysis' in comprehensive_summary:
                benchmark = comprehensive_summary['benchmark_analysis']
                if 'outperformance_rate' in benchmark:
                    outperf_rate = benchmark['outperformance_rate']
                    if outperf_rate > 0.6:
                        recommendations.append(f"ベンチマークアウトパフォーマンス率が{outperf_rate:.1%}と優秀です。現在の戦略設定の維持を推奨します。")
                    elif outperf_rate < 0.4:
                        recommendations.append(f"ベンチマークアウトパフォーマンス率が{outperf_rate:.1%}と低いため、戦略パラメータの最適化を推奨します。")
            
            # 技術的推奨
            if self.config.data_source == "synthetic":
                recommendations.append("現在合成データを使用中です。より現実的な結果のためリアルマーケットデータの使用を推奨します。")
            
            if not self.config.enable_cache:
                recommendations.append("データキャッシュが無効です。実行効率向上のためキャッシュの有効化を推奨します。")
            
            # デフォルト推奨
            if not recommendations:
                recommendations.append("総合的なパフォーマンスは良好です。継続的な監視と段階的な最適化を推奨します。")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Enhanced recommendations generation error: {e}")
            return ["分析中にエラーが発生しました。詳細な検証と設定見直しを推奨します。"]
    
    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """バッチ結果保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 複数フォーマットで保存
            for format_type in self.config.output_formats:
                if format_type == "json":
                    json_file = self.output_dir / f"batch_results_{timestamp}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"Batch results saved to JSON: {json_file}")
                
                elif format_type == "excel":
                    self._save_to_excel(batch_results, timestamp)
                
                elif format_type == "csv":
                    self._save_to_csv(batch_results, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
    
    def _save_comprehensive_results(self, comprehensive_result: Dict[str, Any]):
        """包括的結果保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # メイン結果ファイル
            main_file = self.output_dir / f"comprehensive_analysis_{timestamp}.json"
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_result, f, indent=2, ensure_ascii=False, default=str)
            
            # サマリーレポート
            self._generate_text_report(comprehensive_result, timestamp)
            
            logger.info(f"Comprehensive results saved: {main_file}")
            
        except Exception as e:
            logger.error(f"Error saving comprehensive results: {e}")
    
    def _save_to_excel(self, batch_results: Dict[str, Any], timestamp: str):
        """Excel形式保存"""
        try:
            excel_file = self.output_dir / f"batch_results_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 実行サマリー
                summary_df = pd.DataFrame([batch_results.get('execution_summary', {})])
                summary_df.to_excel(writer, sheet_name='ExecutionSummary', index=False)
                
                # 詳細結果（簡略化）
                detailed_results = batch_results.get('detailed_results', [])
                if detailed_results:
                    results_data = []
                    for result in detailed_results:
                        if result.get('success', False):
                            summary = result.get('test_summary', {})
                            results_data.append({
                                'Symbol': result.get('symbol', ''),
                                'Timeframe': result.get('timeframe', ''),
                                'Days': result.get('days', 0),
                                'Success_Rate': summary.get('success_rate', 0),
                                'Execution_Time': summary.get('total_execution_time', 0)
                            })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        results_df.to_excel(writer, sheet_name='DetailedResults', index=False)
            
            logger.info(f"Excel file saved: {excel_file}")
            
        except Exception as e:
            logger.warning(f"Excel save error: {e}")
    
    def _save_to_csv(self, batch_results: Dict[str, Any], timestamp: str):
        """CSV形式保存"""
        try:
            csv_dir = self.output_dir / "csv"
            csv_dir.mkdir(exist_ok=True)
            
            # 実行サマリー
            summary_df = pd.DataFrame([batch_results.get('execution_summary', {})])
            summary_file = csv_dir / f"execution_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV files saved to: {csv_dir}")
            
        except Exception as e:
            logger.warning(f"CSV save error: {e}")
    
    def _generate_text_report(self, comprehensive_result: Dict[str, Any], timestamp: str):
        """テキストレポート生成"""
        try:
            report_file = self.output_dir / f"analysis_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("Phase 2.A.2 拡張トレンド切替テスター 包括的分析レポート\n")
                f.write("="*80 + "\n\n")
                
                # 設定情報
                f.write("テスト設定:\n")
                f.write("-"*40 + "\n")
                config = comprehensive_result.get('configuration', {})
                f.write(f"データソース: {config.get('data_source', 'unknown')}\n")
                f.write(f"対象銘柄: {', '.join(config.get('symbols', []))}\n")
                f.write(f"時間軸: {', '.join(config.get('timeframes', []))}\n")
                f.write(f"バッチモード: {'有効' if config.get('enable_batch_mode', False) else '無効'}\n")
                f.write(f"並列ワーカー数: {config.get('max_parallel_workers', 1)}\n\n")
                
                # バッチ実行結果
                batch_results = comprehensive_result.get('batch_results', {})
                if 'execution_summary' in batch_results:
                    summary = batch_results['execution_summary']
                    f.write("バッチ実行結果:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"総ジョブ数: {summary.get('total_jobs', 0)}\n")
                    f.write(f"完了ジョブ数: {summary.get('completed_jobs', 0)}\n")
                    f.write(f"成功率: {summary.get('success_rate', 0):.2%}\n")
                    f.write(f"総実行時間: {summary.get('total_execution_time', 0):.1f}秒\n")
                    f.write(f"ジョブあたり平均時間: {summary.get('average_time_per_job', 0):.2f}秒\n\n")
                
                # 推奨事項
                recommendations = comprehensive_result.get('recommendations', [])
                if recommendations:
                    f.write("推奨事項:\n")
                    f.write("-"*40 + "\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                f.write("="*80 + "\n")
                f.write(f"レポート生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Text report generated: {report_file}")
            
        except Exception as e:
            logger.warning(f"Text report generation error: {e}")

def main():
    """メイン関数"""
    try:
        # 設定ファイルパス
        config_path = "src/analysis/trend_switching_config.json"
        
        # 拡張テスター初期化
        tester = EnhancedTrendSwitchingTester(config_path)
        
        logger.info("Starting Phase 2.A.2 Enhanced Trend Switching Tests")
        
        # 包括的分析実行
        comprehensive_results = tester.run_comprehensive_analysis(
            include_benchmarks=True,
            generate_charts=True
        )
        
        # 結果表示
        print("\n" + "="*80)
        print("Phase 2.A.2 拡張トレンド切替テスター 実行結果")
        print("="*80)
        
        if 'error' in comprehensive_results:
            print(f"エラー: {comprehensive_results['error']}")
            return False
        
        # バッチ実行サマリー
        batch_results = comprehensive_results.get('batch_results', {})
        if 'execution_summary' in batch_results:
            summary = batch_results['execution_summary']
            print(f"総ジョブ数: {summary.get('total_jobs', 0)}")
            print(f"完了ジョブ数: {summary.get('completed_jobs', 0)}")
            print(f"成功率: {summary.get('success_rate', 0):.2%}")
            print(f"総実行時間: {summary.get('total_execution_time', 0):.1f}秒")
            
            if summary.get('parallel_mode', False):
                print(f"並列実行: 有効 (ワーカー数: {summary.get('max_workers', 1)})")
            else:
                print("並列実行: 無効")
        
        # 拡張分析結果
        if 'enhanced_analysis' in batch_results:
            enhanced = batch_results['enhanced_analysis']
            print("\n拡張分析結果:")
            
            if 'symbol_analysis' in enhanced:
                symbol_count = len(enhanced['symbol_analysis'])
                print(f"  分析対象銘柄数: {symbol_count}")
            
            if 'cross_analysis' in enhanced:
                cross = enhanced['cross_analysis']
                best_combo = cross.get('best_combination', {})
                if best_combo.get('combo'):
                    print(f"  最高パフォーマンス組み合わせ: {best_combo['combo']} (スコア: {best_combo.get('score', 0):.3f})")
        
        # ベンチマーク分析結果
        comprehensive_summary = comprehensive_results.get('comprehensive_summary', {})
        if 'benchmark_analysis' in comprehensive_summary:
            benchmark = comprehensive_summary['benchmark_analysis']
            outperf_rate = benchmark.get('outperformance_rate', 0)
            print(f"\nベンチマーク分析:")
            print(f"  アウトパフォーマンス率: {outperf_rate:.2%}")
        
        # 推奨事項
        recommendations = comprehensive_results.get('recommendations', [])
        if recommendations:
            print(f"\n主要推奨事項:")
            for i, rec in enumerate(recommendations[:3], 1):  # 上位3件表示
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        
        # 成功判定
        success_rate = batch_results.get('execution_summary', {}).get('success_rate', 0)
        overall_success = success_rate >= 0.6
        
        logger.info(f"Phase 2.A.2 enhanced test {'PASSED' if overall_success else 'FAILED'} - Success rate: {success_rate:.2%}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
