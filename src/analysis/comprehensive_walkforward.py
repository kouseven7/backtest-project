"""
統合ウォークフォワードテストシステム
フェーズ2：包括的なパフォーマンス検証システム

多戦略・多シンボル・多市場環境での統合ウォークフォワードテストを実行し、
市場分類システム・戦略スコアリング・パフォーマンス最適化を統合します。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import sys
import os
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
import json
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 必要なインポート
import data_fetcher
import data_processor
from src.config.logger_config import setup_logger

# 既存のコンポーネントをインポート
try:
    from src.analysis.walkforward_executor import WalkforwardExecutor
    from src.analysis.walkforward_scenarios import WalkforwardScenarios
    from src.analysis.walkforward_result_analyzer import WalkforwardResultAnalyzer
except ImportError as e:
    logging.warning(f"既存ウォークフォワードコンポーネントの一部が利用できません: {e}")

try:
    from src.analysis.market_classification.a_b_market_classifier import ABMarketClassifier
    from src.analysis.market_classification.enhanced_market_detector import EnhancedMarketDetector
except ImportError as e:
    logging.warning(f"市場分類システムの一部が利用できません: {e}")

try:
    from src.strategies.strategy_scoring_model import StrategyScoring
except ImportError as e:
    logging.warning(f"戦略スコアリングが利用できません: {e}")

# 戦略のインポート
AVAILABLE_STRATEGIES = {}
strategy_imports = [
    ("src.strategies.vwap_breakout_strategy", "VWAPBreakoutStrategy"),
    ("src.strategies.vwap_bounce_strategy", "VWAPBounceStrategy"),
    ("src.strategies.breakout_strategy", "BreakoutStrategy"),
    ("src.strategies.gc_strategy", "GCStrategy"),
    ("src.strategies.momentum_investing_strategy", "MomentumInvestingStrategy"),
]

for module_name, class_name in strategy_imports:
    try:
        module = __import__(module_name, fromlist=[class_name])
        AVAILABLE_STRATEGIES[class_name] = getattr(module, class_name)
    except ImportError:
        pass

class ProcessingMode(Enum):
    """処理モード"""
    SEQUENTIAL = "sequential"
    PARALLEL_THREAD = "parallel_thread"
    PARALLEL_PROCESS = "parallel_process"
    HYBRID = "hybrid"
    AUTO = "auto"

class TestStatus(Enum):
    """テスト状況"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TestConfiguration:
    """テスト設定"""
    symbols: List[str]
    strategies: List[str]
    start_date: str
    end_date: str
    window_size_days: int = 252
    step_size_days: int = 21
    min_data_points: int = 100
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    enable_market_classification: bool = True
    enable_strategy_scoring: bool = True
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10
    timeout_seconds: int = 3600
    save_intermediate: bool = True
    output_directory: str = "output/comprehensive_walkforward"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['processing_mode'] = self.processing_mode.value
        return result

@dataclass
class TestProgress:
    """テスト進捗"""
    total_tests: int = 0
    completed_tests: int = 0
    failed_tests: int = 0
    current_test: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    
    @property
    def completion_rate(self) -> float:
        """完了率"""
        if self.total_tests == 0:
            return 0.0
        return self.completed_tests / self.total_tests
    
    @property
    def failure_rate(self) -> float:
        """失敗率"""
        if self.total_tests == 0:
            return 0.0
        return self.failed_tests / self.total_tests

class ComprehensiveWalkforwardTester:
    """統合ウォークフォワードテストシステム"""
    
    def __init__(self, config: TestConfiguration):
        """
        初期化
        
        Args:
            config: テスト設定
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.progress = TestProgress()
        
        # 出力ディレクトリの作成
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果保存用
        self.all_results = []
        self.aggregated_results = {}
        self.market_classification_results = {}
        self.strategy_scoring_results = {}
        
        # 既存システムとの統合
        self._initialize_existing_systems()
        
        # 処理モードの決定
        self._determine_processing_mode()
        
        self.logger.info(f"統合ウォークフォワードテスト初期化完了")
        self.logger.info(f"設定: {config.to_dict()}")

    def _initialize_existing_systems(self):
        """既存システムとの統合初期化"""
        try:
            # 既存のウォークフォワードシステム
            self.scenarios = WalkforwardScenarios()
            self.walkforward_executor = WalkforwardExecutor(self.scenarios)
            self.result_analyzer = WalkforwardResultAnalyzer()
            self.logger.info("既存ウォークフォワードシステムとの統合完了")
        except Exception as e:
            self.logger.warning(f"既存ウォークフォワードシステムとの統合に失敗: {e}")
            self.walkforward_executor = None
            self.result_analyzer = None
        
        # 市場分類システム
        try:
            self.market_classifier = ABMarketClassifier()
            self.market_detector = EnhancedMarketDetector()
            self.logger.info("市場分類システムとの統合完了")
        except Exception as e:
            self.logger.warning(f"市場分類システムとの統合に失敗: {e}")
            self.market_classifier = None
            self.market_detector = None
        
        # 戦略スコアリングシステム
        try:
            self.strategy_scorer = StrategyScoring()
            self.logger.info("戦略スコアリングシステムとの統合完了")
        except Exception as e:
            self.logger.warning(f"戦略スコアリングシステムとの統合に失敗: {e}")
            self.strategy_scorer = None

    def _determine_processing_mode(self):
        """処理モードの自動決定"""
        if self.config.processing_mode == ProcessingMode.AUTO:
            total_combinations = len(self.config.symbols) * len(self.config.strategies)
            
            if total_combinations < 10:
                self.processing_mode = ProcessingMode.SEQUENTIAL
            elif total_combinations < 100:
                self.processing_mode = ProcessingMode.PARALLEL_THREAD
            else:
                self.processing_mode = ProcessingMode.PARALLEL_PROCESS
        else:
            self.processing_mode = self.config.processing_mode
        
        self.logger.info(f"処理モード決定: {self.processing_mode.value}")

    def execute_comprehensive_test(self) -> Dict[str, Any]:
        """統合ウォークフォワードテストの実行"""
        self.logger.info("統合ウォークフォワードテスト開始")
        self.progress.status = TestStatus.RUNNING
        self.progress.start_time = datetime.now()
        
        try:
            # テスト組み合わせの生成
            test_combinations = self._generate_test_combinations()
            self.progress.total_tests = len(test_combinations)
            
            self.logger.info(f"総テスト数: {self.progress.total_tests}")
            
            # 処理モードに応じた実行
            if self.processing_mode == ProcessingMode.SEQUENTIAL:
                results = self._execute_sequential(test_combinations)
            elif self.processing_mode in [ProcessingMode.PARALLEL_THREAD, ProcessingMode.PARALLEL_PROCESS]:
                results = self._execute_parallel(test_combinations)
            elif self.processing_mode == ProcessingMode.HYBRID:
                results = self._execute_hybrid(test_combinations)
            else:
                raise ValueError(f"サポートされていない処理モード: {self.processing_mode}")
            
            # 結果の集約
            aggregated_results = self._aggregate_results(results)
            
            # 市場分類統合
            if self.config.enable_market_classification:
                market_results = self._integrate_market_classification(results)
                aggregated_results['market_classification'] = market_results
            
            # 戦略スコアリング統合
            if self.config.enable_strategy_scoring:
                scoring_results = self._integrate_strategy_scoring(results)
                aggregated_results['strategy_scoring'] = scoring_results
            
            # 結果の保存
            self._save_results(aggregated_results)
            
            self.progress.status = TestStatus.COMPLETED
            self.logger.info("統合ウォークフォワードテスト完了")
            
            return aggregated_results
            
        except Exception as e:
            self.progress.status = TestStatus.FAILED
            self.logger.error(f"統合ウォークフォワードテスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _generate_test_combinations(self) -> List[Dict[str, Any]]:
        """テスト組み合わせの生成"""
        combinations = []
        
        for symbol in self.config.symbols:
            for strategy in self.config.strategies:
                if strategy not in AVAILABLE_STRATEGIES:
                    self.logger.warning(f"戦略 {strategy} が利用できません、スキップします")
                    continue
                
                combination = {
                    'symbol': symbol,
                    'strategy': strategy,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date,
                    'window_size_days': self.config.window_size_days,
                    'step_size_days': self.config.step_size_days
                }
                combinations.append(combination)
        
        return combinations

    def _execute_sequential(self, test_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """逐次実行"""
        self.logger.info("逐次実行モード")
        results = []
        
        for i, combination in enumerate(test_combinations):
            try:
                self.progress.current_test = f"{combination['symbol']}-{combination['strategy']}"
                self.logger.info(f"実行中 ({i+1}/{len(test_combinations)}): {self.progress.current_test}")
                
                result = self._execute_single_test(combination)
                if result:
                    results.append(result)
                    self.progress.completed_tests += 1
                else:
                    self.progress.failed_tests += 1
                
                # 中間保存
                if self.config.save_intermediate and (i + 1) % 10 == 0:
                    self._save_intermediate_results(results, f"sequential_checkpoint_{i+1}")
                
                # 進捗ログ
                if (i + 1) % 5 == 0:
                    self._log_progress()
                    
            except Exception as e:
                self.logger.error(f"テスト失敗 {combination}: {e}")
                self.progress.failed_tests += 1
        
        return results

    def _execute_parallel(self, test_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """並列実行"""
        executor_class = ThreadPoolExecutor if self.processing_mode == ProcessingMode.PARALLEL_THREAD else ProcessPoolExecutor
        max_workers = self.config.max_workers or min(len(test_combinations), os.cpu_count() or 4)
        
        self.logger.info(f"並列実行モード: {self.processing_mode.value}, ワーカー数: {max_workers}")
        
        results = []
        
        with executor_class(max_workers=max_workers) as executor:
            # チャンク単位で処理
            chunks = [test_combinations[i:i + self.config.chunk_size] 
                     for i in range(0, len(test_combinations), self.config.chunk_size)]
            
            for chunk_idx, chunk in enumerate(chunks):
                self.logger.info(f"チャンク {chunk_idx + 1}/{len(chunks)} 処理中")
                
                # 並列実行の開始
                if self.processing_mode == ProcessingMode.PARALLEL_THREAD:
                    future_to_combination = {
                        executor.submit(self._execute_single_test, combination): combination
                        for combination in chunk
                    }
                else:
                    # プロセス並列の場合
                    future_to_combination = {
                        executor.submit(self._execute_single_test_process_safe, combination): combination
                        for combination in chunk
                    }
                
                # 結果の収集
                for future in concurrent.futures.as_completed(future_to_combination, timeout=self.config.timeout_seconds):
                    combination = future_to_combination[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self.progress.completed_tests += 1
                        else:
                            self.progress.failed_tests += 1
                            
                        self.progress.current_test = f"{combination['symbol']}-{combination['strategy']}"
                        
                    except Exception as e:
                        self.logger.error(f"並列テスト失敗 {combination}: {e}")
                        self.progress.failed_tests += 1
                
                # チャンク完了後の中間保存
                if self.config.save_intermediate:
                    self._save_intermediate_results(results, f"parallel_chunk_{chunk_idx + 1}")
                
                self._log_progress()
        
        return results

    def _execute_hybrid(self, test_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ハイブリッド実行（重要なテストは逐次、その他は並列）"""
        self.logger.info("ハイブリッド実行モード")
        
        # 重要度に基づく分類
        high_priority = []
        low_priority = []
        
        for combination in test_combinations:
            # VWAP戦略や主要シンボルは高優先度
            if (combination['strategy'].startswith('VWAP') or 
                combination['symbol'] in ['SPY', 'QQQ', 'AAPL']):
                high_priority.append(combination)
            else:
                low_priority.append(combination)
        
        results = []
        
        # 高優先度を逐次実行
        if high_priority:
            self.logger.info(f"高優先度テスト逐次実行: {len(high_priority)}件")
            sequential_results = self._execute_sequential(high_priority)
            results.extend(sequential_results)
        
        # 低優先度を並列実行
        if low_priority:
            self.logger.info(f"低優先度テスト並列実行: {len(low_priority)}件")
            # 一時的に並列モードに変更
            original_mode = self.processing_mode
            self.processing_mode = ProcessingMode.PARALLEL_THREAD
            parallel_results = self._execute_parallel(low_priority)
            results.extend(parallel_results)
            self.processing_mode = original_mode
        
        return results

    def _execute_single_test(self, combination: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """単一テストの実行"""
        try:
            symbol = combination['symbol']
            strategy_name = combination['strategy']
            
            # データの取得
            data = self._fetch_test_data(combination)
            if data is None or data.empty:
                self.logger.warning(f"データ取得失敗: {symbol}")
                return None
            
            # 既存システムを使用した実行
            if self.walkforward_executor:
                # 既存システムとの統合実行
                scenario_data = {
                    'data': data,
                    'windows': self._generate_windows(data, combination)
                }
                
                walkforward_results = self.walkforward_executor.execute_walkforward_test(
                    symbol, strategy_name, scenario_data
                )
                
                if not walkforward_results:
                    return None
                
                # 結果の拡張
                enhanced_result = self._enhance_test_result(
                    walkforward_results, combination, data
                )
                
                return enhanced_result
            
            else:
                # フォールバック実行
                return self._execute_fallback_test(combination, data)
                
        except Exception as e:
            self.logger.error(f"単一テスト失敗 {combination}: {e}")
            return None

    def _execute_single_test_process_safe(self, combination: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """プロセス並列用の単一テスト実行（シリアライゼーション対応）"""
        try:
            # プロセス間での共有可能な形式に変換
            result = self._execute_single_test(combination)
            if result:
                # 結果を辞書形式に変換してシリアライゼーション可能にする
                return self._serialize_result(result)
            return None
        except Exception as e:
            # エラーログはプロセス内で処理
            return None

    def _fetch_test_data(self, combination: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """テストデータの取得"""
        try:
            symbol = combination['symbol']
            start_date = combination['start_date']
            end_date = combination['end_date']
            
            # データフェッチャーを使用
            fetcher = data_fetcher
            processor = data_processor
            
            # データ取得
            raw_data = fetcher.fetch_stock_data(
                symbol, 
                datetime.strptime(start_date, '%Y-%m-%d'),
                datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            if raw_data is None or raw_data.empty:
                return None
            
            # データ処理
            processed_data = processor.process_data(raw_data)
            
            # 最低データ点数のチェック
            if len(processed_data) < self.config.min_data_points:
                self.logger.warning(f"データ点数不足 {symbol}: {len(processed_data)} < {self.config.min_data_points}")
                return None
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"データ取得エラー {symbol}: {e}")
            return None

    def _generate_windows(self, data: pd.DataFrame, combination: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ウォークフォワードウィンドウの生成"""
        windows = []
        window_size = combination['window_size_days']
        step_size = combination['step_size_days']
        
        start_idx = 0
        while start_idx + window_size < len(data):
            end_idx = start_idx + window_size
            out_of_sample_end = min(end_idx + step_size, len(data))
            
            window = {
                'in_sample_start': start_idx,
                'in_sample_end': end_idx,
                'out_of_sample_start': end_idx,
                'out_of_sample_end': out_of_sample_end,
                'window_data': data.iloc[start_idx:out_of_sample_end].copy()
            }
            windows.append(window)
            
            start_idx += step_size
        
        return windows

    def _enhance_test_result(self, walkforward_results: List[Dict[str, Any]], 
                           combination: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """テスト結果の拡張"""
        enhanced_result = {
            'combination': combination,
            'walkforward_results': walkforward_results,
            'summary_metrics': self._calculate_summary_metrics(walkforward_results),
            'data_quality': self._assess_data_quality(data),
            'execution_timestamp': datetime.now().isoformat()
        }
        
        # 市場分類の追加
        if self.config.enable_market_classification and self.market_classifier:
            try:
                market_classification = self._classify_market_environment(data, combination)
                enhanced_result['market_classification'] = market_classification
            except Exception as e:
                self.logger.warning(f"市場分類失敗: {e}")
        
        # 戦略スコアリングの追加
        if self.config.enable_strategy_scoring and self.strategy_scorer:
            try:
                strategy_score = self._score_strategy_performance(walkforward_results, combination)
                enhanced_result['strategy_score'] = strategy_score
            except Exception as e:
                self.logger.warning(f"戦略スコアリング失敗: {e}")
        
        return enhanced_result

    def _execute_fallback_test(self, combination: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """フォールバック実行（既存システム利用不可時）"""
        try:
            symbol = combination['symbol']
            strategy_name = combination['strategy']
            
            if strategy_name not in AVAILABLE_STRATEGIES:
                return None
            
            strategy_class = AVAILABLE_STRATEGIES[strategy_name]
            
            # 簡易ウォークフォワード実行
            windows = self._generate_windows(data, combination)
            results = []
            
            for window in windows:
                try:
                    window_data = window['window_data']
                    in_sample_data = window_data.iloc[:window['in_sample_end'] - window['in_sample_start']]
                    out_sample_data = window_data.iloc[window['in_sample_end'] - window['in_sample_start']:]
                    
                    # 戦略の初期化と実行
                    strategy = strategy_class()
                    
                    # インサンプルでの最適化（可能な場合）
                    if hasattr(strategy, 'optimize'):
                        strategy.optimize(in_sample_data)
                    
                    # アウトオブサンプルでのテスト
                    backtest_result = strategy.backtest(out_sample_data)
                    
                    if backtest_result is not None and not backtest_result.empty:
                        window_result = {
                            'window_index': len(results),
                            'in_sample_period': f"{in_sample_data.index[0]} to {in_sample_data.index[-1]}",
                            'out_sample_period': f"{out_sample_data.index[0]} to {out_sample_data.index[-1]}",
                            'backtest_result': backtest_result,
                            'metrics': self._calculate_window_metrics(backtest_result)
                        }
                        results.append(window_result)
                
                except Exception as e:
                    self.logger.warning(f"ウィンドウ実行失敗: {e}")
                    continue
            
            return {
                'combination': combination,
                'walkforward_results': results,
                'summary_metrics': self._calculate_summary_metrics(results),
                'execution_timestamp': datetime.now().isoformat(),
                'execution_mode': 'fallback'
            }
            
        except Exception as e:
            self.logger.error(f"フォールバック実行失敗: {e}")
            return None

    def _calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """サマリメトリクスの計算"""
        if not results:
            return {}
        
        try:
            # 全ウィンドウの結果を統合
            all_returns = []
            all_win_rates = []
            all_sharpe_ratios = []
            
            for result in results:
                if 'metrics' in result:
                    metrics = result['metrics']
                    if 'total_return' in metrics:
                        all_returns.append(metrics['total_return'])
                    if 'win_rate' in metrics:
                        all_win_rates.append(metrics['win_rate'])
                    if 'sharpe_ratio' in metrics:
                        all_sharpe_ratios.append(metrics['sharpe_ratio'])
            
            summary = {
                'total_windows': len(results),
                'avg_return': np.mean(all_returns) if all_returns else 0.0,
                'std_return': np.std(all_returns) if all_returns else 0.0,
                'avg_win_rate': np.mean(all_win_rates) if all_win_rates else 0.0,
                'avg_sharpe_ratio': np.mean(all_sharpe_ratios) if all_sharpe_ratios else 0.0,
                'consistency_score': 1.0 - (np.std(all_returns) / (np.mean(all_returns) + 1e-8)) if all_returns else 0.0
            }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"サマリメトリクス計算失敗: {e}")
            return {}

    def _calculate_window_metrics(self, backtest_result: pd.DataFrame) -> Dict[str, float]:
        """ウィンドウメトリクスの計算"""
        try:
            if backtest_result.empty:
                return {}
            
            # 基本的なメトリクス計算
            metrics = {}
            
            if 'Portfolio_Value' in backtest_result.columns:
                portfolio_values = backtest_result['Portfolio_Value'].dropna()
                if not portfolio_values.empty:
                    initial_value = portfolio_values.iloc[0]
                    final_value = portfolio_values.iloc[-1]
                    metrics['total_return'] = (final_value - initial_value) / initial_value
                    
                    # 日次リターン
                    daily_returns = portfolio_values.pct_change().dropna()
                    if not daily_returns.empty:
                        metrics['volatility'] = daily_returns.std()
                        metrics['sharpe_ratio'] = daily_returns.mean() / (daily_returns.std() + 1e-8)
            
            # 取引メトリクス
            if 'Entry_Signal' in backtest_result.columns and 'Exit_Signal' in backtest_result.columns:
                entries = backtest_result['Entry_Signal'].sum()
                exits = backtest_result['Exit_Signal'].sum()
                metrics['total_trades'] = min(entries, exits)
                
                if 'Trade_PnL' in backtest_result.columns:
                    trade_pnl = backtest_result['Trade_PnL'].dropna()
                    if not trade_pnl.empty:
                        winning_trades = (trade_pnl > 0).sum()
                        metrics['win_rate'] = winning_trades / len(trade_pnl) if len(trade_pnl) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"ウィンドウメトリクス計算失敗: {e}")
            return {}

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """データ品質の評価"""
        try:
            quality = {
                'total_points': len(data),
                'missing_values': data.isnull().sum().sum(),
                'date_range': {
                    'start': data.index[0].isoformat() if not data.empty else None,
                    'end': data.index[-1].isoformat() if not data.empty else None
                },
                'completeness': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))) if not data.empty else 0.0
            }
            
            # 価格データの整合性チェック
            if 'Close' in data.columns:
                close_prices = data['Close'].dropna()
                if not close_prices.empty:
                    quality['price_stability'] = 1.0 / (1.0 + close_prices.std() / close_prices.mean())
            
            return quality
            
        except Exception as e:
            self.logger.warning(f"データ品質評価失敗: {e}")
            return {}

    def _classify_market_environment(self, data: pd.DataFrame, combination: Dict[str, Any]) -> Dict[str, Any]:
        """市場環境の分類"""
        try:
            classification = {}
            
            if self.market_classifier:
                # A-B市場分類
                ab_result = self.market_classifier.classify_market_state(data)
                classification['ab_classification'] = ab_result
            
            if self.market_detector:
                # 拡張市場検出
                enhanced_result = self.market_detector.detect_market_regime(data)
                classification['enhanced_detection'] = enhanced_result
            
            return classification
            
        except Exception as e:
            self.logger.warning(f"市場分類失敗: {e}")
            return {}

    def _score_strategy_performance(self, results: List[Dict[str, Any]], combination: Dict[str, Any]) -> Dict[str, Any]:
        """戦略パフォーマンスのスコアリング"""
        try:
            if not self.strategy_scorer:
                return {}
            
            # 結果を戦略スコアリング用の形式に変換
            performance_data = []
            for result in results:
                if 'metrics' in result:
                    performance_data.append(result['metrics'])
            
            if not performance_data:
                return {}
            
            # スコアリングの実行
            score_result = self.strategy_scorer.calculate_comprehensive_score(
                performance_data, combination['strategy']
            )
            
            return score_result
            
        except Exception as e:
            self.logger.warning(f"戦略スコアリング失敗: {e}")
            return {}

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """結果の集約"""
        try:
            if not results:
                return {'summary': '結果なし'}
            
            # 戦略別集約
            strategy_aggregation = {}
            symbol_aggregation = {}
            overall_metrics = []
            
            for result in results:
                combination = result['combination']
                strategy = combination['strategy']
                symbol = combination['symbol']
                
                # 戦略別
                if strategy not in strategy_aggregation:
                    strategy_aggregation[strategy] = []
                strategy_aggregation[strategy].append(result)
                
                # シンボル別
                if symbol not in symbol_aggregation:
                    symbol_aggregation[symbol] = []
                symbol_aggregation[symbol].append(result)
                
                # 全体メトリクス
                if 'summary_metrics' in result:
                    overall_metrics.append(result['summary_metrics'])
            
            # 集約統計の計算
            aggregated = {
                'total_tests': len(results),
                'strategy_breakdown': {
                    strategy: {
                        'test_count': len(tests),
                        'avg_metrics': self._average_metrics([t.get('summary_metrics', {}) for t in tests])
                    }
                    for strategy, tests in strategy_aggregation.items()
                },
                'symbol_breakdown': {
                    symbol: {
                        'test_count': len(tests),
                        'avg_metrics': self._average_metrics([t.get('summary_metrics', {}) for t in tests])
                    }
                    for symbol, tests in symbol_aggregation.items()
                },
                'overall_metrics': self._average_metrics(overall_metrics),
                'execution_summary': {
                    'completion_rate': self.progress.completion_rate,
                    'failure_rate': self.progress.failure_rate,
                    'total_execution_time': (datetime.now() - self.progress.start_time).total_seconds() if self.progress.start_time else 0
                }
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"結果集約失敗: {e}")
            return {'error': str(e)}

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """メトリクスの平均計算"""
        if not metrics_list:
            return {}
        
        # 共通キーを抽出
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        averaged = {}
        for key in all_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                averaged[key] = np.mean(values)
        
        return averaged

    def _integrate_market_classification(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """市場分類結果の統合"""
        try:
            market_results = {
                'classification_summary': {},
                'strategy_performance_by_market': {},
                'market_distribution': {}
            }
            
            # 各結果から市場分類を抽出
            for result in results:
                if 'market_classification' in result:
                    classification = result['market_classification']
                    combination = result['combination']
                    
                    # 市場タイプ別の戦略パフォーマンス
                    if 'ab_classification' in classification:
                        market_type = classification['ab_classification'].get('market_state', 'unknown')
                        strategy = combination['strategy']
                        
                        if market_type not in market_results['strategy_performance_by_market']:
                            market_results['strategy_performance_by_market'][market_type] = {}
                        
                        if strategy not in market_results['strategy_performance_by_market'][market_type]:
                            market_results['strategy_performance_by_market'][market_type][strategy] = []
                        
                        market_results['strategy_performance_by_market'][market_type][strategy].append(
                            result.get('summary_metrics', {})
                        )
            
            # 市場分布の計算
            for market_type, strategies in market_results['strategy_performance_by_market'].items():
                market_results['market_distribution'][market_type] = sum(len(tests) for tests in strategies.values())
            
            return market_results
            
        except Exception as e:
            self.logger.warning(f"市場分類統合失敗: {e}")
            return {}

    def _integrate_strategy_scoring(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """戦略スコアリング結果の統合"""
        try:
            scoring_results = {
                'strategy_rankings': {},
                'score_distribution': {},
                'performance_categories': {}
            }
            
            strategy_scores = {}
            
            for result in results:
                if 'strategy_score' in result:
                    score_data = result['strategy_score']
                    strategy = result['combination']['strategy']
                    
                    if strategy not in strategy_scores:
                        strategy_scores[strategy] = []
                    
                    strategy_scores[strategy].append(score_data)
            
            # 戦略ランキングの計算
            for strategy, scores in strategy_scores.items():
                if scores:
                    avg_score = np.mean([s.get('overall_score', 0) for s in scores if 'overall_score' in s])
                    scoring_results['strategy_rankings'][strategy] = avg_score
            
            # ランキングソート
            scoring_results['strategy_rankings'] = dict(
                sorted(scoring_results['strategy_rankings'].items(), 
                      key=lambda x: x[1], reverse=True)
            )
            
            return scoring_results
            
        except Exception as e:
            self.logger.warning(f"戦略スコアリング統合失敗: {e}")
            return {}

    def _save_results(self, results: Dict[str, Any]):
        """結果の保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON形式で保存
            json_path = self.output_dir / f"comprehensive_walkforward_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # Pickle形式で保存（完全な結果保存用）
            pickle_path = self.output_dir / f"comprehensive_walkforward_results_{timestamp}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"結果保存完了: {json_path}")
            
        except Exception as e:
            self.logger.error(f"結果保存失敗: {e}")

    def _save_intermediate_results(self, results: List[Dict[str, Any]], checkpoint_name: str):
        """中間結果の保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = self.output_dir / f"{checkpoint_name}_{timestamp}.json"
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"中間結果保存: {checkpoint_path}")
            
        except Exception as e:
            self.logger.warning(f"中間結果保存失敗: {e}")

    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """結果のシリアライゼーション"""
        try:
            # DataFrameやその他のオブジェクトを辞書に変換
            serialized = {}
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    serialized[key] = value.to_dict()
                elif isinstance(value, datetime):
                    serialized[key] = value.isoformat()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serialized[key] = value
                else:
                    serialized[key] = str(value)
            
            return serialized
            
        except Exception as e:
            self.logger.warning(f"結果シリアライゼーション失敗: {e}")
            return {}

    def _log_progress(self):
        """進捗ログ"""
        completion_rate = self.progress.completion_rate * 100
        failure_rate = self.progress.failure_rate * 100
        
        self.logger.info(
            f"進捗: {completion_rate:.1f}% "
            f"({self.progress.completed_tests}/{self.progress.total_tests}) "
            f"失敗率: {failure_rate:.1f}% "
            f"現在のテスト: {self.progress.current_test}"
        )
        
        # 推定完了時間の計算
        if self.progress.start_time and self.progress.completed_tests > 0:
            elapsed = datetime.now() - self.progress.start_time
            estimated_total = elapsed * (self.progress.total_tests / self.progress.completed_tests)
            estimated_completion = self.progress.start_time + estimated_total
            self.progress.estimated_completion = estimated_completion
            
            remaining = estimated_completion - datetime.now()
            self.logger.info(f"推定完了時刻: {estimated_completion.strftime('%H:%M:%S')} (残り約{remaining.total_seconds()/60:.0f}分)")

def create_test_configuration(
    symbols: List[str] = None,
    strategies: List[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    processing_mode: str = "auto",
    **kwargs
) -> TestConfiguration:
    """テスト設定の作成ヘルパー"""
    
    # デフォルト値の設定
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    
    if strategies is None:
        strategies = list(AVAILABLE_STRATEGIES.keys())
    
    # 処理モードの変換
    mode_mapping = {
        "sequential": ProcessingMode.SEQUENTIAL,
        "parallel": ProcessingMode.PARALLEL_THREAD,
        "process": ProcessingMode.PARALLEL_PROCESS,
        "hybrid": ProcessingMode.HYBRID,
        "auto": ProcessingMode.AUTO
    }
    
    processing_mode_enum = mode_mapping.get(processing_mode.lower(), ProcessingMode.AUTO)
    
    config = TestConfiguration(
        symbols=symbols,
        strategies=strategies,
        start_date=start_date,
        end_date=end_date,
        processing_mode=processing_mode_enum,
        **kwargs
    )
    
    return config

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="統合ウォークフォワードテストシステム")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"], help="テスト対象シンボル")
    parser.add_argument("--strategies", nargs="+", default=None, help="テスト戦略")
    parser.add_argument("--start-date", default="2022-01-01", help="開始日")
    parser.add_argument("--end-date", default="2023-12-31", help="終了日")
    parser.add_argument("--mode", default="auto", choices=["sequential", "parallel", "process", "hybrid", "auto"], help="処理モード")
    parser.add_argument("--output", default="output/comprehensive_walkforward", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 設定の作成
    config = create_test_configuration(
        symbols=args.symbols,
        strategies=args.strategies,
        start_date=args.start_date,
        end_date=args.end_date,
        processing_mode=args.mode,
        output_directory=args.output
    )
    
    # テスト実行
    tester = ComprehensiveWalkforwardTester(config)
    
    try:
        results = tester.execute_comprehensive_test()
        print(f"\n=== 統合ウォークフォワードテスト完了 ===")
        print(f"総テスト数: {results.get('total_tests', 0)}")
        print(f"完了率: {tester.progress.completion_rate*100:.1f}%")
        print(f"結果保存先: {tester.output_dir}")
        
        # 戦略ランキングの表示
        if 'strategy_scoring' in results and 'strategy_rankings' in results['strategy_scoring']:
            print("\n=== 戦略ランキング ===")
            for strategy, score in results['strategy_scoring']['strategy_rankings'].items():
                print(f"{strategy}: {score:.3f}")
                
    except Exception as e:
        print(f"テスト実行失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
