"""
統合ウォークフォワードテストシステム（市場分類なしバージョン）
フェーズ2：包括的なパフォーマンス検証システム（簡易版）

市場分類システムへの依存を除いた基本的なウォークフォワードテストを実行します。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import sys
import os
import json
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 必要なインポート
import data_fetcher
import data_processor
from src.config.logger_config import setup_logger

# 既存のコンポーネントをインポート（市場分類を除く）
try:
    from src.analysis.walkforward_executor import WalkforwardExecutor
    from src.analysis.walkforward_scenarios import WalkforwardScenarios
    from src.analysis.walkforward_result_analyzer import WalkforwardResultAnalyzer
except ImportError as e:
    logging.warning(f"既存ウォークフォワードコンポーネントの一部が利用できません: {e}")

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
    AUTO = "auto"

class TestStatus(Enum):
    """テスト状況"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

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
    output_directory: str = "output/simple_walkforward"
    
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
    status: TestStatus = TestStatus.PENDING
    
    @property
    def completion_rate(self) -> float:
        """完了率"""
        if self.total_tests == 0:
            return 0.0
        return self.completed_tests / self.total_tests

class SimpleWalkforwardTester:
    """簡易統合ウォークフォワードテストシステム"""
    
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
        
        # 既存システムとの統合
        self._initialize_existing_systems()
        
        self.logger.info(f"簡易ウォークフォワードテスト初期化完了")

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

    def execute_comprehensive_test(self) -> Dict[str, Any]:
        """統合ウォークフォワードテストの実行"""
        self.logger.info("簡易ウォークフォワードテスト開始")
        self.progress.status = TestStatus.RUNNING
        self.progress.start_time = datetime.now()
        
        try:
            # テスト組み合わせの生成
            test_combinations = self._generate_test_combinations()
            self.progress.total_tests = len(test_combinations)
            
            self.logger.info(f"総テスト数: {self.progress.total_tests}")
            
            # 逐次実行
            results = self._execute_sequential(test_combinations)
            
            # 結果の集約
            aggregated_results = self._aggregate_results(results)
            
            # 結果の保存
            self._save_results(aggregated_results)
            
            self.progress.status = TestStatus.COMPLETED
            self.logger.info("簡易ウォークフォワードテスト完了")
            
            return aggregated_results
            
        except Exception as e:
            self.progress.status = TestStatus.FAILED
            self.logger.error(f"簡易ウォークフォワードテスト失敗: {e}")
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
                
                # 進捗ログ
                if (i + 1) % 2 == 0:
                    self._log_progress()
                    
            except Exception as e:
                self.logger.error(f"テスト失敗 {combination}: {e}")
                self.progress.failed_tests += 1
        
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
            'execution_timestamp': datetime.now().isoformat(),
            'market_classification': {'market_state': 'unknown'}  # 市場分類なしバージョン
        }
        
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
                'execution_mode': 'fallback',
                'market_classification': {'market_state': 'unknown'}
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
            
            return quality
            
        except Exception as e:
            self.logger.warning(f"データ品質評価失敗: {e}")
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

    def _save_results(self, results: Dict[str, Any]):
        """結果の保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON形式で保存
            json_path = self.output_dir / f"simple_walkforward_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"結果保存完了: {json_path}")
            
        except Exception as e:
            self.logger.error(f"結果保存失敗: {e}")

    def _log_progress(self):
        """進捗ログ"""
        completion_rate = self.progress.completion_rate * 100
        
        self.logger.info(
            f"進捗: {completion_rate:.1f}% "
            f"({self.progress.completed_tests}/{self.progress.total_tests}) "
            f"現在のテスト: {self.progress.current_test}"
        )

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
        symbols = ["AAPL", "MSFT"]
    
    if strategies is None:
        strategies = list(AVAILABLE_STRATEGIES.keys())
    
    # 処理モードの変換
    mode_mapping = {
        "sequential": ProcessingMode.SEQUENTIAL,
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
    
    parser = argparse.ArgumentParser(description="簡易統合ウォークフォワードテストシステム")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"], help="テスト対象シンボル")
    parser.add_argument("--strategies", nargs="+", default=None, help="テスト戦略")
    parser.add_argument("--start-date", default="2023-01-01", help="開始日")
    parser.add_argument("--end-date", default="2023-06-30", help="終了日")
    parser.add_argument("--output", default="output/simple_walkforward", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 設定の作成
    config = create_test_configuration(
        symbols=args.symbols,
        strategies=args.strategies,
        start_date=args.start_date,
        end_date=args.end_date,
        output_directory=args.output
    )
    
    # テスト実行
    tester = SimpleWalkforwardTester(config)
    
    try:
        results = tester.execute_comprehensive_test()
        print(f"\n=== 簡易ウォークフォワードテスト完了 ===")
        print(f"総テスト数: {results.get('total_tests', 0)}")
        print(f"完了率: {tester.progress.completion_rate*100:.1f}%")
        print(f"結果保存先: {tester.output_dir}")
        
    except Exception as e:
        print(f"テスト実行失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
