"""
Module: Composite Strategy Backtest Engine
File: composite_backtest_engine.py
Description: 
  4-2-2「複合戦略バックテスト機能実装」
  既存システムと統合した複合戦略バックテストエンジン

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 複合戦略バックテスト実行
  - 動的期間分割によるテスト
  - 期待値重視のパフォーマンス評価
  - Excel + 可視化レポート生成
"""

import os
import sys
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムのインポート
try:
    from config.composite_strategy_execution_engine import CompositeStrategyExecutionEngine, ExecutionRequest, ExecutionResponse
    from config.strategy_selector import StrategySelector
    from config.multi_strategy_coordination_manager import MultiStrategyCoordinationManager
    from trend_strategy_switch_tester import TrendStrategySwitchTester
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    logging.getLogger(__name__).warning(f"Some imports failed: {e}")

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BacktestStatus(Enum):
    """バックテストステータス"""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BacktestScenario:
    """バックテストシナリオ"""
    scenario_id: str
    scenario_type: str
    test_period: Tuple[datetime, datetime]
    market_regime: str
    expected_challenges: List[str]
    performance_expectations: Dict[str, float]
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)

@dataclass
class StrategyCombination:
    """戦略組み合わせ"""
    combination_id: str
    name: str
    strategies: List[Dict[str, Any]]
    rebalancing_rules: Dict[str, Any]
    risk_limits: Dict[str, float]
    expected_performance: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """バックテスト結果"""
    backtest_id: str
    combination_id: str
    scenario_id: str
    start_time: datetime
    end_time: datetime
    execution_time: float
    status: BacktestStatus
    performance_metrics: Dict[str, float]
    strategy_results: Dict[str, Any]
    switching_events: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    trades: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None

class CompositeStrategyBacktestEngine:
    """複合戦略バックテストエンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """エンジンの初期化"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # 既存システム統合（軽負荷モード）
        self._initialize_existing_systems()
        
        # バックテスト専用コンポーネント
        self.strategy_combination_manager = None
        self.backtest_scenario_generator = None
        self.performance_calculator = None
        self.result_analyzer = None
        
        # 実行状態管理
        self.current_status = BacktestStatus.IDLE
        self.active_backtests: Dict[str, BacktestResult] = {}
        self.completed_backtests: List[BacktestResult] = []
        
        # パフォーマンス監視
        self.performance_stats = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "total_execution_time": 0.0,
            "engine_start_time": datetime.now()
        }
        
        self.logger.info("CompositeStrategyBacktestEngine initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "backtest", 
                "composite_backtest_config.json"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "backtest_engine": {
                "engine_name": "CompositeStrategyBacktestEngine",
                "version": "1.0.0",
                "execution_mode": "hybrid",
                "max_concurrent_backtests": 3,
                "default_timeout_minutes": 30
            },
            "performance_calculation": {
                "primary_metrics": [
                    "expected_return",
                    "expected_sharpe_ratio",
                    "expected_max_drawdown",
                    "return_consistency",
                    "risk_adjusted_return"
                ]
            }
        }
    
    def _initialize_existing_systems(self):
        """既存システムの初期化（軽負荷モード）"""
        try:
            # 複合戦略実行エンジン
            self.composite_execution_engine = CompositeStrategyExecutionEngine()
            self.logger.info("CompositeStrategyExecutionEngine initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize CompositeStrategyExecutionEngine: {e}")
            self.composite_execution_engine = None
        
        try:
            # 4-2-1トレンド切替テスター（協力連携）
            self.trend_switch_tester = TrendStrategySwitchTester()
            self.logger.info("TrendStrategySwitchTester integrated")
        except Exception as e:
            self.logger.warning(f"Failed to integrate TrendStrategySwitchTester: {e}")
            self.trend_switch_tester = None
        
        try:
            # マルチ戦略調整マネージャー（軽負荷統合）
            self.coordination_manager = MultiStrategyCoordinationManager()
            self.logger.info("MultiStrategyCoordinationManager integrated (light mode)")
        except Exception as e:
            self.logger.warning(f"Failed to integrate MultiStrategyCoordinationManager: {e}")
            self.coordination_manager = None
    
    async def run_composite_backtest(self, 
                                   strategy_combinations: List[Dict],
                                   test_period: Tuple[datetime, datetime],
                                   benchmark_strategy: str = "buy_and_hold") -> Dict[str, Any]:
        """複合戦略バックテスト実行"""
        
        backtest_id = f"composite_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting composite backtest {backtest_id}")
        
        start_time = time.time()
        self.current_status = BacktestStatus.PREPARING
        
        try:
            # 1. テスト環境準備
            test_environment = await self._setup_test_environment(test_period)
            self.logger.info("Test environment prepared")
            
            # 2. 戦略組み合わせ別バックテスト
            backtest_results = []
            self.current_status = BacktestStatus.RUNNING
            
            for combination in strategy_combinations:
                try:
                    result = await self._run_combination_backtest(
                        combination, test_environment, backtest_id
                    )
                    backtest_results.append(result)
                    self.logger.info(f"Completed backtest for combination: {combination.get('combination_id', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Failed to run backtest for combination {combination}: {e}")
                    # エラーが発生しても他の組み合わせのテストを継続
                    continue
            
            # 3. ベンチマーク比較
            benchmark_result = await self._run_benchmark_backtest(
                benchmark_strategy, test_environment
            )
            
            # 4. 結果分析・比較
            self.current_status = BacktestStatus.ANALYZING
            analysis_report = await self._analyze_backtest_results(
                backtest_results, benchmark_result
            )
            
            # 5. レポート生成
            self.current_status = BacktestStatus.REPORTING
            report_paths = await self._generate_comprehensive_report(analysis_report)
            
            # 統計更新
            execution_time = time.time() - start_time
            self.performance_stats["total_backtests"] += 1
            self.performance_stats["successful_backtests"] += 1
            self.performance_stats["total_execution_time"] += execution_time
            
            self.current_status = BacktestStatus.COMPLETED
            self.logger.info(f"Composite backtest {backtest_id} completed in {execution_time:.2f}s")
            
            return {
                "backtest_id": backtest_id,
                "status": "completed",
                "execution_time": execution_time,
                "analysis_report": analysis_report,
                "report_paths": report_paths,
                "backtest_results": backtest_results,
                "benchmark_result": benchmark_result
            }
            
        except Exception as e:
            self.current_status = BacktestStatus.FAILED
            self.logger.error(f"Composite backtest {backtest_id} failed: {e}")
            
            return {
                "backtest_id": backtest_id,
                "status": "failed",
                "error_message": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _setup_test_environment(self, test_period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """テスト環境準備"""
        
        # 市場データの準備
        try:
            # 実際の実装では、data_fetcher等を使用してデータを取得
            # ここではサンプルデータを生成
            market_data = self._generate_sample_market_data(test_period)
            self.logger.info(f"Market data prepared: {len(market_data)} data points")
        except Exception as e:
            self.logger.warning(f"Failed to load real market data: {e}, using synthetic data")
            market_data = self._generate_sample_market_data(test_period)
        
        # トレンド変化期間の検出（4-2-1システム連携）
        trend_periods = []
        if self.trend_switch_tester:
            try:
                trend_periods = await self._detect_trend_change_periods(test_period, market_data)
                self.logger.info(f"Detected {len(trend_periods)} trend change periods")
            except Exception as e:
                self.logger.warning(f"Trend period detection failed: {e}")
        
        return {
            "market_data": market_data,
            "trend_periods": trend_periods,
            "test_period": test_period,
            "data_quality": self._assess_data_quality(market_data)
        }
    
    def _generate_sample_market_data(self, test_period: Tuple[datetime, datetime]) -> pd.DataFrame:
        """サンプル市場データ生成"""
        
        start_date, end_date = test_period
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 基本的な価格データを生成
        n_days = len(date_range)
        np.random.seed(42)  # 再現可能な結果のため
        
        # 価格の生成（幾何ブラウン運動）
        initial_price = 100.0
        daily_returns = np.random.normal(0.0005, 0.02, n_days)  # 平均0.05%、標準偏差2%
        cumulative_returns = np.cumsum(daily_returns)
        prices = initial_price * np.exp(cumulative_returns)
        
        # OHLCV データの作成
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        opens = np.concatenate([[prices[0]], prices[:-1] * (1 + np.random.normal(0, 0.005, n_days-1))])
        volumes = np.random.lognormal(10, 0.5, n_days)
        
        market_data = pd.DataFrame({
            'Date': date_range,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }).set_index('Date')
        
        return market_data
    
    async def _detect_trend_change_periods(self, 
                                         test_period: Tuple[datetime, datetime], 
                                         market_data: pd.DataFrame) -> List[Dict]:
        """トレンド変化期間検出（4-2-1システム連携）"""
        
        trend_periods = []
        try:
            # 簡略化されたトレンド検出
            # 実際の実装では、UnifiedTrendDetectorやTrendStrategySwitchTesterを使用
            
            window = 20  # 20日移動平均
            market_data['SMA20'] = market_data['Close'].rolling(window=window).mean()
            market_data['Trend'] = np.where(
                market_data['Close'] > market_data['SMA20'], 'uptrend', 'downtrend'
            )
            
            # トレンド変化点を検出
            trend_changes = market_data['Trend'].ne(market_data['Trend'].shift()).cumsum()
            
            for trend_id in trend_changes.unique():
                if pd.isna(trend_id):
                    continue
                    
                period_data = market_data[trend_changes == trend_id]
                if len(period_data) > 5:  # 最低5日間のトレンド期間
                    trend_periods.append({
                        'start': period_data.index[0],
                        'end': period_data.index[-1],
                        'trend': period_data['Trend'].iloc[0],
                        'confidence': 0.7,  # 簡略化
                        'volatility': period_data['Close'].pct_change().std()
                    })
            
        except Exception as e:
            self.logger.warning(f"Error in trend detection: {e}")
        
        return trend_periods
    
    def _assess_data_quality(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """データ品質評価"""
        
        total_points = len(market_data)
        missing_points = market_data.isnull().sum().sum()
        
        quality_metrics = {
            "completeness": 1.0 - (missing_points / (total_points * len(market_data.columns))),
            "data_points": total_points,
            "missing_ratio": missing_points / total_points if total_points > 0 else 1.0,
            "volatility": market_data['Close'].pct_change().std() if 'Close' in market_data.columns else 0.0
        }
        
        return quality_metrics
    
    async def _run_combination_backtest(self, 
                                      combination: Dict, 
                                      test_environment: Dict,
                                      backtest_id: str) -> BacktestResult:
        """戦略組み合わせのバックテスト実行"""
        
        combination_id = combination.get('combination_id', 'unknown')
        start_time = time.time()
        
        try:
            # 戦略の準備
            strategies = combination.get('strategies', [])
            market_data = test_environment['market_data']
            
            # 簡略化されたバックテスト実行
            # 実際の実装では、各戦略を実行して結果を統合
            strategy_results = {}
            all_trades = []
            
            for strategy_config in strategies:
                strategy_name = strategy_config.get('strategy_name', 'unknown')
                weight = strategy_config.get('weight', 0.0)
                
                # 戦略結果のシミュレーション
                strategy_result = self._simulate_strategy_performance(
                    strategy_name, weight, market_data
                )
                strategy_results[strategy_name] = strategy_result
                
                if 'trades' in strategy_result:
                    all_trades.extend(strategy_result['trades'])
            
            # パフォーマンス指標の計算
            performance_metrics = self._calculate_performance_metrics(
                strategy_results, market_data
            )
            
            # リスク指標の計算
            risk_metrics = self._calculate_risk_metrics(strategy_results, market_data)
            
            # 結果の作成
            result = BacktestResult(
                backtest_id=f"{backtest_id}_{combination_id}",
                combination_id=combination_id,
                scenario_id="default",
                start_time=datetime.now() - timedelta(seconds=time.time() - start_time),
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                status=BacktestStatus.COMPLETED,
                performance_metrics=performance_metrics,
                strategy_results=strategy_results,
                switching_events=[],  # 後で実装
                risk_metrics=risk_metrics,
                trades=pd.DataFrame(all_trades) if all_trades else None
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in combination backtest {combination_id}: {e}")
            
            return BacktestResult(
                backtest_id=f"{backtest_id}_{combination_id}",
                combination_id=combination_id,
                scenario_id="default",
                start_time=datetime.now() - timedelta(seconds=time.time() - start_time),
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                status=BacktestStatus.FAILED,
                performance_metrics={},
                strategy_results={},
                switching_events=[],
                risk_metrics={},
                error_message=str(e)
            )
    
    def _simulate_strategy_performance(self, 
                                     strategy_name: str, 
                                     weight: float, 
                                     market_data: pd.DataFrame) -> Dict[str, Any]:
        """戦略パフォーマンスのシミュレーション"""
        
        # 基本的なパフォーマンス指標のシミュレーション
        np.random.seed(hash(strategy_name) % 2147483647)  # 戦略名に基づくシード
        
        # 日次リターンの生成
        daily_returns = np.random.normal(0.0008, 0.015, len(market_data)) * weight
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        
        # トレード生成（簡略化）
        n_trades = max(1, int(len(market_data) / 20))  # 20日に1回程度
        trade_dates = np.random.choice(market_data.index, n_trades, replace=False)
        
        trades = []
        for trade_date in trade_dates:
            trade_return = np.random.normal(0.02, 0.05)  # 平均2%、標準偏差5%
            trades.append({
                'date': trade_date,
                'strategy': strategy_name,
                'return': trade_return,
                'weight': weight
            })
        
        return {
            'strategy_name': strategy_name,
            'weight': weight,
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': cumulative_returns[-1],
            'volatility': np.std(daily_returns) * np.sqrt(252),  # 年間ボラティリティ
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """最大ドローダウンの計算"""
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (1 + peak)
        return np.min(drawdown)
    
    def _calculate_performance_metrics(self, 
                                     strategy_results: Dict, 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """期待値重視のパフォーマンス指標計算"""
        
        # 重み付き統合
        total_weight = sum(result['weight'] for result in strategy_results.values())
        
        if total_weight == 0:
            return {}
        
        # 期待リターンの計算
        expected_return = sum(
            result['total_return'] * result['weight'] 
            for result in strategy_results.values()
        ) / total_weight
        
        # 期待シャープレシオの計算
        expected_sharpe = sum(
            result['sharpe_ratio'] * result['weight'] 
            for result in strategy_results.values()
        ) / total_weight
        
        # 期待最大ドローダウンの計算
        expected_max_drawdown = sum(
            result['max_drawdown'] * result['weight'] 
            for result in strategy_results.values()
        ) / total_weight
        
        # リターン一貫性の計算
        returns_list = [result['total_return'] for result in strategy_results.values()]
        return_consistency = 1.0 - (np.std(returns_list) / np.mean(returns_list)) if np.mean(returns_list) != 0 else 0.0
        
        # リスク調整後リターン
        risk_adjusted_return = expected_return / max(abs(expected_max_drawdown), 0.01)
        
        return {
            'expected_return': expected_return,
            'expected_sharpe_ratio': expected_sharpe,
            'expected_max_drawdown': expected_max_drawdown,
            'return_consistency': return_consistency,
            'risk_adjusted_return': risk_adjusted_return,
            'portfolio_volatility': self._calculate_portfolio_volatility(strategy_results),
            'total_strategies': len(strategy_results),
            'weighted_win_rate': self._calculate_weighted_win_rate(strategy_results)
        }
    
    def _calculate_portfolio_volatility(self, strategy_results: Dict) -> float:
        """ポートフォリオボラティリティの計算"""
        
        # 簡略化された計算（相関を無視）
        total_weight = sum(result['weight'] for result in strategy_results.values())
        
        if total_weight == 0:
            return 0.0
        
        weighted_variance = sum(
            (result['volatility'] * result['weight']) ** 2 
            for result in strategy_results.values()
        ) / (total_weight ** 2)
        
        return np.sqrt(weighted_variance)
    
    def _calculate_weighted_win_rate(self, strategy_results: Dict) -> float:
        """重み付き勝率の計算"""
        
        total_weight = sum(result['weight'] for result in strategy_results.values())
        
        if total_weight == 0:
            return 0.0
        
        # 簡略化：各戦略のトレードから勝率を計算
        total_trades = 0
        winning_trades = 0
        
        for result in strategy_results.values():
            trades = result.get('trades', [])
            strategy_total = len(trades)
            strategy_wins = sum(1 for trade in trades if trade.get('return', 0) > 0)
            
            weight_factor = result['weight'] / total_weight
            total_trades += strategy_total * weight_factor
            winning_trades += strategy_wins * weight_factor
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_risk_metrics(self, 
                              strategy_results: Dict, 
                              market_data: pd.DataFrame) -> Dict[str, float]:
        """リスク指標の計算"""
        
        # 基本的なリスク指標
        return {
            'value_at_risk_95': 0.05,  # 簡略化
            'expected_shortfall': 0.07,
            'beta': 1.0,
            'correlation_with_market': 0.7,
            'tracking_error': 0.03,
            'information_ratio': 0.5
        }
    
    async def _run_benchmark_backtest(self, 
                                    benchmark_strategy: str, 
                                    test_environment: Dict) -> Dict[str, Any]:
        """ベンチマークバックテストの実行"""
        
        market_data = test_environment['market_data']
        
        if benchmark_strategy == "buy_and_hold":
            # バイ・アンド・ホールド戦略
            initial_price = market_data['Close'].iloc[0]
            final_price = market_data['Close'].iloc[-1]
            total_return = (final_price - initial_price) / initial_price
            
            daily_returns = market_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            max_dd = self._calculate_max_drawdown(
                (market_data['Close'] / market_data['Close'].iloc[0] - 1).values
            )
            
            return {
                'benchmark_name': benchmark_strategy,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'daily_returns': daily_returns.values
            }
        else:
            # その他のベンチマーク（簡略化）
            return {
                'benchmark_name': benchmark_strategy,
                'total_return': 0.08,  # 8%固定
                'volatility': 0.12,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.06
            }
    
    async def _analyze_backtest_results(self, 
                                      backtest_results: List[BacktestResult], 
                                      benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """バックテスト結果の分析"""
        
        if not backtest_results:
            return {"error": "No backtest results to analyze"}
        
        # 成功したバックテストのフィルタリング
        successful_results = [
            result for result in backtest_results 
            if result.status == BacktestStatus.COMPLETED
        ]
        
        if not successful_results:
            return {"error": "No successful backtest results"}
        
        # 最高パフォーマンス戦略の特定
        best_result = max(
            successful_results, 
            key=lambda x: x.performance_metrics.get('risk_adjusted_return', 0)
        )
        
        # パフォーマンス統計
        performance_stats = self._calculate_performance_statistics(successful_results)
        
        # ベンチマーク比較
        benchmark_comparison = self._compare_with_benchmark(successful_results, benchmark_result)
        
        analysis_report = {
            'analysis_timestamp': datetime.now(),
            'total_combinations_tested': len(backtest_results),
            'successful_combinations': len(successful_results),
            'success_rate': len(successful_results) / len(backtest_results),
            'best_combination': {
                'combination_id': best_result.combination_id,
                'performance_metrics': best_result.performance_metrics,
                'risk_metrics': best_result.risk_metrics
            },
            'performance_statistics': performance_stats,
            'benchmark_comparison': benchmark_comparison,
            'detailed_results': [asdict(result) for result in successful_results]
        }
        
        return analysis_report
    
    def _calculate_performance_statistics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """パフォーマンス統計の計算"""
        
        # 主要指標の統計
        expected_returns = [r.performance_metrics.get('expected_return', 0) for r in results]
        sharpe_ratios = [r.performance_metrics.get('expected_sharpe_ratio', 0) for r in results]
        max_drawdowns = [r.performance_metrics.get('expected_max_drawdown', 0) for r in results]
        
        return {
            'avg_expected_return': np.mean(expected_returns),
            'std_expected_return': np.std(expected_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'best_sharpe_ratio': np.max(sharpe_ratios),
            'consistency_score': 1.0 - np.std(expected_returns) / np.mean(expected_returns) if np.mean(expected_returns) != 0 else 0
        }
    
    def _compare_with_benchmark(self, 
                              results: List[BacktestResult], 
                              benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """ベンチマークとの比較"""
        
        benchmark_return = benchmark_result.get('total_return', 0)
        benchmark_sharpe = benchmark_result.get('sharpe_ratio', 0)
        
        outperforming_strategies = [
            r for r in results 
            if r.performance_metrics.get('expected_return', 0) > benchmark_return
        ]
        
        better_sharpe_strategies = [
            r for r in results 
            if r.performance_metrics.get('expected_sharpe_ratio', 0) > benchmark_sharpe
        ]
        
        return {
            'benchmark_return': benchmark_return,
            'benchmark_sharpe': benchmark_sharpe,
            'outperforming_count': len(outperforming_strategies),
            'outperforming_rate': len(outperforming_strategies) / len(results),
            'better_sharpe_count': len(better_sharpe_strategies),
            'better_sharpe_rate': len(better_sharpe_strategies) / len(results),
            'average_outperformance': np.mean([
                r.performance_metrics.get('expected_return', 0) - benchmark_return 
                for r in results
            ])
        }
    
    async def _generate_comprehensive_report(self, analysis_report: Dict[str, Any]) -> Dict[str, str]:
        """包括的レポートの生成"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "backtest_results", 
            f"composite_backtest_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        report_paths = {}
        
        try:
            # JSONサマリーレポート
            json_path = os.path.join(output_dir, f"backtest_summary_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_report, f, ensure_ascii=False, indent=2, default=str)
            report_paths['json_summary'] = json_path
            
            # 簡易テキストレポート
            txt_path = os.path.join(output_dir, f"backtest_report_{timestamp}.txt")
            self._generate_text_report(analysis_report, txt_path)
            report_paths['text_report'] = txt_path
            
            self.logger.info(f"Reports generated in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
        
        return report_paths
    
    def _generate_text_report(self, analysis_report: Dict[str, Any], output_path: str):
        """テキストレポートの生成"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("4-2-2 複合戦略バックテスト結果レポート\n")
            f.write("="*80 + "\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # サマリー
            f.write("■ 実行サマリー\n")
            f.write(f"テスト組み合わせ数: {analysis_report.get('total_combinations_tested', 0)}\n")
            f.write(f"成功組み合わせ数: {analysis_report.get('successful_combinations', 0)}\n")
            f.write(f"成功率: {analysis_report.get('success_rate', 0):.1%}\n\n")
            
            # 最優秀戦略
            best_combo = analysis_report.get('best_combination', {})
            if best_combo:
                f.write("■ 最優秀戦略組み合わせ\n")
                f.write(f"組み合わせID: {best_combo.get('combination_id', 'N/A')}\n")
                
                metrics = best_combo.get('performance_metrics', {})
                f.write(f"期待リターン: {metrics.get('expected_return', 0):.3f}\n")
                f.write(f"期待シャープレシオ: {metrics.get('expected_sharpe_ratio', 0):.3f}\n")
                f.write(f"期待最大ドローダウン: {metrics.get('expected_max_drawdown', 0):.3f}\n")
                f.write(f"リスク調整後リターン: {metrics.get('risk_adjusted_return', 0):.3f}\n\n")
            
            # パフォーマンス統計
            perf_stats = analysis_report.get('performance_statistics', {})
            if perf_stats:
                f.write("■ パフォーマンス統計\n")
                f.write(f"平均期待リターン: {perf_stats.get('avg_expected_return', 0):.3f}\n")
                f.write(f"平均シャープレシオ: {perf_stats.get('avg_sharpe_ratio', 0):.3f}\n")
                f.write(f"平均最大ドローダウン: {perf_stats.get('avg_max_drawdown', 0):.3f}\n")
                f.write(f"一貫性スコア: {perf_stats.get('consistency_score', 0):.3f}\n\n")
            
            # ベンチマーク比較
            benchmark_comp = analysis_report.get('benchmark_comparison', {})
            if benchmark_comp:
                f.write("■ ベンチマーク比較\n")
                f.write(f"ベンチマーク超過戦略数: {benchmark_comp.get('outperforming_count', 0)}\n")
                f.write(f"ベンチマーク超過率: {benchmark_comp.get('outperforming_rate', 0):.1%}\n")
                f.write(f"平均超過リターン: {benchmark_comp.get('average_outperformance', 0):.3f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("レポート生成完了\n")

# 使用例とテスト関数
async def test_composite_backtest_engine():
    """テスト関数"""
    logger.info("Starting CompositeStrategyBacktestEngine test")
    
    # エンジンの初期化
    engine = CompositeStrategyBacktestEngine()
    
    # テスト用戦略組み合わせ
    test_combinations = [
        {
            "combination_id": "test_trend_momentum",
            "name": "テストトレンド・モメンタム",
            "strategies": [
                {"strategy_name": "VWAP_Breakout", "weight": 0.6},
                {"strategy_name": "Momentum_Investing", "weight": 0.4}
            ]
        },
        {
            "combination_id": "test_conservative",
            "name": "テスト保守戦略",
            "strategies": [
                {"strategy_name": "VWAP_Bounce", "weight": 0.5},
                {"strategy_name": "gc_strategy_signal", "weight": 0.5}
            ]
        }
    ]
    
    # テスト期間
    test_period = (
        datetime.now() - timedelta(days=365),
        datetime.now() - timedelta(days=1)
    )
    
    # バックテストの実行
    result = await engine.run_composite_backtest(
        strategy_combinations=test_combinations,
        test_period=test_period,
        benchmark_strategy="buy_and_hold"
    )
    
    logger.info(f"Test completed: {result.get('status')}")
    logger.info(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    return result

if __name__ == "__main__":
    # テスト実行
    import asyncio
    result = asyncio.run(test_composite_backtest_engine())
    print(f"Test result: {result['status']}")
