"""
DSSMS Phase 2 Task 2.1: 統合パフォーマンス計算機
DSSMSエンジンを優先利用した統合システムのパフォーマンス計算

主要機能:
1. DSSMSエンジンを使用した高精度パフォーマンス計算
2. 統合システム（DSSMS+既存戦略）のメトリクス算出
3. 比較分析（DSSMS単体、戦略単体、統合システム）
4. リアルタイムパフォーマンス追跡
5. 包括的レポート生成

設計方針:
- DSSMSパフォーマンス計算エンジンの再利用
- 統合システム独自メトリクスの追加
- エラーハンドリングとフォールバック機能
- 履歴データとリアルタイムデータの統合
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import traceback

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# DSSMSパフォーマンス計算コンポーネント
try:
    from src.dssms.dssms_performance_calculator_v2 import DSSMSPerformanceCalculatorV2
    from src.dssms.dssms_portfolio_calculator_v2 import DSSMSPortfolioCalculatorV2
    from src.dssms.performance_calculation_bridge import PerformanceCalculationBridge
except ImportError as e:
    print(f"DSSMS performance components import warning: {e}")

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class IntegratedMetrics:
    """統合メトリクス"""
    # 基本パフォーマンス
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # 統合システム特有メトリクス
    dssms_contribution: float  # DSSMS決定による貢献度
    strategy_contribution: float  # 戦略決定による貢献度
    hybrid_efficiency: float  # ハイブリッド効率性
    
    # 切替メトリクス
    total_switches: int
    symbol_switches: int
    strategy_switches: int
    switch_success_rate: float
    
    # リスクメトリクス
    volatility: float
    downside_volatility: float
    var_95: float
    cvar_95: float
    
    # 取引メトリクス
    total_trades: int
    win_rate: float
    profit_factor: float
    average_trade_return: float

@dataclass
class PerformanceComparison:
    """パフォーマンス比較"""
    integrated_metrics: IntegratedMetrics
    dssms_only_metrics: Optional[IntegratedMetrics]
    strategy_only_metrics: Optional[IntegratedMetrics]
    benchmark_metrics: Optional[IntegratedMetrics]
    
    improvement_vs_dssms: Dict[str, float]
    improvement_vs_strategy: Dict[str, float]
    improvement_vs_benchmark: Dict[str, float]

class IntegratedPerformanceCalculator:
    """
    統合パフォーマンス計算機
    
    DSSMSエンジンを優先利用して統合システムの
    包括的なパフォーマンス分析を提供します。
    """
    
    def __init__(self, use_dssms_engine: bool = True):
        """初期化"""
        self.use_dssms_engine = use_dssms_engine
        self.dssms_calculator = None
        self.portfolio_calculator = None
        self.calculation_bridge = None
        
        # パフォーマンス履歴
        self.performance_history = []
        self.calculation_cache = {}
        
        # DSSMSエンジン初期化
        if self.use_dssms_engine:
            self._initialize_dssms_engine()
        
        logger.info("Integrated Performance Calculator initialized")
    
    def _initialize_dssms_engine(self):
        """DSSMSエンジン初期化"""
        try:
            self.dssms_calculator = DSSMSPerformanceCalculatorV2()
            self.portfolio_calculator = DSSMSPortfolioCalculatorV2()
            self.calculation_bridge = PerformanceCalculationBridge()
            logger.info("DSSMS performance engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DSSMS engine: {e}")
            self.use_dssms_engine = False
    
    def calculate_comprehensive_performance(self,
                                          results: Dict[str, Any],
                                          initial_capital: float,
                                          benchmark_data: Optional[pd.DataFrame] = None) -> IntegratedMetrics:
        """包括的パフォーマンス計算"""
        try:
            logger.info("Calculating comprehensive performance metrics")
            
            # 基本データ準備
            daily_values = pd.DataFrame(results.get('daily_values', []))
            trades = pd.DataFrame(results.get('trades', []))
            
            if daily_values.empty:
                logger.warning("No daily values data available")
                return self._create_empty_metrics()
            
            # DSSMSエンジン使用時
            if self.use_dssms_engine and self.dssms_calculator:
                return self._calculate_with_dssms_engine(
                    daily_values, trades, initial_capital, benchmark_data
                )
            else:
                return self._calculate_with_fallback_engine(
                    daily_values, trades, initial_capital, benchmark_data
                )
                
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive performance: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_metrics()
    
    def _calculate_with_dssms_engine(self,
                                   daily_values: pd.DataFrame,
                                   trades: pd.DataFrame,
                                   initial_capital: float,
                                   benchmark_data: Optional[pd.DataFrame]) -> IntegratedMetrics:
        """DSSMSエンジンでの計算"""
        try:
            # ポートフォリオ価値系列
            portfolio_values = daily_values['portfolio_value'].values
            dates = pd.to_datetime(daily_values['date'])
            
            # DSSMSエンジンでの基本メトリクス計算
            dssms_metrics = self.dssms_calculator.calculate_performance_metrics(
                portfolio_values=portfolio_values,
                dates=dates,
                initial_capital=initial_capital
            )
            
            # 統合システム特有メトリクス追加
            integration_metrics = self._calculate_integration_metrics(trades, daily_values)
            
            # 切替メトリクス
            switch_metrics = self._calculate_switch_metrics(trades)
            
            # 取引メトリクス
            trade_metrics = self._calculate_trade_metrics(trades, initial_capital)
            
            # 統合メトリクス作成
            return IntegratedMetrics(
                # 基本パフォーマンス（DSSMSエンジン）
                total_return=dssms_metrics.get('total_return', 0.0),
                annualized_return=dssms_metrics.get('annualized_return', 0.0),
                sharpe_ratio=dssms_metrics.get('sharpe_ratio', 0.0),
                sortino_ratio=dssms_metrics.get('sortino_ratio', 0.0),
                max_drawdown=dssms_metrics.get('max_drawdown', 0.0),
                calmar_ratio=dssms_metrics.get('calmar_ratio', 0.0),
                
                # 統合システム特有
                dssms_contribution=integration_metrics.get('dssms_contribution', 0.0),
                strategy_contribution=integration_metrics.get('strategy_contribution', 0.0),
                hybrid_efficiency=integration_metrics.get('hybrid_efficiency', 0.0),
                
                # 切替メトリクス
                total_switches=switch_metrics.get('total_switches', 0),
                symbol_switches=switch_metrics.get('symbol_switches', 0),
                strategy_switches=switch_metrics.get('strategy_switches', 0),
                switch_success_rate=switch_metrics.get('switch_success_rate', 0.0),
                
                # リスクメトリクス（DSSMSエンジン）
                volatility=dssms_metrics.get('volatility', 0.0),
                downside_volatility=dssms_metrics.get('downside_volatility', 0.0),
                var_95=dssms_metrics.get('var_95', 0.0),
                cvar_95=dssms_metrics.get('cvar_95', 0.0),
                
                # 取引メトリクス
                total_trades=trade_metrics.get('total_trades', 0),
                win_rate=trade_metrics.get('win_rate', 0.0),
                profit_factor=trade_metrics.get('profit_factor', 0.0),
                average_trade_return=trade_metrics.get('average_trade_return', 0.0)
            )
            
        except Exception as e:
            logger.error(f"DSSMS engine calculation failed: {e}")
            return self._calculate_with_fallback_engine(daily_values, trades, initial_capital, benchmark_data)
    
    def _calculate_with_fallback_engine(self,
                                      daily_values: pd.DataFrame,
                                      trades: pd.DataFrame,
                                      initial_capital: float,
                                      benchmark_data: Optional[pd.DataFrame]) -> IntegratedMetrics:
        """フォールバックエンジンでの計算"""
        try:
            logger.info("Using fallback performance calculation engine")
            
            # ポートフォリオ価値系列
            portfolio_values = daily_values['portfolio_value'].values
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[~np.isnan(returns)]
            
            # 基本メトリクス
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # ドローダウン
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # 統合メトリクス
            integration_metrics = self._calculate_integration_metrics(trades, daily_values)
            switch_metrics = self._calculate_switch_metrics(trades)
            trade_metrics = self._calculate_trade_metrics(trades, initial_capital)
            
            return IntegratedMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=0.0,  # 簡易版では計算しない
                max_drawdown=abs(max_drawdown),
                calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
                
                dssms_contribution=integration_metrics.get('dssms_contribution', 0.0),
                strategy_contribution=integration_metrics.get('strategy_contribution', 0.0),
                hybrid_efficiency=integration_metrics.get('hybrid_efficiency', 0.0),
                
                total_switches=switch_metrics.get('total_switches', 0),
                symbol_switches=switch_metrics.get('symbol_switches', 0),
                strategy_switches=switch_metrics.get('strategy_switches', 0),
                switch_success_rate=switch_metrics.get('switch_success_rate', 0.0),
                
                volatility=volatility,
                downside_volatility=0.0,
                var_95=np.percentile(returns, 5) if len(returns) > 0 else 0,
                cvar_95=np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns) > 0 else 0,
                
                total_trades=trade_metrics.get('total_trades', 0),
                win_rate=trade_metrics.get('win_rate', 0.0),
                profit_factor=trade_metrics.get('profit_factor', 0.0),
                average_trade_return=trade_metrics.get('average_trade_return', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Fallback calculation failed: {e}")
            return self._create_empty_metrics()
    
    def _calculate_integration_metrics(self,
                                     trades: pd.DataFrame,
                                     daily_values: pd.DataFrame) -> Dict[str, float]:
        """統合システム特有メトリクス計算"""
        try:
            if trades.empty:
                return {'dssms_contribution': 0.0, 'strategy_contribution': 0.0, 'hybrid_efficiency': 0.0}
            
            # システム別利益計算
            dssms_trades = trades[trades.get('system', '') == 'dssms_only']
            strategy_trades = trades[trades.get('system', '').str.contains('strategy', na=False)]
            
            dssms_profit = dssms_trades['profit'].sum() if 'profit' in dssms_trades.columns and not dssms_trades.empty else 0
            strategy_profit = strategy_trades['profit'].sum() if 'profit' in strategy_trades.columns and not strategy_trades.empty else 0
            total_profit = trades['profit'].sum() if 'profit' in trades.columns else 0
            
            # 貢献度計算
            if total_profit != 0:
                dssms_contribution = dssms_profit / total_profit
                strategy_contribution = strategy_profit / total_profit
            else:
                dssms_contribution = 0.0
                strategy_contribution = 0.0
            
            # ハイブリッド効率性（統合効果の測定）
            # 単純和より良いパフォーマンスが出ているかの指標
            expected_combined = dssms_profit + strategy_profit
            actual_performance = total_profit
            hybrid_efficiency = actual_performance / expected_combined if expected_combined != 0 else 1.0
            
            return {
                'dssms_contribution': dssms_contribution,
                'strategy_contribution': strategy_contribution,
                'hybrid_efficiency': hybrid_efficiency
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate integration metrics: {e}")
            return {'dssms_contribution': 0.0, 'strategy_contribution': 0.0, 'hybrid_efficiency': 0.0}
    
    def _calculate_switch_metrics(self, trades: pd.DataFrame) -> Dict[str, int]:
        """切替メトリクス計算"""
        try:
            if trades.empty:
                return {'total_switches': 0, 'symbol_switches': 0, 'strategy_switches': 0, 'switch_success_rate': 0.0}
            
            # システム切替回数
            system_changes = 0
            if 'system' in trades.columns:
                prev_system = None
                for system in trades['system']:
                    if prev_system is not None and system != prev_system:
                        system_changes += 1
                    prev_system = system
            
            # 戦略切替回数
            strategy_changes = 0
            if 'strategy' in trades.columns:
                prev_strategy = None
                for strategy in trades['strategy']:
                    if prev_strategy is not None and strategy != prev_strategy:
                        strategy_changes += 1
                    prev_strategy = strategy
            
            # 銘柄切替回数
            symbol_changes = 0
            if 'symbol' in trades.columns:
                prev_symbol = None
                for symbol in trades['symbol']:
                    if prev_symbol is not None and symbol != prev_symbol:
                        symbol_changes += 1
                    prev_symbol = symbol
            
            # 切替成功率（切替後の取引で利益が出た割合）
            switch_success_rate = 0.0
            if 'profit' in trades.columns and len(trades) > 1:
                profitable_trades = len(trades[trades['profit'] > 0])
                switch_success_rate = profitable_trades / len(trades)
            
            return {
                'total_switches': system_changes + strategy_changes,
                'symbol_switches': symbol_changes,
                'strategy_switches': strategy_changes,
                'switch_success_rate': switch_success_rate
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate switch metrics: {e}")
            return {'total_switches': 0, 'symbol_switches': 0, 'strategy_switches': 0, 'switch_success_rate': 0.0}
    
    def _calculate_trade_metrics(self,
                               trades: pd.DataFrame,
                               initial_capital: float) -> Dict[str, float]:
        """取引メトリクス計算"""
        try:
            if trades.empty or 'profit' not in trades.columns:
                return {'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'average_trade_return': 0.0}
            
            # 売買ペアのみ（buy-sell）
            sell_trades = trades[trades['action'] == 'sell']
            total_trades = len(sell_trades)
            
            if total_trades == 0:
                return {'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'average_trade_return': 0.0}
            
            # 勝率
            winning_trades = len(sell_trades[sell_trades['profit'] > 0])
            win_rate = winning_trades / total_trades
            
            # プロフィットファクター
            gross_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
            gross_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # 平均取引リターン
            average_trade_return = sell_trades['profit'].mean() / initial_capital
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_trade_return': average_trade_return
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate trade metrics: {e}")
            return {'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'average_trade_return': 0.0}
    
    def _create_empty_metrics(self) -> IntegratedMetrics:
        """空のメトリクス作成"""
        return IntegratedMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            dssms_contribution=0.0,
            strategy_contribution=0.0,
            hybrid_efficiency=0.0,
            total_switches=0,
            symbol_switches=0,
            strategy_switches=0,
            switch_success_rate=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            average_trade_return=0.0
        )
    
    def compare_performance(self,
                          integrated_results: Dict[str, Any],
                          dssms_results: Optional[Dict[str, Any]] = None,
                          strategy_results: Optional[Dict[str, Any]] = None,
                          benchmark_data: Optional[pd.DataFrame] = None,
                          initial_capital: float = 1000000) -> PerformanceComparison:
        """パフォーマンス比較分析"""
        try:
            logger.info("Performing comprehensive performance comparison")
            
            # 統合システムメトリクス
            integrated_metrics = self.calculate_comprehensive_performance(
                integrated_results, initial_capital, benchmark_data
            )
            
            # DSSMS単体メトリクス
            dssms_metrics = None
            if dssms_results:
                dssms_metrics = self.calculate_comprehensive_performance(
                    dssms_results, initial_capital, benchmark_data
                )
            
            # 戦略単体メトリクス
            strategy_metrics = None
            if strategy_results:
                strategy_metrics = self.calculate_comprehensive_performance(
                    strategy_results, initial_capital, benchmark_data
                )
            
            # ベンチマークメトリクス
            benchmark_metrics = None
            if benchmark_data is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(
                    benchmark_data, initial_capital
                )
            
            # 改善度計算
            improvement_vs_dssms = self._calculate_improvement(integrated_metrics, dssms_metrics)
            improvement_vs_strategy = self._calculate_improvement(integrated_metrics, strategy_metrics)
            improvement_vs_benchmark = self._calculate_improvement(integrated_metrics, benchmark_metrics)
            
            return PerformanceComparison(
                integrated_metrics=integrated_metrics,
                dssms_only_metrics=dssms_metrics,
                strategy_only_metrics=strategy_metrics,
                benchmark_metrics=benchmark_metrics,
                improvement_vs_dssms=improvement_vs_dssms,
                improvement_vs_strategy=improvement_vs_strategy,
                improvement_vs_benchmark=improvement_vs_benchmark
            )
            
        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_improvement(self,
                             integrated: IntegratedMetrics,
                             baseline: Optional[IntegratedMetrics]) -> Dict[str, float]:
        """改善度計算"""
        if baseline is None:
            return {}
        
        try:
            improvements = {}
            
            # リターン改善
            if baseline.total_return != 0:
                improvements['total_return'] = (integrated.total_return - baseline.total_return) / abs(baseline.total_return)
            
            # シャープレシオ改善
            if baseline.sharpe_ratio != 0:
                improvements['sharpe_ratio'] = (integrated.sharpe_ratio - baseline.sharpe_ratio) / abs(baseline.sharpe_ratio)
            
            # ドローダウン改善（小さいほど良い）
            if baseline.max_drawdown != 0:
                improvements['max_drawdown'] = (baseline.max_drawdown - integrated.max_drawdown) / baseline.max_drawdown
            
            # 勝率改善
            if baseline.win_rate != 0:
                improvements['win_rate'] = (integrated.win_rate - baseline.win_rate) / baseline.win_rate
            
            return improvements
            
        except Exception as e:
            logger.warning(f"Failed to calculate improvement: {e}")
            return {}
    
    def _calculate_benchmark_metrics(self,
                                   benchmark_data: pd.DataFrame,
                                   initial_capital: float) -> IntegratedMetrics:
        """ベンチマークメトリクス計算"""
        try:
            # ベンチマークリターン計算
            benchmark_prices = benchmark_data['Adj Close']
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # 基本メトリクス
            total_return = (benchmark_prices.iloc[-1] - benchmark_prices.iloc[0]) / benchmark_prices.iloc[0]
            annualized_return = (1 + total_return) ** (252 / len(benchmark_returns)) - 1
            volatility = benchmark_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # ドローダウン
            cumulative = (1 + benchmark_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return IntegratedMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=0.0,
                max_drawdown=abs(max_drawdown),
                calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
                dssms_contribution=0.0,
                strategy_contribution=0.0,
                hybrid_efficiency=1.0,
                total_switches=0,
                symbol_switches=0,
                strategy_switches=0,
                switch_success_rate=0.0,
                volatility=volatility,
                downside_volatility=0.0,
                var_95=benchmark_returns.quantile(0.05),
                cvar_95=benchmark_returns[benchmark_returns <= benchmark_returns.quantile(0.05)].mean(),
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                average_trade_return=0.0
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate benchmark metrics: {e}")
            return self._create_empty_metrics()

# 使用例とテスト関数
def test_performance_calculator():
    """パフォーマンス計算機のテスト"""
    print("=== Integrated Performance Calculator Test ===")
    
    # 計算機初期化
    calculator = IntegratedPerformanceCalculator(use_dssms_engine=True)
    
    # テストデータ生成
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    portfolio_values = 1000000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
    
    daily_values = [
        {'date': date, 'portfolio_value': value, 'cash': value * 0.1, 'position_value': value * 0.9}
        for date, value in zip(dates, portfolio_values)
    ]
    
    trades = []
    for i in range(0, len(dates)-10, 20):
        trades.extend([
            {'date': dates[i], 'symbol': f'Stock{i%3}', 'action': 'buy', 'shares': 100, 'price': 100 + i, 'value': 10000 + i*100, 'system': 'dssms_only'},
            {'date': dates[i+5], 'symbol': f'Stock{i%3}', 'action': 'sell', 'shares': 100, 'price': 105 + i, 'value': 10500 + i*100, 'profit': 500, 'system': 'dssms_only'}
        ])
    
    test_results = {
        'daily_values': daily_values,
        'trades': trades,
        'final_portfolio_value': portfolio_values[-1]
    }
    
    try:
        # パフォーマンス計算
        metrics = calculator.calculate_comprehensive_performance(
            results=test_results,
            initial_capital=1000000
        )
        
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Annualized Return: {metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Win Rate: {metrics.win_rate:.2%}")
        print(f"DSSMS Contribution: {metrics.dssms_contribution:.2%}")
        print(f"Hybrid Efficiency: {metrics.hybrid_efficiency:.3f}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_performance_calculator()
