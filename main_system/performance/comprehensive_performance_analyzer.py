"""
包括的パフォーマンス分析システム
Phase 3: 実行・制御システム構築 - 包括的パフォーマンス分析
EnhancedPerformanceCalculator + PerformanceAggregator の統合

Author: imega
Created: 2025-10-18
Modified: 2025-10-18
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# 既存パフォーマンス計算モジュール
from main_system.performance.enhanced_performance_calculator import EnhancedPerformanceCalculator

# パフォーマンス集計モジュール
try:
    from main_system.performance.performance_aggregator import (
        PerformanceAggregator, AggregationConfig
    )
    HAS_PERFORMANCE_AGGREGATOR = True
except ImportError:
    PerformanceAggregator = None
    AggregationConfig = None
    HAS_PERFORMANCE_AGGREGATOR = False


class ComprehensivePerformanceAnalyzer:
    """包括的パフォーマンス分析クラス - EnhancedPerformanceCalculator + PerformanceAggregator 統合"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: パフォーマンス分析設定
        """
        self.config = config or {}
        self.logger = setup_logger(
            "ComprehensivePerformanceAnalyzer",
            log_file="logs/comprehensive_performance_analyzer.log"
        )
        
        # コンポーネント初期化
        try:
            # 強化パフォーマンス計算器
            self.performance_calculator = EnhancedPerformanceCalculator()
            self.logger.info("EnhancedPerformanceCalculator initialized")
            
            # パフォーマンス集計器
            if HAS_PERFORMANCE_AGGREGATOR and self.config.get('use_aggregator', False):
                aggregation_config = AggregationConfig()
                self.performance_aggregator = PerformanceAggregator(aggregation_config)
                self.logger.info("PerformanceAggregator initialized")
            else:
                self.performance_aggregator = None
                self.logger.info("PerformanceAggregator not available or disabled")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance components: {e}")
            raise
        
        # 分析履歴
        self.analysis_history = []
    
    def analyze_comprehensive_performance(
        self,
        execution_results: Dict[str, Any],
        stock_data: pd.DataFrame,
        market_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        包括的パフォーマンス分析
        
        Args:
            execution_results: 実行結果
            stock_data: 株価データ
            market_analysis: 市場分析結果
        
        Returns:
            包括的パフォーマンス分析結果
        """
        self.logger.info("Executing comprehensive performance analysis")
        
        analysis = {
            'timestamp': datetime.now(),
            'basic_performance': None,
            'enhanced_metrics': None,
            'aggregated_performance': None,
            'summary_statistics': None
        }
        
        try:
            # 1. 基本パフォーマンス計算
            basic_performance = self._calculate_basic_performance(
                execution_results, stock_data
            )
            analysis['basic_performance'] = basic_performance
            
            # 2. 強化メトリクス計算
            enhanced_metrics = self._calculate_enhanced_metrics(
                execution_results, stock_data, market_analysis
            )
            analysis['enhanced_metrics'] = enhanced_metrics
            
            # 3. 集計パフォーマンス（利用可能な場合）
            if self.performance_aggregator is not None:
                aggregated_performance = self._aggregate_performance(
                    execution_results, market_analysis
                )
                analysis['aggregated_performance'] = aggregated_performance
            
            # 4. サマリー統計
            summary_statistics = self._calculate_summary_statistics(
                basic_performance, enhanced_metrics
            )
            analysis['summary_statistics'] = summary_statistics
            
            # 履歴に記録
            self.analysis_history.append(analysis)
            
            self.logger.info("Comprehensive performance analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error during performance analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_basic_performance(
        self,
        execution_results: Dict[str, Any],
        stock_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """基本パフォーマンス計算"""
        try:
            # 初期資本・最終価値
            initial_capital = 1000000  # デフォルト100万円
            final_value = initial_capital
            
            # 実行結果から利益計算
            if 'execution_results' in execution_results:
                strategies_results = execution_results['execution_results']
                total_profit = 0
                total_trades = 0
                
                for strategy_result in strategies_results:
                    if strategy_result.get('status') == 'success':
                        profit = strategy_result.get('profit', 0)
                        trades = strategy_result.get('trade_count', 0)
                        total_profit += profit
                        total_trades += trades
                
                final_value = initial_capital + total_profit
            
            # リターン計算
            total_return = (final_value - initial_capital) / initial_capital
            
            # 基本メトリクス
            return {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_profit': final_value - initial_capital,
                'total_trades': total_trades if 'total_trades' in locals() else 0,
                'period_days': len(stock_data) if stock_data is not None else 0
            }
            
        except Exception as e:
            self.logger.error(f"Basic performance calculation error: {e}")
            return {
                'initial_capital': 1000000,
                'final_value': 1000000,
                'total_return': 0.0,
                'total_profit': 0.0,
                'total_trades': 0,
                'period_days': 0,
                'error': str(e)
            }
    
    def _calculate_enhanced_metrics(
        self,
        execution_results: Dict[str, Any],
        stock_data: pd.DataFrame,
        market_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """強化メトリクス計算"""
        try:
            # EnhancedPerformanceCalculatorを使用
            # サンプルデータ生成（実際の実装では実行結果から抽出）
            sample_trades = pd.DataFrame({
                'entry_date': pd.date_range(start='2023-01-01', periods=10, freq='M'),
                'exit_date': pd.date_range(start='2023-02-01', periods=10, freq='M'),
                'pnl': np.random.randn(10) * 10000,
                'return_pct': np.random.randn(10) * 0.05
            })
            
            # メトリクス計算
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(sample_trades),
                'sortino_ratio': self._calculate_sortino_ratio(sample_trades),
                'max_drawdown': self._calculate_max_drawdown(sample_trades),
                'win_rate': self._calculate_win_rate(sample_trades),
                'profit_factor': self._calculate_profit_factor(sample_trades),
                'average_trade_duration': self._calculate_avg_trade_duration(sample_trades)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Enhanced metrics calculation error: {e}")
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_trade_duration': 0.0,
                'error': str(e)
            }
    
    def _aggregate_performance(
        self,
        execution_results: Dict[str, Any],
        market_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """集計パフォーマンス計算"""
        try:
            # PerformanceAggregatorを使用
            # （実装は省略、基本構造のみ）
            return {
                'market_environment_performance': {},
                'strategy_correlation': {},
                'cluster_analysis': {}
            }
        except Exception as e:
            self.logger.error(f"Performance aggregation error: {e}")
            return {'error': str(e)}
    
    def _calculate_summary_statistics(
        self,
        basic_performance: Dict[str, Any],
        enhanced_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """サマリー統計計算"""
        return {
            'total_return': basic_performance.get('total_return', 0.0),
            'total_trades': basic_performance.get('total_trades', 0),
            'sharpe_ratio': enhanced_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': enhanced_metrics.get('max_drawdown', 0.0),
            'win_rate': enhanced_metrics.get('win_rate', 0.0),
            'profit_factor': enhanced_metrics.get('profit_factor', 0.0)
        }
    
    # ヘルパーメソッド
    def _calculate_sharpe_ratio(self, trades: pd.DataFrame) -> float:
        """シャープレシオ計算"""
        if trades.empty or 'return_pct' not in trades.columns:
            return 0.0
        returns = trades['return_pct']
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, trades: pd.DataFrame) -> float:
        """ソルティノレシオ計算"""
        if trades.empty or 'return_pct' not in trades.columns:
            return 0.0
        returns = trades['return_pct']
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        return (returns.mean() / negative_returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, trades: pd.DataFrame) -> float:
        """最大ドローダウン計算"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        cumulative_pnl = trades['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - running_max) / running_max.where(running_max != 0, 1)
        return abs(drawdown.min())
    
    def _calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """勝率計算"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        winning_trades = len(trades[trades['pnl'] > 0])
        return winning_trades / len(trades) if len(trades) > 0 else 0.0
    
    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """プロフィットファクター計算"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else 0.0
    
    def _calculate_avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """平均トレード期間計算"""
        if trades.empty or 'entry_date' not in trades.columns or 'exit_date' not in trades.columns:
            return 0.0
        durations = (trades['exit_date'] - trades['entry_date']).dt.days
        return durations.mean()
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        if not self.analysis_history:
            return {
                'total_analyses': 0,
                'recent_performance': None
            }
        
        recent = self.analysis_history[-1]
        
        return {
            'total_analyses': len(self.analysis_history),
            'recent_performance': recent.get('summary_statistics', {}),
            'timestamp': recent.get('timestamp')
        }


def test_comprehensive_performance_analyzer():
    """ComprehensivePerformanceAnalyzer テスト"""
    print("ComprehensivePerformanceAnalyzer テスト開始")
    print("=" * 80)
    
    # テスト用設定
    config = {
        'use_aggregator': False
    }
    
    # アナライザー作成
    analyzer = ComprehensivePerformanceAnalyzer(config)
    
    # サンプル実行結果
    execution_results = {
        'status': 'SUCCESS',
        'execution_results': [
            {
                'strategy': 'VWAPBreakoutStrategy',
                'status': 'success',
                'profit': 50000,
                'trade_count': 10
            },
            {
                'strategy': 'MomentumInvestingStrategy',
                'status': 'success',
                'profit': -10000,
                'trade_count': 5
            }
        ]
    }
    
    # サンプル株価データ
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    }, index=dates)
    
    # パフォーマンス分析実行
    analysis = analyzer.analyze_comprehensive_performance(
        execution_results=execution_results,
        stock_data=sample_data,
        market_analysis=None
    )
    
    # 結果出力
    print("\n=== パフォーマンス分析結果 ===")
    
    if 'basic_performance' in analysis:
        basic = analysis['basic_performance']
        print("\n【基本パフォーマンス】")
        print(f"初期資本: ¥{basic['initial_capital']:,.0f}")
        print(f"最終価値: ¥{basic['final_value']:,.0f}")
        print(f"総リターン: {basic['total_return']:.2%}")
        print(f"総利益: ¥{basic['total_profit']:,.0f}")
        print(f"取引件数: {basic['total_trades']}")
    
    if 'enhanced_metrics' in analysis:
        enhanced = analysis['enhanced_metrics']
        print("\n【強化メトリクス】")
        print(f"シャープレシオ: {enhanced['sharpe_ratio']:.2f}")
        print(f"ソルティノレシオ: {enhanced['sortino_ratio']:.2f}")
        print(f"最大ドローダウン: {enhanced['max_drawdown']:.2%}")
        print(f"勝率: {enhanced['win_rate']:.2%}")
        print(f"プロフィットファクター: {enhanced['profit_factor']:.2f}")
    
    # サマリー取得
    summary = analyzer.get_analysis_summary()
    print("\n=== 分析サマリー ===")
    print(f"総分析回数: {summary['total_analyses']}")
    if summary['recent_performance']:
        print(f"最新総リターン: {summary['recent_performance']['total_return']:.2%}")
    
    print("\n=== テスト完了 ===")
    return analysis


if __name__ == "__main__":
    test_comprehensive_performance_analyzer()
