"""
main.py - 次世代マルチ戦略バックテストシステム（簡易版 Phase 1）
シンプルなエントリーポイント - 統合候補モジュール活用版

Phase 1実装: 基本統合システム
- MarketAnalyzer: 市場分析
- DynamicStrategySelector: 動的戦略選択
- IntegratedExecutionManager: 統合実行管理
- UnifiedRiskManager: 統合リスク管理
- ComprehensivePerformanceAnalyzer: 包括的パフォーマンス分析
- ComprehensiveReporter: 包括的レポート生成

Author: imega
Created: 2025-10-18
Modified: 2025-10-18
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ロガー設定
from config.logger_config import setup_logger

# 統合システムインポート
from main_system.market_analysis.market_analyzer import MarketAnalyzer
from main_system.strategy_selection.dynamic_strategy_selector import DynamicStrategySelector
from main_system.execution_control.integrated_execution_manager import IntegratedExecutionManager
from main_system.risk_management.unified_risk_manager import UnifiedRiskManager
from main_system.performance.comprehensive_performance_analyzer import ComprehensivePerformanceAnalyzer
from main_system.reporting.comprehensive_reporter import ComprehensiveReporter

# データ取得用（簡易版）
import pandas as pd
import numpy as np


class MainSystemController:
    """メインシステムコントローラー - 全統合システムの制御"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: システム設定
        """
        self.config = config or {}
        self.logger = setup_logger(
            "MainSystemController",
            log_file="logs/main_system_controller.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info("MainSystemController initialization started")
        self.logger.info("=" * 80)
        
        try:
            # 分析・選択システム初期化
            self.logger.info("Initializing Market Analyzer...")
            self.market_analyzer = MarketAnalyzer()
            
            self.logger.info("Initializing Dynamic Strategy Selector...")
            self.strategy_selector = DynamicStrategySelector()
            
            # 実行・制御システム初期化
            self.logger.info("Initializing Integrated Execution Manager...")
            self.execution_manager = IntegratedExecutionManager(
                config=self.config.get('execution', {})
            )
            
            self.logger.info("Initializing Unified Risk Manager...")
            self.risk_manager = UnifiedRiskManager(
                config=self.config.get('risk_management', {})
            )
            
            # パフォーマンス・レポートシステム初期化
            self.logger.info("Initializing Comprehensive Performance Analyzer...")
            self.performance_analyzer = ComprehensivePerformanceAnalyzer(
                config=self.config.get('performance', {})
            )
            
            self.logger.info("Initializing Comprehensive Reporter...")
            self.reporter = ComprehensiveReporter()
            
            self.logger.info("=" * 80)
            self.logger.info("MainSystemController initialization completed successfully")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MainSystemController: {e}")
            raise
    
    def execute_comprehensive_backtest(
        self,
        ticker: str,
        stock_data: Optional[pd.DataFrame] = None,
        index_data: Optional[pd.DataFrame] = None,
        days_back: int = 365
    ) -> Dict[str, Any]:
        """
        包括的バックテスト実行（Phase 4.2: リアルデータ対応版）
        
        Args:
            ticker: ティッカーシンボル
            stock_data: 株価データ（Noneの場合はyfinanceから取得）
            index_data: インデックスデータ（Noneの場合はyfinanceから取得）
            days_back: 取得日数
        
        Returns:
            包括的バックテスト結果
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting comprehensive backtest for {ticker}")
        self.logger.info("=" * 80)
        
        try:
            # 1. データ取得（Phase 4.2: yfinanceから実データ取得）
            self.logger.info(f"[STEP 1/7] データ取得開始: {ticker}")
            if stock_data is None:
                stock_data, index_data = self._get_real_data(ticker, days_back)
            
            # 2. 市場分析・トレンド判定
            self.logger.info(f"[STEP 2/7] 市場分析実行")
            market_analysis = self.market_analyzer.comprehensive_market_analysis(
                stock_data, index_data
            )
            
            # 3. 動的戦略選択・重み計算
            self.logger.info(f"[STEP 3/7] 動的戦略選択実行")
            strategy_selection = self.strategy_selector.select_optimal_strategies(
                market_analysis, stock_data
            )
            
            # 4. リスク評価・実行制御
            self.logger.info(f"[STEP 4/7] リスク評価・実行制御")
            risk_assessment = self.risk_manager.assess_execution_risk(
                strategy_selection, stock_data, portfolio_value=1000000
            )
            
            # リスク評価で実行拒否の場合は中断
            if not risk_assessment.get('execution_approval', True):
                self.logger.warning("Execution denied by risk assessment")
                return {
                    'status': 'EXECUTION_DENIED',
                    'ticker': ticker,
                    'risk_assessment': risk_assessment,
                    'message': 'Execution denied due to high risk level'
                }
            
            # 5. 戦略実行（動的選択・重み付け）
            self.logger.info(f"[STEP 5/7] 戦略実行開始")
            execution_results = self.execution_manager.execute_dynamic_strategies(
                stock_data=stock_data,
                ticker=ticker,
                selected_strategies=strategy_selection['selected_strategies'],
                strategy_weights=strategy_selection.get('strategy_weights', {})
            )
            
            # 6. 包括的パフォーマンス分析
            self.logger.info(f"[STEP 6/7] パフォーマンス分析")
            performance_results = self.performance_analyzer.analyze_comprehensive_performance(
                execution_results, stock_data, market_analysis
            )
            
            # 7. 包括的レポート生成
            self.logger.info(f"[STEP 7/7] 包括的レポート生成")
            report_results = self.reporter.generate_full_backtest_report(
                execution_results, stock_data, ticker, config=None
            )
            
            # 8. 実行結果統合
            final_results = {
                'status': 'SUCCESS',
                'ticker': ticker,
                'execution_timestamp': datetime.now(),
                'market_analysis': market_analysis,
                'strategy_selection': strategy_selection,
                'risk_assessment': risk_assessment,
                'execution_results': execution_results,
                'performance_results': performance_results,
                'report_results': report_results
            }
            
            self.logger.info("=" * 80)
            self.logger.info(f"[SUCCESS] バックテスト完了")
            self.logger.info(f"[REPORT] レポートパス: {report_results.get('output_directory', 'N/A')}")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"[ERROR] バックテスト実行エラー: {e}", exc_info=True)
            return {
                'status': 'ERROR',
                'ticker': ticker,
                'error': str(e),
                'execution_timestamp': datetime.now()
            }
    
    def _get_real_data(
        self,
        ticker: str,
        days_back: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        実データ取得（Phase 4.2: yfinance統合版）
        
        Args:
            ticker: ティッカーシンボル
            days_back: 取得日数
        
        Returns:
            (株価データ, インデックスデータ)
        """
        self.logger.info(f"Getting real data for {ticker} ({days_back} days)")
        
        try:
            # YFinanceDataFeed初期化
            from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
            data_feed = YFinanceDataFeed()
            
            # 株価データ取得
            stock_data = data_feed.get_stock_data(ticker, days_back=days_back)
            self.logger.info(f"Stock data retrieved: {len(stock_data)} rows")
            
            # インデックスデータ取得（S&P 500）
            index_data = data_feed.get_index_data("^GSPC", days_back=days_back)
            self.logger.info(f"Index data retrieved: {len(index_data)} rows")
            
            return stock_data, index_data
            
        except Exception as e:
            self.logger.error(f"Real data retrieval error: {e}", exc_info=True)
            self.logger.warning("Falling back to sample data generation")
            return self._get_sample_data(ticker, days_back)
    
    def _get_sample_data(
        self,
        ticker: str,
        days_back: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        サンプルデータ生成（フォールバック）
        
        Args:
            ticker: ティッカーシンボル
            days_back: 取得日数
        
        Returns:
            (株価データ, インデックスデータ)
        """
        self.logger.info(f"Generating sample data for {ticker} ({days_back} days)")
        
        # 日付範囲生成
        end_date = datetime.now()
        dates = pd.date_range(
            end=end_date,
            periods=days_back,
            freq='D'
        )
        
        # サンプル株価データ生成
        base_price = 100
        returns = np.random.randn(days_back) * 0.02  # 2% 日次ボラティリティ
        prices = base_price * (1 + returns).cumprod()
        
        stock_data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(days_back) * 0.01),
            'High': prices * (1 + abs(np.random.randn(days_back)) * 0.015),
            'Low': prices * (1 - abs(np.random.randn(days_back)) * 0.015),
            'Close': prices,
            'Adj Close': prices,  # 'Adj Close'カラム追加
            'Volume': np.random.randint(1000000, 10000000, days_back)
        }, index=dates)
        
        # サンプルインデックスデータ生成
        index_returns = np.random.randn(days_back) * 0.015
        index_prices = base_price * (1 + index_returns).cumprod()
        
        index_data = pd.DataFrame({
            'Open': index_prices * (1 + np.random.randn(days_back) * 0.005),
            'High': index_prices * (1 + abs(np.random.randn(days_back)) * 0.01),
            'Low': index_prices * (1 - abs(np.random.randn(days_back)) * 0.01),
            'Close': index_prices,
            'Adj Close': index_prices,  # 'Adj Close'カラム追加
            'Volume': np.random.randint(5000000, 50000000, days_back)
        }, index=dates)
        
        self.logger.info(f"Sample data generated: {len(stock_data)} rows")
        
        return stock_data, index_data


def main():
    """メインエントリーポイント - シンプル化完了"""
    
    print("\n" + "=" * 80)
    print("次世代マルチ戦略バックテストシステム - Phase 1 簡易版")
    print("=" * 80 + "\n")
    
    # システム設定
    config = {
        'execution': {
            'execution_mode': 'simple',
            'broker': {
                'initial_cash': 1000000,
                'commission_per_trade': 1.0
            }
        },
        'risk_management': {
            'use_enhanced_risk': False
        },
        'performance': {
            'use_aggregator': False
        }
    }
    
    # システム初期化
    print("[INFO] システム初期化中...")
    system = MainSystemController(config)
    
    # バックテスト実行
    ticker = "AAPL"  # Phase 4.2: リアル銘柄でテスト
    print(f"\n[INFO] バックテスト実行: {ticker}")
    print("-" * 80 + "\n")
    
    results = system.execute_comprehensive_backtest(ticker, days_back=90)  # 90日間のデータで高速テスト
    
    # 基本結果出力
    print("\n" + "=" * 80)
    print("バックテスト完了")
    print("=" * 80)
    
    if results['status'] == 'SUCCESS':
        print(f"\n✅ ステータス: 成功")
        print(f"銘柄: {results['ticker']}")
        print(f"実行時間: {results['execution_timestamp']}")
        
        # 戦略選択結果
        strategy_selection = results.get('strategy_selection', {})
        selected = strategy_selection.get('selected_strategies', [])
        print(f"\n選択戦略: {', '.join(selected) if selected else 'なし'}")
        
        # パフォーマンス結果
        performance = results.get('performance_results', {})
        summary = performance.get('summary_statistics', {})
        if summary:
            print(f"\n【パフォーマンスサマリー】")
            print(f"  総リターン: {summary.get('total_return', 0):.2%}")
            print(f"  総取引数: {summary.get('total_trades', 0)}")
            print(f"  シャープレシオ: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"  最大ドローダウン: {summary.get('max_drawdown', 0):.2%}")
            print(f"  勝率: {summary.get('win_rate', 0):.2%}")
        
        # レポート結果
        report = results.get('report_results', {})
        if report.get('output_directory'):
            print(f"\nレポート出力ディレクトリ: {report['output_directory']}")
    
    elif results['status'] == 'EXECUTION_DENIED':
        print(f"\n⚠️  ステータス: 実行拒否")
        print(f"理由: {results.get('message', 'リスク評価により実行拒否')}")
        
        risk = results.get('risk_assessment', {})
        print(f"リスクレベル: {risk.get('overall_risk_level', 'N/A')}")
    
    else:  # ERROR
        print(f"\n❌ ステータス: エラー")
        print(f"エラー内容: {results.get('error', '不明なエラー')}")
    
    print("\n" + "=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
