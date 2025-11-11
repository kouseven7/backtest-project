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

# Phase 5-B-12: 共通ユーティリティ（execution_details抽出ロジック統一）
from main_system.execution_control.execution_detail_utils import (
    extract_buy_sell_orders,
    validate_buy_sell_pairing,
    get_execution_detail_summary
)

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
        
        # Phase 5-B-2: データフロー追跡ログ
        self.logger.info(f"[DATA_FLOW_ANALYZER] Input execution_results type: {type(execution_results)}")
        self.logger.info(f"[DATA_FLOW_ANALYZER] Input execution_results keys: {execution_results.keys() if isinstance(execution_results, dict) else 'NOT_DICT'}")
        self.logger.info(f"[DATA_FLOW_ANALYZER] Input stock_data shape: {stock_data.shape}")
        self.logger.info(f"[DATA_FLOW_ANALYZER] Input stock_data columns: {list(stock_data.columns)}")
        
        if isinstance(execution_results, dict) and 'execution_results' in execution_results:
            exec_list = execution_results.get('execution_results', [])
            self.logger.info(f"[DATA_FLOW_ANALYZER] execution_results list length: {len(exec_list)}")
            if exec_list and len(exec_list) > 0:
                self.logger.info(f"[DATA_FLOW_ANALYZER] First execution_result keys: {exec_list[0].keys() if isinstance(exec_list[0], dict) else 'NOT_DICT'}")
        
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
        """
        基本パフォーマンス計算（Phase 2: execution_details対応版）
        
        ComprehensiveReporterと同じロジックでexecution_detailsから実取引データを抽出
        
        copilot-instructions.md準拠:
        - 実データのみ使用
        - statusチェック削除（statusがNoneでも取引データがあれば処理）
        """
        try:
            # Phase 5-B-2: データフロー追跡ログ
            self.logger.info(f"[DATA_FLOW_BASIC] stock_data columns in _calculate_basic_performance: {list(stock_data.columns)}")
            self.logger.info(f"[DATA_FLOW_BASIC] Has Entry_Signal: {'Entry_Signal' in stock_data.columns}")
            self.logger.info(f"[DATA_FLOW_BASIC] Has Exit_Signal: {'Exit_Signal' in stock_data.columns}")
            
            # [DEBUG_PHASE2] execution_results構造確認
            self.logger.info(f"[DEBUG_PHASE2_BASIC] execution_results type: {type(execution_results)}")
            self.logger.info(f"[DEBUG_PHASE2_BASIC] execution_results keys: {execution_results.keys() if isinstance(execution_results, dict) else 'NOT_DICT'}")
            
            # 初期資本・最終価値
            initial_capital = 1000000  # デフォルト100万円
            final_value = initial_capital
            total_profit = 0
            total_trades = 0
            
            # execution_detailsから実取引データを抽出（ComprehensiveReporterと同じロジック）
            if 'execution_results' in execution_results and isinstance(execution_results['execution_results'], list):
                strategies_results = execution_results['execution_results']
                self.logger.info(f"[DEBUG_PHASE2_BASIC] strategies_results length: {len(strategies_results)}")
                
                for idx, strategy_result in enumerate(strategies_results):
                    if not isinstance(strategy_result, dict):
                        continue
                    
                    # execution_detailsの存在確認（statusチェックは削除）
                    if 'execution_details' not in strategy_result:
                        self.logger.info(f"[DEBUG_PHASE2_BASIC] Strategy {idx}: No execution_details")
                        continue
                    
                    execution_details = strategy_result.get('execution_details', [])
                    self.logger.info(
                        f"[DEBUG_PHASE2_BASIC] Strategy {idx} ({strategy_result.get('strategy_name', 'Unknown')}): "
                        f"execution_details length={len(execution_details)}"
                    )
                    
                    # execution_detailsから取引データを抽出
                    trades = self._extract_trades_from_execution_details(execution_details)
                    
                    if trades:
                        # 損益計算
                        strategy_profit = sum(trade.get('pnl', 0) for trade in trades)
                        strategy_trades = len(trades)
                        
                        total_profit += strategy_profit
                        total_trades += strategy_trades
                        
                        self.logger.info(
                            f"[DEBUG_PHASE2_BASIC] Strategy {idx}: "
                            f"extracted {strategy_trades} trades, profit={strategy_profit:.2f}"
                        )
                
                self.logger.info(f"[DEBUG_PHASE2_BASIC] CALCULATED total_profit: {total_profit}, total_trades: {total_trades}")
                final_value = initial_capital + total_profit
            
            # リターン計算
            total_return = (final_value - initial_capital) / initial_capital
            
            result = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_profit': final_value - initial_capital,
                'total_trades': total_trades,
                'period_days': len(stock_data) if stock_data is not None else 0
            }
            
            self.logger.info(f"[DEBUG_PHASE2_BASIC] RESULT: {result}")
            
            return result
            
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
    
    def _extract_trades_from_execution_details(
        self,
        execution_details: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        execution_detailsから取引データを抽出（Phase 5-B-12: 共通ユーティリティ使用版）
        
        copilot-instructions.md準拠:
        - 実データのみ使用
        - BUY/SELLペアの実データのみ抽出
        - データ不足時は空リスト返却（フォールバック禁止）
        - 強制決済（status='force_closed'）も有効な取引として認識
        
        Args:
            execution_details: 実行詳細リスト
        
        Returns:
            取引レコードリスト（pnl, return_pctを含む）
        """
        try:
            trades = []
            
            # Phase 5-B-12: 共通ユーティリティでBUY/SELL抽出
            self.logger.info(
                f"[PHASE_5_B_12] execution_details総数: {len(execution_details)}"
            )
            
            # 共通ユーティリティを使用してBUY/SELL抽出
            buy_orders, sell_orders = extract_buy_sell_orders(
                execution_details,
                logger_instance=self.logger
            )
            
            # ペアリング検証
            pairing_result = validate_buy_sell_pairing(
                buy_orders,
                sell_orders,
                logger_instance=self.logger
            )
            
            # ペア不一致の場合は空リスト返却（copilot-instructions.md: フォールバック禁止）
            if not pairing_result['is_valid']:
                self.logger.warning(
                    f"[FALLBACK_PROHIBITED] {pairing_result['warning_message']} "
                    f"copilot-instructions.md準拠: ダミーデータ補完は実行しません。"
                )
                return []
            
            # ペアリング実行（FIFO方式）
            paired_count = pairing_result['paired_count']
            for i in range(paired_count):
                buy_order = buy_orders[i]
                sell_order = sell_orders[i]
                
                try:
                    # 実データから取引レコード作成
                    entry_price = buy_order.get('executed_price', 0.0)
                    exit_price = sell_order.get('executed_price', 0.0)
                    shares = buy_order.get('quantity', 0)
                    
                    # データ検証
                    if not all([entry_price > 0, exit_price > 0, shares > 0]):
                        self.logger.error(
                            f"[DATA_VALIDATION_FAILED] 不正な取引データ（ペア{i+1}）: "
                            f"entry_price={entry_price}, exit_price={exit_price}, shares={shares}. "
                            f"スキップします（フォールバック禁止）。"
                        )
                        continue
                    
                    # 損益計算（実データに基づく）
                    pnl = (exit_price - entry_price) * shares
                    return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    
                    trade = {
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares
                    }
                    
                    trades.append(trade)
                    
                except Exception as trade_error:
                    self.logger.error(f"[TRADE_EXTRACTION_ERROR] ペア{i+1}: {trade_error}")
                    continue
            
            self.logger.info(
                f"[PHASE_5_B_12] Converted {len(execution_details)} execution details "
                f"to {len(trades)} trade records (BUY={len(buy_orders)}, SELL={len(sell_orders)}, Paired={paired_count})"
            )
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error extracting trades from execution_details: {e}", exc_info=True)
            # copilot-instructions.md: フォールバック禁止
            return []
    
    def _calculate_enhanced_metrics(
        self,
        execution_results: Dict[str, Any],
        stock_data: pd.DataFrame,
        market_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        強化メトリクス計算（Phase 2: 実データ抽出版）
        
        copilot-instructions.md準拠:
        - ダミーデータ生成フォールバック禁止
        - 実データのみ使用
        - データ不足時は0.0を返却
        """
        try:
            # execution_resultsから実取引データを抽出
            trades_list = []
            
            if 'execution_results' in execution_results and isinstance(execution_results['execution_results'], list):
                for strategy_result in execution_results['execution_results']:
                    if isinstance(strategy_result, dict) and 'execution_details' in strategy_result:
                        execution_details = strategy_result.get('execution_details', [])
                        # _extract_trades_from_execution_details()を使用
                        trades = self._extract_trades_from_execution_details(execution_details)
                        trades_list.extend(trades)
            
            # DataFrame変換（実データのみ）
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                self.logger.info(
                    f"[REAL_DATA_ONLY] Extracted {len(trades_df)} real trades for enhanced metrics calculation"
                )
            else:
                # データなし時は空DataFrame（copilot-instructions.md: ダミーデータ生成禁止）
                trades_df = pd.DataFrame()
                self.logger.warning(
                    "[FALLBACK_PROHIBITED] No real trades found in execution_results. "
                    "copilot-instructions.md準拠: ダミーデータは生成しません。"
                )
            
            # メトリクス計算（実データに基づく）
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
                'sortino_ratio': self._calculate_sortino_ratio(trades_df),
                'max_drawdown': self._calculate_max_drawdown(trades_df),
                'win_rate': self._calculate_win_rate(trades_df),
                'profit_factor': self._calculate_profit_factor(trades_df),
                'average_trade_duration': self._calculate_avg_trade_duration(trades_df)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Enhanced metrics calculation error: {e}", exc_info=True)
            # copilot-instructions.md: エラー時もダミーデータ生成禁止、0.0を返却
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
        # [DEBUG_PHASE1] 入力データ確認
        self.logger.info(f"[DEBUG_PHASE1_SUMMARY] basic_performance: {basic_performance}")
        self.logger.info(f"[DEBUG_PHASE1_SUMMARY] enhanced_metrics: {enhanced_metrics}")
        
        result = {
            'total_return': basic_performance.get('total_return', 0.0),
            'total_trades': basic_performance.get('total_trades', 0),
            'sharpe_ratio': enhanced_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': enhanced_metrics.get('max_drawdown', 0.0),
            'win_rate': enhanced_metrics.get('win_rate', 0.0),
            'profit_factor': enhanced_metrics.get('profit_factor', 0.0)
        }
        
        self.logger.info(f"[DEBUG_PHASE1_SUMMARY] RESULT summary_statistics: {result}")
        
        return result
    
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
        """
        最大ドローダウン計算（Phase 4.2-20修正）
        
        修正内容:
        - 分母を累積PnL最大値から初期資本に変更
        - これにより100%を超える異常値を防止
        - ドローダウンは初期資本に対する最大下落率として計算
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        
        # 累積PnLの推移を計算
        cumulative_pnl = trades['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        
        # 初期資本を取得（デフォルト1,000,000円）
        initial_capital = 1000000.0
        
        # ドローダウンを初期資本に対する比率で計算
        # drawdown = (現在の累積PnL - 過去最大の累積PnL) / 初期資本
        drawdown = (cumulative_pnl - running_max) / initial_capital
        
        # 最大ドローダウンを返す（負の値の絶対値）
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
