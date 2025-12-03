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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

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
# numpy削除: _get_sample_data()削除によりnp.random.randn()が不要

# Excel設定読み込み（既存モジュール活用）
from data_fetcher import get_parameters_and_data


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
        days_back: int = 365,
        backtest_start_date: Optional[datetime] = None,
        backtest_end_date: Optional[datetime] = None,
        warmup_days: int = 90
    ) -> Dict[str, Any]:
        """
        包括的バックテスト実行（Phase 4.2: リアルデータ対応版 + ウォームアップ期間対応）
        
        Args:
            ticker: ティッカーシンボル
            stock_data: 株価データ（Noneの場合はyfinanceから取得）
            index_data: インデックスデータ（Noneの場合はyfinanceから取得）
            days_back: 取得日数
            backtest_start_date: バックテスト開始日（取引開始日）
            backtest_end_date: バックテスト終了日
            warmup_days: ウォームアップ期間日数（デフォルト30日）
        
        Returns:
            包括的バックテスト結果
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting comprehensive backtest for {ticker}")
        if backtest_start_date:
            self.logger.info(f"Trading period: {backtest_start_date} to {backtest_end_date}")
            self.logger.info(f"Warmup period: {warmup_days} days")
        self.logger.info("=" * 80)
        
        try:
            # 1. データ取得（Phase 4.2: yfinanceから実データ取得）
            self.logger.info(f"[STEP 1/7] データ取得開始: {ticker}")
            if stock_data is None:
                stock_data, index_data = self._get_real_data(ticker, days_back)
            
            # 1.5 ウォームアップ期間対応（backtest_start_date指定時）
            if backtest_start_date is not None:
                # ウォームアップ開始日を計算
                warmup_start = backtest_start_date - timedelta(days=warmup_days)
                
                # DatetimeIndexに変換（必要なら）
                warmup_start_ts = pd.Timestamp(warmup_start)
                
                # データをウォームアップ開始日以降にフィルタリング
                if isinstance(stock_data.index, pd.DatetimeIndex):
                    available_start = stock_data.index.min()
                    
                    # タイムゾーン統一: available_startがtz-awareならwarmup_start_tsもtz-awareに変換
                    if available_start.tz is not None and warmup_start_ts.tz is None:
                        warmup_start_ts = warmup_start_ts.tz_localize(available_start.tz)
                    elif available_start.tz is None and warmup_start_ts.tz is not None:
                        warmup_start_ts = warmup_start_ts.tz_localize(None)
                    
                    if warmup_start_ts < available_start:
                        raise RuntimeError(
                            f"Insufficient data for warmup period. "
                            f"Required warmup_start: {warmup_start_ts}, "
                            f"Available data starts: {available_start}, "
                            f"Shortage: {(available_start - warmup_start_ts).days} days"
                        )
                    
                    stock_data = stock_data[stock_data.index >= warmup_start_ts]
                    if index_data is not None:
                        # index_dataのインデックスをDatetimeIndexに変換(キャッシュからの読み込み時に文字列の場合がある)
                        if not isinstance(index_data.index, pd.DatetimeIndex):
                            try:
                                index_data.index = pd.to_datetime(index_data.index)
                            except Exception as e:
                                self.logger.warning(f"index_data index conversion failed: {e}, skipping index_data filtering")
                                index_data = None
                        
                        if index_data is not None:
                            index_data = index_data[index_data.index >= warmup_start_ts]
                    
                    # [LOG#3] ウォームアップ期間データ範囲トラッキング
                    self.logger.info(
                        f"[WARMUP_DATA_RANGE] Warmup start: {warmup_start_ts}, "
                        f"Filtered data range: {stock_data.index[0]} ~ {stock_data.index[-1]} ({len(stock_data)} rows)"
                    )
                    
                    # Priority 1-2: データ不足警告ログ (copilot-instructions.md準拠)
                    # 取引0件問題の早期検出: trading_start_date > データ最終日
                    self.logger.info(
                        f"Data filtered: warmup_start={warmup_start_ts}, "
                        f"trading_start={backtest_start_date}, "
                        f"trading_end={backtest_end_date}, "
                        f"data_length={len(stock_data)} rows"
                    )
                    
                    # データ範囲チェック: trading_start_date > データ最終日の場合は警告
                    if len(stock_data) > 0:
                        data_last_date = pd.Timestamp(stock_data.index[-1])
                        trading_start_ts = pd.Timestamp(backtest_start_date)
                        
                        # タイムゾーン統一: data_last_dateがtz-awareならtrading_start_tsもtz-awareに変換
                        if data_last_date.tzinfo is not None and trading_start_ts.tzinfo is None:
                            trading_start_ts = trading_start_ts.tz_localize(data_last_date.tzinfo)
                        elif data_last_date.tzinfo is None and trading_start_ts.tzinfo is not None:
                            trading_start_ts = trading_start_ts.tz_localize(None)
                        
                        self.logger.info(f"[DATA_RANGE_CHECK] データ最終日: {data_last_date.strftime('%Y-%m-%d')}")
                        self.logger.info(f"[DATA_RANGE_CHECK] 取引開始日: {trading_start_ts.strftime('%Y-%m-%d')}")
                        
                        if data_last_date < trading_start_ts:
                            error_msg = (
                                f"[DATA_INSUFFICIENT] trading_start_date({backtest_start_date})がデータ範囲外です。"
                                f"データ最終日: {data_last_date.strftime('%Y-%m-%d')}, "
                                f"取引開始日: {trading_start_ts.strftime('%Y-%m-%d')}。"
                                f"取引0件の可能性があります。バックテストを中断します。"
                            )
                            self.logger.error(error_msg)
                            # copilot-instructions.md準拠: データ不足時はエラーとして中断
                            return {
                                'status': 'DATA_INSUFFICIENT',
                                'ticker': ticker,
                                'error': error_msg,
                                'data_last_date': data_last_date.strftime('%Y-%m-%d'),
                                'trading_start_date': trading_start_ts.strftime('%Y-%m-%d'),
                                'execution_timestamp': datetime.now()
                            }
                    else:
                        error_msg = "[DATA_INSUFFICIENT] stock_dataが空です。取引0件の可能性があります。バックテストを中断します。"
                        self.logger.error(error_msg)
                        # copilot-instructions.md準拠: データ不足時はエラーとして中断
                        return {
                            'status': 'DATA_INSUFFICIENT',
                            'ticker': ticker,
                            'error': error_msg,
                            'execution_timestamp': datetime.now()
                        }
                else:
                    self.logger.warning("stock_data index is not DatetimeIndex, skipping warmup filtering")
            
            # 2. 市場分析・トレンド判定
            self.logger.info(f"[STEP 2/7] 市場分析実行")
            market_analysis = self.market_analyzer.comprehensive_market_analysis(
                stock_data, index_data
            )
            
            # 3. 動的戦略選択・重み計算
            self.logger.info(f"[STEP 3/7] 動的戦略選択実行")
            
            # DynamicStrategySelector復活（Phase 5-B-1）
            # Phase 5-A完了: strategy_not_foundエラー解決、Entry_Idxバグ修正完了
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
            
            # trading_start_date, trading_end_dateをTimestamp化（タイムゾーン統一処理追加）
            trading_start_ts = pd.Timestamp(backtest_start_date) if backtest_start_date else None
            trading_end_ts = pd.Timestamp(backtest_end_date) if backtest_end_date else None
            
            # タイムゾーン統一: stock_dataのインデックスがtz-awareならtrading_start_ts/trading_end_tsもtz-awareに変換
            if isinstance(stock_data.index, pd.DatetimeIndex) and len(stock_data) > 0:
                data_tz = stock_data.index[0].tz
                if data_tz is not None:
                    # データがtz-awareの場合、trading_start_ts/trading_end_tsもtz-awareに変換
                    if trading_start_ts is not None and trading_start_ts.tz is None:
                        trading_start_ts = trading_start_ts.tz_localize(data_tz)
                    if trading_end_ts is not None and trading_end_ts.tz is None:
                        trading_end_ts = trading_end_ts.tz_localize(data_tz)
                else:
                    # データがtz-naiveの場合、trading_start_ts/trading_end_tsもtz-naiveに変換
                    if trading_start_ts is not None and trading_start_ts.tz is not None:
                        trading_start_ts = trading_start_ts.tz_localize(None)
                    if trading_end_ts is not None and trading_end_ts.tz is not None:
                        trading_end_ts = trading_end_ts.tz_localize(None)
            
            # [LOG#8] データ範囲トラッキング
            self.logger.info(
                f"[DATA_RANGE_TRACKING] Stock data range: {stock_data.index[0]} ~ {stock_data.index[-1]} ({len(stock_data)} rows), "
                f"Trading period: {trading_start_ts} ~ {trading_end_ts}"
            )
            
            execution_results = self.execution_manager.execute_dynamic_strategies(
                stock_data=stock_data,
                ticker=ticker,
                selected_strategies=strategy_selection['selected_strategies'],
                strategy_weights=strategy_selection.get('strategy_weights', {}),
                trading_start_date=trading_start_ts,
                trading_end_date=trading_end_ts
            )
            
            # Phase 5-B-2: backtest_signalsを取得（Entry_Signal/Exit_Signal列を含む）
            backtest_signals = None
            if (isinstance(execution_results, dict) and 
                'execution_results' in execution_results and 
                len(execution_results['execution_results']) > 0):
                # 最初の実行結果からbacktest_signalsを取得
                for result in execution_results['execution_results']:
                    if result.get('success') and 'backtest_signals' in result:
                        backtest_signals = result['backtest_signals']
                        break
            
            # backtest_signalsが取得できなければstock_dataを使用（後方互換）
            data_for_analysis = backtest_signals if backtest_signals is not None else stock_data
            
            # 6. 包括的パフォーマンス分析
            self.logger.info(f"[STEP 6/7] パフォーマンス分析")
            performance_results = self.performance_analyzer.analyze_comprehensive_performance(
                execution_results, data_for_analysis, market_analysis
            )
            
            # 7. 包括的レポート生成
            self.logger.info(f"[STEP 7/7] 包括的レポート生成")
            report_results = self.reporter.generate_full_backtest_report(
                execution_results, data_for_analysis, ticker, config=None
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
        
        [CRITICAL] copilot-instructions.md準拠:
        - モック/ダミーデータのフォールバック禁止
        - 実データ取得失敗時はエラーとして処理
        
        Args:
            ticker: ティッカーシンボル
            days_back: 取得日数
        
        Returns:
            (株価データ, インデックスデータ)
            
        Raises:
            RuntimeError: データ取得失敗時
        """
        self.logger.info(f"Getting real data for {ticker} ({days_back} days)")
        
        try:
            # YFinanceDataFeed初期化
            from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
            data_feed = YFinanceDataFeed()
            
            # 株価データ取得
            stock_data = data_feed.get_stock_data(ticker, days_back=days_back)
            self.logger.info(f"Stock data retrieved: {len(stock_data)} rows")
            
            # インデックスデータ取得（.Tで終わる場合は日経225、それ以外はS&P 500）
            index_ticker = "^N225" if ticker.endswith(".T") else "^GSPC"
            self.logger.info(f"Index ticker selected: {index_ticker} for {ticker}")
            index_data = data_feed.get_index_data(index_ticker, days_back=days_back)
            self.logger.info(f"Index data retrieved: {len(index_data)} rows")
            
            return stock_data, index_data
            
        except Exception as e:
            self.logger.error(f"Real data retrieval failed: {e}", exc_info=True)
            # copilot-instructions.md準拠: モックデータフォールバック禁止
            raise RuntimeError(
                f"Failed to retrieve real market data for {ticker}. "
                f"Mock/dummy data fallback is prohibited by copilot-instructions.md. "
                f"Cannot proceed with backtest. Original error: {e}"
            )
    
    # _get_sample_data() メソッドを削除
    # 理由: copilot-instructions.md違反
    # 「モック/ダミー/テストデータを使用するフォールバック禁止」
    # 実データ取得失敗時は_get_real_data()でRuntimeErrorを発生させる


def main():
    """メインエントリーポイント - Excel設定対応版"""
    
    print("\n" + "=" * 80)
    print("次世代マルチ戦略バックテストシステム - Excel設定対応版")
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
            'use_enhanced_risk': False,
            'max_drawdown_threshold': 0.15
        },
        'performance': {
            'use_aggregator': False
        }
    }
    
    # システム初期化
    print("[INFO] システム初期化中...")
    system = MainSystemController(config)
    
    # Excel設定ファイルからパラメータ取得
    print("[INFO] Excel設定ファイル読み込み中...")
    print("        優先順位: backtest_config.xlsm > backtest_config.xlsx > config.csv")
    
    try:
        # get_parameters_and_data()で銘柄・期間・データを取得
        # 引数なしで呼び出すと、Excelから自動取得
        # warmup_days=90を明示的に渡す（2025-12-03変更）
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(warmup_days=90)
        
        print(f"[SUCCESS] Excel設定読み込み完了")
        print(f"        銘柄: {ticker}")
        print(f"        期間: {start_date} ~ {end_date}")
        print(f"        株価データ: {len(stock_data)} rows")
        print(f"        インデックスデータ: {len(index_data) if index_data is not None else 'N/A'} rows")
        
    except FileNotFoundError as e:
        print(f"[WARNING] Excel設定ファイルが見つかりません: {e}")
        print("[INFO] デフォルト設定を使用します: AAPL, 90 days")
        ticker = "AAPL"
        stock_data = None
        index_data = None
        start_date = None
        end_date = None
        
    except Exception as e:
        print(f"[ERROR] Excel設定読み込みエラー: {e}")
        print("[INFO] デフォルト設定を使用します: AAPL, 90 days")
        ticker = "AAPL"
        stock_data = None
        index_data = None
        start_date = None
        end_date = None
    
    # バックテスト実行
    print(f"\n[INFO] バックテスト実行: {ticker}")
    print("-" * 80 + "\n")
    
    # stock_dataが取得済みの場合はそれを使用、なければyfinanceから取得
    if stock_data is not None:
        # start_date/end_dateをdatetimeに変換して渡す
        backtest_start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        backtest_end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        
        results = system.execute_comprehensive_backtest(
            ticker,
            stock_data=stock_data,
            index_data=index_data,
            backtest_start_date=backtest_start,
            backtest_end_date=backtest_end
        )
    else:
        results = system.execute_comprehensive_backtest(ticker, days_back=90)
    
    # 基本結果出力
    print("\n" + "=" * 80)
    print("バックテスト完了")
    print("=" * 80)
    
    if results['status'] == 'SUCCESS':
        # [DEBUG_PHASE1] データフロー追跡
        print(f"\n[DEBUG_PHASE1] results keys: {results.keys()}")
        print(f"[DEBUG_PHASE1] results['status']: {results['status']}")
        
        print(f"\n[OK] ステータス: 成功")
        print(f"銘柄: {results['ticker']}")
        print(f"実行時間: {results['execution_timestamp']}")
        
        # 戦略選択結果
        strategy_selection = results.get('strategy_selection', {})
        selected = strategy_selection.get('selected_strategies', [])
        print(f"\n選択戦略: {', '.join(selected) if selected else 'なし'}")
        
        # パフォーマンス結果
        performance = results.get('performance_results', {})
        print(f"[DEBUG_PHASE1] performance type: {type(performance)}")
        print(f"[DEBUG_PHASE1] performance keys: {performance.keys() if isinstance(performance, dict) else 'NOT_DICT'}")
        
        summary = performance.get('summary_statistics', {})
        print(f"[DEBUG_PHASE1] summary_statistics type: {type(summary)}")
        print(f"[DEBUG_PHASE1] summary_statistics content: {summary}")
        
        if summary:
            print(f"\n【パフォーマンスサマリー】")
            print(f"  総リターン: {summary.get('total_return', 0):.2%}")
            print(f"  総取引数: {summary.get('total_trades', 0)}")
            print(f"  シャープレシオ: {summary.get('sharpe_ratio', 0):.2f}")
            print(f"  最大ドローダウン: {summary.get('max_drawdown', 0):.2%}")
            print(f"  勝率: {summary.get('win_rate', 0):.2%}")
        else:
            print(f"\n[WARNING] summary_statistics is empty or None")
        
        # レポート結果
        report = results.get('report_results', {})
        if report.get('output_directory'):
            print(f"\nレポート出力ディレクトリ: {report['output_directory']}")
    
    elif results['status'] == 'EXECUTION_DENIED':
        print(f"\n[WARNING] ステータス: 実行拒否")
        print(f"理由: {results.get('message', 'リスク評価により実行拒否')}")
        
        risk = results.get('risk_assessment', {})
        print(f"リスクレベル: {risk.get('overall_risk_level', 'N/A')}")
    
    else:  # ERROR
        print(f"\n[ERROR] ステータス: エラー")
        print(f"エラー内容: {results.get('error', '不明なエラー')}")
    
    print("\n" + "=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
