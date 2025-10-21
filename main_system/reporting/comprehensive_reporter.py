"""
包括的レポートシステム - Phase 4.1
IntegratedExecutionManagerの実行結果を統合して包括的なレポートを生成

Purpose:
  - 複数戦略の実行結果を統合
  - テキスト/JSON/CSV形式での出力（Excel出力禁止対応）
  - 期待値分析、パフォーマンス計算、取引分析の統合

Author: GitHub Copilot
Created: 2025-10-17
Version: 1.1 (Phase 4.2-7: JSON Serialization対応)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import json
import numpy as np

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# 既存モジュールのインポート
from main_system.reporting.main_text_reporter import MainTextReporter
from main_system.performance.trade_analyzer import TradeAnalyzer
from main_system.performance.enhanced_performance_calculator import EnhancedPerformanceCalculator
from main_system.performance.data_extraction_enhancer import MainDataExtractor


class SafeJSONEncoder(json.JSONEncoder):
    """
    安全なJSONエンコーダー - Phase 4.2-7
    
    戦略オブジェクトやその他のシリアライズ不可能なオブジェクトを
    文字列表現に変換してJSON出力を可能にする
    """
    def default(self, obj):
        # datetime対応
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # numpy型対応
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # pandas型対応
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        # 戦略オブジェクト対応（クラス名を返す）
        if hasattr(obj, '__class__') and 'Strategy' in obj.__class__.__name__:
            return f"<{obj.__class__.__name__}>"
        
        # その他のオブジェクト（文字列表現）
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


class ComprehensiveReporter:
    """包括的レポート生成クラス"""
    
    def __init__(self, output_base_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            output_base_dir: 出力ベースディレクトリ（Noneの場合はデフォルト）
        """
        self.logger = setup_logger(
            "ComprehensiveReporter",
            log_file="logs/comprehensive_reporter.log"
        )
        
        # 出力ディレクトリ設定
        if output_base_dir is None:
            self.output_base_dir = project_root / "output" / "comprehensive_reports"
        else:
            self.output_base_dir = Path(output_base_dir)
        
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネント初期化
        try:
            self.text_reporter = MainTextReporter()
            # TradeAnalyzerは実行結果ごとに初期化
            self.trade_analyzer = None
            self.performance_calculator = EnhancedPerformanceCalculator()
            self.data_extractor = MainDataExtractor()
            
            self.logger.info("ComprehensiveReporter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    def generate_full_backtest_report(
        self,
        execution_results: Dict[str, Any],
        stock_data: pd.DataFrame,
        ticker: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        バックテスト結果の包括的レポート生成
        
        Args:
            execution_results: IntegratedExecutionManagerの実行結果
            stock_data: 株価データ
            ticker: ティッカーシンボル
            config: 追加設定
        
        Returns:
            Dict[str, Any]: 生成されたレポートの情報
        """
        try:
            self.logger.info(f"Generating comprehensive report for {ticker}")
            
            # Phase 5-B-2: execution_resultsを保存（_generate_text_reportで使用）
            self._current_execution_results = execution_results
            
            # Phase 5-B-2: データフロー追跡ログ
            self.logger.info(f"[DATA_FLOW_REPORTER] execution_results type: {type(execution_results)}")
            self.logger.info(f"[DATA_FLOW_REPORTER] execution_results keys: {execution_results.keys() if isinstance(execution_results, dict) else 'NOT_DICT'}")
            self.logger.info(f"[DATA_FLOW_REPORTER] stock_data shape: {stock_data.shape}")
            self.logger.info(f"[DATA_FLOW_REPORTER] stock_data columns: {list(stock_data.columns)}")
            
            # タイムスタンプ付きディレクトリ作成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.output_base_dir / f"{ticker}_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. データ抽出と分析
            self.logger.info("Step 1: Data extraction and analysis")
            extracted_data = self._extract_and_analyze_data(stock_data, execution_results)
            
            # Phase 5-B-2: 抽出結果ログ
            self.logger.info(f"[DATA_FLOW_REPORTER] extracted_data type: {type(extracted_data)}")
            if isinstance(extracted_data, dict) and 'executed_trades' in extracted_data:
                self.logger.info(f"[DATA_FLOW_REPORTER] Extracted {len(extracted_data['executed_trades'])} trades")
            
            # 2. パフォーマンス計算
            self.logger.info("Step 2: Performance calculation")
            performance_metrics = self._calculate_comprehensive_performance(
                execution_results, extracted_data, stock_data
            )
            
            # 3. 取引分析
            self.logger.info("Step 3: Trade analysis")
            trade_analysis = self._analyze_trades(execution_results, extracted_data)
            
            # 4. テキストレポート生成
            self.logger.info("Step 4: Text report generation")
            text_report_path = self._generate_text_report(
                stock_data, ticker, execution_results, 
                performance_metrics, report_dir
            )
            
            # 5. CSV出力生成
            self.logger.info("Step 5: CSV outputs generation")
            csv_outputs = self._generate_csv_outputs(
                extracted_data, performance_metrics, report_dir, ticker
            )
            
            # 6. JSON出力生成
            self.logger.info("Step 6: JSON outputs generation")
            json_outputs = self._generate_json_outputs(
                execution_results, performance_metrics, 
                trade_analysis, report_dir, ticker
            )
            
            # 7. サマリーレポート生成
            self.logger.info("Step 7: Summary report generation")
            summary_report = self._generate_summary_report(
                execution_results, performance_metrics, 
                trade_analysis, report_dir, ticker
            )
            
            # 統合結果
            comprehensive_result = {
                'status': 'SUCCESS',
                'ticker': ticker,
                'timestamp': timestamp,
                'report_directory': str(report_dir),
                'text_report_path': text_report_path,
                'csv_outputs': csv_outputs,
                'json_outputs': json_outputs,
                'summary_report': summary_report,
                'performance_metrics': performance_metrics,
                'trade_analysis': trade_analysis,
                'execution_summary': {
                    'total_strategies_executed': execution_results.get('total_executions', 0),
                    'successful_strategies': execution_results.get('successful_strategies', 0),
                    'failed_strategies': execution_results.get('failed_strategies', 0)
                }
            }
            
            self.logger.info(f"Comprehensive report generation completed: {report_dir}")
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Error in generate_full_backtest_report: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'ticker': ticker
            }
    
    def _extract_and_analyze_data(
        self,
        stock_data: pd.DataFrame,
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        データ抽出と分析（Phase 4.2-5-3: execution_results統合版）
        
        Args:
            stock_data: 株価データ
            execution_results: 戦略実行結果（execution_detailsを含む）
        
        Returns:
            抽出・分析結果
        """
        try:
            # MainDataExtractorを使用してデータ抽出（バックテストシグナルから）
            extracted_trades = self.data_extractor.extract_accurate_trades(stock_data)
            
            # Phase 4.2-5-3: execution_resultsから実行された取引を追加
            # copilot-instructions.md: 実際の取引件数 > 0 を検証
            executed_trades = self._extract_executed_trades(execution_results)
            
            if executed_trades:
                self.logger.info(f"[OK] Adding {len(executed_trades)} executed trades to report")
                # 実行された取引をマージ
                extracted_trades.extend(executed_trades)
                self.logger.info(f"Total trades after merge: {len(extracted_trades)}")
            else:
                self.logger.warning("No executed trades found in execution_results")
            
            # 基本統計
            analysis_result = {
                'trades': extracted_trades,
                'total_trades': len(extracted_trades),
                'executed_trades_count': len(executed_trades),  # Phase 4.2-5-3: 実行取引数を記録
                'period': {
                    'start_date': stock_data.index[0].strftime("%Y-%m-%d") if len(stock_data) > 0 else 'N/A',
                    'end_date': stock_data.index[-1].strftime("%Y-%m-%d") if len(stock_data) > 0 else 'N/A',
                    'trading_days': len(stock_data)
                },
                'performance': self._calculate_basic_performance(extracted_trades)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Data extraction error: {e}", exc_info=True)
            return {
                'trades': [],
                'total_trades': 0,
                'executed_trades_count': 0,
                'period': {'start_date': 'N/A', 'end_date': 'N/A', 'trading_days': 0},
                'performance': {}
            }
    
    def _extract_executed_trades(
        self,
        execution_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        execution_resultsから実行された取引を抽出（Phase 5-B-1 Step 3-2: フォールバック削除版）
        
        copilot-instructions.md準拠:
        - 実データのみ抽出
        - データ不足時はエラーログ、空リスト返却（フォールバック禁止）
        
        Args:
            execution_results: 戦略実行結果
        
        Returns:
            実行された取引のリスト（MainDataExtractorと同じ形式）
        """
        try:
            executed_trades = []
            
            # execution_resultsの構造を確認
            # IntegratedExecutionManagerからの結果: {'execution_results': [...]}
            # StrategyExecutionManagerからの結果: {'execution_details': [...]}
            
            # パターン1: execution_results['execution_results']（統合実行結果）
            if 'execution_results' in execution_results and isinstance(execution_results['execution_results'], list):
                for result in execution_results['execution_results']:
                    if isinstance(result, dict) and 'execution_details' in result:
                        trades = self._convert_execution_details_to_trades(result['execution_details'])
                        executed_trades.extend(trades)
                        self.logger.info(
                            f"[REAL_DATA] Extracted {len(trades)} trades from "
                            f"strategy: {result.get('strategy_name', 'Unknown')}"
                        )
            
            # パターン2: execution_results['execution_details']（単一戦略結果）
            elif 'execution_details' in execution_results:
                trades = self._convert_execution_details_to_trades(execution_results['execution_details'])
                executed_trades.extend(trades)
                self.logger.info(f"[REAL_DATA] Extracted {len(trades)} trades from execution_details")
            
            # データ検証（copilot-instructions.md: 実際の取引件数 > 0 を検証）
            if not executed_trades:
                self.logger.warning(
                    "[FALLBACK_PROHIBITED] execution_resultsから取引データを抽出できませんでした。"
                    "copilot-instructions.md準拠: ダミーデータは生成しません。"
                )
            else:
                self.logger.info(
                    f"[SUCCESS] Extracted {len(executed_trades)} real trades from execution_results"
                )
            
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error extracting executed trades: {e}", exc_info=True)
            # copilot-instructions.md: フォールバック禁止
            return []
    
    def _convert_execution_details_to_trades(
        self,
        execution_details: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        execution_detailsを取引レコード形式に変換（Phase 5-B-1 Step 3-2: フォールバック削除版）
        
        copilot-instructions.md準拠:
        - ダミーデータ生成フォールバック禁止
        - BUY/SELLペアの実データのみ抽出
        - データ不足時はエラーログのみ、空リスト返却
        
        Args:
            execution_details: 実行詳細リスト
        
        Returns:
            取引レコードリスト（MainDataExtractor形式）
        
        Raises:
            ValueError: データ構造が不正な場合（フォールバック禁止）
        """
        try:
            trades = []
            
            # BUY/SELLペアの抽出（実データのみ）
            # execution_detailsは個別の注文実行記録のリスト
            # BUY→SELLのペアを構成する必要がある
            
            buy_orders = []
            sell_orders = []
            
            for detail in execution_details:
                if not isinstance(detail, dict):
                    continue
                
                # 実行成功した取引のみを対象
                if detail.get('status') != 'executed' or not detail.get('success', False):
                    continue
                
                # BUY/SELL分類
                action = detail.get('action', '').upper()
                if action == 'BUY':
                    buy_orders.append(detail)
                elif action == 'SELL':
                    sell_orders.append(detail)
            
            # BUY/SELLペアリング（FIFO方式）
            # copilot-instructions.md: 実データのみ使用、推測による補完禁止
            if len(buy_orders) != len(sell_orders):
                self.logger.warning(
                    f"[FALLBACK_PROHIBITED] BUY/SELLペア不一致: "
                    f"BUY={len(buy_orders)}, SELL={len(sell_orders)}. "
                    f"copilot-instructions.md準拠: ダミーデータ補完は実行しません。"
                )
                # フォールバック禁止: ペアが成立しない場合は空リスト返却
                return []
            
            # ペアリング実行
            for buy_order, sell_order in zip(buy_orders, sell_orders):
                try:
                    # 実データから取引レコード作成
                    entry_date = buy_order.get('timestamp')
                    exit_date = sell_order.get('timestamp')
                    entry_price = buy_order.get('executed_price', 0.0)
                    exit_price = sell_order.get('executed_price', 0.0)
                    shares = buy_order.get('quantity', 0)
                    
                    # データ検証（copilot-instructions.md: 推測ではなく正確な数値）
                    if not all([entry_date, exit_date, entry_price > 0, exit_price > 0, shares > 0]):
                        self.logger.error(
                            f"[DATA_VALIDATION_FAILED] 不正な取引データ: "
                            f"entry_date={entry_date}, exit_date={exit_date}, "
                            f"entry_price={entry_price}, exit_price={exit_price}, shares={shares}. "
                            f"スキップします（フォールバック禁止）。"
                        )
                        continue
                    
                    # 損益計算（実データに基づく）
                    pnl = (exit_price - entry_price) * shares
                    return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    
                    # 保有期間計算
                    holding_period_days = 0
                    try:
                        if isinstance(entry_date, (pd.Timestamp, datetime)):
                            entry_dt = entry_date
                        else:
                            entry_dt = pd.to_datetime(entry_date)
                        
                        if isinstance(exit_date, (pd.Timestamp, datetime)):
                            exit_dt = exit_date
                        else:
                            exit_dt = pd.to_datetime(exit_date)
                        
                        holding_period_days = (exit_dt - entry_dt).days
                    except Exception as e:
                        self.logger.warning(f"保有期間計算エラー: {e}")
                    
                    # 取引レコード作成（実データのみ）
                    trade_record = {
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'holding_period_days': holding_period_days,
                        'strategy': buy_order.get('symbol', 'Unknown'),
                        'position_value': entry_price * shares,
                        'is_forced_exit': False,
                        'is_executed_trade': True  # Phase 4.2-5-3: 実行取引フラグ
                    }
                    
                    trades.append(trade_record)
                    
                except Exception as e:
                    self.logger.error(f"取引レコード作成エラー: {e}")
                    continue
            
            self.logger.info(
                f"[REAL_DATA_ONLY] Converted {len(trades)} execution details to trade records "
                f"(BUY={len(buy_orders)}, SELL={len(sell_orders)})"
            )
            return trades
            
        except Exception as e:
            self.logger.error(f"Error converting execution details: {e}", exc_info=True)
            # copilot-instructions.md: フォールバック禁止、エラー時は空リスト返却
            return []
    
    def _calculate_basic_performance(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """基本パフォーマンス計算"""
        if not trades:
            return {
                'initial_capital': 1000000,
                'final_portfolio_value': 1000000,
                'total_return': 0.0,
                'win_rate': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'profit_factor': 0.0
            }
        
        # 損益計算
        pnls = [trade.get('pnl', 0) for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        net_profit = total_profit - total_loss
        
        initial_capital = 1000000  # デフォルト初期資本
        final_value = initial_capital + net_profit
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio_value': final_value,
            'total_return': (final_value / initial_capital - 1) if initial_capital > 0 else 0,
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_profit': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean([abs(l) for l in losing_trades]) if losing_trades else 0,
            'max_profit': max(winning_trades) if winning_trades else 0,
            'max_loss': abs(min(losing_trades)) if losing_trades else 0,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': (total_profit / total_loss) if total_loss > 0 else 0
        }
    
    def _calculate_comprehensive_performance(
        self,
        execution_results: Dict[str, Any],
        extracted_data: Dict[str, Any],
        stock_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """包括的パフォーマンス計算"""
        try:
            # EnhancedPerformanceCalculatorを使用
            # 注: 実際の実装は既存モジュールのAPIに合わせて調整が必要
            
            performance_metrics = {
                'basic_metrics': extracted_data.get('performance', {}),
                'period_analysis': extracted_data.get('period', {}),
                'execution_summary': {
                    'status': execution_results.get('status', 'UNKNOWN'),
                    'total_executions': execution_results.get('total_executions', 0),
                    'successful_strategies': execution_results.get('successful_strategies', 0),
                    'failed_strategies': execution_results.get('failed_strategies', 0)
                },
                'trade_statistics': {
                    'total_trades': extracted_data.get('total_trades', 0),
                    'avg_holding_period': self._calculate_avg_holding_period(extracted_data.get('trades', []))
                }
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance calculation error: {e}")
            return {'error': str(e)}
    
    def _calculate_avg_holding_period(self, trades: List[Dict[str, Any]]) -> float:
        """平均保有期間計算"""
        if not trades:
            return 0.0
        
        holding_periods = []
        for trade in trades:
            try:
                entry_date = trade.get('entry_date')
                exit_date = trade.get('exit_date')
                if entry_date and exit_date:
                    if isinstance(entry_date, str):
                        entry_date = pd.to_datetime(entry_date)
                    if isinstance(exit_date, str):
                        exit_date = pd.to_datetime(exit_date)
                    holding_periods.append((exit_date - entry_date).days)
            except Exception:
                continue
        
        return np.mean(holding_periods) if holding_periods else 0.0
    
    def _analyze_trades(
        self,
        execution_results: Dict[str, Any],
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """取引分析"""
        try:
            trades = extracted_data.get('trades', [])
            
            if not trades:
                return {
                    'status': 'NO_TRADES',
                    'total_trades': 0,
                    'strategy_breakdown': {}
                }
            
            # 戦略別分析
            strategy_breakdown = {}
            for trade in trades:
                strategy = trade.get('strategy', 'Unknown')
                if strategy not in strategy_breakdown:
                    strategy_breakdown[strategy] = {
                        'trades': [],
                        'total_pnl': 0,
                        'win_count': 0,
                        'loss_count': 0
                    }
                
                pnl = trade.get('pnl', 0)
                strategy_breakdown[strategy]['trades'].append(trade)
                strategy_breakdown[strategy]['total_pnl'] += pnl
                if pnl > 0:
                    strategy_breakdown[strategy]['win_count'] += 1
                elif pnl < 0:
                    strategy_breakdown[strategy]['loss_count'] += 1
            
            # 戦略別統計計算
            for strategy, data in strategy_breakdown.items():
                total_trades = len(data['trades'])
                data['win_rate'] = data['win_count'] / total_trades if total_trades > 0 else 0
                data['avg_pnl'] = data['total_pnl'] / total_trades if total_trades > 0 else 0
            
            return {
                'status': 'SUCCESS',
                'total_trades': len(trades),
                'strategy_breakdown': strategy_breakdown,
                'top_strategy': max(
                    strategy_breakdown.items(),
                    key=lambda x: x[1]['total_pnl']
                )[0] if strategy_breakdown else 'N/A'
            }
            
        except Exception as e:
            self.logger.error(f"Trade analysis error: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _generate_text_report(
        self,
        stock_data: pd.DataFrame,
        ticker: str,
        execution_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        report_dir: Path
    ) -> str:
        """テキストレポート生成（Phase 5-B-2: execution_results対応版）"""
        try:
            # Phase 5-B-2: execution_resultsを取得
            # self.execution_resultsはgenerate_full_backtest_report()で保存される
            execution_results_to_pass = getattr(self, '_current_execution_results', None)
            
            # MainTextReporterを使用
            text_report_path = self.text_reporter.generate_comprehensive_report(
                stock_data=stock_data,
                ticker=ticker,
                execution_results=execution_results_to_pass,  # Phase 5-B-2追加
                optimized_params=None,  # 最適化パラメータは別途実装時に追加
                output_dir=str(report_dir)
            )
            
            return text_report_path
            
        except Exception as e:
            self.logger.error(f"Text report generation error: {e}")
            return ""
    
    def _generate_csv_outputs(
        self,
        extracted_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        report_dir: Path,
        ticker: str
    ) -> Dict[str, str]:
        """CSV出力生成"""
        csv_outputs = {}
        
        try:
            # 取引履歴CSV
            trades = extracted_data.get('trades', [])
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_csv_path = report_dir / f"{ticker}_trades.csv"
                trades_df.to_csv(trades_csv_path, index=False, encoding='utf-8')
                csv_outputs['trades'] = str(trades_csv_path)
                self.logger.info(f"Trades CSV saved: {trades_csv_path}")
            
            # パフォーマンスサマリーCSV
            performance_summary = []
            basic_metrics = performance_metrics.get('basic_metrics', {})
            for metric_name, metric_value in basic_metrics.items():
                performance_summary.append({
                    'Metric': metric_name,
                    'Value': metric_value
                })
            
            if performance_summary:
                perf_df = pd.DataFrame(performance_summary)
                perf_csv_path = report_dir / f"{ticker}_performance_summary.csv"
                perf_df.to_csv(perf_csv_path, index=False, encoding='utf-8')
                csv_outputs['performance_summary'] = str(perf_csv_path)
                self.logger.info(f"Performance summary CSV saved: {perf_csv_path}")
            
        except Exception as e:
            self.logger.error(f"CSV generation error: {e}")
        
        return csv_outputs
    
    def _generate_json_outputs(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        trade_analysis: Dict[str, Any],
        report_dir: Path,
        ticker: str
    ) -> Dict[str, str]:
        """JSON出力生成"""
        json_outputs = {}
        
        try:
            # 実行結果JSON
            execution_json_path = report_dir / f"{ticker}_execution_results.json"
            with open(execution_json_path, 'w', encoding='utf-8') as f:
                json.dump(execution_results, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            json_outputs['execution_results'] = str(execution_json_path)
            self.logger.info(f"Execution results JSON saved: {execution_json_path}")
            
            # パフォーマンスメトリクスJSON
            metrics_json_path = report_dir / f"{ticker}_performance_metrics.json"
            with open(metrics_json_path, 'w', encoding='utf-8') as f:
                json.dump(performance_metrics, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            json_outputs['performance_metrics'] = str(metrics_json_path)
            self.logger.info(f"Performance metrics JSON saved: {metrics_json_path}")
            
            # 取引分析JSON
            analysis_json_path = report_dir / f"{ticker}_trade_analysis.json"
            # strategy_breakdownのtradesリストを除外（大きすぎる）
            trade_analysis_copy = trade_analysis.copy()
            if 'strategy_breakdown' in trade_analysis_copy:
                for strategy, data in trade_analysis_copy['strategy_breakdown'].items():
                    if 'trades' in data:
                        data['trade_count'] = len(data['trades'])
                        del data['trades']  # 詳細は trades.csv に保存済み
            
            with open(analysis_json_path, 'w', encoding='utf-8') as f:
                json.dump(trade_analysis_copy, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            json_outputs['trade_analysis'] = str(analysis_json_path)
            self.logger.info(f"Trade analysis JSON saved: {analysis_json_path}")
            
        except Exception as e:
            self.logger.error(f"JSON generation error: {e}")
        
        return json_outputs
    
    def _generate_summary_report(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        trade_analysis: Dict[str, Any],
        report_dir: Path,
        ticker: str
    ) -> str:
        """サマリーレポート生成（簡易版テキスト）"""
        try:
            summary_path = report_dir / f"{ticker}_SUMMARY.txt"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"包括的バックテストレポート サマリー\n")
                f.write("=" * 80 + "\n")
                f.write(f"ティッカー: {ticker}\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                # 実行サマリー
                f.write("【実行サマリー】\n")
                f.write(f"  ステータス: {execution_results.get('status', 'UNKNOWN')}\n")
                f.write(f"  実行戦略数: {execution_results.get('total_executions', 0)}\n")
                f.write(f"  成功: {execution_results.get('successful_strategies', 0)}\n")
                f.write(f"  失敗: {execution_results.get('failed_strategies', 0)}\n")
                f.write("\n")
                
                # パフォーマンスサマリー
                basic_metrics = performance_metrics.get('basic_metrics', {})
                f.write("【パフォーマンスサマリー】\n")
                f.write(f"  初期資本: ¥{basic_metrics.get('initial_capital', 0):,.0f}\n")
                f.write(f"  最終ポートフォリオ値: ¥{basic_metrics.get('final_portfolio_value', 0):,.0f}\n")
                f.write(f"  総リターン: {basic_metrics.get('total_return', 0) * 100:.2f}%\n")
                f.write(f"  純利益: ¥{basic_metrics.get('net_profit', 0):,.0f}\n")
                f.write(f"  勝率: {basic_metrics.get('win_rate', 0) * 100:.2f}%\n")
                f.write("\n")
                
                # 取引サマリー
                f.write("【取引サマリー】\n")
                f.write(f"  総取引数: {trade_analysis.get('total_trades', 0)}\n")
                f.write(f"  最優秀戦略: {trade_analysis.get('top_strategy', 'N/A')}\n")
                f.write("\n")
                
                # ファイルリスト
                f.write("【生成ファイル】\n")
                f.write(f"  レポートディレクトリ: {report_dir}\n")
                f.write(f"  - 詳細テキストレポート\n")
                f.write(f"  - 取引履歴CSV\n")
                f.write(f"  - パフォーマンスサマリーCSV\n")
                f.write(f"  - 実行結果JSON\n")
                f.write(f"  - パフォーマンスメトリクスJSON\n")
                f.write(f"  - 取引分析JSON\n")
                f.write("\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Summary report saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            self.logger.error(f"Summary report generation error: {e}")
            return ""


# テスト用関数
def test_comprehensive_reporter():
    """ComprehensiveReporter テスト"""
    from config.logger_config import setup_logger
    
    logger = setup_logger("ComprehensiveReporterTest", log_file="logs/comprehensive_reporter_test.log")
    
    try:
        print("ComprehensiveReporter テスト開始")
        print("="*80)
        
        # サンプルデータ生成
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'High': 102 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Low': 98 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Entry_Signal': [1 if i % 50 == 0 else 0 for i in range(len(dates))],
            'Exit_Signal': [1 if i % 50 == 25 else 0 for i in range(len(dates))],
            'Strategy': ['VWAPBreakoutStrategy' if i % 50 in [0, 25] else '' for i in range(len(dates))]
        }, index=dates)
        
        # サンプル実行結果
        execution_results = {
            'status': 'PARTIAL_SUCCESS',
            'total_executions': 3,
            'successful_strategies': 2,
            'failed_strategies': 1,
            'execution_results': []
        }
        
        # ComprehensiveReporter作成
        reporter = ComprehensiveReporter()
        
        # レポート生成
        result = reporter.generate_full_backtest_report(
            execution_results=execution_results,
            stock_data=sample_data,
            ticker='TEST',
            config=None
        )
        
        print("\n=== ComprehensiveReporter テスト結果 ===")
        print(f"ステータス: {result.get('status')}")
        print(f"レポートディレクトリ: {result.get('report_directory')}")
        print(f"生成ファイル数: CSV={len(result.get('csv_outputs', {}))}, JSON={len(result.get('json_outputs', {}))}")
        print("\n=== テスト完了 ===")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == '__main__':
    test_comprehensive_reporter()
