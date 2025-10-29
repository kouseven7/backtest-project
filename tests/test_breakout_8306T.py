"""
Breakout戦略 動作検証テスト - 8306.T（三菱UFJフィナンシャル・グループ）

このテストの目的:
- strategies/Breakout.pyが正常に動作しているかの確認
- 作成者の意図通りに動作しているかを確認
- マルチ戦略システムのバグ特定のため、戦略に問題がないかの確認

テスト要件:
- 対象ファイル: strategies/Breakout.py（修正禁止）
- データ取得: yfinance経由で8306.T、2024/01/01～2024/12/31
- 初期資金: 1,000,000円、取引単位: 100株、手数料: なし

検証項目:
1. シグナル生成の確認
2. 取引実行の確認
3. パフォーマンス計算
4. データ整合性

Author: Backtest Project Team
Created: 2025-10-26
Last Modified: 2025-10-26
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 必要なモジュールをインポート
from config.logger_config import setup_logger
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Breakout import BreakoutStrategy


class BreakoutStrategyTester:
    """Breakout戦略の包括的テストクラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(
            "BreakoutStrategyTester",
            log_file="logs/test_breakout_8306T.log"
        )
        
        # テスト設定
        self.ticker = "9101.T"
        self.start_date = "2024-01-01"
        self.end_date = "2024-12-31"
        self.initial_capital = 1000000  # 100万円
        self.shares_per_trade = 100  # 100株単位
        
        # 出力ディレクトリ
        self.output_dir = project_root / "tests" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.stock_data = None
        self.strategy = None
        self.backtest_result = None
        
        self.logger.info("=" * 80)
        self.logger.info("Breakout Strategy Tester initialized")
        self.logger.info(f"Ticker: {self.ticker}")
        self.logger.info(f"Period: {self.start_date} to {self.end_date}")
        self.logger.info(f"Initial Capital: JPY {self.initial_capital:,}")
        self.logger.info(f"Shares per Trade: {self.shares_per_trade}")
        self.logger.info("=" * 80)
    
    def fetch_data(self) -> bool:
        """
        yfinanceからデータ取得
        
        Returns:
            bool: 取得成功
        """
        try:
            self.logger.info(f"Fetching data for {self.ticker}...")
            
            data_feed = YFinanceDataFeed()
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or len(self.stock_data) == 0:
                self.logger.error("Data retrieval failed: empty dataset")
                return False
            
            self.logger.info(f"[OK] Data retrieved: {len(self.stock_data)} rows")
            self.logger.info(f"Columns: {list(self.stock_data.columns)}")
            self.logger.info(f"Date range: {self.stock_data.index[0]} to {self.stock_data.index[-1]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data retrieval error: {e}", exc_info=True)
            return False
    
    def initialize_strategy(self) -> bool:
        """
        Breakout戦略を初期化
        
        Returns:
            bool: 初期化成功
        """
        try:
            self.logger.info("Initializing BreakoutStrategy...")
            
            # 戦略初期化（まずデフォルトパラメータで初期化）
            self.strategy = BreakoutStrategy(
                data=self.stock_data.copy(),
                params=None,  # まずデフォルトで初期化
                price_column="Close",  # 配当調整の影響を避けるため Close を使用
                volume_column="Volume"
            )
            
            # ticker を設定（最適化パラメータ読み込みに必要）
            self.strategy.ticker = self.ticker
            
            # 最適化パラメータを読み込み
            try:
                if self.strategy.load_optimized_parameters():
                    self.logger.info("[OK] Loaded optimized parameters")
                else:
                    self.logger.info("[INFO] Using default parameters (no optimized params found)")
            except Exception as e:
                self.logger.warning(f"[WARNING] Failed to load optimized parameters: {e}")
                self.logger.info("[INFO] Using default parameters")
            
            # パラメータ確認（ログ出力）
            try:
                params_repr = getattr(self.strategy, 'params', None)
            except Exception:
                params_repr = None
            self.logger.info(f"[PARAMS] Strategy parameters: {params_repr}")
            self.logger.info(f"[PARAMS] Price column: {getattr(self.strategy, 'price_column', 'Close')}")
            self.logger.info(f"[PARAMS] Volume column: {getattr(self.strategy, 'volume_column', 'Volume')}")
            
            # 最適化パラメータ情報の表示（使用されている場合）
            if hasattr(self.strategy, '_approved_params') and self.strategy._approved_params:
                self.logger.info("[OPTIMIZATION] Using approved optimized parameters:")
                self.logger.info(f"  Parameter ID: {self.strategy._approved_params.get('parameter_id', 'N/A')}")
                self.logger.info(f"  Created at: {self.strategy._approved_params.get('created_at', 'N/A')}")
                self.logger.info(f"  Sharpe ratio: {self.strategy._approved_params.get('sharpe_ratio', 'N/A')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy initialization error: {e}", exc_info=True)
            return False
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行
        
        Returns:
            bool: 実行成功
        """
        try:
            self.logger.info("Running backtest...")
            
            # バックテスト実行（strategies/Breakout.pyのbacktest()メソッドを呼び出し）
            self.backtest_result = self.strategy.backtest()
            
            if self.backtest_result is None:
                self.logger.error("Backtest returned None")
                return False
            
            self.logger.info(f"[OK] Backtest completed: {len(self.backtest_result)} rows")
            self.logger.info(f"Result columns: {list(self.backtest_result.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backtest execution error: {e}", exc_info=True)
            return False
    
    def verify_signals(self) -> Dict[str, Any]:
        """
        シグナル生成の確認
        
        Returns:
            Dict[str, Any]: シグナル検証結果
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("[VERIFICATION 1/4] Signal Generation")
            self.logger.info("=" * 80)
            
            # Entry_Signal == 1 の回数
            entry_signals = self.backtest_result[self.backtest_result['Entry_Signal'] == 1]
            entry_count = len(entry_signals)
            
            # Exit_Signal == -1 の回数
            exit_signals = self.backtest_result[self.backtest_result['Exit_Signal'] == -1]
            exit_count = len(exit_signals)
            
            # エントリーとエグジットの数が一致するか
            signals_match = (entry_count == exit_count)
            
            result = {
                'entry_count': entry_count,
                'exit_count': exit_count,
                'signals_match': signals_match,
                'entry_dates': entry_signals.index.tolist(),
                'exit_dates': exit_signals.index.tolist()
            }
            
            # ログ出力
            self.logger.info(f"Entry signals: {entry_count}")
            self.logger.info(f"Exit signals: {exit_count}")
            self.logger.info(f"Signals match: {signals_match}")
            
            if entry_count > 0:
                self.logger.info(f"Entry dates: {result['entry_dates'][:5]}..." if entry_count > 5 else f"Entry dates: {result['entry_dates']}")
            
            if exit_count > 0:
                self.logger.info(f"Exit dates: {result['exit_dates'][:5]}..." if exit_count > 5 else f"Exit dates: {result['exit_dates']}")
            
            if not signals_match:
                self.logger.warning(f"[WARNING] Entry/Exit count mismatch: {entry_count} vs {exit_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Signal verification error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def verify_execution(self) -> Dict[str, Any]:
        """
        取引実行の確認
        
        Returns:
            Dict[str, Any]: 実行検証結果
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("[VERIFICATION 2/4] Trade Execution")
            self.logger.info("=" * 80)
            
            # Position == 1 が発生したか（エントリー）
            # Position列の存在を安全にチェック
            has_entries = False
            if 'Position' in self.backtest_result.columns:
                position_entries = self.backtest_result[self.backtest_result['Position'] == 1]
                has_entries = len(position_entries) > 0
            
            # Position == 0 が発生したか（エグジット）
            # 注: Positionは初期値が0なので、エントリー後に0に戻ったかを確認
            has_exits = ('Exit_Signal' in self.backtest_result.columns and 
                        (self.backtest_result['Exit_Signal'] == -1).any())
            
            # エントリー日とエグジット日のリスト
            entry_indices = self.backtest_result[self.backtest_result['Entry_Signal'] == 1].index
            exit_indices = self.backtest_result[self.backtest_result['Exit_Signal'] == -1].index
            
            result = {
                'has_entries': has_entries,
                'has_exits': has_exits,
                'entry_dates': entry_indices.tolist(),
                'exit_dates': exit_indices.tolist(),
                'entry_count': len(entry_indices),
                'exit_count': len(exit_indices)
            }
            
            # ログ出力
            self.logger.info(f"Entries occurred: {has_entries}")
            self.logger.info(f"Exits occurred: {has_exits}")
            self.logger.info(f"Entry count: {result['entry_count']}")
            self.logger.info(f"Exit count: {result['exit_count']}")
            
            if result['entry_count'] > 0:
                self.logger.info(f"First entry: {result['entry_dates'][0]}")
                self.logger.info(f"Last entry: {result['entry_dates'][-1]}")
            
            if result['exit_count'] > 0:
                self.logger.info(f"First exit: {result['exit_dates'][0]}")
                self.logger.info(f"Last exit: {result['exit_dates'][-1]}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution verification error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def calculate_performance(self) -> Dict[str, Any]:
        """
        パフォーマンス計算
        
        Returns:
            Dict[str, Any]: パフォーマンス指標
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("[VERIFICATION 3/4] Performance Calculation")
            self.logger.info("=" * 80)
            
            # エントリー・エグジットペアの抽出
            trades = self._extract_trades()
            
            if not trades:
                self.logger.warning("No completed trades found")
                return {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'total_return': 0,
                    'win_rate': 0,
                    'avg_holding_days': 0
                }
            
            # パフォーマンス指標計算
            total_trades = len(trades)
            total_pnl = sum(trade['pnl'] for trade in trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_holding_days = np.mean([t['holding_days'] for t in trades]) if trades else 0
            
            total_return = total_pnl / self.initial_capital
            
            result = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_pnl': total_pnl,
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_holding_days': avg_holding_days,
                'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'max_win': max([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'max_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            }
            
            # ログ出力
            self.logger.info(f"Total trades: {result['total_trades']}")
            self.logger.info(f"Winning trades: {result['winning_trades']}")
            self.logger.info(f"Losing trades: {result['losing_trades']}")
            self.logger.info(f"Total PnL: JPY {result['total_pnl']:,.0f}")
            self.logger.info(f"Total return: {result['total_return']:.2%}")
            self.logger.info(f"Win rate: {result['win_rate']:.2%}")
            self.logger.info(f"Avg holding days: {result['avg_holding_days']:.1f}")
            self.logger.info(f"Avg win: JPY {result['avg_win']:,.0f}")
            self.logger.info(f"Avg loss: JPY {result['avg_loss']:,.0f}")
            self.logger.info(f"Max win: JPY {result['max_win']:,.0f}")
            self.logger.info(f"Max loss: JPY {result['max_loss']:,.0f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Performance calculation error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _extract_trades(self) -> List[Dict[str, Any]]:
        """
        エントリー・エグジットペアから取引データを抽出
        
        Returns:
            List[Dict[str, Any]]: 取引リスト
        """
        trades = []
        
        entry_indices = self.backtest_result[self.backtest_result['Entry_Signal'] == 1].index
        exit_indices = self.backtest_result[self.backtest_result['Exit_Signal'] == -1].index
        
        # ペアリング（FIFO）
        for entry_idx, exit_idx in zip(entry_indices, exit_indices):
            try:
                entry_price = self.backtest_result.loc[entry_idx, self.strategy.price_column]
                exit_price = self.backtest_result.loc[exit_idx, self.strategy.price_column]
                
                # 損益計算（手数料なし）
                pnl = (exit_price - entry_price) * self.shares_per_trade
                
                # 保有期間
                holding_days = (exit_idx - entry_idx).days
                
                trade = {
                    'entry_date': entry_idx,
                    'exit_date': exit_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': (exit_price - entry_price) / entry_price,
                    'holding_days': holding_days
                }
                
                trades.append(trade)
                
            except Exception as e:
                self.logger.error(f"Trade extraction error at {entry_idx}: {e}")
                continue
        
        return trades
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """
        データ整合性の確認
        
        Returns:
            Dict[str, Any]: 整合性検証結果
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("[VERIFICATION 4/4] Data Integrity")
            self.logger.info("=" * 80)
            
            # 1. entry_prices辞書の整合性
            entry_prices_valid = self._verify_entry_prices()
            
            # 2. high_prices辞書の整合性
            high_prices_valid = self._verify_high_prices()
            
            # 3. パラメータ使用状況
            params_valid = self._verify_parameters()
            
            # 4. 未決済ポジション処理
            forced_exit_valid = self._verify_forced_exit()
            
            result = {
                'entry_prices_valid': entry_prices_valid,
                'high_prices_valid': high_prices_valid,
                'parameters_valid': params_valid,
                'forced_exit_valid': forced_exit_valid,
                'overall_valid': all([
                    entry_prices_valid,
                    high_prices_valid,
                    params_valid,
                    forced_exit_valid
                ])
            }
            
            # ログ出力
            self.logger.info(f"entry_prices integrity: {'OK' if entry_prices_valid else 'FAILED'}")
            self.logger.info(f"high_prices integrity: {'OK' if high_prices_valid else 'FAILED'}")
            self.logger.info(f"Parameters validity: {'OK' if params_valid else 'FAILED'}")
            self.logger.info(f"Forced exit handling: {'OK' if forced_exit_valid else 'FAILED'}")
            self.logger.info(f"Overall integrity: {'OK' if result['overall_valid'] else 'FAILED'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data integrity verification error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _verify_entry_prices(self) -> bool:
        """entry_prices辞書の整合性検証"""
        try:
            entry_signals = self.backtest_result[self.backtest_result['Entry_Signal'] == 1]
            
            if len(entry_signals) == 0:
                self.logger.info("[entry_prices] No entry signals to verify")
                return True
            
            # エントリーシグナルがある行のインデックスがentry_pricesに記録されているか
            for idx in entry_signals.index:
                idx_pos = self.backtest_result.index.get_loc(idx)
                
                if idx_pos not in self.strategy.entry_prices and idx not in self.strategy.entry_prices:
                    self.logger.warning(f"[entry_prices] Missing entry price for idx={idx} (pos={idx_pos})")
                    return False
                
                # 記録された価格が正しいか
                recorded_price = self.strategy.entry_prices.get(idx_pos) or self.strategy.entry_prices.get(idx)
                actual_price = self.backtest_result.loc[idx, self.strategy.price_column]
                
                if abs(recorded_price - actual_price) > 0.01:
                    self.logger.warning(
                        f"[entry_prices] Price mismatch at {idx}: "
                        f"recorded={recorded_price}, actual={actual_price}"
                    )
                    return False
            
            self.logger.info(f"[entry_prices] All {len(entry_signals)} entries have valid prices")
            return True
            
        except Exception as e:
            self.logger.error(f"entry_prices verification error: {e}")
            return False
    
    def _verify_high_prices(self) -> bool:
        """high_prices辞書の整合性検証"""
        try:
            entry_signals = self.backtest_result[self.backtest_result['Entry_Signal'] == 1]
            
            if len(entry_signals) == 0:
                self.logger.info("[high_prices] No entry signals to verify")
                return True
            
            # エントリー時に高値が記録されているか
            for idx in entry_signals.index:
                idx_pos = self.backtest_result.index.get_loc(idx)
                
                if idx_pos not in self.strategy.high_prices and idx not in self.strategy.high_prices:
                    self.logger.warning(f"[high_prices] Missing high price for idx={idx} (pos={idx_pos})")
                    return False
            
            self.logger.info(f"[high_prices] All {len(entry_signals)} entries have high prices recorded")
            return True
            
        except Exception as e:
            self.logger.error(f"high_prices verification error: {e}")
            return False
    
    def _verify_parameters(self) -> bool:
        """パラメータ使用状況の検証"""
        try:
            # デフォルトパラメータと実際のパラメータを比較
            default_params = {
                "volume_threshold": 1.2,
                "take_profit": 0.03,
                "look_back": 1,
                "trailing_stop": 0.02,
                "breakout_buffer": 0.01
            }
            
            params_match = True
            for key, default_value in default_params.items():
                actual_value = self.strategy.params.get(key)
                
                if actual_value != default_value:
                    self.logger.warning(
                        f"[parameters] Parameter mismatch: {key} "
                        f"(expected={default_value}, actual={actual_value})"
                    )
                    params_match = False
            
            if params_match:
                self.logger.info("[parameters] All default parameters are correctly applied")
            
            return params_match
            
        except Exception as e:
            self.logger.error(f"Parameters verification error: {e}")
            return False
    
    def _verify_forced_exit(self) -> bool:
        """未決済ポジション処理の検証"""
        try:
            # 最後のエントリーとエグジットを確認
            entry_indices = self.backtest_result[self.backtest_result['Entry_Signal'] == 1].index
            exit_indices = self.backtest_result[self.backtest_result['Exit_Signal'] == -1].index
            
            if len(entry_indices) == 0:
                self.logger.info("[forced_exit] No entries to verify")
                return True
            
            last_entry = entry_indices[-1]
            
            # 最後のエントリーに対応するエグジットがあるか
            if len(exit_indices) == 0 or exit_indices[-1] < last_entry:
                self.logger.warning(
                    f"[forced_exit] Last entry ({last_entry}) has no corresponding exit"
                )
                return False
            
            # バックテスト終了時の強制決済が正しく動作しているか
            # (Entry_Signal == 1 の数 == Exit_Signal == -1 の数)
            if len(entry_indices) != len(exit_indices):
                self.logger.warning(
                    f"[forced_exit] Entry/Exit count mismatch: "
                    f"entries={len(entry_indices)}, exits={len(exit_indices)}"
                )
                return False
            
            self.logger.info("[forced_exit] Forced exit handling is correct")
            return True
            
        except Exception as e:
            self.logger.error(f"Forced exit verification error: {e}")
            return False
    
    def save_results(self, signal_result: Dict, execution_result: Dict, 
                    performance_result: Dict, integrity_result: Dict):
        """
        テスト結果をCSVとテキストで保存
        
        Args:
            signal_result: シグナル検証結果
            execution_result: 実行検証結果
            performance_result: パフォーマンス結果
            integrity_result: 整合性検証結果
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 取引一覧CSV
            if performance_result.get('total_trades', 0) > 0:
                trades = self._extract_trades()
                trades_df = pd.DataFrame(trades)
                trades_csv = self.output_dir / f"breakout_8306T_trades_{timestamp}.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8')
                self.logger.info(f"Trades CSV saved: {trades_csv}")
            
            # 2. サマリーCSV
            summary_data = {
                'Metric': [
                    'Ticker', 'Period Start', 'Period End',
                    'Total Trades', 'Winning Trades', 'Losing Trades',
                    'Total PnL (JPY)', 'Total Return (%)', 'Win Rate (%)',
                    'Avg Holding Days', 'Avg Win (JPY)', 'Avg Loss (JPY)',
                    'Entry Count', 'Exit Count', 'Signals Match',
                    'Data Integrity'
                ],
                'Value': [
                    self.ticker, self.start_date, self.end_date,
                    performance_result.get('total_trades', 0),
                    performance_result.get('winning_trades', 0),
                    performance_result.get('losing_trades', 0),
                    f"{performance_result.get('total_pnl', 0):,.0f}",
                    f"{performance_result.get('total_return', 0) * 100:.2f}",
                    f"{performance_result.get('win_rate', 0) * 100:.2f}",
                    f"{performance_result.get('avg_holding_days', 0):.1f}",
                    f"{performance_result.get('avg_win', 0):,.0f}",
                    f"{performance_result.get('avg_loss', 0):,.0f}",
                    signal_result.get('entry_count', 0),
                    signal_result.get('exit_count', 0),
                    'Yes' if signal_result.get('signals_match', False) else 'No',
                    'OK' if integrity_result.get('overall_valid', False) else 'FAILED'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = self.output_dir / f"breakout_8306T_summary_{timestamp}.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
            self.logger.info(f"Summary CSV saved: {summary_csv}")
            
            # 3. 詳細テキストレポート
            report_path = self.output_dir / f"breakout_8306T_report_{timestamp}.txt"
            self._generate_text_report(
                report_path, signal_result, execution_result,
                performance_result, integrity_result
            )
            
        except Exception as e:
            self.logger.error(f"Results saving error: {e}", exc_info=True)
    
    def _generate_text_report(self, report_path: Path, 
                             signal_result: Dict, execution_result: Dict,
                             performance_result: Dict, integrity_result: Dict):
        """詳細テキストレポート生成"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Breakout Strategy Test Report - 8306.T\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ticker: {self.ticker}\n")
            f.write(f"Period: {self.start_date} to {self.end_date}\n")
            f.write(f"Initial Capital: JPY {self.initial_capital:,}\n")
            f.write(f"Shares per Trade: {self.shares_per_trade}\n")
            f.write("\n")
            
            # シグナル検証
            f.write("1. SIGNAL GENERATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Entry Signals: {signal_result.get('entry_count', 0)}\n")
            f.write(f"Exit Signals: {signal_result.get('exit_count', 0)}\n")
            f.write(f"Signals Match: {'Yes' if signal_result.get('signals_match', False) else 'No'}\n")
            f.write("\n")
            
            # 実行検証
            f.write("2. TRADE EXECUTION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Entries Occurred: {'Yes' if execution_result.get('has_entries', False) else 'No'}\n")
            f.write(f"Exits Occurred: {'Yes' if execution_result.get('has_exits', False) else 'No'}\n")
            f.write(f"Entry Count: {execution_result.get('entry_count', 0)}\n")
            f.write(f"Exit Count: {execution_result.get('exit_count', 0)}\n")
            f.write("\n")
            
            # パフォーマンス
            f.write("3. PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trades: {performance_result.get('total_trades', 0)}\n")
            f.write(f"Winning Trades: {performance_result.get('winning_trades', 0)}\n")
            f.write(f"Losing Trades: {performance_result.get('losing_trades', 0)}\n")
            f.write(f"Total PnL: JPY {performance_result.get('total_pnl', 0):,.0f}\n")
            f.write(f"Total Return: {performance_result.get('total_return', 0) * 100:.2f}%\n")
            f.write(f"Win Rate: {performance_result.get('win_rate', 0) * 100:.2f}%\n")
            f.write(f"Avg Holding Days: {performance_result.get('avg_holding_days', 0):.1f}\n")
            f.write("\n")
            
            # 整合性
            f.write("4. DATA INTEGRITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"entry_prices: {'OK' if integrity_result.get('entry_prices_valid', False) else 'FAILED'}\n")
            f.write(f"high_prices: {'OK' if integrity_result.get('high_prices_valid', False) else 'FAILED'}\n")
            f.write(f"Parameters: {'OK' if integrity_result.get('parameters_valid', False) else 'FAILED'}\n")
            f.write(f"Forced Exit: {'OK' if integrity_result.get('forced_exit_valid', False) else 'FAILED'}\n")
            f.write(f"Overall: {'OK' if integrity_result.get('overall_valid', False) else 'FAILED'}\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Text report saved: {report_path}")
    
    def run_full_test(self) -> bool:
        """
        包括テスト実行
        
        Returns:
            bool: テスト成功
        """
        try:
            # ステップ1: データ取得
            if not self.fetch_data():
                return False
            
            # ステップ2: 戦略初期化
            if not self.initialize_strategy():
                return False
            
            # ステップ3: バックテスト実行
            if not self.run_backtest():
                return False
            
            # ステップ4: 検証実行
            signal_result = self.verify_signals()
            execution_result = self.verify_execution()
            performance_result = self.calculate_performance()
            integrity_result = self.verify_data_integrity()
            
            # ステップ5: 結果保存
            self.save_results(
                signal_result, execution_result,
                performance_result, integrity_result
            )
            
            # テスト成功条件
            test_success = (
                signal_result.get('entry_count', 0) > 0 and
                signal_result.get('signals_match', False) and
                integrity_result.get('overall_valid', False)
            )
            
            self.logger.info("=" * 80)
            if test_success:
                self.logger.info("[SUCCESS] Test completed successfully")
            else:
                self.logger.warning("[WARNING] Test completed with warnings")
            self.logger.info("=" * 80)
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"Full test execution error: {e}", exc_info=True)
            return False


def main():
    """メインエントリーポイント"""
    print("\n" + "=" * 80)
    print("Breakout Strategy Test - 8306.T")
    print("=" * 80 + "\n")
    
    tester = BreakoutStrategyTester()
    success = tester.run_full_test()
    
    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] Test completed successfully")
    else:
        print("[WARNING] Test completed with warnings - check logs for details")
    print("=" * 80 + "\n")
    
    return success


if __name__ == "__main__":
    main()
