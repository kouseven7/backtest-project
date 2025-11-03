"""
test_momentum_investing_standalone_9101T.py - MomentumInvestingStrategy 単体テスト

main_new.py実行結果との比較検証用スクリプト
- 銘柄: 9101.T
- 期間: 2024-01-01 ~ 2024-12-31
- パラメータ: デフォルト
- 出力: CSV + テキストレポート

主な機能:
- data_fetcherから実データ取得（main_new.pyと同じソース）
- MomentumInvestingStrategy単体実行
- 取引履歴CSV出力（trades.csv互換）
- 包括的テキストレポート生成
- main_new.pyとの比較分析

統合コンポーネント:
- data_fetcher: データ取得
- strategies.Momentum_Investing: 戦略実行
- indicators: 各種テクニカル指標

セーフティ機能/注意事項:
- copilot-instructions.md準拠: 実データのみ使用
- モック/ダミーデータのフォールバック禁止
- 実際の実行結果を検証して報告

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from data_fetcher import get_parameters_and_data
from strategies.Momentum_Investing import MomentumInvestingStrategy


class MomentumInvestingStandaloneTest:
    """MomentumInvestingStrategy 単体テストクラス"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        初期化
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        self.logger = setup_logger(
            "MomentumInvestingStandaloneTest",
            log_file="logs/momentum_investing_standalone_test.log"
        )
        
        # 実行結果格納
        self.stock_data = None
        self.index_data = None
        self.backtest_result = None
        self.trades = []
        self.performance_metrics = {}
        
        # 出力ディレクトリ
        self.output_dir = Path("output/standalone_tests") / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized test for {ticker} ({start_date} to {end_date})")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def step1_fetch_data(self) -> bool:
        """
        STEP 1: データ取得（data_fetcherから実データ取得）
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: データ取得開始")
        self.logger.info("=" * 80)
        
        try:
            # data_fetcher.get_parameters_and_data()を使用
            ticker_result, start_result, end_result, stock_data, index_data = get_parameters_and_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            self.stock_data = stock_data
            self.index_data = index_data
            
            self.logger.info(f"データ取得成功:")
            self.logger.info(f"  銘柄: {ticker_result}")
            self.logger.info(f"  期間: {start_result} ~ {end_result}")
            self.logger.info(f"  株価データ: {len(stock_data)} 行")
            self.logger.info(f"  インデックスデータ: {len(index_data) if index_data is not None else 'N/A'} 行")
            self.logger.info(f"  株価データカラム: {stock_data.columns.tolist()}")
            
            # データ検証
            if len(stock_data) == 0:
                self.logger.error("株価データが空です")
                return False
            
            # 必須カラム確認
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                self.logger.error(f"必須カラムが不足: {missing_columns}")
                return False
            
            self.logger.info("STEP 1: データ取得完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 1 失敗: {e}", exc_info=True)
            return False
    
    def step2_initialize_strategy(self) -> bool:
        """
        STEP 2: 戦略初期化（デフォルトパラメータ）
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: 戦略初期化開始")
        self.logger.info("=" * 80)
        
        try:
            # MomentumInvestingStrategy初期化（デフォルトパラメータ）
            self.strategy = MomentumInvestingStrategy(
                data=self.stock_data,
                params=None,  # デフォルトパラメータ使用
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            self.logger.info("戦略初期化成功")
            self.logger.info(f"使用パラメータ:")
            for key, value in self.strategy.params.items():
                self.logger.info(f"  {key}: {value}")
            
            # 戦略固有の初期化処理
            self.strategy.initialize_strategy()
            
            self.logger.info("STEP 2: 戦略初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 2 失敗: {e}", exc_info=True)
            return False
    
    def step3_run_backtest(self) -> bool:
        """
        STEP 3: バックテスト実行
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: バックテスト実行開始")
        self.logger.info("=" * 80)
        
        try:
            # バックテスト実行
            self.backtest_result = self.strategy.backtest()
            
            self.logger.info("バックテスト実行成功")
            self.logger.info(f"結果データ形状: {self.backtest_result.shape}")
            self.logger.info(f"結果カラム: {self.backtest_result.columns.tolist()}")
            
            # シグナル統計
            entry_count = (self.backtest_result['Entry_Signal'] == 1).sum()
            exit_count = (self.backtest_result['Exit_Signal'] == -1).sum()
            
            self.logger.info(f"エントリーシグナル数: {entry_count}")
            self.logger.info(f"エグジットシグナル数: {exit_count}")
            
            if entry_count != exit_count:
                self.logger.warning(f"警告: エントリーとエグジットの数が不一致 (Entry={entry_count}, Exit={exit_count})")
            
            self.logger.info("STEP 3: バックテスト実行完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 3 失敗: {e}", exc_info=True)
            return False
    
    def step4_extract_trades(self) -> bool:
        """
        STEP 4: 取引履歴抽出
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: 取引履歴抽出開始")
        self.logger.info("=" * 80)
        
        try:
            trades = []
            
            # エントリーシグナルの位置を取得
            entry_signals = self.backtest_result[self.backtest_result['Entry_Signal'] == 1]
            exit_signals = self.backtest_result[self.backtest_result['Exit_Signal'] == -1]
            
            self.logger.info(f"エントリー数: {len(entry_signals)}")
            self.logger.info(f"エグジット数: {len(exit_signals)}")
            
            # エントリーとエグジットをペアリング
            for entry_date, entry_row in entry_signals.iterrows():
                # このエントリー以降の最初のエグジットを探す
                future_exits = exit_signals[exit_signals.index > entry_date]
                
                if len(future_exits) == 0:
                    self.logger.warning(f"エントリー {entry_date} に対応するエグジットが見つかりません")
                    continue
                
                exit_date = future_exits.index[0]
                exit_row = self.backtest_result.loc[exit_date]
                
                # 取引レコード作成
                entry_price = entry_row['Adj Close']
                exit_price = exit_row['Adj Close']
                shares = 100  # main_new.pyと合わせる
                
                pnl = (exit_price - entry_price) * shares
                return_pct = (exit_price - entry_price) / entry_price
                holding_period_days = (exit_date - entry_date).days
                
                trade = {
                    'entry_date': entry_date.isoformat(),
                    'exit_date': exit_date.isoformat(),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'holding_period_days': holding_period_days,
                    'strategy': 'MomentumInvestingStrategy',
                    'position_value': entry_price * shares,
                    'is_forced_exit': False,
                    'is_executed_trade': True
                }
                
                trades.append(trade)
            
            self.trades = trades
            
            self.logger.info(f"取引履歴抽出完了: {len(trades)} 件")
            
            if len(trades) == 0:
                self.logger.warning("警告: 取引が0件です")
            
            self.logger.info("STEP 4: 取引履歴抽出完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 4 失敗: {e}", exc_info=True)
            return False
    
    def step5_calculate_metrics(self) -> bool:
        """
        STEP 5: パフォーマンス指標計算
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 5: パフォーマンス指標計算開始")
        self.logger.info("=" * 80)
        
        try:
            if len(self.trades) == 0:
                self.logger.warning("取引が0件のため、指標計算をスキップします")
                self.performance_metrics = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'average_pnl': 0.0,
                    'average_holding_days': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'profit_factor': 0.0
                }
                return True
            
            # 基本統計
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # PnL統計
            pnls = [t['pnl'] for t in self.trades]
            total_pnl = sum(pnls)
            average_pnl = np.mean(pnls)
            max_profit = max(pnls) if pnls else 0.0
            max_loss = min(pnls) if pnls else 0.0
            
            # 保有期間
            holding_days = [t['holding_period_days'] for t in self.trades]
            average_holding_days = np.mean(holding_days) if holding_days else 0.0
            
            # プロフィットファクター
            total_profit = sum([p for p in pnls if p > 0])
            total_loss = abs(sum([p for p in pnls if p <= 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
            
            self.performance_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_pnl': average_pnl,
                'average_holding_days': average_holding_days,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_factor': profit_factor
            }
            
            self.logger.info("パフォーマンス指標:")
            for key, value in self.performance_metrics.items():
                self.logger.info(f"  {key}: {value}")
            
            self.logger.info("STEP 5: パフォーマンス指標計算完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 5 失敗: {e}", exc_info=True)
            return False
    
    def step6_output_csv(self) -> bool:
        """
        STEP 6: CSV出力
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 6: CSV出力開始")
        self.logger.info("=" * 80)
        
        try:
            # trades.csv
            trades_csv_path = self.output_dir / f"{self.ticker}_trades_standalone.csv"
            
            if len(self.trades) > 0:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(trades_csv_path, index=False)
                self.logger.info(f"Trades CSV保存: {trades_csv_path}")
            else:
                self.logger.warning("取引が0件のため、trades.csvは作成されません")
            
            self.logger.info("STEP 6: CSV出力完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 6 失敗: {e}", exc_info=True)
            return False
    
    def step7_output_report(self) -> bool:
        """
        STEP 7: テキストレポート出力
        
        Returns:
            成功した場合True
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 7: テキストレポート出力開始")
        self.logger.info("=" * 80)
        
        try:
            report_path = self.output_dir / f"{self.ticker}_standalone_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("MomentumInvestingStrategy 単体テストレポート\n")
                f.write("=" * 80 + "\n")
                f.write(f"銘柄コード: {self.ticker}\n")
                f.write(f"期間: {self.start_date} ~ {self.end_date}\n")
                f.write(f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"データ行数: {len(self.stock_data)}\n")
                f.write("\n")
                
                f.write("1. 実行条件\n")
                f.write("-" * 80 + "\n")
                f.write("戦略: MomentumInvestingStrategy\n")
                f.write("パラメータ: デフォルト\n")
                f.write(f"  sma_short: {self.strategy.params['sma_short']}\n")
                f.write(f"  sma_long: {self.strategy.params['sma_long']}\n")
                f.write(f"  rsi_period: {self.strategy.params['rsi_period']}\n")
                f.write(f"  rsi_lower: {self.strategy.params['rsi_lower']}\n")
                f.write(f"  rsi_upper: {self.strategy.params['rsi_upper']}\n")
                f.write(f"  take_profit: {self.strategy.params['take_profit']:.2%}\n")
                f.write(f"  stop_loss: {self.strategy.params['stop_loss']:.2%}\n")
                f.write(f"  max_hold_days: {self.strategy.params['max_hold_days']}\n")
                f.write("\n")
                
                f.write("2. パフォーマンスサマリー\n")
                f.write("-" * 80 + "\n")
                metrics = self.performance_metrics
                f.write(f"総取引数: {metrics['total_trades']}\n")
                f.write(f"勝ちトレード数: {metrics['winning_trades']}\n")
                f.write(f"負けトレード数: {metrics['losing_trades']}\n")
                f.write(f"勝率: {metrics['win_rate']:.2%}\n")
                f.write(f"総PnL: {metrics['total_pnl']:.2f}円\n")
                f.write(f"平均PnL: {metrics['average_pnl']:.2f}円\n")
                f.write(f"平均保有期間: {metrics['average_holding_days']:.2f}日\n")
                f.write(f"最大利益: {metrics['max_profit']:.2f}円\n")
                f.write(f"最大損失: {metrics['max_loss']:.2f}円\n")
                f.write(f"プロフィットファクター: {metrics['profit_factor']:.2f}\n")
                f.write("\n")
                
                if len(self.trades) > 0:
                    f.write("3. 取引詳細（最初の10件）\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'No':<5} {'Entry Date':<12} {'Exit Date':<12} {'Entry Price':>12} {'Exit Price':>12} {'PnL':>12} {'Hold Days':>10}\n")
                    f.write("-" * 80 + "\n")
                    
                    for i, trade in enumerate(self.trades[:10], 1):
                        entry_date = trade['entry_date'][:10]
                        exit_date = trade['exit_date'][:10]
                        f.write(f"{i:<5} {entry_date:<12} {exit_date:<12} {trade['entry_price']:>12.2f} {trade['exit_price']:>12.2f} {trade['pnl']:>12.2f} {trade['holding_period_days']:>10}\n")
                    
                    if len(self.trades) > 10:
                        f.write(f"... および他 {len(self.trades) - 10} 件\n")
                    f.write("\n")
                
                f.write("4. main_new.py との比較\n")
                f.write("-" * 80 + "\n")
                f.write("比較対象:\n")
                f.write("  output/comprehensive_reports/9101.T_20251030_163444/9101.T_trades.csv\n")
                f.write("  output/comprehensive_reports/9101.T_20251030_163444/main_comprehensive_report_9101.T_20251030_163444.txt\n")
                f.write("\n")
                f.write("main_new.pyの結果:\n")
                f.write("  総取引数: 22\n")
                f.write("  勝率: 0.00%\n")
                f.write("  総PnL: -4,850円\n")
                f.write("  平均PnL: -220円\n")
                f.write("\n")
                f.write("単体テストの結果:\n")
                f.write(f"  総取引数: {metrics['total_trades']}\n")
                f.write(f"  勝率: {metrics['win_rate']:.2%}\n")
                f.write(f"  総PnL: {metrics['total_pnl']:.2f}円\n")
                f.write(f"  平均PnL: {metrics['average_pnl']:.2f}円\n")
                f.write("\n")
                
                # 差異分析
                main_trades = 22
                main_pnl = -4850
                diff_trades = metrics['total_trades'] - main_trades
                diff_pnl = metrics['total_pnl'] - main_pnl
                
                f.write("差異分析:\n")
                f.write(f"  取引数差異: {diff_trades:+d}\n")
                f.write(f"  PnL差異: {diff_pnl:+.2f}円\n")
                f.write("\n")
                
                if diff_trades == 0 and abs(diff_pnl) < 100:
                    f.write("結論: 単体テストとmain_new.pyの結果はほぼ一致しています。\n")
                else:
                    f.write("結論: 単体テストとmain_new.pyの結果に差異があります。\n")
                    f.write("原因候補:\n")
                    f.write("  - 初期資金の違い\n")
                    f.write("  - ポジションサイズ計算の違い\n")
                    f.write("  - 手数料計算の違い\n")
                    f.write("  - その他の実行環境の違い\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("レポート終了\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"テキストレポート保存: {report_path}")
            self.logger.info("STEP 7: テキストレポート出力完了")
            return True
            
        except Exception as e:
            self.logger.error(f"STEP 7 失敗: {e}", exc_info=True)
            return False
    
    def run_all_steps(self) -> bool:
        """全ステップ実行"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MomentumInvestingStrategy 単体テスト開始")
        self.logger.info("=" * 80 + "\n")
        
        steps = [
            ("STEP 1: データ取得", self.step1_fetch_data),
            ("STEP 2: 戦略初期化", self.step2_initialize_strategy),
            ("STEP 3: バックテスト実行", self.step3_run_backtest),
            ("STEP 4: 取引履歴抽出", self.step4_extract_trades),
            ("STEP 5: パフォーマンス指標計算", self.step5_calculate_metrics),
            ("STEP 6: CSV出力", self.step6_output_csv),
            ("STEP 7: テキストレポート出力", self.step7_output_report)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"\n実行中: {step_name}")
            if not step_func():
                self.logger.error(f"{step_name} 失敗")
                return False
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MomentumInvestingStrategy 単体テスト完了")
        self.logger.info(f"出力ディレクトリ: {self.output_dir}")
        self.logger.info("=" * 80 + "\n")
        
        return True


def main():
    """メインエントリーポイント"""
    print("\n" + "=" * 80)
    print("MomentumInvestingStrategy 単体テスト (9101.T, 2024-01-01~2024-12-31)")
    print("=" * 80 + "\n")
    
    # テスト条件
    ticker = "9101.T"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    print(f"銘柄: {ticker}")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"パラメータ: デフォルト")
    print("\n")
    
    # テスト実行
    test = MomentumInvestingStandaloneTest(ticker, start_date, end_date)
    success = test.run_all_steps()
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] テスト完了")
        print("=" * 80)
        print(f"\n出力ディレクトリ: {test.output_dir}")
        print(f"\n主要指標:")
        print(f"  総取引数: {test.performance_metrics['total_trades']}")
        print(f"  勝率: {test.performance_metrics['win_rate']:.2%}")
        print(f"  総PnL: {test.performance_metrics['total_pnl']:.2f}円")
        print(f"  平均保有期間: {test.performance_metrics['average_holding_days']:.2f}日")
        
        print(f"\nmain_new.pyとの比較:")
        print(f"  main_new.py: 総取引数=22, 総PnL=-4,850円")
        print(f"  単体テスト: 総取引数={test.performance_metrics['total_trades']}, 総PnL={test.performance_metrics['total_pnl']:.2f}円")
        
    else:
        print("\n" + "=" * 80)
        print("[ERROR] テスト失敗")
        print("=" * 80)
        print("ログファイルを確認してください: logs/momentum_investing_standalone_test.log")
    
    print("\n")
    
    return test if success else None


if __name__ == "__main__":
    main()
