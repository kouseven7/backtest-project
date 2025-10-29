"""
test_momentum_investing_8306T.py - MomentumInvestingStrategy 単体テスト

テスト対象: strategies/Momentum_Investing.py
テスト銘柄: 8306.T（三菱UFJフィナンシャル・グループ）
テスト期間: 2024/01/01 ~ 2024/12/31

テスト目的:
- MomentumInvestingStrategyが正常に動作しているかの確認
- 作成者の意図通りに動作しているかを確認
- マルチ戦略システムのバグ特定のため、戦略に問題がないかの確認

検証項目:
1. シグナル生成の確認（Entry_Signal, Exit_Signal）
2. 取引実行の確認（エントリー日、エグジット日）
3. パフォーマンス計算（総損益、総損益率）
4. データ整合性（エントリー/エグジット数の一致）

Author: Backtest Project Team
Created: 2025-10-23
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import logging

# プロジェクトパス設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# テスト対象モジュール
from strategies.Momentum_Investing import MomentumInvestingStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# ロガー設定（出力ディレクトリを事前作成）
log_dir = Path("tests/results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'test_momentum_8306T.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class MomentumStrategyTester:
    """MomentumInvestingStrategy テストクラス"""
    
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
        self.stock_data = None
        self.strategy = None
        self.result = None
        
        logger.info("=" * 80)
        logger.info("MomentumInvestingStrategy 単体テスト開始")
        logger.info("=" * 80)
        logger.info(f"テスト銘柄: {ticker}")
        logger.info(f"テスト期間: {start_date} ~ {end_date}")
        logger.info("")
    
    def fetch_data(self) -> bool:
        """
        yfinanceからデータ取得
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        
        try:
            data_feed = YFinanceDataFeed()
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"[SUCCESS] データ取得完了: {len(self.stock_data)} 行")
            logger.info(f"  カラム: {self.stock_data.columns.tolist()}")
            logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            logger.info(f"  最初の価格: {self.stock_data['Close'].iloc[0]:.2f} 円")
            logger.info(f"  最後の価格: {self.stock_data['Close'].iloc[-1]:.2f} 円")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] データ取得失敗: {e}")
            return False
    
    def initialize_strategy(self) -> bool:
        """
        戦略初期化
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 2] 戦略初期化")
        logger.info("-" * 80)
        
        try:
            # デフォルトパラメータで初期化
            self.strategy = MomentumInvestingStrategy(
                data=self.stock_data,
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            logger.info("[SUCCESS] MomentumInvestingStrategy 初期化完了")
            logger.info(f"  戦略パラメータ:")
            for key, value in self.strategy.params.items():
                logger.info(f"    {key}: {value}")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 戦略初期化失敗: {e}")
            return False
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 3] バックテスト実行")
        logger.info("-" * 80)
        
        try:
            self.result = self.strategy.backtest()
            
            logger.info("[SUCCESS] バックテスト実行完了")
            logger.info(f"  結果データ行数: {len(self.result)}")
            logger.info(f"  結果カラム: {self.result.columns.tolist()}")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] バックテスト実行失敗: {e}")
            return False
    
    def verify_signals(self) -> Dict[str, Any]:
        """
        シグナル生成の検証
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 4] シグナル生成の検証")
        logger.info("-" * 80)
        
        # エントリーシグナルのカウント
        entry_signals = self.result[self.result['Entry_Signal'] == 1]
        entry_count = len(entry_signals)
        
        # エグジットシグナルのカウント
        exit_signals = self.result[self.result['Exit_Signal'] == -1]
        exit_count = len(exit_signals)
        
        # 整合性チェック
        signals_match = (entry_count == exit_count)
        
        logger.info(f"[SIGNAL CHECK]")
        logger.info(f"  Entry_Signal == 1: {entry_count} 回")
        logger.info(f"  Exit_Signal == -1: {exit_count} 回")
        logger.info(f"  整合性: {'OK' if signals_match else 'NG - 不一致!'}")
        
        if not signals_match:
            logger.warning(f"  [WARNING] エントリーとエグジットの数が一致しません!")
        
        logger.info("")
        
        return {
            'entry_count': entry_count,
            'exit_count': exit_count,
            'signals_match': signals_match
        }
    
    def extract_trades(self) -> List[Dict[str, Any]]:
        """
        取引詳細の抽出
        
        Returns:
            list: 取引リスト
        """
        logger.info("[STEP 5] 取引詳細の抽出")
        logger.info("-" * 80)
        
        trades = []
        entry_signals = self.result[self.result['Entry_Signal'] == 1]
        exit_signals = self.result[self.result['Exit_Signal'] == -1]
        
        # エントリーとエグジットをペアリング
        for entry_idx, exit_idx in zip(entry_signals.index, exit_signals.index):
            entry_date = entry_idx
            exit_date = exit_idx
            entry_price = self.result.loc[entry_idx, 'Adj Close']
            exit_price = self.result.loc[exit_idx, 'Adj Close']
            
            # 損益計算（100株仮定）
            shares = 100
            pnl = (exit_price - entry_price) * shares
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # 保有期間計算
            holding_days = (exit_date - entry_date).days
            
            trade = {
                'entry_date': entry_date.strftime("%Y-%m-%d"),
                'exit_date': exit_date.strftime("%Y-%m-%d"),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'return_pct': return_pct,
                'holding_days': holding_days
            }
            trades.append(trade)
        
        logger.info(f"[TRADES EXTRACTED] {len(trades)} 件の取引を抽出")
        logger.info("")
        
        return trades
    
    def calculate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        パフォーマンス計算
        
        Args:
            trades: 取引リスト
        
        Returns:
            dict: パフォーマンス指標
        """
        logger.info("[STEP 6] パフォーマンス計算")
        logger.info("-" * 80)
        
        if not trades:
            logger.warning("[WARNING] 取引が0件のため、パフォーマンス計算をスキップ")
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_return_pct': 0
            }
        
        # 総損益計算
        total_pnl = sum(trade['pnl'] for trade in trades)
        
        # 総損益率計算（累積リターン）
        total_return_pct = sum(trade['return_pct'] for trade in trades)
        
        # 平均保有期間
        avg_holding_days = sum(trade['holding_days'] for trade in trades) / len(trades)
        
        logger.info(f"[PERFORMANCE]")
        logger.info(f"  総取引回数: {len(trades)} 回")
        logger.info(f"  総損益: {total_pnl:,.2f} 円")
        logger.info(f"  総損益率: {total_return_pct:.2f} %")
        logger.info(f"  平均保有期間: {avg_holding_days:.1f} 日")
        logger.info("")
        
        return {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_holding_days': avg_holding_days
        }
    
    def verify_entry_prices(self) -> Dict[str, Any]:
        """
        entry_prices辞書の検証
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 7] entry_prices辞書の検証")
        logger.info("-" * 80)
        
        entry_prices_count = len(self.strategy.entry_prices)
        expected_count = (self.result['Entry_Signal'] == 1).sum()
        
        prices_match = (entry_prices_count == expected_count)
        
        logger.info(f"[ENTRY_PRICES CHECK]")
        logger.info(f"  記録されたentry_prices: {entry_prices_count} 件")
        logger.info(f"  期待されるentry_prices: {expected_count} 件")
        logger.info(f"  整合性: {'OK' if prices_match else 'NG - 不一致!'}")
        
        if not prices_match:
            logger.warning(f"  [WARNING] entry_prices辞書の件数が不一致!")
        
        logger.info("")
        
        return {
            'entry_prices_count': entry_prices_count,
            'expected_count': expected_count,
            'prices_match': prices_match
        }
    
    def save_results(self, trades: List[Dict[str, Any]], performance: Dict[str, Any]) -> None:
        """
        結果をCSVに保存
        
        Args:
            trades: 取引リスト
            performance: パフォーマンス指標
        """
        logger.info("[STEP 8] 結果保存")
        logger.info("-" * 80)
        
        try:
            # 出力ディレクトリ作成
            output_dir = Path("tests/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 取引履歴CSV
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_csv = output_dir / "momentum_8306T_trades.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
                logger.info(f"[SUCCESS] 取引履歴CSV保存: {trades_csv}")
            
            # サマリーCSV
            summary_data = {
                'ticker': [self.ticker],
                'start_date': [self.start_date],
                'end_date': [self.end_date],
                'total_trades': [performance['total_trades']],
                'total_pnl': [performance['total_pnl']],
                'total_return_pct': [performance['total_return_pct']],
                'avg_holding_days': [performance.get('avg_holding_days', 0)]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = output_dir / "momentum_8306T_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            logger.info(f"[SUCCESS] サマリーCSV保存: {summary_csv}")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"[ERROR] 結果保存失敗: {e}")
    
    def print_trade_details(self, trades: List[Dict[str, Any]]) -> None:
        """
        取引詳細をログ出力
        
        Args:
            trades: 取引リスト
        """
        logger.info("[取引詳細]")
        logger.info("-" * 80)
        
        if not trades:
            logger.info("  取引なし")
            logger.info("")
            return
        
        for i, trade in enumerate(trades, 1):
            logger.info(f"  [{i}] エントリー: {trade['entry_date']} @ {trade['entry_price']:.2f}円")
            logger.info(f"      エグジット: {trade['exit_date']} @ {trade['exit_price']:.2f}円")
            logger.info(f"      保有期間: {trade['holding_days']}日")
            logger.info(f"      損益: {trade['pnl']:,.2f}円 ({trade['return_pct']:.2f}%)")
            logger.info("")
    
    def run_full_test(self) -> bool:
        """
        フルテスト実行
        
        Returns:
            bool: テスト成功時True
        """
        # Step 1: データ取得
        if not self.fetch_data():
            logger.error("[TEST FAILED] データ取得失敗")
            return False
        
        # Step 2: 戦略初期化
        if not self.initialize_strategy():
            logger.error("[TEST FAILED] 戦略初期化失敗")
            return False
        
        # Step 3: バックテスト実行
        if not self.run_backtest():
            logger.error("[TEST FAILED] バックテスト実行失敗")
            return False
        
        # Step 4: シグナル検証
        signal_result = self.verify_signals()
        
        # Step 5: 取引抽出
        trades = self.extract_trades()
        
        # Step 6: パフォーマンス計算
        performance = self.calculate_performance(trades)
        
        # Step 7: entry_prices検証
        entry_prices_result = self.verify_entry_prices()
        
        # Step 8: 結果保存
        self.save_results(trades, performance)
        
        # Step 9: 取引詳細出力
        self.print_trade_details(trades)
        
        # テスト成功条件チェック
        logger.info("[TEST RESULT]")
        logger.info("=" * 80)
        
        # 条件1: エントリー回数 > 0
        condition1 = signal_result['entry_count'] > 0
        logger.info(f"  条件1（エントリー回数 > 0）: {'PASS' if condition1 else 'FAIL'}")
        
        # 条件2: エントリー回数 == エグジット回数
        condition2 = signal_result['signals_match']
        logger.info(f"  条件2（エントリー == エグジット）: {'PASS' if condition2 else 'FAIL'}")
        
        # 条件3: entry_prices辞書の整合性
        condition3 = entry_prices_result['prices_match']
        logger.info(f"  条件3（entry_prices整合性）: {'PASS' if condition3 else 'FAIL'}")
        
        # 総合判定
        all_passed = condition1 and condition2 and condition3
        logger.info("")
        logger.info(f"  総合結果: {'TEST PASSED' if all_passed else 'TEST FAILED'}")
        logger.info("=" * 80)
        
        return all_passed


def main():
    """メインエントリーポイント"""
    # テスト設定
    ticker = "8306.T"  # 三菱UFJフィナンシャル・グループ
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # テスト実行
    tester = MomentumStrategyTester(ticker, start_date, end_date)
    success = tester.run_full_test()
    
    # 終了コード
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
