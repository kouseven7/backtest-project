"""
GC Strategy Verification Test (Version 2) - 8306.T

改善点:
1. 明確なデータフロー（backtest戻り値を直接使用）
2. 詳細なログ（各取引のentry_idx/exit_idxを記録）
3. 検証機能強化（シグナル整合性の複数チェック）
4. デバッグモード対応（DEBUG_BACKTEST環境変数）
5. copilot-instructions.md完全準拠

主な機能:
- yfinanceデータ取得（モックデータ禁止）
- GCStrategy実行（strategy.backtest()必須）
- 取引詳細抽出（entry_idx/exit_idx記録）
- シグナル整合性検証（複数の方法）
- 結果CSV出力（trades/cross_events/summary）

統合コンポーネント:
- strategies.gc_strategy_signal: テスト対象のGC戦略
- main_system.data_acquisition.yfinance_data_feed: データ取得
- strategies.base_strategy: バックテスト実行基盤

セーフティ機能/注意事項:
- モックデータ使用禁止（yfinanceから実データ取得）
- strategy.backtest()呼び出し必須
- 実行結果の検証必須（推測での報告禁止）
- エラーは隠蔽せず明示的に報告

Author: Backtest Project Team
Created: 2025-10-24
Last Modified: 2025-10-24
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.gc_strategy_signal import GCStrategy

# ログ設定
log_dir = Path("tests/results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'test_gc_strategy_v2.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class GCStrategyTesterV2:
    """GC戦略テスター Version 2"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, params: dict, test_name: str = "default"):
        """
        初期化
        
        Args:
            ticker: 銘柄コード
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            params: 戦略パラメータ
            test_name: テスト名
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.params = params
        self.test_name = test_name
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.strategy: Optional[GCStrategy] = None
        self.backtest_result: Optional[pd.DataFrame] = None
        
        logger.info("=" * 80)
        logger.info(f"GC Strategy Tester V2 - {test_name}")
        logger.info("=" * 80)
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Parameters: {params}")
        logger.info("")
    
    def step1_fetch_data(self) -> bool:
        """STEP 1: yfinanceからデータ取得"""
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        
        try:
            data_feed = YFinanceDataFeed()
            self.raw_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.raw_data is None or len(self.raw_data) == 0:
                logger.error("[FAIL] データ取得失敗またはデータが空")
                return False
            
            logger.info(f"[SUCCESS] データ取得完了")
            logger.info(f"  データ行数: {len(self.raw_data)}")
            logger.info(f"  期間: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
            logger.info(f"  カラム: {list(self.raw_data.columns)}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] データ取得エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step2_initialize_strategy(self) -> bool:
        """STEP 2: 戦略初期化"""
        logger.info("[STEP 2] 戦略初期化")
        logger.info("-" * 80)
        
        try:
            self.strategy = GCStrategy(
                data=self.raw_data.copy(),  # データのコピーを渡す
                params=self.params,
                price_column="Adj Close"
            )
            
            logger.info(f"[SUCCESS] GCStrategy初期化完了")
            logger.info(f"  使用パラメータ: {self.strategy.params}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] 戦略初期化エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step3_run_backtest(self) -> bool:
        """STEP 3: バックテスト実行（copilot-instructions.md必須）"""
        logger.info("[STEP 3] バックテスト実行")
        logger.info("-" * 80)
        
        try:
            # strategy.backtest()呼び出し（戻り値を直接使用）
            self.backtest_result = self.strategy.backtest()
            
            if self.backtest_result is None:
                logger.error("[FAIL] バックテスト結果がNone")
                return False
            
            # 必須カラムの存在確認
            required_columns = ['Entry_Signal', 'Exit_Signal', 'Position', 'Strategy']
            missing_columns = [col for col in required_columns if col not in self.backtest_result.columns]
            
            if missing_columns:
                logger.error(f"[FAIL] 必須カラムが見つかりません: {missing_columns}")
                return False
            
            logger.info(f"[SUCCESS] バックテスト実行完了")
            logger.info(f"  結果データ行数: {len(self.backtest_result)}")
            logger.info(f"  結果カラム: {list(self.backtest_result.columns)}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] バックテストエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step4_verify_signals(self) -> bool:
        """STEP 4: シグナル整合性検証（複数の方法）"""
        logger.info("[STEP 4] シグナル整合性検証")
        logger.info("-" * 80)
        
        try:
            # 検証1: Entry_Signal/Exit_Signalのカウント
            entry_count = (self.backtest_result['Entry_Signal'] == 1).sum()
            exit_count = (self.backtest_result['Exit_Signal'] == -1).sum()
            
            logger.info(f"[検証1] シグナルカウント")
            logger.info(f"  Entry_Signal == 1: {entry_count} 回")
            logger.info(f"  Exit_Signal == -1: {exit_count} 回")
            
            # 検証2: Positionの遷移カウント
            position_changes = self.backtest_result[self.backtest_result['Position'].diff() != 0]
            logger.info(f"[検証2] Position変化: {len(position_changes)} 回")
            
            # 検証3: DataFrameのインデックス確認
            logger.info(f"[検証3] インデックス整合性")
            logger.info(f"  raw_data行数: {len(self.raw_data)}")
            logger.info(f"  backtest_result行数: {len(self.backtest_result)}")
            logger.info(f"  インデックス型: {type(self.backtest_result.index)}")
            
            # 検証4: Entry/Exitの一致確認
            if entry_count == 0:
                logger.error("[FAIL] エントリー回数が0です")
                return False
            
            if entry_count != exit_count:
                logger.warning(f"[WARNING] エントリー({entry_count}) とエグジット({exit_count}) が不一致")
                # これは警告だが、テストは続行
            else:
                logger.info(f"[SUCCESS] エントリー/エグジット一致: {entry_count} 回")
            
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] シグナル検証エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step5_extract_trades(self) -> List[Dict]:
        """STEP 5: 取引詳細抽出（entry_idx/exit_idx記録）"""
        logger.info("[STEP 5] 取引詳細抽出")
        logger.info("-" * 80)
        
        trades = []
        
        try:
            # Entry_Signal==1の行を取得
            entry_signals = self.backtest_result[self.backtest_result['Entry_Signal'] == 1]
            
            for entry_idx, entry_row in entry_signals.iterrows():
                entry_date = entry_idx
                entry_price = entry_row['Adj Close']
                entry_position_idx = self.backtest_result.index.get_loc(entry_idx)
                
                # このエントリーに対応するExit_Signalを探す
                # エントリー日以降でExit_Signal==-1の最初の日
                exit_signals_after = self.backtest_result.loc[entry_date:][self.backtest_result['Exit_Signal'] == -1]
                
                if len(exit_signals_after) > 0:
                    exit_idx = exit_signals_after.index[0]
                    exit_row = self.backtest_result.loc[exit_idx]
                    exit_date = exit_idx
                    exit_price = exit_row['Adj Close']
                    exit_position_idx = self.backtest_result.index.get_loc(exit_idx)
                    
                    # 損益計算
                    pnl = exit_price - entry_price
                    pnl_pct = (pnl / entry_price) * 100
                    hold_days = (exit_date - entry_date).days
                    
                    trade = {
                        'entry_idx': entry_position_idx,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_idx': exit_position_idx,
                        'exit_date': exit_date,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'hold_days': hold_days
                    }
                    
                    trades.append(trade)
                    
                    logger.info(f"  取引#{len(trades)}: entry_idx={entry_position_idx}, exit_idx={exit_position_idx}, "
                               f"PnL={pnl:.2f} ({pnl_pct:.2f}%), 保有={hold_days}日")
                else:
                    logger.warning(f"  エントリー{entry_date}に対応するイグジットが見つかりません")
            
            logger.info(f"[SUCCESS] {len(trades)} 件の取引を抽出")
            logger.info("")
            
            return trades
            
        except Exception as e:
            logger.error(f"[FAIL] 取引抽出エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def step6_calculate_performance(self, trades: List[Dict]) -> Dict:
        """STEP 6: パフォーマンス計算"""
        logger.info("[STEP 6] パフォーマンス計算")
        logger.info("-" * 80)
        
        if len(trades) == 0:
            logger.warning("[WARNING] 取引データが空です")
            return {}
        
        try:
            # 総損益
            total_pnl = sum(t['pnl'] for t in trades)
            total_pnl_pct = sum(t['pnl_pct'] for t in trades)
            
            # 勝率
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0
            
            # 平均保有期間
            avg_hold_days = sum(t['hold_days'] for t in trades) / len(trades)
            
            performance = {
                'total_trades': len(trades),
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'win_rate': win_rate,
                'winning_trades': len(winning_trades),
                'losing_trades': len(trades) - len(winning_trades),
                'avg_hold_days': avg_hold_days
            }
            
            logger.info(f"[PERFORMANCE]")
            logger.info(f"  総取引回数: {performance['total_trades']} 回")
            logger.info(f"  勝率: {performance['win_rate']:.1f}% ({performance['winning_trades']}勝 {performance['losing_trades']}敗)")
            logger.info(f"  総損益: {performance['total_pnl']:.2f} 円")
            logger.info(f"  総損益率: {performance['total_pnl_pct']:.2f}%")
            logger.info(f"  平均保有期間: {performance['avg_hold_days']:.1f} 日")
            logger.info("")
            
            return performance
            
        except Exception as e:
            logger.error(f"[FAIL] パフォーマンス計算エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def step7_save_results(self, trades: List[Dict], performance: Dict) -> bool:
        """STEP 7: 結果をCSVに保存"""
        logger.info("[STEP 7] 結果保存")
        logger.info("-" * 80)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. バックテスト結果全体を保存
            backtest_file = log_dir / f"backtest_result_{self.test_name}_{timestamp}.csv"
            self.backtest_result.to_csv(backtest_file)
            logger.info(f"  [1] バックテスト結果: {backtest_file}")
            
            # 2. 取引詳細を保存
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                trades_file = log_dir / f"trades_{self.test_name}_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False)
                logger.info(f"  [2] 取引詳細: {trades_file}")
            
            # 3. パフォーマンスサマリーを保存
            if performance:
                summary_file = log_dir / f"summary_{self.test_name}_{timestamp}.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== GC Strategy Test Summary ({self.test_name}) ===\n\n")
                    f.write(f"Ticker: {self.ticker}\n")
                    f.write(f"Period: {self.start_date} to {self.end_date}\n")
                    f.write(f"Parameters: {self.params}\n\n")
                    f.write(f"=== Performance ===\n")
                    f.write(f"Total Trades: {performance['total_trades']}\n")
                    f.write(f"Win Rate: {performance['win_rate']:.1f}%\n")
                    f.write(f"Total PnL: {performance['total_pnl']:.2f}\n")
                    f.write(f"Total PnL %: {performance['total_pnl_pct']:.2f}%\n")
                    f.write(f"Avg Hold Days: {performance['avg_hold_days']:.1f}\n")
                logger.info(f"  [3] サマリー: {summary_file}")
            
            logger.info(f"[SUCCESS] 結果保存完了")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] 結果保存エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_full_test(self) -> bool:
        """フルテスト実行"""
        
        # STEP 1: データ取得
        if not self.step1_fetch_data():
            return False
        
        # STEP 2: 戦略初期化
        if not self.step2_initialize_strategy():
            return False
        
        # STEP 3: バックテスト実行（必須）
        if not self.step3_run_backtest():
            return False
        
        # STEP 4: シグナル検証
        if not self.step4_verify_signals():
            return False
        
        # STEP 5: 取引抽出
        trades = self.step5_extract_trades()
        
        # STEP 6: パフォーマンス計算
        performance = self.step6_calculate_performance(trades)
        
        # STEP 7: 結果保存
        self.step7_save_results(trades, performance)
        
        # 最終判定
        logger.info("[FINAL RESULT]")
        logger.info("=" * 80)
        
        entry_count = (self.backtest_result['Entry_Signal'] == 1).sum()
        exit_count = (self.backtest_result['Exit_Signal'] == -1).sum()
        
        # 成功条件
        conditions = {
            'データ取得成功': self.raw_data is not None,
            'バックテスト実行成功': self.backtest_result is not None,
            'エントリー > 0': entry_count > 0,
            'エントリー == エグジット': entry_count == exit_count,
            '取引抽出成功': len(trades) > 0
        }
        
        for condition, result in conditions.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"  {condition}: {status}")
        
        test_passed = all(conditions.values())
        
        logger.info("")
        if test_passed:
            logger.info("総合結果: TEST PASSED")
        else:
            logger.info("総合結果: TEST FAILED")
        logger.info("=" * 80)
        logger.info("")
        
        return test_passed


def main():
    """メイン関数"""
    
    # テスト設定
    ticker = "8306.T"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # テスト1: トレンドフィルター無効
    logger.info("\n" + "=" * 80)
    logger.info("[TEST 1] トレンドフィルター無効")
    logger.info("=" * 80 + "\n")
    
    params_test1 = {
        "short_window": 5,
        "long_window": 25,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.03,
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True,
        "trend_filter_enabled": False
    }
    
    tester1 = GCStrategyTesterV2(ticker, start_date, end_date, params_test1, "trend_filter_off")
    test1_passed = tester1.run_full_test()
    
    # テスト2: トレンドフィルター有効
    logger.info("\n" + "=" * 80)
    logger.info("[TEST 2] トレンドフィルター有効")
    logger.info("=" * 80 + "\n")
    
    params_test2 = {
        "short_window": 5,
        "long_window": 25,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.03,
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True,
        "trend_filter_enabled": True,
        "allowed_trends": ["uptrend"]
    }
    
    tester2 = GCStrategyTesterV2(ticker, start_date, end_date, params_test2, "trend_filter_on")
    test2_passed = tester2.run_full_test()
    
    # 最終結果
    logger.info("\n" + "=" * 80)
    logger.info("最終結果")
    logger.info("=" * 80)
    logger.info(f"Test 1 (trend_filter=OFF): {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Test 2 (trend_filter=ON):  {'PASSED' if test2_passed else 'FAILED'}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
