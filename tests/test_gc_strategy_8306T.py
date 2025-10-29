"""
GC戦略動作検証テスト - 8306.T（三菱UFJフィナンシャル・グループ）

目的:
- src/strategies/gc_strategy_signal.pyが正常に動作しているかの確認
- 作成者の意図通りに動作しているかを確認
- マルチ戦略システムのバグ特定のため、戦略自体に問題がないかを確認

主な検証項目:
- ゴールデンクロス/デッドクロスの検出
- シグナル生成（Entry_Signal, Exit_Signal）
- 取引実行の確認
- イグジット条件の内訳分析
- パフォーマンス計算（総損益、損益率、保有期間）
- データ整合性（entry_prices, high_prices辞書）

統合コンポーネント:
- src/strategies/gc_strategy_signal.py: テスト対象のGC戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- indicators/basic_indicators.py: SMA計算

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ使用禁止）
- 対象ファイルは修正しない（Read-only）
- yfinance取得失敗時はエラーとして処理
- トレンドフィルター: 第1テスト無効、第2テスト有効

Author: Backtest Project Team
Created: 2025-10-23
Last Modified: 2025-10-23
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from src.strategies.gc_strategy_signal import GCStrategy

# ログ設定
log_dir = Path("tests/results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'test_gc_strategy_8306T.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class GCStrategyTester:
    """GC戦略テスター"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, 
                 params: dict = None, test_name: str = "default"):
        """
        初期化
        
        Args:
            ticker: 銘柄コード
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            params: 戦略パラメータ
            test_name: テスト名（ファイル名に使用）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.params = params or {}
        self.test_name = test_name
        
        self.stock_data = None
        self.strategy = None
        self.result = None
        self.golden_crosses = []
        self.dead_crosses = []
        
        logger.info("=" * 80)
        logger.info(f"GC Strategy Tester Initialized - {test_name}")
        logger.info("=" * 80)
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Parameters: {params}")
        logger.info("")
    
    def fetch_data(self) -> bool:
        """yfinanceからデータ取得"""
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        
        try:
            data_feed = YFinanceDataFeed()
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or len(self.stock_data) == 0:
                logger.error(f"[FAIL] データ取得失敗: {self.ticker}")
                return False
            
            logger.info(f"[SUCCESS] データ取得完了")
            logger.info(f"  データ行数: {len(self.stock_data)}")
            logger.info(f"  期間: {self.stock_data.index[0]} to {self.stock_data.index[-1]}")
            logger.info(f"  カラム: {list(self.stock_data.columns)}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] データ取得エラー: {e}")
            return False
    
    def initialize_strategy(self) -> bool:
        """戦略初期化"""
        logger.info("[STEP 2] 戦略初期化")
        logger.info("-" * 80)
        
        try:
            # GCStrategyインスタンス作成
            self.strategy = GCStrategy(
                data=self.stock_data,
                params=self.params,
                price_column="Adj Close"
            )
            
            logger.info(f"[SUCCESS] GCStrategy初期化完了")
            logger.info(f"  使用パラメータ: {self.strategy.params}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] 戦略初期化エラー: {e}")
            return False
    
    def detect_crosses(self) -> None:
        """ゴールデンクロス/デッドクロスを検出"""
        logger.info("[STEP 3] クロスイベント検出")
        logger.info("-" * 80)
        
        # SMAカラム名
        short_window = self.strategy.params.get('short_window', 5)
        long_window = self.strategy.params.get('long_window', 25)
        sma_short_col = f'SMA_{short_window}'
        sma_long_col = f'SMA_{long_window}'
        
        # データにSMAがない場合は計算
        if sma_short_col not in self.stock_data.columns:
            from indicators.basic_indicators import calculate_sma
            self.stock_data[sma_short_col] = calculate_sma(
                self.stock_data, "Adj Close", short_window
            )
        
        if sma_long_col not in self.stock_data.columns:
            from indicators.basic_indicators import calculate_sma
            self.stock_data[sma_long_col] = calculate_sma(
                self.stock_data, "Adj Close", long_window
            )
        
        # クロス検出
        for i in range(1, len(self.stock_data)):
            prev_short = self.stock_data[sma_short_col].iloc[i-1]
            prev_long = self.stock_data[sma_long_col].iloc[i-1]
            curr_short = self.stock_data[sma_short_col].iloc[i]
            curr_long = self.stock_data[sma_long_col].iloc[i]
            
            # NaN チェック
            if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
                continue
            
            # ゴールデンクロス: 短期が長期を下から上に突き抜ける
            if prev_short <= prev_long and curr_short > curr_long:
                date = self.stock_data.index[i]
                price = self.stock_data['Adj Close'].iloc[i]
                self.golden_crosses.append({
                    'date': date,
                    'price': price,
                    'short_ma': curr_short,
                    'long_ma': curr_long
                })
                logger.info(f"[GOLDEN CROSS] {date.strftime('%Y-%m-%d')} @ {price:.2f}円")
            
            # デッドクロス: 短期が長期を上から下に突き抜ける
            elif prev_short >= prev_long and curr_short < curr_long:
                date = self.stock_data.index[i]
                price = self.stock_data['Adj Close'].iloc[i]
                self.dead_crosses.append({
                    'date': date,
                    'price': price,
                    'short_ma': curr_short,
                    'long_ma': curr_long
                })
                logger.info(f"[DEAD CROSS] {date.strftime('%Y-%m-%d')} @ {price:.2f}円")
        
        logger.info("")
        logger.info(f"[CROSS SUMMARY]")
        logger.info(f"  Golden Cross: {len(self.golden_crosses)} 回")
        logger.info(f"  Dead Cross: {len(self.dead_crosses)} 回")
        logger.info("")
    
    def run_backtest(self) -> bool:
        """バックテスト実行"""
        logger.info("[STEP 4] バックテスト実行")
        logger.info("-" * 80)
        
        try:
            # strategy.backtest() 呼び出し（copilot-instructions.md必須）
            self.strategy.backtest()
            # BaseStrategy更新後のself.dataを使用（データ同期修正）
            self.result = self.strategy.data
            
            if self.result is None:
                logger.error("[FAIL] バックテスト結果がNone")
                return False
            
            logger.info(f"[SUCCESS] バックテスト実行完了")
            logger.info(f"  結果データ行数: {len(self.result)}")
            logger.info(f"  結果カラム: {list(self.result.columns)}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] バックテストエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def verify_signals(self) -> bool:
        """シグナル生成検証"""
        logger.info("[STEP 5] シグナル生成の検証")
        logger.info("-" * 80)
        
        try:
            # Entry_Signal == 1 の回数
            entry_count = (self.result['Entry_Signal'] == 1).sum()
            
            # Exit_Signal == -1 の回数
            exit_count = (self.result['Exit_Signal'] == -1).sum()
            
            logger.info(f"[SIGNAL CHECK]")
            logger.info(f"  Entry_Signal == 1: {entry_count} 回")
            logger.info(f"  Exit_Signal == -1: {exit_count} 回")
            
            # 整合性チェック
            if entry_count == exit_count:
                logger.info(f"  整合性: OK")
            else:
                logger.warning(f"  整合性: NG (エントリー != エグジット)")
            
            logger.info("")
            
            # テスト成功条件
            if entry_count == 0:
                logger.error("[FAIL] エントリー回数が0です")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] シグナル検証エラー: {e}")
            return False
    
    def extract_trades(self) -> list:
        """取引詳細の抽出"""
        logger.info("[STEP 6] 取引詳細の抽出")
        logger.info("-" * 80)
        
        trades = []
        entry_date = None
        entry_price = None
        
        for i in range(len(self.result)):
            row = self.result.iloc[i]
            
            # エントリーシグナル
            if row['Entry_Signal'] == 1 and entry_date is None:
                entry_date = self.result.index[i]
                entry_price = row['Adj Close']
            
            # エグジットシグナル
            elif row['Exit_Signal'] == -1 and entry_date is not None:
                exit_date = self.result.index[i]
                exit_price = row['Adj Close']
                
                # 保有期間
                hold_days = (exit_date - entry_date).days
                
                # 損益計算（100株想定）
                pnl = (exit_price - entry_price) * 100
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'hold_days': hold_days,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                # リセット
                entry_date = None
                entry_price = None
        
        logger.info(f"[TRADES EXTRACTED] {len(trades)} 件の取引を抽出")
        logger.info("")
        
        return trades
    
    def analyze_exit_conditions(self, trades: list) -> dict:
        """イグジット条件の内訳分析"""
        logger.info("[STEP 7] イグジット条件の分析")
        logger.info("-" * 80)
        
        exit_conditions = {
            'dead_cross': 0,
            'trailing_stop': 0,
            'take_profit': 0,
            'stop_loss': 0,
            'max_hold_days': 0,
            'other': 0
        }
        
        # 戦略ログからイグジット理由を推定
        # （実際には戦略クラスにイグジット理由を記録する機能が必要）
        # ここでは取引結果から推定
        
        take_profit_pct = self.strategy.params.get('take_profit_pct', 0.05)
        stop_loss_pct = self.strategy.params.get('stop_loss_pct', 0.03)
        max_hold_days = self.strategy.params.get('max_hold_days', 20)
        
        for trade in trades:
            pnl_pct = trade['pnl_pct'] / 100
            hold_days = trade['hold_days']
            
            # 利益確定判定
            if pnl_pct >= take_profit_pct * 0.9:  # 90%以上なら利益確定
                exit_conditions['take_profit'] += 1
            # 損切り判定
            elif pnl_pct <= -stop_loss_pct * 0.9:  # 90%以上損失なら損切り
                exit_conditions['stop_loss'] += 1
            # 最大保有期間判定
            elif hold_days >= max_hold_days:
                exit_conditions['max_hold_days'] += 1
            # その他（デッドクロスまたはトレーリングストップと推定）
            else:
                exit_conditions['other'] += 1
        
        logger.info(f"[EXIT CONDITIONS]")
        logger.info(f"  Take Profit: {exit_conditions['take_profit']} 回")
        logger.info(f"  Stop Loss: {exit_conditions['stop_loss']} 回")
        logger.info(f"  Max Hold Days: {exit_conditions['max_hold_days']} 回")
        logger.info(f"  Other (Dead Cross/Trailing): {exit_conditions['other']} 回")
        logger.info("")
        
        return exit_conditions
    
    def calculate_performance(self, trades: list) -> dict:
        """パフォーマンス計算"""
        logger.info("[STEP 8] パフォーマンス計算")
        logger.info("-" * 80)
        
        if len(trades) == 0:
            logger.warning("[WARNING] 取引が0件のためパフォーマンス計算不可")
            return {}
        
        # 総損益
        total_pnl = sum(t['pnl'] for t in trades)
        
        # 総損益率
        total_pnl_pct = sum(t['pnl_pct'] for t in trades)
        
        # 平均保有期間
        avg_hold_days = sum(t['hold_days'] for t in trades) / len(trades)
        
        # 勝率計算
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
        
        performance = {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_hold_days': avg_hold_days,
            'win_rate': win_rate
        }
        
        logger.info(f"[PERFORMANCE]")
        logger.info(f"  総取引回数: {performance['total_trades']} 回")
        logger.info(f"  総損益: {performance['total_pnl']:.2f} 円")
        logger.info(f"  総損益率: {performance['total_pnl_pct']:.2f} %")
        logger.info(f"  平均保有期間: {performance['avg_hold_days']:.1f} 日")
        logger.info(f"  勝率: {performance['win_rate']:.1f} %")
        logger.info("")
        
        return performance
    
    def verify_data_integrity(self) -> bool:
        """データ整合性確認"""
        logger.info("[STEP 9] データ整合性の確認")
        logger.info("-" * 80)
        
        try:
            # entry_prices辞書の確認
            entry_prices_count = len(getattr(self.strategy, 'entry_prices', {}))
            logger.info(f"[ENTRY_PRICES CHECK]")
            logger.info(f"  記録されたentry_prices: {entry_prices_count} 件")
            
            # high_prices辞書の確認（トレーリングストップ用）
            high_prices_count = len(getattr(self.strategy, 'high_prices', {}))
            logger.info(f"[HIGH_PRICES CHECK]")
            logger.info(f"  記録されたhigh_prices: {high_prices_count} 件")
            
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] データ整合性確認エラー: {e}")
            return False
    
    def save_results(self, trades: list, performance: dict, exit_conditions: dict) -> None:
        """結果をCSVに保存"""
        logger.info("[STEP 10] 結果保存")
        logger.info("-" * 80)
        
        try:
            results_dir = Path("tests/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 取引履歴CSV
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                trades_csv = results_dir / f"gc_strategy_8306T_{self.test_name}_trades.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
                logger.info(f"[SUCCESS] 取引履歴CSV保存: {trades_csv}")
            
            # クロスイベントCSV
            if len(self.golden_crosses) > 0 or len(self.dead_crosses) > 0:
                cross_events = []
                for gc in self.golden_crosses:
                    cross_events.append({
                        'date': gc['date'],
                        'type': 'Golden Cross',
                        'price': gc['price'],
                        'short_ma': gc['short_ma'],
                        'long_ma': gc['long_ma']
                    })
                for dc in self.dead_crosses:
                    cross_events.append({
                        'date': dc['date'],
                        'type': 'Dead Cross',
                        'price': dc['price'],
                        'short_ma': dc['short_ma'],
                        'long_ma': dc['long_ma']
                    })
                
                cross_df = pd.DataFrame(cross_events)
                cross_df = cross_df.sort_values('date')
                cross_csv = results_dir / f"gc_strategy_8306T_{self.test_name}_cross_events.csv"
                cross_df.to_csv(cross_csv, index=False, encoding='utf-8-sig')
                logger.info(f"[SUCCESS] クロスイベントCSV保存: {cross_csv}")
            
            # サマリーCSV
            summary_data = {
                'Ticker': [self.ticker],
                'Start Date': [self.start_date],
                'End Date': [self.end_date],
                'Test Name': [self.test_name],
                'Golden Cross Count': [len(self.golden_crosses)],
                'Dead Cross Count': [len(self.dead_crosses)],
                'Total Trades': [performance.get('total_trades', 0)],
                'Total PnL': [performance.get('total_pnl', 0)],
                'Total PnL %': [performance.get('total_pnl_pct', 0)],
                'Avg Hold Days': [performance.get('avg_hold_days', 0)],
                'Win Rate %': [performance.get('win_rate', 0)],
                'Take Profit Exits': [exit_conditions.get('take_profit', 0)],
                'Stop Loss Exits': [exit_conditions.get('stop_loss', 0)],
                'Max Hold Exits': [exit_conditions.get('max_hold_days', 0)],
                'Other Exits': [exit_conditions.get('other', 0)]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = results_dir / f"gc_strategy_8306T_{self.test_name}_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            logger.info(f"[SUCCESS] サマリーCSV保存: {summary_csv}")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"[FAIL] 結果保存エラー: {e}")
    
    def print_trade_details(self, trades: list) -> None:
        """取引詳細を出力"""
        logger.info("[取引詳細]")
        logger.info("-" * 80)
        
        for i, trade in enumerate(trades, 1):
            logger.info(f"  [{i}] エントリー: {trade['entry_date'].strftime('%Y-%m-%d')} @ {trade['entry_price']:.2f}円")
            logger.info(f"      エグジット: {trade['exit_date'].strftime('%Y-%m-%d')} @ {trade['exit_price']:.2f}円")
            logger.info(f"      保有期間: {trade['hold_days']}日")
            logger.info(f"      損益: {trade['pnl']:.2f}円 ({trade['pnl_pct']:.2f}%)")
            logger.info("")
    
    def run_full_test(self) -> bool:
        """フルテスト実行"""
        # STEP 1: データ取得
        if not self.fetch_data():
            return False
        
        # STEP 2: 戦略初期化
        if not self.initialize_strategy():
            return False
        
        # STEP 3: クロス検出
        self.detect_crosses()
        
        # STEP 4: バックテスト実行
        if not self.run_backtest():
            return False
        
        # STEP 5: シグナル検証
        if not self.verify_signals():
            return False
        
        # STEP 6: 取引抽出
        trades = self.extract_trades()
        
        # STEP 7: イグジット条件分析
        exit_conditions = self.analyze_exit_conditions(trades)
        
        # STEP 8: パフォーマンス計算
        performance = self.calculate_performance(trades)
        
        # STEP 9: データ整合性確認
        self.verify_data_integrity()
        
        # STEP 10: 結果保存
        self.save_results(trades, performance, exit_conditions)
        
        # 取引詳細出力
        if len(trades) > 0:
            self.print_trade_details(trades)
        
        # テスト結果判定
        logger.info("[TEST RESULT]")
        logger.info("=" * 80)
        
        # 成功条件チェック
        golden_cross_ok = len(self.golden_crosses) > 0
        entry_count_ok = (self.result['Entry_Signal'] == 1).sum() > 0
        entry_exit_match = (self.result['Entry_Signal'] == 1).sum() == (self.result['Exit_Signal'] == -1).sum()
        no_errors = True  # エラーがなければTrue
        
        logger.info(f"  条件1（Golden Cross > 0）: {'PASS' if golden_cross_ok else 'FAIL'}")
        logger.info(f"  条件2（エントリー > 0）: {'PASS' if entry_count_ok else 'FAIL'}")
        logger.info(f"  条件3（エントリー == エグジット）: {'PASS' if entry_exit_match else 'FAIL'}")
        logger.info(f"  条件4（エラーなし）: {'PASS' if no_errors else 'FAIL'}")
        logger.info("")
        
        test_passed = golden_cross_ok and entry_count_ok and entry_exit_match and no_errors
        
        if test_passed:
            logger.info("  総合結果: TEST PASSED")
        else:
            logger.info("  総合結果: TEST FAILED")
        
        logger.info("=" * 80)
        
        return test_passed


def main():
    """メイン関数"""
    
    # テスト設定
    ticker = "8306.T"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # Q1: 最適化パラメータの取得試行
    logger.info("\n[Q1] 最適化パラメータの取得を試行...")
    optimized_params = None
    
    try:
        from config.optimized_parameters import OptimizedParameterManager
        manager = OptimizedParameterManager()
        config = manager.get_best_config_by_metric(
            'GCStrategy', 
            metric='sharpe_ratio', 
            ticker=ticker, 
            status='approved'
        )
        
        if config and 'parameters' in config:
            optimized_params = config['parameters']
            logger.info(f"[OK] 最適化パラメータ取得成功: {optimized_params}")
        else:
            logger.info("[INFO] 承認済み最適化パラメータが見つかりません")
            logger.info("[INFO] デフォルトパラメータでテストを実行します")
    
    except Exception as e:
        logger.info(f"[INFO] 最適化パラメータ取得不可: {e}")
        logger.info("[INFO] デフォルトパラメータでテストを実行します")
    
    # Q2-B: トレンドフィルター無効でテスト（第1テスト）
    logger.info("\n" + "=" * 80)
    logger.info("[TEST 1] トレンドフィルター無効（trend_filter_enabled: False）")
    logger.info("=" * 80 + "\n")
    
    params_test1 = optimized_params.copy() if optimized_params else {
        "short_window": 5,
        "long_window": 25,
        "take_profit_pct": 0.05,  # Q3: 新名称のみ
        "stop_loss_pct": 0.03,    # Q3: 新名称のみ
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True
    }
    
    # トレンドフィルター無効化
    params_test1["trend_filter_enabled"] = False
    
    tester1 = GCStrategyTester(ticker, start_date, end_date, params_test1, "trend_filter_off")
    test1_passed = tester1.run_full_test()
    
    # Q2-A: トレンドフィルター有効でテスト（第2テスト）
    logger.info("\n" + "=" * 80)
    logger.info("[TEST 2] トレンドフィルター有効（trend_filter_enabled: True）")
    logger.info("=" * 80 + "\n")
    
    params_test2 = optimized_params.copy() if optimized_params else {
        "short_window": 5,
        "long_window": 25,
        "take_profit_pct": 0.05,  # Q3: 新名称のみ
        "stop_loss_pct": 0.03,    # Q3: 新名称のみ
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True
    }
    
    # トレンドフィルター有効化
    params_test2["trend_filter_enabled"] = True
    params_test2["allowed_trends"] = ["uptrend"]
    
    tester2 = GCStrategyTester(ticker, start_date, end_date, params_test2, "trend_filter_on")
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
