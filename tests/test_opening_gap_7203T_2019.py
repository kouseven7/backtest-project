"""
Opening_Gap.py 動作検証テスト - 7203.T（2019年データ）
過学習検証: 最適化に使われていないデータでテスト

目的:
1. 最適化前のデータ(2019年)で性能検証
2. 過学習の有無を確認
3. 最適化データ(2020-2024)との性能比較
4. Out-of-sample性能の検証

主な機能:
- yfinanceを使用した実データ取得（7203.T 2019年 + ^DJI）
- Opening_Gap戦略のバックテスト実行（最適化パラメータ使用）
- 2019年データでの性能測定
- 過学習検出のための性能比較
- 詳細なトレード分析

統合コンポーネント:
- strategies/Opening_Gap.py: テスト対象戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- config/optimized_params/opening_gap_7203.T_2025-06-19.json: 最適化パラメータ
- config/logger_config.py: ロガー設定

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ使用禁止）
- strategies/Opening_Gap.pyは修正しない
- 2019年データ（最適化前）を使用
- 最適化パラメータをそのまま適用
- 過学習の兆候を検出

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap import OpeningGapStrategy


class OpeningGap7203Tester2019:
    """
    Opening_Gap戦略テストクラス（7203.T 2019年データ）
    
    最適化に使われていないデータで過学習を検証
    """
    
    def __init__(self, ticker: str = "7203.T", start_date: str = "2019-01-01", end_date: str = "2019-12-31"):
        """
        初期化
        
        Args:
            ticker: テスト対象銘柄（デフォルト: 7203.T）
            start_date: テスト期間開始日（2019-01-01）
            end_date: テスト期間終了日（2019-12-31）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # ロガー設定
        self.logger = setup_logger(
            "OpeningGap7203Tester2019",
            log_file="logs/test_opening_gap_7203T_2019.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"Opening_Gap.py 過学習検証テスト開始（2019年データ）")
        self.logger.info(f"銘柄: {ticker}, 期間: {start_date} ~ {end_date}")
        self.logger.info(f"注意: 最適化データ期間(2020-2024)外のデータを使用")
        self.logger.info("=" * 80)
        
        # データフィード初期化
        self.data_feed = YFinanceDataFeed()
        
        # データ格納
        self.stock_data: Optional[pd.DataFrame] = None
        self.dow_data: Optional[pd.DataFrame] = None
        self.strategy: Optional[OpeningGapStrategy] = None
        self.backtest_result: Optional[pd.DataFrame] = None
        
        # 最適化パラメータ
        self.optimized_params: Optional[Dict[str, Any]] = None
        
        # 検証結果
        self.validation_results: Dict[str, Any] = {}
    
    def run_full_test(self) -> Dict[str, Any]:
        """
        フルテスト実行
        
        Returns:
            テスト結果サマリー
        """
        self.logger.info("\n[PHASE 1] データ取得開始（2019年）")
        self.logger.info("-" * 80)
        
        try:
            # Step 1: 最適化パラメータ読み込み
            self._load_optimized_params()
            
            # Step 2: データ取得（2019年）
            self._fetch_data()
            
            # Step 3: 戦略初期化
            self._initialize_strategy()
            
            # Step 4: バックテスト実行
            self._run_backtest()
            
            # Step 5: 検証実行
            self.logger.info("\n[PHASE 2] 検証ロジック実行開始")
            self.logger.info("-" * 80)
            self._validate_all()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[SUCCESS] Phase 1-2: データ取得・バックテスト・検証完了（2019年）")
            self.logger.info("=" * 80)
            
            return {
                'status': 'SUCCESS_PHASE2',
                'ticker': self.ticker,
                'test_period': '2019',
                'stock_data_rows': len(self.stock_data),
                'dow_data_rows': len(self.dow_data),
                'backtest_result_rows': len(self.backtest_result) if self.backtest_result is not None else 0,
                'validation_results': self.validation_results,
                'optimized_params': self.optimized_params
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] テスト実行エラー: {e}", exc_info=True)
            return {
                'status': 'FAILED',
                'ticker': self.ticker,
                'test_period': '2019',
                'error': str(e)
            }
    
    def _load_optimized_params(self) -> None:
        """最適化パラメータの読み込み"""
        self.logger.info("\n[OPTIMIZED_PARAMS] 最適化パラメータ読み込み")
        self.logger.info("-" * 80)
        
        param_file = project_root / "config" / "optimized_params" / "opening_gap_7203.T_2025-06-19.json"
        
        if not param_file.exists():
            raise FileNotFoundError(f"最適化パラメータファイルが見つかりません: {param_file}")
        
        with open(param_file, 'r', encoding='utf-8') as f:
            param_data = json.load(f)
        
        self.optimized_params = param_data['parameters'].copy()
        
        self.logger.info(f"[SUCCESS] 最適化パラメータ読み込み完了")
        self.logger.info(f"  ファイル: {param_file.name}")
        self.logger.info(f"  最適化期間: {param_data['optimization_details']['data_period']['start_date']} ~ {param_data['optimization_details']['data_period']['end_date']}")
        self.logger.info(f"  テスト期間: {self.start_date} ~ {self.end_date}")
        self.logger.info(f"  [CRITICAL] 最適化期間外のデータを使用（過学習検証）")
    
    def _fetch_data(self) -> None:
        """データ取得（2019年）"""
        self.logger.info(f"\n[DATA_FETCH] 株価データ取得: {self.ticker} (2019年)")
        
        try:
            self.stock_data = self.data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or len(self.stock_data) == 0:
                raise RuntimeError(
                    f"Failed to retrieve stock data for {self.ticker} (2019). "
                    "Mock/dummy data fallback is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] 株価データ取得完了: {len(self.stock_data)} rows")
            self.logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 株価データ取得失敗: {e}")
            raise
        
        self.logger.info(f"\n[DATA_FETCH] DOWデータ取得: ^DJI (2019年)")
        
        try:
            self.dow_data = self.data_feed.get_stock_data(
                ticker="^DJI",
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.dow_data is None or len(self.dow_data) == 0:
                raise RuntimeError(
                    "Failed to retrieve DOW data (^DJI) (2019). "
                    "Mock/dummy data fallback is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] DOWデータ取得完了: {len(self.dow_data)} rows")
            
        except Exception as e:
            self.logger.error(f"[ERROR] DOWデータ取得失敗: {e}")
            raise
    
    def _initialize_strategy(self) -> None:
        """OpeningGapStrategy初期化"""
        self.logger.info("\n[STRATEGY_INIT] OpeningGapStrategy初期化")
        self.logger.info("-" * 80)
        
        try:
            strategy_params = {k: v for k, v in self.optimized_params.items() if k != 'score'}
            
            self.strategy = OpeningGapStrategy(
                data=self.stock_data,
                dow_data=self.dow_data,
                params=strategy_params,
                price_column="Adj Close"
            )
            
            self.logger.info("[SUCCESS] OpeningGapStrategy初期化完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] OpeningGapStrategy初期化失敗: {e}")
            raise
    
    def _run_backtest(self) -> None:
        """バックテスト実行"""
        self.logger.info("\n[BACKTEST] バックテスト実行開始（2019年）")
        self.logger.info("-" * 80)
        
        try:
            self.backtest_result = self.strategy.backtest()
            
            if self.backtest_result is None:
                raise RuntimeError(
                    "strategy.backtest() returned None. "
                    "Skipping backtest is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] バックテスト実行完了: {len(self.backtest_result)} rows")
            
            # 互換性レイヤー
            if 'Position_Size' in self.backtest_result.columns:
                self.backtest_result['Position'] = (
                    self.backtest_result['Position_Size'] > 0
                ).astype(int)
            
            # シグナル集計
            entry_count = (self.backtest_result['Entry_Signal'] == 1).sum()
            exit_count = (self.backtest_result['Exit_Signal'] == -1).sum()
            
            self.logger.info(f"\n[BACKTEST_SUMMARY]")
            self.logger.info(f"  Entry_Signal == 1: {entry_count} 回")
            self.logger.info(f"  Exit_Signal == -1: {exit_count} 回")
            
        except Exception as e:
            self.logger.error(f"[ERROR] バックテスト実行失敗: {e}")
            raise
    
    def _validate_all(self) -> None:
        """Phase 2: 検証ロジック実行"""
        try:
            self._validate_signal_generation()
            self._validate_trade_execution()
            self._validate_performance_calculation()
            
            self.logger.info("\n[SUCCESS] Phase 2: 全検証カテゴリ完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 検証実行エラー: {e}")
            raise
    
    def _validate_signal_generation(self) -> None:
        """検証1: シグナル生成検証"""
        self.logger.info("\n[VALIDATION 1] シグナル生成検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        entry_count = (df['Entry_Signal'] == 1).sum()
        exit_count = (df['Exit_Signal'] == -1).sum()
        
        self.logger.info(f"[SIGNAL_COUNT] Entry: {entry_count}, Exit: {exit_count}")
        
        self.validation_results['signal_generation'] = {
            'entry_count': int(entry_count),
            'exit_count': int(exit_count)
        }
    
    def _validate_trade_execution(self) -> None:
        """検証2: トレード実行検証"""
        self.logger.info("\n[VALIDATION 2] トレード実行検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        trades = []
        entry_date = None
        entry_price = None
        
        for idx, row in df.iterrows():
            if row['Entry_Signal'] == 1:
                entry_date = idx
                entry_price = row['Open'] if 'Open' in df.columns else row['Adj Close']
            elif row['Exit_Signal'] == -1 and entry_date is not None:
                exit_price = row['Adj Close']
                hold_days = (idx - entry_date).days
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0.0
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'pnl_pct': float(pnl_pct),
                    'hold_days': hold_days
                })
                
                entry_date = None
                entry_price = None
        
        self.logger.info(f"[TRADE_COUNT] 総取引数: {len(trades)} 件")
        
        if trades:
            self.logger.info("\n[TRADE_EXAMPLES] 最初の5件のトレード:")
            for i, trade in enumerate(trades[:5], 1):
                self.logger.info(
                    f"  #{i}: {trade['entry_date'].strftime('%Y-%m-%d')} → "
                    f"{trade['exit_date'].strftime('%Y-%m-%d')} "
                    f"P&L: {trade['pnl_pct']:+.2f}%"
                )
        
        self.validation_results['trade_execution'] = {
            'total_trades': len(trades),
            'trades': trades
        }
    
    def _validate_performance_calculation(self) -> None:
        """検証3: パフォーマンス計算"""
        self.logger.info("\n[VALIDATION 3] パフォーマンス計算（2019年データ）")
        self.logger.info("-" * 80)
        
        trades = self.validation_results.get('trade_execution', {}).get('trades', [])
        
        if not trades:
            self.logger.warning("[WARNING] トレードが0件")
            self.validation_results['performance'] = {}
            return
        
        pnl_list = [t['pnl_pct'] for t in trades]
        total_pnl = sum(pnl_list)
        
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / len(trades)) * 100
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        self.logger.info(f"[P&L] 総P&L: {total_pnl:.2f}%")
        self.logger.info(f"[WIN_RATE] 勝率: {win_rate:.1f}% ({win_count}勝 / {loss_count}敗)")
        self.logger.info(f"[AVG] 平均利益: {avg_win:.2f}%, 平均損失: {avg_loss:.2f}%")
        
        # 過学習判定
        self.logger.info("\n[OVERFITTING_CHECK] 過学習判定:")
        self.logger.info(f"  最適化時の勝率: 66.67%")
        self.logger.info(f"  2019年の勝率: {win_rate:.1f}%")
        
        if win_rate < 50:
            self.logger.warning(f"  [WARNING] 勝率が50%未満: 過学習の可能性あり")
        elif win_rate < 60:
            self.logger.info(f"  [INFO] 勝率低下: 最適化期間との差異あり")
        else:
            self.logger.info(f"  [OK] 勝率維持: 過学習の兆候なし")
        
        self.validation_results['performance'] = {
            'total_pnl_pct': float(total_pnl),
            'win_rate_pct': float(win_rate),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'avg_win_pct': float(avg_win),
            'avg_loss_pct': float(avg_loss),
            'overfitting_suspected': win_rate < 50
        }


def main():
    """メインエントリーポイント"""
    print("\n" + "=" * 80)
    print("Opening_Gap.py 過学習検証テスト - 7203.T（2019年データ）")
    print("=" * 80 + "\n")
    
    tester = OpeningGap7203Tester2019(
        ticker="7203.T",
        start_date="2019-01-01",
        end_date="2019-12-31"
    )
    
    result = tester.run_full_test()
    
    print("\n" + "=" * 80)
    print("テスト結果（2019年データ - 過学習検証）")
    print("=" * 80)
    print(f"ステータス: {result['status']}")
    
    if result['status'] == 'SUCCESS_PHASE2':
        val_results = result.get('validation_results', {})
        
        sig_gen = val_results.get('signal_generation', {})
        print(f"\n[シグナル生成]")
        print(f"  Entry: {sig_gen.get('entry_count', 0)}回")
        print(f"  Exit: {sig_gen.get('exit_count', 0)}回")
        
        trade_exec = val_results.get('trade_execution', {})
        print(f"\n[トレード実行]")
        print(f"  総取引数: {trade_exec.get('total_trades', 0)}件")
        
        perf = val_results.get('performance', {})
        print(f"\n[パフォーマンス - 2019年]")
        print(f"  総P&L: {perf.get('total_pnl_pct', 0):.2f}%")
        print(f"  勝率: {perf.get('win_rate_pct', 0):.1f}%")
        print(f"  勝ち: {perf.get('win_count', 0)}件")
        print(f"  負け: {perf.get('loss_count', 0)}件")
        
        print(f"\n[過学習判定]")
        print(f"  最適化時の勝率: 66.67%")
        print(f"  2019年の勝率: {perf.get('win_rate_pct', 0):.1f}%")
        
        if perf.get('overfitting_suspected', False):
            print(f"  判定: 過学習の可能性あり")
        else:
            print(f"  判定: 要確認")
        
        print(f"\n[比較参考]")
        print(f"  2024年データ(trend_filter=True): 41件, 0%勝率")
        print(f"  2024年データ(trend_filter=False): 108件, 0%勝率")
    
    print("\n" + "=" * 80 + "\n")
    
    return result


if __name__ == "__main__":
    main()
