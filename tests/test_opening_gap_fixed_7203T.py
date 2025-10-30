"""
Opening_Gap_Fixed.py 動作検証テスト - 7203.T
同日Entry/Exit問題の修正版が正常に動作するか確認

目的:
1. Opening_Gap_Fixed.pyが正常に動作するか確認
2. 同日Entry/Exit問題が解決されているか検証
3. オリジナル版との結果比較
4. ポジション管理が正しく機能しているか確認

主な機能:
- yfinanceを使用した実データ取得（7203.T + ^DJI）
- Opening_Gap_Fixed戦略のバックテスト実行
- ポジション管理の検証
- 同日Entry/Exit問題の検出
- オリジナル版との比較

統合コンポーネント:
- strategies/Opening_Gap_Fixed.py: テスト対象戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- config/logger_config.py: ロガー設定

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ使用禁止）
- strategies/Opening_Gap_Fixed.pyは修正しない
- 実際のバックテスト実行を必須とする
- 同日Entry/Exit問題を厳密にチェック

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy


class OpeningGapFixedTester:
    """
    Opening_Gap_Fixed戦略テストクラス
    
    同日Entry/Exit問題の修正版をテスト
    """
    
    def __init__(self, ticker: str = "7203.T", start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """初期化"""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # ロガー設定
        self.logger = setup_logger(
            "OpeningGapFixedTester",
            log_file="logs/test_opening_gap_fixed_7203T.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"Opening_Gap_Fixed.py 動作検証テスト開始")
        self.logger.info(f"銘柄: {ticker}, 期間: {start_date} ~ {end_date}")
        self.logger.info("=" * 80)
        
        # データフィード初期化
        self.data_feed = YFinanceDataFeed()
        
        # データ格納
        self.stock_data: Optional[pd.DataFrame] = None
        self.dow_data: Optional[pd.DataFrame] = None
        self.strategy: Optional[OpeningGapFixedStrategy] = None
        self.backtest_result: Optional[pd.DataFrame] = None
        
        # 最適化パラメータ
        self.optimized_params: Optional[Dict[str, Any]] = None
        
        # 検証結果
        self.validation_results: Dict[str, Any] = {}
    
    def run_full_test(self) -> Dict[str, Any]:
        """フルテスト実行"""
        self.logger.info("\n[PHASE 1] データ取得開始")
        self.logger.info("-" * 80)
        
        try:
            # Step 1: 最適化パラメータ読み込み
            self._load_optimized_params()
            
            # Step 2: データ取得
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
            self.logger.info("[SUCCESS] Phase 1-2: データ取得・バックテスト・検証完了")
            self.logger.info("=" * 80)
            
            return {
                'status': 'SUCCESS_PHASE2',
                'ticker': self.ticker,
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
    
    def _fetch_data(self) -> None:
        """データ取得"""
        self.logger.info(f"\n[DATA_FETCH] 株価データ取得: {self.ticker}")
        
        try:
            self.stock_data = self.data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or len(self.stock_data) == 0:
                raise RuntimeError(
                    f"Failed to retrieve stock data for {self.ticker}. "
                    "Mock/dummy data fallback is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] 株価データ取得完了: {len(self.stock_data)} rows")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 株価データ取得失敗: {e}")
            raise
        
        self.logger.info(f"\n[DATA_FETCH] DOWデータ取得: ^DJI")
        
        try:
            self.dow_data = self.data_feed.get_stock_data(
                ticker="^DJI",
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.dow_data is None or len(self.dow_data) == 0:
                raise RuntimeError(
                    "Failed to retrieve DOW data (^DJI). "
                    "Mock/dummy data fallback is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] DOWデータ取得完了: {len(self.dow_data)} rows")
            
        except Exception as e:
            self.logger.error(f"[ERROR] DOWデータ取得失敗: {e}")
            raise
    
    def _initialize_strategy(self) -> None:
        """OpeningGapFixedStrategy初期化"""
        self.logger.info("\n[STRATEGY_INIT] OpeningGapFixedStrategy初期化")
        self.logger.info("-" * 80)
        
        try:
            strategy_params = {k: v for k, v in self.optimized_params.items() if k != 'score'}
            
            self.strategy = OpeningGapFixedStrategy(
                data=self.stock_data,
                dow_data=self.dow_data,
                params=strategy_params,
                price_column="Adj Close"
            )
            
            self.logger.info("[SUCCESS] OpeningGapFixedStrategy初期化完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] OpeningGapFixedStrategy初期化失敗: {e}")
            raise
    
    def _run_backtest(self) -> None:
        """バックテスト実行"""
        self.logger.info("\n[BACKTEST] バックテスト実行開始")
        self.logger.info("-" * 80)
        
        try:
            self.backtest_result = self.strategy.backtest()
            
            if self.backtest_result is None:
                raise RuntimeError(
                    "strategy.backtest() returned None. "
                    "Skipping backtest is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] バックテスト実行完了: {len(self.backtest_result)} rows")
            
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
            self._validate_same_day_entry_exit()
            self._validate_position_management()
            self._validate_trade_execution()
            self._validate_performance()
            
            self.logger.info("\n[SUCCESS] Phase 2: 全検証カテゴリ完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 検証実行エラー: {e}")
            raise
    
    def _validate_same_day_entry_exit(self) -> None:
        """検証1: 同日Entry/Exit問題のチェック"""
        self.logger.info("\n[VALIDATION 1] 同日Entry/Exit問題チェック")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        # 同日にEntry_Signal=1とExit_Signal=-1が両方立っている日を検出
        same_day_issues = (
            (df['Entry_Signal'] == 1) & 
            (df['Exit_Signal'] == -1)
        ).sum()
        
        if same_day_issues > 0:
            self.logger.error(f"[ERROR] 同日Entry/Exit問題検出: {same_day_issues}件")
            # 詳細をログ出力
            problem_dates = df[(df['Entry_Signal'] == 1) & (df['Exit_Signal'] == -1)].index
            for date in problem_dates[:5]:  # 最初の5件を表示
                self.logger.error(f"  問題日: {date}")
        else:
            self.logger.info(f"[OK] 同日Entry/Exit問題なし")
        
        self.validation_results['same_day_entry_exit'] = {
            'issue_count': int(same_day_issues),
            'fixed': same_day_issues == 0
        }
    
    def _validate_position_management(self) -> None:
        """検証2: ポジション管理の検証"""
        self.logger.info("\n[VALIDATION 2] ポジション管理検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        # Position_Sizeカラムの存在確認
        if 'Position_Size' not in df.columns:
            self.logger.error("[ERROR] Position_Sizeカラムが存在しません")
            self.validation_results['position_management'] = {
                'position_size_exists': False,
                'position_logic_valid': False
            }
            return
        
        # ポジションサイズが0→1→0と正しく変化しているかチェック
        position_transitions = []
        for i in range(1, len(df)):
            prev_pos = df['Position_Size'].iloc[i-1]
            curr_pos = df['Position_Size'].iloc[i]
            
            if prev_pos != curr_pos:
                position_transitions.append({
                    'date': df.index[i],
                    'from': prev_pos,
                    'to': curr_pos
                })
        
        self.logger.info(f"[POSITION_TRANSITIONS] ポジション変化: {len(position_transitions)}回")
        
        # 最初の5件を表示
        for trans in position_transitions[:5]:
            self.logger.info(f"  {trans['date']}: {trans['from']:.1f} → {trans['to']:.1f}")
        
        # 異常なポジション遷移を検出（0→1, 1→0以外）
        invalid_transitions = [
            trans for trans in position_transitions
            if not ((trans['from'] == 0.0 and trans['to'] == 1.0) or
                    (trans['from'] == 1.0 and trans['to'] == 0.0))
        ]
        
        if invalid_transitions:
            self.logger.warning(f"[WARNING] 異常なポジション遷移: {len(invalid_transitions)}件")
        else:
            self.logger.info(f"[OK] ポジション遷移が正常")
        
        self.validation_results['position_management'] = {
            'position_size_exists': True,
            'transition_count': len(position_transitions),
            'invalid_transitions': len(invalid_transitions),
            'position_logic_valid': len(invalid_transitions) == 0
        }
    
    def _validate_trade_execution(self) -> None:
        """検証3: トレード実行検証"""
        self.logger.info("\n[VALIDATION 3] トレード実行検証")
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
    
    def _validate_performance(self) -> None:
        """検証4: パフォーマンス計算"""
        self.logger.info("\n[VALIDATION 4] パフォーマンス計算")
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
        
        self.validation_results['performance'] = {
            'total_pnl_pct': float(total_pnl),
            'win_rate_pct': float(win_rate),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'avg_win_pct': float(avg_win),
            'avg_loss_pct': float(avg_loss)
        }


def main():
    """メインエントリーポイント"""
    print("\n" + "=" * 80)
    print("Opening_Gap_Fixed.py 動作検証テスト - 7203.T")
    print("=" * 80 + "\n")
    
    tester = OpeningGapFixedTester(
        ticker="7203.T",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    result = tester.run_full_test()
    
    print("\n" + "=" * 80)
    print("テスト結果（Opening_Gap_Fixed）")
    print("=" * 80)
    print(f"ステータス: {result['status']}")
    
    if result['status'] == 'SUCCESS_PHASE2':
        val_results = result.get('validation_results', {})
        
        same_day = val_results.get('same_day_entry_exit', {})
        print(f"\n[検証1] 同日Entry/Exit問題:")
        print(f"  問題件数: {same_day.get('issue_count', 0)}件")
        print(f"  修正済み: {'はい' if same_day.get('fixed', False) else 'いいえ'}")
        
        pos_mgmt = val_results.get('position_management', {})
        print(f"\n[検証2] ポジション管理:")
        print(f"  Position_Size存在: {'はい' if pos_mgmt.get('position_size_exists', False) else 'いいえ'}")
        print(f"  ポジション遷移: {pos_mgmt.get('transition_count', 0)}回")
        print(f"  異常遷移: {pos_mgmt.get('invalid_transitions', 0)}件")
        print(f"  管理ロジック正常: {'はい' if pos_mgmt.get('position_logic_valid', False) else 'いいえ'}")
        
        trade_exec = val_results.get('trade_execution', {})
        print(f"\n[検証3] トレード実行:")
        print(f"  総取引数: {trade_exec.get('total_trades', 0)}件")
        
        perf = val_results.get('performance', {})
        print(f"\n[検証4] パフォーマンス:")
        print(f"  総P&L: {perf.get('total_pnl_pct', 0):.2f}%")
        print(f"  勝率: {perf.get('win_rate_pct', 0):.1f}%")
    
    print("\n" + "=" * 80 + "\n")
    
    return result


if __name__ == "__main__":
    main()
