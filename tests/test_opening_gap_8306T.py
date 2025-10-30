"""
Opening_Gap.py 動作検証テスト - 8306.T（三菱UFJ）
リアルデータを使用したバックテスト検証

目的:
1. Opening_Gap.pyが正常に動作しているかの確認
2. 作成者の意図通りに動作しているかの確認（ギャップアップ戦略として）
3. マルチ戦略システムに起きているバグの特定（戦略側に問題がないかの確認）
4. 同日Entry/Exit問題の有無の確認（Opening_Gap_Fixed.pyとの比較準備）

主な機能:
- yfinanceを使用した実データ取得（8306.T + ^DJI）
- Opening_Gap戦略のバックテスト実行
- シグナル生成の検証
- 取引実行の検証
- パフォーマンス計算
- データ整合性チェック
- ギャップアップ条件の検証

統合コンポーネント:
- strategies/Opening_Gap.py: テスト対象戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- indicators/unified_trend_detector.py: トレンド判定
- config/logger_config.py: ロガー設定

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ使用禁止）
- strategies/Opening_Gap.pyは修正しない
- yfinanceデータ取得失敗時はエラーとして処理
- DOWデータ取得失敗もエラーとして処理

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

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap import OpeningGapStrategy


class OpeningGapTester:
    """
    Opening_Gap戦略テストクラス
    
    実データを使用した包括的な動作検証を実施
    """
    
    def __init__(self, ticker: str = "8306.T", start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """
        初期化
        
        Args:
            ticker: テスト対象銘柄（デフォルト: 8306.T）
            start_date: テスト期間開始日
            end_date: テスト期間終了日
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # ロガー設定
        self.logger = setup_logger(
            "OpeningGapTester",
            log_file="logs/test_opening_gap_8306T.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"Opening_Gap.py 動作検証テスト開始")
        self.logger.info(f"銘柄: {ticker}, 期間: {start_date} ~ {end_date}")
        self.logger.info("=" * 80)
        
        # データフィード初期化
        self.data_feed = YFinanceDataFeed()
        
        # データ格納
        self.stock_data: Optional[pd.DataFrame] = None
        self.dow_data: Optional[pd.DataFrame] = None
        self.strategy: Optional[OpeningGapStrategy] = None
        self.backtest_result: Optional[pd.DataFrame] = None
        
        # 検証結果
        self.validation_results: Dict[str, Any] = {}
    
    def run_full_test(self) -> Dict[str, Any]:
        """
        フルテスト実行
        
        Returns:
            テスト結果サマリー
        """
        self.logger.info("\n[PHASE 1] データ取得開始")
        self.logger.info("-" * 80)
        
        try:
            # Step 1: データ取得
            self._fetch_data()
            
            # Step 2: 戦略初期化
            self._initialize_strategy()
            
            # Step 3: バックテスト実行
            self._run_backtest()
            
            # Step 4: 検証実行（Phase 2）
            self.logger.info("\n[PHASE 2] 検証ロジック実行開始")
            self.logger.info("-" * 80)
            self._validate_all()
            
            # Step 5: レポート生成（Phase 3で実装）
            # self._generate_reports()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[SUCCESS] Phase 1-2: データ取得・バックテスト・検証完了")
            self.logger.info("=" * 80)
            
            return {
                'status': 'SUCCESS_PHASE2',
                'stock_data_rows': len(self.stock_data),
                'dow_data_rows': len(self.dow_data),
                'backtest_result_rows': len(self.backtest_result) if self.backtest_result is not None else 0,
                'validation_results': self.validation_results
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] テスト実行エラー: {e}", exc_info=True)
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def _fetch_data(self) -> None:
        """
        データ取得（yfinance使用）
        
        copilot-instructions.md準拠:
        - モックデータ使用禁止
        - データ取得失敗時はエラーとして処理
        """
        self.logger.info(f"[DATA_FETCH] 株価データ取得: {self.ticker}")
        
        try:
            # 株価データ取得（8306.T）
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
            self.logger.info(f"  カラム: {list(self.stock_data.columns)}")
            self.logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 株価データ取得失敗: {e}")
            raise RuntimeError(
                f"Failed to retrieve stock data for {self.ticker}: {e}. "
                "Mock/dummy data fallback is prohibited by copilot-instructions.md"
            ) from e
        
        self.logger.info(f"\n[DATA_FETCH] DOWデータ取得: ^DJI")
        
        try:
            # DOWデータ取得（^DJI）
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
            self.logger.info(f"  カラム: {list(self.dow_data.columns)}")
            self.logger.info(f"  期間: {self.dow_data.index[0]} ~ {self.dow_data.index[-1]}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] DOWデータ取得失敗: {e}")
            raise RuntimeError(
                f"Failed to retrieve DOW data (^DJI): {e}. "
                "Mock/dummy data fallback is prohibited by copilot-instructions.md"
            ) from e
    
    def _initialize_strategy(self) -> None:
        """
        OpeningGapStrategy初期化
        """
        self.logger.info("\n[STRATEGY_INIT] OpeningGapStrategy初期化")
        self.logger.info("-" * 80)
        
        try:
            # デフォルトパラメータで初期化（Opening_Gap.pyの作成者意図を確認）
            self.strategy = OpeningGapStrategy(
                data=self.stock_data,
                dow_data=self.dow_data,
                params=None,  # デフォルトパラメータ使用
                price_column="Adj Close"
            )
            
            # パラメータ確認
            self.logger.info("[STRATEGY_PARAMS] 戦略パラメータ:")
            for key, value in self.strategy.params.items():
                self.logger.info(f"  {key}: {value}")
            
            self.logger.info("[SUCCESS] OpeningGapStrategy初期化完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] OpeningGapStrategy初期化失敗: {e}")
            raise
    
    def _run_backtest(self) -> None:
        """
        バックテスト実行
        
        copilot-instructions.md準拠:
        - strategy.backtest()の呼び出しをスキップしない
        - 実際の実行結果を確認
        """
        self.logger.info("\n[BACKTEST] バックテスト実行開始")
        self.logger.info("-" * 80)
        
        try:
            # バックテスト実行（必須）
            self.backtest_result = self.strategy.backtest()
            
            if self.backtest_result is None:
                raise RuntimeError(
                    "strategy.backtest() returned None. "
                    "Skipping backtest is prohibited by copilot-instructions.md"
                )
            
            self.logger.info(f"[SUCCESS] バックテスト実行完了")
            self.logger.info(f"  結果データ: {len(self.backtest_result)} rows")
            self.logger.info(f"  カラム: {list(self.backtest_result.columns)}")
            
            # 互換性レイヤー: Position_SizeからPositionカラムを生成
            if 'Position_Size' in self.backtest_result.columns:
                self.backtest_result['Position'] = (
                    self.backtest_result['Position_Size'] > 0
                ).astype(int)
                self.logger.info("[COMPAT] Position カラムを Position_Size から生成")
            elif 'Position' not in self.backtest_result.columns:
                # どちらもない場合はエラー
                raise RuntimeError(
                    "Neither 'Position' nor 'Position_Size' column found in backtest result"
                )
            
            # シグナル集計（基本確認）
            entry_count = (self.backtest_result['Entry_Signal'] == 1).sum()
            exit_count = (self.backtest_result['Exit_Signal'] == -1).sum()
            position_days = (self.backtest_result['Position'] == 1).sum()
            position_size_days = (self.backtest_result['Position_Size'] > 0).sum() if 'Position_Size' in self.backtest_result.columns else 0
            
            self.logger.info(f"\n[BACKTEST_SUMMARY]")
            self.logger.info(f"  Entry_Signal == 1: {entry_count} 回")
            self.logger.info(f"  Exit_Signal == -1: {exit_count} 回")
            self.logger.info(f"  Position == 1: {position_days} 日")
            if 'Position_Size' in self.backtest_result.columns:
                self.logger.info(f"  Position_Size > 0: {position_size_days} 日")
            
            # 同日Entry/Exit問題の検証
            same_day_issues = (
                (self.backtest_result['Entry_Signal'] == 1) & 
                (self.backtest_result['Exit_Signal'] == -1)
            ).sum()
            
            if same_day_issues > 0:
                self.logger.warning(
                    f"[WARNING] 同日Entry/Exit問題を検出: {same_day_issues} 件"
                )
                # 該当日のリスト
                same_day_dates = self.backtest_result[
                    (self.backtest_result['Entry_Signal'] == 1) & 
                    (self.backtest_result['Exit_Signal'] == -1)
                ].index.tolist()
                self.logger.warning(f"  該当日: {same_day_dates[:5]}...")  # 最初の5件のみ表示
            else:
                self.logger.info("[OK] 同日Entry/Exit問題なし")
            
            # copilot-instructions.md準拠: 実際の取引件数 > 0 を検証
            if entry_count == 0:
                self.logger.warning(
                    "[WARNING] エントリー回数が0です。"
                    "ギャップアップ条件が厳しすぎる可能性があります。"
                )
            
            # エントリーとエグジットの数が一致するかチェック
            if entry_count != exit_count:
                self.logger.warning(
                    f"[WARNING] エントリー回数({entry_count})とエグジット回数({exit_count})が一致しません"
                )
            
        except Exception as e:
            self.logger.error(f"[ERROR] バックテスト実行失敗: {e}")
            raise
    
    def _validate_all(self) -> None:
        """
        Phase 2: 検証ロジック実行
        
        5つの検証カテゴリ:
        1. シグナル生成検証
        2. トレード実行検証
        3. パフォーマンス計算
        4. データ整合性検証
        5. 戦略パラメータ検証
        """
        try:
            # 検証1: シグナル生成検証
            self._validate_signal_generation()
            
            # 検証2: トレード実行検証
            self._validate_trade_execution()
            
            # 検証3: パフォーマンス計算
            self._validate_performance_calculation()
            
            # 検証4: データ整合性検証
            self._validate_data_integrity()
            
            # 検証5: 戦略パラメータ検証
            self._validate_strategy_parameters()
            
            self.logger.info("\n[SUCCESS] Phase 2: 全検証カテゴリ完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] 検証実行エラー: {e}")
            raise
    
    def _validate_signal_generation(self) -> None:
        """
        検証1: シグナル生成検証
        
        確認項目:
        - Entry_Signal == 1 の回数
        - Exit_Signal == -1 の回数
        - Entry/Exitの対応関係
        - 同日Entry/Exit問題の有無
        """
        self.logger.info("\n[VALIDATION 1] シグナル生成検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        # Entry_Signalカウント
        entry_count = (df['Entry_Signal'] == 1).sum()
        exit_count = (df['Exit_Signal'] == -1).sum()
        
        self.logger.info(f"[SIGNAL_COUNT] Entry: {entry_count}, Exit: {exit_count}")
        
        # 同日Entry/Exit検証
        same_day_count = (
            (df['Entry_Signal'] == 1) & 
            (df['Exit_Signal'] == -1)
        ).sum()
        
        if same_day_count > 0:
            self.logger.warning(f"[ISSUE] 同日Entry/Exit問題検出: {same_day_count} 件")
            same_day_dates = df[
                (df['Entry_Signal'] == 1) & 
                (df['Exit_Signal'] == -1)
            ].index.tolist()
            self.logger.warning(f"  該当日: {same_day_dates}")
        else:
            self.logger.info("[OK] 同日Entry/Exit問題なし")
        
        # Entry/Exit数の一致確認
        if entry_count != exit_count:
            self.logger.warning(
                f"[ISSUE] Entry({entry_count})とExit({exit_count})の数が不一致"
            )
        else:
            self.logger.info(f"[OK] Entry/Exit数の一致確認: {entry_count}件")
        
        # 検証結果を保存
        self.validation_results['signal_generation'] = {
            'entry_count': int(entry_count),
            'exit_count': int(exit_count),
            'same_day_issues': int(same_day_count),
            'entry_exit_match': entry_count == exit_count
        }
    
    def _validate_trade_execution(self) -> None:
        """
        検証2: トレード実行検証
        
        確認項目:
        - Position遷移の正当性（0→1→0のパターン）
        - エントリー日とエグジット日の記録
        - 平均保有日数
        - 最長保有日数
        """
        self.logger.info("\n[VALIDATION 2] トレード実行検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        # トレード抽出（Entry → Exit のペア）
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
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'hold_days': hold_days,
                    'pnl_pct': float(pnl_pct)
                })
                
                entry_date = None
                entry_price = None
        
        trade_count = len(trades)
        self.logger.info(f"[TRADE_COUNT] 総取引数: {trade_count} 件")
        
        if trade_count > 0:
            # 保有日数統計
            hold_days_list = [t['hold_days'] for t in trades]
            avg_hold_days = np.mean(hold_days_list)
            max_hold_days = max(hold_days_list)
            min_hold_days = min(hold_days_list)
            
            self.logger.info(f"[HOLD_DAYS] 平均: {avg_hold_days:.1f}日, 最長: {max_hold_days}日, 最短: {min_hold_days}日")
            
            # 最初の3件のトレード詳細
            self.logger.info("\n[TRADE_EXAMPLES] 最初の3件のトレード:")
            for i, trade in enumerate(trades[:3], 1):
                self.logger.info(
                    f"  #{i}: {trade['entry_date'].strftime('%Y-%m-%d')} → "
                    f"{trade['exit_date'].strftime('%Y-%m-%d')} "
                    f"({trade['hold_days']}日) P&L: {trade['pnl_pct']:.2f}%"
                )
            
            # 検証結果を保存
            self.validation_results['trade_execution'] = {
                'total_trades': trade_count,
                'avg_hold_days': float(avg_hold_days),
                'max_hold_days': int(max_hold_days),
                'min_hold_days': int(min_hold_days),
                'trades': trades  # 全トレードデータ保存
            }
        else:
            self.logger.warning("[WARNING] トレードが1件もありません")
            self.validation_results['trade_execution'] = {
                'total_trades': 0,
                'avg_hold_days': 0.0,
                'max_hold_days': 0,
                'min_hold_days': 0,
                'trades': []
            }
    
    def _validate_performance_calculation(self) -> None:
        """
        検証3: パフォーマンス計算
        
        確認項目:
        - 総P&L（全トレード合計）
        - 勝率（勝ちトレード数 / 総トレード数）
        - 平均利益/平均損失
        - 最大ドローダウン
        - プロフィットファクター
        """
        self.logger.info("\n[VALIDATION 3] パフォーマンス計算")
        self.logger.info("-" * 80)
        
        trades = self.validation_results.get('trade_execution', {}).get('trades', [])
        
        if len(trades) == 0:
            self.logger.warning("[WARNING] トレードが0件のためパフォーマンス計算不可")
            self.validation_results['performance'] = {}
            return
        
        # P&L統計
        pnl_list = [t['pnl_pct'] for t in trades]
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        
        # 勝ち/負けトレード
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / len(trades)) * 100 if len(trades) > 0 else 0.0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        # プロフィットファクター
        total_win = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 最大ドローダウン計算
        cumulative_pnl = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0.0
        
        self.logger.info(f"[P&L] 総P&L: {total_pnl:.2f}%, 平均P&L: {avg_pnl:.2f}%")
        self.logger.info(f"[WIN_RATE] 勝率: {win_rate:.1f}% ({win_count}勝 / {loss_count}敗)")
        self.logger.info(f"[AVG] 平均利益: {avg_win:.2f}%, 平均損失: {avg_loss:.2f}%")
        self.logger.info(f"[RISK] 最大ドローダウン: {max_drawdown:.2f}%")
        self.logger.info(f"[PF] プロフィットファクター: {profit_factor:.2f}")
        
        # 検証結果を保存
        self.validation_results['performance'] = {
            'total_pnl_pct': float(total_pnl),
            'avg_pnl_pct': float(avg_pnl),
            'win_rate_pct': float(win_rate),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'avg_win_pct': float(avg_win),
            'avg_loss_pct': float(avg_loss),
            'max_drawdown_pct': float(max_drawdown),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.99
        }
    
    def _validate_data_integrity(self) -> None:
        """
        検証4: データ整合性検証
        
        確認項目:
        - entry_prices dict の整合性
        - high_prices dict の整合性
        - Position列の連続性
        - 強制清算の検出
        """
        self.logger.info("\n[VALIDATION 4] データ整合性検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        # Position列の連続性検証
        position_changes = df['Position'].diff()
        invalid_transitions = position_changes[
            (position_changes != 0) & 
            (position_changes != 1) & 
            (position_changes != -1)
        ]
        
        if len(invalid_transitions) > 0:
            self.logger.warning(
                f"[ISSUE] 不正なPosition遷移を検出: {len(invalid_transitions)} 件"
            )
        else:
            self.logger.info("[OK] Position列の連続性確認")
        
        # Entry/Exit対応の整合性
        entry_indices = df[df['Entry_Signal'] == 1].index.tolist()
        exit_indices = df[df['Exit_Signal'] == -1].index.tolist()
        
        unmatched_entries = []
        for entry_idx in entry_indices:
            # 対応するExitを探す
            matching_exits = [e for e in exit_indices if e > entry_idx]
            if not matching_exits:
                unmatched_entries.append(entry_idx)
        
        if unmatched_entries:
            self.logger.warning(
                f"[ISSUE] 対応するExitがないEntry: {len(unmatched_entries)} 件"
            )
            self.logger.info(f"  該当日: {[str(d) for d in unmatched_entries[:5]]}")
        else:
            self.logger.info("[OK] 全てのEntryに対応するExitが存在")
        
        # 検証結果を保存
        self.validation_results['data_integrity'] = {
            'invalid_transitions': len(invalid_transitions),
            'unmatched_entries': len(unmatched_entries),
            'integrity_ok': len(invalid_transitions) == 0 and len(unmatched_entries) == 0
        }
    
    def _validate_strategy_parameters(self) -> None:
        """
        検証5: 戦略パラメータ検証
        
        確認項目:
        - gap_threshold = 1% の確認
        - stop_loss = 2% の確認
        - take_profit = 5% の確認
        - その他パラメータの妥当性
        """
        self.logger.info("\n[VALIDATION 5] 戦略パラメータ検証")
        self.logger.info("-" * 80)
        
        params = self.strategy.params
        
        # 重要パラメータの確認
        gap_threshold = params.get('gap_threshold', None)
        stop_loss = params.get('stop_loss', None)
        take_profit = params.get('take_profit', None)
        max_hold_days = params.get('max_hold_days', None)
        
        self.logger.info(f"[PARAM] gap_threshold: {gap_threshold} (期待値: 0.01 = 1%)")
        self.logger.info(f"[PARAM] stop_loss: {stop_loss} (期待値: 0.02 = 2%)")
        self.logger.info(f"[PARAM] take_profit: {take_profit} (期待値: 0.05 = 5%)")
        self.logger.info(f"[PARAM] max_hold_days: {max_hold_days} (期待値: 5日)")
        
        # パラメータ妥当性チェック
        issues = []
        if gap_threshold != 0.01:
            issues.append(f"gap_threshold が期待値(0.01)と異なる: {gap_threshold}")
        if stop_loss != 0.02:
            issues.append(f"stop_loss が期待値(0.02)と異なる: {stop_loss}")
        if take_profit != 0.05:
            issues.append(f"take_profit が期待値(0.05)と異なる: {take_profit}")
        
        if issues:
            for issue in issues:
                self.logger.warning(f"[ISSUE] {issue}")
        else:
            self.logger.info("[OK] 全パラメータが期待値と一致")
        
        # 検証結果を保存
        self.validation_results['strategy_parameters'] = {
            'gap_threshold': float(gap_threshold) if gap_threshold is not None else None,
            'stop_loss': float(stop_loss) if stop_loss is not None else None,
            'take_profit': float(take_profit) if take_profit is not None else None,
            'max_hold_days': int(max_hold_days) if max_hold_days is not None else None,
            'all_params': {k: v for k, v in params.items()},
            'params_match_expected': len(issues) == 0
        }


def main():
    """
    メインエントリーポイント
    """
    print("\n" + "=" * 80)
    print("Opening_Gap.py 動作検証テスト - Phase 1-2")
    print("=" * 80 + "\n")
    
    # テスター初期化
    tester = OpeningGapTester(
        ticker="8306.T",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Phase 1実行（データ取得・バックテスト実行）
    result = tester.run_full_test()
    
    # 結果出力
    print("\n" + "=" * 80)
    print("Phase 2 テスト結果")
    print("=" * 80)
    print(f"ステータス: {result['status']}")
    
    if result['status'] == 'SUCCESS_PHASE2':
        print(f"\n[OK] Phase 1-2完了")
        print(f"  株価データ: {result['stock_data_rows']} rows")
        print(f"  DOWデータ: {result['dow_data_rows']} rows")
        print(f"  バックテスト結果: {result['backtest_result_rows']} rows")
        
        # 検証結果サマリー
        val_results = result.get('validation_results', {})
        
        # シグナル生成
        sig_gen = val_results.get('signal_generation', {})
        print(f"\n[検証1] シグナル生成:")
        print(f"  Entry: {sig_gen.get('entry_count', 0)}回, Exit: {sig_gen.get('exit_count', 0)}回")
        print(f"  同日Entry/Exit: {sig_gen.get('same_day_issues', 0)}件")
        
        # トレード実行
        trade_exec = val_results.get('trade_execution', {})
        print(f"\n[検証2] トレード実行:")
        print(f"  総取引数: {trade_exec.get('total_trades', 0)}件")
        print(f"  平均保有: {trade_exec.get('avg_hold_days', 0):.1f}日")
        
        # パフォーマンス
        perf = val_results.get('performance', {})
        print(f"\n[検証3] パフォーマンス:")
        print(f"  総P&L: {perf.get('total_pnl_pct', 0):.2f}%")
        print(f"  勝率: {perf.get('win_rate_pct', 0):.1f}%")
        print(f"  最大DD: {perf.get('max_drawdown_pct', 0):.2f}%")
        
        # データ整合性
        integrity = val_results.get('data_integrity', {})
        print(f"\n[検証4] データ整合性:")
        print(f"  整合性OK: {integrity.get('integrity_ok', False)}")
        
        # 戦略パラメータ
        params = val_results.get('strategy_parameters', {})
        print(f"\n[検証5] 戦略パラメータ:")
        print(f"  gap_threshold: {params.get('gap_threshold', 'N/A')}")
        print(f"  stop_loss: {params.get('stop_loss', 'N/A')}")
        print(f"  take_profit: {params.get('take_profit', 'N/A')}")
        
        print(f"\n[NEXT] Phase 3: レポート生成")
    else:
        print(f"\n[ERROR] テスト失敗")
        print(f"  エラー: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80 + "\n")
    
    return result


if __name__ == "__main__":
    main()
