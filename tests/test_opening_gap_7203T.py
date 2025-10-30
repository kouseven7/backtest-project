"""
Opening_Gap.py 動作検証テスト - 7203.T（トヨタ自動車）
最適化パラメータを使用したバックテスト検証

目的:
1. 最適化パラメータ（opening_gap_7203.T_2025-06-19.json）の動作確認
2. デフォルトパラメータ（8306.T）との比較
3. 最適化による改善効果の検証
4. トレンドフィルター設定の確認

主な機能:
- yfinanceを使用した実データ取得（7203.T + ^DJI）
- Opening_Gap戦略のバックテスト実行（最適化パラメータ使用）
- シグナル生成の検証
- トレード実行の検証
- パフォーマンス計算
- データ整合性チェック
- 8306.T結果との比較

統合コンポーネント:
- strategies/Opening_Gap.py: テスト対象戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- indicators/unified_trend_detector.py: トレンド判定
- config/optimized_params/opening_gap_7203.T_2025-06-19.json: 最適化パラメータ
- config/logger_config.py: ロガー設定

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ使用禁止）
- strategies/Opening_Gap.pyは修正しない
- yfinanceデータ取得失敗時はエラーとして処理
- DOWデータ取得失敗もエラーとして処理
- 最適化パラメータの明示的な指定

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


class OpeningGap7203Tester:
    """
    Opening_Gap戦略テストクラス（7203.T + 最適化パラメータ）
    
    実データと最適化パラメータを使用した包括的な動作検証を実施
    """
    
    def __init__(self, ticker: str = "7203.T", start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """
        初期化
        
        Args:
            ticker: テスト対象銘柄（デフォルト: 7203.T）
            start_date: テスト期間開始日
            end_date: テスト期間終了日
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # ロガー設定
        self.logger = setup_logger(
            "OpeningGap7203Tester",
            log_file="logs/test_opening_gap_7203T.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"Opening_Gap.py 動作検証テスト開始（最適化パラメータ使用）")
        self.logger.info(f"銘柄: {ticker}, 期間: {start_date} ~ {end_date}")
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
        self.logger.info("\n[PHASE 1] データ取得開始")
        self.logger.info("-" * 80)
        
        try:
            # Step 1: 最適化パラメータ読み込み
            self._load_optimized_params()
            
            # Step 2: データ取得
            self._fetch_data()
            
            # Step 3: 戦略初期化（最適化パラメータ使用）
            self._initialize_strategy()
            
            # Step 4: バックテスト実行
            self._run_backtest()
            
            # Step 5: 検証実行（Phase 2）
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
        """
        最適化パラメータの読み込み
        
        config/optimized_params/opening_gap_7203.T_2025-06-19.json
        """
        self.logger.info("\n[OPTIMIZED_PARAMS] 最適化パラメータ読み込み")
        self.logger.info("-" * 80)
        
        param_file = project_root / "config" / "optimized_params" / "opening_gap_7203.T_2025-06-19.json"
        
        if not param_file.exists():
            raise FileNotFoundError(
                f"最適化パラメータファイルが見つかりません: {param_file}"
            )
        
        with open(param_file, 'r', encoding='utf-8') as f:
            param_data = json.load(f)
        
        self.optimized_params = param_data['parameters']
        
        self.logger.info(f"[SUCCESS] 最適化パラメータ読み込み完了")
        self.logger.info(f"  ファイル: {param_file.name}")
        self.logger.info(f"  最適化日: {param_data['optimization_date']}")
        self.logger.info(f"  スコア: {param_data['parameters']['score']:.2f}")
        self.logger.info(f"\n[OPTIMIZED_PARAMS] パラメータ詳細:")
        for key, value in self.optimized_params.items():
            if key != 'score':
                self.logger.info(f"  {key}: {value}")
    
    def _fetch_data(self) -> None:
        """
        データ取得（yfinance使用）
        
        copilot-instructions.md準拠:
        - モックデータ使用禁止
        - データ取得失敗時はエラーとして処理
        """
        self.logger.info(f"\n[DATA_FETCH] 株価データ取得: {self.ticker}")
        
        try:
            # 株価データ取得（7203.T）
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
        OpeningGapStrategy初期化（最適化パラメータ使用）
        """
        self.logger.info("\n[STRATEGY_INIT] OpeningGapStrategy初期化（最適化パラメータ使用）")
        self.logger.info("-" * 80)
        
        try:
            # 最適化パラメータから'score'を除外
            strategy_params = {k: v for k, v in self.optimized_params.items() if k != 'score'}
            
            # 重要: trend_filter_enabledの確認
            if 'trend_filter_enabled' not in strategy_params:
                self.logger.warning("[WARNING] trend_filter_enabledが最適化パラメータに含まれていません。デフォルト値を使用します。")
            
            self.strategy = OpeningGapStrategy(
                data=self.stock_data,
                dow_data=self.dow_data,
                params=strategy_params,  # 最適化パラメータを明示的に指定
                price_column="Adj Close"
            )
            
            # パラメータ確認
            self.logger.info("[STRATEGY_PARAMS] 実際に使用されるパラメータ:")
            for key, value in self.strategy.params.items():
                self.logger.info(f"  {key}: {value}")
            
            # トレンドフィルター設定の確認
            trend_filter_enabled = self.strategy.params.get('trend_filter_enabled', False)
            allowed_trends = self.strategy.params.get('allowed_trends', [])
            self.logger.info(f"\n[TREND_FILTER] トレンドフィルター設定:")
            self.logger.info(f"  trend_filter_enabled: {trend_filter_enabled}")
            self.logger.info(f"  allowed_trends: {allowed_trends}")
            
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
                    "最適化パラメータの条件が厳しすぎる可能性があります。"
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
        """検証1: シグナル生成検証"""
        self.logger.info("\n[VALIDATION 1] シグナル生成検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
        entry_count = (df['Entry_Signal'] == 1).sum()
        exit_count = (df['Exit_Signal'] == -1).sum()
        
        self.logger.info(f"[SIGNAL_COUNT] Entry: {entry_count}, Exit: {exit_count}")
        
        same_day_count = (
            (df['Entry_Signal'] == 1) & 
            (df['Exit_Signal'] == -1)
        ).sum()
        
        if same_day_count > 0:
            self.logger.warning(f"[ISSUE] 同日Entry/Exit問題検出: {same_day_count} 件")
        else:
            self.logger.info("[OK] 同日Entry/Exit問題なし")
        
        if entry_count != exit_count:
            self.logger.warning(
                f"[ISSUE] Entry({entry_count})とExit({exit_count})の数が不一致"
            )
        else:
            self.logger.info(f"[OK] Entry/Exit数の一致確認: {entry_count}件")
        
        self.validation_results['signal_generation'] = {
            'entry_count': int(entry_count),
            'exit_count': int(exit_count),
            'same_day_issues': int(same_day_count),
            'entry_exit_match': entry_count == exit_count
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
            hold_days_list = [t['hold_days'] for t in trades]
            avg_hold_days = np.mean(hold_days_list)
            max_hold_days = max(hold_days_list)
            min_hold_days = min(hold_days_list)
            
            self.logger.info(f"[HOLD_DAYS] 平均: {avg_hold_days:.1f}日, 最長: {max_hold_days}日, 最短: {min_hold_days}日")
            
            self.logger.info("\n[TRADE_EXAMPLES] 最初の3件のトレード:")
            for i, trade in enumerate(trades[:3], 1):
                self.logger.info(
                    f"  #{i}: {trade['entry_date'].strftime('%Y-%m-%d')} → "
                    f"{trade['exit_date'].strftime('%Y-%m-%d')} "
                    f"({trade['hold_days']}日) P&L: {trade['pnl_pct']:.2f}%"
                )
            
            self.validation_results['trade_execution'] = {
                'total_trades': trade_count,
                'avg_hold_days': float(avg_hold_days),
                'max_hold_days': int(max_hold_days),
                'min_hold_days': int(min_hold_days),
                'trades': trades
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
        """検証3: パフォーマンス計算"""
        self.logger.info("\n[VALIDATION 3] パフォーマンス計算")
        self.logger.info("-" * 80)
        
        trades = self.validation_results.get('trade_execution', {}).get('trades', [])
        
        if len(trades) == 0:
            self.logger.warning("[WARNING] トレードが0件のためパフォーマンス計算不可")
            self.validation_results['performance'] = {}
            return
        
        pnl_list = [t['pnl_pct'] for t in trades]
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / len(trades)) * 100 if len(trades) > 0 else 0.0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        total_win = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        cumulative_pnl = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0.0
        
        self.logger.info(f"[P&L] 総P&L: {total_pnl:.2f}%, 平均P&L: {avg_pnl:.2f}%")
        self.logger.info(f"[WIN_RATE] 勝率: {win_rate:.1f}% ({win_count}勝 / {loss_count}敗)")
        self.logger.info(f"[AVG] 平均利益: {avg_win:.2f}%, 平均損失: {avg_loss:.2f}%")
        self.logger.info(f"[RISK] 最大ドローダウン: {max_drawdown:.2f}%")
        self.logger.info(f"[PF] プロフィットファクター: {profit_factor:.2f}")
        
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
        """検証4: データ整合性検証"""
        self.logger.info("\n[VALIDATION 4] データ整合性検証")
        self.logger.info("-" * 80)
        
        df = self.backtest_result
        
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
        
        entry_indices = df[df['Entry_Signal'] == 1].index.tolist()
        exit_indices = df[df['Exit_Signal'] == -1].index.tolist()
        
        unmatched_entries = []
        for entry_idx in entry_indices:
            matching_exits = [e for e in exit_indices if e > entry_idx]
            if not matching_exits:
                unmatched_entries.append(entry_idx)
        
        if unmatched_entries:
            self.logger.warning(
                f"[ISSUE] 対応するExitがないEntry: {len(unmatched_entries)} 件"
            )
        else:
            self.logger.info("[OK] 全てのEntryに対応するExitが存在")
        
        self.validation_results['data_integrity'] = {
            'invalid_transitions': len(invalid_transitions),
            'unmatched_entries': len(unmatched_entries),
            'integrity_ok': len(invalid_transitions) == 0 and len(unmatched_entries) == 0
        }
    
    def _validate_strategy_parameters(self) -> None:
        """検証5: 戦略パラメータ検証"""
        self.logger.info("\n[VALIDATION 5] 戦略パラメータ検証（最適化パラメータ）")
        self.logger.info("-" * 80)
        
        params = self.strategy.params
        
        gap_threshold = params.get('gap_threshold', None)
        stop_loss = params.get('stop_loss', None)
        take_profit = params.get('take_profit', None)
        max_hold_days = params.get('max_hold_days', None)
        dow_filter_enabled = params.get('dow_filter_enabled', None)
        volatility_filter = params.get('volatility_filter', None)
        
        self.logger.info(f"[PARAM] gap_threshold: {gap_threshold} (最適化値: 0.02 = 2%)")
        self.logger.info(f"[PARAM] stop_loss: {stop_loss} (最適化値: 0.01 = 1%)")
        self.logger.info(f"[PARAM] take_profit: {take_profit} (最適化値: 0.03 = 3%)")
        self.logger.info(f"[PARAM] max_hold_days: {max_hold_days} (最適化値: 7日)")
        self.logger.info(f"[PARAM] dow_filter_enabled: {dow_filter_enabled} (最適化値: True)")
        self.logger.info(f"[PARAM] volatility_filter: {volatility_filter} (最適化値: True)")
        
        # 最適化パラメータとの比較
        issues = []
        if gap_threshold != 0.02:
            issues.append(f"gap_threshold が最適化値(0.02)と異なる: {gap_threshold}")
        if stop_loss != 0.01:
            issues.append(f"stop_loss が最適化値(0.01)と異なる: {stop_loss}")
        if take_profit != 0.03:
            issues.append(f"take_profit が最適化値(0.03)と異なる: {take_profit}")
        if max_hold_days != 7:
            issues.append(f"max_hold_days が最適化値(7)と異なる: {max_hold_days}")
        
        if issues:
            for issue in issues:
                self.logger.warning(f"[ISSUE] {issue}")
        else:
            self.logger.info("[OK] 全パラメータが最適化値と一致")
        
        self.validation_results['strategy_parameters'] = {
            'gap_threshold': float(gap_threshold) if gap_threshold is not None else None,
            'stop_loss': float(stop_loss) if stop_loss is not None else None,
            'take_profit': float(take_profit) if take_profit is not None else None,
            'max_hold_days': int(max_hold_days) if max_hold_days is not None else None,
            'dow_filter_enabled': bool(dow_filter_enabled) if dow_filter_enabled is not None else None,
            'volatility_filter': bool(volatility_filter) if volatility_filter is not None else None,
            'all_params': {k: v for k, v in params.items()},
            'params_match_expected': len(issues) == 0
        }


def main():
    """
    メインエントリーポイント
    """
    print("\n" + "=" * 80)
    print("Opening_Gap.py 動作検証テスト - 7203.T（最適化パラメータ）")
    print("=" * 80 + "\n")
    
    # テスター初期化
    tester = OpeningGap7203Tester(
        ticker="7203.T",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Phase 1-2実行
    result = tester.run_full_test()
    
    # 結果出力
    print("\n" + "=" * 80)
    print("Phase 2 テスト結果（7203.T + 最適化パラメータ）")
    print("=" * 80)
    print(f"ステータス: {result['status']}")
    
    if result['status'] == 'SUCCESS_PHASE2':
        print(f"\n[OK] Phase 1-2完了")
        print(f"  銘柄: {result['ticker']}")
        print(f"  株価データ: {result['stock_data_rows']} rows")
        print(f"  DOWデータ: {result['dow_data_rows']} rows")
        print(f"  バックテスト結果: {result['backtest_result_rows']} rows")
        
        val_results = result.get('validation_results', {})
        
        sig_gen = val_results.get('signal_generation', {})
        print(f"\n[検証1] シグナル生成:")
        print(f"  Entry: {sig_gen.get('entry_count', 0)}回, Exit: {sig_gen.get('exit_count', 0)}回")
        print(f"  同日Entry/Exit: {sig_gen.get('same_day_issues', 0)}件")
        
        trade_exec = val_results.get('trade_execution', {})
        print(f"\n[検証2] トレード実行:")
        print(f"  総取引数: {trade_exec.get('total_trades', 0)}件")
        print(f"  平均保有: {trade_exec.get('avg_hold_days', 0):.1f}日")
        
        perf = val_results.get('performance', {})
        print(f"\n[検証3] パフォーマンス:")
        print(f"  総P&L: {perf.get('total_pnl_pct', 0):.2f}%")
        print(f"  勝率: {perf.get('win_rate_pct', 0):.1f}%")
        print(f"  最大DD: {perf.get('max_drawdown_pct', 0):.2f}%")
        print(f"  プロフィットファクター: {perf.get('profit_factor', 0):.2f}")
        
        integrity = val_results.get('data_integrity', {})
        print(f"\n[検証4] データ整合性:")
        print(f"  整合性OK: {integrity.get('integrity_ok', False)}")
        
        params = val_results.get('strategy_parameters', {})
        print(f"\n[検証5] 戦略パラメータ:")
        print(f"  gap_threshold: {params.get('gap_threshold', 'N/A')}")
        print(f"  stop_loss: {params.get('stop_loss', 'N/A')}")
        print(f"  take_profit: {params.get('take_profit', 'N/A')}")
        print(f"  dow_filter_enabled: {params.get('dow_filter_enabled', 'N/A')}")
        print(f"  volatility_filter: {params.get('volatility_filter', 'N/A')}")
        
        print(f"\n[NEXT] Phase 3: 8306.Tとの比較分析")
    else:
        print(f"\n[ERROR] テスト失敗")
        print(f"  エラー: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80 + "\n")
    
    return result


if __name__ == "__main__":
    main()
