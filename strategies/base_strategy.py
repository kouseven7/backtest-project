"""
BaseStrategy - 全戦略の基底クラス（バックテスト実行・シグナル生成の統一インターフェース）

全ての投資戦略の基底クラスを提供します。エントリー・イグジットシグナルの生成インターフェース、
バックテスト実行、ログ記録などの共通機能を実装。各戦略クラスはこのBaseStrategyを継承して実装します。

主な機能:
- backtest()メソッド: 全期間一括バックテスト実行（現行方式）
- generate_entry_signal()/generate_exit_signal(): シグナル生成インターフェース
- ルックアヘッドバイアス防止: 翌日始値エントリー（data['Open'].iloc[idx + 1]）
- トレーリングストップ・資金管理機能
- デバッグログ機能（DEBUG_BACKTEST=1環境変数で有効化）
- ウォームアップ期間対応（150日推奨）

統合コンポーネント:
- DSSMS統合: dssms_integrated_main.py経由での動的銘柄切替対応
- data_fetcher: get_parameters_and_data()によるデータ取得
- ComprehensiveReporter: 統一出力エンジン経由での結果出力
- MainSystemController: main_new.py経由でのマルチ戦略制御

セーフティ機能/注意事項:
- ルックアヘッドバイアス禁止（3原則）: 前日データ判断・翌日始値エントリー・取引コスト考慮
- 【設計問題】現在のbacktest()は全期間一括判定のため、DSSMS日次判断と不一致
- 【Phase 3実装予定】backtest_daily()メソッド: 日次判定でリアルトレード対応
- 【重要】新戦略実装時はcopilot-instructions.mdのルックアヘッドバイアス防止ルール必須遵守
- 【監視】トレード件数0の場合: インジケーター.shift(1)未適用・ウォームアップ期間不足を確認

Author: Backtest Project Team
Created: 2023-01-01
Last Modified: 2025-12-30
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import logging

class BaseStrategy:
    """
    BaseStrategyは、全戦略に共通する基本処理（パラメータ初期化、エントリー／イグジット判定、ログ出力など）を実装する基底クラスです。
    各戦略は、このクラスを継承して固有のシグナル生成ロジックを実装してください。
    """
    def __init__(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        基本戦略の初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（カスタマイズ可能）
        """
        self.data = data
        self.params = params or {}
        self.logger = self._setup_logger()
        
        # エントリー価格を記録する辞書（派生クラスで使用可能）
        if not hasattr(self, 'entry_prices'):
            self.entry_prices = {}
        
        # トレーリングストップ用の最高価格を記録する辞書（派生クラスで使用可能）
        if not hasattr(self, 'high_prices'):
            self.high_prices = {}
        
        self.initialize_strategy()
        
    def _setup_logger(self) -> logging.Logger:
        """
        ロガーの初期設定を行う。
        
        Returns:
            logging.Logger: 設定されたロガーインスタンス
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:  # 既にハンドラが設定されていない場合のみ
            # デバッグログを有効化（環境変数で制御可能）
            import os
            log_level = logging.DEBUG if os.getenv('DEBUG_BACKTEST') == '1' else logging.INFO
            logger.setLevel(log_level)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # コンソールにログを出力
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # ファイルにもログを出力（オプション）
            try:
                file_handler = logging.FileHandler('logs/backtest.log')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except:
                logger.warning("ログファイルにアクセスできません。コンソールログのみ使用します。")
                
        return logger

    def initialize_strategy(self) -> None:
        """
        戦略固有の初期化を行う。
        派生クラスで必要に応じてオーバーライドできる。
        """
        self.logger.info(f"{self.__class__.__name__} 初期化: パラメータ = {self.params}")

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        派生クラスが実装する必要がある。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        raise NotImplementedError("派生クラスはgenerate_entry_signalメソッドを実装してください")

    def generate_exit_signal(self, idx: int, entry_idx: int = -1) -> int:
        """
        イグジットシグナルを生成する。
        派生クラスが実装する必要がある。
        
        Parameters:
            idx (int): 現在のインデックス
            entry_idx (int): エントリー時のインデックス（オプション、デフォルト-1）
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        raise NotImplementedError("派生クラスはgenerate_exit_signalメソッドを実装してください")

    def log_trade(self, message: str) -> None:
        """
        取引関連のログメッセージを記録する。
        
        Parameters:
            message (str): ログメッセージ
        """
        self.logger.info(message)
        
    def get_latest_entry_price(self, idx: int) -> Optional[float]:
        """
        指定されたインデックスより前の最新のエントリー価格を取得する。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            Optional[float]: 最新のエントリー価格（存在しない場合はNone）
        """
        entry_signals = self.data[self.data['Entry_Signal'] == 1].index
        previous_entries = [i for i in entry_signals if i < self.data.index[idx]]
        
        if not previous_entries:
            return None
            
        latest_entry_idx = previous_entries[-1]
        return self.data.loc[latest_entry_idx, 'Adj Close']
        
    def backtest(self, trading_start_date: Optional[pd.Timestamp] = None,
                 trading_end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        戦略のバックテストを実行する標準メソッド。
        必要に応じて各戦略でオーバーライドできます。
        
        Args:
            trading_start_date: 取引開始日（この日以降のみシグナル生成）
            trading_end_date: 取引終了日（この日以前のみシグナル生成）
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        # [WARMUP_DEBUG] ウォームアップ期間パラメータ確認
        self.logger.info(
            f"[WARMUP_DEBUG] backtest() called: "
            f"trading_start_date={trading_start_date}, "
            f"trading_end_date={trading_end_date}, "
            f"strategy={self.__class__.__name__}"
        )
        
        # シグナル列の初期化
        result = self.data.copy()  # データのコピーを作成して元のデータに影響を与えない
        result['Entry_Signal'] = 0
        result['Exit_Signal'] = 0
        result['Exit_Reason'] = 'none'  # Task 1実装: エグジット理由列を追加
        result['Position'] = 0  # ポジション管理列を追加（0: なし, 1: ロング）
        
        # 戦略名を追加
        result['Strategy'] = self.__class__.__name__
        
        # [WARMUP_DEBUG] データ範囲確認
        if len(result) > 0:
            self.logger.info(
                f"[WARMUP_DEBUG] Data range: "
                f"start={result.index[0]}, "
                f"end={result.index[-1]}, "
                f"rows={len(result)}"
            )
        
        # インデックスが日時型になっていることを確認
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.DatetimeIndex(result.index)
                self.logger.info("インデックスをDatetimeIndexに変換しました")
            except Exception as e:
                self.logger.warning(f"インデックス変換エラー: {e}")

        in_position = False
        entry_idx = -1
        
        # 価格カラムを決定（派生クラスで指定されている場合はそれを使用）
        price_column = getattr(self, 'price_column', 'Adj Close')
        if price_column not in result.columns:
            price_column = 'Close'
        
        # タイムゾーン統一処理（trading_start_date/trading_end_date指定時）
        # current_date (result.index[idx]) がtz-awareの場合、trading_start_date/trading_end_dateもtz-awareに変換
        trading_start_date_unified = trading_start_date
        trading_end_date_unified = trading_end_date
        
        if len(result) > 0:
            first_date = result.index[0]
            if first_date.tz is not None:
                # データインデックスがtz-awareの場合
                if trading_start_date is not None and trading_start_date.tz is None:
                    trading_start_date_unified = trading_start_date.tz_localize(first_date.tz)
                    self.logger.info(
                        f"[WARMUP_DEBUG] Timezone conversion: "
                        f"trading_start_date {trading_start_date} -> {trading_start_date_unified}"
                    )
                if trading_end_date is not None and trading_end_date.tz is None:
                    trading_end_date_unified = trading_end_date.tz_localize(first_date.tz)
                    self.logger.info(
                        f"[WARMUP_DEBUG] Timezone conversion: "
                        f"trading_end_date {trading_end_date} -> {trading_end_date_unified}"
                    )
            else:
                # データインデックスがtz-naiveの場合
                if trading_start_date is not None and trading_start_date.tz is not None:
                    trading_start_date_unified = trading_start_date.tz_localize(None)
                    self.logger.info(
                        f"[WARMUP_DEBUG] Timezone removal: "
                        f"trading_start_date {trading_start_date} -> {trading_start_date_unified}"
                    )
                if trading_end_date is not None and trading_end_date.tz is not None:
                    trading_end_date_unified = trading_end_date.tz_localize(None)
                    self.logger.info(
                        f"[WARMUP_DEBUG] Timezone removal: "
                        f"trading_end_date {trading_end_date} -> {trading_end_date_unified}"
                    )
        
        # 各日にちについてシグナルを計算
        entry_count = 0
        exit_count = 0
        warmup_filtered_count = 0  # ウォームアップ期間でフィルタされた日数
        
        # [DATA_STRUCTURE_LOG] データ構造検証ログ（2025-12-28 Solution A検証用）
        self.logger.info(
            f"[DATA_STRUCTURE] result shape: {result.shape}, "
            f"dates: {result.index[0]} ~ {result.index[-1]}, "
            f"loop_range: 0 ~ {len(result) - 2} (range(len(result)-1))"
        )
        if trading_start_date_unified is not None:
            try:
                target_date_position = result.index.get_loc(trading_start_date_unified) if trading_start_date_unified in result.index else -1
                self.logger.info(
                    f"[DATA_STRUCTURE] trading_start_date={trading_start_date_unified}, "
                    f"position_in_array={target_date_position}"
                )
            except Exception as e:
                self.logger.warning(f"[DATA_STRUCTURE] target_date position check failed: {e}")
        
        # ルックアヘッドバイアス対策: エントリー価格参照時に翌日始値を使用
        # 日次ウォームアップ方式対応: 最終行もループに含める（2025-12-29修正）
        for idx in range(len(result)):
            current_date = result.index[idx]
            
            # ウォームアップ期間チェック（trading_start_date指定時）
            in_trading_period = True
            if trading_start_date_unified is not None:
                if current_date < trading_start_date_unified:
                    in_trading_period = False
                    warmup_filtered_count += 1
                    if warmup_filtered_count == 1:  # 最初のフィルタ時のみログ出力
                        self.logger.info(
                            f"[WARMUP_FILTER] Warmup period detected: "
                            f"current_date={current_date} < trading_start={trading_start_date_unified}"
                        )
            if trading_end_date_unified is not None:
                if current_date > trading_end_date_unified:
                    in_trading_period = False
                    if idx == len(result) - 1 or result.index[idx + 1] <= trading_end_date_unified:
                        self.logger.info(
                            f"[WARMUP_FILTER] After trading period: "
                            f"current_date={current_date} > trading_end={trading_end_date_unified}"
                        )
            
            # ポジションを持っていない場合のみエントリーシグナルをチェック
            if not in_position and in_trading_period:
                # 【選択肢D無効化】2025-12-28 Solution A実装
                # 理由: Option A（日次方式）により累積期間方式の問題は解消済み
                # in_trading_period内であればエントリー許可（従来のOption D制約を削除）
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    result.at[result.index[idx], 'Entry_Signal'] = 1
                    result.at[result.index[idx], 'Position'] = 1
                    in_position = True
                    entry_idx = idx
                    entry_count += 1
                    
                    # ルックアヘッドバイアス禁止対応（2026-01-22修正）
                    # idx日のシグナル → idx+1日の始値で執行（copilot-instructions.md準拠）
                    # Phase 2統合: スリッページ・取引コスト対応（2025-12-23追加）
                    # デフォルト: slippage=0.001（0.1%）、transaction_cost=0.0（0%）
                    if idx + 1 < len(result):
                        next_open = float(result['Open'].iloc[idx + 1])
                    else:
                        # 最終日の場合は当日終値を使用（例外処理）
                        next_open = float(result['Adj Close'].iloc[idx])
                        self.logger.warning(
                            f"[ENTRY] Final day entry: using Close instead of next Open. "
                            f"idx={idx}, date={result.index[idx]}"
                        )
                    
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = next_open * (1 + slippage + transaction_cost)
                    self.entry_prices[idx] = entry_price
                    
                    # デバッグログ: エントリー記録（Phase 2対応）
                    self.logger.debug(
                        f"[ENTRY #{entry_count}] idx={idx}, date={result.index[idx]}, "
                        f"next_open={next_open:.2f}, entry_price={entry_price:.2f}, "
                        f"slippage+cost={slippage+transaction_cost:.4f}, in_position={in_position}"
                    )
            
            # ポジションを持っている場合のみイグジットシグナルをチェック
            elif in_position:
                # ポジションを前日から引き継ぐ
                if idx > 0:
                    result.at[result.index[idx], 'Position'] = result['Position'].iloc[idx-1]
                
                # 日次ウォームアップ方式対応: 最終日の特別処理（2025-12-29修正）
                if idx + 1 >= len(result):
                    self.logger.debug(
                        f"[EXIT_FINAL_DAY] 最終日のイグジット判定: idx={idx}, date={result.index[idx]}"
                    )
                
                # entry_idxを渡してgenerate_exit_signalを呼び出す
                exit_result = self.generate_exit_signal(idx, entry_idx=entry_idx)
                
                # Task 1実装: タプル返り値対応（signal, reason）
                if isinstance(exit_result, tuple):
                    exit_signal, exit_reason = exit_result
                else:
                    # 後方互換性: int返り値のサポート
                    exit_signal = exit_result
                    exit_reason = 'unknown'
                
                # デバッグログ: イグジット判定
                if exit_signal == -1:
                    exit_count += 1
                    result.at[result.index[idx], 'Exit_Signal'] = -1
                    result.at[result.index[idx], 'Position'] = 0
                    result.at[result.index[idx], 'Exit_Reason'] = exit_reason
                    
                    # デバッグログ: イグジット記録
                    self.logger.debug(f"[EXIT #{exit_count}] idx={idx}, date={result.index[idx]}, exit_signal={exit_signal}, exit_reason={exit_reason}, in_position(before)={in_position}")
                    
                    in_position = False
                    entry_idx = -1
        
        # バックテスト終了時に未決済のポジションがある場合は、最終日に強制決済
        if in_position and entry_idx >= 0:
            last_idx = len(result) - 1
            result.at[result.index[last_idx], 'Exit_Signal'] = -1
            result.at[result.index[last_idx], 'Position'] = 0
            result.at[result.index[last_idx], 'Exit_Reason'] = 'force_close'
            self.logger.info(f"バックテスト終了時のオープンポジションを強制決済: エントリー日={result.index[entry_idx]}, 決済日={result.index[last_idx]}")

        # エントリーとエグジットの回数を検証
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        
        # [Profit_Loss計算] Priority 1実装（2026-01-22）
        # 目的: PF・勝率・シャープレシオ計算可能にする
        # copilot-instructions.md準拠: 実データのみ使用、フォールバック禁止
        result['Profit_Loss'] = 0.0
        result['Entry_Price'] = np.nan
        result['Exit_Price'] = np.nan
        result['Trade_ID'] = 0  # 取引IDを追加（後続分析用）
        
        trade_id = 0
        current_entry_idx = None
        current_entry_price = None
        
        for idx in range(len(result)):
            # エントリー記録
            if result['Entry_Signal'].iloc[idx] == 1:
                trade_id += 1
                current_entry_idx = idx
                current_entry_price = self.entry_prices.get(idx)
                
                if current_entry_price is None:
                    self.logger.error(
                        f"[PL_CALC_ERROR] Entry price not found for idx={idx}, date={result.index[idx]}. "
                        f"This should never happen (copilot-instructions.md violation)."
                    )
                    raise RuntimeError(f"Entry price missing at idx={idx}. This indicates a bug in backtest() logic.")
                
                result.at[result.index[idx], 'Entry_Price'] = current_entry_price
                result.at[result.index[idx], 'Trade_ID'] = trade_id
                
                self.logger.debug(
                    f"[PL_CALC] Trade #{trade_id} Entry: idx={idx}, date={result.index[idx]}, "
                    f"entry_price={current_entry_price:.2f}"
                )
            
            # エグジット記録・損益計算
            if result['Exit_Signal'].iloc[idx] == -1:
                if current_entry_idx is None or current_entry_price is None:
                    self.logger.error(
                        f"[PL_CALC_ERROR] Exit without entry at idx={idx}, date={result.index[idx]}. "
                        f"This should never happen."
                    )
                    raise RuntimeError(f"Exit without entry at idx={idx}. This indicates a bug in backtest() logic.")
                
                # エグジット価格計算（ルックアヘッドバイアス防止: 翌日始値）
                # Phase 1b修正: idx日目のシグナルでidx+1日目の始値で執行
                if idx + 1 < len(result):
                    exit_price = float(result['Open'].iloc[idx + 1])
                else:
                    # 最終日の場合は終値を使用（例外処理）
                    exit_price = float(result['Adj Close'].iloc[idx])
                    self.logger.warning(
                        f"[PL_CALC] Final day exit: using Close instead of next Open. "
                        f"idx={idx}, date={result.index[idx]}, exit_price={exit_price:.2f}"
                    )
                
                # Phase 2統合: スリッページ・取引コスト対応（エグジット時）
                slippage = self.params.get("slippage", 0.001)
                transaction_cost = self.params.get("transaction_cost", 0.0)
                exit_price_adjusted = exit_price * (1 - slippage - transaction_cost)
                
                # 損益計算
                profit_loss = exit_price_adjusted - current_entry_price
                result.at[result.index[idx], 'Exit_Price'] = exit_price_adjusted
                result.at[result.index[idx], 'Profit_Loss'] = profit_loss
                result.at[result.index[idx], 'Trade_ID'] = trade_id
                
                self.logger.debug(
                    f"[PL_CALC] Trade #{trade_id} Exit: idx={idx}, date={result.index[idx]}, "
                    f"exit_price={exit_price:.2f}, exit_price_adjusted={exit_price_adjusted:.2f}, "
                    f"profit_loss={profit_loss:.2f} (entry={current_entry_price:.2f})"
                )
                
                # リセット
                current_entry_idx = None
                current_entry_price = None
        
        # [Profit_Loss統計] バックテスト結果サマリーに追加
        total_profit = result[result['Profit_Loss'] > 0]['Profit_Loss'].sum()
        total_loss = abs(result[result['Profit_Loss'] < 0]['Profit_Loss'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        win_count = (result['Profit_Loss'] > 0).sum()
        loss_count = (result['Profit_Loss'] < 0).sum()
        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0.0
        
        self.logger.info(
            f"[WARMUP_SUMMARY] Backtest completed: "
            f"strategy={self.__class__.__name__}, "
            f"total_rows={len(result)}, "
            f"warmup_filtered={warmup_filtered_count}, "
            f"trading_rows={len(result) - warmup_filtered_count}, "
            f"entry_signals={entry_count}, "
            f"exit_signals={exit_count}"
        )
        
        self.logger.info(
            f"[PL_SUMMARY] Profit Factor={profit_factor:.2f}, Win Rate={win_rate:.1%} "
            f"({win_count}W/{loss_count}L), Total Profit={total_profit:.2f}, Total Loss={total_loss:.2f}"
        )
        
        if entry_count != exit_count:
            self.logger.warning(f"エントリー ({entry_count}) とエグジット ({exit_count}) の回数が一致しません！")
        
        # データを更新（派生クラスのgenerate_exit_signalがself.dataを参照する場合に備えて）
        self.data = result
            
        return result

    def backtest_daily(self, current_date, stock_data: pd.DataFrame, 
                      existing_position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        日次バックテスト実行（Phase 3-A MVP版）
        
        リアルトレード対応の日次判定システム。その日のみを対象とし、
        既存ポジション情報を考慮してエントリー/エグジット判定を行う。
        
        Args:
            current_date: 判定対象日（datetime）
            stock_data: current_dateまでのデータ（ウォームアップ含む）
            existing_position: 既存のポジション情報（銘柄切替時に使用）
                {
                    'symbol': str,           # 保有銘柄コード
                    'quantity': int,         # 保有株数
                    'entry_price': float,    # エントリー価格
                    'entry_date': datetime   # エントリー日
                }
        
        Returns:
            {
                'action': 'entry'|'exit'|'hold',  # 実行アクション
                'signal': 1|-1|0,                 # シグナル値（1:買い、-1:売り、0:何もしない）
                'price': float,                   # 実行価格（翌日始値想定）
                'shares': int,                    # 取引株数
                'reason': str                     # 判定理由
            }
        """
        from datetime import datetime, timedelta
        
        # Phase 3-A MVP実装: 既存backtest()をラップして新インターフェースを提供
        # 後方互換性を保ちながら、段階的に日次対応ロジックに移行
        
        try:
            # current_dateの型確認・変換
            if isinstance(current_date, str):
                current_date = pd.Timestamp(current_date)
            elif not isinstance(current_date, (pd.Timestamp, datetime)):
                current_date = pd.Timestamp(current_date)
            
            # 一時的にself.dataを更新（既存backtest()との互換性のため）
            # Option B実装: インジケーター保持しつつ基本データのみ更新
            original_data = self.data.copy()
            
            # 基本データ（価格・ボリューム）のみ更新、インジケーター保持
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    # インデックスが一致する部分のみ安全に更新
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            # デバッグログ: データ更新状況
            self.logger.debug(
                f"[backtest_daily] Option B data update: "
                f"updated_columns={updated_columns}, "
                f"common_dates={len(self.data.index.intersection(stock_data.index)) if len(updated_columns) > 0 else 0}"
            )
            
            # 単日バックテスト実行（current_date のみを対象）
            trading_start_date = current_date
            trading_end_date = current_date + timedelta(days=1)  # 翌日まで（翌日始値取得のため）
            
            # ログ記録
            self.logger.debug(
                f"[backtest_daily] {self.__class__.__name__}: "
                f"current_date={current_date.strftime('%Y-%m-%d')}, "
                f"existing_position={existing_position is not None}"
            )
            
            # 既存backtest()メソッドを呼び出し
            result_df = self.backtest(trading_start_date, trading_end_date)
            
            # current_dateのエントリー/エグジットシグナルを確認
            current_date_data = result_df[result_df.index.date == current_date.date()]
            
            if len(current_date_data) == 0:
                # 対象日のデータが存在しない場合
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'No data available for {current_date.strftime("%Y-%m-%d")}'
                }
            
            current_row = current_date_data.iloc[0]
            entry_signal = current_row.get('Entry_Signal', 0)
            exit_signal = current_row.get('Exit_Signal', 0)
            
            # アクション決定ロジック
            if existing_position is not None and exit_signal == -1:
                # 既存ポジションありでエグジットシグナル発生
                try:
                    # 翌日始値を取得（ルックアヘッドバイアス防止）
                    next_day_idx = result_df.index.get_loc(current_row.name) + 1
                    if next_day_idx < len(result_df):
                        exit_price = result_df.iloc[next_day_idx]['Open']
                    else:
                        exit_price = current_row.get('Close', 0.0)  # フォールバック（最終日の場合）
                except:
                    exit_price = current_row.get('Close', 0.0)
                
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'{self.__class__.__name__}: Exit signal detected'
                }
                
            elif existing_position is None and entry_signal == 1:
                # ポジションなしでエントリーシグナル発生
                try:
                    # 翌日始値を取得（ルックアヘッドバイアス防止）
                    next_day_idx = result_df.index.get_loc(current_row.name) + 1
                    if next_day_idx < len(result_df):
                        entry_price = result_df.iloc[next_day_idx]['Open']
                        # スリッページ考慮（copilot-instructions.md推奨0.1%）
                        slippage = 0.001
                        entry_price = entry_price * (1 + slippage)
                    else:
                        entry_price = current_row.get('Close', 0.0)  # フォールバック
                except:
                    entry_price = current_row.get('Close', 0.0)
                
                # 標準的な取引株数計算（資金の10%程度を想定）
                shares = int(100000 / entry_price) if entry_price > 0 else 0  # 暫定実装
                
                return {
                    'action': 'entry',
                    'signal': 1,
                    'price': float(entry_price),
                    'shares': shares,
                    'reason': f'{self.__class__.__name__}: Entry signal detected'
                }
            
            else:
                # シグナルなし、または既存ポジションと不一致
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'{self.__class__.__name__}: No action required (entry={entry_signal}, exit={exit_signal}, has_position={existing_position is not None})'
                }
        
        except Exception as e:
            self.logger.error(f"backtest_daily error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Error: {str(e)}'
            }
        
        finally:
            # 元のデータを復元
            if 'original_data' in locals():
                self.data = original_data
