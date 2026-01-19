"""
Module: gc_strategy_signal
File: gc_strategy_signal.py
Description: 
  移動平均線のゴールデンクロス（短期線が長期線を上抜け）とデッドクロス（短期線が長期線を下抜け）を
  検出して取引シグナルを生成する戦略を実装しています。上昇トレンドの確認と合わせて使用することで
  精度を高め、適切な利確・損切り条件も設定しています。

Author: kouseven7
Created: 2023-02-25
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.trend_analysis
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from indicators.trend_analysis import detect_trend
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend, detect_unified_trend_with_confidence

class GCStrategy(BaseStrategy):
    """
    GC戦略（ゴールデンクロス戦略）の実装クラス。
    短期移動平均と長期移動平均のゴールデンクロス／デッドクロスを基にエントリー／イグジットシグナルを生成し、
    Excelから取得した戦略パラメータ（例: 利益確定％、損切割合％、短期・長期移動平均期間）を反映させます。
    """
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close", ticker: str = None):
        """
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（例: {"short_window": 5, "long_window": 25, ...}）
            price_column (str): インジケーター計算に使用する価格カラム（デフォルトは "Adj Close"）
            ticker (str, optional): 銘柄コード（最適化パラメータ読み込み用）
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.ticker = ticker  # 銘柄コードを保存（最適化パラメータ用）
        self.entry_prices = {}  # エントリー価格を記録する辞書
        self.high_prices = {}  # トレーリングストップ用の最高価格を記録する辞書
        
        # 指定された価格カラムが存在するか確認、なければ 'Close' を代用
        if self.price_column not in data.columns:
            price_column = "Close"
            self.price_column = price_column
        
        # デフォルトパラメータの設定（Phase 2最終推奨パラメータ: 2026-01-18最適化完了）
        default_params = {
            "short_window": 5,       # 短期移動平均期間
            "long_window": 25,       # 長期移動平均期間
            "take_profit": 0.15,     # 利益確定（15%）← Phase 2最終推奨（Phase 1/2最適化完了）
            "stop_loss": 0.03,       # ストップロス（3%）← Phase 2-1最適化完了（2%は悪化、5%は横ばい）
            "trailing_stop_pct": 0.05,  # トレーリングストップ（5%）← Phase 2最終推奨（Phase 1最適化完了）
            "max_hold_days": 300,    # 最大保有期間（300日、実質無効化）
            "exit_on_death_cross": True,  # デッドクロスでイグジットするかどうか
            
            # トレンドフィルター設定
            "trend_filter_enabled": False,  # デフォルトは無効（マルチ戦略システムで既にフィルタリング済み）  # 統一トレンド判定の有効化
            "allowed_trends": ["uptrend"]  # 許可するトレンド（上昇トレンド）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # 辞書を初期化（再実行時のクリーンアップ）
        self.entry_prices = {}
        self.high_prices = {}
        
        # 戦略パラメータの読み込み
        self.short_window = int(self.params.get("short_window", 5))
        self.long_window = int(self.params.get("long_window", 25))
        
        # Phase 1c修正: 移動平均線にshift(1)を適用（ルックアヘッドバイアス修正）
        # 理由: idx日目の移動平均がidx日目の価格を含むのはルックアヘッドバイアス
        # リアルトレードでは前日までのデータで当日の判断を行う
        # 移動平均線の計算（存在しない場合のみ）
        if f"SMA_{self.short_window}" not in self.data.columns:
            self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean().shift(1)
        if f"SMA_{self.long_window}" not in self.data.columns:
            self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean().shift(1)
        
        # 統一トレンド検出器の初期化
        # 最新時点でのトレンド判定をコンソールに出力
        if len(self.data) > 0:
            try:
                trend, confidence = detect_unified_trend_with_confidence(
                    self.data, self.price_column, strategy="Golden_Cross"
                )
                self.logger.info(f"現在のトレンド: {trend}, 信頼度: {confidence:.1%}")
            except Exception as e:
                self.logger.warning(f"トレンド判定エラー: {e}")
        
        self.logger.info(
            f"GCStrategy initialized with short_window={self.short_window}, long_window={self.long_window}, "
            f"take_profit_pct={self.params.get('take_profit_pct', self.params.get('take_profit', 0.05))}, "
            f"stop_loss_pct={self.params.get('stop_loss_pct', self.params.get('stop_loss', 0.03))}"
        )
        
        # 移動平均は上記（Lines 82-85）で既に計算済み（存在しない場合のみ）
        # 重複計算を削除し、GC_Signalのみ追加計算
        
        # ベクトル化操作: ゴールデンクロスシグナル
        self.data['GC_Signal'] = np.where(
            (self.data[f'SMA_{self.short_window}'] > self.data[f'SMA_{self.long_window}']) & 
            (self.data[f'SMA_{self.short_window}'].shift(1) <= self.data[f'SMA_{self.long_window}'].shift(1)),
            1, 0
        )

    def generate_entry_signal(self, idx: int) -> int:
        """
        指定されたインデックス位置でのエントリーシグナルを生成する。
        短期移動平均が長期移動平均を上回り、かつトレンドが上昇トレンドの場合、1を返す。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        if idx < self.long_window:  # 長期移動平均の計算に必要な日数分のデータがない場合
            return 0
            
        # トレンド確認（統一トレンド判定を使用）
        use_trend_filter = self.params.get("trend_filter_enabled", False)
        if use_trend_filter:
            trend = detect_unified_trend(
                self.data.iloc[:idx + 1], 
                self.price_column, 
                strategy="Golden_Cross",
                method="combined"  # 複合メソッドを使用
            )
            allowed_trends = self.params.get("allowed_trends", ["uptrend"])
            # 許可されたトレンドでのみエントリー
            if trend not in allowed_trends:
                return 0  # トレンド不適合
        
        short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
        long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
        
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0

        # 前日のSMA値を取得してクロス判定
        prev_short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx-1]
        prev_long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx-1]
        
        # Task 2実装 (2026-01-16): トレンド継続中のエントリー条件緩和
        # 目的: 11ヶ月3取引 → エントリー機会増加、利益向上
        # 背景: DSSMSがパーフェクトオーダー銘柄を選択 → 既にGC済み → エントリー不可
        # 対策: トレンド継続中（両MA上昇中）でもエントリー許可
        
        # 従来のゴールデンクロス（短期MAが長期MAを下から上に抜けた）
        golden_cross = short_sma > long_sma and prev_short_sma <= prev_long_sma
        
        # トレンド継続中のエントリー条件（新規追加）
        # 条件: 短期MA > 長期MA かつ 両MA上昇中
        uptrend_continuation = (
            short_sma > long_sma and 
            short_sma > prev_short_sma and  # 短期MA上昇中
            long_sma > prev_long_sma         # 長期MA上昇中
        )
        
        # 緩和後のエントリー条件（ゴールデンクロス or トレンド継続中）
        entry_signal = golden_cross or uptrend_continuation

        if entry_signal:
            return 1
            
        return 0

    def generate_exit_signal(self, idx: int, entry_idx: int = -1) -> int:
        """
        イグジットシグナルを生成する
        
        Parameters:
            idx (int): 現在のインデックス
            entry_idx (int): エントリー時のインデックス（BaseStrategyから渡される）
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < self.params["long_window"]:
            return 0
        
        # entry_idxが渡されていない場合はシグナルを返さない
        # （BaseStrategy.backtest()は必ずentry_idxを渡すため、ここには来ない）
        if entry_idx < 0:
            self.logger.debug(f"[EXIT CHECK] idx={idx}, entry_idx={entry_idx} (< 0), returning 0")
            return 0
        
        # エントリー価格を取得
        entry_price = self.entry_prices.get(entry_idx)
        
        # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でイグジット
        current_price = float(self.data['Open'].iloc[idx + 1])
        
        # デバッグログ: 価格情報
        self.logger.debug(f"[EXIT CHECK] idx={idx}, entry_idx={entry_idx}, entry_price={entry_price}, current_price={current_price:.2f} (next_day_open)")
        
        # entry_priceがNoneの場合はエラー（フォールバック禁止）
        if entry_price is None:
            error_msg = f"CRITICAL ERROR: エントリー価格がNoneです。entry_idx={entry_idx}, idx={idx}, date={self.data.index[idx]}"
            self.logger.error(error_msg)
            self.logger.error(f"  entry_prices辞書の内容: {self.entry_prices}")
            self.logger.error(f"  BaseStrategy.backtest()がentry_idxでエントリー価格を記録していない可能性があります")
            raise ValueError(error_msg)
                
        # 1. デッドクロスでイグジット（オプション）
        if self.params.get("exit_on_death_cross", True):
            short_ma = self.data[f'SMA_{self.params["short_window"]}'].iloc[idx]
            long_ma = self.data[f'SMA_{self.params["long_window"]}'].iloc[idx]
            prev_short_ma = self.data[f'SMA_{self.params["short_window"]}'].iloc[idx-1]
            prev_long_ma = self.data[f'SMA_{self.params["long_window"]}'].iloc[idx-1]
        
            # デッドクロス（短期MAが長期MAを下回る）
            if prev_short_ma >= prev_long_ma and short_ma < long_ma:
                self.logger.info(f"デッドクロスによるイグジット: 日付={self.data.index[idx]}")
                self.logger.debug(f"[EXIT REASON] Death Cross: prev_short={prev_short_ma:.2f}, prev_long={prev_long_ma:.2f}, short={short_ma:.2f}, long={long_ma:.2f}")
                return -1
    
        # 2. トレーリングストップ
        if entry_idx not in self.high_prices:
            self.high_prices[entry_idx] = entry_price
        else:
            self.high_prices[entry_idx] = max(self.high_prices[entry_idx], current_price)
    
        trailing_stop = self.high_prices[entry_idx] * (1 - self.params.get("trailing_stop_pct", 0.03))
        self.logger.debug(f"[TRAILING] high_price={self.high_prices[entry_idx]:.2f}, trailing_stop={trailing_stop:.2f}, current_price={current_price:.2f} (next_day_open)")
        
        if current_price < trailing_stop:
            self.logger.info(f"トレーリングストップによるイグジット: 日付={self.data.index[idx]}")
            self.logger.debug(f"[EXIT REASON] Trailing Stop: {current_price:.2f} (next_day_open) < {trailing_stop:.2f}")
            return -1
    
        # 3. 利益確定
        take_profit_price = entry_price * (1 + self.params.get("take_profit", 0.05))
        if current_price >= take_profit_price:
            self.logger.info(f"利益確定によるイグジット: 日付={self.data.index[idx]}")
            self.logger.debug(f"[EXIT REASON] Take Profit: {current_price:.2f} (next_day_open) >= {take_profit_price:.2f}")
            return -1
    
        # 4. 損切り
        stop_loss_price = entry_price * (1 - self.params.get("stop_loss", 0.03))
        if current_price <= stop_loss_price:
            self.logger.info(f"損切りによるイグジット: 日付={self.data.index[idx]}")
            self.logger.debug(f"[EXIT REASON] Stop Loss: {current_price:.2f} (next_day_open) <= {stop_loss_price:.2f}")
            return -1
    
        # 5. 最大保有期間
        days_held = idx - entry_idx
        if days_held >= self.params.get("max_hold_days", 20):
            self.logger.info(f"最大保有期間によるイグジット: 日付={self.data.index[idx]}")
            return -1
    
        return 0

    def load_optimized_parameters(self) -> bool:
        """
        最適化されたパラメータを読み込み
        Returns:
            bool: 読み込み成功
        """
        try:
            from config.optimized_parameters import OptimizedParameterManager
            manager = OptimizedParameterManager()
            ticker = getattr(self, 'ticker', 'DEFAULT')
            # GC戦略用の承認済みパラメータ取得
            params = manager.get_best_config_by_metric('GCStrategy', metric='sharpe_ratio', ticker=ticker, status='approved')
            if params and 'parameters' in params:
                self.params.update(params['parameters'])
                self._approved_params = params
                print(f"[OK] 最適化パラメータを読み込みました (ID: {params.get('parameter_id', 'N/A')})")
                return True
            else:
                print(f"[WARNING] 承認済みの最適化パラメータが見つかりません")
                return False
        except Exception as e:
            print(f"[ERROR] 最適化パラメータの読み込みでエラー: {e}")
            return False

    def run_optimized_strategy(self):
        """
        最適化パラメータで戦略を実行
        Returns:
            pd.DataFrame: バックテスト結果
        """
        loaded = self.load_optimized_parameters()
        if loaded and hasattr(self, '_approved_params'):
            print(f"\n[CHART] 使用パラメータ: {self._approved_params.get('parameters', {})}")
            print(f"   作成日時: {self._approved_params.get('created_at', 'N/A')}")
            print(f"   シャープレシオ: {self._approved_params.get('performance_metrics', {}).get('sharpe_ratio', 'N/A')}")
        else:
            print(f"[CHART] デフォルトパラメータを使用: {self.params}")
        return self.backtest()

    def get_optimization_info(self):
        """
        最適化情報を取得
        Returns:
            dict: 最適化情報
        """
        info = {
            'using_optimized_params': hasattr(self, '_approved_params') and self._approved_params is not None,
            'default_params': {
                "short_window": 5,
                "long_window": 25,
                "take_profit": 0.05,
                "stop_loss": 0.03,
                "trailing_stop_pct": 0.03,
                "max_hold_days": 20,
                "exit_on_death_cross": True
            },
            'current_params': self.params
        }
        if hasattr(self, '_approved_params') and self._approved_params:
            info['optimized_params'] = self._approved_params
        return info

    def backtest_daily(self, current_date, stock_data: pd.DataFrame, existing_position=None, **kwargs):
        """
        日次バックテスト実行（Phase 3-C Day 11実装）
        
        GCStrategy専用のbacktest_daily()実装。templates/backtest_daily_template.pyパターンを活用。
        
        Cycle 26修正: **kwargs追加
        - 理由: force_close時にentry_symbol_dataがkwargsで渡される（Cycle 7修正）
        
        Cycle 27修正: entry_symbol_data使用
        - force_close時はentry_symbol_data（元の銘柄）でエグジット価格を取得
        - 問題: 全跨銘柄取引で切替先の価格が入っていた（5202→2768: 3325円 vs 実際383円）
        
        Parameters:
            current_date: 判定対象日（datetime/pd.Timestamp/str）
            stock_data: current_dateまでのデータ（ウォームアップ含む）
            existing_position: 既存のポジション情報（銘柄切替時に使用）
                {
                    'symbol': str,           # 保有銘柄コード
                    'quantity': int,         # 保有株数
                    'entry_price': float,    # エントリー価格
                    'entry_date': datetime,  # エントリー日
                    'entry_idx': int         # エントリー時のインデックス（オプション）
                }
        
        Returns:
            {
                'action': 'entry'|'exit'|'hold',  # 実行アクション
                'signal': 1|-1|0,                 # シグナル値（1:買い、-1:売り、0:何もしない）
                'price': float,                   # 実行価格（翌日始値想定）
                'shares': int,                    # 取引株数
                'reason': str                     # 判定理由
            }
        
        Note:
            - copilot-instructions.md準拠（ルックアヘッドバイアス防止）
            - 既存generate_entry_signal/exit_signal活用
            - Entry_Signal依存なし（翌日始値対応済み）
        """
        print(f"[GC_DEBUG] backtest_daily() called: current_date={current_date}, stock_data.shape={stock_data.shape}")
        
        # Phase 1: current_dateの型変換・検証
        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        elif not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        print(f"[GC_DEBUG] Phase 1 complete: current_date={current_date}, type={type(current_date)}")
        
        # Cycle 23修正: Breakout.py Cycle 20修正を参考にタイムゾーン統一
        if current_date.tz is not None:
            current_date = current_date.tz_localize(None)
            print(f"[GC_DEBUG] current_date tz-naive conversion完了")
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)
            print(f"[GC_DEBUG] stock_data.index tz-naive conversion完了")
        
        # Phase 2: データ整合性チェック
        print(f"[GC_DEBUG] Phase 2 check: current_date={current_date}, current_date in index={current_date in stock_data.index}")
        
        if current_date not in stock_data.index:
            print(f"[GC_DEBUG] Phase 2 early return: current_date NOT in index")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'GCStrategy: No data available for {current_date.strftime("%Y-%m-%d")}'
            }
        
        print(f"[GC_DEBUG] Phase 2 passed, proceeding to Phase 3")
        
        # Phase 3: ウォームアップ期間考慮
        # Cycle 23修正: Breakout.py Cycle 19修正を参考に、戦略固有の最小期間のみを要求
        # 理由: DSSMSがwarmup_days=150で既にデータ拡大しているため、戦略側で150日要求は不要
        current_idx = stock_data.index.get_loc(current_date)
        min_required = self.long_window  # GCStrategyはlong_window=25のみ必要
        
        print(f"[GC_WARMUP_DEBUG] current_idx={current_idx}, min_required={min_required}, long_window={self.long_window}")
        
        if current_idx < min_required:
            print(f"[GC_WARMUP_DEBUG] INSUFFICIENT DATA: current_idx={current_idx} < min_required={min_required}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'GCStrategy: Insufficient warmup data. Required: {min_required}, Available: {current_idx}'
            }
        
        # Phase 4: データ更新（Option B方式）
        original_data = self.data.copy()
        
        try:
            # BaseStrategy.backtest_daily()の Option B ロジックを活用
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            self.logger.debug(f"[GCStrategy.backtest_daily] Data updated: {updated_columns}")
            
            # SMAカラムの更新（必要に応じて）
            if f"SMA_{self.short_window}" not in self.data.columns or f"SMA_{self.long_window}" not in self.data.columns:
                self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean().shift(1)
                self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean().shift(1)
                self.logger.debug(f"[GCStrategy.backtest_daily] SMA columns recalculated")
            
            # Phase 5: 既存ポジション処理分岐
            print(f"[GC_DEBUG] Phase 5: existing_position={existing_position is not None}")
            
            # Cycle 27修正: entry_symbol_dataをkwargsから取得
            entry_symbol_data = kwargs.get('entry_symbol_data', None)
            
            if existing_position is not None:
                # 既存ポジションあり: エグジット判定
                print(f"[GC_DEBUG] Calling _handle_exit_logic_daily, entry_symbol_data={entry_symbol_data is not None}")
                return self._handle_exit_logic_daily(current_idx, existing_position, stock_data, current_date, entry_symbol_data)
            else:
                # 既存ポジションなし: エントリー判定
                print(f"[GC_DEBUG] Calling _handle_entry_logic_daily")
                return self._handle_entry_logic_daily(current_idx, stock_data, current_date)
        
        finally:
            # データの復元
            self.data = original_data

    def _handle_exit_logic_daily(self, current_idx: int, existing_position: dict, 
                                  stock_data: pd.DataFrame, current_date: pd.Timestamp,
                                  entry_symbol_data: pd.DataFrame = None):
        """
        エグジット判定ロジック（GCStrategy専用）
        
        既存のgenerate_exit_signal()を活用。Entry_Signal依存なし。
        
        Cycle 27修正: entry_symbol_data対応
        - force_close時（entry_symbol_data提供時）は元の銘柄のデータでエグジット価格を取得
        - 通常時は現在の銘柄（stock_data）でエグジット価格を取得
        
        Parameters:
            entry_symbol_data: force_close時の元の銘柄データ（オプション）
        
        Returns:
            dict: {'action': 'exit'|'hold', 'signal': -1|0, 'price': float, 'shares': int, 'reason': str}
        """
        try:
            # entry_idxを取得（existing_positionから、またはcurrent_idxをフォールバック）
            entry_idx = existing_position.get('entry_idx', current_idx)
            
            # force_closeフラグ確認
            is_force_close = existing_position.get('force_close', False)
            
            # Cycle 27修正: force_close時はentry_symbol_dataを使用
            if is_force_close and entry_symbol_data is not None:
                data_for_exit = entry_symbol_data
                self.logger.info(f"[GC_EXIT] force_close=True: entry_symbol_dataを使用（{len(entry_symbol_data)}行）")
            else:
                data_for_exit = stock_data
                self.logger.debug(f"[GC_EXIT] force_close={is_force_close}: stock_dataを使用")
            
            # entry_pricesとhigh_pricesを準備（generate_exit_signalが依存）
            if entry_idx not in self.entry_prices:
                self.entry_prices[entry_idx] = existing_position.get('entry_price', data_for_exit.iloc[current_idx]['Close'])
            
            # generate_exit_signal呼び出し（既存実装活用）
            exit_signal = self.generate_exit_signal(current_idx, entry_idx)
            
            if exit_signal == -1:
                # エグジットシグナル発生
                # Cycle 27修正: data_for_exitからエグジット価格を取得
                if current_idx + 1 < len(data_for_exit):
                    exit_price = data_for_exit.iloc[current_idx + 1]['Open']
                else:
                    # 最終日の場合
                    exit_price = data_for_exit.iloc[current_idx]['Close']
                    self.logger.warning(f"[GCStrategy] Using Close price fallback for final day: {current_date}")
                
                self.logger.info(f"[GC_EXIT] exit_price={exit_price}, source={'entry_symbol_data' if is_force_close and entry_symbol_data is not None else 'stock_data'}")
                
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'GCStrategy: Exit signal detected on {current_date.strftime("%Y-%m-%d")}'
                }
            else:
                # エグジットシグナルなし: ホールド
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'GCStrategy: Holding position from {current_date.strftime("%Y-%m-%d")}'
                }
        
        except Exception as e:
            self.logger.error(f"[GCStrategy] Exit logic error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'GCStrategy: Exit logic error: {str(e)}'
            }

    def _handle_entry_logic_daily(self, current_idx: int, stock_data: pd.DataFrame, 
                                   current_date: pd.Timestamp):
        """
        エントリー判定ロジック（GCStrategy専用）
        
        既存のgenerate_entry_signal()を活用。
        
        Returns:
            dict: {'action': 'entry'|'hold', 'signal': 1|0, 'price': float, 'shares': int, 'reason': str}
        """
        try:
            # Cycle 23デバッグ: generate_entry_signal呼び出し前の状態確認
            short_sma = self.data[f"SMA_{self.short_window}"].iloc[current_idx]
            long_sma = self.data[f"SMA_{self.long_window}"].iloc[current_idx]
            self.logger.info(f"[GC_ENTRY_DEBUG] current_idx={current_idx}, date={current_date}, "
                           f"short_sma={short_sma:.2f}, long_sma={long_sma:.2f}")
            
            # generate_entry_signal呼び出し（既存実装活用）
            entry_signal = self.generate_entry_signal(current_idx)
            
            self.logger.info(f"[GC_ENTRY_DEBUG] entry_signal={entry_signal}")
            
            if entry_signal == 1:
                # エントリーシグナル発生
                if current_idx + 1 < len(stock_data):
                    entry_price = stock_data.iloc[current_idx + 1]['Open']
                    
                    # スリッページ・取引コスト適用
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = entry_price * (1 + slippage + transaction_cost)
                    
                    # ポジションサイズ計算
                    shares = self._calculate_position_size_daily(entry_price)
                    
                    return {
                        'action': 'entry',
                        'signal': 1,
                        'price': float(entry_price),
                        'shares': shares,
                        'reason': f'GCStrategy: Golden Cross entry signal on {current_date.strftime("%Y-%m-%d")}'
                    }
                else:
                    # 最終日の場合（エントリー不可）
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': f'GCStrategy: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                    }
            else:
                # エントリーシグナルなし: ホールド
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'GCStrategy: No golden cross signal on {current_date.strftime("%Y-%m-%d")}'
                }
        
        except Exception as e:
            self.logger.error(f"[GCStrategy] Entry logic error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'GCStrategy: Entry logic error: {str(e)}'
            }

    def _calculate_position_size_daily(self, entry_price: float) -> int:
        """
        ポジションサイズ計算（GCStrategy専用）
        
        固定金額方式（10万円相当）、100株単位。
        
        Returns:
            int: 取引株数
        """
        target_amount = self.params.get("position_amount", 100000)  # 10万円相当
        
        if entry_price > 0:
            shares = int(target_amount / entry_price)
            # 最小単位調整（100株単位）
            shares = max(100, shares // 100 * 100)
            return shares
        else:
            return 0

# テストコード
if __name__ == "__main__":
    import numpy as np
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'Adj Close': np.random.random(100) * 100
    }, index=dates)

    # GC戦略の実行
    strategy = GCStrategy(df)

# optimization/configs/gc_strategy_optimization.py
"""
GC戦略の最適化設定ファイル
"""

# GC戦略の最適化パラメータ
PARAM_GRID = {
    "short_window": [5, 10, 15, 20],           # 短期移動平均期間
    "long_window": [25, 50, 100, 200],         # 長期移動平均期間
    "take_profit": [0.03, 0.05, 0.08, 0.1],    # 利益確定レベル
    "stop_loss": [0.02, 0.03, 0.05],           # ストップロスレベル
    "trailing_stop_pct": [0.02, 0.03, 0.05],   # トレーリングストップの割合
    "max_hold_days": [10, 15, 20, 30],         # 最大保有期間
    "exit_on_death_cross": [True, False],      # デッドクロスでイグジット
    "confirmation_days": [1, 2, 3],            # クロス確認日数
    "ma_type": ["SMA", "EMA"],                 # 移動平均の種類
}

# パラメータの説明
PARAM_DESCRIPTIONS = {
    "short_window": "短期移動平均の期間 - 小さいほど反応が早い",
    "long_window": "長期移動平均の期間 - 大きいほどトレンドを捉える",
    "take_profit": "利益確定レベル - エントリー価格からの上昇率",
    "stop_loss": "ストップロスレベル - エントリー価格からの下落率",
    "trailing_stop_pct": "トレーリングストップの割合 - 高値からの下落率",
    "max_hold_days": "最大保有期間 - この日数を超えると強制イグジット",
    "exit_on_death_cross": "デッドクロス発生時にイグジットするかどうか",
    "confirmation_days": "ゴールデンクロス後、確認する日数",
    "ma_type": "移動平均の種類（SMA: 単純移動平均、EMA: 指数移動平均）",
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "win_rate", "weight": 0.6},
    {"name": "risk_adjusted_return", "weight": 0.7}
]