"""
Module: contrarian_strategy
File: contrarian_strategy.py
Description: 
  過度な売られ場面で反発を狙う逆張り戦略を実装しています。
  RSIの過売り状態やギャップダウン、ピンバー形成などの反転サインを検出し、
  レンジ相場でこれらの条件が揃った際にエントリーします。短期の利食いと
  適切な損切り設定で勝率とリスクリワード比の向上を図ります。

Author: kouseven7
Created: 2023-04-10
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.trend_analysis
  - config.optimized_parameters
  - validation.validators.contrarian_validator
"""

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_rsi
from indicators.trend_analysis import detect_trend
from config.optimized_parameters import OptimizedParameterManager
from validation.validators.contrarian_validator import ContrarianParameterValidator
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend, detect_unified_trend_with_confidence

class ContrarianStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close"):
        """
        逆張り戦略の初期化。
        """
        self.price_column = price_column
        self.entry_prices = {}  # エントリー価格を記録
        self.high_prices = {}   # トレーリングストップ用の最高値を記録

        # デフォルトパラメータ
        default_params = {
            "rsi_period": 14,        # RSI計算期間
            "rsi_oversold": 30,      # RSI過売り閾値
            "gap_threshold": 0.02,   # ギャップダウン閾値（A: 5%→2%に緩和）
            "stop_loss": 0.04,       # ストップロス
            "take_profit": 0.05,     # 利益確定
            "pin_bar_ratio": 2.0,    # ピンバー判定比率
            "max_hold_days": 5,      # 最大保有日数
            "rsi_exit_level": 50,    # RSI中立域でのイグジット
            "trailing_stop_pct": 0.02,  # トレーリングストップ率
            
            # トレンドフィルター設定（Option 4: レンジ相場のみに戻す）
            "trend_filter_enabled": True,  # トレンドフィルター有効化
            "allowed_trends": ["range-bound"],  # レンジ相場のみ許可
            
            # Phase 2: スリッページ・取引コスト（2025-12-23追加）
            "slippage": 0.001,         # スリッページ（0.1%、買い注文は不利な方向）
            "transaction_cost": 0.0    # 取引コスト（0%、オプション）
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)

    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        # RSIを計算してデータに追加
        # ルックアヘッドバイアス修正: shift(1)を追加して前日のRSIを使用
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], period=self.params["rsi_period"]).shift(1)
        
        # Openカラムの確認（ピンバー判定に必要）
        if 'Open' not in self.data.columns:
            raise ValueError("ピンバー判定にはOpenカラムが必要です")
        
        # 統一トレンド判定の初期結果を表示（データがある場合）
        if len(self.data) > 20:  # データが十分ある場合のみ
            try:
                # 現在のトレンドを判定
                trend = detect_unified_trend(
                    self.data,
                    price_column=self.price_column,
                    strategy="contrarian_strategy",
                    method="combined"
                )
                print(f"初期トレンド判定: {trend} (contrarian_strategy)")
                
                # 信頼度付き判定
                trend_detector = UnifiedTrendDetector(
                    self.data,
                    price_column=self.price_column,
                    strategy_name="contrarian_strategy",
                    method="combined"
                )
                _, confidence = trend_detector.detect_trend_with_confidence()
                print(f"トレンド判定信頼度: {confidence:.2f}")
            except Exception as e:
                print(f"トレンド判定初期化エラー: {e}")

    def _is_valid_pinbar(self, high: float, low: float, close: float, open_price: float) -> bool:
        """
        改善版ピンバー判定（案B - 緩和版）
        
        条件:
        1. 下ヒゲの最小値チェック（1.0円以上）← 2.0円から緩和
        2. 上ヒゲが下ヒゲの2倍以上
        3. 上ヒゲが全体レンジの50%以上
        4. 実体が小さい（全体レンジの30%以下）
        
        Args:
            high: 高値
            low: 安値
            close: 終値
            open_price: 始値
            
        Returns:
            bool: ピンバーと判定された場合True
        """
        upper_shadow = high - max(close, open_price)
        lower_shadow = min(close, open_price) - low
        body = abs(close - open_price)
        total_range = high - low
        
        # 条件1: 下ヒゲの最小値チェック（緩和: 2.0円 → 1.0円）
        if lower_shadow < 1.0:
            return False
        
        # 条件2: 上ヒゲが下ヒゲの2倍以上
        if upper_shadow <= self.params["pin_bar_ratio"] * lower_shadow:
            return False
        
        # 条件3: 上ヒゲが全体レンジの50%以上
        if total_range > 0 and upper_shadow < 0.5 * total_range:
            return False
        
        # 条件4: 実体が小さい（全体レンジの30%以下）
        if total_range > 0 and body > 0.3 * total_range:
            return False
        
        return True

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        
        Issue調査報告20260210修正: ウォームアップ期間フィルタリング追加
        - trading_start_date未満の日付ではエントリーシグナルを0に設定
        - バックテスト期間外のエントリーを防止
        """
        # ウォームアップ期間フィルタリング（Issue調査報告20260210対応）
        if hasattr(self, 'trading_start_date') and self.trading_start_date is not None:
            try:
                current_date_at_idx = self.data.index[idx]
                # pd.Timestampに変換して比較
                if not isinstance(current_date_at_idx, pd.Timestamp):
                    current_date_at_idx = pd.Timestamp(current_date_at_idx)
                if not isinstance(self.trading_start_date, pd.Timestamp):
                    trading_start_ts = pd.Timestamp(self.trading_start_date)
                else:
                    trading_start_ts = self.trading_start_date
                
                # タイムゾーン統一
                if current_date_at_idx.tz is not None:
                    current_date_at_idx = current_date_at_idx.tz_localize(None)
                if trading_start_ts.tz is not None:
                    trading_start_ts = trading_start_ts.tz_localize(None)
                
                if current_date_at_idx < trading_start_ts:
                    self.logger.debug(
                        f"[WARMUP_SKIP] ウォームアップ期間のためエントリースキップ: "
                        f"{current_date_at_idx.strftime('%Y-%m-%d')} < {trading_start_ts.strftime('%Y-%m-%d')}"
                    )
                    return 0  # エントリー禁止
            except Exception as e:
                self.logger.warning(f"[WARMUP_FILTER_ERROR] trading_start_date比較エラー: {e}")
        
        if idx < 5:  # 過去データが不足している場合
            return 0

        rsi = self.data['RSI'].iloc[idx]
        current_price = self.data[self.price_column].iloc[idx]
        previous_close = self.data[self.price_column].iloc[idx - 1]

        # ギャップダウンの判定
        gap_down = current_price < previous_close * (1.0 - self.params["gap_threshold"])

        # ピンバーの判定（改善版）
        if 'High' in self.data.columns and 'Low' in self.data.columns and 'Open' in self.data.columns:
            high = self.data['High'].iloc[idx]
            low = self.data['Low'].iloc[idx]
            open_price = self.data['Open'].iloc[idx]
            pin_bar = self._is_valid_pinbar(high, low, current_price, open_price)
        else:
            pin_bar = False

        # トレンド判定（統一トレンド判定インターフェースを使用）
        if self.params["trend_filter_enabled"]:
            # 統一トレンド判定インターフェースを使用
            trend = detect_unified_trend(
                self.data.iloc[:idx + 1], 
                price_column=self.price_column,
                strategy="contrarian_strategy",
                method="combined"  # 複合メソッドを使用
            )
            # 許可されたトレンド内にあるか確認
            if trend not in self.params["allowed_trends"]:
                return 0
        # trend_filter_enabled=Falseの場合、トレンドチェックをスキップ

        # エントリー条件（B: RSI条件を両方に適用）
        # 条件1: RSI過売り + ギャップダウン
        if rsi <= self.params["rsi_oversold"] and gap_down:
            # Phase 1修正: エントリー価格記録を削除（backtest()で翌日始値を記録するため）
            # self.entry_prices[idx] = current_price  # ← 削除
            return 1
        # 条件2: RSI過売り + ピンバー（RSI条件追加）
        if rsi <= self.params["rsi_oversold"] and pin_bar:
            # Phase 1修正: エントリー価格記録を削除（backtest()で翌日始値を記録するため）
            # self.entry_prices[idx] = current_price  # ← 削除
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        """
        if idx < 1:
            return 0

        # ポジション状態管理を追加
        # 現在までのエントリー・エグジット数を計算
        current_entries = (self.data['Entry_Signal'].iloc[:idx+1] == 1).sum()
        current_exits = abs((self.data['Exit_Signal'].iloc[:idx+1] == -1).sum())
        
        # アクティブなポジションがない場合はエグジット不可
        if current_entries <= current_exits:
            return 0

        # 最新のエントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0

        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        entry_price = self.entry_prices.get(latest_entry_idx)
        if entry_price is None:
            return 0

        # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日の終値を見てからidx日の終値でイグジットすることは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でイグジット
        # 注意: idx+1アクセスの安全性はbacktest()の`for idx in range(len(self.data) - 1)`で確保済み
        current_price = self.data['Open'].iloc[idx + 1]

        # RSIによるイグジット
        current_rsi = self.data['RSI'].iloc[idx]
        if current_rsi >= self.params["rsi_exit_level"]:
            return -1

        # トレーリングストップ
        if latest_entry_idx not in self.high_prices:
            self.high_prices[latest_entry_idx] = entry_price
        # Phase 1b修正: 当日高値で更新（当日終値ではなく）
        # 理由: トレーリングストップは当日高値を基準とするのが一般的
        # 注意: idx日の高値は既に確定済みの情報なので使用可能
        self.high_prices[latest_entry_idx] = max(self.high_prices[latest_entry_idx], self.data['High'].iloc[idx])
        trailing_stop_price = self.high_prices[latest_entry_idx] * (1.0 - self.params["trailing_stop_pct"])
        if current_price <= trailing_stop_price:
            return -1

        # 利益確定
        if current_price >= entry_price * (1.0 + self.params["take_profit"]):
            return -1

        # ストップロス
        if current_price <= entry_price * (1.0 - self.params["stop_loss"]):
            return -1

        # 最大保有日数
        days_held = idx - latest_entry_idx
        if days_held >= self.params["max_hold_days"]:
            return -1

        return 0

    def backtest(self, trading_start_date=None, trading_end_date=None):
        """
        バックテストを実行する。
        
        Parameters:
            trading_start_date (datetime, optional): 取引開始日（この日以降にシグナル生成開始）
            trading_end_date (datetime, optional): 取引終了日（この日以前までシグナル生成）
        """
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        # Phase 1修正: 最終日を除外してidx+1アクセスを安全に（ルックアヘッドバイアス修正）
        # 理由: エントリー価格を翌日始値（idx+1）に変更するため、最終日でのIndexError回避
        for idx in range(len(self.data) - 1):
            # 取引期間フィルタリング（BaseStrategy.backtest()と同じロジック）
            if trading_start_date is not None or trading_end_date is not None:
                current_date = self.data.index[idx]
                in_trading_period = True
                
                if trading_start_date is not None and current_date < trading_start_date:
                    in_trading_period = False
                if trading_end_date is not None and current_date > trading_end_date:
                    in_trading_period = False
                
                if not in_trading_period:
                    # 取引期間外はシグナル生成をスキップ
                    continue
            # エントリーシグナル
            if not self.data['Entry_Signal'].iloc[max(0, idx - 1):idx + 1].any():
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    # Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
                    # Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
                    # 理由: idx日の終値を見てからidx日の終値で買うことは不可能
                    # リアルトレードでは翌日（idx+1日目）の始値でエントリー
                    next_day_open = self.data['Open'].iloc[idx + 1]
                    
                    # Phase 2: スリッページ・取引コスト適用（買い注文は不利な方向）
                    # デフォルト: slippage=0.001（0.1%）、transaction_cost=0.0（0%）
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = next_day_open * (1 + slippage + transaction_cost)
                    self.entry_prices[idx] = entry_price

            # イグジットシグナル
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

        return self.data

    def backtest_daily(self, current_date, stock_data, existing_position=None, trading_start_date=None, **kwargs):
        """
        ContrarianStrategy 日次バックテスト実行
        
        Phase 3-C Day 10実装: Contrarian戦略でのbacktest_daily()実装
        
        Sprint 1.5修正 (2026-02-09): force_close対応
        - **kwargs追加: entry_symbol_dataを受け取れるように拡張
        - force_close時はentry_symbol_dataで決済価格を取得
        
        Issue調査報告20260210修正: trading_start_date追加
        - ウォームアップ期間（trading_start_date未満）のエントリー防止
        - generate_entry_signal()でフィルタリング実行
        
        Parameters:
            current_date (datetime): 判定対象日
            stock_data (pd.DataFrame): 最新の株価データ
            trading_start_date: バックテスト開始日（この日以降のみエントリー許可）
            existing_position (dict, optional): 既存ポジション情報
                {
                    'symbol': str,
                    'quantity': int,
                    'entry_price': float,
                    'entry_date': datetime,
                    'entry_idx': int,
                    'force_close': bool,         # Sprint 1.5: 強制決済フラグ
                    'entry_symbol': str          # Cycle 7: エントリー銘柄コード
                }
            **kwargs: 追加引数
                - entry_symbol_data (pd.DataFrame): force_close時の元の銘柄データ
                
        Returns:
            dict: {
                'action': 'entry'|'exit'|'hold',
                'signal': 1|-1|0,
                'price': float,
                'shares': int,
                'reason': str
            }
        """
        # Issue調査報告20260210修正: trading_start_dateを保存（generate_entry_signal()で使用）
        self.trading_start_date = trading_start_date
        if trading_start_date is not None:
            self.logger.info(f"[WARMUP_FILTER] trading_start_date設定: {trading_start_date.strftime('%Y-%m-%d') if hasattr(trading_start_date, 'strftime') else trading_start_date}")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Phase 1: current_dateの型変換・検証
        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        elif not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
            
        # Phase 2: データ整合性チェック
        if current_date not in stock_data.index:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Contrarian: No data available for {current_date.strftime("%Y-%m-%d")}'
            }
            
        # Phase 3: ウォームアップ期間考慮
        current_idx = stock_data.index.get_loc(current_date)
        warmup_period = 150
        rsi_period = self.params.get("rsi_period", 14)
        min_required = max(warmup_period, rsi_period, 5)  # RSI期間と最小過去データ（5日）を考慮
        
        if current_idx < min_required:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Contrarian: Insufficient warmup data. Required: {min_required}, Available: {current_idx}'
            }
        
        # Phase 4: データ更新（Option B方式）
        original_data = self.data.copy()
        
        try:
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            # RSIも更新（initialize_strategy()で計算済み、shift(1)適用済み）
            if 'RSI' in self.data.columns:
                # RSIは既にshift(1)適用済みなので再計算は不要
                pass
            
            logger.debug(f"[Contrarian.backtest_daily] Data updated: {updated_columns}")
            
            # Phase 5: 既存ポジション処理分岐
            if existing_position is not None:
                # エグジット判定（簡易版: Entry_Signal依存を回避）
                # Sprint 1.5修正: entry_symbol_dataをkwargsから取得して渡す
                entry_symbol_data = kwargs.get('entry_symbol_data', None)
                return self._handle_exit_logic_daily(current_idx, existing_position, stock_data, current_date, entry_symbol_data)
            else:
                # エントリー判定
                return self._handle_entry_logic_daily(current_idx, stock_data, current_date)
        
        finally:
            # データ復元
            self.data = original_data
    
    def _handle_exit_logic_daily(self, current_idx, existing_position, stock_data, current_date, entry_symbol_data=None):
        """
        エグジット判定ロジック（backtest_daily用簡易版）
        
        generate_exit_signal()がEntry_Signal依存のため、
        直接エグジット条件を判定する簡易実装
        
        Cycle 27修正: entry_symbol_data対応
        - force_close時（entry_symbol_data提供時）は元の銘柄のデータでエグジット価格を取得
        - 通常時は現在の銘柄（stock_data）でエグジット価格を取得
        
        Sprint 1.5修正 (2026-02-09): force_close強制決済実装
        - force_close=True の場合、エグジット条件に関わらず強制決済
        - 銘柄切替時の旧ポジション自動決済を保証
        
        Parameters:
            current_idx: 現在のインデックス
            existing_position: 既存ポジション情報
                {
                    'entry_idx': int,
                    'quantity': int,
                    'entry_price': float,
                    'entry_date': datetime,
                    'force_close': bool,         # Sprint 1.5: 強制決済フラグ
                    'entry_symbol': str          # Cycle 7: エントリー銘柄コード
                }
            stock_data: 現在の銘柄データ
            current_date: 判定日時
            entry_symbol_data: force_close時の元の銘柄データ（オプション）
        
        Returns:
            dict: {'action': 'exit'|'hold', 'signal': -1|0, 'price': float, 'shares': int, 'reason': str}
        
        Note:
            - copilot-instructions.md準拠（ルックアヘッドバイアス防止、フォールバック制限）
            - force_close=True の場合、エグジット条件判定をスキップして強制決済
            - 最終日フォールバックは限定的使用（copilot-instructions.md Section 68-73準拠）
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # existing_positionからエントリー情報取得
            entry_price = existing_position.get('entry_price', 0)
            entry_date = existing_position.get('entry_date')
            entry_idx = existing_position.get('entry_idx', current_idx)
            is_force_close = existing_position.get('force_close', False)
            
            # Cycle 27修正: force_close時はentry_symbol_dataを使用
            if is_force_close and entry_symbol_data is not None:
                data_for_exit = entry_symbol_data
                logger.info(
                    f"[CONTRARIAN_EXIT] force_close=True: entry_symbol_dataを使用 "
                    f"(rows={len(entry_symbol_data)}, symbol={existing_position.get('entry_symbol', 'Unknown')})"
                )
            else:
                data_for_exit = stock_data
                logger.debug(f"[CONTRARIAN_EXIT] force_close={is_force_close}: stock_dataを使用")
            
            if entry_price == 0:
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': existing_position.get('quantity', 0),
                    'reason': 'Contrarian: Invalid entry price in existing_position'
                }
            
            # ============================================================
            # Sprint 1.5修正: force_close強制決済（最優先処理）
            # ============================================================
            # 【削除禁止】このブロックは銘柄切替時の旧ポジション決済に必須
            # 削除すると、ポジションが残り続け、all_transactions.csvが不完全になる
            # 参照: MULTI_POSITION_IMPLEMENTATION_PLAN.md Sprint 1.5
            # ============================================================
            if is_force_close:
                # 強制決済ログ（デバッグ用詳細情報）
                entry_symbol = existing_position.get('entry_symbol', 'Unknown')
                quantity = existing_position.get('quantity', 0)
                
                logger.warning(
                    f"[CONTRARIAN_FORCE_CLOSE] 銘柄切替による強制決済を実行"
                )
                logger.info(
                    f"[CONTRARIAN_FORCE_CLOSE] Position Info: "
                    f"entry_symbol={entry_symbol}, "
                    f"entry_price={entry_price:.2f}円, "
                    f"quantity={quantity}株, "
                    f"date={current_date.strftime('%Y-%m-%d')}"
                )
                
                # エグジット価格取得（entry_symbol_data優先）
                if current_idx + 1 < len(data_for_exit):
                    # 標準: 翌日始値（ルックアヘッドバイアス防止）
                    exit_price = data_for_exit.iloc[current_idx + 1]['Open']
                    data_source = "entry_symbol_data" if entry_symbol_data is not None else "stock_data"
                    price_type = "Open (next day)"
                else:
                    # フォールバック: 最終日は当日終値
                    # 理由: current_idx + 1 が存在しない境界条件
                    # copilot-instructions.md Section 68-73: 限定的フォールバック使用を許可
                    exit_price = data_for_exit.iloc[current_idx]['Close']
                    data_source = f"{'entry_symbol_data' if entry_symbol_data is not None else 'stock_data'} (Close fallback)"
                    price_type = "Close (final day)"
                    logger.warning(
                        f"[CONTRARIAN_FORCE_CLOSE] Final day exit: using Close price fallback. "
                        f"idx={current_idx}, date={current_date.strftime('%Y-%m-%d')}"
                    )
                
                # 決済価格ログ
                logger.info(
                    f"[CONTRARIAN_FORCE_CLOSE] Exit Price: {exit_price:.2f}円 "
                    f"(source={data_source}, type={price_type})"
                )
                
                # 損益計算（参考情報）
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0
                
                logger.info(
                    f"[CONTRARIAN_FORCE_CLOSE] P&L: {pnl:,.0f}円 ({pnl_pct:+.2f}%), "
                    f"entry={entry_price:.2f}円 → exit={exit_price:.2f}円"
                )
                
                # 強制決済実行
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': quantity,
                    'reason': f'Force close due to symbol switch from {entry_symbol}'
                }
            # ============================================================
            # Sprint 1.5修正ここまで
            # ============================================================
            
            # ------------------------------------------------------------
            # 通常のエグジット判定（force_close=False）
            # ------------------------------------------------------------
            # 翌日始値でエグジット（ルックアヘッドバイアス防止）
            # Cycle 27修正: data_for_exitを使用
            if current_idx + 1 < len(data_for_exit):
                exit_price = data_for_exit.iloc[current_idx + 1]['Open']
            else:
                # 最終日フォールバック
                exit_price = data_for_exit.iloc[current_idx]['Close']
                logger.warning(f"[Contrarian] Using Close price fallback for final day: {current_date}")
            
            # エグジット条件判定
            # 1. RSIによるイグジット（中立域）
            current_rsi = self.data['RSI'].iloc[current_idx]
            if pd.notna(current_rsi) and current_rsi >= self.params["rsi_exit_level"]:
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Contrarian: RSI exit on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # 2. 利益確定
            if exit_price >= entry_price * (1.0 + self.params["take_profit"]):
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Contrarian: Take profit on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # 3. ストップロス
            if exit_price <= entry_price * (1.0 - self.params["stop_loss"]):
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Contrarian: Stop loss on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # 4. トレーリングストップ
            # エントリーから現在までの高値を取得
            if entry_date is not None:
                try:
                    entry_loc = stock_data.index.get_loc(entry_date)
                    high_since_entry = stock_data.iloc[entry_loc:current_idx+1]['High'].max()
                except:
                    high_since_entry = entry_price
            else:
                high_since_entry = entry_price
            
            trailing_stop_price = high_since_entry * (1.0 - self.params["trailing_stop_pct"])
            if exit_price <= trailing_stop_price:
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Contrarian: Trailing stop on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # 5. 最大保有日数
            if entry_date is not None:
                days_held = (current_date - entry_date).days
                if days_held >= self.params["max_hold_days"]:
                    return {
                        'action': 'exit',
                        'signal': -1,
                        'price': float(exit_price),
                        'shares': existing_position.get('quantity', 0),
                        'reason': f'Contrarian: Max hold days on {current_date.strftime("%Y-%m-%d")}'
                    }
            
            # エグジット条件に該当せず: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': existing_position.get('quantity', 0),
                'reason': f'Contrarian: Holding position from {current_date.strftime("%Y-%m-%d")}'
            }
        
        except Exception as e:
            # エラーハンドリング（copilot-instructions.md準拠: エラー隠蔽禁止）
            logger.error(f"[Contrarian] Exit logic error: {e}", exc_info=True)
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Contrarian: Exit logic error: {str(e)}'
            }
    
    def _handle_entry_logic_daily(self, current_idx, stock_data, current_date):
        """
        エントリー判定ロジック（backtest_daily用）
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # generate_entry_signal()を使用してエントリー判定
            entry_signal = self.generate_entry_signal(current_idx)
            
            if entry_signal == 1:
                # 翌日始値でエントリー + スリッページ
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
                        'reason': f'Contrarian: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                    }
                else:
                    # 最終日の場合エントリー不可
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': f'Contrarian: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                    }
            else:
                # エントリーシグナルなし
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'Contrarian: No entry signal on {current_date.strftime("%Y-%m-%d")}'
                }
        
        except Exception as e:
            logger.error(f"[Contrarian] Entry logic error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Contrarian: Entry logic error: {str(e)}'
            }
    
    def _calculate_position_size_daily(self, entry_price):
        """
        ポジションサイズ計算（backtest_daily用）
        """
        target_amount = self.params.get("position_amount", 100000)
        
        if entry_price > 0:
            shares = int(target_amount / entry_price)
            shares = max(100, shares // 100 * 100)
            return shares
        else:
            return 0

    def load_optimized_parameters(self, ticker: str = None):
        """
        承認済みの最適化パラメータを自動適用
        """
        manager = OptimizedParameterManager()
        params = manager.load_approved_params("ContrarianStrategy", ticker)
        if params:
            self.params.update(params)
        return self.params

    def run_optimized_strategy(self, ticker: str = None):
        """
        最適化パラメータを自動適用してバックテスト実行
        """
        self.load_optimized_parameters(ticker)
        return self.backtest()

    def get_optimization_info(self, ticker: str = None):
        """
        最適化パラメータのメタ情報を取得
        """
        manager = OptimizedParameterManager()
        configs = manager.list_available_configs(strategy_name="ContrarianStrategy", ticker=ticker)
        return configs

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100
    }, index=dates)

    strategy = ContrarianStrategy(df, price_column='Adj Close')
    result = strategy.backtest()
    print(result[['Adj Close', 'RSI', 'Entry_Signal', 'Exit_Signal']].tail())