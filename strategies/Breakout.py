"""
Module: Breakout
File: Breakout.py
Description: 
  ブレイクアウト（価格の節目突破）戦略を実装したクラスです。前日高値を
  出来高増加を伴って上抜けた場合にエントリーし、利益確定や高値からの
  反落でイグジットします。シンプルながら効果的なモメンタム戦略の一つです。

Author: kouseven7
Created: 2023-03-20
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Close", volume_column: str = "Volume"):
        """
        ブレイクアウト戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        
        Note:
            price_columnは "Close" を使用してください。"Adj Close" (調整後終値) を使用すると、
            配当調整により過去の価格が下方修正され、High (未調整) との比較が不正確になります。
            これにより配当支払い銘柄でシグナルが生成されなくなります。
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.volume_column = volume_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        self.high_prices = {}   # 高値を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "volume_threshold": 1.2,   # 出来高増加率の閾値（20%）
            "take_profit": 0.03,       # 利益確定（3%）
            "look_back": 1,            # 前日からのブレイクアウトを見る日数
            "trailing_stop": 0.02,     # トレーリングストップ（高値から2%下落）
            "breakout_buffer": 0.01,   # ブレイクアウト判定の閾値（1%）
            "slippage": 0.001,         # Phase 2: スリッページ（0.1%、買い注文は不利な方向）
            "transaction_cost": 0.0    # Phase 2: 取引コスト（0%、オプション）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 前日高値を上抜けた場合
        - 出来高が前日比で20%増加している

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        look_back = self.params["look_back"]
        
        # Cycle 20デバッグログ追加
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_ENTRY_SIG] generate_entry_signal開始: idx={idx}, look_back={look_back}, data.shape={self.data.shape}")
            self.log_trade(f"[DEBUG_ENTRY_SIG] generate_entry_signal開始: idx={idx}, look_back={look_back}, "
                         f"data.shape={self.data.shape}")
        
        if idx < look_back:  # 過去データが必要
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_ENTRY_SIG] 早期リターン: idx({idx}) < look_back({look_back})")
                self.log_trade(f"[DEBUG_ENTRY_SIG] 早期リターン: idx({idx}) < look_back({look_back})")
            return 0
            
        if 'High' not in self.data.columns:
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_ENTRY_SIG] 早期リターン: Highカラムなし, columns={self.data.columns.tolist()}")
                self.log_trade(f"[DEBUG_ENTRY_SIG] 早期リターン: Highカラムなし, columns={self.data.columns.tolist()}")
            return 0  # 高値データがない場合

        current_price = self.data[self.price_column].iloc[idx]
        previous_high = self.data['High'].iloc[idx - look_back]
        
        # 出来高データの確認
        if self.volume_column not in self.data.columns:
            volume_increase = False  # 出来高データがない場合はシグナルを出さない
        else:
            current_volume = self.data[self.volume_column].iloc[idx]
            previous_volume = self.data[self.volume_column].iloc[idx - look_back]
            # 出来高が前日比で指定率以上増加している
            volume_increase = current_volume > previous_volume * self.params["volume_threshold"]

        # 前日高値を上抜けた場合（上抜け幅をパラメータ化）
        price_breakout = current_price > previous_high * (1 + self.params["breakout_buffer"])
        
        # Cycle 19デバッグログ追加
        if os.getenv("DEBUG_BACKTEST"):
            date_str = self.data.index[idx]
            self.log_trade(f"[DEBUG_ENTRY] idx={idx}, date={date_str}, "
                         f"price={current_price:.2f}, prev_high={previous_high:.2f}, "
                         f"vol={current_volume if self.volume_column in self.data.columns else 'N/A'}, "
                         f"prev_vol={previous_volume if self.volume_column in self.data.columns else 'N/A'}, "
                         f"price_breakout={price_breakout}, volume_increase={volume_increase}")

        if price_breakout and volume_increase:
            # Cycle 19修正: idx+1アクセスを削除（IndexError回避）
            # generate_entry_signal()は1/0のみを返す
            # 価格計算は_handle_entry_logic_daily()で実施
            self.log_trade(f"Breakout エントリーシグナル: 日付={self.data.index[idx]}, 前日高値={previous_high}")
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        条件:
        - エントリー株価を3%超えた場合に利確
        - 高値を下回った場合に損切り

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 過去データが必要
            return 0
            
        # エントリー価格と高値を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリーインデックス（日付）を取得
        latest_entry_date = entry_indices[-1]
        # インデックス位置（整数）を取得
        latest_entry_pos = self.data.index.get_loc(latest_entry_date)

        # Phase 1a修正: フォールバック処理も翌日始値を使用（ルックアヘッドバイアス修正）
        # Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
        if latest_entry_date not in self.entry_prices:
            next_day_pos = latest_entry_pos + 1
            if next_day_pos < len(self.data):
                next_day_open = self.data['Open'].iloc[next_day_pos]
                # Phase 2: スリッページ・取引コスト適用
                slippage = self.params.get("slippage", 0.001)
                transaction_cost = self.params.get("transaction_cost", 0.0)
                entry_price = next_day_open * (1 + slippage + transaction_cost)
                self.entry_prices[latest_entry_date] = entry_price
            else:
                # 最終日の場合は当日始値を使用（境界条件の妥協案）
                # Phase 2: この場合もスリッページ適用
                current_open = self.data['Open'].iloc[latest_entry_pos]
                slippage = self.params.get("slippage", 0.001)
                transaction_cost = self.params.get("transaction_cost", 0.0)
                entry_price = current_open * (1 + slippage + transaction_cost)
                self.entry_prices[latest_entry_date] = entry_price
            
        if latest_entry_date not in self.high_prices:
            next_day_pos = latest_entry_pos + 1
            if next_day_pos < len(self.data):
                if 'High' in self.data.columns:
                    self.high_prices[latest_entry_date] = self.data['High'].iloc[next_day_pos]
                else:
                    self.high_prices[latest_entry_date] = self.data['Open'].iloc[next_day_pos]
            else:
                # 最終日の場合
                if 'High' in self.data.columns:
                    self.high_prices[latest_entry_date] = self.data['High'].iloc[latest_entry_pos]
                else:
                    self.high_prices[latest_entry_date] = self.data['Open'].iloc[latest_entry_pos]
            
        entry_price = self.entry_prices[latest_entry_date]
        high_price = self.high_prices[latest_entry_date]
        
        # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日の終値を見てからidx日の終値でイグジットすることは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でイグジット
        # 注意: idx+1アクセスの安全性はbacktest()の`for idx in range(len(self.data) - 1)`で確保済み
        current_price = self.data['Open'].iloc[idx + 1]
        
        # 現在の高値を更新（トレーリングストップのために）
        if 'High' in self.data.columns and self.data['High'].iloc[idx] > high_price:
            high_price = self.data['High'].iloc[idx]
            self.high_prices[latest_entry_date] = high_price

        # 利確条件
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # 損切条件（高値からの反落）
        trailing_stop_level = 1 - self.params["trailing_stop"]
        if current_price < high_price * trailing_stop_level:  # 高値からtrailing_stop%下落したら損切り
            self.log_trade(f"Breakout イグジットシグナル: 高値から反落 日付={self.data.index[idx]}, 価格={current_price}, 高値={high_price}")
            return -1

        return 0

    def backtest(self, trading_start_date=None, trading_end_date=None):
        """
        ブレイクアウト戦略のバックテストを実行する。
        
        Parameters:
            trading_start_date (datetime, optional): 取引開始日（この日以降にシグナル生成開始）
            trading_end_date (datetime, optional): 取引終了日（この日以前までシグナル生成）
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        
        # ポジション管理変数
        in_position = False
        last_entry_idx = None

        # 各日にちについてシグナルを計算
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
            
            # ポジションを持っていない場合のみエントリーシグナルをチェック
            if not in_position:
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                    in_position = True
                    last_entry_idx = idx
            
            # ポジションを持っている場合のみイグジットシグナルをチェック
            elif in_position:
                exit_signal = self.generate_exit_signal(idx)
                if exit_signal == -1:
                    self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                    in_position = False
                    last_entry_idx = None

        return self.data

    def backtest_daily(self, current_date, stock_data, existing_position=None, **kwargs):
        """
        BreakoutStrategy 日次バックテスト実行
        
        Phase 3-C Day 9実装: Breakout戦略でのbacktest_daily()実装
        
        Cycle 26修正: **kwargs追加
        - 理由: force_close時にentry_symbol_dataがkwargsで渡される（Cycle 7修正）
        
        Cycle 27修正: entry_symbol_data使用
        - force_close時はentry_symbol_data（元の銘柄）でエグジット価格を取得
        
        Parameters:
            current_date (datetime): 判定対象日
            stock_data (pd.DataFrame): 最新の株価データ
            existing_position (dict, optional): 既存ポジション情報
            **kwargs: 追加引数（entry_symbol_data等）
                {
                    'symbol': str,
                    'quantity': int,
                    'entry_price': float,
                    'entry_date': datetime,
                    'entry_idx': int
                }
                
        Returns:
            dict: {
                'action': 'entry'|'exit'|'hold',
                'signal': 1|-1|0,
                'price': float,
                'shares': int,
                'reason': str
            }
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Cycle 20: 関数呼び出し確認用print()
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_BACKTEST_DAILY] backtest_daily()呼び出し確認: current_date={current_date}")
            print(f"[DEBUG_BACKTEST_DAILY] stock_data.shape={stock_data.shape}")
            print(f"[DEBUG_BACKTEST_DAILY] stock_data.index[0]={stock_data.index[0]}, stock_data.index[-1]={stock_data.index[-1]}")
            print(f"[DEBUG_BACKTEST_DAILY] current_date in stock_data.index: {current_date in stock_data.index}")
        
        # Phase 1: current_dateの型変換・検証
        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        elif not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        # Cycle 20修正: タイムゾーン統一（tz-naiveに変換）
        if current_date.tz is not None:
            current_date = current_date.tz_localize(None)
        
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_PHASE1] current_date={current_date}, type={type(current_date)}, tz={current_date.tz}")
            print(f"[DEBUG_PHASE1] stock_data.index has tz: {stock_data.index.tz is not None}")
            if stock_data.index.tz is not None:
                print(f"[DEBUG_PHASE1] stock_data.index.tz: {stock_data.index.tz}")
            
        # Phase 2: データ整合性チェック（タイムゾーン統一後）
        # stock_dataのインデックスもtz-naiveに変換
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)
        
        if current_date not in stock_data.index:
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_PHASE2] 早期リターン: current_date({current_date})がstock_data.indexに存在しない")
                print(f"[DEBUG_PHASE2] stock_data.index available: {stock_data.index.tolist()[:5]} ... {stock_data.index.tolist()[-5:]}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Breakout: No data available for {current_date.strftime("%Y-%m-%d")}'
            }
        
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_PHASE2] データ整合性OK: current_date({current_date}) in stock_data.index")
            
        # Phase 3: ウォームアップ期間考慮
        current_idx = stock_data.index.get_loc(current_date)
        look_back = self.params.get("look_back", 1)
        # Cycle 19修正: warmup_periodチェックを削除（DSSMSが既にwarmup込みデータを渡すため）
        # min_required = max(warmup_period, look_back) だと常にmin_required=150でエントリー不可
        min_required = look_back
        
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_PHASE3] current_idx={current_idx}, min_required={min_required}, stock_data.shape={stock_data.shape}")
        
        if current_idx < min_required:
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_PHASE3] 早期リターン: Insufficient data (current_idx={current_idx} < min_required={min_required})")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Breakout: Insufficient data for look_back. Required: {min_required}, Available: {current_idx}'
            }
        
        # Phase 4: データ更新（Option B方式）
        original_data = self.data.copy()
        
        # Cycle 20デバッグログ追加
        if os.getenv("DEBUG_BACKTEST"):
            print(f"[DEBUG_DATA_UPDATE] データ更新前: self.data.shape={self.data.shape}")
            print(f"[DEBUG_DATA_UPDATE] self.data.index: [0]={self.data.index[0]}, [-1]={self.data.index[-1]}")
            print(f"[DEBUG_DATA_UPDATE] stock_data: shape={stock_data.shape}, index: [0]={stock_data.index[0]}, [-1]={stock_data.index[-1]}")
            print(f"[DEBUG_DATA_UPDATE] current_date={current_date}, current_idx={current_idx}")
            if current_idx < len(self.data):
                print(f"[DEBUG_DATA_UPDATE] self.data.index[{current_idx}]={self.data.index[current_idx]}")
            logger.info(f"[DEBUG_DATA_UPDATE] データ更新前: self.data.shape={self.data.shape}, "
                      f"self.data.index[0]={self.data.index[0]}, self.data.index[-1]={self.data.index[-1]}")
            logger.info(f"[DEBUG_DATA_UPDATE] stock_data: shape={stock_data.shape}, "
                      f"index[0]={stock_data.index[0]}, index[-1]={stock_data.index[-1]}")
            logger.info(f"[DEBUG_DATA_UPDATE] current_date={current_date}, current_idx={current_idx}")
            if current_idx < len(self.data):
                logger.info(f"[DEBUG_DATA_UPDATE] self.data.index[{current_idx}]={self.data.index[current_idx]}")
        
        try:
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            logger.debug(f"[Breakout.backtest_daily] Data updated: {updated_columns}")
            
            # Cycle 20: 分岐確認用print()
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_BRANCH] existing_position: {existing_position is not None}, type={type(existing_position)}")
            
            # Cycle 27修正: entry_symbol_dataをkwargsから取得
            entry_symbol_data = kwargs.get('entry_symbol_data', None)
            
            # Phase 5: 既存ポジション処理分岐
            if existing_position is not None:
                # エグジット判定（簡易版: Entry_Signal依存を回避）
                if os.getenv("DEBUG_BACKTEST"):
                    print(f"[DEBUG_BRANCH] エグジットロジックへ: existing_position={existing_position}")
                return self._handle_exit_logic_daily(current_idx, existing_position, stock_data, current_date, entry_symbol_data)
            else:
                # エントリー判定
                if os.getenv("DEBUG_BACKTEST"):
                    print(f"[DEBUG_BRANCH] エントリーロジックへ")
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
                logger.info(f"[BREAKOUT_EXIT] force_close=True: entry_symbol_dataを使用（{len(entry_symbol_data)}行）")
            else:
                data_for_exit = stock_data
            
            if entry_price == 0:
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': existing_position.get('quantity', 0),
                    'reason': 'Breakout: Invalid entry price in existing_position'
                }
            
            # 翌日始値でエグジット（ルックアヘッドバイアス防止）
            # Cycle 27修正: data_for_exitを使用
            if current_idx + 1 < len(data_for_exit):
                exit_price = data_for_exit.iloc[current_idx + 1]['Open']
            else:
                # 最終日フォールバック
                exit_price = data_for_exit.iloc[current_idx]['Close']
                logger.warning(f"[Breakout] Using Close price fallback for final day: {current_date}")
            
            logger.info(f"[BREAKOUT_EXIT] exit_price={exit_price}, source={'entry_symbol_data' if is_force_close and entry_symbol_data is not None else 'stock_data'}")
            
            # エグジット条件判定
            # 1. 利益確定
            if exit_price >= entry_price * (1 + self.params["take_profit"]):
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Breakout: Take profit on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # 2. トレーリングストップ（高値からの反落）
            # エントリーから現在までの高値を取得
            if entry_date is not None:
                try:
                    entry_loc = stock_data.index.get_loc(entry_date)
                    high_since_entry = stock_data.iloc[entry_loc:current_idx+1]['High'].max()
                except:
                    high_since_entry = exit_price
            else:
                high_since_entry = exit_price
            
            trailing_stop_level = high_since_entry * (1 - self.params["trailing_stop"])
            if exit_price < trailing_stop_level:
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'Breakout: Trailing stop on {current_date.strftime("%Y-%m-%d")}'
                }
            
            # エグジット条件に該当せず: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': existing_position.get('quantity', 0),
                'reason': f'Breakout: Holding position from {current_date.strftime("%Y-%m-%d")}'
            }
        
        except Exception as e:
            logger.error(f"[Breakout] Exit logic error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Breakout: Exit logic error: {str(e)}'
            }
    
    def _handle_entry_logic_daily(self, current_idx, stock_data, current_date):
        """
        エントリー判定ロジック（backtest_daily用）
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Cycle 20デバッグログ追加
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_ENTRY_LOGIC] _handle_entry_logic_daily呼び出し: current_idx={current_idx}, date={current_date}")
                logger.info(f"[DEBUG_ENTRY_LOGIC] _handle_entry_logic_daily呼び出し: current_idx={current_idx}, "
                          f"date={current_date}, stock_data.shape={stock_data.shape}")
            
            # generate_entry_signal()を使用してエントリー判定
            entry_signal = self.generate_entry_signal(current_idx)
            
            # Cycle 20デバッグログ追加
            if os.getenv("DEBUG_BACKTEST"):
                print(f"[DEBUG_ENTRY_LOGIC] generate_entry_signal()返り値: entry_signal={entry_signal}")
                logger.info(f"[DEBUG_ENTRY_LOGIC] generate_entry_signal()返り値: entry_signal={entry_signal}")
            
            if entry_signal == 1:
                # Cycle 20デバッグログ追加
                if os.getenv("DEBUG_BACKTEST"):
                    print(f"[DEBUG_ENTRY_CHECK] entry_signal=1, current_idx={current_idx}, len(stock_data)={len(stock_data)}")
                    print(f"[DEBUG_ENTRY_CHECK] current_idx + 1 < len(stock_data): {current_idx + 1 < len(stock_data)}")
                
                # 翌日始値でエントリー + スリッページ
                if current_idx + 1 < len(stock_data):
                    entry_price = stock_data.iloc[current_idx + 1]['Open']
                    
                    # スリッページ・取引コスト適用
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = entry_price * (1 + slippage + transaction_cost)
                    
                    # Cycle 19修正: entry_prices/high_pricesをここで初期化
                    # generate_entry_signal()から移動
                    entry_date = self.data.index[current_idx]
                    self.entry_prices[entry_date] = entry_price
                    
                    # 高値の初期値も翌日高値を使用（トレーリングストップの起点）
                    if 'High' in stock_data.columns:
                        self.high_prices[entry_date] = stock_data.iloc[current_idx + 1]['High']
                    else:
                        self.high_prices[entry_date] = stock_data.iloc[current_idx + 1]['Open']
                    
                    # ポジションサイズ計算
                    shares = self._calculate_position_size_daily(entry_price)
                    
                    result = {
                        'action': 'entry',
                        'signal': 1,
                        'price': float(entry_price),
                        'shares': shares,
                        'reason': f'Breakout: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                    }
                    
                    # Cycle 20デバッグログ追加
                    if os.getenv("DEBUG_BACKTEST"):
                        print(f"[DEBUG_RETURN] _handle_entry_logic_daily returning: {result}")
                        logger.info(f"[DEBUG_RETURN] _handle_entry_logic_daily returning: {result}")
                    
                    return result
                else:
                    # 最終日の場合エントリー不可
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': f'Breakout: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                    }
            else:
                # エントリーシグナルなし
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'Breakout: No entry signal on {current_date.strftime("%Y-%m-%d")}'
                }
        
        except Exception as e:
            logger.error(f"[Breakout] Entry logic error: {e}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Breakout: Entry logic error: {str(e)}'
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

    def run_optimized_strategy(self) -> pd.DataFrame:
        """
        最適化されたパラメータを使用して戦略を実行
        
        Returns:
            pd.DataFrame: 戦略実行結果
        """
        # 最適化パラメータの読み込み
        if hasattr(self, 'optimization_mode') and self.optimization_mode and not self.load_optimized_parameters():
            print(f"[WARNING] 最適化パラメータの読み込みに失敗しました。デフォルトパラメータを使用します。")
        
        # 使用するパラメータの表示
        if hasattr(self, '_approved_params') and self._approved_params:
            print(f"[OK] 最適化パラメータを使用:")
            print(f"   パラメータID: {self._approved_params.get('parameter_id', 'N/A')}")
            print(f"   作成日時: {self._approved_params.get('created_at', 'N/A')}")
            print(f"   シャープレシオ: {self._approved_params.get('sharpe_ratio', 'N/A')}")
            print(f"   パラメータ: {self._approved_params.get('parameters', {})}")
        else:
            print(f"[CHART] デフォルトパラメータを使用: {self.params}")
        
        # 戦略実行
        return self.backtest()
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        最適化情報を取得
        
        Returns:
            dict: 最適化情報
        """
        info = {
            'optimization_mode': getattr(self, 'optimization_mode', False),
            'using_optimized_params': getattr(self, '_approved_params', None) is not None,
            'default_params': {
                "volume_threshold": 1.2,
                "take_profit": 0.03,
                "look_back": 1,
                "trailing_stop": 0.02,
                "breakout_buffer": 0.01
            },
            'current_params': self.params
        }
        
        if hasattr(self, '_approved_params') and self._approved_params:
            info['optimized_params'] = self._approved_params
        
        return info
    
    def load_optimized_parameters(self) -> bool:
        """
        最適化されたパラメータを読み込み
        
        Returns:
            bool: 読み込み成功
        """
        try:
            from config.optimized_parameters import OptimizedParameterManager
            
            manager = OptimizedParameterManager()
            
            # データの時間範囲から銘柄を推定
            ticker = getattr(self, 'ticker', 'DEFAULT')
            
            # 承認済みの最適化パラメータを取得
            params = manager.load_approved_params('breakout', ticker)
            
            if params:
                # パラメータを更新
                self.params.update(params['parameters'])
                self._approved_params = params
                print(f"[OK] 最適化パラメータを読み込みました (Date: {params.get('optimization_date', 'N/A')})")
                return True
            else:
                print(f"[WARNING] 承認済みの最適化パラメータが見つかりません (ticker: {ticker})")
                return False
                
        except Exception as e:
            print(f"[ERROR] 最適化パラメータの読み込みでエラー: {e}")
            return False

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(100) * 100,
        'High': np.random.random(100) * 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=dates)

    # ブレイクアウト戦略の実行
    strategy = BreakoutStrategy(df)
    result = strategy.backtest()
    print(result)