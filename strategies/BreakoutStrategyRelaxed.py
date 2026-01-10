"""
DSSMS取引0件問題 - エントリー条件緩和テスト用BreakoutStrategy

通常のBreakoutStrategyのvolume_threshold=1.2を1.0に緩和して
DSSMSでエントリーが発生するかテストします。

Author: Backtest Project Team
Created: 2026-01-10
Last Modified: 2026-01-10
"""

import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_sma

class BreakoutStrategyRelaxed(BaseStrategy):
    """エントリー条件を緩和したBreakoutStrategy（テスト用）"""
    
    def __init__(self, data):
        super().__init__(data)
        
        # パラメータ（DSSMS取引0件問題対応：条件緩和）
        self.volume_threshold = 0.8  # 大幅緩和（20%減少も許容）でエントリー機会を増加
        self.take_profit = 0.03
        self.look_back = 1
        self.trailing_stop = 0.02
        self.breakout_buffer = 0.01
        self.slippage = 0.001
        self.transaction_cost = 0.0
        
        # ロガー設定（強制DEBUG有効化）
        self.logger = logging.getLogger('BreakoutStrategyRelaxed')
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"BreakoutStrategyRelaxed 初期化: volume_threshold={self.volume_threshold} (DSSMS取引0件問題対応緩和)")
        self.logger.debug("[FORCE_DEBUG] BreakoutStrategyRelaxed DEBUGログ有効化")
        
        # インジケーター計算
        self.data['SMA_5'] = calculate_sma(self.data, 'Adj Close', 5)
        self.data['SMA_25'] = calculate_sma(self.data, 'Adj Close', 25)
        
    def generate_entry_signal(self, idx):
        """エントリーシグナル生成（緩和条件）"""
        try:
            # データ範囲チェック（前日データ使用のため idx-1 >= 25、翌日始値取得のため idx < len-1）
            if idx < 26 or idx >= len(self.data) - 1:
                return 0
            
            # ルックアヘッドバイアス回避：前日データで判定
            prev_price = self.data['Adj Close'].iloc[idx - 1]
            prev_volume = self.data['Volume'].iloc[idx - 1]
            prev_prev_volume = self.data['Volume'].iloc[idx - 2] if idx > 1 else prev_volume
            
            prev_sma_5 = self.data['SMA_5'].iloc[idx - 1]
            prev_sma_25 = self.data['SMA_25'].iloc[idx - 1]
            
            # ブレイクアウト条件（前日データ使用、条件を大幅緩和）
            # 従来: prev_price > prev_sma_5 and prev_sma_5 > prev_sma_25
            # 緩和: prev_price > prev_sma_25 のみ（SMA5>SMA25条件を削除）
            price_breakout = prev_price > prev_sma_25
            
            # ボリューム条件（前日データ使用、0.8に緩和済み）
            volume_condition = (prev_volume / prev_prev_volume) >= self.volume_threshold if prev_prev_volume > 0 else True  # 0除算時はTrue
            
            # デバッグログ（前日データ使用）
            volume_ratio = prev_volume / prev_prev_volume if prev_prev_volume > 0 else 0
            self.logger.debug(f"[RELAXED] idx={idx}, prev_price={prev_price:.2f}, prev_SMA5={prev_sma_5:.2f}, prev_SMA25={prev_sma_25:.2f}")
            self.logger.debug(f"[RELAXED] volume_ratio={volume_ratio:.2f}, threshold={self.volume_threshold}")
            self.logger.debug(f"[RELAXED] price_breakout={price_breakout} (緩和: prev_price > prev_sma_25), volume_condition={volume_condition}")
            
            if price_breakout and volume_condition:
                self.logger.info(f"[RELAXED_ENTRY] idx={idx}で緩和条件エントリーシグナル発生！（price>{prev_sma_25:.2f}, volume_ratio={volume_ratio:.2f}）")
                return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"[RELAXED] Entry signal error at idx {idx}: {e}")
            return 0
    
    def generate_exit_signal(self, idx, entry_price, entry_idx):
        """エグジットシグナル生成（変更なし）"""
        try:
            if idx >= len(self.data):
                return 1  # データ終了で強制決済
            
            current_price = self.data['Adj Close'].iloc[idx]
            
            # Take profit
            if current_price >= entry_price * (1 + self.take_profit):
                return 1
            
            # Stop loss (trailing stop)
            if current_price <= entry_price * (1 - self.trailing_stop):
                return 1
            
            # Death cross
            sma_5 = self.data['SMA_5'].iloc[idx]
            sma_25 = self.data['SMA_25'].iloc[idx]
            
            if sma_5 < sma_25:
                return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"[RELAXED] Exit signal error at idx {idx}: {e}")
            return 1
    
    def backtest_daily(self, current_date, stock_data=None, existing_position=None, **kwargs):
        """日次バックテスト（DSSMSから呼び出される）- 完全な取引サイクル対応"""
        try:
            self.logger.debug(f"[RELAXED_DAILY] backtest_daily開始: current_date={current_date}")
            
            # Phase 1: current_dateの型変換・検証
            if isinstance(current_date, str):
                current_date = pd.Timestamp(current_date)
            elif not isinstance(current_date, pd.Timestamp):
                current_date = pd.Timestamp(current_date)
                
            self.logger.debug(f"[RELAXED_DAILY] 型変換完了: current_date={current_date}")
                
            # Phase 2: データ整合性チェック（詳細ログ付き）
            self.logger.debug(f"[DATA_DEBUG] self.data.shape: {self.data.shape}")
            self.logger.debug(f"[DATA_DEBUG] self.data.index type: {type(self.data.index[0]) if len(self.data.index) > 0 else 'Empty'}")
            self.logger.debug(f"[DATA_DEBUG] self.data.index range: {self.data.index[0]} ~ {self.data.index[-1]}")
            self.logger.debug(f"[DATA_DEBUG] self.data.columns: {self.data.columns.tolist()}")
            self.logger.debug(f"[DATA_DEBUG] current_date: {current_date} ({type(current_date)})")
            
            # タイムゾーン不整合修正: current_dateをタイムゾーン付きに変換
            if self.data.index.tz is not None and current_date.tz is None:
                current_date = current_date.tz_localize(self.data.index.tz)
                self.logger.debug(f"[TIMEZONE_FIX] current_dateにタイムゾーン追加: {current_date}")
                
            if current_date not in self.data.index:
                self.logger.warning(f"[RELAXED_DAILY] データなし: {current_date}")
                self.logger.warning(f"[DATA_DEBUG] self.data.indexの最後の5件: {self.data.index[-5:].tolist()}")
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'BreakoutRelaxed: No data available for {current_date.strftime("%Y-%m-%d")}'
                }
            
            # Phase 2.5: 既存ポジション処理（エグジット判定）
            idx = self.data.index.get_loc(current_date)
            self.logger.debug(f"[RELAXED_DAILY] データインデックス取得: idx={idx}, date={current_date}")
            
            if existing_position is not None:
                self.logger.info(f"[RELAXED_DAILY] 既存ポジション確認: {existing_position}")
                
                # エグジットシグナル判定
                entry_price = existing_position.get('entry_price', 0)
                entry_idx = existing_position.get('entry_idx', idx - 1)
                shares = existing_position.get('shares', existing_position.get('quantity', 0))  # sharesまたはquantity
                is_force_close = existing_position.get('force_close', False)  # 銘柄切替フラグ
                
                # Cycle 7修正: force_close時はentry_symbolのデータから終値を取得
                if is_force_close:
                    # Cycle 7: entry_symbol_dataが渡されていればそのデータから終値取得
                    entry_symbol_data = kwargs.get('entry_symbol_data', None)
                    entry_symbol = existing_position.get('entry_symbol', '')
                    
                    if entry_symbol_data is not None and len(entry_symbol_data) > 0:
                        # entry_symbolの実データから前日終値を取得（ルックアヘッドバイアス回避）
                        # current_dateがentry_symbol_dataに存在するか確認
                        if current_date in entry_symbol_data.index:
                            entry_symbol_idx = entry_symbol_data.index.get_loc(current_date)
                            if entry_symbol_idx > 0:
                                exit_price = entry_symbol_data['Adj Close'].iloc[entry_symbol_idx - 1] * (1 - self.slippage)
                            else:
                                exit_price = entry_symbol_data['Adj Close'].iloc[entry_symbol_idx] * (1 - self.slippage)
                            
                            self.logger.info(
                                f"[CYCLE7_FORCE_CLOSE] {current_date.strftime('%Y-%m-%d')}: entry_symbol={entry_symbol}の実データから決済, "
                                f"entry_price={entry_price:.2f}, exit_price={exit_price:.2f}, shares={shares}"
                            )
                        else:
                            # current_dateがデータになければ最終日の終値を使用
                            exit_price = entry_symbol_data['Adj Close'].iloc[-1] * (1 - self.slippage)
                            self.logger.warning(
                                f"[CYCLE7_FORCE_CLOSE] {current_date.strftime('%Y-%m-%d')}がentry_symbolデータに存在せず、最終日終値使用: "
                                f"exit_price={exit_price:.2f}"
                            )
                    else:
                        # entry_symbol_dataがない場合はフォールバック（前日終値）
                        self.logger.warning(f"[CYCLE7_FORCE_CLOSE] entry_symbol_dataなし、フォールバック: 現在銘柄の前日終値使用")
                        if idx > 0:
                            exit_price = self.data['Adj Close'].iloc[idx - 1] * (1 - self.slippage)
                        else:
                            exit_price = self.data['Adj Close'].iloc[idx] * (1 - self.slippage)
                    
                    return {
                        'action': 'sell',
                        'signal': -1,
                        'price': exit_price,
                        'shares': shares,
                        'reason': f'BreakoutRelaxed: Force close (symbol switch from {entry_symbol})',
                        'status': 'force_closed'  # execution_detailsで識別用
                    }
                
                # 通常エグジット判定
                exit_signal = self.generate_exit_signal(idx, entry_price, entry_idx)
                self.logger.info(f"[RELAXED_DAILY] エグジットシグナル判定: {exit_signal} (idx={idx})")
                
                if exit_signal > 0:
                    # エグジット価格は当日始値（現実的取引）
                    exit_price = self.data['Open'].iloc[idx] * (1 - self.slippage)  # 売りスリッページ
                    
                    self.logger.info(f"[RELAXED_DAILY] {current_date.strftime('%Y-%m-%d')}: SELLシグナル, price={exit_price:.2f}, shares={shares}")
                    
                    return {
                        'action': 'sell',
                        'signal': -1,
                        'price': exit_price,
                        'shares': shares,
                        'reason': 'BreakoutRelaxed: Exit signal triggered'
                    }
                else:
                    # ポジション継続
                    self.logger.debug(f"[RELAXED_DAILY] ポジション継続: exit_signal={exit_signal}")
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': 'BreakoutRelaxed: Position held'
                    }
            
            # Phase 3: 新規エントリー判定（ポジションなしの場合のみ）
            entry_signal = self.generate_entry_signal(idx)
            self.logger.info(f"[RELAXED_DAILY] エントリーシグナル判定結果: {entry_signal} (idx={idx})")
            
            if entry_signal > 0:
                # エントリー価格は翌日始値（ルックアヘッドバイアス回避）
                if idx < len(self.data) - 1:
                    next_day_open = self.data['Open'].iloc[idx + 1]
                    entry_price = next_day_open * (1 + self.slippage)
                    
                    # Cycle 4-2: 実残高での株数計算（2026-01-10修正）
                    # 修正前: 固定1,000,000円でshares計算（残高管理なし）
                    # 修正後: available_cash引数から実際の現金残高を使用
                    available_cash = kwargs.get('available_cash', 1000000)  # デフォルトは初期資金
                    shares_raw = int(available_cash / entry_price)
                    shares = (shares_raw // 100) * 100  # 100株単位に調整
                    
                    self.logger.info(
                        f"[RELAXED_DAILY] {current_date.strftime('%Y-%m-%d')}: BUYシグナル, price={entry_price:.2f}, shares={shares}, "
                        f"available_cash={available_cash:,.0f}円"
                    )
                    
                    return {
                        'action': 'buy',
                        'signal': 1,
                        'price': entry_price,
                        'shares': shares,
                        'entry_idx': idx,  # エグジット判定用
                        'reason': 'BreakoutRelaxed: Relaxed volume condition met'
                    }
                else:
                    self.logger.warning(f"[RELAXED_DAILY] 翌日データなし: idx={idx}, len={len(self.data)}")
            
            self.logger.debug(f"[RELAXED_DAILY] HOLDを返す: entry_signal={entry_signal}")
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': 'BreakoutRelaxed: No entry signal'
            }
            
        except Exception as e:
            self.logger.error(f"[RELAXED_DAILY] backtest_daily error for {current_date}: {e}", exc_info=True)
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'BreakoutRelaxed: Error - {str(e)}'
            }