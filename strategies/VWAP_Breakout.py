"""
Module: VWAP_Breakout
File: VWAP_Breakout.py
Description: 
  出来高加重平均価格(VWAP)のブレイクアウトを検出する戦略クラスを実装しています。
  市場全体の上昇トレンドを確認しながら、株価がVWAPを上抜けし、出来高増加と
  テクニカル指標の確認による複合的な判断でエントリー・イグジットします。

Author: kouseven7
Created: 2023-02-15
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.volume_analysis
  - indicators.momentum_indicators
"""

import pandas as pd
import numpy as np
import sys
import logging
import math

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend
from indicators.basic_indicators import calculate_sma, calculate_vwap
from indicators.volume_analysis import detect_volume_increase
from indicators.momentum_indicators import calculate_macd
from indicators.basic_indicators import calculate_rsi

# Loggerの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class VWAPBreakoutStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, index_data: pd.DataFrame = None, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        VWAPアウトブレイク戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            index_data (pd.DataFrame, optional): 市場全体のインデックスデータ（Noneの場合はstock_dataを使用）
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        # 戦略固有の属性を先に設定
        # TODO #12: index_dataがNoneの場合は株価データをコピーして使用（緊急対応）
        if index_data is None:
            index_data = data.copy()
            logger.info("VWAPBreakoutStrategy: index_data not provided, using stock_data as proxy")
        
        self.index_data = index_data
        self.price_column = price_column
        self.volume_column = volume_column
        
        # デフォルトパラメータの設定
        default_params = {
            # --- リスクリワード重視 ---
            "stop_loss": 0.03,    # 3% ストップロス（浅め～標準）
            "take_profit": 0.15,  # 15% 利益確定（広め）

            # --- エントリー頻度調整 ---
            "sma_short": 10,      # 短期移動平均
            "sma_long": 30,       # 長期移動平均
            "volume_threshold": 1.2, # 出来高増加（やや緩め）

            # --- シンプル化 ---
            "confirmation_bars": 1,             # ブレイク確認バー数 (0=即時エントリー, 1=1本確認)
            "breakout_min_percent": 0.003,      # 最小ブレイク率 (0=無効化, 0.3%有効)
            "trailing_stop": 0.05,              # トレーリングストップ（やや広め）
            "trailing_start_threshold": 0.03,   # トレーリング開始閾値 (3%の利益でトレーリング開始)
            "max_holding_period": 10,           # 最大保有期間 (日数)

            # --- フィルター・特殊機能は無効化 ---
            "market_filter_method": "none",    # 市場フィルター方式 (none=無効, sma=シンプル, macd=MACD)
            "rsi_filter_enabled": False,        # RSIフィルターの有効化
            "atr_filter_enabled": False,        # ATRフィルターの有効化
            "partial_exit_enabled": False,      # 部分利確の有効化（無効）
            "partial_exit_threshold": 0.07,     # 部分利確の閾値（7%でデフォルト）
            "partial_exit_portion": 0.5,        # 部分利確の割合（50%でデフォルト）

            # --- その他（将来拡張用・固定値） ---
            "rsi_period": 14,                   # RSI計算期間
            "volume_increase_mode": "simple", # 出来高増加判定方式 (simple/average/exponential)

            # --- Phase 2: スリッページ・取引コスト（2025-12-23追加） ---
            "slippage": 0.001,               # スリッページ（0.1%、買い注文は不利な方向）
            "transaction_cost": 0.0          # 取引コスト（0%、オプション）
        }
        
        # 親クラスの初期化（デフォルトパラメータとユーザーパラメータをマージ）
        merged_params = {**default_params, **(params or {})}
        super().__init__(data, merged_params)
        
    def initialize_strategy(self):
        """
        戦略の初期化処理
        """
        super().initialize_strategy()
        
        # 必要なインジケーターを計算してデータに追加
        sma_short = self.params["sma_short"]
        sma_long = self.params["sma_long"]
        rsi_period = self.params["rsi_period"]
        
        # ルックアヘッドバイアス修正: 全てのインジケーターにshift(1)を適用して前日の値を使用
        self.data['SMA_' + str(sma_short)] = calculate_sma(self.data, self.price_column, sma_short).shift(1)
        self.data['SMA_' + str(sma_long)] = calculate_sma(self.data, self.price_column, sma_long).shift(1)
        self.data['VWAP'] = calculate_vwap(self.data, self.price_column, self.volume_column).shift(1)
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], rsi_period).shift(1)
        
        # MACDも同様にshift(1)を適用
        macd_raw, signal_raw = calculate_macd(self.data, self.price_column)
        self.data['MACD'] = macd_raw.shift(1)
        self.data['Signal_Line'] = signal_raw.shift(1)

        # 市場全体のトレンドを確認するためのインデックスの移動平均線
        if self.index_data is not None:
            self.index_data['SMA_' + str(sma_short)] = calculate_sma(self.index_data, self.price_column, sma_short).shift(1)
            self.index_data['SMA_' + str(sma_long)] = calculate_sma(self.index_data, self.price_column, sma_long).shift(1)

        logger.info(f"[init] data.columns: {self.data.columns.tolist()}")
        logger.info(f"[init] data.index[:5]: {self.data.index[:5].tolist()}")
        logger.info(f"[init] data.shape: {self.data.shape}")

    def is_market_uptrend(self, idx: int) -> bool:
        """
        市場全体が上昇トレンドにあるかを確認する。
        market_filter_methodパラメータに基づいて、異なるトレンド判定方法を使用します。
        """
        # パラメータで指定されたフィルター方式を取得
        filter_method = self.params.get("market_filter_method", "none")
        
        # フィルターが無効な場合は常にTrue
        if filter_method == "none":
            return True
            
        # インデックスデータが利用できない場合は常にTrue
        if self.index_data is None:
            logger.warning(f"[market_trend] インデックスデータが利用できないため、市場フィルターをスキップします")
            return True
            
        # 共通の検証: 十分なデータがあるか
        sma_short = self.params["sma_short"]
        sma_long = self.params["sma_long"]
        if idx < sma_long or len(self.index_data) <= idx:
            # データ不足の場合、安全にTrue
            return True
            
        # SMA方式: 短期SMA > 長期SMA
        if filter_method == "sma":
            sma_short_key = f'SMA_{sma_short}'
            sma_long_key = f'SMA_{sma_long}'
            
            # インデックスデータに必要なSMAがあることを確認
            if sma_short_key not in self.index_data.columns or sma_long_key not in self.index_data.columns:
                logger.warning(f"[market_trend] インデックスデータにSMAカラムがありません: {sma_short_key}, {sma_long_key}")
                return True
                
            # 上昇トレンド判定: 短期SMA > 長期SMA
            index_sma_short = self.index_data[sma_short_key].iloc[idx]
            index_sma_long = self.index_data[sma_long_key].iloc[idx]
            
            uptrend = index_sma_short > index_sma_long
            if not uptrend:
                logger.debug(f"[market_trend] idx={idx}: 市場SMAトレンドNG: short={index_sma_short}, long={index_sma_long}")
            return uptrend
            
        # MACD方式: MACD > シグナルライン
        elif filter_method == "macd":
            # インデックスデータにMACDがあることを確認
            if 'MACD' not in self.index_data.columns or 'Signal_Line' not in self.index_data.columns:
                # MACDがなければ計算
                if self.price_column in self.index_data.columns:
                    from indicators.momentum_indicators import calculate_macd
                    self.index_data['MACD'], self.index_data['Signal_Line'] = calculate_macd(self.index_data, self.price_column)
                else:
                    logger.warning(f"[market_trend] インデックスデータにMACDを計算できません")
                    return True
                    
            # 上昇トレンド判定: MACD > シグナルライン
            macd = self.index_data['MACD'].iloc[idx]
            signal = self.index_data['Signal_Line'].iloc[idx]
            
            uptrend = macd > signal
            if not uptrend:
                logger.debug(f"[market_trend] idx={idx}: 市場MACDトレンドNG: MACD={macd}, Signal={signal}")
            return uptrend
            
        # RSIプラス方式: RSI > 50 (加えてSMAチェックも)
        elif filter_method == "rsi_plus":
            # 基本的なSMAチェック
            sma_short_key = f'SMA_{sma_short}'
            sma_long_key = f'SMA_{sma_long}'
            
            if 'RSI' not in self.index_data.columns:
                # RSIがなければ計算
                if self.price_column in self.index_data.columns:
                    from indicators.basic_indicators import calculate_rsi
                    self.index_data['RSI'] = calculate_rsi(self.index_data[self.price_column], self.params["rsi_period"])
                else:
                    logger.warning(f"[market_trend] インデックスデータにRSIを計算できません")
                    return True
            
            # インデックスデータに必要なSMAがあることを確認
            if sma_short_key not in self.index_data.columns or sma_long_key not in self.index_data.columns:
                logger.warning(f"[market_trend] インデックスデータにSMAカラムがありません: {sma_short_key}, {sma_long_key}")
                return True
                
            # 上昇トレンド判定: RSI > 50 および 短期SMA > 長期SMA
            rsi = self.index_data['RSI'].iloc[idx]
            index_sma_short = self.index_data[sma_short_key].iloc[idx]
            index_sma_long = self.index_data[sma_long_key].iloc[idx]
            
            uptrend = rsi > 50 and index_sma_short > index_sma_long
            if not uptrend:
                logger.debug(f"[market_trend] idx={idx}: 市場RSIプラストレンドNG: RSI={rsi}, short={index_sma_short}, long={index_sma_long}")
            return uptrend
            
        # デフォルトでは常にTrue
        return True

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件を満たさない場合は理由をDEBUGログに出す（デバッグ用）
        
        Issue調査報告20260210修正: ウォームアップ期間フィルタリング追加
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
                    logger.debug(
                        f"[WARMUP_SKIP] ウォームアップ期間のためエントリースキップ: "
                        f"{current_date_at_idx.strftime('%Y-%m-%d')} < {trading_start_ts.strftime('%Y-%m-%d')}"
                    )
                    return 0  # エントリー禁止
            except Exception as e:
                logger.warning(f"[WARMUP_FILTER_ERROR] trading_start_date比較エラー: {e}")
        
        try:
            # 必要なデータ準備
            sma_short_key = 'SMA_' + str(self.params["sma_short"])
            sma_long_key = 'SMA_' + str(self.params["sma_long"])
            
            # 最小データ量チェックを緩和（最適化のためより多くの取引機会を得る）
            min_required = max(self.params["sma_long"], 20)  # 少なくともsma_longか20日分は必要
            if idx < min_required:
                logger.debug(f"[entry] idx={idx}: データ不足 (必要期間={min_required})")
                return 0
                
            # 基本データ取得
            current_price = self.data[self.price_column].iloc[idx]
            sma_short = self.data[sma_short_key].iloc[idx]
            sma_long = self.data[sma_long_key].iloc[idx]
            vwap = self.data['VWAP'].iloc[idx]
            previous_vwap = self.data['VWAP'].iloc[idx - 1]
            current_volume = self.data[self.volume_column].iloc[idx]
            previous_volume = self.data[self.volume_column].iloc[idx - 1]
            
            # Series型のままの場合はスカラー化（Phase 2修正：2025-12-23）
            if isinstance(current_price, pd.Series):
                current_price = current_price.values[0]
            if isinstance(sma_short, pd.Series):
                sma_short = sma_short.values[0]
            if isinstance(sma_long, pd.Series):
                sma_long = sma_long.values[0]
            if isinstance(vwap, pd.Series):
                vwap = vwap.values[0]
            if isinstance(previous_vwap, pd.Series):
                previous_vwap = previous_vwap.values[0]
            if isinstance(current_volume, pd.Series):
                current_volume = current_volume.values[0]
            if isinstance(previous_volume, pd.Series):
                previous_volume = previous_volume.values[0]
            
            # RSIフィルターが有効な場合
            if self.params.get("rsi_filter_enabled", False):
                rsi = self.data['RSI'].iloc[idx]
                if rsi > 70:  # 買われ過ぎ状態ではエントリーしない
                    logger.debug(f"[entry] idx={idx}: RSIフィルターNG (RSI={rsi})")
                    return 0
            
            # 市場全体が上昇トレンドにあるか確認
            if not self.is_market_uptrend(idx):
                logger.debug(f"[entry] idx={idx}: 市場トレンドNG")
                return 0
                
            # トレンド確認 - 価格がSMAより上にあるかで判定
            # 緩和条件: (current_price > sma_long)だけでチェック
            if not (current_price > sma_long):
                logger.debug(f"[entry] idx={idx}: MA順序NG(緩和) current={current_price}, sma_long={sma_long}")
                return 0
            # 移動平均線の条件 - 短期MAが直近より上か、長期MAが直近より上のいずれか（条件緩和）
            ma_rising = (sma_short > self.data[sma_short_key].iloc[idx - 1]) or (sma_long > self.data[sma_long_key].iloc[idx - 1])
            if not ma_rising:
                logger.debug(f"[entry] idx={idx}: MA上昇NG")
                return 0
                
            # VWAPの条件 - 強い上昇トレンドでは必ずしもVWAP上昇を要求しない
            if current_price < sma_short and not (vwap > previous_vwap):
                logger.debug(f"[entry] idx={idx}: VWAP上昇NG vwap={vwap}, prev={previous_vwap}")
                return 0
                
            # VWAPブレイク条件 - 以下の2つパターンを検出
            # 1. 直接的なブレイクアウト: 現在価格 > VWAP かつ 前日価格 <= VWAP
            # 2. または、既にVWAP上にあり、一定以上の上昇: 現在価格 > VWAP * 1.003
            vwap_breakout = (current_price > vwap and self.data[self.price_column].iloc[idx - 1] <= vwap)
            vwap_above_threshold = (current_price > vwap * (1 + self.params.get("breakout_min_percent", 0.003)))
            
            if not (vwap_breakout or vwap_above_threshold):
                logger.debug(f"[entry] idx={idx}: VWAPブレイクNG current={current_price}, vwap={vwap}")
                return 0
                
            # ブレイク持続確認（確認バーが0の場合はスキップ）
            if self.params.get("confirmation_bars", 0) > 0:
                confirm_ok = True
                for i in range(1, min(self.params["confirmation_bars"] + 1, idx + 1)):
                    if self.data[self.price_column].iloc[idx - i] <= vwap:
                        logger.debug(f"[entry] idx={idx}: 確認バーNG idx-i={idx-i}")
                        confirm_ok = False
                        break
                if not confirm_ok:
                    return 0
                    
            # ATRフィルターが有効な場合
            if self.params.get("atr_filter_enabled", False):
                if 'BB_Std' in self.data.columns:  # ボリンジャーバンドのStdを代用
                    bb_std = self.data['BB_Std'].iloc[idx]
                    price_range = self.data['High'].iloc[idx] - self.data['Low'].iloc[idx]
                    if price_range < bb_std * 0.5:  # ボラティリティが低すぎる
                        logger.debug(f"[entry] idx={idx}: ATRフィルターNG 日中レンジ={price_range}, BB標準偏差={bb_std}")
                        return 0
                    
            # 出来高増加条件
            volume_mode = self.params.get("volume_increase_mode", "simple")
            if volume_mode == "simple":
                vol_increase = current_volume >= previous_volume * self.params["volume_threshold"]
            elif volume_mode == "average":
                vol_avg = self.data[self.volume_column].iloc[idx-5:idx].mean()
                vol_increase = current_volume >= vol_avg * self.params["volume_threshold"]
            else:
                vol_increase = detect_volume_increase(current_volume, previous_volume, threshold=self.params["volume_threshold"])
                
            if not vol_increase:
                logger.debug(f"[entry] idx={idx}: 出来高増加NG current={current_volume}, prev={previous_volume}")
                return 0
            logger.info(f"VWAP Breakout エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}")
            return 1
        except Exception as e:
            logger.error(f"[entry] idx={idx}: 例外発生: {e}", exc_info=True)
            return 0

    def generate_exit_signal(self, idx: int, entry_idx: int = None) -> int:
        """
        イグジットシグナルを生成する。
        
        Parameters:
            idx (int): 現在のインデックス
            entry_idx (int): エントリー時のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1 or entry_idx is None:
            return 0
        
        # Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
        # 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
        # リアルトレードでは翌日（idx+1日目）の始値でイグジット
        current_price = self.data['Open'].iloc[idx + 1]
        
        vwap = self.data['VWAP'].iloc[idx]
        entry_price = self.data[self.price_column].iloc[entry_idx]
        # ATR（代用としてVWAPの2%）
        atr = vwap * 0.02
        # VWAPを下回った場合
        if current_price < vwap:
            self.log_trade(f"VWAP Breakout イグジットシグナル: VWAP下抜け 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        # ストップロス条件
        if current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"VWAP Breakout イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        # 利益確定条件
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"VWAP Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        # 高度なトレーリングストップ
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= self.params.get("trailing_start_threshold", 0):
            # Phase 1b修正: idx日目の高値を除外（idx-1日目までの高値を使用）
            high_since_entry = self.data['High'].iloc[entry_idx:idx].max()
            trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
            if current_price <= trailing_stop:
                self.log_trade(f"VWAP Breakout イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
                return -1
        # 部分利確ロジック
        if self.params.get("partial_exit_enabled", False):
            # 必要なパラメータが存在するか確認
            if "partial_exit_threshold" in self.params and "partial_exit_portion" in self.params:
                profit_pct = (current_price - entry_price) / entry_price
                partial_exit_threshold = self.params.get("partial_exit_threshold", 0.07)  # デフォルト値: 7%
                if profit_pct >= partial_exit_threshold and 'Partial_Exit' not in self.data.columns:
                    self.data['Partial_Exit'] = 0
                    self.data.at[self.data.index[idx], 'Partial_Exit'] = 1
                    partial_exit_portion = self.params.get("partial_exit_portion", 0.5)  # デフォルト値: 50%
                    self.log_trade(f"VWAP Breakout 部分利確シグナル: {partial_exit_portion*100}% 利確 日付={self.data.index[idx]}, 価格={current_price}")
                    return 0
            else:
                # 必要なパラメータがない場合はログを出力して処理をスキップ
                logger.warning(f"部分利確が有効ですが、必要なパラメータ(partial_exit_threshold/partial_exit_portion)が設定されていません")
        # RSIやMACDの反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:
            self.log_trade(f"VWAP Breakout イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        if macd < signal_line and self.data['MACD'].iloc[idx-1] >= self.data['Signal_Line'].iloc[idx-1]:
            self.log_trade(f"VWAP Breakout イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        return 0

    def backtest(self, trading_start_date=None, trading_end_date=None):
        """
        VWAPアウトブレイク戦略のバックテストを実行する。
        
        Parameters:
            trading_start_date (datetime, optional): 取引開始日（この日以降にシグナル生成開始）
            trading_end_date (datetime, optional): 取引終了日（この日以前までシグナル生成）
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        self.data['Position'] = 0  # ポジション状態を追加
        self.data['Entry_Price'] = np.nan  # エントリー価格を記録
        
        # エントリーインデックスを整数型として初期化（NaNではなく-1を使用）
        self.data['Entry_Idx'] = pd.Series(-1, index=self.data.index, dtype='int64')  # 明示的に整数型を指定

        # バックテストループ（Phase 1修正: 最終日を除外してidx+1アクセスを安全に）
        for idx in range(len(self.data) - 1):
            current_price = self.data[self.price_column].iloc[idx]
            
            # 取引期間フィルタリング（BaseStrategy.backtest()と同じロジック）
            if trading_start_date is not None or trading_end_date is not None:
                current_date = self.data.index[idx]
                in_trading_period = True
                
                if trading_start_date is not None and current_date < trading_start_date:
                    in_trading_period = False
                if trading_end_date is not None and current_date > trading_end_date:
                    in_trading_period = False
                
                if not in_trading_period:
                    # 取引期間外はポジション状態のみ伝播
                    if idx > 0:
                        self.data.loc[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                        if self.data['Position'].iloc[idx-1] == 1:
                            self.data.loc[self.data.index[idx], 'Entry_Idx'] = self.data['Entry_Idx'].iloc[idx-1]
                    continue
            
            # 前日までのポジション状態を確認
            if idx > 0:
                self.data.loc[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                # Entry_Idxも伝播（ポジション保有中は同じEntry_Idxを維持）
                if self.data['Position'].iloc[idx-1] == 1:
                    self.data.loc[self.data.index[idx], 'Entry_Idx'] = self.data['Entry_Idx'].iloc[idx-1]
            
            # ポジションがない場合、エントリーシグナルをチェック
            if self.data['Position'].iloc[idx] == 0:
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    # エントリーシグナルあり
                    self.data.loc[self.data.index[idx], 'Entry_Signal'] = 1
                    self.data.loc[self.data.index[idx], 'Position'] = 1
                    
                    # Phase 1修正: Entry_Priceを翌日始値に変更（ルックアヘッドバイアス修正）
                    # Phase 2修正: スリッページ適用（2025-12-23追加）
                    next_day_open = self.data['Open'].iloc[idx + 1]
                    
                    # Series型のままの場合はスカラー化
                    if isinstance(next_day_open, pd.Series):
                        next_day_open = next_day_open.values[0]
                    
                    # スリッページ・取引コスト適用
                    slippage = self.params.get("slippage", 0.001)
                    transaction_cost = self.params.get("transaction_cost", 0.0)
                    entry_price = next_day_open * (1 + slippage + transaction_cost)
                    
                    self.data.loc[self.data.index[idx], 'Entry_Price'] = entry_price
                    self.data.loc[self.data.index[idx], 'Entry_Idx'] = idx
            
            # ポジションがある場合、イグジットシグナルをチェック
            elif self.data['Position'].iloc[idx] == 1:
                # エントリーインデックスを取得（伝播により正しく設定されているはず）
                entry_idx_val = self.data['Entry_Idx'].iloc[idx]
                
                # Entry_Idxが無効な場合はエラー（フォールバック削除、copilot-instructions.md準拠）
                if pd.isna(entry_idx_val) or entry_idx_val == -1:
                    raise ValueError(
                        f"Entry_Idx伝播エラー: idx={idx}, 日付={self.data.index[idx]}, "
                        f"entry_idx_val={entry_idx_val}. ポジション保有中にEntry_Idxが無効です。"
                    )
                
                entry_idx = int(entry_idx_val)
                
                # エントリー後の最大保有期間チェック
                days_held = idx - entry_idx
                if days_held >= self.params.get("max_holding_period", 10):
                    # 最大保有期間に達したらイグジット
                    self.data.loc[self.data.index[idx], 'Exit_Signal'] = -1
                    self.data.loc[self.data.index[idx], 'Position'] = 0
                    self.log_trade(f"VWAP Breakout イグジットシグナル: 最大保有期間到達 ({days_held}日) 日付={self.data.index[idx]}, 価格={current_price}")
                else:
                    # 通常のイグジットシグナルをチェック
                    exit_signal = self.generate_exit_signal(idx, entry_idx)
                    if exit_signal == -1:
                        self.data.loc[self.data.index[idx], 'Exit_Signal'] = -1
                        self.data.loc[self.data.index[idx], 'Position'] = 0
        
        return self.data

    def backtest_daily(self, current_date, stock_data, existing_position=None, trading_start_date=None, **kwargs):
        """
        VWAPBreakoutStrategy 日次バックテスト実行
        
        Phase 3-A Step A2実装: VWAPBreakout戦略での実証実装
        
        Cycle 26修正: **kwargs追加
        - 理由: force_close時にentry_symbol_dataがkwargsで渡される（Cycle 7修正）
        
        Cycle 27修正: entry_symbol_data使用
        - force_close時はentry_symbol_data（元の銘柄）でエグジット価格を取得
        
        Sprint 1.5修正: force_close強制決済実装（2026-02-09）
        - 銘柄切替時（existing_position['force_close']=True）は無条件で決済
        - generate_exit_signal()をスキップし、即座にエグジット
        - 旧銘柄のentry_symbol_dataで翌日始値決済（ルックアヘッドバイアス防止）
        - [VWAP_FORCE_CLOSE]ログタグで追跡可能
        
        Parameters:
            current_date (datetime): 判定対象日
            stock_data (pd.DataFrame): 最新の株価データ
            existing_position (dict, optional): 既存ポジション情報
                - force_close (bool): 強制決済フラグ（銘柄切替時True）
                - entry_symbol (str): エントリー時の銘柄コード
                - entry_price (float): エントリー価格
                - quantity (int): 保有数量
            **kwargs: 追加引数
                - entry_symbol_data (pd.DataFrame): 旧銘柄の価格データ（force_close時必須）
            
        Returns:
            dict: {
                'action': 'entry'|'exit'|'hold',
                'signal': 1|-1|0,
                'price': float,
                'shares': int,
                'reason': str
            }
            
        実装内容:
        1. current_dateのindexを特定
        2. ウォームアップ期間考慮（150日）
        3. 前日データのみでVWAPインジケーター計算（shift(1)適用済み）
        4. エントリー判定（ルックアヘッドバイアス防止）
        5. 翌日始値エントリー価格設定
        """
        from datetime import timedelta
        import pandas as pd
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Issue調査報告20260210修正: trading_start_dateを保存（generate_entry_signal()で使用）
        self.trading_start_date = trading_start_date
        if trading_start_date is not None:
            logger.info(f"[WARMUP_FILTER] trading_start_date設定: {trading_start_date.strftime('%Y-%m-%d') if hasattr(trading_start_date, 'strftime') else trading_start_date}")
        
        # Phase 1: current_dateの型変換・検証
        if isinstance(current_date, str):
            current_date = pd.Timestamp(current_date)
        elif not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        # Cycle 24修正: タイムゾーン統一（Breakout.py Cycle 20, GCStrategy Cycle 23パターン）
        # 理由: stock_data.indexは+09:00タイムゾーン付き、current_dateはtz-naiveの可能性
        if current_date.tz is not None:
            current_date = current_date.tz_localize(None)
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)
        
        # Phase 2: データ整合性チェック
        if current_date not in stock_data.index:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'VWAPBreakout: No data available for {current_date.strftime("%Y-%m-%d")}'
            }
            
        # Phase 3: ウォームアップ期間考慮
        # Cycle 24修正: DSSMSがwarmup_days=150で既にデータ拡大済み（GCStrategy Cycle 23パターン）
        # 戦略はsma_long期間分のみ必要（Breakout.py: look_back=5, GCStrategy: long_window=25）
        current_idx = stock_data.index.get_loc(current_date)
        min_required = self.params.get("sma_long", 30)  # VWAPBreakout固有: sma_long期間
        
        if current_idx < min_required:
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'VWAPBreakout: Insufficient warmup data. Required: {min_required}, Available: {current_idx}'
            }
        
        # Phase 4: データ更新（Option B方式を活用）
        # 既存のself.dataを一時保存
        original_data = self.data.copy()
        
        try:
            # BaseStrategy.backtest_daily()の Option B ロジックを活用
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            updated_columns = []
            
            for col in basic_columns:
                if col in stock_data.columns and col in self.data.columns:
                    # インデックスが一致する部分のみ安全に更新
                    common_index = self.data.index.intersection(stock_data.index)
                    if len(common_index) > 0:
                        self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                        updated_columns.append(col)
            
            logger.debug(f"[VWAPBreakout.backtest_daily] Data updated: {updated_columns}")
            
            # Phase 5: 前日データで判定（ルックアヘッドバイアス防止）
            # 注意: インジケーターは既にinitialize_strategy()でshift(1)適用済み
            
            # Cycle 27修正: entry_symbol_dataをkwargsから取得
            entry_symbol_data = kwargs.get('entry_symbol_data', None)
            is_force_close = existing_position.get('force_close', False) if existing_position else False
            
            # force_close時はentry_symbol_dataを使用
            if is_force_close and entry_symbol_data is not None:
                data_for_exit = entry_symbol_data
                logger.info(f"[VWAP_EXIT] force_close=True: entry_symbol_dataを使用（{len(entry_symbol_data)}行）")
            else:
                data_for_exit = stock_data
            
            # 現在のポジション状態を確認
            if existing_position is not None:
                # ============================================================
                # Sprint 1.5修正: force_close強制決済（最優先処理）
                # ============================================================
                # 【削除禁止】このブロックは銘柄切替時の旧ポジション決済に必須
                # 削除すると、ポジションが残り続け、all_transactions.csvが不完全になる
                # 参照: MULTI_POSITION_IMPLEMENTATION_PLAN.md Sprint 1.5
                # ============================================================
                if is_force_close:
                    entry_symbol = existing_position.get('entry_symbol', 'Unknown')
                    entry_price = existing_position.get('entry_price', 0.0)
                    quantity = existing_position.get('quantity', 0)
                    
                    logger.warning(
                        f"[VWAP_FORCE_CLOSE] 銘柄切替による強制決済を実行\n"
                        f"  旧銘柄: {entry_symbol}\n"
                        f"  エントリー価格: {entry_price:.2f}円\n"
                        f"  保有数量: {quantity}株\n"
                        f"  決済日: {current_date.strftime('%Y-%m-%d')}"
                    )
                    
                    # 旧銘柄のデータで決済価格を取得
                    # Cycle 27修正: entry_symbol_data（旧銘柄）を使用
                    if entry_symbol_data is not None:
                        data_for_exit = entry_symbol_data
                        logger.info(f"[VWAP_FORCE_CLOSE] entry_symbol_dataを使用（{len(entry_symbol_data)}行、銘柄={entry_symbol}）")
                    else:
                        logger.error(f"[VWAP_FORCE_CLOSE] entry_symbol_dataが未提供。stock_dataで代替。")
                        data_for_exit = stock_data
                    
                    # 翌日始値で決済（ルックアヘッドバイアス防止）
                    exit_price = None
                    try:
                        if current_idx + 1 < len(data_for_exit):
                            exit_price = data_for_exit.iloc[current_idx + 1]['Open']
                            logger.info(f"[VWAP_FORCE_CLOSE] 翌日始値で決済: {exit_price:.2f}円")
                        else:
                            # 最終日フォールバック（copilot-instructions.md制約: 境界条件のみ許可）
                            exit_price = data_for_exit.iloc[current_idx]['Close']
                            logger.warning(
                                f"[VWAP_FORCE_CLOSE] 最終日のため終値で決済: {exit_price:.2f}円\n"
                                f"  （境界条件フォールバック: copilot-instructions.md準拠）"
                            )
                        
                        # 損益計算
                        pnl = (exit_price - entry_price) * quantity
                        pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0
                        
                        logger.warning(
                            f"[VWAP_FORCE_CLOSE] 強制決済完了\n"
                            f"  決済価格: {exit_price:.2f}円\n"
                            f"  損益: {pnl:,.0f}円 ({pnl_pct:+.2f}%)"
                        )
                        
                        # 強制決済を実行（早期return）
                        return {
                            'action': 'exit',
                            'signal': -1,
                            'price': float(exit_price),
                            'shares': quantity,
                            'reason': f'VWAPBreakout: Force close due to symbol switch on {current_date.strftime("%Y-%m-%d")}'
                        }
                    except Exception as e:
                        logger.error(
                            f"[VWAP_FORCE_CLOSE] 決済価格取得エラー: {e}\n"
                            f"  現在インデックス: {current_idx}\n"
                            f"  データ長: {len(data_for_exit)}"
                        )
                        # エラー時もエグジットを返す（ポジションを残さない）
                        fallback_price = existing_position.get('entry_price', 0.0)
                        logger.warning(f"[VWAP_FORCE_CLOSE] フォールバック: エントリー価格で決済 ({fallback_price:.2f}円)")
                        return {
                            'action': 'exit',
                            'signal': -1,
                            'price': float(fallback_price),
                            'shares': quantity,
                            'reason': f'VWAPBreakout: Force close (error fallback) on {current_date.strftime("%Y-%m-%d")}'
                        }
                # ============================================================
                # 通常のエグジット判定（force_close=Falseの場合のみ実行）
                # ============================================================
                # 既存ポジションあり: エグジット判定
                entry_idx = existing_position.get('entry_idx', current_idx)
                exit_signal = self.generate_exit_signal(current_idx, entry_idx)
                
                if exit_signal == -1:
                    # エグジットシグナル発生
                    try:
                        # Phase 6: 翌日始値でエグジット（ルックアヘッドバイアス防止）
                        # Cycle 27修正: data_for_exitを使用
                        if current_idx + 1 < len(data_for_exit):
                            exit_price = data_for_exit.iloc[current_idx + 1]['Open']
                        else:
                            # 最終日の場合（フォールバック: copilot-instructions.md制約により限定的使用）
                            exit_price = data_for_exit.iloc[current_idx]['Close']
                            logger.warning(f"[VWAPBreakout.backtest_daily] Using Close price fallback for final day: {current_date}")
                        
                        return {
                            'action': 'exit',
                            'signal': -1,
                            'price': float(exit_price),
                            'shares': existing_position.get('quantity', 0),
                            'reason': f'VWAPBreakout: Exit signal detected on {current_date.strftime("%Y-%m-%d")}'
                        }
                    except Exception as e:
                        logger.error(f"[VWAPBreakout.backtest_daily] Exit price calculation failed: {e}")
                        return {
                            'action': 'hold',
                            'signal': 0,
                            'price': 0.0,
                            'shares': 0,
                            'reason': f'VWAPBreakout: Exit price calculation error: {str(e)}'
                        }
                else:
                    # エグジットシグナルなし: ホールド
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': existing_position.get('quantity', 0),
                        'reason': f'VWAPBreakout: Holding position from {current_date.strftime("%Y-%m-%d")}'
                    }
            else:
                # 既存ポジションなし: エントリー判定
                entry_signal = self.generate_entry_signal(current_idx)
                
                if entry_signal == 1:
                    # エントリーシグナル発生
                    try:
                        # Phase 6: 翌日始値でエントリー + スリッページ（ルックアヘッドバイアス防止）
                        if current_idx + 1 < len(stock_data):
                            entry_price = stock_data.iloc[current_idx + 1]['Open']
                            
                            # スリッページ・取引コスト適用（copilot-instructions.md推奨0.1%）
                            slippage = self.params.get("slippage", 0.001)
                            transaction_cost = self.params.get("transaction_cost", 0.0)
                            entry_price = entry_price * (1 + slippage + transaction_cost)
                            
                            # 標準的な取引株数計算（資金の10%程度を想定）
                            shares = int(100000 / entry_price) if entry_price > 0 else 0
                            
                            return {
                                'action': 'entry',
                                'signal': 1,
                                'price': float(entry_price),
                                'shares': shares,
                                'reason': f'VWAPBreakout: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                            }
                        else:
                            # 最終日の場合（エントリー不可）
                            return {
                                'action': 'hold',
                                'signal': 0,
                                'price': 0.0,
                                'shares': 0,
                                'reason': f'VWAPBreakout: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                            }
                    except Exception as e:
                        logger.error(f"[VWAPBreakout.backtest_daily] Entry price calculation failed: {e}")
                        return {
                            'action': 'hold',
                            'signal': 0,
                            'price': 0.0,
                            'shares': 0,
                            'reason': f'VWAPBreakout: Entry price calculation error: {str(e)}'
                        }
                else:
                    # エントリーシグナルなし: ホールド
                    return {
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0,
                        'reason': f'VWAPBreakout: No entry signal on {current_date.strftime("%Y-%m-%d")}'
                    }
                    
        finally:
            # データの復元（元の状態に戻す）
            self.data = original_data
            
        # デフォルト: ホールド
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'VWAPBreakout: Default hold action for {current_date.strftime("%Y-%m-%d")}'
        }

def apply_strategies(stock_data: pd.DataFrame, index_data: pd.DataFrame):
    """
    複数の戦略を適用し、シグナルを生成します。
    """
    strategies = {
        "VWAP Breakout.py": VWAPBreakoutStrategy(stock_data, index_data),
        "Momentum Investing.py": MomentumInvestingStrategy(stock_data),
        "Breakout.py": BreakoutStrategy(stock_data)
    }

    signals = {}
    for strategy_name, strategy in strategies.items():
        signals[strategy_name] = strategy.generate_entry_signal(idx=len(stock_data) - 1)

    # 戦略の優先順位に基づいてシグナルをソート
    prioritized_strategies = risk_manager.prioritize_strategies(signals)

    # シグナルを適用
    for strategy_name in prioritized_strategies:
        if signals[strategy_name] == 1:  # シグナルが発生している場合
            if risk_manager.check_position_size(strategy_name):
                risk_manager.update_position(strategy_name, 1)
                logger.info(f"{strategy_name} のシグナルに基づきポジションを追加しました。")
            else:
                logger.info(f"{strategy_name} のポジションサイズが上限に達しているため、エントリーをスキップしました。")

def get_parameters_and_data():
    """
    Excel設定ファイルからパラメータ取得と市場データ取得（キャッシュ利用）を行います。
    Returns:
        ticker (str), start_date (str), end_date (str), stock_data (pd.DataFrame), index_data (pd.DataFrame)
    """
    from config.error_handling import read_excel_parameters, fetch_stock_data
    from config.cache_manager import get_cache_filepath, save_cache

    # 設定ファイルからパラメータを取得
    config_file = r"C:\Users\imega\Documents\my_backtest_project\config\backtest_config.xlsx"
    config_df = read_excel_parameters(config_file, "銘柄設定")
    ticker = config_df["銘柄"].iloc[0]
    start_date = config_df["開始日"].iloc[0].strftime('%Y-%m-%d')
    end_date = config_df["終了日"].iloc[0].strftime('%Y-%m-%d')
    logger.info(f"パラメータ取得: {ticker}, {start_date}, {end_date}")

    # データ取得
    cache_filepath = get_cache_filepath(ticker, start_date, end_date)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    save_cache(stock_data, cache_filepath)

    # 市場全体のインデックスデータを取得
    index_ticker = "^GSPC"  # 例: S&P 500 のティッカー
    index_data = fetch_stock_data(index_ticker, start_date, end_date)

    # 'Adj Close' がない場合は 'Close' を代用
    if 'Adj Close' not in stock_data.columns:
        logger.warning(f"'{ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を代用します。")
        stock_data['Adj Close'] = stock_data['Close']

    if 'Adj Close' not in index_data.columns:
        logger.warning(f"'{index_ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を代用します。")
        index_data['Adj Close'] = index_data['Close']

    # カラムが MultiIndex になっている場合はフラット化
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    if isinstance(index_data.columns, pd.MultiIndex):
        index_data.columns = index_data.columns.get_level_values(0)

    return ticker, start_date, end_date, stock_data, index_data

# テストコード
if __name__ == "__main__":
    # ダミーデータの作成
    dates = pd.date_range(start="2022-01-01", periods=200, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(200) * 100,
        'High': np.random.random(200) * 100,
        'Low': np.random.random(200) * 100,
        'Adj Close': np.random.random(200) * 100,
        'Volume': np.random.randint(100, 1000, 200)
    }, index=dates)

    # 市場全体のインデックスデータの作成
    index_data = pd.DataFrame({
        'Adj Close': np.random.random(200) * 100
    }, index=dates)

    # VWAPアウトブレイク戦略の実行
    strategy = VWAPBreakoutStrategy(df, index_data)
    result = strategy.backtest()
    print(result)

    # 取引シグナルの確認
    print("\n--- Entry_Signal ---")
    print(result['Entry_Signal'].value_counts())
    print(result['Entry_Signal'].describe())
    print("\n--- Exit_Signal ---")
    print(result['Exit_Signal'].value_counts())
    print(result['Exit_Signal'].describe())
    # 取引結果や日次損益があれば出力
    for col in ['取引結果', '日次損益', '損益', 'Profit', 'PnL']:
        if col in result.columns:
            print(f"\n--- {col} ---")
            print(result[col].value_counts(dropna=False))
            print(result[col].describe())
        else:
            print(f"カラム '{col}' は存在しません。")