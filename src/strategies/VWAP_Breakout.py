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
    def __init__(self, data: pd.DataFrame, index_data: pd.DataFrame, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        VWAPアウトブレイク戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            index_data (pd.DataFrame): 市場全体のインデックスデータ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        # 戦略固有の属性を先に設定
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
        
        self.data['SMA_' + str(sma_short)] = calculate_sma(self.data, self.price_column, sma_short)
        self.data['SMA_' + str(sma_long)] = calculate_sma(self.data, self.price_column, sma_long)
        self.data['VWAP'] = calculate_vwap(self.data, self.price_column, self.volume_column)
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], rsi_period)
        self.data['MACD'], self.data['Signal_Line'] = calculate_macd(self.data, self.price_column)

        # 市場全体のトレンドを確認するためのインデックスの移動平均線
        if self.index_data is not None:
            self.index_data['SMA_' + str(sma_short)] = calculate_sma(self.index_data, self.price_column, sma_short)
            self.index_data['SMA_' + str(sma_long)] = calculate_sma(self.index_data, self.price_column, sma_long)

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
        """
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
        current_price = self.data[self.price_column].iloc[idx]
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
            high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()
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

    def backtest(self):
        """
        VWAPアウトブレイク戦略のバックテストを実行する。
        
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

        # バックテストループ
        for idx in range(len(self.data)):
            current_price = self.data[self.price_column].iloc[idx]
            
            # 前日までのポジション状態を確認
            if idx > 0:
                self.data.loc[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
            
            # ポジションがない場合、エントリーシグナルをチェック
            if self.data['Position'].iloc[idx] == 0:
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    # エントリーシグナルあり
                    self.data.loc[self.data.index[idx], 'Entry_Signal'] = 1
                    self.data.loc[self.data.index[idx], 'Position'] = 1
                    self.data.loc[self.data.index[idx], 'Entry_Price'] = current_price
                    self.data.loc[self.data.index[idx], 'Entry_Idx'] = idx
            
            # ポジションがある場合、イグジットシグナルをチェック
            elif self.data['Position'].iloc[idx] == 1:
                # エントリーインデックスを取得
                entry_idx_val = self.data['Entry_Idx'].iloc[idx]
                
                # エントリーインデックスが無効値(-1)またはNaNの場合は現在のインデックスを使用
                if pd.isna(entry_idx_val) or entry_idx_val == -1:
                    logger.warning(f"有効なEntry_Idxがありません: idx={idx}, 日付={self.data.index[idx]}. 現在のインデックスを使用します。")
                    entry_idx = idx  # フォールバックとして現在のインデックスを使用
                else:
                    try:
                        entry_idx = int(entry_idx_val)
                    except (ValueError, TypeError):
                        logger.warning(f"Entry_Idxの変換に失敗しました: {entry_idx_val}. 現在のインデックスを使用します。")
                        entry_idx = idx  # 変換失敗時も現在のインデックスを使用
                
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