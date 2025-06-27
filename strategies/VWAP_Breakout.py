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

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
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
            # --- リスクリワード調整（最適化結果から） ---
            "stop_loss": 0.05,    # 5% ストップロス（最適化結果から）
            "take_profit": 0.1,   # 10% 利益確定（最適化結果から）

            # --- エントリー条件調整（トレンドに追随） ---
            "sma_short": 10,      # 短期移動平均（10日）
            "sma_long": 20,       # 中期移動平均（20日）
            "volume_threshold": 1.3, # 出来高増加条件を強化（より確度の高いシグナル）

            # --- シグナル品質向上 ---
            "confirmation_bars": 1,             # 1日の確認期間でフェイクブレイクを減らす
            "breakout_min_percent": 0.005,      # 0.5%のブレイク率で意味のあるブレイクのみ検出
            "trailing_stop": 0.05,              # トレーリングストップ（5%）
            "trailing_start_threshold": 0.04,   # 利益が4%出た時点でトレーリング開始
            "max_holding_period": 15,           # 中期トレンドも捉えられるよう保有期間延長（15日）

            # --- インテリジェント・フィルター ---
            "market_filter_method": "macd",     # MACDベースの市場フィルター（より先行指標効果）
            "rsi_filter_enabled": True,         # RSIフィルターを有効化
            "atr_filter_enabled": True,         # ATRフィルターを有効化（ボラティリティ考慮）
            "partial_exit_enabled": True,       # 部分利確の有効化（利益確保）
            "partial_exit_threshold": 0.07,     # 7%で部分利確
            "partial_exit_portion": 0.5,        # 50%のポジションを利確

            # --- その他（最適化を通して検証） ---
            "rsi_period": 14,                   # RSI計算期間
            "rsi_lower": 30,                    # RSI下限値（過売り）
            "rsi_upper": 70,                    # RSI上限値（過買い）
            "volume_increase_mode": "average",  # 出来高増加判定方式（平均対比）
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
        （デバッグ用：常にTrueを返す）
        """
        return True

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        改善されたロジックでより高品質なシグナルを生成
        """
        try:
            sma_short_key = 'SMA_' + str(self.params["sma_short"])
            sma_long_key = 'SMA_' + str(self.params["sma_long"])
            
            # データ不足チェック
            if idx < max(self.params["sma_long"], 20):
                logger.debug(f"[entry] idx={idx}: データ不足 (必要データ数={max(self.params['sma_long'], 20)})")
                return 0
            
            # 基本指標の取得
            current_price = self.data[self.price_column].iloc[idx]
            prev_price = self.data[self.price_column].iloc[idx-1]
            sma_short = self.data[sma_short_key].iloc[idx]
            sma_long = self.data[sma_long_key].iloc[idx]
            vwap = self.data['VWAP'].iloc[idx]
            previous_vwap = self.data['VWAP'].iloc[idx - 1]
            current_volume = self.data[self.volume_column].iloc[idx]
            
            # 動的な変数取得
            rsi = self.data['RSI'].iloc[idx] if 'RSI' in self.data.columns else 50
            macd = self.data['MACD'].iloc[idx] if 'MACD' in self.data.columns else 0
            signal = self.data['Signal_Line'].iloc[idx] if 'Signal_Line' in self.data.columns else 0
            
            # --- 市場環境フィルター（より高度な判断） ---
            if self.params.get("market_filter_method") == "macd":
                # MACDによる市場トレンド判断
                if self.index_data is not None:
                    if 'MACD' in self.index_data.columns and 'Signal_Line' in self.index_data.columns:
                        index_macd = self.index_data['MACD'].iloc[idx]
                        index_signal = self.index_data['Signal_Line'].iloc[idx]
                        if not (index_macd > index_signal):
                            logger.debug(f"[entry] idx={idx}: 市場MACDフィルターNG index_macd={index_macd}, index_signal={index_signal}")
                            return 0
            elif self.params.get("market_filter_method") == "sma":
                # SMAによる市場トレンド判断
                if self.index_data is not None:
                    index_price = self.index_data[self.price_column].iloc[idx]
                    index_sma = self.index_data[sma_short_key].iloc[idx]
                    if not (index_price > index_sma):
                        logger.debug(f"[entry] idx={idx}: 市場SMAフィルターNG index_price={index_price}, index_sma={index_sma}")
                        return 0
            
            # --- 高精度エントリー条件 ---
            
            # 1. トレンド条件：短期SMAが長期SMAを上回る
            trend_condition = sma_short > sma_long
            if not trend_condition:
                logger.debug(f"[entry] idx={idx}: トレンド条件NG sma_short={sma_short}, sma_long={sma_long}")
                return 0
            
            # 2. VWAPブレイク条件：価格がVWAPを一定割合以上上回る
            min_breakout_pct = self.params.get("breakout_min_percent", 0.005)
            vwap_break_condition = current_price > vwap * (1 + min_breakout_pct)
            if not vwap_break_condition:
                logger.debug(f"[entry] idx={idx}: VWAPブレイク条件NG current={current_price}, vwap*(1+{min_breakout_pct})={vwap*(1+min_breakout_pct)}")
                return 0
                
            # 3. ブレイク確認：連続して閾値を超えているか確認
            if self.params.get("confirmation_bars", 1) > 0:
                bars_above_vwap = 0
                for i in range(min(self.params["confirmation_bars"] + 1, idx + 1)):
                    if self.data[self.price_column].iloc[idx - i] > vwap:
                        bars_above_vwap += 1
                    else:
                        break
                
                if bars_above_vwap < self.params["confirmation_bars"]:
                    logger.debug(f"[entry] idx={idx}: 確認バー条件NG bars_above_vwap={bars_above_vwap}, required={self.params['confirmation_bars']}")
                    return 0
            
            # 4. 出来高増加条件：平均対比でより精度の高い判定
            if self.params.get("volume_increase_mode") == "average":
                # 過去10日の平均出来高を計算
                avg_volume = self.data[self.volume_column].iloc[idx-10:idx].mean()
                volume_condition = current_volume > avg_volume * self.params["volume_threshold"]
                if not volume_condition:
                    logger.debug(f"[entry] idx={idx}: 出来高条件(平均比較)NG current={current_volume}, avg*{self.params['volume_threshold']}={avg_volume*self.params['volume_threshold']}")
                    return 0
            else:
                # 前日比の単純な出来高増加
                prev_volume = self.data[self.volume_column].iloc[idx - 1]
                volume_condition = detect_volume_increase(current_volume, prev_volume, threshold=self.params["volume_threshold"])
                if not volume_condition:
                    logger.debug(f"[entry] idx={idx}: 出来高増加NG current={current_volume}, prev={prev_volume}")
                    return 0
            
            # 5. RSIフィルター（過買い/過売り状態を避ける）
            if self.params.get("rsi_filter_enabled", False):
                rsi_lower = self.params.get("rsi_lower", 30)
                rsi_upper = self.params.get("rsi_upper", 70)
                
                # RSIが過買いでも過売りでもない中間域での取引を推奨
                if rsi > rsi_upper or rsi < rsi_lower:
                    logger.debug(f"[entry] idx={idx}: RSIフィルターNG rsi={rsi}, 許容範囲={rsi_lower}〜{rsi_upper}")
                    return 0
            
            # 6. ATRフィルター（ボラティリティ考慮）
            if self.params.get("atr_filter_enabled", False) and 'ATR' in self.data.columns:
                atr = self.data['ATR'].iloc[idx]
                # 日次変動率に対するATRの比率で判断（値動きが大きすぎる場合は避ける）
                price_change_pct = abs((current_price - prev_price) / prev_price)
                if price_change_pct > atr * 1.5:  # ATRの1.5倍以上の価格変動がある場合は不安定と判断
                    logger.debug(f"[entry] idx={idx}: ATRフィルターNG price_change={price_change_pct}, atr*1.5={atr*1.5}")
                    return 0
            
            # すべての条件を満たしたらエントリーシグナルを出す
            logger.info(f"VWAP Breakout エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}, RSI={rsi:.1f}")
            return 1
            
        except Exception as e:
            logger.error(f"[entry] idx={idx}: 例外発生: {e}", exc_info=True)
            return 0

    def generate_exit_signal(self, idx: int, entry_idx: int = None) -> int:
        """
        イグジットシグナルを生成する。
        より洗練されたイグジットロジックを実装
        
        Parameters:
            idx (int): 現在のインデックス
            entry_idx (int): エントリー時のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1 or entry_idx is None:
            return 0
        
        try:
            # 基本指標の取得
            current_price = self.data[self.price_column].iloc[idx]
            prev_price = self.data[self.price_column].iloc[idx-1]
            vwap = self.data['VWAP'].iloc[idx]
            entry_price = self.data[self.price_column].iloc[entry_idx]
            sma_short_key = 'SMA_' + str(self.params["sma_short"])
            
            # 価格変動率の計算
            profit_pct = (current_price - entry_price) / entry_price
            
            # 保有期間の計算
            days_held = idx - entry_idx
            
            # --- 1. ストップロス条件 (より洗練されたリスク管理) ---
            # 基本ストップロス
            stop_loss_pct = self.params["stop_loss"]
            
            # 保有期間に応じて損切り幅を調整（長期保有なら緩め、短期なら厳しめ）
            if days_held > 5:
                # 長期保有の場合、ボラティリティに応じて損切り幅を調整
                if 'ATR' in self.data.columns:
                    atr = self.data['ATR'].iloc[idx]
                    # ATRが大きい（ボラティリティが高い）場合は損切り幅を広げる
                    atr_multiplier = 2.0
                    dynamic_stop = min(atr * atr_multiplier / entry_price, stop_loss_pct * 1.5)
                    stop_loss_pct = max(stop_loss_pct, dynamic_stop)
            
            # 損切りチェック
            if current_price <= entry_price * (1 - stop_loss_pct):
                self.log_trade(f"VWAP Breakout イグジットシグナル: ストップロス({stop_loss_pct*100:.1f}%) 日付={self.data.index[idx]}, 価格={current_price}")
                return -1
                
            # --- 2. 連続的な利益確定条件 ---
            # 基本利益確定率
            take_profit = self.params["take_profit"]
            
            # 利益が大きいほど利確の閾値を上げる（トレンドを活かす）
            if profit_pct >= take_profit:
                self.log_trade(f"VWAP Breakout イグジットシグナル: 利益確定({take_profit*100:.1f}%) 日付={self.data.index[idx]}, 価格={current_price}")
                return -1
                
            # --- 3. 高度なトレーリングストップ ---
            trailing_start = self.params.get("trailing_start_threshold", 0.04)
            if profit_pct >= trailing_start:
                # 最高値からの下落率でトレーリングストップを判断
                high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()
                
                # トレンド強度に応じてトレーリングストップの幅を動的に調整
                trailing_stop_pct = self.params["trailing_stop"]
                
                # トレンド強度に応じた調整（強いトレンドならより緩めに）
                if 'MACD' in self.data.columns:
                    macd = self.data['MACD'].iloc[idx]
                    # MACDが強い場合は、トレーリングストップをより緩め（トレンドを活かす）
                    if macd > 0.5:  # 強いトレンドの目安
                        trailing_stop_pct *= 1.2  # 20%緩めるだけ
                
                trailing_stop_price = high_since_entry * (1 - trailing_stop_pct)
                if current_price <= trailing_stop_price:
                    self.log_trade(f"VWAP Breakout イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}, 最高値={high_since_entry}")
                    return -1
            
            # --- 4. テクニカル指標による早期警告 ---
            
            # VWAP下抜けとマルチファクターの組み合わせ
            price_change = (current_price - prev_price) / prev_price
            
            # 価格がVWAPを完全に下回り、かつ他の条件が揃ったときにイグジット
            if current_price < vwap * 0.995:  # VWAPの0.5%以上下回った場合
                # 下記条件のいずれかを満たす場合にイグジット
                vwap_exit_conditions = []
                
                # a) 下落中
                vwap_exit_conditions.append(price_change < -0.005)  # 0.5%以上の下落
                
                # b) 短期移動平均を下抜け
                if sma_short_key in self.data.columns:
                    vwap_exit_conditions.append(current_price < self.data[sma_short_key].iloc[idx])
                
                # c) RSIが過買いからの下落開始
                if 'RSI' in self.data.columns:
                    rsi = self.data['RSI'].iloc[idx]
                    prev_rsi = self.data['RSI'].iloc[idx-1]
                    rsi_falling_from_high = rsi < prev_rsi and prev_rsi > 70
                    vwap_exit_conditions.append(rsi_falling_from_high)
                
                # いずれかの条件が成立すればイグジット
                if any(vwap_exit_conditions):
                    reason = "VWAP下抜け+"
                    if price_change < -0.005:
                        reason += "下落中"
                    elif sma_short_key in self.data.columns and current_price < self.data[sma_short_key].iloc[idx]:
                        reason += "MA下抜け"
                    else:
                        reason += "RSI下落"
                    
                    self.log_trade(f"VWAP Breakout イグジットシグナル: {reason} 日付={self.data.index[idx]}, 価格={current_price}")
                    return -1
            
            # --- 5. MACDクロスによる高精度イグジット ---
            if 'MACD' in self.data.columns and 'Signal_Line' in self.data.columns:
                macd = self.data['MACD'].iloc[idx]
                signal_line = self.data['Signal_Line'].iloc[idx]
                prev_macd = self.data['MACD'].iloc[idx-1]
                prev_signal = self.data['Signal_Line'].iloc[idx-1]
                
                # MACDがシグナルラインを下抜け、かつ利益が出ている場合のみ（損失時は避ける）
                if macd < signal_line and prev_macd >= prev_signal and profit_pct > 0:
                    self.log_trade(f"VWAP Breakout イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
                    return -1
            
            # --- 6. 部分利確の最適化 ---
            if self.params.get("partial_exit_enabled", False):
                partial_threshold = self.params.get("partial_exit_threshold", 0.07)
                
                if profit_pct >= partial_threshold:
                    if 'Partial_Exit' not in self.data.columns:
                        self.data['Partial_Exit'] = 0
                    
                    # 部分利確の実行（既に部分利確していない場合のみ）
                    if self.data.at[self.data.index[idx], 'Partial_Exit'] == 0:
                        portion = self.params.get("partial_exit_portion", 0.5)
                        self.data.at[self.data.index[idx], 'Partial_Exit'] = portion
                        self.log_trade(f"VWAP Breakout 部分利確シグナル: {portion*100}% 利確 日付={self.data.index[idx]}, 価格={current_price}, 利益率={profit_pct*100:.1f}%")
            
            # イグジットシグナルなし
            return 0
            
        except Exception as e:
            logger.error(f"[exit] idx={idx}, entry_idx={entry_idx}: 例外発生: {e}", exc_info=True)
            return 0

    def backtest(self):
        """
        VWAPアウトブレイク戦略のバックテストを実行する。
        改善されたバックテストロジックとATR計算を追加
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        try:
            # 必要なインジケーターが計算されているか確認
            if 'ATR' not in self.data.columns and self.params.get("atr_filter_enabled", False):
                # ATR計算（ボラティリティフィルター用）
                from indicators.bollinger_atr import calculate_atr
                # calculate_atrはDataFrameを返すので、ATR列だけを取得
                atr_df = calculate_atr(self.data, price_column=self.price_column)
                self.data['ATR'] = atr_df['ATR']
                logger.info("ATR計算を実行しました")
                
            # シグナル列の初期化
            self.data['Entry_Signal'] = 0
            self.data['Exit_Signal'] = 0
            self.data['Position'] = 0  # ポジション状態を追加
            self.data['Entry_Price'] = np.nan  # エントリー価格を記録
            self.data['Entry_Idx'] = np.nan  # エントリーインデックスを記録
            self.data['Partial_Exit'] = 0.0  # 部分利確フラグを初期化（float型で）
            
            # トレード統計用
            trade_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
            }
    
            # バックテストループ
            for idx in range(len(self.data)):
                try:
                    current_price = self.data[self.price_column].iloc[idx]
                    
                    # 前日までのポジション状態を確認
                    if idx > 0:
                        self.data.loc[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                        # ポジションを引き継ぐ場合はEntry_Idx、Entry_Price、Partial_Exitも引き継ぐ
                        if self.data['Position'].iloc[idx] == 1:
                            # 常に前日の値を引き継ぎ（上書き）して一貫性を保つ
                            if not pd.isna(self.data['Entry_Idx'].iloc[idx-1]):
                                self.data.loc[self.data.index[idx], 'Entry_Idx'] = self.data['Entry_Idx'].iloc[idx-1]
                                self.data.loc[self.data.index[idx], 'Entry_Price'] = self.data['Entry_Price'].iloc[idx-1]
                                self.data.loc[self.data.index[idx], 'Partial_Exit'] = self.data['Partial_Exit'].iloc[idx-1]
                
                    # ポジションがない場合、エントリーシグナルをチェック
                    if self.data['Position'].iloc[idx] == 0:
                        entry_signal = self.generate_entry_signal(idx)
                        if entry_signal == 1:
                            # エントリーシグナルあり
                            self.data.loc[self.data.index[idx], 'Entry_Signal'] = 1
                            self.data.loc[self.data.index[idx], 'Position'] = 1
                            self.data.loc[self.data.index[idx], 'Entry_Price'] = current_price
                            self.data.loc[self.data.index[idx], 'Entry_Idx'] = idx
                            self.data.loc[self.data.index[idx], 'Partial_Exit'] = 0
                            trade_stats['total_trades'] += 1
                            
                    # ポジションがある場合、イグジットシグナルをチェック
                    elif self.data['Position'].iloc[idx] == 1:
                        # エントリーインデックスを取得（NaNチェック強化）
                        entry_idx_val = self.data['Entry_Idx'].iloc[idx]
                        if pd.isna(entry_idx_val):
                            # Entry_Idxが設定されていない場合はエラーログを出力して次のインデックスへ
                            logger.warning(f"Position=1だがEntry_Idxが設定されていません(idx={idx})")
                            continue
                        
                        try:
                            entry_idx = int(float(entry_idx_val))  # 念のため一旦floatに変換してからint化
                        except (ValueError, TypeError) as e:
                            logger.error(f"エントリーインデックス変換エラー: {e}, entry_idx_val={entry_idx_val}, idx={idx}")
                            continue
                        
                        entry_price = self.data['Entry_Price'].iloc[idx]
                        profit_pct = (current_price - entry_price) / entry_price
                        
                        # エントリー後の最大保有期間チェック
                        days_held = idx - entry_idx
                        if days_held >= self.params.get("max_holding_period", 10):
                            # 最大保有期間に達したらイグジット
                            self.data.loc[self.data.index[idx], 'Exit_Signal'] = -1
                            self.data.loc[self.data.index[idx], 'Position'] = 0
                            self.log_trade(f"VWAP Breakout イグジットシグナル: 最大保有期間到達 ({days_held}日) 日付={self.data.index[idx]}, 価格={current_price}")
                            
                            # 取引統計の更新
                            profit = current_price - entry_price
                            trade_stats['total_profit'] += profit
                            
                            if profit > 0:
                                trade_stats['winning_trades'] += 1
                                trade_stats['max_profit'] = max(trade_stats['max_profit'], profit)
                            else:
                                trade_stats['losing_trades'] += 1
                                trade_stats['max_loss'] = min(trade_stats['max_loss'], profit)
                        else:
                            # 通常のイグジットシグナルをチェック
                            exit_signal = self.generate_exit_signal(idx, entry_idx)
                            if exit_signal == -1:
                                self.data.loc[self.data.index[idx], 'Exit_Signal'] = -1
                                self.data.loc[self.data.index[idx], 'Position'] = 0
                                
                                # 取引統計の更新
                                profit = current_price - entry_price
                                trade_stats['total_profit'] += profit
                                
                                if profit > 0:
                                    trade_stats['winning_trades'] += 1
                                    trade_stats['max_profit'] = max(trade_stats['max_profit'], profit)
                                else:
                                    trade_stats['losing_trades'] += 1
                                    trade_stats['max_loss'] = min(trade_stats['max_loss'], profit)
                except Exception as e:
                    logger.error(f"バックテストループでエラー発生 idx={idx}: {e}")
                    continue
            
            # 取引統計をログに出力
            win_rate = trade_stats['winning_trades'] / trade_stats['total_trades'] * 100 if trade_stats['total_trades'] > 0 else 0
            avg_profit = trade_stats['total_profit'] / trade_stats['total_trades'] if trade_stats['total_trades'] > 0 else 0
            
            logger.info(f"バックテスト完了: 総取引数={trade_stats['total_trades']}, 勝率={win_rate:.1f}%, 平均利益={avg_profit:.2f}")
            logger.info(f"最大利益={trade_stats['max_profit']:.2f}, 最大損失={trade_stats['max_loss']:.2f}, 総損益={trade_stats['total_profit']:.2f}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)
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