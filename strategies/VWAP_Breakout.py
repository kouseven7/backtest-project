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
            # --- リスクリワード重視（より現実的な設定） ---
            "stop_loss": 0.02,    # 2% ストップロス（より早めの損切り）
            "take_profit": 0.08,  # 8% 利益確定（より現実的な設定）

            # --- エントリー頻度調整（より多く） ---
            "sma_short": 5,       # より短期の移動平均（5日）
            "sma_long": 20,       # より短期の長期線（20日）
            "volume_threshold": 1.1, # 出来高増加条件を緩和（10%増）

            # --- シンプル化と高速対応 ---
            "confirmation_bars": 0,             # 確認バー無し（即時エントリー）
            "breakout_min_percent": 0.0,        # ブレイク率チェック無効化
            "trailing_stop": 0.03,              # より敏感なトレーリングストップ（3%）
            "trailing_start_threshold": 0.02,   # より早く利益を守る（2%の利益でトレーリング開始）
            "max_holding_period": 7,            # 短めの保有期間（7日）

            # --- フィルター・特殊機能（選択的に有効化） ---
            "market_filter_method": "sma",      # シンプルなSMAフィルターを使用
            "rsi_filter_enabled": True,         # RSIフィルターを有効化（過買い回避）
            "atr_filter_enabled": False,        # ATRフィルターの有効化
            "partial_exit_enabled": True,       # 部分利確の有効化（利益確保）
            "partial_exit_threshold": 0.05,     # 5%で部分利確
            "partial_exit_portion": 0.3,        # 30%のポジションを利確

            # --- その他（将来拡張用・固定値） ---
            "rsi_period": 14,                   # RSI計算期間
            "volume_increase_mode": "simple",   # 出来高増加判定方式
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
        条件を満たさない場合は理由をDEBUGログに出す（デバッグ用）
        """
        try:
            sma_short_key = 'SMA_' + str(self.params["sma_short"])
            sma_long_key = 'SMA_' + str(self.params["sma_long"])
            if idx < self.params["sma_long"]:
                logger.debug(f"[entry] idx={idx}: データ不足 (sma_long={self.params['sma_long']})")
                return 0
                
            # 基本指標の取得
            current_price = self.data[self.price_column].iloc[idx]
            sma_short = self.data[sma_short_key].iloc[idx]
            sma_long = self.data[sma_long_key].iloc[idx]
            vwap = self.data['VWAP'].iloc[idx]
            previous_vwap = self.data['VWAP'].iloc[idx - 1]
            current_volume = self.data[self.volume_column].iloc[idx]
            previous_volume = self.data[self.volume_column].iloc[idx - 1]
            
            # RSI値の取得
            rsi = self.data['RSI'].iloc[idx] if 'RSI' in self.data.columns else 50
            
            # --- 市場フィルター（より効果的なフィルター） ---
            market_filter_ok = True
            if self.params.get("market_filter_method") == "sma":
                # インデックスデータがある場合のみ確認
                if self.index_data is not None:
                    index_price = self.index_data[self.price_column].iloc[idx]
                    index_sma = self.index_data[sma_short_key].iloc[idx]
                    market_filter_ok = index_price > index_sma
                    if not market_filter_ok:
                        logger.debug(f"[entry] idx={idx}: 市場フィルターNG index_price={index_price}, index_sma={index_sma}")
                        return 0
            
            # --- RSIフィルター（過買い/過売り状態を避ける） ---
            if self.params.get("rsi_filter_enabled", False):
                if rsi > 70:  # 過買い状態ではエントリーしない
                    logger.debug(f"[entry] idx={idx}: RSI過買いNG rsi={rsi}")
                    return 0
            
            # --- 価格/移動平均線条件（緩和版） ---
            # 価格が長期移動平均線の上にあるかのみ確認
            if not (current_price > sma_long):
                logger.debug(f"[entry] idx={idx}: 長期MA上NG current={current_price}, sma_long={sma_long}")
                return 0
            
            # --- VWAPブレイク条件（緩和版） ---
            # 単純に価格がVWAPを上回っているかのみ確認
            if not (current_price > vwap):
                logger.debug(f"[entry] idx={idx}: VWAPブレイクNG current={current_price}, vwap={vwap}")
                return 0
            
            # --- 確認バー条件（オプション） ---
            if self.params.get("confirmation_bars", 0) > 0:
                for i in range(1, min(self.params["confirmation_bars"] + 1, idx + 1)):
                    if self.data[self.price_column].iloc[idx - i] <= vwap:
                        logger.debug(f"[entry] idx={idx}: 確認バーNG idx-i={idx-i}")
                        return 0

            # --- 出来高条件（緩和版） ---
            # 出来高増加の閾値を下げる（パラメータで調整可能）
            if not detect_volume_increase(current_volume, previous_volume, threshold=self.params["volume_threshold"]):
                logger.debug(f"[entry] idx={idx}: 出来高増加NG current={current_volume}, prev={previous_volume}")
                return 0
                
            # すべての条件を満たしたらエントリーシグナルを出す
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
            
        # 基本指標の取得
        current_price = self.data[self.price_column].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        entry_price = self.data[self.price_column].iloc[entry_idx]
        
        # 価格変動率の計算
        profit_pct = (current_price - entry_price) / entry_price
        
        # --- ストップロス条件 (トレンド強度に応じて動的調整) ---
        stop_loss = self.params["stop_loss"]
        if current_price <= entry_price * (1 - stop_loss):
            self.log_trade(f"VWAP Breakout イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # --- 利益確定条件 ---
        take_profit = self.params["take_profit"]
        if current_price >= entry_price * (1 + take_profit):
            self.log_trade(f"VWAP Breakout イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # --- トレーリングストップ（より早く発動） ---
        trailing_start = self.params.get("trailing_start_threshold", 0.02)  # デフォルト2%
        if profit_pct >= trailing_start:
            # 最高値からの下落率でトレーリングストップを判断
            high_since_entry = self.data['High'].iloc[entry_idx:idx+1].max()
            trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
            if current_price <= trailing_stop:
                self.log_trade(f"VWAP Breakout イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}, 最高値={high_since_entry}")
                return -1
                
        # --- VWAP下抜けによるイグジット（上昇トレンド崩れの早期察知） ---
        # 単純なVWAP下抜けに加えて、前日比の下落率も考慮
        prev_price = self.data[self.price_column].iloc[idx-1]
        price_change = (current_price - prev_price) / prev_price
        
        # 価格がVWAPを下回り、なおかつ下落中の場合
        if current_price < vwap and price_change < 0:
            self.log_trade(f"VWAP Breakout イグジットシグナル: VWAP下抜け+下落中 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # --- RSI/MACDによるイグジット ---
        rsi = self.data['RSI'].iloc[idx] if 'RSI' in self.data.columns else 50
        
        # RSIが高水準から下落した場合
        if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:
            self.log_trade(f"VWAP Breakout イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
            
        # MACDクロスによるイグジット
        if 'MACD' in self.data.columns and 'Signal_Line' in self.data.columns:
            macd = self.data['MACD'].iloc[idx]
            signal_line = self.data['Signal_Line'].iloc[idx]
            prev_macd = self.data['MACD'].iloc[idx-1]
            prev_signal = self.data['Signal_Line'].iloc[idx-1]
            
            if macd < signal_line and prev_macd >= prev_signal:
                self.log_trade(f"VWAP Breakout イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
                return -1
        
        # --- 部分利確ロジック ---
        if self.params.get("partial_exit_enabled", False) and profit_pct >= self.params.get("partial_exit_threshold", 0.05):
            if 'Partial_Exit' not in self.data.columns:
                self.data['Partial_Exit'] = 0
            
            # 部分利確の実行（既に部分利確していない場合のみ）
            if self.data.at[self.data.index[idx], 'Partial_Exit'] == 0:
                self.data.at[self.data.index[idx], 'Partial_Exit'] = self.params.get("partial_exit_portion", 0.5)
                self.log_trade(f"VWAP Breakout 部分利確シグナル: {self.params.get('partial_exit_portion', 0.5)*100}% 利確 日付={self.data.index[idx]}, 価格={current_price}")
                
        # イグジットシグナルなし
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
        self.data['Entry_Idx'] = np.nan  # エントリーインデックスを記録

        # バックテストループ
        for idx in range(len(self.data)):
            current_price = self.data[self.price_column].iloc[idx]            # 前日までのポジション状態を確認
            if idx > 0:
                self.data.loc[self.data.index[idx], 'Position'] = self.data['Position'].iloc[idx-1]
                # ポジションを引き継ぐ場合はEntry_IdxとEntry_Priceも引き継ぐ
                if self.data['Position'].iloc[idx] == 1:
                    # 常に前日のEntry_IdxとEntry_Priceを引き継ぎ（上書き）して一貫性を保つ
                    if not pd.isna(self.data['Entry_Idx'].iloc[idx-1]):
                        self.data.loc[self.data.index[idx], 'Entry_Idx'] = self.data['Entry_Idx'].iloc[idx-1]
                        self.data.loc[self.data.index[idx], 'Entry_Price'] = self.data['Entry_Price'].iloc[idx-1]
            
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
                # エントリーインデックスを取得（NaNチェック強化）
                entry_idx_val = self.data['Entry_Idx'].iloc[idx]
                if pd.isna(entry_idx_val):                    # Entry_Idxが設定されていない場合はエラーログを出力して次のインデックスへ
                    self.logger.warning(f"Position=1だがEntry_Idxが設定されていません(idx={idx})")
                    continue
                
                try:
                    entry_idx = int(float(entry_idx_val))  # 念のため一旦floatに変換してからint化
                except (ValueError, TypeError) as e:
                    self.logger.error(f"エントリーインデックス変換エラー: {e}, entry_idx_val={entry_idx_val}, idx={idx}")
                    continue
                
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