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
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_sma, calculate_vwap
from indicators.volume_analysis import detect_volume_increase
from indicators.momentum_indicators import calculate_macd
from indicators.basic_indicators import calculate_rsi

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
            "sma_short": 20,
            "sma_long": 50,
            "rsi_period": 14,
            "volume_threshold": 1.2,
            "take_profit": 0.10,  # 10%
            "stop_loss": 0.05,    # 5%
            "trailing_stop": 0.03,  # 3%
            "confirmation_bars": 1,             # ブレイク確認バー数 (0=即時エントリー)
            "breakout_min_percent": 0.005,      # 最小ブレイク率 (例: 0.5%以上の上抜け)
            "atr_filter_enabled": False,        # ボラティリティフィルターの有効化
            "atr_period": 14,                   # ATR計算期間
            "atr_min_threshold": 0.01,          # 最小ATR閾値 (相対値)
            "atr_max_threshold": 0.03,          # 最大ATR閾値 (相対値)
            "volume_increase_mode": "simple",   # 出来高増加判定方式 (simple/average/exponential)
            "volume_lookback_period": 5,        # 出来高比較期間
            "bullish_candle_required": True,    # 陽線要求 (True=陽線形成時のみエントリー)
            "trailing_start_threshold": 0.03,   # トレーリング開始閾値 (3%の利益でトレーリング開始)
            "max_holding_period": 10,           # 最大保有期間 (日数)
            "partial_exit_enabled": False,      # 部分利確の有効化
            "partial_exit_threshold": 0.05,     # 部分利確の発動閾値
            "partial_exit_portion": 0.5,        # 一部利確の割合
            "reversal_exit_enabled": False,     # 反転イグジットの有効化
            "reversal_bars_threshold": 2,       # 反転確認バー数
            "vwap_recross_exit": True,          # VWAP再クロス時のイグジット
            "market_filter_method": "sma",      # 市場フィルター方式 (sma/ema/macd/combined)
            "market_condition_threshold": 0.01, # 市場上昇トレンド判定閾値
            "rsi_filter_enabled": False,        # RSIフィルターの有効化
            "rsi_lower_bound": 40,              # RSI下限値 (この値以下ではエントリーしない)
            "rsi_upper_bound": 70,              # RSI上限値 (この値以上でイグジット検討)
            "multiple_index_confirmation": False,# 複数指数による確認
            "risk_per_trade": 0.02,             # トレードごとのリスク (総資産の2%)
            "max_open_positions": 5,            # 最大保有ポジション数
            "drawdown_stop_threshold": 0.1,     # ドローダウン停止閾値 (10%)
            "consecutive_loss_limit": 3,        # 連続損失制限
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

    def is_market_uptrend(self, idx: int) -> bool:
        """
        市場全体が上昇トレンドにあるかを確認する。

        Parameters:
            idx (int): 現在のインデックス

        Returns:
            bool: 市場全体が上昇トレンドにある場合は True、それ以外は False
        """
        if self.index_data is None or idx < self.params["sma_long"]:
            return False

        sma_short_key = 'SMA_' + str(self.params["sma_short"])
        sma_long_key = 'SMA_' + str(self.params["sma_long"])
        
        index_price = self.index_data[self.price_column].iloc[idx]
        index_sma_short = self.index_data[sma_short_key].iloc[idx]
        index_sma_long = self.index_data[sma_long_key].iloc[idx]

        # 市場全体が上昇トレンドにある条件
        return index_price > index_sma_short > index_sma_long and \
               self.index_data[sma_short_key].iloc[idx] > self.index_data[sma_short_key].iloc[idx - 1] and \
               self.index_data[sma_long_key].iloc[idx] > self.index_data[sma_long_key].iloc[idx - 1]

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        sma_short_key = 'SMA_' + str(self.params["sma_short"])
        sma_long_key = 'SMA_' + str(self.params["sma_long"])
        
        if idx < self.params["sma_long"]:  # 必要な履歴データがない場合
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        sma_short = self.data[sma_short_key].iloc[idx]
        sma_long = self.data[sma_long_key].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        previous_vwap = self.data['VWAP'].iloc[idx - 1]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # 市場全体が上昇トレンドにあるか確認
        if not self.is_market_uptrend(idx):
            return 0

        # 株価が短期移動平均線や長期移動平均線の上に位置している
        if not (current_price > sma_short > sma_long):
            return 0

        # 移動平均線が上昇している
        if not (sma_short > self.data[sma_short_key].iloc[idx - 1] and 
                sma_long > self.data[sma_long_key].iloc[idx - 1]):
            return 0

        # VWAPが上昇している
        if not (vwap > previous_vwap):
            return 0

        # VWAPを上抜けしている
        vwap_breakout = current_price > vwap and self.data[self.price_column].iloc[idx - 1] <= vwap
        if not vwap_breakout:
            return 0

        # ブレイク率チェック
        if self.params.get("breakout_min_percent", 0) > 0:
            min_breakout = vwap * (1 + self.params["breakout_min_percent"])
            if current_price <= min_breakout:
                return 0  # 最小ブレイク率に達していない

        # ブレイク持続確認
        if self.params.get("confirmation_bars", 0) > 0:
            confirmation_needed = True
            for i in range(1, min(self.params["confirmation_bars"] + 1, idx + 1)):
                if self.data[self.price_column].iloc[idx - i] <= vwap:
                    return 0  # 確認期間内にVWAPより下になった

        # 出来高が増加している
        if not detect_volume_increase(current_volume, previous_volume, threshold=self.params["volume_threshold"]):
            return 0

        # すべての条件を満たした場合、エントリーシグナルを返す
        self.log_trade(f"VWAP Breakout エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}")
        return 1

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 必要な履歴データがない場合
            return 0
            
        current_price = self.data[self.price_column].iloc[idx]
        vwap = self.data['VWAP'].iloc[idx]
        
        # エントリー価格がない場合（ポジションがない場合）はシグナルなし
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        entry_price = self.data[self.price_column].iloc[latest_entry_idx]
        
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
        
        # トレーリング開始閾値を超えた場合のみトレーリングストップを適用
        if profit_pct >= self.params.get("trailing_start_threshold", 0):
            high_since_entry = self.data['High'].iloc[latest_entry_idx:idx+1].max()
            trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
            
            if current_price <= trailing_stop:
                self.log_trade(f"VWAP Breakout イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
                return -1

        # 部分利確ロジック
        if self.params.get("partial_exit_enabled", False):
            profit_pct = (current_price - entry_price) / entry_price
            
            # 部分利確の条件を満たす場合
            if profit_pct >= self.params["partial_exit_threshold"] and 'Partial_Exit' not in self.data.columns:
                # 部分利確用のカラムを追加
                self.data['Partial_Exit'] = 0
                self.data.at[self.data.index[idx], 'Partial_Exit'] = 1
                
                # ログ
                self.log_trade(f"VWAP Breakout 部分利確シグナル: {self.params['partial_exit_portion']*100}% 利確 日付={self.data.index[idx]}, 価格={current_price}")
                
                # 全ポジションイグジットしない (部分利確なのでリターン値は0)
                return 0

        # RSIやMACDの反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > 70 and rsi < self.data['RSI'].iloc[idx - 1]:  # RSIが70以上から急落
            self.log_trade(f"VWAP Breakout イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        if macd < signal_line and self.data['MACD'].iloc[idx-1] >= self.data['Signal_Line'].iloc[idx-1]:  # MACDがシグナルラインを下抜け
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

        # 各日にちについてシグナルを計算
        for idx in range(len(self.data)):
            # エントリーシグナルを確認
            entry_signal = self.generate_entry_signal(idx)
            if entry_signal == 1:
                self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
            
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

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