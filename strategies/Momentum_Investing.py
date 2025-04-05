"""
Module: Momentum_Investing
File: Momentum_Investing.py
Description: 
  モメンタム（勢い）に基づいた投資戦略を実装します。トレンドが強く出ている銘柄に投資し、
  移動平均線やRSI、MACD、出来高などの複合的な指標を用いて、上昇トレンドの継続性を
  判断します。適切なエントリー・イグジットポイントを算出し、リスク管理も考慮しています。

Author: kouseven7
Created: 2023-03-10
Modified: 2025-04-02

Dependencies:
  - strategies.base_strategy
  - indicators.basic_indicators
  - indicators.momentum_indicators
  - indicators.volume_analysis
  - indicators.volatility_indicators
"""

import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from strategies.base_strategy import BaseStrategy
from indicators.basic_indicators import calculate_sma, calculate_rsi
from indicators.momentum_indicators import calculate_macd
from indicators.volume_analysis import detect_volume_increase
from indicators.volatility_indicators import calculate_atr

class MomentumInvestingStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        モメンタム戦略の初期化。

        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（オーバーライド用）
            price_column (str): 株価カラム名（デフォルトは "Adj Close"）
            volume_column (str): 出来高カラム名（デフォルトは "Volume"）
        """
        # 戦略固有の属性を先に設定
        self.price_column = price_column
        self.volume_column = volume_column
        self.entry_prices = {}  # エントリー価格を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "sma_short": 20,
            "sma_long": 50,
            "rsi_period": 14,
            "rsi_lower": 50,
            "rsi_upper": 70,
            "volume_threshold": 1.2,
            "take_profit": 0.10,  # 10%
            "stop_loss": 0.05,    # 5%
            "trailing_stop": 0.03  # 3%
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
        self.data['RSI'] = calculate_rsi(self.data[self.price_column], rsi_period)
        self.data['MACD'], self.data['Signal_Line'] = calculate_macd(self.data, self.price_column)
        self.data['ATR'] = calculate_atr(self.data, self.price_column)

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 株価が20日MAおよび50日MAの上にある
        - 20日MAが50日MAを上抜けしており、両者が上昇傾向にある
        - RSIが50以上で過熱状態ではない
        - MACDラインがシグナルラインを上抜けしている
        - 出来高が増加している
        - プルバックからの反発、またはブレイクアウト

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        sma_short_key = 'SMA_' + str(self.params["sma_short"])
        sma_long_key = 'SMA_' + str(self.params["sma_long"])
        rsi_lower = self.params["rsi_lower"]
        rsi_upper = self.params["rsi_upper"]
        
        if idx < self.params["sma_long"]:  # 必要な履歴データがない場合
            return 0

        current_price = self.data[self.price_column].iloc[idx]
        sma_short = self.data[sma_short_key].iloc[idx]
        sma_long = self.data[sma_long_key].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        current_volume = self.data[self.volume_column].iloc[idx]
        previous_volume = self.data[self.volume_column].iloc[idx - 1]

        # 株価が20日MAおよび50日MAの上にある
        if not (current_price > sma_short > sma_long):
            return 0

        # 20日MAが50日MAを上抜けしており、両者が上昇傾向にある
        if not (sma_short > sma_long and 
                self.data[sma_short_key].iloc[idx - 1] < self.data[sma_long_key].iloc[idx - 1]):
            return 0

        # RSIが50以上で過熱状態ではない
        if not (rsi_lower <= rsi < rsi_upper):
            return 0

        # MACDラインがシグナルラインを上抜けしている
        if not (macd > signal_line and 
                self.data['MACD'].iloc[idx - 1] <= self.data['Signal_Line'].iloc[idx - 1]):
            return 0

        # 出来高が増加している
        if not detect_volume_increase(current_volume, previous_volume, 
                                     threshold=self.params["volume_threshold"]):
            return 0

        # プルバックからの反発
        if current_price < sma_short and current_price > sma_short * 0.98:
            self.entry_prices[idx] = current_price
            self.log_trade(f"Momentum Investing エントリーシグナル (プルバック): 日付={self.data.index[idx]}, 価格={current_price}")
            return 1

        # ブレイクアウト（直近の抵抗線を上抜け）
        recent_high = self.data['High'].iloc[max(0, idx - 10):idx].max()
        if current_price > recent_high:
            self.entry_prices[idx] = current_price
            self.log_trade(f"Momentum Investing エントリーシグナル (ブレイクアウト): 日付={self.data.index[idx]}, 価格={current_price}")
            return 1

        return 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        条件:
        - エントリー価格から3～5%の下落でストップロス
        - ATRに基づいたストップロス
        - 利益が伸びた場合、トレーリングストップを設定
        - 株価が短期MA（20日MA）を下回った場合
        - RSIが70以上から急落、またはMACDがシグナルラインを下抜けした場合
        - 高値更新が止まり、ダイバージェンスが発生した場合

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        if idx < 1:  # 必要な履歴データがない場合
            return 0
            
        # エントリー価格を取得
        entry_indices = self.data[self.data['Entry_Signal'] == 1].index
        if len(entry_indices) == 0 or entry_indices[-1] >= self.data.index[idx]:
            return 0
            
        # 最新のエントリー価格を取得
        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
        if latest_entry_idx not in self.entry_prices:
            # 記録されていない場合は価格を取得して記録
            self.entry_prices[latest_entry_idx] = self.data[self.price_column].iloc[latest_entry_idx]
            
        entry_price = self.entry_prices[latest_entry_idx]
        current_price = self.data[self.price_column].iloc[idx]
        atr = self.data['ATR'].iloc[idx]
        sma_short_key = 'SMA_' + str(self.params["sma_short"])
            
        # ストップロス条件（ATRベースまたはパーセンテージベース）
        if current_price <= entry_price - atr or current_price <= entry_price * (1 - self.params["stop_loss"]):
            self.log_trade(f"Momentum Investing イグジットシグナル: ストップロス 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # 利益確定条件（目標利益）
        if current_price >= entry_price * (1 + self.params["take_profit"]):
            self.log_trade(f"Momentum Investing イグジットシグナル: 利益確定 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # トレーリングストップ条件
        high_since_entry = self.data['High'].iloc[latest_entry_idx:idx+1].max()
        trailing_stop = high_since_entry * (1 - self.params["trailing_stop"])
        if current_price <= trailing_stop:
            self.log_trade(f"Momentum Investing イグジットシグナル: トレーリングストップ 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # 移動平均線のブレイク
        sma_short = self.data[sma_short_key].iloc[idx]
        if current_price < sma_short:
            self.log_trade(f"Momentum Investing イグジットシグナル: 移動平均線ブレイク 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # モメンタム指標の反転
        rsi = self.data['RSI'].iloc[idx]
        macd = self.data['MACD'].iloc[idx]
        signal_line = self.data['Signal_Line'].iloc[idx]
        if rsi > self.params["rsi_upper"] and rsi < self.data['RSI'].iloc[idx - 1]:  # RSIが70以上から急落
            self.log_trade(f"Momentum Investing イグジットシグナル: RSI反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1
        if macd < signal_line and self.data['MACD'].iloc[idx-1] >= self.data['Signal_Line'].iloc[idx-1]:  # MACDがシグナルラインを下抜け
            self.log_trade(f"Momentum Investing イグジットシグナル: MACD反転 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        # チャートパターンの崩壊（高値更新が止まる）
        recent_high = self.data['High'].iloc[max(0, idx - 10):idx].max()
        if current_price < recent_high * (1 - self.params["trailing_stop"]):  # 直近高値の3%下
            self.log_trade(f"Momentum Investing イグジットシグナル: チャートパターン崩壊 日付={self.data.index[idx]}, 価格={current_price}")
            return -1

        return 0

    def backtest(self):
        """
        モメンタム戦略のバックテストを実行する。
        """
        # シグナル列の初期化
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0

        # 各日にちについてシグナルを計算
        for idx in range(len(self.data)):
            # Entry_Signalがまだ立っていない場合のみエントリーシグナルをチェック
            if not self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1].any():
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
            
            # イグジットシグナルを確認
            exit_signal = self.generate_exit_signal(idx)
            if exit_signal == -1:
                self.data.at[self.data.index[idx], 'Exit_Signal'] = -1

        return self.data

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

    # モメンタム戦略の実行
    strategy = MomentumInvestingStrategy(df)
    result = strategy.backtest()
    print(result)