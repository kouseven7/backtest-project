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
from typing import Dict, Any
from strategies.base_strategy import BaseStrategy
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend, detect_unified_trend_with_confidence

class BreakoutStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params=None, price_column: str = "Adj Close", volume_column: str = "Volume"):
        """
        ブレイクアウト戦略の初期化。

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
        self.high_prices = {}   # 高値を記録する辞書
        
        # デフォルトパラメータの設定
        default_params = {
            "volume_threshold": 1.2,   # 出来高増加率の閾値（20%）
            "take_profit": 0.03,       # 利益確定（3%）
            "look_back": 1,            # 前日からのブレイクアウトを見る日数
            "trailing_stop": 0.02,     # トレーリングストップ（高値から2%下落）
            "breakout_buffer": 0.01,    # ブレイクアウト判定の閾値（1%）
            
            # トレンドフィルター設定
            "trend_filter_enabled": True,  # 統一トレンド判定の有効化
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
        
        # 統一トレンド検出器の初期化
        # 最新時点でのトレンド判定をコンソールに出力
        if len(self.data) > 0:
            try:
                trend, confidence = detect_unified_trend_with_confidence(
                    self.data, self.price_column, strategy="Breakout"
                )
                self.logger.info(f"現在のトレンド: {trend}, 信頼度: {confidence:.1%}")
            except Exception as e:
                self.logger.warning(f"トレンド判定エラー: {e}")
                
    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        条件:
        - 前日高値を上抜けた場合
        - 出来高が前日比で20%増加している
        - トレンド判定が上昇トレンド（オプション）

        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        # 十分なデータがない場合はエントリーなし
        look_back = self.params.get("look_back", 1)
        if idx <= look_back:
            return 0
            
        # トレンド確認（統一トレンド判定を使用）
        use_trend_filter = self.params.get("trend_filter_enabled", False)
        if use_trend_filter:
            trend = detect_unified_trend(self.data.iloc[:idx + 1], self.price_column, strategy="Breakout")
            allowed_trends = self.params.get("allowed_trends", ["uptrend"])
            # 許可されたトレンドでのみエントリー
            if trend not in allowed_trends:
                return 0  # トレンド不適合
        
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

        if price_breakout and volume_increase:
            # エントリー価格と高値を記録
            self.entry_prices[idx] = current_price
            if 'High' in self.data.columns:
                self.high_prices[idx] = self.data['High'].iloc[idx]
            else:
                self.high_prices[idx] = current_price
                
            self.log_trade(f"Breakout エントリーシグナル: 日付={self.data.index[idx]}, 価格={current_price}, 前日高値={previous_high}")
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

        if latest_entry_date not in self.entry_prices:
            self.entry_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        if latest_entry_date not in self.high_prices and 'High' in self.data.columns:
            self.high_prices[latest_entry_date] = self.data['High'].iloc[latest_entry_pos]
        elif latest_entry_date not in self.high_prices:
            self.high_prices[latest_entry_date] = self.data[self.price_column].iloc[latest_entry_pos]
            
        entry_price = self.entry_prices[latest_entry_date]
        high_price = self.high_prices[latest_entry_date]
        current_price = self.data[self.price_column].iloc[idx]
        
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

    def backtest(self):
        """
        ブレイクアウト戦略のバックテストを実行する。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
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

    def run_optimized_strategy(self) -> pd.DataFrame:
        """
        最適化されたパラメータを使用して戦略を実行
        
        Returns:
            pd.DataFrame: 戦略実行結果
        """
        # 最適化パラメータの読み込み
        if hasattr(self, 'optimization_mode') and self.optimization_mode and not self.load_optimized_parameters():
            print(f"⚠️ 最適化パラメータの読み込みに失敗しました。デフォルトパラメータを使用します。")
        
        # 使用するパラメータの表示
        if hasattr(self, '_approved_params') and self._approved_params:
            print(f"✅ 最適化パラメータを使用:")
            print(f"   パラメータID: {self._approved_params.get('parameter_id', 'N/A')}")
            print(f"   作成日時: {self._approved_params.get('created_at', 'N/A')}")
            print(f"   シャープレシオ: {self._approved_params.get('sharpe_ratio', 'N/A')}")
            print(f"   パラメータ: {self._approved_params.get('parameters', {})}")
        else:
            print(f"📊 デフォルトパラメータを使用: {self.params}")
        
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
            params = manager.get_latest_approved_parameters('breakout', ticker)
            
            if params:
                # パラメータを更新
                self.params.update(params['parameters'])
                self._approved_params = params
                print(f"✅ 最適化パラメータを読み込みました (ID: {params.get('parameter_id', 'N/A')})")
                return True
            else:
                print(f"⚠️ 承認済みの最適化パラメータが見つかりません")
                return False
                
        except Exception as e:
            print(f"❌ 最適化パラメータの読み込みでエラー: {e}")
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