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
        
        if idx < look_back:  # 過去データが必要
            return 0
            
        if 'High' not in self.data.columns:
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

        if price_breakout and volume_increase:
            # Phase 1a修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス修正）
            # Phase 2修正: スリッページ・取引コスト対応（2025-12-23追加）
            # 理由: idx日の終値を見てからidx日の終値で買うことは不可能
            # リアルトレードでは翌日（idx+1日目）の始値でエントリー
            next_day_open = self.data['Open'].iloc[idx + 1]
            
            # Phase 2: スリッページ・取引コスト適用（買い注文は不利な方向）
            # デフォルト: slippage=0.001（0.1%）、transaction_cost=0.0（0%）
            slippage = self.params.get("slippage", 0.001)
            transaction_cost = self.params.get("transaction_cost", 0.0)
            entry_price = next_day_open * (1 + slippage + transaction_cost)
            self.entry_prices[idx] = entry_price  # スリッページ適用後の価格を記録
            
            # 高値の初期値も翌日始値を使用（トレーリングストップの起点）
            if 'High' in self.data.columns:
                # 翌日の高値を使用
                self.high_prices[idx] = self.data['High'].iloc[idx + 1]
            else:
                self.high_prices[idx] = next_day_open
                
            self.log_trade(f"Breakout エントリーシグナル: 日付={self.data.index[idx]}, 価格={next_day_open}, 前日高値={previous_high}")
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