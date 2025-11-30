"""
Module: enhanced_base_strategy
File: enhanced_base_strategy.py
Description: 
  BaseStrategyを拡張し、ポジション管理機能を強化した基底クラス。
  同日のエントリー/エグジット問題を解決するポジション状態追跡機能を実装し、
  すべての戦略で一貫したポジション管理を可能にします。

Author: imega
Created: 2025-10-15
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import BaseStrategy

class EnhancedBaseStrategy(BaseStrategy):
    """
    EnhancedBaseStrategyは、BaseStrategyを拡張し、ポジション管理機能を強化した基底クラスです。
    同日のエントリー/エグジット問題を解決するポジション状態追跡機能を実装し、
    すべての戦略で一貫したポジション管理を可能にします。
    """
    
    def __init__(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        拡張基本戦略の初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（カスタマイズ可能）
        """
        # 親クラスの初期化
        super().__init__(data, params)
        
        # ポジション管理用の追加属性
        self.entry_prices = {}  # エントリー価格を記録する辞書（キー: インデックス、値: エントリー価格）
        self.high_prices = {}   # トレーリングストップ用の最高値を記録する辞書
        self.price_column = (params or {}).get("price_column", "Adj Close")
        
        # 現在のポジションサイズ
        self.current_position = 0.0
        
        # ロガーの取得
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def backtest(self) -> pd.DataFrame:
        """
        バックテスト実行メソッド（オーバーライド）
        拡張されたバックテストメソッドを呼び出します。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルとポジション状態が追加されたデータフレーム
        """
        # 拡張されたバックテストメソッドを呼び出す
        return self.backtest_with_position_tracking()
    
    def backtest_with_position_tracking(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
        """
        ポジション追跡機能を備えた戦略バックテストを実行する拡張メソッド。
        同日Entry/Exit問題を回避するために、ポジション状態を明示的に追跡します。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルとポジション状態が追加されたデータフレーム
        """
        # シグナル列の初期化
        result = self.data.copy()
        result['Entry_Signal'] = 0
        result['Exit_Signal'] = 0
        result['Position_Size'] = 0.0  # ポジションサイズ追跡用
        
        # 戦略名を追加
        result['Strategy'] = self.__class__.__name__
        
        # インデックスが日時型になっていることを確認
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.DatetimeIndex(result.index)
                self.logger.info("インデックスをDatetimeIndexに変換しました")
            except Exception as e:
                self.logger.warning(f"インデックス変換エラー: {e}")
        
        # バックテストループ
        for idx in range(len(result)):
            # 取引期間フィルタリング
            if trading_start_date is not None or trading_end_date is not None:
                current_date = result.index[idx]
                in_trading_period = True
                if trading_start_date is not None and current_date < trading_start_date:
                    in_trading_period = False
                if trading_end_date is not None and current_date > trading_end_date:
                    in_trading_period = False
                if not in_trading_period:
                    # 取引期間外でもPosition_Sizeを維持
                    if idx > 0:
                        result.at[result.index[idx], 'Position_Size'] = float(result['Position_Size'].iloc[idx-1])
                    continue
            
            try:
                # 現在のポジションサイズを取得（前日のポジションサイズ）
                current_position = float(result['Position_Size'].iloc[idx-1]) if idx > 0 else 0.0
                self.current_position = current_position
                
                # ポジションがない場合のみエントリーシグナルをチェック
                if current_position == 0.0:
                    # エントリーシグナルの生成
                    entry_signal = self.generate_entry_signal(idx)
                    if entry_signal == 1:
                        result.at[result.index[idx], 'Entry_Signal'] = 1
                        result.at[result.index[idx], 'Position_Size'] = 1.0
                        self.current_position = 1.0
                
                # ポジションがある場合のみイグジットシグナルをチェック
                elif current_position > 0.0:
                    exit_signal = self.generate_exit_signal(idx)
                    if exit_signal != 0:  # 0以外のイグジットシグナル（-1など）
                        result.at[result.index[idx], 'Exit_Signal'] = exit_signal
                        result.at[result.index[idx], 'Position_Size'] = 0.0
                        self.current_position = 0.0
                else:
                    # ポジションサイズが前日から変わらない場合、前日と同じ値をコピー
                    if idx > 0:
                        result.at[result.index[idx], 'Position_Size'] = float(result['Position_Size'].iloc[idx-1])
            except Exception as e:
                self.logger.error(f"バックテストループエラー（idx={idx}）: {e}")
        
        # リターン・パフォーマンス計算を追加
        if 'Position_Size' in result.columns:
            try:
                # 価格変化率の計算
                returns = result[self.price_column].pct_change()
                
                # 戦略リターンの計算（ポジションサイズに基づく）
                result['Strategy_Return'] = result['Position_Size'].shift(1) * returns
                
                # 累積リターンの計算
                result['Cumulative_Return'] = (1 + result['Strategy_Return']).cumprod()
            except Exception as e:
                self.logger.error(f"リターン計算エラー: {e}")
        
        # バックテスト終了時に未決済のポジションがある場合は、最終日に強制決済
        last_position = result['Position_Size'].iloc[-1]
        if last_position > 0.0:
            last_idx = len(result) - 1
            
            # エントリー日を特定
            entry_indices = result[result['Entry_Signal'] == 1].index
            if len(entry_indices) > 0:
                latest_entry_idx = result.index.get_loc(entry_indices[-1])
                result.at[result.index[last_idx], 'Exit_Signal'] = -1
                result.at[result.index[last_idx], 'Position_Size'] = 0.0
                
                # エントリー価格と最終価格を取得してリターンを計算
                entry_price = self.entry_prices.get(latest_entry_idx)
                if entry_price is None and 'Open' in result.columns:
                    entry_price = float(result['Open'].iloc[latest_entry_idx])
                elif entry_price is None:
                    entry_price = float(result[self.price_column].iloc[latest_entry_idx])
                
                final_price = float(result[self.price_column].iloc[last_idx])
                return_pct = (final_price / entry_price - 1) * 100 if entry_price else 0.0
                
                self.logger.info(f"バックテスト終了時のオープンポジションを強制決済: 決済日={result.index[last_idx]}, 損益={return_pct:.2f}%")
        
        return result
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] != 0).sum()
        
        if entry_count != exit_count:
            self.logger.warning(f"エントリー ({entry_count}) とエグジット ({exit_count}) の回数が一致しません！")
            
        return result
    
    def record_entry_price(self, idx: int, price: float) -> None:
        """
        エントリー価格を記録する
        
        Parameters:
            idx (int): インデックス
            price (float): エントリー価格
        """
        self.entry_prices[idx] = price
    
    def update_trailing_high(self, idx: int, price: float) -> None:
        """
        トレーリングストップのための最高値を更新する
        
        Parameters:
            idx (int): インデックス
            price (float): 現在の価格
        """
        current_high = self.high_prices.get(idx-1, price) if idx > 0 else price
        self.high_prices[idx] = max(current_high, price)
    
    # デフォルトのbacktestメソッドを拡張版で上書き
    def backtest(self, trading_start_date=None, trading_end_date=None) -> pd.DataFrame:
        """
        戦略のバックテストを実行する拡張メソッド。
        ポジション追跡機能を使用してバックテストを行います。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルとポジション状態が追加されたデータフレーム
        """
        return self.backtest_with_position_tracking(trading_start_date, trading_end_date)