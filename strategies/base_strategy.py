"""
Module: base_strategy
File: base_strategy.py
Description: 
  全ての投資戦略の基底クラスを提供します。エントリー・イグジットシグナルの生成インターフェース、
  バックテスト実行、ログ記録などの共通機能を実装しています。
  各戦略クラスはこのBaseStrategyを継承して実装します。

Author: kouseven7
Created: 2023-01-01
Modified: 2025-04-02

Dependencies:
  - pandas
  - numpy
  - logging
"""

# ファイル: strategies/base_strategy.py
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import logging

class BaseStrategy:
    """
    BaseStrategyは、全戦略に共通する基本処理（パラメータ初期化、エントリー／イグジット判定、ログ出力など）を実装する基底クラスです。
    各戦略は、このクラスを継承して固有のシグナル生成ロジックを実装してください。
    """
    def __init__(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """
        基本戦略の初期化。
        
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict, optional): 戦略パラメータ（カスタマイズ可能）
        """
        self.data = data
        self.params = params or {}
        self.logger = self._setup_logger()
        self.initialize_strategy()
        
    def _setup_logger(self) -> logging.Logger:
        """
        ロガーの初期設定を行う。
        
        Returns:
            logging.Logger: 設定されたロガーインスタンス
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:  # 既にハンドラが設定されていない場合のみ
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # コンソールにログを出力
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # ファイルにもログを出力（オプション）
            try:
                file_handler = logging.FileHandler('logs/backtest.log')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except:
                logger.warning("ログファイルにアクセスできません。コンソールログのみ使用します。")
                
        return logger

    def initialize_strategy(self) -> None:
        """
        戦略固有の初期化を行う。
        派生クラスで必要に応じてオーバーライドできる。
        """
        self.logger.info(f"{self.__class__.__name__} 初期化: パラメータ = {self.params}")

    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナルを生成する。
        派生クラスが実装する必要がある。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: エントリーシグナル（1: エントリー, 0: なし）
        """
        raise NotImplementedError("派生クラスはgenerate_entry_signalメソッドを実装してください")

    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナルを生成する。
        派生クラスが実装する必要がある。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        """
        raise NotImplementedError("派生クラスはgenerate_exit_signalメソッドを実装してください")

    def log_trade(self, message: str) -> None:
        """
        取引関連のログメッセージを記録する。
        
        Parameters:
            message (str): ログメッセージ
        """
        self.logger.info(message)
        
    def get_latest_entry_price(self, idx: int) -> Optional[float]:
        """
        指定されたインデックスより前の最新のエントリー価格を取得する。
        
        Parameters:
            idx (int): 現在のインデックス
            
        Returns:
            Optional[float]: 最新のエントリー価格（存在しない場合はNone）
        """
        entry_signals = self.data[self.data['Entry_Signal'] == 1].index
        previous_entries = [i for i in entry_signals if i < self.data.index[idx]]
        
        if not previous_entries:
            return None
            
        latest_entry_idx = previous_entries[-1]
        return self.data.loc[latest_entry_idx, 'Adj Close']
        
    def backtest(self) -> pd.DataFrame:
        """
        戦略のバックテストを実行する標準メソッド。
        必要に応じて各戦略でオーバーライドできます。
        
        Returns:
            pd.DataFrame: エントリー/イグジットシグナルが追加されたデータフレーム
        """
        # シグナル列の初期化
        result = self.data.copy()  # データのコピーを作成して元のデータに影響を与えない
        result['Entry_Signal'] = 0
        result['Exit_Signal'] = 0
        
        # 戦略名を追加
        result['Strategy'] = self.__class__.__name__
        
        # インデックスが日時型になっていることを確認
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.DatetimeIndex(result.index)
                self.logger.info("インデックスをDatetimeIndexに変換しました")
            except Exception as e:
                self.logger.warning(f"インデックス変換エラー: {e}")

        in_position = False
        entry_idx = -1
        
        # 各日にちについてシグナルを計算
        for idx in range(len(result)):
            # ポジションを持っていない場合のみエントリーシグナルをチェック
            if not in_position:
                entry_signal = self.generate_entry_signal(idx)
                if entry_signal == 1:
                    result.at[result.index[idx], 'Entry_Signal'] = 1
                    in_position = True
                    entry_idx = idx
            
            # ポジションを持っている場合のみイグジットシグナルをチェック
            elif in_position:
                exit_signal = self.generate_exit_signal(idx)
                if exit_signal == -1:
                    result.at[result.index[idx], 'Exit_Signal'] = -1
                    in_position = False
                    entry_idx = -1
        
        # バックテスト終了時に未決済のポジションがある場合は、最終日に強制決済
        if in_position and entry_idx >= 0:
            last_idx = len(result) - 1
            result.at[result.index[last_idx], 'Exit_Signal'] = -1
            self.logger.info(f"バックテスト終了時のオープンポジションを強制決済: エントリー日={result.index[entry_idx]}, 決済日={result.index[last_idx]}")

        # エントリーとエグジットの回数を検証
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        
        if entry_count != exit_count:
            self.logger.warning(f"エントリー ({entry_count}) とエグジット ({exit_count}) の回数が一致しません！")
            
        return result
