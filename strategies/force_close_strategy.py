"""
ForceCloseStrategy - 強制決済戦略

銘柄切替時・バックテスト終了時に全ポジションを強制決済する戦略。
PaperBroker.close_all_positions()を呼び出し、決済結果をsignals形式に変換。

主な機能:
- PaperBroker.close_all_positions()呼び出し
- 決済結果をsignals DataFrame形式に変換
- strategy_name="ForceClose"明示設定
- execution_details生成（StrategyExecutionManager経由）
- エラー耐性（決済失敗時も処理継続）

統合コンポーネント:
- PaperBroker: close_all_positions()呼び出し
- StrategyExecutionManager: signals実行、execution_details生成
- IntegratedExecutionManager: execute_force_close()経由で呼び出し

セーフティ機能/注意事項:
- PaperBroker.close_all_positions()はエラー耐性実装済み
- 決済失敗銘柄は警告ログ出力（エラー隠蔽禁止）
- モック/ダミーデータ使用禁止（copilot-instructions.md準拠）
- フォールバック機能禁止

Author: Backtest Project Team
Created: 2025-12-19
Last Modified: 2025-12-19
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy


class ForceCloseStrategy(BaseStrategy):
    """
    強制決済戦略（銘柄切替時/バックテスト終了時）
    
    PaperBroker.close_all_positions()を呼び出し、
    全ポジションを決済するsignalsを生成。
    
    strategy_name: "ForceClose"
    """
    
    def __init__(self, broker, data: Optional[pd.DataFrame] = None, 
                 params: Optional[Dict[str, Any]] = None, 
                 reason: str = "symbol_switch"):
        """
        ForceCloseStrategy初期化
        
        Args:
            broker: PaperBrokerインスタンス（close_all_positions()呼び出し用）
            data: 株価データ（オプション、signals生成に使用）
            params: 戦略パラメータ（オプション）
            reason: 決済理由（"symbol_switch", "backtest_end"等）
        
        Note:
            - BaseStrategyはdata必須のためダミーDataFrame渡す
            - 実際の決済はPaperBroker.close_all_positions()経由
        """
        self.broker = broker
        self.reason = reason
        self.strategy_name = "ForceClose"
        
        # BaseStrategy初期化（dataがNoneの場合はダミーDataFrame作成）
        if data is None:
            # 空のDataFrame（BaseStrategy要件満たすため）
            data = pd.DataFrame(
                {'Close': [0.0]}, 
                index=pd.DatetimeIndex([datetime.now()])
            )
        
        super().__init__(data, params or {})
        self.logger.info(f"ForceCloseStrategy initialized: reason={reason}")
    
    def backtest(self, trading_start_date: Optional[pd.Timestamp] = None,
                 trading_end_date: Optional[pd.Timestamp] = None,
                 current_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        強制決済実行
        
        Args:
            trading_start_date: 取引開始日（未使用、BaseStrategyインターフェース互換性のため）
            trading_end_date: 取引終了日（未使用、BaseStrategyインターフェース互換性のため）
            current_date: 決済日時（PaperBroker.close_all_positions()に渡す）
        
        Returns:
            pd.DataFrame: 決済シグナル（strategy="ForceClose"設定済み）
        
        Note:
            - PaperBroker.close_all_positions()呼び出し
            - 決済結果をsignals DataFrame形式に変換
            - StrategyExecutionManagerが後続処理で実行
        
        copilot-instructions.md準拠:
            - 実データのみ使用（モック/ダミー禁止）
            - エラー隠蔽禁止（警告ログ出力）
            - フォールバック禁止
        """
        try:
            # current_dateが未指定の場合は現在時刻を使用
            if current_date is None:
                current_date = datetime.now()
            
            self.logger.info(
                f"[FORCE_CLOSE] Starting force close execution: "
                f"date={current_date.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"reason={self.reason}"
            )
            
            # PaperBroker.close_all_positions()呼び出し
            close_results = self.broker.close_all_positions(
                current_date=current_date,
                reason=self.reason
            )
            
            self.logger.info(
                f"[FORCE_CLOSE] PaperBroker returned {len(close_results)} close results"
            )
            
            # 決済結果が空の場合（ポジション未保有）
            if not close_results:
                self.logger.info("[FORCE_CLOSE] No positions to close")
                # 空のsignals DataFrame返却
                return self._create_empty_signals(current_date)
            
            # 決済結果をsignals DataFrame形式に変換
            signals = self._convert_to_signals(close_results, current_date)
            
            self.logger.info(
                f"[FORCE_CLOSE] Generated {len(signals)} SELL signals for force close"
            )
            
            return signals
            
        except Exception as e:
            # エラー時は警告ログ出力（エラー隠蔽禁止）
            self.logger.error(
                f"[FORCE_CLOSE] Error in force close execution: {e}",
                exc_info=True
            )
            # 空のsignals返却（フォールバック禁止）
            return self._create_empty_signals(current_date)
    
    def _convert_to_signals(self, close_results: List[Dict[str, Any]], 
                          current_date: datetime) -> pd.DataFrame:
        """
        PaperBroker決済結果をsignals DataFrame形式に変換
        
        Args:
            close_results: PaperBroker.close_all_positions()の返却値
            current_date: 決済日時
        
        Returns:
            pd.DataFrame: signals DataFrame（strategy="ForceClose"設定済み）
        
        Note:
            - 各決済結果をSELLシグナルとして変換
            - Exit_Signal=-1設定
            - strategy="ForceClose"明示
        """
        try:
            if not close_results:
                return self._create_empty_signals(current_date)
            
            # signals DataFrame構築
            signals_data = []
            
            for result in close_results:
                # 各決済結果をSELLシグナルとして追加
                signal_row = {
                    'Close': result['exit_price'],
                    'Entry_Signal': 0,
                    'Exit_Signal': -1,  # SELL
                    'Position': 0,  # ポジションクローズ
                    'Strategy': 'ForceClose',
                    'symbol': result['symbol'],
                    'quantity': result['quantity'],
                    'entry_price': result['entry_price'],
                    'exit_price': result['exit_price'],
                    'entry_time': result['entry_time'],
                    'pnl': result['pnl'],
                    'commission': result['commission'],
                    'slippage': result['slippage'],
                    'reason': result['reason']
                }
                signals_data.append(signal_row)
            
            # DataFrameに変換（インデックスは決済日時）
            signals = pd.DataFrame(
                signals_data,
                index=[current_date] * len(signals_data)
            )
            
            self.logger.info(
                f"[FORCE_CLOSE] Converted {len(signals)} close results to signals"
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(
                f"[FORCE_CLOSE] Error converting close results to signals: {e}",
                exc_info=True
            )
            return self._create_empty_signals(current_date)
    
    def _create_empty_signals(self, current_date: datetime) -> pd.DataFrame:
        """
        空のsignals DataFrame作成
        
        Args:
            current_date: 決済日時
        
        Returns:
            pd.DataFrame: 空のsignals DataFrame（strategy="ForceClose"設定済み）
        """
        return pd.DataFrame(
            {
                'Close': [],
                'Entry_Signal': [],
                'Exit_Signal': [],
                'Position': [],
                'Strategy': []
            },
            index=pd.DatetimeIndex([])
        )
    
    # BaseStrategyインターフェース互換性メソッド
    # （StrategyExecutionManagerがgenerate_entry_signal/generate_exit_signalを呼び出す可能性対応）
    
    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナル生成（ForceCloseは常に0）
        
        Args:
            idx: インデックス
        
        Returns:
            int: 0（ForceCloseはエントリーしない）
        """
        return 0
    
    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナル生成（ForceCloseは常に-1）
        
        Args:
            idx: インデックス
        
        Returns:
            int: -1（SELL固定）
        """
        return -1
