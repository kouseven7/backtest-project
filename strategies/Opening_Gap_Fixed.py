"""
Module: Opening_Gap_Fixed

*** 警告: この戦略は使用不可と結論されました ***

Phase B-3検証結果（2025-10-30）:
- 親クラスOpeningGapStrategyが2022-2024データで壊滅的性能を示したため、
  この修正版も使用不可と判断されました。
- 元の戦略の結果: 全銘柄で3.7%～19.1%勝率, -100%～-231% P&L
- 結論: 同日Entry/Exit問題の修正だけでは根本的な性能問題は解決できません

このファイルは参考・アーカイブ目的でのみ保持されています。
実運用・バックテストでは使用しないでください。

Description:
  Opening Gapストラテジーの修正版。
  同日のEntry/Exit問題を修正したバージョン。

Author: imega
Created: 2025-10-15
Modified: 2025-10-30 (Phase B-3完了: 使用不可と結論)
"""

from strategies.Opening_Gap import OpeningGapStrategy
import pandas as pd
from typing import Optional, Dict, Any

class OpeningGapFixedStrategy(OpeningGapStrategy):
    """
    同日Entry/Exit問題を修正したOpeningGapStrategy
    
    元のOpeningGapStrategyを継承し、backtest()メソッドのみを
    ポジション管理を適切に行うように修正しています。
    """
    
    def __init__(self, data: pd.DataFrame, dow_data: pd.DataFrame, params: Optional[Dict[str, Any]] = None, price_column: str = "Adj Close"):
        """初期化は親クラスと同じ"""
        super().__init__(data, dow_data, params, price_column)
    
    def backtest(self):
        """バックテストに一部利確機能を追加（ポジション管理改善版）"""
        self.data['Entry_Signal'] = 0
        self.data['Exit_Signal'] = 0
        self.data['Position_Size'] = 0.0  # ポジションサイズ追跡用
        self.data['Partial_Exit'] = 0     # 一部利確用
        
        # バックテストループ
        for idx in range(len(self.data)):
            current_position = self.data['Position_Size'].iloc[idx-1] if idx > 0 else 0.0
            
            # ポジションがない場合のみエントリーシグナルをチェック
            if current_position == 0.0:
                entry_signal_window = self.data['Entry_Signal'].iloc[max(0, idx-1):idx+1]
                if not bool(entry_signal_window.values.any()):
                    entry_signal = self.generate_entry_signal(idx)
                    if entry_signal == 1:
                        self.data.at[self.data.index[idx], 'Entry_Signal'] = 1
                        self.data.at[self.data.index[idx], 'Position_Size'] = 1.0  # エントリー時にポジションサイズを1に設定
            
            # ポジションがある場合のみイグジットシグナルをチェック
            elif current_position > 0.0:
                exit_signal = self.generate_exit_signal(idx)
                if exit_signal == -1:
                    self.data.at[self.data.index[idx], 'Exit_Signal'] = -1
                    self.data.at[self.data.index[idx], 'Position_Size'] = 0.0  # イグジット時にポジションサイズを0に設定

            # 一部利確処理（ポジションがある場合のみ）
            if current_position > 0.0 and self.params.get("partial_exit_enabled", False) and idx > 0:
                if self.data['Partial_Exit'].iloc[idx-1] == 0:
                    entry_indices = self.data[self.data['Entry_Signal'] == 1].index
                    if len(entry_indices) > 0:
                        latest_entry_idx = self.data.index.get_loc(entry_indices[-1])
                        entry_price = self.entry_prices.get(latest_entry_idx)
                        current_price = self.data[self.price_column].iloc[idx]
                        
                        # 利益率が閾値を超えたら一部利確
                        if entry_price and (current_price / entry_price - 1) >= self.params["partial_exit_threshold"]:
                            portion = self.params["partial_exit_portion"]
                            self.data.at[self.data.index[idx], 'Partial_Exit'] = portion
                            self.data.at[self.data.index[idx], 'Position_Size'] -= portion
                            self.log_trade(f"一部利確 {portion*100}%: 日付={self.data.index[idx]}")

        return self.data