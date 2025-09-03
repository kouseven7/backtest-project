
def _prepare_excel_data_improved(self) -> pd.DataFrame:
    """
    DSSMSバックテストデータをExcel出力システム用に変換（改善版）
    各銘柄切り替えを個別の取引として正確に分離
    """
    try:
        import pandas as pd
        
        # 各銘柄切り替えを個別の取引として処理
        trades = []
        
        for i, switch in enumerate(self.switch_history):
            trade = {
                'Date': getattr(switch, 'switch_time', None),
                'Strategy': f"DSSMS_{getattr(switch, 'to_symbol', 'Unknown')}",
                'Symbol': getattr(switch, 'to_symbol', 'Unknown'),
                'Entry_Date': getattr(switch, 'switch_time', None),
                'Exit_Date': None,  # 次の切り替え日または最終日
                'Entry_Price': 0,   # 要計算
                'Exit_Price': 0,    # 要計算
                'Quantity': 0,      # 要計算
                'Trade_Amount': getattr(switch, 'portfolio_value_after', 0) - getattr(switch, 'portfolio_value_before', 0),
                'Commission': 0,    # 要取得
                'Trade_Result': 0,  # 要計算
                'Holding_Days': 0   # 要計算
            }
            
            # 次の切り替え日をExit_Dateとして設定
            if i + 1 < len(self.switch_history):
                trade['Exit_Date'] = getattr(self.switch_history[i + 1], 'switch_time', None)
            else:
                # 最後の取引は期間終了日
                trade['Exit_Date'] = self.portfolio_history[-1].get('date') if self.portfolio_history else None
            
            trades.append(trade)
        
        # DataFrameに変換
        trades_df = pd.DataFrame(trades)
        
        return trades_df
        
    except Exception as e:
        self.logger.error(f"Excel用データ準備エラー（改善版）: {e}")
        return pd.DataFrame()
