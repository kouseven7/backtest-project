"""
DSSMS Excel出力修正 Phase 3: 損益計算の修正
各取引の正確な損益計算を実装
"""

def improved_trade_conversion_with_accurate_pnl():
    """
    改良版の取引変換ロジック - 正確な損益計算付き
    """
    
    fixed_method = '''
def _convert_switches_to_trades(self) -> list:
    """
    switch_historyを個別取引リストに変換（損益計算修正版）
    
    Returns:
        list: 個別取引のリスト
    """
    try:
        trades = []
        
        self.logger.info(f"銘柄切り替え履歴を個別取引に変換中: {len(self.switch_history)}件")
        
        for i, switch in enumerate(self.switch_history):
            # 切り替え情報の取得
            switch_time = getattr(switch, 'switch_time', None) or getattr(switch, 'timestamp', None)
            from_symbol = getattr(switch, 'from_symbol', None)
            to_symbol = getattr(switch, 'to_symbol', None)
            
            # 正確な損益・コスト情報を取得
            profit_loss = getattr(switch, 'profit_loss_at_switch', 0)
            switch_cost = getattr(switch, 'switch_cost', 0)
            holding_period = getattr(switch, 'holding_period_hours', 0)
            
            # Portfolio価値の変化
            portfolio_before = getattr(switch, 'portfolio_value_before', 0)
            portfolio_after = getattr(switch, 'portfolio_value_after', 0)
            
            # 理由・トリガー情報
            reason = getattr(switch, 'reason', 'DSSMS切り替え')
            trigger = getattr(switch, 'trigger', 'daily_evaluation')
            
            if not switch_time:
                self.logger.warning(f"切り替え{i+1}: 日時情報なし")
                continue
            
            # 前のポジションのExit取引（初回以外）
            if i > 0 and from_symbol:
                # 前回の切り替えからこの切り替えまでの損益を計算
                prev_switch = self.switch_history[i-1]
                prev_portfolio_value = getattr(prev_switch, 'portfolio_value_after', portfolio_before)
                
                # この期間の実際の損益（手数料除く）
                period_pnl = portfolio_before - prev_portfolio_value + switch_cost/2
                
                exit_trade = {
                    'trade_id': f"DSSMS_EXIT_{i}",
                    'date': switch_time,
                    'symbol': from_symbol,
                    'action': 'SELL',
                    'strategy': f"DSSMS_{trigger}",
                    'entry_date': getattr(prev_switch, 'switch_time', switch_time),
                    'exit_date': switch_time,
                    'pnl': period_pnl,  # 正確な期間損益
                    'holding_period_hours': holding_period,
                    'switch_cost': switch_cost / 2,  # ExitとEntryで分割
                    'reason': f"Exit_{reason}",
                    'portfolio_value_before': prev_portfolio_value,
                    'portfolio_value_after': portfolio_before,
                    'trade_type': 'EXIT'
                }
                trades.append(exit_trade)
                
                self.logger.debug(f"Exit取引: {from_symbol} 損益={period_pnl:.0f}円")
            
            # 新しいポジションのEntry取引
            if to_symbol:
                entry_trade = {
                    'trade_id': f"DSSMS_ENTRY_{i+1}",
                    'date': switch_time,
                    'symbol': to_symbol,
                    'action': 'BUY',
                    'strategy': f"DSSMS_{trigger}",
                    'entry_date': switch_time,
                    'exit_date': None,  # 次の切り替えまたは期間終了
                    'pnl': 0,  # Entry時点では未実現
                    'holding_period_hours': 0,  # 未完了
                    'switch_cost': switch_cost / 2,  # ExitとEntryで分割
                    'reason': f"Entry_{reason}",
                    'portfolio_value_before': portfolio_before,
                    'portfolio_value_after': portfolio_after,
                    'trade_type': 'ENTRY'
                }
                trades.append(entry_trade)
                
                self.logger.debug(f"Entry取引: {to_symbol}")
        
        # 最後のポジションの決済処理
        if trades and len(self.switch_history) > 0:
            last_switch = self.switch_history[-1]
            final_portfolio_value = getattr(last_switch, 'portfolio_value_after', 0)
            
            # 最後のエントリーの最終決済
            last_entry_trades = [t for t in trades if t['trade_type'] == 'ENTRY']
            if last_entry_trades:
                last_entry = last_entry_trades[-1]
                initial_value = last_entry['portfolio_value_after']
                final_pnl = final_portfolio_value - initial_value
                
                final_exit = {
                    'trade_id': f"DSSMS_FINAL_EXIT",
                    'date': last_switch.switch_time,
                    'symbol': last_entry['symbol'],
                    'action': 'SELL',
                    'strategy': 'DSSMS_FINAL',
                    'entry_date': last_entry['entry_date'],
                    'exit_date': last_switch.switch_time,
                    'pnl': final_pnl,
                    'holding_period_hours': getattr(last_switch, 'holding_period_hours', 0),
                    'switch_cost': 0,  # 最終決済は手数料なし
                    'reason': 'Final_Settlement',
                    'portfolio_value_before': initial_value,
                    'portfolio_value_after': final_portfolio_value,
                    'trade_type': 'FINAL_EXIT'
                }
                trades.append(final_exit)
                
                self.logger.debug(f"最終決済: 損益={final_pnl:.0f}円")
        
        self.logger.info(f"個別取引変換完了: {len(trades)}件の取引生成")
        
        # 損益合計をチェック
        total_pnl = sum(t['pnl'] for t in trades)
        self.logger.info(f"計算された総損益: {total_pnl:.0f}円")
        
        return trades
        
    except Exception as e:
        self.logger.error(f"切り替え→取引変換エラー: {e}")
        return []
'''
    
    return fixed_method

def explain_pnl_calculation_fix():
    """損益計算修正の説明"""
    
    explanation = '''
🔧 損益計算修正のポイント:

❌ 修正前の問題:
- 各取引の損益が全期間のポートフォリオ価値として計算
- 11,873,981,430円という異常な値
- 実際の切り替え損益が反映されていない

✅ 修正後の改善:
- 期間損益 = 現在のポートフォリオ価値 - 前回のポートフォリオ価値
- 手数料を適切に考慮
- 最終決済の正確な処理
- 現実的な損益額

📊 計算式:
```
Exit取引の損益 = portfolio_before - prev_portfolio_after + 手数料調整
Entry取引の損益 = 0 (未実現)
最終決済損益 = final_portfolio_value - last_entry_value
```

🎯 期待される結果:
- 117件の取引履歴
- 現実的な損益額（数万円〜数十万円レベル）
- 正確な勝率計算
- 実用的なパフォーマンス分析
'''
    
    print(explanation)

if __name__ == "__main__":
    print("DSSMS Excel出力修正 Phase 3: 損益計算の修正")
    
    fixed_method = improved_trade_conversion_with_accurate_pnl()
    print("修正されたメソッド:")
    print(fixed_method)
    
    explain_pnl_calculation_fix()
    
    print("\n次のステップ:")
    print("1. src/dssms/dssms_backtester.py の _convert_switches_to_trades メソッドを上記で置換")
    print("2. python src\\dssms\\dssms_backtester.py で修正版をテスト")
    print("3. verify_dssms_fix.py で結果を検証")
