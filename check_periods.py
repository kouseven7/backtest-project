"""
Task 11実装前後のバックテスト期間を比較
"""
import json
from datetime import datetime

def check_period(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trades = data.get('execution_details', [])
    
    if not trades:
        print(f"\n{label}: No trades")
        return
    
    dates = [t['timestamp'] for t in trades]
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    print(f"\n{label}:")
    print(f"  Total trades: {len(trades)}")
    print(f"  BUY: {len(buy_trades)}, SELL: {len(sell_trades)}")
    print(f"  Period: {min(dates)} to {max(dates)}")
    
    # 2023-01-13の8306を確認
    trades_20230113 = [t for t in trades if t['timestamp'] == '2023-01-13' and t['symbol'] == '8306']
    if trades_20230113:
        print(f"  2023-01-13 8306 trades: {len(trades_20230113)}")
        for t in trades_20230113:
            print(f"    {t['action']} {t['strategy_name']} quantity={t['quantity']}")

# 実装前
check_period('output/dssms_integration/dssms_20251208_124418/dssms_execution_results.json', 
             'Task 11実装前')

# 実装後
check_period('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', 
             'Task 11実装後')
