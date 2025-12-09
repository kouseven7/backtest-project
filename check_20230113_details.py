"""
2023-01-13の8306取引の詳細を確認
"""
import json

def check_8306_trades(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trades = data.get('execution_details', [])
    
    # 2023-01-13の8306を抽出
    trades_20230113 = [t for t in trades if '2023-01-13' in t['timestamp'] and t['symbol'] == '8306']
    
    print(f"\n{label}:")
    print(f"  2023-01-13の8306取引: {len(trades_20230113)}件")
    
    for i, t in enumerate(trades_20230113, 1):
        print(f"\n  [{i}] {t['action']} {t['strategy_name']}")
        print(f"      Price: {t.get('execution_price', t.get('price', 'N/A'))}, Quantity: {t.get('quantity', 'N/A')}")
        print(f"      Status: {t.get('status', 'N/A')}")
        print(f"      Timestamp: {t['timestamp']}")
        print(f"      All keys: {list(t.keys())}")

# 実装前
check_8306_trades('output/dssms_integration/dssms_20251208_124418/dssms_execution_results.json', 
                  'Task 11実装前')

# 実装後
check_8306_trades('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', 
                  'Task 11実装後')
