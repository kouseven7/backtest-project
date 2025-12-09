"""BUY=0件の銘柄の詳細確認"""
import json
import pprint

result = json.load(open('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', 'r', encoding='utf-8'))
data = result.get('execution_details', [])

symbols_no_buy = ['4063', '4568', '9735', '8267']

for symbol in symbols_no_buy:
    sells = [d for d in data if d['symbol'] == symbol and d['action'] == 'SELL']
    print(f'\n[{symbol}] SELL件数: {len(sells)}件')
    
    for i, sell in enumerate(sells, 1):
        print(f"\nSELL {i}:")
        print(f"  timestamp: {sell.get('timestamp', 'N/A')}")
        print(f"  strategy: {sell.get('strategy_name', 'N/A')}")
        print(f"  quantity: {sell.get('quantity', 'N/A')}")
        print(f"  price: {sell.get('executed_price', 'N/A')}")
        print(f"  status: {sell.get('status', 'N/A')}")
