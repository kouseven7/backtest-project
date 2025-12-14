"""
execution_details分析スクリプト - 9ペア→3レコード謎の解明

このスクリプトはexecution_detailsの実データを解析し、
データ検証条件をパスするかを検証します。

copilot-instructions.md準拠:
- 実データ検証
- 推測ではなく正確な数値を報告
"""

import json
from pathlib import Path
from collections import defaultdict

# JSONファイル読み込み
json_path = Path("output/dssms_integration/dssms_20251214_213349/dssms_execution_results.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("execution_details分析 - データ検証条件チェック")
print("=" * 80)

# BUY/SELL分類
buy_orders = []
sell_orders = []

for detail in execution_details:
    action = detail.get('action')
    if action == 'BUY':
        buy_orders.append(detail)
    elif action == 'SELL':
        sell_orders.append(detail)

print(f"\nBUY注文: {len(buy_orders)}件")
print(f"SELL注文: {len(sell_orders)}件")

# 銘柄別にグループ化
buy_by_symbol = defaultdict(list)
sell_by_symbol = defaultdict(list)

for buy in buy_orders:
    symbol = buy.get('symbol')
    if symbol:
        buy_by_symbol[symbol].append(buy)
        
for sell in sell_orders:
    symbol = sell.get('symbol')
    if symbol:
        sell_by_symbol[symbol].append(sell)

# すべての銘柄について銘柄別FIFOペアリング
all_symbols = set(buy_by_symbol.keys()) | set(sell_by_symbol.keys())

print(f"\n処理対象銘柄数: {len(all_symbols)}")
print(f"BUY銘柄: {len(buy_by_symbol)}")
print(f"SELL銘柄: {len(sell_by_symbol)}")

total_pairs = 0
valid_pairs = 0
invalid_pairs = 0

print("\n" + "=" * 80)
print("銘柄別ペアリング詳細")
print("=" * 80)

for symbol in sorted(all_symbols):
    buys = buy_by_symbol.get(symbol, [])
    sells = sell_by_symbol.get(symbol, [])
    paired_count = min(len(buys), len(sells))
    
    print(f"\n【銘柄: {symbol}】")
    print(f"  BUY件数: {len(buys)}")
    print(f"  SELL件数: {len(sells)}")
    print(f"  ペア数: {paired_count}")
    
    total_pairs += paired_count
    
    # 各ペアのデータ検証
    for i in range(paired_count):
        buy_order = buys[i]
        sell_order = sells[i]
        
        entry_date = buy_order.get('timestamp')
        exit_date = sell_order.get('timestamp')
        entry_price = buy_order.get('executed_price', 0.0)
        exit_price = sell_order.get('executed_price', 0.0)
        shares = buy_order.get('quantity', 0)
        
        # データ検証条件: if not all([entry_date, exit_date, entry_price > 0, exit_price > 0, shares > 0])
        is_valid = all([entry_date, exit_date, entry_price > 0, exit_price > 0, shares > 0])
        
        if is_valid:
            valid_pairs += 1
            status = "✓ VALID"
        else:
            invalid_pairs += 1
            status = "✗ INVALID (データ検証失敗)"
        
        print(f"\n  ペア{i+1}: {status}")
        print(f"    BUY  - timestamp: {entry_date}, executed_price: {entry_price}, quantity: {shares}")
        print(f"    SELL - timestamp: {exit_date}, executed_price: {exit_price}, quantity: {sell_order.get('quantity', 0)}")
        print(f"    戦略: {buy_order.get('strategy_name', 'Unknown')}")
        print(f"    検証項目:")
        print(f"      - entry_date存在: {bool(entry_date)}")
        print(f"      - exit_date存在: {bool(exit_date)}")
        print(f"      - entry_price > 0: {entry_price > 0} (値: {entry_price})")
        print(f"      - exit_price > 0: {exit_price > 0} (値: {exit_price})")
        print(f"      - shares > 0: {shares > 0} (値: {shares})")

print("\n" + "=" * 80)
print("最終サマリー")
print("=" * 80)
print(f"合計ペア数: {total_pairs}")
print(f"有効ペア数: {valid_pairs}")
print(f"無効ペア数: {invalid_pairs}")
print(f"\n期待される取引レコード数: {valid_pairs}件")
print(f"実際の取引レコード数: 3件 (dssms_trade_analysis.jsonより)")
print(f"\n差分: {valid_pairs - 3}件のペアが謎のスキップ")
