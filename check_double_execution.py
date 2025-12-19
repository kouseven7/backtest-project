"""
二重取引の確認

調査目的:
1. DSSMS由来のswitch取引（7件）
2. main_new.py由来の通常取引
3. 同一銘柄・同一日付の重複確認
"""
import json

json_path = 'output/dssms_integration/dssms_20251219_121821/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("二重取引の確認")
print("=" * 80)

# execution_type別に分類
by_exec_type = {
    'trade': [],
    'force_close': [],
    'switch': []
}

for detail in execution_details:
    exec_type = detail.get('execution_type', 'unknown')
    if exec_type in by_exec_type:
        by_exec_type[exec_type].append(detail)

print(f"\n[1] execution_type別の件数:")
print(f"  trade: {len(by_exec_type['trade'])}件")
print(f"  force_close: {len(by_exec_type['force_close'])}件")
print(f"  switch: {len(by_exec_type['switch'])}件")

# 同一銘柄・同一日付の重複チェック
print("\n" + "=" * 80)
print("[2] 同一銘柄・同一日付の取引確認")
print("=" * 80)

# 銘柄・日付・アクションごとにグループ化
from collections import defaultdict
groups = defaultdict(list)

for detail in execution_details:
    symbol = detail.get('symbol')
    timestamp = detail.get('timestamp', '')[:10]  # 日付部分のみ
    action = detail.get('action')
    exec_type = detail.get('execution_type')
    
    key = f"{symbol}_{timestamp}_{action}"
    groups[key].append({
        'exec_type': exec_type,
        'strategy_name': detail.get('strategy_name'),
        'price': detail.get('executed_price'),
        'quantity': detail.get('quantity')
    })

# 重複（2件以上）があるキーを抽出
duplicates = {k: v for k, v in groups.items() if len(v) > 1}

if duplicates:
    print(f"\n重複取引: {len(duplicates)}件\n")
    for key, details in sorted(duplicates.items()):
        symbol, date, action = key.split('_')
        print(f"[{symbol}] {date} {action}: {len(details)}件")
        for i, d in enumerate(details):
            print(f"  [{i+1}] exec_type={d['exec_type']}, "
                  f"strategy={d['strategy_name']}, "
                  f"price={d['price']:.2f}, "
                  f"qty={d['quantity']:.2f}")
else:
    print("\n重複取引なし")

# strategy_name別の集計
print("\n" + "=" * 80)
print("[3] strategy_name別の集計")
print("=" * 80)

strategy_counts = defaultdict(lambda: {'BUY': 0, 'SELL': 0})
for detail in execution_details:
    strategy = detail.get('strategy_name', 'unknown')
    action = detail.get('action', 'unknown')
    strategy_counts[strategy][action] += 1

for strategy in sorted(strategy_counts.keys()):
    buy = strategy_counts[strategy]['BUY']
    sell = strategy_counts[strategy]['SELL']
    print(f"{strategy:<40}: BUY={buy:>2}, SELL={sell:>2}")

# DSSMS由来とmain_new.py由来の分離
print("\n" + "=" * 80)
print("[4] DSSMS由来 vs main_new.py由来")
print("=" * 80)

dssms_strategies = ['DSSMS_SymbolSwitch', 'DSSMS_BacktestEndForceClose', 'ForceClose']
main_new_strategies = ['GCStrategy', 'VWAPBreakoutStrategy', 'ContrarianStrategy']

dssms_count = sum(1 for d in execution_details if d.get('strategy_name') in dssms_strategies)
main_new_count = sum(1 for d in execution_details if d.get('strategy_name') in main_new_strategies)
other_count = len(execution_details) - dssms_count - main_new_count

print(f"DSSMS由来: {dssms_count}件")
print(f"main_new.py由来: {main_new_count}件")
print(f"その他: {other_count}件")
print(f"総計: {len(execution_details)}件")
