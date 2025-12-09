"""DSSMS_SymbolSwitchの詳細分析"""
import json
from collections import defaultdict

result = json.load(open('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', 'r', encoding='utf-8'))
data = result.get('execution_details', [])

# DSSMS_SymbolSwitchのBUYとSELLを集計
dssms_switch = [d for d in data if 'DSSMS_SymbolSwitch' in d.get('strategy_name', '')]

buy_count = sum(1 for d in dssms_switch if d['action'] == 'BUY')
sell_count = sum(1 for d in dssms_switch if d['action'] == 'SELL')

print("=== DSSMS_SymbolSwitch分析 ===")
print(f"総件数: {len(dssms_switch)}件")
print(f"  BUY: {buy_count}件")
print(f"  SELL: {sell_count}件")
print(f"  差分: {sell_count - buy_count}件（SELL{'超過' if sell_count > buy_count else '不足'}）\n")

# 日付順にソート
dssms_switch_sorted = sorted(dssms_switch, key=lambda x: x.get('timestamp', ''))

print("=== DSSMS_SymbolSwitch取引履歴（最初の20件） ===")
print(f"{'No':>3} {'日付':<12} {'Action':<6} {'銘柄':<8} {'数量':>12}")
print("-" * 50)

for i, d in enumerate(dssms_switch_sorted[:20], 1):
    date = d.get('timestamp', 'N/A')[:10]
    action = d.get('action', 'N/A')
    symbol = d.get('symbol', 'N/A')
    quantity = d.get('quantity', 0)
    print(f"{i:>3} {date:<12} {action:<6} {symbol:<8} {quantity:>12.2f}")

# ペアリング確認（同日のBUY/SELLをペアにする）
print("\n=== DSSMS_SymbolSwitch ペアリング確認（最初の10日） ===")

by_date = defaultdict(lambda: {'BUY': [], 'SELL': []})
for d in dssms_switch:
    date = d.get('timestamp', '')[:10]
    action = d.get('action', '')
    by_date[date][action].append(d)

dates = sorted(by_date.keys())[:10]

for date in dates:
    buys = by_date[date]['BUY']
    sells = by_date[date]['SELL']
    print(f"\n{date}:")
    print(f"  BUY: {len(buys)}件", end="")
    if buys:
        print(f" - {', '.join([b['symbol'] for b in buys])}")
    else:
        print()
    print(f"  SELL: {len(sells)}件", end="")
    if sells:
        print(f" - {', '.join([s['symbol'] for s in sells])}")
    else:
        print()

print("\n=== 銘柄切替パターンの推定 ===")
print("注意: DSSMS_SymbolSwitchは「SELL（旧銘柄） → BUY（新銘柄）」のペア")
print("BUY=0件の銘柄は、DSSMS側の初期銘柄（execution_detailsに初期BUY記録なし）")
