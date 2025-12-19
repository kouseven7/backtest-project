"""
BUY/SELLペア不一致の詳細分析スクリプト

調査目的: execution_details内のBUY/SELL件数を実際に集計し、
          なぜペア不一致が発生するのか根本原因を特定する

根拠データ:
- dssms_execution_results.json内のexecution_details
- ComprehensiveReporterのログ出力
"""
import json
import pandas as pd
from collections import defaultdict

# 1. execution_results.jsonから実際のexecution_detailsを読み込み
json_path = 'output/dssms_integration/dssms_20251217_002959/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("【A1】execution_details実件数分析")
print("=" * 80)

# 2. action別集計
action_count = defaultdict(int)
execution_type_count = defaultdict(int)
strategy_count = defaultdict(int)

for detail in execution_details:
    action = detail.get('action', '')
    exec_type = detail.get('execution_type', '')
    strategy = detail.get('strategy_name', '')
    
    action_count[action] += 1
    execution_type_count[exec_type] += 1
    strategy_count[strategy] += 1

print(f"\n[集計1] action別件数:")
for action, count in sorted(action_count.items()):
    print(f"  {action}: {count}件")

print(f"\n[集計2] execution_type別件数:")
for exec_type, count in sorted(execution_type_count.items()):
    print(f"  {exec_type}: {count}件")

print(f"\n[集計3] strategy_name別件数:")
for strategy, count in sorted(strategy_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {strategy}: {count}件")

# 3. 銘柄別BUY/SELL集計
symbol_buy_sell = defaultdict(lambda: {'BUY': [], 'SELL': []})

for i, detail in enumerate(execution_details):
    action = detail.get('action', '')
    symbol = detail.get('symbol', '')
    timestamp = detail.get('timestamp', '')
    strategy = detail.get('strategy_name', '')
    exec_type = detail.get('execution_type', '')
    
    if action in ['BUY', 'SELL'] and symbol:
        symbol_buy_sell[symbol][action].append({
            'index': i,
            'timestamp': timestamp,
            'strategy': strategy,
            'execution_type': exec_type
        })

print(f"\n[集計4] 銘柄別BUY/SELL件数:")
print(f"{'銘柄':<10} {'BUY':>5} {'SELL':>5} {'差分':>5}")
print("-" * 30)

total_buy = 0
total_sell = 0

for symbol in sorted(symbol_buy_sell.keys()):
    buy_count = len(symbol_buy_sell[symbol]['BUY'])
    sell_count = len(symbol_buy_sell[symbol]['SELL'])
    diff = buy_count - sell_count
    
    total_buy += buy_count
    total_sell += sell_count
    
    print(f"{symbol:<10} {buy_count:>5} {sell_count:>5} {diff:>+5}")

print("-" * 30)
print(f"{'合計':<10} {total_buy:>5} {total_sell:>5} {total_buy - total_sell:>+5}")

# 4. ForceClose戦略のSELL記録確認
print(f"\n[調査A2] ForceClose戦略のSELL記録:")
force_close_sells = [
    detail for detail in execution_details
    if detail.get('action') == 'SELL' and 'ForceClose' in detail.get('strategy_name', '')
]
print(f"  ForceClose SELL件数: {len(force_close_sells)}")

if len(force_close_sells) > 0:
    print(f"  最初の3件:")
    for i, detail in enumerate(force_close_sells[:3]):
        print(f"    [{i+1}] 銘柄={detail.get('symbol')}, "
              f"timestamp={detail.get('timestamp')}, "
              f"strategy={detail.get('strategy_name')}")

# 5. VWAPBreakoutStrategy戦略の詳細確認
print(f"\n[調査A2-2] VWAPBreakoutStrategy BUY/SELL:")
vwap_buys = [d for d in execution_details if d.get('action') == 'BUY' and d.get('strategy_name') == 'VWAPBreakoutStrategy']
vwap_sells = [d for d in execution_details if d.get('action') == 'SELL' and d.get('strategy_name') == 'VWAPBreakoutStrategy']

print(f"  VWAPBreakoutStrategy BUY: {len(vwap_buys)}件")
print(f"  VWAPBreakoutStrategy SELL: {len(vwap_sells)}件")
print(f"  差分: {len(vwap_buys) - len(vwap_sells)}件")

# 6. execution_type=tradeのみの集計
print(f"\n[重要] execution_type='trade'のみの集計:")
trade_buys = [d for d in execution_details if d.get('action') == 'BUY' and d.get('execution_type') == 'trade']
trade_sells = [d for d in execution_details if d.get('action') == 'SELL' and d.get('execution_type') == 'trade']

print(f"  trade BUY: {len(trade_buys)}件")
print(f"  trade SELL: {len(trade_sells)}件")
print(f"  差分: {len(trade_buys) - len(trade_sells)}件")

# 7. 最後の10件のexecution_detailsを確認
print(f"\n[確認] execution_detailsの最後10件:")
for i in range(max(0, len(execution_details) - 10), len(execution_details)):
    detail = execution_details[i]
    print(f"  [{i}] action={detail.get('action', 'N/A'):<5}, "
          f"symbol={detail.get('symbol', 'N/A'):<6}, "
          f"timestamp={detail.get('timestamp', 'N/A')[:10]}, "
          f"strategy={detail.get('strategy_name', 'N/A')[:20]}")

print("\n" + "=" * 80)
print("分析完了")
print("=" * 80)
