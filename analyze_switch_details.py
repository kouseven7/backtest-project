"""
execution_type='switch'の詳細確認スクリプト

調査目的: DSSMS銘柄切替がどのような取引を行っているのか確認
"""
import json

json_path = 'output/dssms_integration/dssms_20251217_214451/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("【B1】DSSMS銘柄切替（execution_type='switch'）詳細確認")
print("=" * 80)

# execution_type='switch'を抽出
switch_details = [
    (i, detail) for i, detail in enumerate(execution_details)
    if detail.get('execution_type') == 'switch'
]

print(f"\n[確認1] execution_type='switch'の件数: {len(switch_details)}件\n")

# 各銘柄切替の詳細表示
for idx, (i, detail) in enumerate(switch_details):
    print(f"[{idx+1}] index={i}")
    print(f"    symbol: {detail.get('symbol')}")
    print(f"    action: {detail.get('action')}")
    print(f"    quantity: {detail.get('quantity')}")
    print(f"    executed_price: {detail.get('executed_price')}")
    print(f"    timestamp: {detail.get('timestamp')}")
    print(f"    strategy_name: {detail.get('strategy_name')}")
    print(f"    profit_pct: {detail.get('profit_pct')}")
    print(f"    close_return: {detail.get('close_return')}")
    print()

# 前後の取引を確認（銘柄切替の前後で何が起きているか）
print("\n[確認2] 銘柄切替の前後の取引:")
for idx, (i, detail) in enumerate(switch_details[:3]):  # 最初の3件
    print(f"\n--- 銘柄切替 {idx+1} の前後 ---")
    
    # 前の取引
    if i > 0:
        prev = execution_details[i-1]
        print(f"  [前] index={i-1}, action={prev.get('action')}, "
              f"symbol={prev.get('symbol')}, "
              f"execution_type={prev.get('execution_type')}, "
              f"strategy={prev.get('strategy_name')}")
    
    # 銘柄切替
    print(f"  [切替] index={i}, action={detail.get('action')}, "
          f"symbol={detail.get('symbol')}, "
          f"execution_type={detail.get('execution_type')}, "
          f"strategy={detail.get('strategy_name')}")
    
    # 次の取引
    if i < len(execution_details) - 1:
        next_detail = execution_details[i+1]
        print(f"  [次] index={i+1}, action={next_detail.get('action')}, "
              f"symbol={next_detail.get('symbol')}, "
              f"execution_type={next_detail.get('execution_type')}, "
              f"strategy={next_detail.get('strategy_name')}")

# BUY/SELL集計
print("\n[確認3] 銘柄切替のBUY/SELL内訳:")
switch_buys = [d for _, d in switch_details if d.get('action') == 'BUY']
switch_sells = [d for _, d in switch_details if d.get('action') == 'SELL']

print(f"  BUY: {len(switch_buys)}件")
print(f"  SELL: {len(switch_sells)}件")

# 同一銘柄の切替回数
from collections import Counter
switch_symbols = [d.get('symbol') for _, d in switch_details]
symbol_counts = Counter(switch_symbols)

print(f"\n[確認4] 銘柄別の切替回数:")
for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {symbol}: {count}回")

print("\n" + "=" * 80)
print("結論:")
print("- 銘柄切替の実態を確認しました")
print("=" * 80)
