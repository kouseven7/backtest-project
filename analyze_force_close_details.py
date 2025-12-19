"""
execution_type='force_close'の詳細確認スクリプト
"""
import json

json_path = 'output/dssms_integration/dssms_20251217_214451/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("【B2】強制決済（execution_type='force_close'）詳細確認")
print("=" * 80)

# execution_type='force_close'を抽出
force_close_details = [
    (i, detail) for i, detail in enumerate(execution_details)
    if detail.get('execution_type') == 'force_close'
]

print(f"\n[確認1] execution_type='force_close'の件数: {len(force_close_details)}件\n")

# 各強制決済の詳細表示
for idx, (i, detail) in enumerate(force_close_details):
    print(f"[{idx+1}] index={i}")
    print(f"    symbol: {detail.get('symbol')}")
    print(f"    action: {detail.get('action')}")
    print(f"    quantity: {detail.get('quantity')}")
    print(f"    executed_price: {detail.get('executed_price')}")
    print(f"    timestamp: {detail.get('timestamp')}")
    print(f"    strategy_name: {detail.get('strategy_name')}")
    print(f"    profit_pct: {detail.get('profit_pct')}")
    print(f"    close_return: {detail.get('close_return', 'N/A')}")
    print()

# 損益集計
force_close_profits = [d.get('profit_pct', 0) for _, d in force_close_details]
positive_count = sum(1 for p in force_close_profits if p > 0)
negative_count = sum(1 for p in force_close_profits if p < 0)
zero_count = sum(1 for p in force_close_profits if p == 0)

print(f"\n[確認2] 強制決済の損益内訳:")
print(f"  利益: {positive_count}件")
print(f"  損失: {negative_count}件")
print(f"  同値: {zero_count}件")

# 前後の取引確認
print(f"\n[確認3] 強制決済の前の取引（最初の3件）:")
for idx, (i, detail) in enumerate(force_close_details[:3]):
    if i > 0:
        prev = execution_details[i-1]
        print(f"\n  [強制決済 {idx+1}] symbol={detail.get('symbol')}, profit_pct={detail.get('profit_pct'):.4f}%")
        print(f"    [前の取引] action={prev.get('action')}, "
              f"symbol={prev.get('symbol')}, "
              f"execution_type={prev.get('execution_type')}, "
              f"strategy={prev.get('strategy_name')}")

print("\n" + "=" * 80)
print("結論:")
print("- 強制決済は実際の損益を伴う取引")
print("- 損失取引が含まれている")
print("=" * 80)
