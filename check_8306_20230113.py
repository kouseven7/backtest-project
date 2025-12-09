import json

# Task 11実装前
data_before = json.load(open('output/dssms_integration/dssms_20251208_124418/dssms_execution_results.json', encoding='utf-8'))
details_before = [d for d in data_before['execution_details'] if d.get('symbol')=='8306' and '2023-01-13' in d.get('timestamp', '')]

print(f"=== Task 11実装前 (2023-01-13の8306) ===")
print(f"Count={len(details_before)}")
for d in details_before:
    print(f"Action={d.get('action')}, Strategy={d.get('strategy_name')}, Price={d.get('executed_price', 'N/A')}, Quantity={d.get('quantity', 'N/A')}")

# Task 11実装後
data_after = json.load(open('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', encoding='utf-8'))
details_after = [d for d in data_after['execution_details'] if d.get('symbol')=='8306' and '2023-01-13' in d.get('timestamp', '')]

print(f"\n=== Task 11実装後 (2023-01-13の8306) ===")
print(f"Count={len(details_after)}")
for d in details_after:
    print(f"Action={d.get('action')}, Strategy={d.get('strategy_name')}, Price={d.get('executed_price', 'N/A')}, Quantity={d.get('quantity', 'N/A')}")
