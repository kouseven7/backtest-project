import pandas as pd
import json

# execution_results.jsonから取引詳細を確認
with open('output/comprehensive_reports/9101.T_20251120_114716/execution_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_results'][0]['execution_details']

print("=== 4月の全取引 ===")
for detail in execution_details:
    timestamp = detail.get('timestamp', '')
    if '2024-04' in timestamp or '2024-05' in timestamp[:7]:
        print(f"{timestamp}: {detail.get('action')} {detail.get('quantity')} @ {detail.get('price'):.2f}")

print("\n=== 4/23と4/30の取引詳細 ===")
for detail in execution_details:
    timestamp = detail.get('timestamp', '')
    if '2024-04-23' in timestamp or '2024-04-30' in timestamp:
        print(json.dumps(detail, indent=2, ensure_ascii=False))
