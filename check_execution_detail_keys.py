import json

# 9101.T_execution_results.jsonからexecution_detailsを確認
with open('output/comprehensive_reports/9101.T_20251120_115359/9101.T_execution_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data.get('execution_details', [])

print("=" * 80)
print("タスク3: execution_detailsのキー名確認")
print("=" * 80)

# 最初の取引（4/23 BUY）のキーを確認
if execution_details:
    first_trade = execution_details[0]
    print("\n最初の取引（4/23 BUY）のキー:")
    for key in first_trade.keys():
        print(f"  {key}: {first_trade[key]}")
    
    print("\n\n重要なキー:")
    print(f"  'price'キーの存在: {'price' in first_trade}")
    print(f"  'executed_price'キーの存在: {'executed_price' in first_trade}")
    
    if 'price' in first_trade:
        print(f"  'price'の値: {first_trade['price']}")
    if 'executed_price' in first_trade:
        print(f"  'executed_price'の値: {first_trade['executed_price']}")
