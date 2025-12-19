"""
ForceClose実データの詳細確認
"""
import json

json_path = 'output/dssms_integration/dssms_20251219_121821/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("ForceClose実データの詳細確認")
print("=" * 80)

# ForceClose取引を抽出
force_closes = [d for d in execution_details if 'ForceClose' in d.get('strategy_name', '')]

print(f"\n[1] ForceClose取引: {len(force_closes)}件\n")

for i, detail in enumerate(force_closes):
    print(f"[{i+1}] strategy_name: {detail.get('strategy_name')}")
    print(f"    action: {detail.get('action')}")
    print(f"    success: {detail.get('success')}")
    print(f"    status: {detail.get('status')}")
    print(f"    execution_type: {detail.get('execution_type')}")
    print(f"    symbol: {detail.get('symbol')}")
    print(f"    timestamp: {detail.get('timestamp')}")
    
    # is_valid_trade()のロジック
    success = detail.get('success', False)
    action = detail.get('action', '').upper()
    exec_type = detail.get('execution_type', 'trade')
    
    will_pass = success and action in ['BUY', 'SELL'] and exec_type in ['trade', 'force_close']
    
    print(f"    is_valid_trade()判定: {'通過' if will_pass else '除外'}")
    print()

# is_valid_trade()通過シミュレーション
valid_buy = 0
valid_sell = 0

for detail in execution_details:
    success = detail.get('success', False)
    action = detail.get('action', '').upper()
    exec_type = detail.get('execution_type', 'trade')
    
    if success and action in ['BUY', 'SELL'] and exec_type in ['trade', 'force_close']:
        if action == 'BUY':
            valid_buy += 1
        elif action == 'SELL':
            valid_sell += 1

print("=" * 80)
print("[2] is_valid_trade()通過後の集計:")
print(f"  BUY: {valid_buy}")
print(f"  SELL: {valid_sell}")
print(f"  差分: {abs(valid_buy - valid_sell)}")
print("=" * 80)
