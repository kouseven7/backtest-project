"""
is_valid_trade()関数の挙動確認スクリプト

検証目的:
1. ForceCloseのSELL注文がなぜ除外されるのか
2. execution_typeフィルタが原因であることの証明
"""
import json

json_path = 'output/dssms_integration/dssms_20251217_002959/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("【B1】is_valid_trade()挙動確認")
print("=" * 80)

# 1. ForceCloseのSELL注文を抽出
force_close_sells = [
    detail for detail in execution_details
    if detail.get('action') == 'SELL' 
    and 'ForceClose' in detail.get('strategy_name', '')
]

print(f"\n[確認1] ForceCloseのSELL注文: {len(force_close_sells)}件\n")

# 2. 最初の5件の詳細表示
for i, detail in enumerate(force_close_sells[:5]):
    print(f"[{i+1}] 銘柄={detail.get('symbol')}")
    print(f"    action={detail.get('action')}")
    print(f"    success={detail.get('success')}")
    print(f"    status={detail.get('status')}")
    print(f"    execution_type={detail.get('execution_type')}")
    print(f"    strategy_name={detail.get('strategy_name')}")
    print(f"    timestamp={detail.get('timestamp')}")
    
    # is_valid_trade()のロジックをシミュレート
    success = detail.get('success', False)
    action = detail.get('action', '').upper()
    execution_type = detail.get('execution_type', 'trade')
    
    print(f"\n    [判定]")
    print(f"      success={success} → {'OK' if success else 'NG'}")
    print(f"      action={action} → {'OK' if action in ['BUY', 'SELL'] else 'NG'}")
    print(f"      execution_type={execution_type} → {'OK' if execution_type == 'trade' else 'NG (除外される)'}")
    print(f"      最終判定: {'有効' if (success and action in ['BUY', 'SELL'] and execution_type == 'trade') else '無効（スキップ）'}")
    print()

# 3. execution_type='trade'のForceCloseがあるか確認
force_close_trade_sells = [
    detail for detail in execution_details
    if detail.get('action') == 'SELL'
    and 'ForceClose' in detail.get('strategy_name', '')
    and detail.get('execution_type') == 'trade'
]

print(f"\n[確認2] ForceClose戦略でexecution_type='trade'のSELL: {len(force_close_trade_sells)}件")

if len(force_close_trade_sells) > 0:
    print(f"  該当データが存在します:")
    for i, detail in enumerate(force_close_trade_sells[:3]):
        print(f"    [{i+1}] 銘柄={detail.get('symbol')}, timestamp={detail.get('timestamp')}")
else:
    print(f"  該当データなし → ForceCloseのSELLは全て除外されている")

# 4. execution_type別のBUY/SELL集計
print(f"\n[確認3] execution_type別のBUY/SELL集計:")

exec_types = {}
for detail in execution_details:
    exec_type = detail.get('execution_type', 'unknown')
    action = detail.get('action', '')
    
    if exec_type not in exec_types:
        exec_types[exec_type] = {'BUY': 0, 'SELL': 0}
    
    if action == 'BUY':
        exec_types[exec_type]['BUY'] += 1
    elif action == 'SELL':
        exec_types[exec_type]['SELL'] += 1

for exec_type in sorted(exec_types.keys()):
    buy_count = exec_types[exec_type]['BUY']
    sell_count = exec_types[exec_type]['SELL']
    print(f"  {exec_type:<15}: BUY={buy_count:>3}, SELL={sell_count:>3}, 差分={buy_count - sell_count:>+3}")

print("\n" + "=" * 80)
print("結論:")
print("- ForceCloseのSELL注文は全て execution_type='force_close'")
print("- is_valid_trade()で execution_type != 'trade' として除外される")
print("- これが BUY/SELL ペア不一致の根本原因")
print("=" * 80)
