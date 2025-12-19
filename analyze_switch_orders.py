"""
switch取引の実態調査

目的:
1. switchのBUY/SELLが実際に注文を出しているのか?
2. DSSMSが直接シグナルを出しているのか?
3. 記録用なのか、実際の取引なのか?
"""
import json

json_path = 'output/dssms_integration/dssms_20251219_121821/dssms_execution_results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("switch取引の実態調査")
print("=" * 80)

# switch取引を抽出
switches = [d for d in execution_details if d.get('execution_type') == 'switch']

print(f"\n[1] switch取引: {len(switches)}件\n")

for i, detail in enumerate(switches):
    print(f"[{i+1}] symbol: {detail.get('symbol')}")
    print(f"    action: {detail.get('action')}")
    print(f"    strategy_name: {detail.get('strategy_name')}")
    print(f"    timestamp: {detail.get('timestamp')}")
    print(f"    executed_price: {detail.get('executed_price')}")
    print(f"    quantity: {detail.get('quantity')}")
    print(f"    success: {detail.get('success')}")
    print(f"    status: {detail.get('status')}")
    print(f"    profit_pct: {detail.get('profit_pct')}")
    print(f"    close_return: {detail.get('close_return')}")
    
    # order情報があるか確認
    if 'order' in detail:
        print(f"    order: {detail.get('order')[:100]}...")  # 最初の100文字
    else:
        print(f"    order: なし")
    
    print()

# switchのペアを確認
print("=" * 80)
print("[2] switchのペア分析")
print("=" * 80)

switch_pairs = {}
for detail in switches:
    symbol = detail.get('symbol')
    action = detail.get('action')
    
    if symbol not in switch_pairs:
        switch_pairs[symbol] = {'BUY': [], 'SELL': []}
    
    switch_pairs[symbol][action].append({
        'timestamp': detail.get('timestamp'),
        'price': detail.get('executed_price'),
        'quantity': detail.get('quantity')
    })

print("\n銘柄ごとのswitch BUY/SELLペア:")
for symbol, actions in switch_pairs.items():
    buy_count = len(actions['BUY'])
    sell_count = len(actions['SELL'])
    print(f"\n銘柄 {symbol}:")
    print(f"  BUY: {buy_count}件")
    for buy in actions['BUY']:
        print(f"    - {buy['timestamp']}: {buy['price']}円 x {buy['quantity']:.2f}株")
    print(f"  SELL: {sell_count}件")
    for sell in actions['SELL']:
        print(f"    - {sell['timestamp']}: {sell['price']}円 x {sell['quantity']:.2f}株")
    
    if buy_count == sell_count:
        print(f"  状態: ペア成立")
    else:
        print(f"  状態: 不一致（BUY={buy_count}, SELL={sell_count}）")

# 実際の損益が発生しているか確認
print("\n" + "=" * 80)
print("[3] switchの損益確認")
print("=" * 80)

total_switch_pnl = 0
for detail in switches:
    close_return = detail.get('close_return')
    if close_return is not None:
        total_switch_pnl += close_return
        print(f"銘柄 {detail.get('symbol')} {detail.get('action')}: {close_return:,.2f}円")

print(f"\nswitch取引の総損益: {total_switch_pnl:,.2f}円")

if abs(total_switch_pnl) > 0.01:
    print("結論: switchは実際の損益を伴う取引（実注文）")
else:
    print("結論: switchは記録用（実注文なし）")
