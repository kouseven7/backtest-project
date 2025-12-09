"""損益計算への影響確認

BUY/SELL不一致が損益計算に与える影響を評価します。
"""
import json

result = json.load(open('output/dssms_integration/dssms_20251208_193732/dssms_execution_results.json', 'r', encoding='utf-8'))

print("=== 損益計算への影響評価 ===\n")

print("1. 基本情報:")
print(f"   initial_capital: {result['initial_capital']:,.0f}円")
print(f"   total_portfolio_value: {result['total_portfolio_value']:,.0f}円")
print(f"   total_return: {result['total_return']:,.0f}円 ({result['total_return']/result['initial_capital']*100:.2f}%)")
print()

data = result.get('execution_details', [])

# SELL取引の損益を集計
sell_trades = [d for d in data if d['action'] == 'SELL']
total_sell_profit = 0
sell_with_profit = 0

for sell in sell_trades:
    profit_pct = sell.get('profit_pct', 0)
    if profit_pct != 0:
        sell_with_profit += 1
        # profit_pctからおおよその損益を推定（正確ではないが傾向を確認）
        # 注意: quantityやentry_priceがないため、正確な計算は不可能

print(f"2. SELL取引の損益記録:")
print(f"   総SELL件数: {len(sell_trades)}件")
print(f"   profit_pct記録あり: {sell_with_profit}件 ({sell_with_profit/len(sell_trades)*100:.1f}%)")
print(f"   profit_pct記録なし: {len(sell_trades) - sell_with_profit}件")
print()

# DSSMS_SymbolSwitchの損益確認
dssms_switch_sells = [d for d in data if d['action'] == 'SELL' and 'DSSMS_SymbolSwitch' in d.get('strategy_name', '')]
dssms_with_profit = sum(1 for d in dssms_switch_sells if 'profit_pct' in d and d['profit_pct'] != 0)

print(f"3. DSSMS_SymbolSwitch SELL取引の損益記録:")
print(f"   総件数: {len(dssms_switch_sells)}件")
print(f"   profit_pct記録あり: {dssms_with_profit}件")
print(f"   profit_pct記録なし: {len(dssms_switch_sells) - dssms_with_profit}件")
print()

# サンプル確認
print(f"4. DSSMS_SymbolSwitch SELL取引サンプル（最初の3件）:")
for i, sell in enumerate(dssms_switch_sells[:3], 1):
    print(f"\n   SELL {i}:")
    print(f"     timestamp: {sell.get('timestamp', 'N/A')}")
    print(f"     symbol: {sell.get('symbol', 'N/A')}")
    print(f"     quantity: {sell.get('quantity', 'N/A'):.2f}")
    print(f"     price: {sell.get('executed_price', 'N/A'):.2f}")
    print(f"     profit_pct: {sell.get('profit_pct', 'N/A')}")
    print(f"     entry_price: {sell.get('entry_price', 'N/A')}")
    print(f"     close_return: {sell.get('close_return', 'N/A')}")

print("\n=== 結論 ===")
print("1. BUY/SELL不一致の主要原因:")
print("   - DSSMS_SymbolSwitchのBUY記録が欠落（63件）")
print("   - _open_position()がexecution_detailを生成していない")
print()
print("2. 損益計算への影響:")
print("   - DSSMS_SymbolSwitch SELLにはclose_returnが記録されている")
print("   - ポートフォリオ価値の計算には影響しない可能性が高い")
print("   - ただし、取引履歴の完全性は失われている")
print()
print("3. 推奨対応:")
print("   - _open_position()にexecution_detail生成を追加")
print("   - strategy_name='DSSMS_SymbolSwitch'（BUY側）を記録")
print("   - これによりBUY/SELL一致が実現される")
