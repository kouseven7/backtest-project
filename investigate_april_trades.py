import json
import pandas as pd

# 9101.T_execution_results.jsonから取引詳細を抽出
with open('output/comprehensive_reports/9101.T_20251120_115359/9101.T_execution_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data.get('execution_details', [])

print("=" * 80)
print("タスク1: execution_detailsの価格確認")
print("=" * 80)

# 4/23と4/30の取引を抽出
april_trades = []
for detail in execution_details:
    timestamp = detail.get('timestamp', '')
    if '2024-04-23' in timestamp or '2024-04-30' in timestamp:
        april_trades.append({
            'timestamp': timestamp,
            'action': detail.get('action'),
            'quantity': detail.get('quantity'),
            'executed_price': detail.get('executed_price'),
            'strategy': detail.get('strategy_name')
        })

print("\n4月の取引詳細（execution_details）:")
for trade in april_trades:
    print(f"  {trade['timestamp']}: {trade['action']} {trade['quantity']}株 @ {trade['executed_price']:.2f}円 (戦略: {trade['strategy']})")

# 計算検証
if len(april_trades) >= 2:
    buy_trade = april_trades[0]
    sell_trade = april_trades[1]
    
    print("\n計算検証（execution_details価格）:")
    print(f"  4/23 BUY: {buy_trade['quantity']}株 × {buy_trade['executed_price']:.2f}円 = {buy_trade['quantity'] * buy_trade['executed_price']:,.2f}円")
    print(f"  初期現金: 1,000,000円")
    print(f"  BUY後現金残高: 1,000,000 - {buy_trade['quantity'] * buy_trade['executed_price']:,.2f} = {1000000 - buy_trade['quantity'] * buy_trade['executed_price']:,.2f}円")
    print()
    print(f"  4/30 SELL: {sell_trade['quantity']}株 × {sell_trade['executed_price']:.2f}円 = {sell_trade['quantity'] * sell_trade['executed_price']:,.2f}円")
    print(f"  SELL後現金残高: {1000000 - buy_trade['quantity'] * buy_trade['executed_price']:,.2f} + {sell_trade['quantity'] * sell_trade['executed_price']:,.2f} = {1000000 - buy_trade['quantity'] * buy_trade['executed_price'] + sell_trade['quantity'] * sell_trade['executed_price']:,.2f}円")
    print(f"  SELL後ポジション: 0株")
    print(f"  SELL後ポートフォリオ総額: {1000000 - buy_trade['quantity'] * buy_trade['executed_price'] + sell_trade['quantity'] * sell_trade['executed_price']:,.2f}円")
