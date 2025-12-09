"""
BUY/SELL件数確認スクリプト（修正後）
"""
import json
from pathlib import Path
from datetime import datetime

# 最新の出力ディレクトリを取得
output_base = Path("output/dssms_integration")
latest_dir = max(output_base.glob("dssms_*"), key=lambda p: p.stat().st_mtime)
print(f"最新ディレクトリ: {latest_dir.name}")
print(f"作成日時: {datetime.fromtimestamp(latest_dir.stat().st_mtime)}")
print()

# execution_results.json読み込み
results_file = latest_dir / "dssms_execution_results.json"
if not results_file.exists():
    print(f"エラー: {results_file} が見つかりません")
    exit(1)

with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# execution_details分析
details = data.get('execution_details', [])
print(f"execution_details総数: {len(details)}件")

# BUY/SELL集計
buy_orders = [x for x in details if x.get('action') == 'BUY']
sell_orders = [x for x in details if x.get('action') == 'SELL']
print(f"BUY: {len(buy_orders)}件")
print(f"SELL: {len(sell_orders)}件")
print(f"差分: {abs(len(buy_orders) - len(sell_orders))}件")
print()

# DSSMS_SymbolSwitch集計
dssms_buy = [x for x in buy_orders if x.get('strategy_name') == 'DSSMS_SymbolSwitch']
dssms_sell = [x for x in sell_orders if x.get('strategy_name') == 'DSSMS_SymbolSwitch']
print(f"DSSMS_SymbolSwitch:")
print(f"  BUY: {len(dssms_buy)}件")
print(f"  SELL: {len(dssms_sell)}件")
print(f"  差分: {abs(len(dssms_buy) - len(dssms_sell))}件")
print()

# 戦略別集計
strategies = {}
for detail in details:
    strategy = detail.get('strategy_name', 'Unknown')
    action = detail.get('action', 'Unknown')
    key = f"{strategy}_{action}"
    strategies[key] = strategies.get(key, 0) + 1

print("戦略別集計:")
for key, count in sorted(strategies.items()):
    print(f"  {key}: {count}件")
print()

# 期待値との比較
print("期待値との比較:")
print(f"  修正前DSSMS_SymbolSwitch BUY: 0件")
print(f"  修正後DSSMS_SymbolSwitch BUY: {len(dssms_buy)}件 ← 修正効果")
print(f"  修正前BUY/SELL差分: 97件（SELL超過）")
print(f"  修正後BUY/SELL差分: {abs(len(buy_orders) - len(sell_orders))}件")

# DSSMS_SymbolSwitchサンプル表示
if len(dssms_buy) > 0:
    print()
    print("DSSMS_SymbolSwitch BUYサンプル（最初の3件）:")
    for i, detail in enumerate(dssms_buy[:3], 1):
        print(f"  [{i}] symbol={detail.get('symbol')}, timestamp={detail.get('timestamp')}, "
              f"quantity={detail.get('quantity')}, executed_price={detail.get('executed_price')}")
