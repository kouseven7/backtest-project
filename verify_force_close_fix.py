"""
ForceClose除外修正の検証スクリプト

修正内容:
- execution_detail_utils.py Line 160
- 変更前: if execution_type != 'trade':
- 変更後: if execution_type not in ['trade', 'force_close']:

期待される結果:
1. BUY/SELL件数: BUY=48, SELL=48（以前はSELL=31）
2. dssms_trades.csv: 48件の取引記録（以前は31件）
3. ForceClose SELL 18件がCSVに含まれる
4. 勝率が正確になる（100%から実際の値へ）
"""
import json
import csv
from pathlib import Path

print("=" * 80)
print("ForceClose除外修正の検証")
print("=" * 80)

# 最新のバックテスト結果を取得
output_dir = Path('output/dssms_integration')
latest_dir = max(output_dir.glob('dssms_*'), key=lambda p: p.stat().st_mtime)

print(f"\n[検証対象] {latest_dir.name}")
print("-" * 80)

# 1. execution_results.jsonの確認
json_path = latest_dir / 'dssms_execution_results.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

# execution_type別の集計
buy_count = {'trade': 0, 'force_close': 0, 'switch': 0, 'total': 0}
sell_count = {'trade': 0, 'force_close': 0, 'switch': 0, 'total': 0}

for detail in execution_details:
    action = detail.get('action')
    exec_type = detail.get('execution_type', 'trade')
    
    if action == 'BUY':
        buy_count[exec_type] = buy_count.get(exec_type, 0) + 1
        buy_count['total'] += 1
    elif action == 'SELL':
        sell_count[exec_type] = sell_count.get(exec_type, 0) + 1
        sell_count['total'] += 1

print("\n[1] execution_details全体のBUY/SELL集計:")
print(f"  BUY総数: {buy_count['total']}")
print(f"    - trade: {buy_count.get('trade', 0)}")
print(f"    - force_close: {buy_count.get('force_close', 0)}")
print(f"    - switch: {buy_count.get('switch', 0)}")
print(f"  SELL総数: {sell_count['total']}")
print(f"    - trade: {sell_count.get('trade', 0)}")
print(f"    - force_close: {sell_count.get('force_close', 0)}")
print(f"    - switch: {sell_count.get('switch', 0)}")

# 2. is_valid_trade()の挙動シミュレーション
valid_buy = 0
valid_sell = 0

for detail in execution_details:
    success = detail.get('success', False)
    action = detail.get('action', '').upper()
    exec_type = detail.get('execution_type', 'trade')
    
    # is_valid_trade()のロジック
    if success and action in ['BUY', 'SELL'] and exec_type in ['trade', 'force_close']:
        if action == 'BUY':
            valid_buy += 1
        elif action == 'SELL':
            valid_sell += 1

print(f"\n[2] is_valid_trade()通過後のBUY/SELL:")
print(f"  BUY: {valid_buy}")
print(f"  SELL: {valid_sell}")
print(f"  差分: {abs(valid_buy - valid_sell)}")

if valid_buy == valid_sell:
    print("  結果: OK（完全一致）")
else:
    print(f"  結果: NG（{abs(valid_buy - valid_sell)}件の差分あり）")

# 3. dssms_trades.csvの件数確認
csv_path = latest_dir / 'dssms_trades.csv'
if csv_path.exists():
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        trades = list(reader)
    
    print(f"\n[3] dssms_trades.csv:")
    print(f"  取引件数: {len(trades)}")
    print(f"  期待値: {valid_sell}件")
    
    if len(trades) == valid_sell:
        print("  結果: OK")
    else:
        print(f"  結果: NG（{abs(len(trades) - valid_sell)}件の差分）")
    
    # ForceClose取引の確認
    force_close_trades = [t for t in trades if 'ForceClose' in t.get('strategy', '')]
    print(f"\n[4] ForceClose取引:")
    print(f"  件数: {len(force_close_trades)}")
    print(f"  期待値: {sell_count.get('force_close', 0)}件")
    
    if len(force_close_trades) > 0:
        print("  結果: OK（ForceCloseがCSVに含まれている）")
        # 最初の3件表示
        print("\n  サンプル（最初の3件）:")
        for i, trade in enumerate(force_close_trades[:3]):
            print(f"    [{i+1}] entry_date={trade.get('entry_date')}, "
                  f"exit_date={trade.get('exit_date')}, "
                  f"pnl={trade.get('pnl')}, "
                  f"strategy={trade.get('strategy')}")
    else:
        print("  結果: NG（ForceCloseがCSVに含まれていない）")
else:
    print(f"\n[3] dssms_trades.csv: ファイルが存在しません")

# 5. 総合判定
print("\n" + "=" * 80)
print("総合判定:")
print("=" * 80)

success_checks = 0
total_checks = 4

if valid_buy == valid_sell:
    print("[OK] BUY/SELL完全一致")
    success_checks += 1
else:
    print(f"[NG] BUY/SELL不一致: BUY={valid_buy}, SELL={valid_sell}")

if csv_path.exists() and len(trades) == valid_sell:
    print("[OK] CSV取引件数一致")
    success_checks += 1
else:
    print("[NG] CSV取引件数不一致")

if sell_count.get('force_close', 0) > 0:
    print(f"[OK] ForceClose SELL検出: {sell_count.get('force_close', 0)}件")
    success_checks += 1
else:
    print("[NG] ForceClose SELL未検出")

if len(force_close_trades) > 0:
    print(f"[OK] ForceCloseがCSVに含まれる: {len(force_close_trades)}件")
    success_checks += 1
else:
    print("[NG] ForceCloseがCSVに含まれない")

print(f"\n成功率: {success_checks}/{total_checks} ({success_checks/total_checks*100:.1f}%)")

if success_checks == total_checks:
    print("\n結論: 修正成功 - ForceClose除外問題が解決されました")
else:
    print(f"\n結論: 修正未完了 - {total_checks - success_checks}個の問題が残っています")
