"""
インデント修正検証スクリプト

修正後のコードが正しく動作するかを検証するため、
_convert_execution_details_to_trades()関数を単体でテストします。

copilot-instructions.md準拠:
- 実データ検証
- 推測ではなく正確な数値を報告
"""

import sys
import json
from pathlib import Path

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_system.reporting.comprehensive_reporter import ComprehensiveReporter

# JSONファイル読み込み
json_path = Path("output/dssms_integration/dssms_20251214_213349/dssms_execution_results.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

execution_details = data['execution_details']

print("=" * 80)
print("インデント修正後の検証")
print("=" * 80)

# ComprehensiveReporter初期化
reporter = ComprehensiveReporter()

# _convert_execution_details_to_trades()を実行
print(f"\n実行前: execution_details件数 = {len(execution_details)}")

trades, open_positions = reporter._convert_execution_details_to_trades(execution_details)

print(f"\n実行後:")
print(f"  - 取引レコード数: {len(trades)}")
print(f"  - 保有中ポジション数: {len(open_positions)}")

print("\n" + "=" * 80)
print("検証結果")
print("=" * 80)

if len(trades) == 9:
    print("✓ 成功: 取引レコード数が9件（期待値）")
else:
    print(f"✗ 失敗: 取引レコード数が{len(trades)}件（期待値: 9件）")

print("\n取引レコード詳細:")
for i, trade in enumerate(trades, 1):
    print(f"  {i}. {trade['strategy']:25s} | Entry: {trade['entry_date']} | PnL: {trade['pnl']:,.2f}")

print("\n銘柄別集計:")
from collections import defaultdict
strategy_counts = defaultdict(int)
for trade in trades:
    strategy_counts[trade['strategy']] += 1

for strategy, count in sorted(strategy_counts.items()):
    print(f"  - {strategy}: {count}件")
