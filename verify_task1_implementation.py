"""
Task 1実装検証スクリプト - SymbolSwitchManager完全版への切替確認

目的:
- 完全版がロードされているか確認
- 設定値が正しく反映されているか確認
- 初期化時間を測定

Author: Backtest Project Team
Created: 2026-01-13
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Task 1実装検証: SymbolSwitchManager完全版への切替")
print("="*80)

# 初期化時間測定
print("\n[1] 初期化時間測定...")
start = time.time()

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

bt = DSSMSIntegratedBacktester()
init_time = time.time() - start

print(f"初期化時間: {init_time:.3f}秒")

# SymbolSwitchManagerの型確認
print(f"\n[2] SymbolSwitchManager型確認...")
print(f"型: {type(bt.switch_manager).__name__}")
print(f"モジュール: {type(bt.switch_manager).__module__}")

# 設定値確認
print(f"\n[3] 設定値確認...")
switch_config = bt.config.get('symbol_switch', {})
print(f"min_holding_days: {switch_config.get('min_holding_days', 'N/A')}")
print(f"max_switches_per_month: {switch_config.get('max_switches_per_month', 'N/A')}")
print(f"switch_cost_rate: {switch_config.get('switch_cost_rate', 'N/A')}")

# メソッド存在確認
print(f"\n[4] 完全版メソッド存在確認...")
methods_to_check = [
    '_check_min_holding_period',
    '_count_recent_switches',
    '_check_monthly_switch_limit',
    '_evaluate_switch_cost_efficiency'
]

for method_name in methods_to_check:
    exists = hasattr(bt.switch_manager, method_name)
    status = "✅" if exists else "❌"
    print(f"{status} {method_name}: {'存在' if exists else '存在しない'}")

# 結果判定
print(f"\n{'='*80}")
print("検証結果サマリー")
print(f"{'='*80}")

is_complete_version = type(bt.switch_manager).__name__ == 'SymbolSwitchManager'
config_correct = (
    switch_config.get('min_holding_days') == 10 and
    switch_config.get('max_switches_per_month') == 5
)

print(f"完全版ロード: {'✅ 成功' if is_complete_version else '❌ 失敗'}")
print(f"設定値最適化: {'✅ 成功' if config_correct else '❌ 失敗'}")
print(f"初期化時間: {init_time:.3f}秒 ({'✅ 問題なし' if init_time < 2.0 else '❌ 遅い'})")

if is_complete_version and config_correct:
    print("\n✅ Task 1実装完了: 完全版への切替成功")
else:
    print("\n❌ Task 1実装に問題あり")
    if not is_complete_version:
        print("  - 完全版がロードされていません")
    if not config_correct:
        print("  - 設定値が正しく反映されていません")

print(f"{'='*80}\n")
