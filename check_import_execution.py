"""
SymbolSwitchManager import実行パス分析
main()が実行されているか確認
"""

import time

print("=== Import実行パス分析 ===")
start_time = time.time()

print("1. sys.pathチェック...")
import sys
print(f"   __name__ = {__name__}")

print("2. SymbolSwitchManagerインポート開始...")
try:
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    import_time = time.time() - start_time
    print(f"[OK] インポート成功: {import_time*1000:.1f}ms")
except Exception as e:
    import_time = time.time() - start_time
    print(f"[ERROR] インポートエラー: {e}")
    print(f"   時間: {import_time*1000:.1f}ms")

print("3. インポート分析完了")