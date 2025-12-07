"""
インポートパス問題調査用スクリプト
直接実行 vs モジュール実行時のsys.pathの違いを確認
"""
import sys
import os

print("=" * 80)
print("sys.path確認 (上位5件)")
print("=" * 80)
for i, path in enumerate(sys.path[:5]):
    print(f"{i}: {path}")

print("\n" + "=" * 80)
print("__file__とカレントディレクトリ")
print("=" * 80)
print(f"__file__: {__file__}")
print(f"Current Dir: {os.getcwd()}")
print(f"Script Dir: {os.path.dirname(os.path.abspath(__file__))}")

print("\n" + "=" * 80)
print("dssms_backtester_v3.py の場所")
print("=" * 80)
dss_path = os.path.join(os.path.dirname(__file__), "src", "dssms", "dssms_backtester_v3.py")
print(f"Expected Path: {dss_path}")
print(f"Exists: {os.path.exists(dss_path)}")

print("\n" + "=" * 80)
print("インポートテスト")
print("=" * 80)

# パターン1: 絶対パスなし (失敗予想 in モジュール実行)
try:
    import dssms_backtester_v3
    print("[OK] import dssms_backtester_v3: SUCCESS")
except ImportError as e:
    print(f"[ERROR] import dssms_backtester_v3: FAILED - {e}")

# パターン2: from src.dssms
try:
    from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
    print("[OK] from src.dssms.dssms_backtester_v3: SUCCESS")
except ImportError as e:
    print(f"[ERROR] from src.dssms.dssms_backtester_v3: FAILED - {e}")

print("\n" + "=" * 80)
print("調査完了")
print("=" * 80)
