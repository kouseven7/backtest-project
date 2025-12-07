"""
修正後のインポート動作確認テスト
両方の実行方法でDSS Core V3がインポートできることを確認
"""
import sys
import os

print("=" * 80)
print("修正後インポートテスト")
print("=" * 80)

# テスト1: 絶対インポート (プロジェクトルートからのパス)
try:
    from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
    print("[OK] from src.dssms.dssms_backtester_v3 import DSSBacktesterV3: SUCCESS")
    print(f"  DSSBacktesterV3: {DSSBacktesterV3}")
except ImportError as e:
    print(f"[ERROR] from src.dssms.dssms_backtester_v3: FAILED - {e}")

# テスト2: dssms_integrated_main.py のインポートチェック
try:
    # Line 73の利用可能性チェックをシミュレート
    from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
    dss_available = True
    print(f"[OK] dss_available: {dss_available}")
except ImportError:
    dss_available = False
    print(f"[ERROR] dss_available: {dss_available}")

print("\n" + "=" * 80)
print("テスト完了")
print("=" * 80)
