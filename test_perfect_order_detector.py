#!/usr/bin/env python3
"""
PerfectOrderDetector テストスクリプト
"""

import sys
sys.path.append('src')

from dssms.perfect_order_detector import PerfectOrderDetector

# 基本インポートテスト
print("=== PerfectOrderDetector テスト ===")
print("✓ インポート成功")

# インスタンス化テスト
detector = PerfectOrderDetector()
print("✓ インスタンス化成功")

# 設定値確認
print(f"設定値:")
for timeframe, params in detector.timeframes.items():
    print(f"  {timeframe}: {params}")

# 利用可能メソッド確認
public_methods = [m for m in dir(detector) if not m.startswith("_")]
print(f"利用可能メソッド: {public_methods}")

print("✓ PerfectOrderDetector 基本テスト完了")