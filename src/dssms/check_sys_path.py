"""
モジュール実行時のsys.path確認用 (src/dssms配下)
"""
import sys
print("=" * 80)
print("モジュール実行時のsys.path (上位5件)")
print("=" * 80)
for i, path in enumerate(sys.path[:5]):
    print(f"{i}: {path}")
