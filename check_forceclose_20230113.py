import re

# task11_backtest.logから2023-01-13のForceClose関連ログを抽出
with open('task11_backtest.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print("=== 2023-01-13 ForceClose関連ログ ===")
for i, line in enumerate(lines):
    if '2023-01-13' in line and 'FORCE_CLOSE' in line and '8306' in line:
        print(f"Line {i+1}: {line.strip()}")
        if i+1 < len(lines) and i-1 >= 0:
            print(f"  Before: {lines[i-1].strip()}")
            print(f"  After: {lines[i+1].strip()}")
        print()
