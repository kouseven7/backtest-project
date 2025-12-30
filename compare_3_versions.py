import subprocess
import re

# 3つの時点のコードを取得
commits = {
    "重複問題対応前": "ec60722~1",
    "重複問題対応後": "ec60722",
    "現在": "HEAD"
}

for label, commit in commits.items():
    print(f"\n{'='*60}")
    print(f"{label} ({commit})")
    print(f"{'='*60}")
    
    # git showでコードを取得
    cmd = f'git show {commit}:strategies/base_strategy.py'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        continue
    
    lines = result.stdout.split('\n')
    
    # Line 260-290付近を抽出（エントリーロジック部分）
    start = 260
    end = 295
    
    print(f"\nLine {start}-{end}付近（エントリーロジック）:")
    print("-" * 60)
    for i in range(start-1, min(end, len(lines))):
        print(f"{i+1:4d}: {lines[i]}")
