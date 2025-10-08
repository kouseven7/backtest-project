"""
SymbolSwitchManager段階的軽量化テスト
メソッドを段階的に削除してボトルネックを特定
"""

import time
import os
import shutil

def create_test_version(method_count=None):
    """テスト版作成"""
    original_file = r"src\dssms\symbol_switch_manager.py"
    test_file = r"symbol_switch_manager_test.py"
    
    # 元ファイル読み込み
    with open(original_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if method_count is None:
        # 完全版
        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return "full"
    
    # メソッド削除版作成
    output_lines = []
    method_found = 0
    skip_method = False
    indent_level = 0
    
    for line in lines:
        # メソッド開始検出
        if line.strip().startswith('def ') and not line.strip().startswith('def __init__'):
            method_found += 1
            if method_found > method_count:
                skip_method = True
                indent_level = len(line) - len(line.lstrip())
                continue
        
        # スキップ中のメソッド終了検出
        if skip_method:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else float('inf')
            if line.strip() and current_indent <= indent_level:
                skip_method = False
            else:
                continue
        
        output_lines.append(line)
    
    # テストファイル作成
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    return f"methods_{method_count}"

def test_import_time(version_name):
    """インポート時間測定"""
    print(f"=== {version_name} インポートテスト ===")
    
    start_time = time.time()
    try:
        # 既存モジュールをクリア
        import sys
        if 'symbol_switch_manager_test' in sys.modules:
            del sys.modules['symbol_switch_manager_test']
        
        from symbol_switch_manager_test import SymbolSwitchManager
        import_time = (time.time() - start_time) * 1000
        print(f"[OK] {version_name}: {import_time:.1f}ms")
        return import_time
        
    except Exception as e:
        import_time = (time.time() - start_time) * 1000
        print(f"[ERROR] {version_name} エラー: {e}")
        print(f"   時間: {import_time:.1f}ms")
        return import_time

def main():
    """段階的テスト実行"""
    print("=== SymbolSwitchManager段階的軽量化テスト ===")
    
    # テストケース
    test_cases = [
        ("最小版 (メソッド2個)", 2),
        ("中版 (メソッド5個)", 5),
        ("大版 (メソッド10個)", 10),
        ("完全版", None)
    ]
    
    results = []
    
    for name, method_count in test_cases:
        version_name = create_test_version(method_count)
        import_time = test_import_time(name)
        results.append((name, import_time))
        
        # クリーンアップ
        if os.path.exists("symbol_switch_manager_test.py"):
            time.sleep(0.1)  # ファイルロック回避
    
    # 結果表示
    print(f"\n=== 段階的テスト結果 ===")
    for name, import_time in results:
        print(f"{name}: {import_time:.1f}ms")
    
    # 重い処理特定
    if len(results) >= 2:
        diff = results[-1][1] - results[0][1] 
        print(f"\n重い処理: {diff:.1f}ms")
        if diff > 1000:
            print("[WARNING] メソッド定義に1秒以上の重い処理があります！")

if __name__ == "__main__":
    main()