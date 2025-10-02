"""
新規Pythonプロセスでの初回インポート時間測定
キャッシュを回避した正確な測定
"""

import subprocess
import sys
import time

def test_clean_import(module_path, class_name):
    """新規プロセスでのインポート時間測定"""
    script = f"""
import time
start_time = time.time()

from {module_path} import {class_name}

import_time = (time.time() - start_time) * 1000
print(f"IMPORT_TIME: {{import_time:.1f}}ms")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 結果から時間を抽出
        for line in result.stdout.split('\n'):
            if line.startswith('IMPORT_TIME:'):
                time_str = line.split('IMPORT_TIME: ')[1].replace('ms', '')
                return float(time_str)
        
        return None
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

def main():
    """比較テスト実行"""
    print("=== 新規プロセスでの初回インポート比較 ===")
    
    # 高速版テスト
    print("1. 高速版テスト...")
    fast_time = test_clean_import('src.dssms.symbol_switch_manager_fast', 'SymbolSwitchManagerFast')
    
    # 元版テスト
    print("2. 元版テスト...")
    original_time = test_clean_import('src.dssms.symbol_switch_manager', 'SymbolSwitchManager')
    
    # 結果表示
    print(f"\n=== 比較結果（新規プロセス） ===")
    
    if fast_time and original_time:
        print(f"高速版: {fast_time:.1f}ms")
        print(f"元版: {original_time:.1f}ms")
        
        if fast_time < original_time:
            improvement = original_time - fast_time
            speed_ratio = original_time / fast_time
            print(f"✅ 改善: {improvement:.1f}ms ({speed_ratio:.1f}x高速化)")
        else:
            print(f"❌ 高速版が遅い: {fast_time - original_time:.1f}ms差")
    else:
        print("❌ 測定失敗")
        if fast_time:
            print(f"高速版のみ: {fast_time:.1f}ms")
        if original_time:
            print(f"元版のみ: {original_time:.1f}ms")

if __name__ == "__main__":
    main()