"""
TODO-PERF-001 Phase 2完了後の最終パフォーマンス測定
"""

import time
import subprocess
import sys

def measure_execution_time():
    """main.py実行時間測定"""
    print("=== TODO-PERF-001 Phase 2完了後 最終パフォーマンス測定 ===")
    
    start_time = time.time()
    
    try:
        # main.py実行
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True,
            text=True,
            timeout=300  # 5分タイムアウト
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        print(f"✅ 実行完了: {execution_time:.0f}ms")
        
        if result.returncode == 0:
            print(f"🎉 正常終了")
        else:
            print(f"⚠️ 異常終了: return code {result.returncode}")
        
        # 目標達成確認
        target_time = 1500  # ms
        if execution_time <= target_time:
            print(f"🎯 目標達成！ {execution_time:.0f}ms ≤ {target_time}ms")
        else:
            remaining = execution_time - target_time
            print(f"🔄 あと{remaining:.0f}ms短縮が必要")
        
        return execution_time
        
    except subprocess.TimeoutExpired:
        execution_time = (time.time() - start_time) * 1000
        print(f"❌ タイムアウト: {execution_time:.0f}ms")
        return execution_time
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        print(f"❌ 実行エラー: {e}")
        return execution_time

if __name__ == "__main__":
    execution_time = measure_execution_time()
    
    print(f"\n=== 最終結果 ===")
    print(f"実行時間: {execution_time:.0f}ms")
    print(f"目標時間: 1500ms")
    
    improvement_from_original = 5000 - execution_time  # 元は5000ms以上
    if improvement_from_original > 0:
        print(f"改善量: {improvement_from_original:.0f}ms")
        print(f"改善率: {improvement_from_original/5000*100:.1f}%")