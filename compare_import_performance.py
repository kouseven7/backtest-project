"""
軽量版と完全版のインポート時間比較テスト
"""

import time

def test_lite_import():
    """軽量版インポートテスト"""
    print("=== 軽量版インポートテスト ===")
    start_time = time.time()
    
    from symbol_switch_manager_lite import SymbolSwitchManagerLite
    
    import_time = (time.time() - start_time) * 1000
    print(f"[OK] 軽量版インポート: {import_time:.1f}ms")
    
    return import_time

def test_full_import():
    """完全版インポートテスト"""
    print("=== 完全版インポートテスト ===")
    start_time = time.time()
    
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    
    import_time = (time.time() - start_time) * 1000
    print(f"[OK] 完全版インポート: {import_time:.1f}ms")
    
    return import_time

if __name__ == "__main__":
    print("=== インポート時間比較テスト ===")
    
    lite_time = test_lite_import()
    full_time = test_full_import()
    
    print(f"\n=== 比較結果 ===")
    print(f"軽量版: {lite_time:.1f}ms")
    print(f"完全版: {full_time:.1f}ms")
    print(f"差異: {full_time - lite_time:.1f}ms ({(full_time/lite_time):.1f}x)")
    
    if full_time > lite_time * 10:
        print("[WARNING] 完全版に重い処理があります！")