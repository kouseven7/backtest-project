"""
極限軽量版インポート時間測定
絶対最小限のインポートでの測定
"""

import time

def measure_minimal_imports():
    """絶対最小限のインポート測定"""
    print("=== 極限軽量版分析 ===")
    
    components = []
    
    # 1. sys, os のみ
    start = time.perf_counter()
    import sys
    import os
    basic_time = (time.perf_counter() - start) * 1000
    components.append(("sys, os", basic_time))
    
    # 2. datetimeを段階的追加
    start = time.perf_counter()
    from datetime import datetime, timedelta
    datetime_time = (time.perf_counter() - start) * 1000
    components.append(("datetime", datetime_time))
    
    # 3. TYPE_CHECKINGのみ
    start = time.perf_counter()
    from typing import TYPE_CHECKING
    type_checking_time = (time.perf_counter() - start) * 1000
    components.append(("TYPE_CHECKING", type_checking_time))
    
    # 4. importlib.util
    start = time.perf_counter()  
    import importlib.util
    importlib_time = (time.perf_counter() - start) * 1000
    components.append(("importlib.util", importlib_time))
    
    # 5. UltraLight関数定義
    start = time.perf_counter()
    def _load_symbol_switch_manager_ultra_light():
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ultra_light_path = os.path.join(current_dir, "src", "dssms", "symbol_switch_manager_ultra_light.py")
            
            if os.path.exists(ultra_light_path):
                spec = importlib.util.spec_from_file_location("symbol_switch_manager_ultra_light", ultra_light_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.SymbolSwitchManagerUltraLight
        except Exception:
            pass
        return None
        
    func_def_time = (time.perf_counter() - start) * 1000
    components.append(("関数定義", func_def_time))
    
    # 6. UltraLight実行
    start = time.perf_counter()
    SymbolSwitchManager = _load_symbol_switch_manager_ultra_light()
    ultra_exec_time = (time.perf_counter() - start) * 1000
    components.append(("UltraLight実行", ultra_exec_time))
    
    # 7. クラス定義（軽量版）
    start = time.perf_counter()
    
    class DSSMSIntegrationError(Exception):
        pass
    
    class MinimalBacktester:
        def __init__(self, config=None):
            self.config = config or {}
            self._logger = None
    
    class_def_time = (time.perf_counter() - start) * 1000  
    components.append(("クラス定義", class_def_time))
    
    return components

def main():
    """極限軽量版測定実行"""
    total_start = time.perf_counter()
    
    components = measure_minimal_imports()
    
    total_time = (time.perf_counter() - total_start) * 1000
    
    print("\n📊 極限軽量版インポート時間:")
    
    measured_total = sum(comp[1] for comp in components)
    
    for name, time_ms in components:
        if measured_total > 0:
            percentage = (time_ms / measured_total) * 100
            print(f"   {name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        else:
            print(f"   {name}: {time_ms:.1f}ms")
    
    print(f"\n📊 測定合計: {measured_total:.1f}ms")
    print(f"📊 実際の総時間: {total_time:.1f}ms")
    print(f"📊 未測定分: {total_time - measured_total:.1f}ms")
    
    # 1.2ms目標との比較
    target = 1.2
    remaining = total_time - target
    
    print(f"\n🎯 極限最適化結果:")
    print(f"   現在: {total_time:.1f}ms")
    print(f"   目標: {target:.1f}ms")
    print(f"   残り: {remaining:.1f}ms")
    
    if remaining <= 0:
        print("   ✅ 目標達成！")
    elif remaining < 5:
        print("   🔸 ほぼ目標達成")
        print("   💡 さらなる最適化: クラス定義簡素化、不要処理除去")
    else:
        print("   ⚠️ さらなる最適化が必要")
        
        # 最重要ボトルネック特定
        max_component = max(components, key=lambda x: x[1])
        print(f"   🔥 最大ボトルネック: {max_component[0]} ({max_component[1]:.1f}ms)")

if __name__ == "__main__":
    main()