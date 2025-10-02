"""
dssms_integrated_main.py 残り65ms詳細分析
コンポーネント別インポート時間測定

作成: 2025年10月2日
目的: 1.2ms目標達成のための最終最適化
"""

import time
import sys
import importlib.util

def measure_component_import_times():
    """各コンポーネントのインポート時間を詳細測定"""
    print("=== dssms_integrated_main.py 65ms詳細分析 ===")
    
    components = []
    total_start = time.perf_counter()
    
    # 1. 基本的なビルトインインポート
    start = time.perf_counter()
    import sys
    import os
    basic_time = (time.perf_counter() - start) * 1000
    components.append(("sys, os", basic_time))
    
    # 2. datetime, timedelta
    start = time.perf_counter()
    from datetime import datetime, timedelta
    datetime_time = (time.perf_counter() - start) * 1000
    components.append(("datetime", datetime_time))
    
    # 3. typing (重い可能性)
    start = time.perf_counter()
    from typing import Dict, List, Any, Optional, Tuple
    typing_time = (time.perf_counter() - start) * 1000
    components.append(("typing", typing_time))
    
    # 4. logging
    start = time.perf_counter()
    import logging
    logging_time = (time.perf_counter() - start) * 1000
    components.append(("logging", logging_time))
    
    # 5. time (注意: 名前衝突)
    start = time.perf_counter()
    import time as time_module
    time_time = (time.perf_counter() - start) * 1000
    components.append(("time", time_time))
    
    # 6. pathlib (重い可能性)
    start = time.perf_counter()
    from pathlib import Path
    pathlib_time = (time.perf_counter() - start) * 1000
    components.append(("pathlib", pathlib_time))
    
    # 7. json
    start = time.perf_counter()
    import json
    json_time = (time.perf_counter() - start) * 1000
    components.append(("json", json_time))
    
    # 8. argparse (コマンドライン処理、重い可能性)
    start = time.perf_counter()
    import argparse
    argparse_time = (time.perf_counter() - start) * 1000
    components.append(("argparse", argparse_time))
    
    # 9. importlib.util
    start = time.perf_counter()
    import importlib.util
    importlib_time = (time.perf_counter() - start) * 1000
    components.append(("importlib.util", importlib_time))
    
    # 10. UltraLight版のロード
    start = time.perf_counter()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dssms_dir = os.path.join(current_dir, "src", "dssms")
    ultra_light_path = os.path.join(src_dssms_dir, "symbol_switch_manager_ultra_light.py")
    
    if os.path.exists(ultra_light_path):
        spec = importlib.util.spec_from_file_location("symbol_switch_manager_ultra_light", ultra_light_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ultralight_time = (time.perf_counter() - start) * 1000
            components.append(("UltraLight", ultralight_time))
        else:
            ultralight_time = 0
            components.append(("UltraLight (失敗)", ultralight_time))
    else:
        ultralight_time = 0
        components.append(("UltraLight (不存在)", ultralight_time))
    
    # 11. DSSMSIntegratedBacktesterクラス定義（実際のクラスのロード）
    start = time.perf_counter()  
    # クラス定義は測定困難なので、残りの時間を推定
    class_def_time = 0  # プレースホルダー
    components.append(("クラス定義", class_def_time))
    
    total_time = (time.perf_counter() - total_start) * 1000
    components.append(("総測定時間", total_time))
    
    return components

def analyze_heavy_components(components):
    """重いコンポーネントの分析"""
    print("\n📊 コンポーネント別インポート時間:")
    
    # 総時間除く
    import_components = components[:-1]
    total_measured = sum(comp[1] for comp in import_components)
    
    # 降順ソート
    sorted_components = sorted(import_components, key=lambda x: x[1], reverse=True)
    
    for name, time_ms in sorted_components:
        if total_measured > 0:
            percentage = (time_ms / total_measured) * 100
            print(f"   {name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        else:
            print(f"   {name}: {time_ms:.1f}ms")
    
    print(f"\n📊 測定済み合計: {total_measured:.1f}ms")
    print(f"📊 実際の総時間: {components[-1][1]:.1f}ms")
    
    return sorted_components

def suggest_optimizations(sorted_components):
    """最適化提案"""
    print("\n🔧 最適化提案:")
    
    heavy_threshold = 5.0  # 5ms以上を重いとする
    heavy_imports = [comp for comp in sorted_components if comp[1] > heavy_threshold]
    
    if not heavy_imports:
        print("   ✅ すべてのインポートが軽量（5ms未満）")
        print("   📋 次の最適化候補:")
        print("      1. クラス定義の簡素化")
        print("      2. メソッド定義の遅延ロード")
        print("      3. 不要なメソッド・属性の除去")
        return
    
    print(f"   🔥 重いインポート（{len(heavy_imports)}個）:")
    for name, time_ms in heavy_imports:
        print(f"      - {name}: {time_ms:.1f}ms")
        
        # 個別最適化提案
        if "typing" in name.lower():
            print(f"        💡 typing最適化: TYPE_CHECKING使用、必要最小限のみインポート")
        elif "pathlib" in name.lower():
            print(f"        💡 pathlib最適化: os.path使用、またはstr操作で代替")
        elif "argparse" in name.lower():
            print(f"        💡 argparse最適化: sys.argv直接解析、または遅延インポート")
        elif "datetime" in name.lower():
            print(f"        💡 datetime最適化: time.time()使用、または遅延インポート")
        elif "json" in name.lower():
            print(f"        💡 json最適化: 遅延インポート、またはliteral_eval使用")

def main():
    """メイン実行"""
    try:
        components = measure_component_import_times()
        sorted_components = analyze_heavy_components(components)
        suggest_optimizations(sorted_components)
        
        # 目標との比較
        total_measured = sum(comp[1] for comp in components[:-1])
        target = 1.2
        remaining = total_measured - target
        
        print(f"\n🎯 目標達成状況:")
        print(f"   現在: {total_measured:.1f}ms")
        print(f"   目標: {target:.1f}ms")
        print(f"   残り削減必要: {remaining:.1f}ms ({remaining/total_measured*100:.1f}%)")
        
        if remaining <= 0:
            print("   ✅ 目標達成済み！")
        elif remaining < 10:
            print("   🔸 あと少しで達成")
        else:
            print("   ⚠️ さらなる最適化が必要")
            
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()