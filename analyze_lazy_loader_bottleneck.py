"""
lazy_loader ボトルネック詳細分析
DSSMSLazyModules.get_symbol_switch_manager()の重い処理調査

作成: 2025年10月2日
目的: lazy_loader 2832.6msの真の原因特定
"""

import time

def analyze_lazy_loader_bottleneck():
    """lazy_loader ボトルネック段階的分析"""
    print("=== lazy_loader ボトルネック詳細分析 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # 1. 基本クラスのインポート
    print("1. 基本LazyLoaderクラス定義時間")
    start = time.perf_counter()
    try:
        import importlib
        import logging
        from typing import Any, Dict, Callable
        from functools import wraps
        
        class TestLazyLoader:
            def __init__(self):
                self._loaded_modules: Dict[str, Any] = {}
                self._import_times: Dict[str, float] = {}
        
        basic_time = (time.perf_counter() - start) * 1000
        print(f"   基本LazyLoaderクラス: {basic_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ 基本クラスエラー: {e}")
        basic_time = 0
    
    # 2. DSSMSLazyModulesクラス単体
    print("\n2. DSSMSLazyModulesクラス定義時間")
    start = time.perf_counter()
    try:
        class TestDSSMSLazyModules:
            @staticmethod
            def get_test():
                return None, False
        
        dssms_class_time = (time.perf_counter() - start) * 1000
        print(f"   DSSMSLazyModulesクラス: {dssms_class_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ DSSMSLazyModulesエラー: {e}")
        dssms_class_time = 0
    
    # 3. 実際のlazy_loader全体
    print("\n3. 実際のlazy_loader全体インポート")
    start = time.perf_counter()
    try:
        from src.dssms.lazy_loader import LazyLoader
        lazy_loader_import_time = (time.perf_counter() - start) * 1000
        print(f"   LazyLoader import: {lazy_loader_import_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ LazyLoader importエラー: {e}")
        lazy_loader_import_time = 0
    
    # 4. DSSMSLazyModules単体
    print("\n4. DSSMSLazyModules単体インポート")
    start = time.perf_counter()
    try:
        from src.dssms.lazy_loader import DSSMSLazyModules
        dssms_modules_time = (time.perf_counter() - start) * 1000
        print(f"   DSSMSLazyModules import: {dssms_modules_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ DSSMSLazyModulesエラー: {e}")
        dssms_modules_time = 0
    
    # 5. lazy_import, lazy_class_import
    print("\n5. デコレータ関数インポート")
    start = time.perf_counter()
    try:
        from src.dssms.lazy_loader import lazy_import, lazy_class_import
        decorators_time = (time.perf_counter() - start) * 1000
        print(f"   デコレータ関数: {decorators_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ デコレータエラー: {e}")
        decorators_time = 0
    
    # 6. 全体再測定
    print("\n6. 全体統合インポート再測定")
    start = time.perf_counter()
    try:
        from src.dssms.lazy_loader import DSSMSLazyModules, lazy_import, lazy_class_import
        full_import_time = (time.perf_counter() - start) * 1000
        print(f"   全体統合インポート: {full_import_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ 全体エラー: {e}")
        full_import_time = 0
    
    return {
        'basic_time': basic_time,
        'dssms_class_time': dssms_class_time,
        'lazy_loader_import_time': lazy_loader_import_time,
        'dssms_modules_time': dssms_modules_time,
        'decorators_time': decorators_time,
        'full_import_time': full_import_time
    }

def analyze_symbol_switch_manager_call():
    """get_symbol_switch_manager()呼び出し分析"""
    print("\n7. get_symbol_switch_manager()実行時間分析")
    
    try:
        from src.dssms.lazy_loader import lazy_modules
        
        # 実際の呼び出し
        start = time.perf_counter()
        SymbolSwitchManagerClass, available = lazy_modules.get_symbol_switch_manager()
        call_time = (time.perf_counter() - start) * 1000
        
        print(f"   get_symbol_switch_manager(): {call_time:.1f}ms")
        print(f"   結果: {type(SymbolSwitchManagerClass).__name__ if SymbolSwitchManagerClass else 'None'}")
        print(f"   利用可能: {available}")
        
        return call_time
        
    except Exception as e:
        print(f"   ❌ get_symbol_switch_manager()エラー: {e}")
        return 0

def generate_optimization_recommendations(results):
    """最適化推奨案生成"""
    print("\n=== 最適化推奨案 ===")
    
    total_measured = sum(results.values())
    print(f"📊 分析合計時間: {total_measured:.1f}ms")
    
    # 主要ボトルネック特定
    bottleneck = max(results.items(), key=lambda x: x[1])
    print(f"🔴 最大ボトルネック: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")
    
    print("\n🎯 推奨最適化策:")
    
    if results['full_import_time'] > 1000:
        print("1. 【最優先】lazy_loader全体の簡素化")
        print("   - 不要な機能削除")
        print("   - 軽量版作成")
        print("   - 直接インポートへの回帰検討")
    
    if results['dssms_modules_time'] > 500:
        print("2. 【高優先】DSSMSLazyModulesの軽量化")
        print("   - get_symbol_switch_manager()の簡素化")
        print("   - 静的メソッドの見直し")
    
    print("3. 【代替案】lazy_loader完全除去")
    print("   - 直接インポート+条件分岐への回帰")
    print("   - シンプルな解決策")

def main():
    """メイン実行"""
    try:
        results = analyze_lazy_loader_bottleneck()
        call_time = analyze_symbol_switch_manager_call()
        generate_optimization_recommendations(results)
        
        print(f"\n📋 重要発見:")
        print(f"lazy_loaderは最適化のつもりが、実際は{results['full_import_time']:.1f}msの重い処理")
        print(f"シンプルな直接インポートの方が高速の可能性あり")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()