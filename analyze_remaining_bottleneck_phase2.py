"""
SymbolSwitchManagerFast単独インポート時間測定
他のモジュール影響を排除した純粋測定

作成: 2025年10月2日  
目的: 2405.7msボトルネックの詳細原因特定
"""

import time
import importlib.util
import os
import sys

def test_pure_symbol_switch_manager_fast():
    """SymbolSwitchManagerFast単独の純粋インポート時間測定"""
    print("=== SymbolSwitchManagerFast単独インポート時間測定 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # 測定前のモジュール数
    modules_before = len(sys.modules)
    print(f"測定前モジュール数: {modules_before}")
    
    # 1. 直接ファイルパスインポート（dssms_integrated_main.pyと同じ方式）
    print("\n1. 直接ファイルパスインポート（dssms_integrated_main.py方式）")
    start = time.perf_counter()
    
    try:
        # 相対パスから絶対パスを取得
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fast_path = os.path.join(current_dir, "src", "dssms", "symbol_switch_manager_fast.py")
        print(f"   ファイルパス: {fast_path}")
        
        # ファイル存在確認
        if not os.path.exists(fast_path):
            print(f"   [ERROR] ファイルが存在しません: {fast_path}")
            return
        
        # 直接ファイルインポート
        spec = importlib.util.spec_from_file_location("symbol_switch_manager_fast", fast_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            SymbolSwitchManagerFast = module.SymbolSwitchManagerFast
            
            direct_import_time = (time.perf_counter() - start) * 1000
            print(f"   直接ファイルインポート: {direct_import_time:.1f}ms")
            
            # 測定後のモジュール数
            modules_after = len(sys.modules)
            new_modules = modules_after - modules_before
            print(f"   追加モジュール数: {new_modules}")
            
            # 初期化テスト
            start_init = time.perf_counter()
            config = {'switch_management': {'switch_cost_rate': 0.001}}
            instance = SymbolSwitchManagerFast(config)
            init_time = (time.perf_counter() - start_init) * 1000
            print(f"   初期化時間: {init_time:.1f}ms")
            
        else:
            print(f"   [ERROR] specまたはloaderが無効")
            
    except Exception as e:
        direct_import_time = 0
        print(f"   [ERROR] 直接インポートエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 比較: 通常のfromインポート（重い版）
    print("\n2. 通常のfromインポート（参考・重い版）")
    start = time.perf_counter()
    try:
        # 一度クリア（既にインポート済みの場合）
        modules_to_clear = [m for m in sys.modules.keys() if 'symbol_switch' in m or 'src.dssms' in m]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
        normal_import_time = (time.perf_counter() - start) * 1000
        print(f"   通常のfromインポート: {normal_import_time:.1f}ms")
        
        # 追加モジュール数
        modules_final = len(sys.modules)
        total_new_modules = modules_final - modules_before
        print(f"   総追加モジュール数: {total_new_modules}")
        
    except Exception as e:
        normal_import_time = 0
        print(f"   [ERROR] 通常インポートエラー: {e}")
    
    # 3. 結果比較
    print(f"\n=== 結果比較 ===")
    if direct_import_time > 0 and normal_import_time > 0:
        improvement = normal_import_time - direct_import_time
        improvement_rate = (improvement / normal_import_time) * 100
        print(f"[CHART] 直接インポート: {direct_import_time:.1f}ms")
        print(f"[CHART] 通常インポート: {normal_import_time:.1f}ms")
        print(f"[TARGET] 改善効果: {improvement:.1f}ms削減 ({improvement_rate:.1f}%改善)")
        
        if direct_import_time < 10:
            print("[OK] 直接インポートで大幅最適化成功")
        elif direct_import_time < 100:
            print("[WARNING] 直接インポートで中程度改善、追加最適化推奨")
        else:
            print("[ERROR] 直接インポートでも重い、根本的問題あり")
    else:
        print("[ERROR] 測定失敗 - エラーの詳細確認が必要")

def analyze_remaining_bottleneck():
    """残りボトルネック要因分析"""
    print("\n=== 残りボトルネック要因分析 ===")
    
    print("[SEARCH] 考えられる原因:")
    print("1. SymbolSwitchManagerFast内の重い依存ライブラリ")
    print("2. クラス定義・メソッド定義の処理時間")
    print("3. モジュール実行時の初期化処理")
    print("4. Python自体のインポート機構のオーバーヘッド")
    
    print("\n[IDEA] 次の最適化候補:")
    print("1. SymbolSwitchManagerFast内部の軽量化")
    print("2. 最小限機能のみの超軽量版作成")
    print("3. 遅延初期化・lazy loading導入")
    print("4. インポート回避・動的生成方式")

def main():
    """メイン実行"""
    try:
        test_pure_symbol_switch_manager_fast()
        analyze_remaining_bottleneck()
        
        print("\n=== TODO-PERF-005 Phase 2分析完了 ===")
        print("[OK] 直接パスインポート効果測定完了")
        print("[SEARCH] 残りボトルネック要因特定")
        print("[TARGET] 次Phase最適化方針決定")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()