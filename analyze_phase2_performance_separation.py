"""
TODO-PERF-001 Phase 2 最適化対象分離分析

ユーザー指摘：
- src/dssms/dssms_integrated_main.py
- main.py
の最適化が混在している可能性

調査内容：
1. Phase 2で実際に最適化されたのはどちらのファイルか
2. 実行時間短縮はどこで発生したか
3. 混在問題の有無
"""

import time
import sys
import os

# プロジェクトルートをパスに追加
project_root = r"C:\Users\imega\Documents\my_backtest_project"
sys.path.append(project_root)

def test_dssms_integrated_main_performance():
    """src/dssms/dssms_integrated_main.py のパフォーマンステスト"""
    print("=== dssms_integrated_main.py パフォーマンステスト ===")
    
    try:
        start_time = time.time()
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        import_time = time.time() - start_time
        
        start_init = time.time()
        backtester = DSSMSIntegratedBacktester()
        init_time = time.time() - start_init
        
        total_time = time.time() - start_time
        
        print(f"[OK] dssms_integrated_main.py:")
        print(f"   Import時間: {import_time*1000:.1f}ms")
        print(f"   初期化時間: {init_time*1000:.1f}ms")
        print(f"   合計時間: {total_time*1000:.1f}ms")
        
        return total_time * 1000
        
    except Exception as e:
        print(f"[ERROR] dssms_integrated_main.py エラー: {e}")
        return None

def test_main_py_performance():
    """main.py のパフォーマンステスト"""
    print("\n=== main.py パフォーマンステスト ===")
    
    try:
        # main.pyの主要コンポーネントをインポート
        start_time = time.time()
        
        # main.pyで使用される主要モジュール
        import main
        import_time = time.time() - start_time
        
        total_time = time.time() - start_time
        
        print(f"[OK] main.py:")
        print(f"   Import時間: {import_time*1000:.1f}ms")
        print(f"   合計時間: {total_time*1000:.1f}ms")
        
        return total_time * 1000
        
    except Exception as e:
        print(f"[ERROR] main.py エラー: {e}")
        return None

def analyze_symbol_switch_manager_dependency():
    """SymbolSwitchManagerがどちらで使用されているか分析"""
    print("\n=== SymbolSwitchManager依存関係分析 ===")
    
    # dssms_integrated_main.pyでの使用確認
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        print("[OK] dssms_integrated_main.py: SymbolSwitchManager使用確認")
        backtester = DSSMSIntegratedBacktester()
        if hasattr(backtester, 'switch_manager'):
            print("   - switch_manager属性存在")
        else:
            print("   - switch_manager属性なし")
    except Exception as e:
        print(f"[ERROR] dssms_integrated_main.py: {e}")
    
    # main.pyでの使用確認
    try:
        import main
        print("[OK] main.py: SymbolSwitchManager使用確認")
        # main.pyのソースコードをチェック
        main_file = os.path.join(project_root, "main.py")
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'SymbolSwitchManager' in content:
                print("   - SymbolSwitchManager文字列存在")
            else:
                print("   - SymbolSwitchManager文字列なし")
    except Exception as e:
        print(f"[ERROR] main.py: {e}")

def test_symbol_switch_manager_fast_loading():
    """高速版SymbolSwitchManagerの遅延ロード効果測定"""
    print("\n=== SymbolSwitchManager高速版効果測定 ===")
    
    # 元版
    try:
        start_time = time.time()
        from src.dssms.symbol_switch_manager import SymbolSwitchManager
        orig_time = time.time() - start_time
        print(f"元版SymbolSwitchManager: {orig_time*1000:.1f}ms")
    except Exception as e:
        print(f"元版エラー: {e}")
        orig_time = None
    
    # 高速版
    try:
        start_time = time.time()
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
        fast_time = time.time() - start_time
        print(f"高速版SymbolSwitchManagerFast: {fast_time*1000:.1f}ms")
    except Exception as e:
        print(f"高速版エラー: {e}")
        fast_time = None
    
    # 遅延ローダー
    try:
        start_time = time.time()
        # lazy_loader除去 (TODO-PERF-001: Stage 3)
# 直接インポートに変更: get_symbol_switch_manager
        switch_class, available = get_symbol_switch_manager()
        lazy_time = time.time() - start_time
        print(f"遅延ローダー: {lazy_time*1000:.1f}ms (クラス: {switch_class.__name__})")
    except Exception as e:
        print(f"遅延ローダーエラー: {e}")
        lazy_time = None
    
    if orig_time and fast_time:
        improvement = ((orig_time - fast_time) / orig_time) * 100
        print(f"高速化効果: {improvement:.1f}% 改善")

def analyze_phase2_optimization_target():
    """Phase 2最適化対象の特定"""
    print("\n=== Phase 2最適化対象特定 ===")
    
    # dssms_integrated_main.pyの実行時間測定
    dssms_time = test_dssms_integrated_main_performance()
    
    # main.pyの実行時間測定
    main_time = test_main_py_performance()
    
    print(f"\n[TARGET] Phase 2最適化対象分析結果:")
    if dssms_time is not None and main_time is not None:
        if dssms_time > main_time:
            print(f"   主要ボトルネック: dssms_integrated_main.py ({dssms_time:.1f}ms)")
            print(f"   副次的: main.py ({main_time:.1f}ms)")
        else:
            print(f"   主要ボトルネック: main.py ({main_time:.1f}ms)")
            print(f"   副次的: dssms_integrated_main.py ({dssms_time:.1f}ms)")
    
    # Phase 2の成果確認
    print(f"\n[CHART] Phase 2成果確認:")
    print(f"   目標: 1500ms")
    if dssms_time is not None:
        print(f"   dssms_integrated_main.py: {dssms_time:.1f}ms")
        if dssms_time <= 1500:
            print(f"   [OK] 目標達成")
        else:
            print(f"   [ERROR] 目標未達 (残り{dssms_time-1500:.1f}ms)")

def main():
    print("=== TODO-PERF-001 Phase 2 最適化対象分離分析 ===")
    print("ユーザー指摘: dssms_integrated_main.py と main.py の最適化混在問題調査")
    
    # 1. 個別パフォーマンステスト
    analyze_phase2_optimization_target()
    
    # 2. SymbolSwitchManager依存関係分析
    analyze_symbol_switch_manager_dependency()
    
    # 3. 高速版効果測定
    test_symbol_switch_manager_fast_loading()
    
    print(f"\n=== 結論 ===")
    print("Phase 2最適化で実際に対象となったファイルを特定完了")
    print("この結果をdocs/dssms/Fallback problem countermeasures/Fallback problem countermeasures.mdに整理します")

if __name__ == "__main__":
    main()