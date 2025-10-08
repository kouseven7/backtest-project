"""
Phase 2最適化修正テストスクリプト
dssms_integrated_main.pyの軽量版切り替え修正の動作確認

作成: 2025年10月2日
目的: TODO-PERF-002 Phase 2修正の検証
"""

import time
import sys
import os

def test_phase2_optimization_fix():
    """Phase 2最適化修正の動作確認"""
    print("=== Phase 2最適化修正テスト ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    print("1. 修正後のDSSMSIntegratedBacktester インポート時間測定")
    start_import = time.perf_counter()
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        import_time = (time.perf_counter() - start_import) * 1000
        print(f"   [CHART] DSSMSIntegratedBacktester: {import_time:.1f}ms")
        
        target_time = 1.2
        if import_time <= target_time:
            print(f"   [OK] Phase 2目標達成! ({target_time}ms以下)")
            improvement = True
        else:
            remaining = import_time - target_time
            print(f"   [WARNING] Phase 2目標未達成: 残り{remaining:.1f}ms短縮必要")
            improvement = False
            
    except Exception as e:
        print(f"   [ERROR] インポートエラー: {e}")
        return False
    
    print()
    print("2. 軽量版SymbolSwitchManager使用確認")
    try:
        config = {
            'symbol_switch': {
                'switch_cost_rate': 0.001,
                'min_holding_days': 1,
                'max_switches_per_month': 10
            }
        }
        
        start_init = time.perf_counter()
        backtester = DSSMSIntegratedBacktester(config)
        init_time = (time.perf_counter() - start_init) * 1000
        
        print(f"   [CHART] 初期化時間: {init_time:.1f}ms")
        
        # コンポーネント初期化
        switch_manager = backtester.ensure_components()
        
        if switch_manager:
            switch_manager_type = type(switch_manager).__name__
            print(f"   [SEARCH] 使用中SymbolSwitchManager: {switch_manager_type}")
            
            if switch_manager_type == "SymbolSwitchManagerFast":
                print(f"   [OK] 軽量版使用中（Phase 2修正成功）")
                fast_version_used = True
            elif switch_manager_type == "SymbolSwitchManager":
                print(f"   [ERROR] 重い元版使用中（Phase 2修正未適用）")
                fast_version_used = False
            else:
                print(f"   [WARNING] 不明なSymbolSwitchManager: {switch_manager_type}")
                fast_version_used = False
        else:
            print(f"   [ERROR] SymbolSwitchManager初期化失敗")
            fast_version_used = False
            
    except Exception as e:
        print(f"   [ERROR] 初期化エラー: {e}")
        fast_version_used = False
    
    print()  
    print("3. lazy_loader統計記録確認")
    try:
        # lazy_loader除去 (TODO-PERF-001: Stage 3)
# 直接インポートに変更: lazy_modules
        stats = # lazy_modules除去: get_import_stats()
        
        if stats:
            print("   [UP] 遅延ロード統計 (使用記録):")
            for module, load_time in stats.items():
                print(f"      {module}: {load_time:.1f}ms")
            stats_recorded = True
        else:
            print("   [ERROR] 遅延ロード統計が空（未使用）")
            stats_recorded = False
            
    except Exception as e:
        print(f"   [ERROR] lazy_loader統計エラー: {e}")
        stats_recorded = False
    
    print()
    print("=== Phase 2修正結果サマリー ===")
    
    results = {
        'import_time_improved': improvement,
        'fast_version_used': fast_version_used,
        'stats_recorded': stats_recorded
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"[CHART] 修正成功率: {success_rate:.1f}% ({success_count}/{total_count})")
    
    if success_rate >= 66.7:  # 2/3以上
        print("[OK] Phase 2修正おおむね成功")
        if success_rate < 100:
            print("[WARNING] 一部課題残存 - 追加修正必要")
    else:
        print("[ERROR] Phase 2修正要再検討")
    
    return results

def identify_remaining_issues():
    """残存課題特定"""
    print("\n4. 残存課題特定")
    
    # SymbolSwitchManagerFast性能問題調査
    print("   [SEARCH] SymbolSwitchManagerFast性能逆転問題:")
    try:
        start_fast = time.perf_counter()
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
        fast_import_time = (time.perf_counter() - start_fast) * 1000
        
        start_orig = time.perf_counter()
        from src.dssms.symbol_switch_manager import SymbolSwitchManager
        orig_import_time = (time.perf_counter() - start_orig) * 1000
        
        print(f"      SymbolSwitchManagerFast: {fast_import_time:.1f}ms")
        print(f"      SymbolSwitchManager: {orig_import_time:.1f}ms")
        
        if fast_import_time > orig_import_time:
            ratio = fast_import_time / orig_import_time
            print(f"      [ERROR] 逆転問題確認: 軽量版が{ratio:.1f}倍重い")
            print(f"      [LIST] 次のタスク: SymbolSwitchManagerFast実装見直し必要")
        else:
            print(f"      [OK] 性能逆転解決済み")
            
    except Exception as e:
        print(f"      [ERROR] 性能測定エラー: {e}")

def main():
    """メイン実行"""
    try:
        results = test_phase2_optimization_fix()
        identify_remaining_issues()
        
        print("\n[LIST] 次のステップ:")
        if not results['fast_version_used']:
            print("1. 【最優先】SymbolSwitchManagerFast使用確認・統合調査")
        if not results['stats_recorded']:
            print("2. 【高優先】lazy_loader統計記録機能修正")
        print("3. 【中優先】SymbolSwitchManagerFast性能逆転問題解決")
        print("4. 【低優先】Phase 3最適化準備（yfinance遅延ローディング）")
        
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()