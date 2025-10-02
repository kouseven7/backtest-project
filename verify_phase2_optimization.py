"""
DSSMS Phase 2最適化検証スクリプト
dssms_integrated_main.pyの実行時間とボトルネック分析

作成: 2025年10月2日
目的: Phase 2最適化の実態調査
"""

import time
import sys
import os

def measure_import_time():
    """インポート時間測定"""
    print("=== DSSMS Phase 2最適化検証 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # Phase 2で最適化されたはずの軽量版確認
    print("1. SymbolSwitchManagerFast軽量版確認")
    start_fast = time.perf_counter()
    try:
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
        fast_time = (time.perf_counter() - start_fast) * 1000
        print(f"   ✅ SymbolSwitchManagerFast: {fast_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ SymbolSwitchManagerFast: エラー - {e}")
    
    print()
    
    # 元版の重い処理確認
    print("2. 元版SymbolSwitchManager（重い）確認")
    start_orig = time.perf_counter()
    try:
        from src.dssms.symbol_switch_manager import SymbolSwitchManager
        orig_time = (time.perf_counter() - start_orig) * 1000
        print(f"   ⚠️ SymbolSwitchManager: {orig_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ SymbolSwitchManager: エラー - {e}")
    
    print()
    
    # 最も重要: dssms_integrated_main.py本体
    print("3. dssms_integrated_main.py インポート時間")
    start_main = time.perf_counter()
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        main_time = (time.perf_counter() - start_main) * 1000
        print(f"   📊 DSSMSIntegratedBacktester: {main_time:.1f}ms")
        
        # Phase 2の目標: 2763ms → 1.2ms (99.96%改善)
        target_time = 1.2
        if main_time <= target_time:
            print(f"   ✅ Phase 2目標達成! ({target_time}ms以下)")
        else:
            improvement_needed = main_time - target_time
            print(f"   ❌ Phase 2目標未達成: 残り{improvement_needed:.1f}ms短縮必要")
            
    except Exception as e:
        print(f"   ❌ DSSMSIntegratedBacktester: エラー - {e}")
    
    print()
    
    # lazy_loader統計確認
    print("4. lazy_loader統計確認")
    try:
        from src.dssms.lazy_loader import lazy_modules
        stats = lazy_modules.get_import_stats()
        if stats:
            print("   📈 遅延ロード統計:")
            for module, load_time in stats.items():
                print(f"      {module}: {load_time:.1f}ms")
        else:
            print("   ⚠️ 遅延ロード未使用")
    except Exception as e:
        print(f"   ❌ lazy_loader統計: エラー - {e}")

def measure_initialization_time():
    """初期化時間測定"""
    print("\n5. DSSMSIntegratedBacktester初期化時間")
    
    try:
        # 基本設定
        config = {
            'symbol_switch': {
                'switch_cost_rate': 0.001,
                'min_holding_days': 1,
                'max_switches_per_month': 10
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_rate': 0.05
            }
        }
        
        start_init = time.perf_counter()
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        backtester = DSSMSIntegratedBacktester(config)
        init_time = (time.perf_counter() - start_init) * 1000
        
        print(f"   📊 初期化時間: {init_time:.1f}ms")
        
        # どのコンポーネントが使用されているか確認
        switch_manager_type = type(backtester.switch_manager).__name__ if hasattr(backtester, 'switch_manager') and backtester.switch_manager else "None"
        print(f"   🔍 使用中SymbolSwitchManager: {switch_manager_type}")
        
        if switch_manager_type == "SymbolSwitchManagerFast":
            print("   ✅ 軽量版使用中（Phase 2最適化成功）")
        elif switch_manager_type == "SymbolSwitchManager":
            print("   ❌ 重い元版使用中（Phase 2最適化未適用）")
        else:
            print("   ⚠️ SymbolSwitchManager未初期化")
            
    except Exception as e:
        print(f"   ❌ 初期化エラー: {e}")

def main():
    """メイン実行"""
    try:
        measure_import_time()
        measure_initialization_time()
        
        print("\n=== Phase 2最適化検証完了 ===")
        print("\n📋 次のステップ:")
        print("1. dssms_integrated_main.pyの軽量版切り替え修正")
        print("2. 残ボトルネック特定（データ取得・戦略実行・Excel出力）")
        print("3. Phase 3最適化計画策定")
        
    except Exception as e:
        print(f"検証エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()