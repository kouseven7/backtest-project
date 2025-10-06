"""
lazy loading統合後のSymbolSwitchManager高速化テスト
"""

import time
# lazy_loader除去 (TODO-PERF-001: Stage 3)
# 直接インポートに変更: lazy_modules

def test_lazy_symbol_switch_manager():
    """lazy loading経由でのSymbolSwitchManager取得テスト"""
    print("=== Lazy Loading SymbolSwitchManagerテスト ===")
    
    start_time = time.time()
    
    # lazy loading経由で高速版取得
    SymbolSwitchManagerClass, available = # lazy_modules除去: get_symbol_switch_manager()
    
    load_time = (time.time() - start_time) * 1000
    print(f"✅ lazy loading取得: {load_time:.1f}ms")
    
    if available:
        print(f"取得したクラス: {SymbolSwitchManagerClass.__name__}")
        
        # 基本動作テスト
        config = {
            'switch_management': {
                'switch_cost_rate': 0.001,
                'min_holding_days': 1,
                'max_switches_per_month': 10,
                'cost_threshold': 0.001
            }
        }
        
        ssm = SymbolSwitchManagerClass(config)
        evaluation = ssm.evaluate_symbol_switch(None, '7203', None)
        print(f"✅ 動作確認: {evaluation['should_switch']} ({evaluation['reason']})")
        
        return load_time
    else:
        print("❌ SymbolSwitchManagerが取得できません")
        return None

def test_direct_import_comparison():
    """直接インポートとの比較"""
    print("\n=== 直接インポート比較テスト ===")
    
    start_time = time.time()
    from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
    direct_time = (time.time() - start_time) * 1000
    print(f"✅ 直接インポート: {direct_time:.1f}ms")
    
    return direct_time

def main():
    """統合テスト実行"""
    print("=== Lazy Loading統合テスト ===")
    
    lazy_time = test_lazy_symbol_switch_manager()
    direct_time = test_direct_import_comparison()
    
    if lazy_time and direct_time:
        print(f"\n=== 比較結果 ===")
        print(f"Lazy Loading: {lazy_time:.1f}ms")
        print(f"直接インポート: {direct_time:.1f}ms")
        
        if lazy_time < 100:
            print("🎉 Lazy Loading版が100ms未満を達成！")
        
        # 統計表示
        stats = # lazy_modules除去: get_import_stats()
        print(f"\n=== インポート統計 ===")
        for module, time_ms in stats.items():
            print(f"  {module}: {time_ms:.1f}ms")

if __name__ == "__main__":
    main()