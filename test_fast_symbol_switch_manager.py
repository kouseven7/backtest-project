"""
高速版SymbolSwitchManagerのインポート時間テスト
"""

import time

def test_fast_version():
    """高速版インポートテスト"""
    print("=== 高速版SymbolSwitchManagerテスト ===")
    
    start_time = time.time()
    from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
    import_time = (time.time() - start_time) * 1000
    print(f"✅ 高速版インポート: {import_time:.1f}ms")
    
    # 動作テスト
    config = {
        'switch_management': {
            'switch_cost_rate': 0.001,
            'min_holding_days': 1,
            'max_switches_per_month': 10,
            'cost_threshold': 0.001
        }
    }
    
    ssm = SymbolSwitchManagerFast(config)
    
    # 基本動作確認
    evaluation = ssm.evaluate_symbol_switch(None, '7203', None)
    print(f"✅ 評価結果: {evaluation['should_switch']} ({evaluation['reason']})")
    
    return import_time

def test_original_version():
    """元版インポートテスト（比較用）"""
    print("=== 元版SymbolSwitchManagerテスト ===")
    
    start_time = time.time()
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    import_time = (time.time() - start_time) * 1000
    print(f"✅ 元版インポート: {import_time:.1f}ms")
    
    return import_time

if __name__ == "__main__":
    print("=== SymbolSwitchManager高速化比較テスト ===")
    
    fast_time = test_fast_version()
    original_time = test_original_version()
    
    print(f"\n=== 比較結果 ===")
    print(f"高速版: {fast_time:.1f}ms")
    print(f"元版: {original_time:.1f}ms")
    print(f"改善: {original_time - fast_time:.1f}ms ({original_time/fast_time:.1f}x高速化)")
    
    if fast_time < 100:
        print("🎉 高速版が100ms未満を達成！")