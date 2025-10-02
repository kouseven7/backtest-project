"""
実際のDSSMSIntegratedBacktesterインポート詳細分析
クラス初期化時の遅延インポート問題調査
"""

import time
import sys

def measure_actual_dssms_import():
    """実際のDSSMSIntegratedBacktesterインポート測定"""
    print("=== 実際のDSSMSIntegratedBacktester詳細分析 ===")
    
    # 段階別測定
    print("1. インポート文の実行時間測定")
    
    start = time.perf_counter()
    from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
    import_time = (time.perf_counter() - start) * 1000
    print(f"   from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester: {import_time:.1f}ms")
    
    # 2. クラス初期化時間測定
    print("\n2. クラス初期化時間測定")
    
    config = {
        'initial_capital': 1000000,
        'symbol_switch': {
            'switch_cost_rate': 0.001,
            'min_holding_days': 1
        }
    }
    
    start = time.perf_counter()
    try:
        backtester = DSSMSIntegratedBacktester(config)
        init_time = (time.perf_counter() - start) * 1000
        print(f"   DSSMSIntegratedBacktester初期化: {init_time:.1f}ms")
        
        # 3. 初期化時の遅延インポート調査
        print("\n3. 初期化時の遅延インポート調査")
        print(f"   dss_core属性: {backtester.dss_core}")
        print(f"   advanced_ranking_engine属性: {backtester.advanced_ranking_engine}")
        
        # 属性アクセス時のインポート測定
        if hasattr(backtester, '_ensure_components_loaded'):
            print("\n4. _ensure_components_loaded実行時間測定")
            start = time.perf_counter()
            backtester._ensure_components_loaded()
            ensure_time = (time.perf_counter() - start) * 1000
            print(f"   _ensure_components_loaded: {ensure_time:.1f}ms")
        
        return {
            'import_time': import_time,
            'init_time': init_time,
            'ensure_time': ensure_time if 'ensure_time' in locals() else 0
        }
        
    except Exception as e:
        print(f"   ❌ 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return {
            'import_time': import_time,
            'init_time': 0,
            'ensure_time': 0
        }

def analyze_modules_loaded():
    """ロード済みモジュール分析"""
    print("\n=== ロード済みモジュール分析 ===")
    
    modules_before = len(sys.modules)
    heavy_modules_before = [m for m in sys.modules if any(lib in m for lib in ['pandas', 'numpy', 'yfinance', 'openpyxl', 'matplotlib'])]
    
    print(f"初期ロード済みモジュール数: {modules_before}")
    print(f"初期重いライブラリ: {len(heavy_modules_before)}")
    for mod in heavy_modules_before[:5]:  # 最初の5個のみ表示
        print(f"   - {mod}")
    
    # DSSMSIntegratedBacktesterインポート実行
    result = measure_actual_dssms_import()
    
    modules_after = len(sys.modules)
    heavy_modules_after = [m for m in sys.modules if any(lib in m for lib in ['pandas', 'numpy', 'yfinance', 'openpyxl', 'matplotlib'])]
    
    print(f"\nDSSMS後ロード済みモジュール数: {modules_after}")
    print(f"DSSMS後重いライブラリ: {len(heavy_modules_after)}")
    
    new_modules = modules_after - modules_before
    new_heavy_modules = [m for m in heavy_modules_after if m not in heavy_modules_before]
    
    print(f"新規ロードモジュール数: {new_modules}")
    print(f"新規重いライブラリ: {len(new_heavy_modules)}")
    
    if new_heavy_modules:
        print("新規重いライブラリ詳細:")
        for mod in new_heavy_modules[:10]:  # 最初の10個のみ表示
            print(f"   - {mod}")
    
    return result

def main():
    """メイン実行"""
    try:
        result = analyze_modules_loaded()
        
        print("\n=== 総合分析結果 ===")
        print(f"📊 インポート時間: {result['import_time']:.1f}ms")
        print(f"📊 初期化時間: {result['init_time']:.1f}ms")
        print(f"📊 コンポーネント確保時間: {result['ensure_time']:.1f}ms")
        
        total_measured = result['import_time'] + result['init_time'] + result['ensure_time']
        print(f"📊 測定総時間: {total_measured:.1f}ms")
        
        # 2854ms問題との比較
        expected_problem_time = 2854.4
        if total_measured < expected_problem_time / 2:
            print(f"⚠️ 大きな差異発見: 予想{expected_problem_time:.1f}ms vs 実測{total_measured:.1f}ms")
            print("🔍 可能性:")
            print("   1. 測定タイミングの違い（クラス取得 vs インスタンス作成）")
            print("   2. 他の遅延インポートが未発火")
            print("   3. 初回実行vs2回目実行の差")
            print("   4. キャッシュ効果の影響")
        else:
            print("✅ 測定値が妥当範囲")
            
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()