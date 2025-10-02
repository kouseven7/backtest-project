"""
TODO-PERF-003前提条件分析スクリプト
Phase 3最適化対象の妥当性検証

作成: 2025年10月2日
目的: インポート時間 vs 実行時間の混同問題解決
"""

import time
import sys

def analyze_current_performance_status():
    """現在のパフォーマンス状況を正確に分析"""
    print("=== TODO-PERF-003前提条件分析 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    results = {}
    
    # 1. 現在のDSSMSIntegratedBacktesterインポート時間測定
    print("1. 【インポート時間】現在の状況測定")
    start = time.perf_counter()
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        import_time = (time.perf_counter() - start) * 1000
        results['dssms_import_time'] = import_time
        print(f"   DSSMSIntegratedBacktesterインポート時間: {import_time:.1f}ms")
        
        # 目標との比較
        target_import = 1.2
        remaining_import = import_time - target_import
        if remaining_import > 0:
            print(f"   📊 TODO-PERF-005残り: {remaining_import:.1f}ms削減必要")
            print(f"   🎯 インポート時間最適化: {remaining_import/import_time*100:.1f}%削減で達成")
        else:
            print(f"   ✅ インポート時間目標達成済み")
        
    except Exception as e:
        print(f"   ❌ インポートエラー: {e}")
        results['dssms_import_time'] = 0
        import traceback
        traceback.print_exc()
    
    # 2. 実行時間サンプル測定（軽量テスト）
    print("\n2. 【実行時間】軽量テスト実行")
    if results.get('dssms_import_time', 0) > 0:
        start = time.perf_counter()
        try:
            # 軽量設定で短期間テスト
            config = {
                'initial_capital': 1000000,
                'symbol_switch': {
                    'switch_cost_rate': 0.001,
                    'min_holding_days': 1
                }
            }
            backtester = DSSMSIntegratedBacktester(config)
            init_time = (time.perf_counter() - start) * 1000
            results['init_time'] = init_time
            print(f"   初期化時間: {init_time:.1f}ms")
            
            # 軽量バックテスト（1日のみ）
            from datetime import datetime, timedelta
            start_exec = time.perf_counter()
            
            # 1日だけの軽量実行テスト
            test_date = datetime(2024, 1, 5)  # 営業日
            
            # バックテスト実行時間測定（軽量版）
            print(f"   軽量バックテスト実行中...")
            # 実際の実行は省略（時間がかかりすぎるため）
            exec_time_estimate = 0  # 実際の測定は別途必要
            results['exec_time_sample'] = exec_time_estimate
            
            print(f"   ⚠️ 実行時間測定は別途詳細分析が必要")
            
        except Exception as e:
            print(f"   ❌ 実行テストエラー: {e}")
            results['init_time'] = 0
            results['exec_time_sample'] = 0
    
    # 3. yfinance・openpyxl等の重いライブラリ影響調査
    print("\n3. 【重いライブラリ】影響度調査")
    
    # yfinanceインポート時間
    modules_before = len(sys.modules)
    start = time.perf_counter()
    try:
        import yfinance as yf
        yfinance_import_time = (time.perf_counter() - start) * 1000
        modules_after = len(sys.modules)
        yfinance_modules = modules_after - modules_before
        results['yfinance_import_time'] = yfinance_import_time
        print(f"   yfinanceインポート時間: {yfinance_import_time:.1f}ms")
        print(f"   yfinance追加モジュール数: {yfinance_modules}")
    except Exception as e:
        print(f"   ❌ yfinanceインポートエラー: {e}")
        results['yfinance_import_time'] = 0
    
    # openpyxlインポート時間
    modules_before = len(sys.modules)
    start = time.perf_counter()
    try:
        import openpyxl
        openpyxl_import_time = (time.perf_counter() - start) * 1000
        modules_after = len(sys.modules)
        openpyxl_modules = modules_after - modules_before
        results['openpyxl_import_time'] = openpyxl_import_time
        print(f"   openpyxlインポート時間: {openpyxl_import_time:.1f}ms")
        print(f"   openpyxl追加モジュール数: {openpyxl_modules}")
    except Exception as e:
        print(f"   ❌ openpyxlインポートエラー: {e}")
        results['openpyxl_import_time'] = 0
    
    return results

def analyze_todo_perf_003_validity(results):
    """TODO-PERF-003の妥当性分析"""
    print("\n=== TODO-PERF-003妥当性分析 ===")
    
    dssms_import = results.get('dssms_import_time', 0)
    yfinance_import = results.get('yfinance_import_time', 0)
    openpyxl_import = results.get('openpyxl_import_time', 0)
    
    print("🔍 前提条件検証:")
    
    # 1. インポート時間 vs 実行時間の混同チェック
    print("1. 測定対象の整合性:")
    print(f"   現在のDSSMSIntegratedBacktesterインポート時間: {dssms_import:.1f}ms")
    print(f"   TODO-PERF-003で言及されたyfinance: 957.5ms")
    print(f"   実測yfinanceインポート時間: {yfinance_import:.1f}ms")
    
    if yfinance_import > 0:
        discrepancy = abs(957.5 - yfinance_import)
        if discrepancy > 100:
            print(f"   ⚠️ 大きな乖離: {discrepancy:.1f}ms差 - 測定条件が異なる可能性")
        else:
            print(f"   ✅ 概ね一致: {discrepancy:.1f}ms差")
    
    # 2. 最適化対象の優先度分析
    print("\n2. 最適化対象優先度:")
    import_bottlenecks = [
        ('DSSMSIntegratedBacktester', dssms_import),
        ('yfinance', yfinance_import),
        ('openpyxl', openpyxl_import)
    ]
    import_bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    for i, (component, time_ms) in enumerate(import_bottlenecks):
        if time_ms > 0:
            print(f"   {i+1}. {component}: {time_ms:.1f}ms")
    
    # 3. Phase 3の方向性提案
    print("\n3. Phase 3方向性提案:")
    
    if dssms_import > 1.2:
        remaining = dssms_import - 1.2
        print(f"   🎯 優先課題: TODO-PERF-005完了 (残り{remaining:.1f}ms削減)")
        print(f"   📋 対象: DSSMSIntegratedBacktesterインポート時間最適化")
        phase3_priority = "DSSMS_IMPORT_OPTIMIZATION"
    elif yfinance_import > 100 or openpyxl_import > 100:
        print(f"   🎯 次期課題: 重いライブラリインポート最適化")
        print(f"   📋 対象: yfinance ({yfinance_import:.1f}ms), openpyxl ({openpyxl_import:.1f}ms)")
        phase3_priority = "HEAVY_LIBRARY_OPTIMIZATION"
    else:
        print(f"   ✅ インポート時間最適化完了")
        print(f"   📋 次段階: 実行時間最適化に移行可能")
        phase3_priority = "EXECUTION_TIME_OPTIMIZATION"
    
    return {
        'phase3_priority': phase3_priority,
        'dssms_import_remaining': max(0, dssms_import - 1.2),
        'heavy_libraries_impact': yfinance_import + openpyxl_import,
        'recommendation': generate_recommendation(phase3_priority, results)
    }

def generate_recommendation(priority, results):
    """推奨事項生成"""
    recommendations = []
    
    if priority == "DSSMS_IMPORT_OPTIMIZATION":
        remaining = results.get('dssms_import_time', 0) - 1.2
        recommendations.append(f"TODO-PERF-005継続: DSSMSIntegratedBacktester残り{remaining:.1f}ms削減")
        recommendations.append("超軽量版SymbolSwitchManager統合完了")
        recommendations.append("他の重いコンポーネント特定・最適化")
        
    elif priority == "HEAVY_LIBRARY_OPTIMIZATION":
        yfinance_time = results.get('yfinance_import_time', 0)
        openpyxl_time = results.get('openpyxl_import_time', 0)
        recommendations.append(f"yfinance遅延ローディング実装 ({yfinance_time:.1f}ms削減可能)")
        recommendations.append(f"openpyxl遅延ローディング実装 ({openpyxl_time:.1f}ms削減可能)")
        recommendations.append("@lazy_import統合による条件付きインポート")
        
    elif priority == "EXECUTION_TIME_OPTIMIZATION":
        recommendations.append("インポート時間最適化完了 - 実行時間最適化に移行")
        recommendations.append("バックテスト実行時間の詳細分析")
        recommendations.append("データ取得・処理・計算の最適化")
    
    return recommendations

def main():
    """メイン実行"""
    try:
        # 現在の状況分析
        results = analyze_current_performance_status()
        
        # TODO-PERF-003妥当性分析
        analysis = analyze_todo_perf_003_validity(results)
        
        # 結果サマリー
        print("\n=== 分析結果サマリー ===")
        print(f"🎯 Phase 3優先度: {analysis['phase3_priority']}")
        print(f"📊 DSSMS残り削減: {analysis['dssms_import_remaining']:.1f}ms")
        print(f"📊 重ライブラリ影響: {analysis['heavy_libraries_impact']:.1f}ms")
        
        print("\n📋 推奨事項:")
        for i, rec in enumerate(analysis['recommendation'], 1):
            print(f"   {i}. {rec}")
        
        print("\n🔍 TODO-PERF-003検証結果:")
        if analysis['phase3_priority'] == "DSSMS_IMPORT_OPTIMIZATION":
            print("   ⚠️ 前提条件不適切: まずTODO-PERF-005完了が必要")
        elif analysis['phase3_priority'] == "HEAVY_LIBRARY_OPTIMIZATION":
            print("   ✅ Phase 3として適切: 重いライブラリ最適化が有効")
        else:
            print("   📈 段階移行: 実行時間最適化フェーズへ")
        
        return analysis
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()