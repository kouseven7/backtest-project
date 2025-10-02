"""
SymbolSwitchManagerFast詳細ボトルネック分析
2746.4msのインポート時間の原因を詳細特定

作成: 2025年10月2日
目的: TODO-PERF-005の段階的最適化方針決定
"""

import time
import sys
import importlib.util

def measure_import_step_by_step():
    """SymbolSwitchManagerFastのインポート段階別測定"""
    print("=== SymbolSwitchManagerFast詳細ボトルネック分析 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    measurements = {}
    
    # 1. 基本インポート測定
    print("1. 基本Pythonライブラリインポート時間")
    basic_libs = ['datetime', 'typing', 'logging']
    for lib in basic_libs:
        start = time.perf_counter()
        if lib in sys.modules:
            del sys.modules[lib]
        __import__(lib)
        elapsed = (time.perf_counter() - start) * 1000
        measurements[f'basic_{lib}'] = elapsed
        print(f"   {lib}: {elapsed:.1f}ms")
    
    # 2. ファイル読み込みテスト
    print("\n2. symbol_switch_manager_fast.pyファイル読み込み時間")
    fast_file_path = "src/dssms/symbol_switch_manager_fast.py"
    start = time.perf_counter()
    try:
        with open(fast_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        file_read_time = (time.perf_counter() - start) * 1000
        measurements['file_read'] = file_read_time
        print(f"   ファイル読み込み: {file_read_time:.1f}ms")
        print(f"   ファイルサイズ: {len(content)}文字")
    except Exception as e:
        print(f"   ❌ ファイル読み込みエラー: {e}")
        measurements['file_read'] = 0
    
    # 3. 段階別インポート測定
    print("\n3. 段階別インポート測定")
    
    # 3.1 モジュール準備
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location(
        "symbol_switch_manager_fast", 
        fast_file_path
    )
    spec_time = (time.perf_counter() - start) * 1000
    measurements['spec_creation'] = spec_time
    print(f"   モジュールspec作成: {spec_time:.1f}ms")
    
    # 3.2 モジュール作成
    start = time.perf_counter()
    module = importlib.util.module_from_spec(spec)
    module_time = (time.perf_counter() - start) * 1000
    measurements['module_creation'] = module_time
    print(f"   モジュール作成: {module_time:.1f}ms")
    
    # 3.3 モジュール実行（最も重い処理）
    start = time.perf_counter()
    try:
        spec.loader.exec_module(module)
        exec_time = (time.perf_counter() - start) * 1000
        measurements['module_execution'] = exec_time
        print(f"   モジュール実行: {exec_time:.1f}ms")
        
        # 3.4 クラス取得
        start = time.perf_counter()
        SymbolSwitchManagerFast = getattr(module, 'SymbolSwitchManagerFast')
        class_get_time = (time.perf_counter() - start) * 1000
        measurements['class_access'] = class_get_time
        print(f"   クラス取得: {class_get_time:.1f}ms")
        
        # 3.5 クラス初期化テスト
        start = time.perf_counter()
        config = {'switch_management': {'switch_cost_rate': 0.001}}
        instance = SymbolSwitchManagerFast(config)
        init_time = (time.perf_counter() - start) * 1000
        measurements['class_init'] = init_time
        print(f"   クラス初期化: {init_time:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ モジュール実行エラー: {e}")
        measurements['module_execution'] = 0
        measurements['class_access'] = 0
        measurements['class_init'] = 0
        
        import traceback
        print("   エラー詳細:")
        traceback.print_exc()
    
    # 4. 比較測定: 通常のインポート
    print("\n4. 通常のインポート方式との比較")
    start = time.perf_counter()
    try:
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast
        normal_import_time = (time.perf_counter() - start) * 1000
        measurements['normal_import'] = normal_import_time
        print(f"   通常のfromインポート: {normal_import_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ 通常インポートエラー: {e}")
        measurements['normal_import'] = 0
    
    return measurements

def analyze_bottleneck_causes(measurements):
    """ボトルネック原因分析とレポート生成"""
    print("\n=== ボトルネック原因分析 ===")
    
    # 最大時間項目特定
    sorted_items = sorted(measurements.items(), key=lambda x: x[1], reverse=True)
    print("📊 処理時間ランキング:")
    for i, (item, time_ms) in enumerate(sorted_items[:5]):
        print(f"   {i+1}. {item}: {time_ms:.1f}ms")
    
    # 分析結果
    total_measured = sum(measurements.values())
    main_bottleneck = sorted_items[0] if sorted_items else ('unknown', 0)
    
    print(f"\n🔍 分析結果:")
    print(f"   測定合計時間: {total_measured:.1f}ms")
    print(f"   最大ボトルネック: {main_bottleneck[0]} ({main_bottleneck[1]:.1f}ms)")
    
    # 最適化推奨
    print(f"\n🎯 最適化推奨:")
    if main_bottleneck[1] > 1000:
        print(f"   🔴 緊急: {main_bottleneck[0]}の大幅削減が必要")
    elif main_bottleneck[1] > 100:
        print(f"   🟡 重要: {main_bottleneck[0]}の中程度削減が必要")  
    else:
        print(f"   🟢 軽微: 全体的な軽微最適化で改善可能")
    
    # 具体的改善案
    if 'module_execution' in measurements and measurements['module_execution'] > 1000:
        print(f"   💡 改善案: モジュール実行時の重い処理を遅延化")
    if 'normal_import' in measurements and measurements['normal_import'] > 1000:
        print(f"   💡 改善案: インポート依存関係の見直し・軽量化")
    
    return {
        'main_bottleneck': main_bottleneck,
        'total_time': total_measured,
        'optimization_priority': sorted_items
    }

def generate_optimization_plan(analysis_result):
    """最適化計画生成"""
    print("\n=== 段階的最適化計画 ===")
    
    main_bottleneck, main_time = analysis_result['main_bottleneck']
    total_time = analysis_result['total_time']
    
    print("🗺️ Phase-by-Phase最適化方針:")
    
    # Phase 1: 主要ボトルネック
    if main_time > 1000:
        print(f"   Phase 1: {main_bottleneck}最適化 (削減目標: {main_time:.1f}ms → 100ms)")
        phase1_target = main_time - 100
    else:
        print(f"   Phase 1: 軽微最適化 (削減目標: 全体50%削減)")
        phase1_target = total_time * 0.5
    
    # Phase 2: 残存最適化
    print(f"   Phase 2: 残存ボトルネック最適化")
    
    # Phase 3: 目標達成確認
    target_time = 1.0
    print(f"   Phase 3: 最終目標確認 (目標: {target_time}ms)")
    
    remaining_reduction = total_time - target_time
    if remaining_reduction > 0:
        print(f"   📈 必要削減量: {remaining_reduction:.1f}ms (現在比{remaining_reduction/total_time*100:.1f}%削減)")
    else:
        print(f"   ✅ 既に目標達成済み")
    
    return {
        'phase1_target_reduction': phase1_target if main_time > 1000 else total_time * 0.5,
        'total_target_reduction': remaining_reduction,
        'feasibility': 'high' if remaining_reduction < total_time * 0.9 else 'challenging'
    }

def main():
    """メイン実行"""
    try:
        # 詳細測定実行
        measurements = measure_import_step_by_step()
        
        # ボトルネック分析
        analysis = analyze_bottleneck_causes(measurements)
        
        # 最適化計画生成
        plan = generate_optimization_plan(analysis)
        
        # 結果サマリー
        print("\n=== TODO-PERF-005実装サマリー ===")
        print(f"🔍 分析完了: {len(measurements)}項目測定")
        print(f"🎯 主要ボトルネック: {analysis['main_bottleneck'][0]} ({analysis['main_bottleneck'][1]:.1f}ms)")
        print(f"📊 削減必要量: {plan['total_target_reduction']:.1f}ms")
        print(f"🚀 実現可能性: {plan['feasibility']}")
        
        if plan['feasibility'] == 'high':
            print("\n✅ TODO-PERF-005は段階的実装で達成可能")
        else:
            print("\n⚠️ TODO-PERF-005は困難 - 追加戦略検討必要")
        
        print("\n📋 次ステップ:")
        print("1. Phase 1: 主要ボトルネック最適化実装")
        print("2. 効果測定・検証")
        print("3. Phase 2-3: 残存最適化")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()