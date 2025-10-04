#!/usr/bin/env python3
"""
TODO-PERF-001現状分析レポート - 2025年10月4日版
TODO-PERF-003/006解決後の実際のパフォーマンス状況を詳細調査

分析対象:
1. DSSMSIntegratedBacktester全体のインポート時間
2. lazy_loader統合状況（TODO-PERF-002確認）
3. SymbolSwitchManager最適化状況（Phase 2問題確認）
4. yfinance/openpyxlボトルネック現状（Phase 3）
5. システム全体実行時間現状
"""

import time
import sys
import os
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def measure_import_time(module_name, description=""):
    """インポート時間測定ヘルパー"""
    try:
        start_time = time.perf_counter()
        module = __import__(module_name, fromlist=[''])
        end_time = time.perf_counter()
        import_time = (end_time - start_time) * 1000
        return import_time, module, None
    except Exception as e:
        return None, None, str(e)

def analyze_dssms_performance():
    """DSSMS全体パフォーマンス分析"""
    print("=" * 80)
    print("TODO-PERF-001現状分析レポート - 2025年10月4日版")
    print("=" * 80)
    
    results = {}
    
    # 1. DSSMSIntegratedBacktester現状測定
    print("\n1. DSSMSIntegratedBacktester インポート時間測定:")
    import_time, module, error = measure_import_time('src.dssms.dssms_integrated_main')
    if error:
        print(f"   ❌ インポートエラー: {error}")
        results['dssms_import'] = {'status': 'error', 'error': error}
    else:
        print(f"   ✅ インポート時間: {import_time:.1f}ms")
        results['dssms_import'] = {'status': 'success', 'time_ms': import_time}
    
    # 2. lazy_loader統合状況確認
    print("\n2. lazy_loader統合状況確認:")
    try:
        # lazy_loaderファイルの存在確認
        lazy_loader_path = project_root / "src" / "utils" / "lazy_import_manager.py"
        if lazy_loader_path.exists():
            print(f"   ✅ lazy_loader存在: {lazy_loader_path}")
            
            # dssms_integrated_main.pyでの使用状況確認
            dssms_main_path = project_root / "src" / "dssms" / "dssms_integrated_main.py"
            if dssms_main_path.exists():
                with open(dssms_main_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'lazy_loader' in content:
                        print("   ⚠️  lazy_loader参照発見 - 完全除去未完了")
                        results['lazy_loader_integration'] = 'partial_removal'
                    else:
                        print("   ✅ lazy_loader参照なし - 完全除去済み")
                        results['lazy_loader_integration'] = 'fully_removed'
        else:
            print("   ℹ️  lazy_loader不存在 - 完全除去済み")
            results['lazy_loader_integration'] = 'fully_removed'
    except Exception as e:
        print(f"   ❌ lazy_loader確認エラー: {e}")
        results['lazy_loader_integration'] = f'error: {e}'
    
    # 3. SymbolSwitchManager最適化状況
    print("\n3. SymbolSwitchManager最適化状況:")
    try:
        # UltraLight版の存在確認
        ultra_light_path = project_root / "src" / "dssms" / "symbol_switch_manager_ultra_light.py"
        if ultra_light_path.exists():
            # UltraLight版インポート測定
            ul_time, ul_module, ul_error = measure_import_time('src.dssms.symbol_switch_manager_ultra_light')
            if ul_error:
                print(f"   ❌ UltraLight版エラー: {ul_error}")
                results['ultra_light'] = {'status': 'error', 'error': ul_error}
            else:
                print(f"   ✅ UltraLight版インポート: {ul_time:.1f}ms")
                results['ultra_light'] = {'status': 'success', 'time_ms': ul_time}
        else:
            print("   ❌ UltraLight版未発見")
            results['ultra_light'] = {'status': 'not_found'}
        
        # 通常版インポート測定
        normal_time, normal_module, normal_error = measure_import_time('src.dssms.symbol_switch_manager')
        if normal_error:
            print(f"   ❌ 通常版エラー: {normal_error}")
            results['normal_switch_manager'] = {'status': 'error', 'error': normal_error}
        else:
            print(f"   ✅ 通常版インポート: {normal_time:.1f}ms")
            results['normal_switch_manager'] = {'status': 'success', 'time_ms': normal_time}
    except Exception as e:
        print(f"   ❌ SymbolSwitchManager確認エラー: {e}")
        results['switch_manager_analysis'] = f'error: {e}'
    
    # 4. 重いライブラリボトルネック現状
    print("\n4. 重いライブラリボトルネック現状:")
    heavy_libraries = [
        ('yfinance', 'yfinance'),
        ('openpyxl', 'openpyxl'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy')
    ]
    
    library_times = {}
    for lib_name, import_name in heavy_libraries:
        lib_time, lib_module, lib_error = measure_import_time(import_name)
        if lib_error:
            print(f"   ❌ {lib_name}: エラー {lib_error}")
            library_times[lib_name] = {'status': 'error', 'error': lib_error}
        else:
            print(f"   📊 {lib_name}: {lib_time:.1f}ms")
            library_times[lib_name] = {'status': 'success', 'time_ms': lib_time}
    
    results['heavy_libraries'] = library_times
    
    # 5. システム全体実行時間評価
    print("\n5. システム全体実行時間評価（簡易テスト実行）:")
    try:
        start_time = time.perf_counter()
        
        # 簡易バックテスト実行シミュレーション
        if results['dssms_import']['status'] == 'success':
            from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
            
            # 軽量設定での初期化テスト
            init_start = time.perf_counter()
            config = {
                'symbols': ['7203', '9984'],  # 軽量テスト
                'start_date': '2024-01-01',
                'end_date': '2024-01-05',
                'risk_free_rate': 0.01
            }
            backtester = DSSMSIntegratedBacktester(config)
            init_end = time.perf_counter()
            
            init_time = (init_end - init_start) * 1000
            print(f"   ✅ 初期化時間: {init_time:.1f}ms")
            
            # 合計時間
            total_time = (init_end - start_time) * 1000
            print(f"   📊 合計時間（インポート+初期化）: {total_time:.1f}ms")
            
            results['system_execution'] = {
                'status': 'success',
                'init_time_ms': init_time,
                'total_time_ms': total_time
            }
        else:
            print("   ⚠️  DSSMSインポートエラーのため実行テスト不可")
            results['system_execution'] = {'status': 'skipped', 'reason': 'import_failed'}
    except Exception as e:
        print(f"   ❌ 実行テストエラー: {e}")
        results['system_execution'] = {'status': 'error', 'error': str(e)}
    
    # 6. TODO-PERF-001問題点評価
    print("\n6. TODO-PERF-001問題点評価:")
    evaluation = analyze_perf_001_status(results)
    
    print("\n" + "=" * 80)
    print("分析完了 - 詳細結果をreports/で確認可能")
    print("=" * 80)
    
    return results, evaluation

def analyze_perf_001_status(results):
    """TODO-PERF-001問題の現在の状況評価"""
    evaluation = {
        'phase2_lazy_loader_issue': 'unknown',
        'phase2_symbol_switch_issue': 'unknown', 
        'phase3_heavy_libraries': 'unknown',
        'overall_performance': 'unknown',
        'remaining_issues': []
    }
    
    # Phase 2 lazy_loader問題評価
    if results.get('lazy_loader_integration') == 'fully_removed':
        evaluation['phase2_lazy_loader_issue'] = 'resolved'
        print("   ✅ Phase 2 lazy_loader問題: 解決済み")
    elif results.get('lazy_loader_integration') == 'partial_removal':
        evaluation['phase2_lazy_loader_issue'] = 'partially_resolved'
        evaluation['remaining_issues'].append('lazy_loader参照が残存')
        print("   ⚠️  Phase 2 lazy_loader問題: 部分解決")
    else:
        print("   ❌ Phase 2 lazy_loader問題: 不明")
    
    # Phase 2 SymbolSwitchManager問題評価
    dssms_time = results.get('dssms_import', {}).get('time_ms', 0)
    if dssms_time < 100:  # 100ms以下なら良好
        evaluation['phase2_symbol_switch_issue'] = 'resolved'
        print(f"   ✅ Phase 2 SymbolSwitchManager問題: 解決済み ({dssms_time:.1f}ms)")
    elif dssms_time < 1000:  # 1秒以下なら改善済み
        evaluation['phase2_symbol_switch_issue'] = 'improved'
        evaluation['remaining_issues'].append(f'SymbolSwitchManager最適化不十分 ({dssms_time:.1f}ms)')
        print(f"   ⚠️  Phase 2 SymbolSwitchManager問題: 改善済みだが不十分 ({dssms_time:.1f}ms)")
    else:
        evaluation['phase2_symbol_switch_issue'] = 'unresolved'
        evaluation['remaining_issues'].append(f'SymbolSwitchManager重大ボトルネック ({dssms_time:.1f}ms)')
        print(f"   ❌ Phase 2 SymbolSwitchManager問題: 未解決 ({dssms_time:.1f}ms)")
    
    # Phase 3重いライブラリ評価
    heavy_libs = results.get('heavy_libraries', {})
    yfinance_time = heavy_libs.get('yfinance', {}).get('time_ms', 0)
    openpyxl_time = heavy_libs.get('openpyxl', {}).get('time_ms', 0)
    
    if yfinance_time > 500 or openpyxl_time > 200:
        evaluation['phase3_heavy_libraries'] = 'unresolved'
        evaluation['remaining_issues'].append(f'重いライブラリボトルネック (yfinance:{yfinance_time:.0f}ms, openpyxl:{openpyxl_time:.0f}ms)')
        print(f"   ❌ Phase 3重いライブラリ: 未解決 (yfinance:{yfinance_time:.0f}ms, openpyxl:{openpyxl_time:.0f}ms)")
    else:
        evaluation['phase3_heavy_libraries'] = 'acceptable' 
        print(f"   ✅ Phase 3重いライブラリ: 許容範囲 (yfinance:{yfinance_time:.0f}ms, openpyxl:{openpyxl_time:.0f}ms)")
    
    # 全体パフォーマンス評価
    total_time = results.get('system_execution', {}).get('total_time_ms', 0)
    if total_time < 1500:  # 目標1500ms
        evaluation['overall_performance'] = 'target_achieved'
        print(f"   ✅ 全体パフォーマンス: 目標達成 ({total_time:.1f}ms < 1500ms)")
    elif total_time < 3000:  # 3秒以下なら改善済み
        evaluation['overall_performance'] = 'improved'
        evaluation['remaining_issues'].append(f'目標1500ms未達成 ({total_time:.1f}ms)')
        print(f"   ⚠️  全体パフォーマンス: 改善済みだが目標未達成 ({total_time:.1f}ms)")
    else:
        evaluation['overall_performance'] = 'poor'
        evaluation['remaining_issues'].append(f'深刻なパフォーマンス問題 ({total_time:.1f}ms)')
        print(f"   ❌ 全体パフォーマンス: 深刻な問題 ({total_time:.1f}ms)")
    
    # 残存課題数評価
    remaining_count = len(evaluation['remaining_issues'])
    if remaining_count == 0:
        print(f"\n   🎯 TODO-PERF-001総合評価: ✅ 完全解決")
    elif remaining_count <= 2:
        print(f"\n   🎯 TODO-PERF-001総合評価: ⚠️  部分解決 ({remaining_count}件残存)")
    else:
        print(f"\n   🎯 TODO-PERF-001総合評価: ❌ 重大問題残存 ({remaining_count}件)")
    
    return evaluation

if __name__ == "__main__":
    try:
        results, evaluation = analyze_dssms_performance()
        
        # 結果保存
        import json
        import datetime
        
        # レポート生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            'timestamp': timestamp,
            'analysis_results': results,
            'evaluation': evaluation,
            'conclusion': {
                'todo_perf_001_status': 'partially_resolved' if evaluation['remaining_issues'] else 'resolved',
                'remaining_issues_count': len(evaluation['remaining_issues']),
                'remaining_issues': evaluation['remaining_issues'],
                'next_priorities': []
            }
        }
        
        # 次の優先事項決定
        if evaluation['phase2_symbol_switch_issue'] == 'unresolved':
            report_data['conclusion']['next_priorities'].append('Phase 2 SymbolSwitchManager最適化完了')
        if evaluation['phase3_heavy_libraries'] == 'unresolved':
            report_data['conclusion']['next_priorities'].append('Phase 3 重いライブラリ遅延インポート実装')
        if evaluation['overall_performance'] in ['poor', 'improved']:
            report_data['conclusion']['next_priorities'].append('システム全体最適化継続')
        
        # レポート保存
        os.makedirs('reports', exist_ok=True)
        report_path = f'reports/todo_perf_001_analysis_{timestamp}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 詳細レポート保存: {report_path}")
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback
        traceback.print_exc()