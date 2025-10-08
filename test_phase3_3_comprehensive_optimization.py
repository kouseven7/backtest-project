"""
Phase 3-3 統合効果測定・追加最適化判定テスト
作成: 2025年10月3日

TODO-PERF-003 Phase 3全体実装効果測定
yfinance + openpyxl遅延インポートによる1456.6ms削減効果を総合検証
"""

import sys
import os
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def measure_full_system_performance_original():
    """従来システム全体性能測定"""
    print("=== 従来システム全体性能測定（直接インポート） ===")
    
    # 新プロセス起動で純粋測定
    import subprocess
    script_content = '''
import time
start = time.perf_counter()

# 重いライブラリ直接インポート
import yfinance as yf
# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, Reference

total_time = (time.perf_counter() - start) * 1000
print(f"RESULT:{total_time:.1f}")
'''
    
    try:
        with open("temp_original_test.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        result = subprocess.run([sys.executable, "temp_original_test.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('RESULT:'):
                    import_time = float(line.split(':')[1])
                    print(f"従来システム重いライブラリインポート: {import_time:.1f}ms")
                    
                    os.remove("temp_original_test.py")
                    return {'total_import_time_ms': import_time}
        else:
            print(f"従来システム測定エラー: {result.stderr}")
            
    except Exception as e:
        print(f"従来システム測定例外: {e}")
    
    finally:
        if os.path.exists("temp_original_test.py"):
            os.remove("temp_original_test.py")
    
    return {'total_import_time_ms': 0, 'error': 'measurement_failed'}

def measure_full_system_performance_optimized():
    """最適化システム全体性能測定"""
    print("\n=== 最適化システム全体性能測定（遅延インポート） ===")
    
    start_total = time.perf_counter()
    
    # 遅延インポートマネージャー読み込み
    start_manager = time.perf_counter()
    from src.utils.lazy_import_manager import get_yfinance, get_openpyxl, get_lazy_import_stats, get_total_optimization_effect
    manager_time = (time.perf_counter() - start_manager) * 1000
    print(f"LazyImportManager読み込み: {manager_time:.1f}ms")
    
    # DSSMS Core コンポーネント読み込み
    start_dssms = time.perf_counter()
    try:
        from src.dssms.nikkei225_screener import Nikkei225Screener
        from src.dssms.dssms_data_manager import DSSMSDataManager
        dssms_time = (time.perf_counter() - start_dssms) * 1000
        print(f"DSSMS Core読み込み: {dssms_time:.1f}ms")
    except Exception as e:
        print(f"DSSMS Core読み込みエラー: {e}")
        dssms_time = 0
    
    # Excel Exporter読み込み
    start_excel = time.perf_counter()
    try:
        # 型注釈エラーを回避するため、インポートのみテスト
        from output import dssms_excel_exporter_v2
        excel_time = (time.perf_counter() - start_excel) * 1000
        print(f"Excel Exporter読み込み: {excel_time:.1f}ms")
    except Exception as e:
        print(f"Excel Exporter読み込みエラー: {e}")
        excel_time = 0
    
    # 実際の重いライブラリ使用テスト
    start_heavy = time.perf_counter()
    try:
        yf = get_yfinance()
        ticker = yf.Ticker("7203.T")  # 軽量テスト
        
        openpyxl = get_openpyxl()
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: wb = openpyxl.Workbook()
        
        heavy_time = (time.perf_counter() - start_heavy) * 1000
        print(f"重いライブラリ実使用: {heavy_time:.1f}ms")
    except Exception as e:
        print(f"重いライブラリ使用エラー: {e}")
        heavy_time = 0
    
    total_time = (time.perf_counter() - start_total) * 1000
    print(f"最適化システム全体: {total_time:.1f}ms")
    
    # 統計取得
    stats = get_lazy_import_stats()
    effect = get_total_optimization_effect()
    
    return {
        'manager_time_ms': manager_time,
        'dssms_time_ms': dssms_time,
        'excel_time_ms': excel_time,
        'heavy_lib_time_ms': heavy_time,
        'total_time_ms': total_time,
        'stats': stats,
        'effect': effect
    }

def calculate_phase3_comprehensive_effect(original_result, optimized_result):
    """Phase 3統合効果計算"""
    print("\n=== Phase 3統合効果計算 ===")
    
    original_time = original_result.get('total_import_time_ms', 0)
    optimized_time = optimized_result.get('total_time_ms', 0)
    
    if original_time > 0 and optimized_time > 0:
        time_saved = original_time - optimized_time
        improvement_rate = (time_saved / original_time) * 100
        
        print(f"従来システム時間: {original_time:.1f}ms")
        print(f"最適化システム時間: {optimized_time:.1f}ms")
        print(f"削減時間: {time_saved:.1f}ms")
        print(f"改善率: {improvement_rate:.1f}%")
        
        # Phase 3目標達成
        target_reduction = 1456.6  # yfinance 1201.8ms + openpyxl 254.7ms
        achievement_rate = (time_saved / target_reduction) * 100
        print(f"Phase 3目標削減時間: {target_reduction:.1f}ms")
        print(f"Phase 3目標達成率: {achievement_rate:.1f}%")
        
        # 個別コンポーネント効果詳細
        stats = optimized_result.get('stats', {})
        print(f"\n個別最適化効果:")
        for module, data in stats.items():
            load_time = data.get('import_time_ms', 0)
            print(f"  {module}: {load_time:.3f}ms削減")
        
        return {
            'original_time_ms': original_time,
            'optimized_time_ms': optimized_time,
            'time_saved_ms': time_saved,
            'improvement_rate_pct': improvement_rate,
            'target_reduction_ms': target_reduction,
            'achievement_rate_pct': achievement_rate,
            'phase3_success': achievement_rate >= 80.0  # 80%以上で成功判定
        }
    
    return {'error': 'calculation_failed'}

def assess_additional_optimization_needs(comprehensive_effect):
    """追加最適化必要性判定"""
    print("\n=== 追加最適化必要性判定 ===")
    
    if comprehensive_effect.get('error'):
        print("[ERROR] 効果計算エラーのため判定不可")
        return {'needs_additional': True, 'reason': 'calculation_error'}
    
    achievement_rate = comprehensive_effect.get('achievement_rate_pct', 0)
    time_saved = comprehensive_effect.get('time_saved_ms', 0)
    
    print(f"Phase 3達成状況:")
    print(f"  目標達成率: {achievement_rate:.1f}%")
    print(f"  削減時間: {time_saved:.1f}ms")
    
    # 判定ロジック
    if achievement_rate >= 100.0:
        print("[OK] Phase 3目標完全達成 - 追加最適化不要")
        return {
            'needs_additional': False, 
            'status': 'complete',
            'reason': f'目標超過達成({achievement_rate:.1f}%)'
        }
    elif achievement_rate >= 80.0:
        print("[OK] Phase 3目標概ね達成 - 追加最適化任意")
        return {
            'needs_additional': False, 
            'status': 'acceptable',
            'reason': f'十分な達成率({achievement_rate:.1f}%)'
        }
    elif achievement_rate >= 50.0:
        print("[WARNING] Phase 3部分達成 - 軽微な追加最適化推奨")
        return {
            'needs_additional': True, 
            'status': 'partial',
            'reason': f'中程度達成率({achievement_rate:.1f}%) - 軽微調整',
            'suggestions': ['型注釈最適化', 'キャッシュ機構改善', 'インポート順序最適化']
        }
    else:
        print("[ERROR] Phase 3目標未達 - 本格的追加最適化必要")
        return {
            'needs_additional': True, 
            'status': 'insufficient',
            'reason': f'低達成率({achievement_rate:.1f}%) - 本格対応必要',
            'suggestions': ['追加ライブラリ遅延化', 'アーキテクチャ見直し', 'ボトルネック再調査']
        }

def main():
    """Phase 3-3統合効果測定・追加最適化判定メイン実行"""
    print("Phase 3-3 統合効果測定・追加最適化判定開始")
    print("=" * 70)
    
    # 1. 従来システム性能測定
    original_result = measure_full_system_performance_original()
    
    # 2. 最適化システム性能測定
    optimized_result = measure_full_system_performance_optimized()
    
    # 3. 統合効果計算
    comprehensive_effect = calculate_phase3_comprehensive_effect(original_result, optimized_result)
    
    # 4. 追加最適化必要性判定
    optimization_assessment = assess_additional_optimization_needs(comprehensive_effect)
    
    # 5. 最終サマリー
    print("\n" + "=" * 70)
    print("Phase 3-3 最終判定サマリー")
    print("=" * 70)
    
    if not comprehensive_effect.get('error'):
        print(f"[OK] Phase 3統合最適化結果:")
        print(f"   削減時間: {comprehensive_effect['time_saved_ms']:.1f}ms")
        print(f"   改善率: {comprehensive_effect['improvement_rate_pct']:.1f}%")
        print(f"   目標達成率: {comprehensive_effect['achievement_rate_pct']:.1f}%")
        print(f"   Phase 3成功: {'[OK] YES' if comprehensive_effect['phase3_success'] else '[ERROR] NO'}")
    
    print(f"\n[OK] 追加最適化判定:")
    print(f"   必要性: {'❗ 必要' if optimization_assessment['needs_additional'] else '[OK] 不要'}")
    print(f"   状況: {optimization_assessment['status']}")
    print(f"   理由: {optimization_assessment['reason']}")
    
    if 'suggestions' in optimization_assessment:
        print(f"   推奨対応:")
        for suggestion in optimization_assessment['suggestions']:
            print(f"     - {suggestion}")
    
    # 次段階への提言
    print(f"\n[LIST] 次段階提言:")
    if optimization_assessment['status'] == 'complete':
        print("   • TODO-PERF-003 Phase 3完了宣言")
        print("   • 次のボトルネック特定・Phase 4計画立案")
        print("   • 性能監視・維持管理体制構築")
    elif optimization_assessment['status'] == 'acceptable':
        print("   • TODO-PERF-003 Phase 3実質完了")
        print("   • ドキュメント更新・効果記録")
        print("   • 次の大きなボトルネック特定")
    else:
        print("   • Phase 3追加対応実施")
        print("   • 根本原因再調査")
        print("   • 段階的改善継続")
    
    print("\nPhase 3-3統合効果測定完了")

if __name__ == "__main__":
    main()