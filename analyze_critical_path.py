"""
TODO-PERF-001 最終段階: Critical Path分析による最適化
残り604ms最適化のための深度分析
"""

import time
import sys
import os
from typing import Dict, Any
import importlib.util

def analyze_module_import_times():
    """個別モジュールのインポート時間分析"""
    print("=== Critical Path分析: 個別モジュールインポート時間 ===")
    
    # 重要モジュールのインポート時間測定
    critical_modules = [
        'src.dssms.symbol_switch_manager',
        'src.dssms.data_cache_manager', 
        'src.dssms.performance_tracker',
        'src.dssms.dssms_excel_exporter',
        'src.dssms.dssms_report_generator',
        'src.dssms.nikkei225_screener',
        'src.dssms.lazy_loader',
        'config.logger_config'
    ]
    
    import_times = {}
    
    for module_name in critical_modules:
        # 各モジュールを個別にインポート測定
        start_time = time.perf_counter()
        try:
            # sys.modulesから削除して再インポート強制
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            module = importlib.import_module(module_name)
            import_time = (time.perf_counter() - start_time) * 1000
            import_times[module_name] = import_time
            print(f"[OK] {module_name}: {import_time:.1f}ms")
            
        except ImportError as e:
            import_time = 0
            import_times[module_name] = import_time
            print(f"[WARNING]  {module_name}: FAILED ({e})")
        except Exception as e:
            import_time = 0
            import_times[module_name] = import_time
            print(f"[ERROR] {module_name}: ERROR ({e})")
    
    # 結果ソート
    sorted_imports = sorted(import_times.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== 重いモジュールランキング ===")
    total_time = sum(import_times.values())
    for module, time_ms in sorted_imports:
        percentage = (time_ms / total_time) * 100 if total_time > 0 else 0
        print(f"{time_ms:6.1f}ms ({percentage:4.1f}%) - {module}")
    
    print(f"\n合計測定時間: {total_time:.1f}ms")
    
    # 最適化推奨
    heavy_modules = [m for m, t in sorted_imports if t > 100]  # 100ms以上
    if heavy_modules:
        print(f"\n[LIST] 最適化推奨モジュール (100ms+): {len(heavy_modules)}個")
        for module, time_ms in [(m, import_times[m]) for m in heavy_modules]:
            print(f"  - {module}: {time_ms:.1f}ms")
    
    return import_times

def calculate_optimization_potential():
    """最適化ポテンシャル分析"""
    print("\n=== TODO-PERF-001 最適化ポテンシャル分析 ===")
    
    current_time = 2104  # 現在の初期化時間
    target_time = 1500   # 目標時間
    remaining_optimization = current_time - target_time
    
    # ベースライン改善履歴
    baseline = 2682
    improvements = [
        ('pandas/numpy除去', 404),
        ('DSSMSコンポーネント遅延化', 521),
        ('logger最適化', -158)  # 逆効果
    ]
    
    total_improvement = sum(imp for _, imp in improvements)
    current_improvement_rate = (total_improvement / baseline) * 100
    
    print(f"ベースライン: {baseline}ms")
    print(f"現在値: {current_time}ms")
    print(f"目標値: {target_time}ms")
    print(f"残り最適化必要: {remaining_optimization}ms")
    print(f"現在改善率: {current_improvement_rate:.1f}%")
    
    print("\n改善履歴:")
    for desc, imp in improvements:
        if imp > 0:
            print(f"  [OK] {desc}: -{imp}ms")
        else:
            print(f"  [ERROR] {desc}: +{abs(imp)}ms (逆効果)")
    
    # 最適化戦略提案
    print(f"\n[LIST] 残り{remaining_optimization}ms最適化戦略:")
    print("  1. Import文の条件分岐最適化 (~200ms)")
    print("  2. 重いモジュールの完全遅延化 (~250ms)")  
    print("  3. モジュール依存関係の最適化 (~150ms)")
    print("  4. 不要なグローバル変数初期化削除 (~100ms)")

if __name__ == "__main__":
    # プロジェクトルートをパスに追加
    project_root = os.path.dirname(__file__)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    import_times = analyze_module_import_times()
    calculate_optimization_potential()