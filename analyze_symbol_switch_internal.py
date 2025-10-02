"""
symbol_switch_manager内部処理プロファイリング
TODO-PERF-001 Phase 2: 詳細分析
"""

import time
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.append(project_root)

def analyze_symbol_switch_manager_internal():
    """SymbolSwitchManager内部処理の詳細分析"""
    print("=== SymbolSwitchManager内部処理分析 ===")
    
    # 段階的インポート時間測定
    stages = [
        ("basic imports", lambda: __import_basic()),
        ("config creation", lambda: __create_config()),
        ("SymbolSwitchManager import", lambda: __import_class()),
        ("SymbolSwitchManager instantiation", lambda: __instantiate_class()),
        ("method calls", lambda: __call_methods())
    ]
    
    times = {}
    
    for stage_name, stage_func in stages:
        start_time = time.perf_counter()
        try:
            result = stage_func()
            stage_time = (time.perf_counter() - start_time) * 1000
            times[stage_name] = stage_time
            print(f"✅ {stage_name}: {stage_time:.1f}ms")
        except Exception as e:
            stage_time = (time.perf_counter() - start_time) * 1000
            times[stage_name] = stage_time
            print(f"❌ {stage_name}: {stage_time:.1f}ms (Error: {e})")
    
    # 結果分析
    total_time = sum(times.values())
    print(f"\n=== 分析結果 ===")
    print(f"合計時間: {total_time:.1f}ms")
    
    for stage, time_ms in sorted(times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_ms / total_time) * 100 if total_time > 0 else 0
        print(f"{time_ms:6.1f}ms ({percentage:4.1f}%) - {stage}")
    
    return times

def __import_basic():
    """基本モジュールインポート"""
    from datetime import datetime, timedelta
    from typing import Dict, List, Any, Optional
    import logging
    return True

def __create_config():
    """設定作成"""
    config = {
        'switch_management': {
            'switch_cost_rate': 0.001,
            'min_holding_days': 1,
            'max_switches_per_month': 10,
            'cost_threshold': 0.001
        }
    }
    return config

def __import_class():
    """SymbolSwitchManagerクラスインポート"""
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    return SymbolSwitchManager

def __instantiate_class():
    """SymbolSwitchManagerインスタンス化"""
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    config = __create_config()
    return SymbolSwitchManager(config)

def __call_methods():
    """メソッド呼び出しテスト"""
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
    from datetime import datetime
    
    config = __create_config()
    manager = SymbolSwitchManager(config)
    
    # 軽量メソッド呼び出し
    stats = manager.get_switch_statistics()
    return stats

if __name__ == "__main__":
    result_times = analyze_symbol_switch_manager_internal()
    
    # 最適化提案
    print(f"\n=== TODO-PERF-001 Phase 2 最適化提案 ===")
    
    heaviest_stage = max(result_times.items(), key=lambda x: x[1])
    print(f"最重要最適化対象: {heaviest_stage[0]} ({heaviest_stage[1]:.1f}ms)")
    
    if heaviest_stage[1] > 1000:
        print("⚠️ 1秒以上の重い処理発見！詳細調査必要")
        print("推奨対策:")
        print("  1. クラス定義の軽量化")
        print("  2. 初期化処理の遅延化")
        print("  3. モジュール依存の最小化")
    else:
        print("ℹ️ 個別処理は軽量、統合時の累積効果が問題")