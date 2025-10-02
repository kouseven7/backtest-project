"""
TODO-PERF-004 残りボトルネック詳細分析スクリプト
DSSMSIntegratedBacktesterの各コンポーネント別インポート時間測定

作成: 2025年10月2日
目的: 2826.5msの真のボトルネック特定と最適化優先度決定
"""

import time
import sys
import os

def measure_individual_imports():
    """個別インポート時間詳細測定"""
    print("=== TODO-PERF-004 ボトルネック詳細分析 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # 基本ライブラリ
    print("1. 基本ライブラリインポート時間")
    basic_libs = [
        ('datetime', 'from datetime import datetime, timedelta'),
        ('typing', 'from typing import Dict, List, Any, Optional, Tuple'),
        ('logging', 'import logging'),
        ('time', 'import time'),
        ('pathlib', 'from pathlib import Path'),
        ('json', 'import json'),
        ('argparse', 'import argparse')
    ]
    
    basic_total = 0
    for lib_name, import_cmd in basic_libs:
        start = time.perf_counter()
        try:
            exec(import_cmd)
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {lib_name}: {load_time:.1f}ms")
            basic_total += load_time
        except Exception as e:
            print(f"   ❌ {lib_name}: エラー - {e}")
    
    print(f"   📊 基本ライブラリ合計: {basic_total:.1f}ms")
    print()
    
    # lazy_loader自体
    print("2. lazy_loader インポート時間")
    start = time.perf_counter()
    try:
        from src.dssms.lazy_loader import DSSMSLazyModules, lazy_import, lazy_class_import
        lazy_loader_time = (time.perf_counter() - start) * 1000
        print(f"   lazy_loader: {lazy_loader_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ lazy_loader: エラー - {e}")
        lazy_loader_time = 0
    print()
    
    # 重いライブラリの実際の使用確認
    print("3. 重いライブラリ遅延ロード確認")
    heavy_libs = [
        ('yfinance', 'import yfinance as yf'),
        ('openpyxl', 'import openpyxl'),
        ('pandas', 'import pandas as pd'),
        ('numpy', 'import numpy as np')
    ]
    
    heavy_total = 0
    for lib_name, import_cmd in heavy_libs:
        start = time.perf_counter()
        try:
            exec(import_cmd)
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {lib_name}: {load_time:.1f}ms")
            if load_time > 100:  # 100ms以上が重いライブラリ
                print(f"      🔴 重いライブラリ検出: {load_time:.1f}ms")
            heavy_total += load_time
        except Exception as e:
            print(f"   ❌ {lib_name}: エラー - {e}")
    
    print(f"   📊 重いライブラリ合計: {heavy_total:.1f}ms")
    print()
    
    return basic_total, lazy_loader_time, heavy_total

def measure_dssms_components():
    """DSSMSコンポーネント別測定"""
    print("4. DSSMSコンポーネント別インポート時間")
    
    dssms_components = [
        ('dssms_backtester_v3', 'DSSBacktesterV3'),
        ('src.dssms.advanced_ranking_system.advanced_ranking_engine', 'AdvancedRankingEngine'),
        ('config.risk_management', 'RiskManagement'),
        ('src.dssms.data_cache_manager', 'DataCacheManager'),
        ('src.dssms.performance_tracker', 'PerformanceTracker'),
        ('src.dssms.dssms_excel_exporter', 'DSSMSExcelExporter'),
        ('src.dssms.dssms_report_generator', 'DSSMSReportGenerator'),
        ('src.dssms.nikkei225_screener', 'Nikkei225Screener')
    ]
    
    component_total = 0
    bottlenecks = []
    
    for module_name, class_name in dssms_components:
        start = time.perf_counter()
        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {class_name}: {load_time:.1f}ms")
            
            if load_time > 50:  # 50ms以上をボトルネック候補
                bottlenecks.append((class_name, load_time))
                print(f"      ⚠️ ボトルネック候補: {load_time:.1f}ms")
            
            component_total += load_time
        except Exception as e:
            print(f"   ❌ {class_name}: エラー - {e}")
    
    print(f"   📊 DSSMSコンポーネント合計: {component_total:.1f}ms")
    print()
    
    if bottlenecks:
        print("   🎯 最適化優先度（ボトルネック順）:")
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        for i, (component, load_time) in enumerate(bottlenecks[:5], 1):
            print(f"      {i}. {component}: {load_time:.1f}ms")
    
    return component_total, bottlenecks

def measure_dssms_integrated_main():
    """DSSMSIntegratedBacktesterクラス定義時間"""
    print("\n5. DSSMSIntegratedBacktesterクラス定義時間")
    
    start = time.perf_counter()
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        class_def_time = (time.perf_counter() - start) * 1000
        print(f"   DSSMSIntegratedBacktester: {class_def_time:.1f}ms")
        
        return class_def_time
    except Exception as e:
        print(f"   ❌ DSSMSIntegratedBacktester: エラー - {e}")
        return 0

def calculate_optimization_strategy(basic_total, lazy_loader_time, heavy_total, 
                                  component_total, bottlenecks, class_def_time):
    """最適化戦略計算"""
    print("\n=== 最適化戦略分析 ===")
    
    total_measured = basic_total + lazy_loader_time + heavy_total + component_total
    print(f"📊 測定合計時間: {total_measured:.1f}ms")
    print(f"📊 実際のDSSMSIntegratedBacktester: {class_def_time:.1f}ms")
    print(f"📊 差分（未特定ボトルネック）: {class_def_time - total_measured:.1f}ms")
    
    print("\n🎯 最適化優先順位:")
    
    # 優先度計算
    strategies = []
    
    if heavy_total > 500:
        strategies.append(("重いライブラリ遅延ローディング", heavy_total, "最高"))
    
    for component, load_time in bottlenecks:
        if load_time > 100:
            strategies.append((f"{component}軽量化", load_time, "高"))
    
    if class_def_time - total_measured > 500:
        strategies.append(("未特定ボトルネック調査", class_def_time - total_measured, "高"))
    
    for i, (strategy, time_saving, priority) in enumerate(strategies, 1):
        print(f"   {i}. 【{priority}優先】{strategy}: {time_saving:.1f}ms削減可能")
    
    # 目標達成可能性
    total_savings = sum(saving for _, saving, _ in strategies)
    final_time = class_def_time - total_savings
    
    print(f"\n📈 最適化効果予測:")
    print(f"   現在: {class_def_time:.1f}ms")
    print(f"   削減可能: {total_savings:.1f}ms")
    print(f"   予測最終: {final_time:.1f}ms")
    print(f"   目標1.2ms: {'✅ 達成可能' if final_time <= 1.2 else '❌ 追加対策必要'}")

def main():
    """メイン実行"""
    try:
        basic_total, lazy_loader_time, heavy_total = measure_individual_imports()
        component_total, bottlenecks = measure_dssms_components()
        class_def_time = measure_dssms_integrated_main()
        
        calculate_optimization_strategy(basic_total, lazy_loader_time, heavy_total,
                                      component_total, bottlenecks, class_def_time)
        
        print("\n📋 次のステップ:")
        print("1. 最高優先度の最適化実施")
        print("2. 段階的効果測定")
        print("3. 目標1.2ms達成まで継続")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()