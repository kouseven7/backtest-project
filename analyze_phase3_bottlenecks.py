"""
Phase 3最適化候補特定スクリプト
DSSMS実行時間6780msのボトルネック分析

作成: 2025年10月2日
目的: データ取得・戦略実行・Excel出力の詳細分析
"""

import time
import sys
import os
from datetime import datetime, timedelta

def analyze_data_fetching_bottleneck():
    """データ取得ボトルネック分析"""
    print("=== Phase 3最適化候補分析 ===")
    print("📊 1. データ取得コンポーネント分析")
    
    # yfinance インポート時間
    start = time.perf_counter()
    try:
        import yfinance as yf
        yf_time = (time.perf_counter() - start) * 1000
        print(f"   yfinance インポート: {yf_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ yfinance エラー: {e}")
    
    # データフェッチャー類
    start = time.perf_counter()
    try:
        from data_fetcher import DataFetcher
        fetcher_time = (time.perf_counter() - start) * 1000
        print(f"   DataFetcher インポート: {fetcher_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ DataFetcher エラー: {e}")
    
    # データ処理
    start = time.perf_counter()
    try:
        from data_processor import DataProcessor  
        processor_time = (time.perf_counter() - start) * 1000
        print(f"   DataProcessor インポート: {processor_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ DataProcessor エラー: {e}")

def analyze_strategy_execution_bottleneck():
    """戦略実行ボトルネック分析"""
    print("\n📈 2. 戦略実行コンポーネント分析")
    
    # 主要戦略
    strategies = [
        ('strategies.vwap_breakout_strategy', 'VWAPBreakoutStrategy'),
        ('strategies.bollinger_bands_strategy', 'BollingerBandsStrategy'),
        ('strategies.rsi_strategy', 'RSIStrategy'),
        ('strategies.moving_average_strategy', 'MovingAverageStrategy')
    ]
    
    total_strategy_time = 0
    for module_name, class_name in strategies:
        start = time.perf_counter()
        try:
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {class_name}: {load_time:.1f}ms")
            total_strategy_time += load_time
        except Exception as e:
            print(f"   ❌ {class_name}: エラー - {e}")
    
    print(f"   📊 戦略合計: {total_strategy_time:.1f}ms")

def analyze_excel_output_bottleneck():
    """Excel出力ボトルネック分析"""
    print("\n📄 3. Excel出力コンポーネント分析")
    
    # openpyxl（Excel処理）
    start = time.perf_counter()
    try:
        import openpyxl
        openpyxl_time = (time.perf_counter() - start) * 1000
        print(f"   openpyxl インポート: {openpyxl_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ openpyxl エラー: {e}")
    
    # Excelエクスポーター
    start = time.perf_counter()
    try:
        from output.excel_exporter import ExcelExporter
        exporter_time = (time.perf_counter() - start) * 1000
        print(f"   ExcelExporter インポート: {exporter_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ ExcelExporter エラー: {e}")
    
    # シミュレーションハンドラー
    start = time.perf_counter()
    try:
        from output.simulation_handler import SimulationHandler
        handler_time = (time.perf_counter() - start) * 1000
        print(f"   SimulationHandler インポート: {handler_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ SimulationHandler エラー: {e}")

def analyze_heavy_libraries():
    """重いライブラリ分析"""
    print("\n🔬 4. 重いライブラリインポート分析")
    
    heavy_libs = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('scipy', None)
    ]
    
    total_heavy_time = 0
    for lib_name, alias in heavy_libs:
        start = time.perf_counter()
        try:
            if alias:
                exec(f"import {lib_name} as {alias}")
            else:
                exec(f"import {lib_name}")
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {lib_name}: {load_time:.1f}ms")
            total_heavy_time += load_time
        except Exception as e:
            print(f"   ❌ {lib_name}: エラー - {e}")
    
    print(f"   📊 重いライブラリ合計: {total_heavy_time:.1f}ms")

def analyze_dssms_specific_bottlenecks():
    """DSSMS固有ボトルネック分析"""
    print("\n🎯 5. DSSMS固有コンポーネント分析")
    
    dssms_components = [
        ('src.dssms.dssms_backtester', 'DSSMSBacktester'),
        ('src.dssms.advanced_ranking_system.advanced_ranking_engine', 'AdvancedRankingEngine'),
        ('src.dssms.hierarchical_ranking_system', 'HierarchicalRankingSystem'),
        ('src.dssms.integration_bridge', 'IntegrationBridge')
    ]
    
    total_dssms_time = 0
    for module_name, class_name in dssms_components:
        start = time.perf_counter()
        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            load_time = (time.perf_counter() - start) * 1000
            print(f"   {class_name}: {load_time:.1f}ms")
            total_dssms_time += load_time
        except Exception as e:
            print(f"   ❌ {class_name}: エラー - {e}")
    
    print(f"   📊 DSSMS固有合計: {total_dssms_time:.1f}ms")

def calculate_optimization_priorities():
    """最適化優先度計算"""
    print("\n📋 6. Phase 3最適化優先度まとめ")
    print("⚠️ 注意: 実際の実行時間6780msとインポート時間は異なります")
    print("   実行時間 = インポート時間 + 初期化時間 + 処理実行時間")
    print()
    print("🎯 推奨Phase 3最適化順序:")
    print("   1. 【高優先】重いライブラリ遅延ローディング（pandas, numpy等）")
    print("   2. 【高優先】DSSMS固有コンポーネント最適化")
    print("   3. 【中優先】戦略実行処理の軽量化")
    print("   4. 【中優先】データ取得キャッシュ最適化")
    print("   5. 【低優先】Excel出力処理最適化")

def main():
    """メイン実行"""
    try:
        analyze_data_fetching_bottleneck()
        analyze_strategy_execution_bottleneck()
        analyze_excel_output_bottleneck() 
        analyze_heavy_libraries()
        analyze_dssms_specific_bottlenecks()
        calculate_optimization_priorities()
        
        print("\n=== Phase 3最適化候補分析完了 ===")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()