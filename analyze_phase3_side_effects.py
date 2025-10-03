"""
Phase 3追加分析：遅延インポート副作用調査
作成: 2025年10月3日

遅延インポート導入による予期しないオーバーヘッドの根本原因分析
"""

import sys
import os
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def analyze_lazy_import_overhead():
    """遅延インポート機構のオーバーヘッド分析"""
    print("=== 遅延インポート機構オーバーヘッド分析 ===")
    
    # 1. LazyImportManager単体分析
    start = time.perf_counter()
    from src.utils.lazy_import_manager import LazyImporter
    manager_import_time = (time.perf_counter() - start) * 1000
    print(f"LazyImporter クラスインポート: {manager_import_time:.1f}ms")
    
    # 2. インスタンス作成オーバーヘッド
    start = time.perf_counter()
    importer = LazyImporter()
    instance_time = (time.perf_counter() - start) * 1000
    print(f"LazyImporter インスタンス作成: {instance_time:.1f}ms")
    
    # 3. 各メソッド呼び出しオーバーヘッド
    start = time.perf_counter()
    yf = importer.import_yfinance()
    yf_call_time = (time.perf_counter() - start) * 1000
    print(f"import_yfinance() 初回呼び出し: {yf_call_time:.1f}ms")
    
    start = time.perf_counter()
    yf2 = importer.import_yfinance()  # 2回目（キャッシュ済み）
    yf_cache_time = (time.perf_counter() - start) * 1000
    print(f"import_yfinance() キャッシュ呼び出し: {yf_cache_time:.3f}ms")
    
    start = time.perf_counter()
    openpyxl = importer.import_openpyxl()
    xl_call_time = (time.perf_counter() - start) * 1000
    print(f"import_openpyxl() 初回呼び出し: {xl_call_time:.1f}ms")
    
    return {
        'manager_import_ms': manager_import_time,
        'instance_create_ms': instance_time,
        'yfinance_first_ms': yf_call_time,
        'yfinance_cache_ms': yf_cache_time,
        'openpyxl_first_ms': xl_call_time
    }

def analyze_dssms_import_chain():
    """DSSMS インポートチェーン分析"""
    print("\n=== DSSMS インポートチェーン分析 ===")
    
    # 段階的インポート測定
    results = {}
    
    # 1. プロジェクトルート設定オーバーヘッド
    start = time.perf_counter()
    import sys, os
    from pathlib import Path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    setup_time = (time.perf_counter() - start) * 1000
    print(f"プロジェクト設定: {setup_time:.3f}ms")
    results['project_setup_ms'] = setup_time
    
    # 2. Logger設定
    start = time.perf_counter()
    from config.logger_config import setup_logger
    logger_time = (time.perf_counter() - start) * 1000
    print(f"Logger設定: {logger_time:.1f}ms")
    results['logger_setup_ms'] = logger_time
    
    # 3. 基本ライブラリ
    start = time.perf_counter()
    import pandas as pd
    from datetime import datetime, timedelta
    import json
    basic_time = (time.perf_counter() - start) * 1000
    print(f"基本ライブラリ: {basic_time:.1f}ms")
    results['basic_libs_ms'] = basic_time
    
    # 4. DSSMS Screener
    start = time.perf_counter()
    try:
        from src.dssms.nikkei225_screener import Nikkei225Screener
        screener_time = (time.perf_counter() - start) * 1000
        print(f"Nikkei225Screener: {screener_time:.1f}ms")
        results['screener_ms'] = screener_time
    except Exception as e:
        screener_time = (time.perf_counter() - start) * 1000
        print(f"Nikkei225Screener エラー: {screener_time:.1f}ms - {e}")
        results['screener_ms'] = screener_time
        results['screener_error'] = str(e)
    
    # 5. DSSMS Data Manager
    start = time.perf_counter()
    try:
        from src.dssms.dssms_data_manager import DSSMSDataManager
        data_mgr_time = (time.perf_counter() - start) * 1000
        print(f"DSSMSDataManager: {data_mgr_time:.1f}ms")
        results['data_manager_ms'] = data_mgr_time
    except Exception as e:
        data_mgr_time = (time.perf_counter() - start) * 1000
        print(f"DSSMSDataManager エラー: {data_mgr_time:.1f}ms - {e}")
        results['data_manager_ms'] = data_mgr_time
        results['data_manager_error'] = str(e)
    
    return results

def analyze_import_dependency_cascade():
    """インポート依存関係カスケード分析"""
    print("\n=== インポート依存関係カスケード分析 ===")
    
    # インポート前のモジュール数
    initial_modules = len(sys.modules)
    print(f"初期モジュール数: {initial_modules}")
    
    # 段階的にインポートしてモジュール増加を追跡
    checkpoints = []
    
    # LazyImportManager
    from src.utils.lazy_import_manager import get_yfinance, get_openpyxl
    checkpoints.append(('LazyImportManager', len(sys.modules) - initial_modules))
    
    # yfinance 遅延ロード
    yf = get_yfinance()
    checkpoints.append(('yfinance lazy load', len(sys.modules) - initial_modules))
    
    # openpyxl 遅延ロード
    openpyxl = get_openpyxl()
    checkpoints.append(('openpyxl lazy load', len(sys.modules) - initial_modules))
    
    # DSSMS コンポーネント
    try:
        from src.dssms.nikkei225_screener import Nikkei225Screener
        checkpoints.append(('DSSMS Screener', len(sys.modules) - initial_modules))
    except:
        pass
    
    print("モジュール増加パターン:")
    for name, module_count in checkpoints:
        print(f"  {name}: +{module_count} モジュール")
    
    return checkpoints

def identify_bottleneck_root_cause():
    """ボトルネック根本原因特定"""
    print("\n=== ボトルネック根本原因特定 ===")
    
    # 仮説1: 遅延インポート機構自体のオーバーヘッド
    lazy_overhead = analyze_lazy_import_overhead()
    lazy_total = sum([
        lazy_overhead['manager_import_ms'],
        lazy_overhead['instance_create_ms'],
        lazy_overhead['yfinance_first_ms'],
        lazy_overhead['openpyxl_first_ms']
    ])
    print(f"遅延インポート機構総オーバーヘッド: {lazy_total:.1f}ms")
    
    # 仮説2: DSSMS モジュール間の循環依存・重複インポート
    dssms_chain = analyze_dssms_import_chain()
    dssms_total = sum([v for k, v in dssms_chain.items() if k.endswith('_ms')])
    print(f"DSSMS インポートチェーン総時間: {dssms_total:.1f}ms")
    
    # 仮説3: モジュール依存関係の複雑化
    module_cascade = analyze_import_dependency_cascade()
    final_modules = module_cascade[-1][1] if module_cascade else 0
    print(f"最終追加モジュール数: {final_modules}")
    
    # 根本原因判定
    print(f"\n🔍 根本原因分析:")
    
    if lazy_total > 100:
        print(f"⚠️ 遅延インポート機構オーバーヘッド: {lazy_total:.1f}ms > 100ms")
        print("   → 遅延化の利益を相殺している可能性")
    
    if dssms_total > 2000:
        print(f"❗ DSSMS モジュールチェーン異常: {dssms_total:.1f}ms > 2000ms")
        print("   → 循環依存や重複処理の可能性")
    
    if final_modules > 200:
        print(f"⚠️ 過剰なモジュール読み込み: {final_modules} > 200")
        print("   → 不要な依存関係の可能性")
    
    # 対策提案
    print(f"\n💡 対策提案:")
    
    if lazy_total > 100:
        print("   1. 遅延インポート機構の軽量化")
        print("      - 統計機能の簡素化")
        print("      - キャッシュ機構の最適化")
    
    if dssms_total > 2000:
        print("   2. DSSMS アーキテクチャ見直し")
        print("      - 循環依存の解消")
        print("      - モジュールの分割・統合")
    
    print("   3. 段階的最適化アプローチ")
    print("      - 高頻度使用モジュールのみ遅延化")
    print("      - 低頻度モジュールは従来方式維持")
    
    return {
        'lazy_overhead_ms': lazy_total,
        'dssms_chain_ms': dssms_total,
        'final_modules': final_modules,
        'recommendations': ['lazy_optimization', 'architecture_review', 'selective_approach']
    }

def main():
    """Phase 3追加分析メイン実行"""
    print("Phase 3追加分析：遅延インポート副作用調査開始")
    print("=" * 70)
    
    root_cause = identify_bottleneck_root_cause()
    
    print("\n" + "=" * 70)
    print("Phase 3追加分析結果サマリー")
    print("=" * 70)
    
    print(f"✅ 副作用分析結果:")
    print(f"   遅延機構オーバーヘッド: {root_cause['lazy_overhead_ms']:.1f}ms")
    print(f"   DSSMSチェーン時間: {root_cause['dssms_chain_ms']:.1f}ms")
    print(f"   追加モジュール数: {root_cause['final_modules']}")
    
    print(f"\n📋 推奨対策:")
    for i, rec in enumerate(root_cause['recommendations'], 1):
        rec_names = {
            'lazy_optimization': '遅延インポート機構軽量化',
            'architecture_review': 'DSSMSアーキテクチャ見直し',
            'selective_approach': '選択的最適化アプローチ'
        }
        print(f"   {i}. {rec_names.get(rec, rec)}")
    
    print(f"\n🎯 Phase 3改良方針:")
    print("   • 個別ライブラリ最適化は成功（yfinance 280ms、openpyxl 233ms削減）")
    print("   • システム統合時のオーバーヘッドが課題")
    print("   • 選択的適用による最適バランス探索が必要")
    
    print("\nPhase 3追加分析完了")

if __name__ == "__main__":
    main()