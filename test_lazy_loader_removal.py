"""
lazy_loader除去効果測定スクリプト
最小限修正後の実行時間測定

作成: 2025年10月2日
目的: lazy_loader除去による効果の即時確認
"""

import time

def test_lazy_loader_removal():
    """lazy_loader除去効果テスト"""
    print("=== lazy_loader除去効果測定 ===")
    print(f"実行日時: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # 1. 基本的なインポート（lazy_loader除去前のベースライン）
    print("1. 基本ライブラリインポート時間（ベースライン）")
    start = time.perf_counter()
    import sys, os, logging, json
    basic_time = (time.perf_counter() - start) * 1000
    print(f"   基本ライブラリ: {basic_time:.1f}ms")
    
    # 2. SymbolSwitchManager直接インポート（軽量版優先）
    print("\n2. SymbolSwitchManager直接インポート（軽量版優先）")
    start = time.perf_counter()
    try:
        from src.dssms.symbol_switch_manager_fast import SymbolSwitchManagerFast as SymbolSwitchManager
        direct_import_time = (time.perf_counter() - start) * 1000
        print(f"   SymbolSwitchManagerFast: {direct_import_time:.1f}ms")
        used_version = "Fast"
    except ImportError:
        try:
            from src.dssms.symbol_switch_manager import SymbolSwitchManager
            direct_import_time = (time.perf_counter() - start) * 1000
            print(f"   SymbolSwitchManager: {direct_import_time:.1f}ms")
            used_version = "Original"
        except ImportError as e:
            print(f"   ❌ SymbolSwitchManagerインポートエラー: {e}")
            direct_import_time = 0
            used_version = "None"
    
    # 3. 修正版DSSMSIntegratedBacktesterのインポート試行
    print("\n3. 修正版DSSMSIntegratedBacktesterインポート")
    start = time.perf_counter()
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        modified_import_time = (time.perf_counter() - start) * 1000
        print(f"   修正版DSSMSIntegratedBacktester: {modified_import_time:.1f}ms")
        
        # 目標との比較
        target_time = 1.2
        if modified_import_time <= target_time:
            print(f"   ✅ 目標達成! ({target_time}ms以下)")
            improvement_status = "成功"
        else:
            remaining = modified_import_time - target_time
            print(f"   ⚠️ 目標未達成: 残り{remaining:.1f}ms短縮必要")
            improvement_status = "部分的"
        
        # 従来との比較（2826.5msベースライン）
        baseline = 2826.5
        improvement = baseline - modified_import_time
        improvement_rate = (improvement / baseline) * 100
        print(f"   📊 改善効果: {improvement:.1f}ms削減 ({improvement_rate:.1f}%改善)")
        
    except Exception as e:
        print(f"   ❌ 修正版インポートエラー: {e}")
        modified_import_time = 0
        improvement_status = "失敗"
        
        # エラー詳細
        import traceback
        print("   エラー詳細:")
        traceback.print_exc()
    
    # 4. 簡易初期化テスト
    if modified_import_time > 0:
        print("\n4. 簡易初期化テスト")
        start = time.perf_counter()
        try:
            config = {
                'symbol_switch': {
                    'switch_cost_rate': 0.001,
                    'min_holding_days': 1
                }
            }
            backtester = DSSMSIntegratedBacktester(config)
            init_time = (time.perf_counter() - start) * 1000
            print(f"   初期化時間: {init_time:.1f}ms")
            print(f"   使用中SymbolSwitchManager: {used_version}")
            
        except Exception as e:
            print(f"   ❌ 初期化エラー: {e}")
    
    return {
        'basic_time': basic_time,
        'direct_import_time': direct_import_time,
        'modified_import_time': modified_import_time,
        'improvement_status': improvement_status,
        'used_version': used_version
    }

def main():
    """メイン実行"""
    try:
        results = test_lazy_loader_removal()
        
        print("\n=== lazy_loader除去効果サマリー ===")
        print(f"📊 基本ライブラリ: {results['basic_time']:.1f}ms")
        print(f"📊 SymbolSwitchManager直接: {results['direct_import_time']:.1f}ms")
        print(f"📊 修正版DSSMSIntegratedBacktester: {results['modified_import_time']:.1f}ms")
        print(f"📊 使用バージョン: {results['used_version']}")
        print(f"🎯 改善状況: {results['improvement_status']}")
        
        if results['improvement_status'] == "成功":
            print("\n✅ lazy_loader除去による大幅最適化成功！")
        elif results['improvement_status'] == "部分的":
            print("\n⚠️ 部分的改善 - 追加最適化が必要")
        else:
            print("\n❌ 除去失敗 - 問題修正が必要")
        
        print("\n📋 次のステップ:")
        if results['improvement_status'] != "成功":
            print("1. エラー修正・コンパイル問題解決")
            print("2. lazy_class_import完全除去")
            print("3. 再測定・効果確認")
        else:
            print("1. 成功確認・ドキュメント更新")
            print("2. Phase 2目標達成報告")
            print("3. Phase 3準備")
        
    except Exception as e:
        print(f"測定エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()