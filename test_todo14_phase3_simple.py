#!/usr/bin/env python3
"""
TODO #14 Phase 3: 簡易統合テスト

Phase 3の自動データ供給機能の基本動作確認
"""

import sys
import os
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必要モジュールのインポート
try:
    from config.multi_strategy_manager import MultiStrategyManager
    from real_market_data_fetcher import fetch_strategy_required_data
    print("✅ 必要モジュールのインポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)

def create_sample_stock_data():
    """テスト用の株価データ作成"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    data = {
        'Date': dates,
        'Open': [100 + i for i in range(20)],
        'High': [105 + i for i in range(20)],
        'Low': [95 + i for i in range(20)],
        'Close': [102 + i for i in range(20)],
        'Volume': [1000000 + i*10000 for i in range(20)]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def test_fetch_strategy_required_data():
    """fetch_strategy_required_data関数のテスト"""
    print("\n🔧 fetch_strategy_required_data関数テスト")
    
    stock_data = create_sample_stock_data()
    
    try:
        # VWAPBreakoutStrategy用データ取得テスト
        print("📋 VWAPBreakoutStrategy用データ取得...")
        vwap_data = fetch_strategy_required_data('VWAPBreakoutStrategy', stock_data)
        
        if 'index_data' in vwap_data and vwap_data['index_data'] is not None:
            print(f"✅ VWAPBreakoutStrategy用index_data取得成功: {len(vwap_data['index_data'])} rows")
            print(f"   Columns: {list(vwap_data['index_data'].columns)}")
        else:
            print("❌ VWAPBreakoutStrategy用index_data取得失敗")
            return False
        
        # OpeningGapStrategy用データ取得テスト
        print("📋 OpeningGapStrategy用データ取得...")
        gap_data = fetch_strategy_required_data('OpeningGapStrategy', stock_data)
        
        if 'dow_data' in gap_data and gap_data['dow_data'] is not None:
            print(f"✅ OpeningGapStrategy用dow_data取得成功: {len(gap_data['dow_data'])} rows")
            print(f"   Columns: {list(gap_data['dow_data'].columns)}")
        else:
            print("❌ OpeningGapStrategy用dow_data取得失敗")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ fetch_strategy_required_dataテストエラー: {str(e)}")
        return False

def test_multi_strategy_manager_initialization():
    """MultiStrategyManagerの初期化テスト"""
    print("\n🔧 MultiStrategyManager初期化テスト")
    
    try:
        manager = MultiStrategyManager()
        print("✅ MultiStrategyManager基本初期化成功")
        
        # システム初期化の実行
        print("📋 システム初期化実行...")
        init_success = manager.initialize_systems()
        
        if init_success:
            print("✅ システム初期化成功")
            
            # 初期化確認
            if hasattr(manager, 'strategy_registry') and manager.strategy_registry:
                print(f"✅ Strategy registry確認: {len(manager.strategy_registry)} strategies")
                return True
            else:
                print("⚠️  Strategy registryは空ですが、初期化は成功")
                return True  # 空でも初期化成功とみなす
        else:
            print("❌ システム初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ MultiStrategyManager初期化エラー: {str(e)}")
        return False

def main():
    """簡易統合テスト実行"""
    print("=" * 60)
    print("TODO #14 Phase 3: 簡易統合テスト")
    print("=" * 60)
    
    results = []
    
    # Step 1: fetch_strategy_required_data関数テスト
    print("📋 Step 1: fetch_strategy_required_data関数テスト")
    results.append(test_fetch_strategy_required_data())
    
    # Step 2: MultiStrategyManager初期化テスト
    print("\n📋 Step 2: MultiStrategyManager初期化テスト")
    results.append(test_multi_strategy_manager_initialization())
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 簡易統合テスト結果サマリー")
    print("=" * 60)
    
    test_names = [
        "fetch_strategy_required_data関数テスト",
        "MultiStrategyManager初期化テスト"
    ]
    
    success_count = 0
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
        if result:
            success_count += 1
    
    overall_success = success_count == len(results)
    print(f"\n📊 総合結果: {success_count}/{len(results)} tests passed")
    
    if overall_success:
        print("🎉 Phase 3簡易統合テスト成功 - 基本機能正常動作")
        print("📋 次ステップ: より詳細な統合テストまたはPhase 4実装")
    else:
        print("⚠️  Phase 3簡易統合テスト失敗 - 基本機能に問題あり")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)