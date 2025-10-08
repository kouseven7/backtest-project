#!/usr/bin/env python3
"""
TODO #14 Phase 3: 自動データ供給システム統合テスト

テスト項目:
1. VWAPBreakoutStrategy の index_data 自動供給
2. OpeningGapStrategy の dow_data 自動供給  
3. Phase 1 エラー停止からPhase 3 自動供給への強化確認
4. 手動データ提供時の動作確認

実行方法:
python test_todo14_phase3_integration.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必要モジュールのインポート
try:
    from config.multi_strategy_manager import MultiStrategyManager
    from real_market_data_fetcher import RealMarketDataFetcher, fetch_strategy_required_data
    from config.logger_config import setup_logger
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

def test_phase3_auto_supply_vwap():
    """Phase 3: VWAPBreakoutStrategy 自動データ供給テスト"""
    print("\n🔧 Phase 3テスト: VWAPBreakoutStrategy自動データ供給")
    
    manager = MultiStrategyManager()  # loggerパラメータ削除
    stock_data = create_sample_stock_data()
    
    try:
        # Phase 3: index_data を提供せずに自動供給をテスト
        print("📋 テスト: index_data未提供時の自動供給...")
        strategy_instance = manager.get_strategy_instance(
            strategy_name='VWAPBreakoutStrategy',
            data=stock_data,
            params={'vwap_window': 20, 'breakout_threshold': 0.02}
        )
        
        # 戦略インスタンスの確認
        if hasattr(strategy_instance, 'index_data'):
            print(f"✅ 自動供給成功: index_data有り ({len(strategy_instance.index_data)} rows)")
            print(f"   Index data columns: {list(strategy_instance.index_data.columns)}")
            return True
        else:
            print("❌ 自動供給失敗: index_dataが設定されていません")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3自動供給エラー: {str(e)}")
        return False

def test_phase3_auto_supply_opening_gap():
    """Phase 3: OpeningGapStrategy 自動データ供給テスト"""
    print("\n🔧 Phase 3テスト: OpeningGapStrategy自動データ供給")
    
    manager = MultiStrategyManager()  # loggerパラメータ削除
    stock_data = create_sample_stock_data()
    
    try:
        # Phase 3: dow_data を提供せずに自動供給をテスト
        print("📋 テスト: dow_data未提供時の自動供給...")
        strategy_instance = manager.get_strategy_instance(
            strategy_name='OpeningGapStrategy',
            data=stock_data,
            params={'gap_threshold': 0.02, 'confirmation_period': 5}
        )
        
        # 戦略インスタンスの確認
        if hasattr(strategy_instance, 'dow_data'):
            print(f"✅ 自動供給成功: dow_data有り ({len(strategy_instance.dow_data)} rows)")
            print(f"   Dow data columns: {list(strategy_instance.dow_data.columns)}")
            return True
        else:
            print("❌ 自動供給失敗: dow_dataが設定されていません")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3自動供給エラー: {str(e)}")
        return False

def test_manual_data_provision():
    """手動データ提供時の動作確認テスト"""
    print("\n🔧 手動データ提供テスト")
    
    manager = MultiStrategyManager()  # loggerパラメータ削除
    stock_data = create_sample_stock_data()
    
    # 手動データ作成
    manual_index_data = create_sample_stock_data()
    manual_index_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # N225形式
    
    try:
        print("📋 テスト: 手動でindex_data提供...")
        strategy_instance = manager.get_strategy_instance(
            strategy_name='VWAPBreakoutStrategy',
            data=stock_data,
            params={'vwap_window': 20, 'breakout_threshold': 0.02},
            index_data=manual_index_data
        )
        
        if hasattr(strategy_instance, 'index_data'):
            print(f"✅ 手動データ提供成功: index_data有り ({len(strategy_instance.index_data)} rows)")
            return True
        else:
            print("❌ 手動データ提供失敗")
            return False
            
    except Exception as e:
        print(f"❌ 手動データ提供エラー: {str(e)}")
        return False

def test_real_data_fetcher_direct():
    """RealMarketDataFetcher直接テスト（Phase 2確認）"""
    print("\n🔧 RealMarketDataFetcher直接テスト")
    
    try:
        # RealMarketDataFetcher直接インスタンス化
        fetcher = RealMarketDataFetcher()
        
        # 実データ取得テスト（正しいメソッド名使用）
        print("📋 N225データ取得テスト...")
        n225_data = fetcher.fetch_required_market_data('N225', '2024-01-01', '2024-01-31')
        
        if n225_data is not None and len(n225_data) > 0:
            print(f"✅ N225データ取得成功: {len(n225_data)} rows")
            print(f"   Columns: {list(n225_data.columns)}")
        else:
            print("❌ N225データ取得失敗")
            return False
        
        print("📋 DJIデータ取得テスト...")
        dji_data = fetcher.fetch_required_market_data('DJI', '2024-01-01', '2024-01-31')
        
        if dji_data is not None and len(dji_data) > 0:
            print(f"✅ DJIデータ取得成功: {len(dji_data)} rows")
            print(f"   Columns: {list(dji_data.columns)}")
        else:
            print("❌ DJIデータ取得失敗")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ RealMarketDataFetcher直接テストエラー: {str(e)}")
        return False

def main():
    """Phase 3統合テスト実行"""
    print("=" * 60)
    print("TODO #14 Phase 3: 自動データ供給システム統合テスト")
    print("=" * 60)
    
    results = []
    
    # Phase 2確認（RealMarketDataFetcher）
    print("📋 Step 1: Phase 2機能確認（RealMarketDataFetcher）")
    results.append(test_real_data_fetcher_direct())
    
    # Phase 3テスト（自動供給）
    print("\n📋 Step 2: Phase 3機能テスト（自動データ供給統合）")
    results.append(test_phase3_auto_supply_vwap())
    results.append(test_phase3_auto_supply_opening_gap())
    
    # 手動データ提供テスト
    print("\n📋 Step 3: 手動データ提供動作確認")
    results.append(test_manual_data_provision())
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 Phase 3統合テスト結果サマリー")
    print("=" * 60)
    
    test_names = [
        "RealMarketDataFetcher直接テスト",
        "VWAPBreakoutStrategy自動供給テスト", 
        "OpeningGapStrategy自動供給テスト",
        "手動データ提供テスト"
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
        print("🎉 Phase 3統合テスト成功 - 自動データ供給システム正常動作")
        print("📋 次ステップ: Phase 4 (MarketDataQualityValidator) 実装準備完了")
    else:
        print("⚠️  Phase 3統合テスト部分的失敗 - 問題箇所の調査が必要")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)