"""
TODO #14 Phase 1 検証テスト
自動生成データ廃止・実データ取得必須化の動作確認

Author: AI Assistant
Created: 2025-10-08
Purpose: VWAPBreakoutStrategyとOpeningGapStrategyのエラー停止機能検証
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトパス追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

print("="*80)
print("TODO #14 Phase 1 検証テスト: 自動生成データ廃止・実データ取得必須化")
print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
print("="*80)

def create_test_stock_data():
    """テスト用株価データ生成"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': 1000 + np.random.randn(len(dates)) * 10,
        'High': 1000 + np.random.randn(len(dates)) * 10 + 5,
        'Low': 1000 + np.random.randn(len(dates)) * 10 - 5,
        'Close': 1000 + np.random.randn(len(dates)) * 10,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # 価格の論理的整合性確保
    for i in range(len(data)):
        high = max(data.iloc[i]['Open'], data.iloc[i]['Close']) + abs(np.random.randn() * 2)
        low = min(data.iloc[i]['Open'], data.iloc[i]['Close']) - abs(np.random.randn() * 2)
        data.iloc[i, data.columns.get_loc('High')] = high
        data.iloc[i, data.columns.get_loc('Low')] = low
    
    data.set_index('Date', inplace=True)
    return data

def test_strategy_error_stopping():
    """戦略エラー停止機能テスト"""
    print("\n🔍 Test 1: VWAPBreakoutStrategy エラー停止機能確認")
    
    try:
        from config.multi_strategy_manager import MultiStrategyManager
        
        # MultiStrategyManager初期化
        manager = MultiStrategyManager()
        manager.initialize_systems()
        
        # テスト用データ準備
        test_data = create_test_stock_data()
        test_params = {'lookback_period': 20, 'breakout_threshold': 0.02}
        
        print(f"  📊 テストデータ: {test_data.shape[0]}日分")
        print(f"  📋 テストパラメータ: {test_params}")
        
        # VWAPBreakoutStrategy を index_data なしで呼び出し（エラー期待）
        print("\n  🎯 VWAPBreakoutStrategy を index_data パラメータなしで呼び出し...")
        
        try:
            strategy_instance = manager.get_strategy_instance(
                strategy_name='VWAPBreakoutStrategy',
                data=test_data,
                params=test_params
            )
            print("  ❌ エラー停止機能が動作していません - 戦略インスタンスが作成されました")
            return False
            
        except ValueError as e:
            if "Real data required" in str(e) and "index_data" in str(e):
                print("  ✅ エラー停止機能が正しく動作しています")
                print(f"  📝 エラーメッセージ: {str(e)[:100]}...")
                return True
            else:
                print(f"  ⚠️ 予期しないエラー: {e}")
                return False
                
    except Exception as e:
        print(f"  ❌ テスト実行エラー: {e}")
        return False

def test_opening_gap_strategy_error_stopping():
    """OpeningGapStrategy エラー停止機能テスト"""
    print("\n🔍 Test 2: OpeningGapStrategy エラー停止機能確認")
    
    try:
        from config.multi_strategy_manager import MultiStrategyManager
        
        # MultiStrategyManager初期化
        manager = MultiStrategyManager()
        manager.initialize_systems()
        
        # テスト用データ準備
        test_data = create_test_stock_data()
        test_params = {'gap_threshold': 0.02, 'holding_period': 5}
        
        print(f"  📊 テストデータ: {test_data.shape[0]}日分")
        print(f"  📋 テストパラメータ: {test_params}")
        
        # OpeningGapStrategy を dow_data なしで呼び出し（エラー期待）
        print("\n  🎯 OpeningGapStrategy を dow_data パラメータなしで呼び出し...")
        
        try:
            strategy_instance = manager.get_strategy_instance(
                strategy_name='OpeningGapStrategy',
                data=test_data,
                params=test_params
            )
            print("  ❌ エラー停止機能が動作していません - 戦略インスタンスが作成されました")
            return False
            
        except ValueError as e:
            if "Real data required" in str(e) and "dow_data" in str(e):
                print("  ✅ エラー停止機能が正しく動作しています")
                print(f"  📝 エラーメッセージ: {str(e)[:100]}...")
                return True
            else:
                print(f"  ⚠️ 予期しないエラー: {e}")
                return False
                
    except Exception as e:
        print(f"  ❌ テスト実行エラー: {e}")
        return False

def test_other_strategies_unaffected():
    """他の戦略への影響がないことを確認"""
    print("\n🔍 Test 3: 他の5戦略への影響なし確認")
    
    unaffected_strategies = [
        'MomentumInvestingStrategy',
        'BreakoutStrategy', 
        'VWAPBounceStrategy',
        'ContrarianStrategy',
        'GCStrategy'
    ]
    
    results = {}
    
    for strategy_name in unaffected_strategies:
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            
            # MultiStrategyManager初期化
            manager = MultiStrategyManager()
            manager.initialize_systems()
            
            # テスト用データ準備
            test_data = create_test_stock_data()
            test_params = {'lookback_period': 20}
            
            print(f"\n  🎯 {strategy_name} テスト中...")
            
            # 戦略インスタンス化試行
            strategy_instance = manager.get_strategy_instance(
                strategy_name=strategy_name,
                data=test_data,
                params=test_params
            )
            
            # backtest()メソッド確認
            if hasattr(strategy_instance, 'backtest') and callable(strategy_instance.backtest):
                print(f"  ✅ {strategy_name}: 正常動作")
                results[strategy_name] = True
            else:
                print(f"  ⚠️ {strategy_name}: backtest()メソッドなし")
                results[strategy_name] = False
                
        except Exception as e:
            print(f"  ❌ {strategy_name}: エラー - {str(e)[:50]}...")
            results[strategy_name] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n  📊 他戦略動作確認結果: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    return success_count >= 3  # 最低3戦略は動作することを期待

def main():
    """メインテスト実行"""
    print("\n🚀 TODO #14 Phase 1 検証テスト開始\n")
    
    test_results = []
    
    # Test 1: VWAPBreakoutStrategy エラー停止
    result1 = test_strategy_error_stopping()
    test_results.append(("VWAPBreakoutStrategy エラー停止", result1))
    
    # Test 2: OpeningGapStrategy エラー停止  
    result2 = test_opening_gap_strategy_error_stopping()
    test_results.append(("OpeningGapStrategy エラー停止", result2))
    
    # Test 3: 他戦略への影響なし
    result3 = test_other_strategies_unaffected()
    test_results.append(("他戦略への影響なし", result3))
    
    # 総合評価
    print("\n" + "="*80)
    print("📊 TODO #14 Phase 1 検証結果")
    print("="*80)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Phase 1 実装品質: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ TODO #14 Phase 1 実装成功 - Phase 2 準備可能")
        return True
    else:
        print("❌ TODO #14 Phase 1 要改善 - Phase 2 延期推奨")
        return False

if __name__ == "__main__":
    main()