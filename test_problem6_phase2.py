# Problem 6 Phase 2: PortfolioDataManager拡張機能テスト
"""
独立したPhase 2機能テスト
talib依存回避のため直接import
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 直接importでtalibエラー回避
sys.path.append(str(project_root / 'src' / 'dssms'))

try:
    from portfolio_data_manager import create_unified_portfolio_manager, DataValidationLevel
    from datetime import datetime, timedelta
    
    print("=== Problem 6 Phase 2: PortfolioDataManager拡張機能テスト ===")
    
    # 統一マネージャ作成
    manager = create_unified_portfolio_manager("basic")
    print(f"統一マネージャ作成成功: {type(manager)}")
    
    # Phase 2拡張機能テスト
    # 1. 統一値保存テスト
    base_date = datetime(2024, 1, 1)
    test_values = [1000000, 1001000, 1002000, 999000, 1005000]
    
    success_count = 0
    for i, value in enumerate(test_values):
        date = base_date + timedelta(days=i)
        if manager.store_unified_value(date, value, f"test_source_{i}"):
            success_count += 1
    
    print(f"統一値保存テスト: {success_count}/{len(test_values)}件成功")
    
    # 2. 統一値取得テスト
    test_date = base_date + timedelta(days=2)
    retrieved_value = manager.get_unified_value(test_date, 0.0)
    expected_value = test_values[2]
    
    print(f"統一値取得テスト: {retrieved_value} (期待値: {expected_value})")
    assert abs(retrieved_value - expected_value) < 1e-6, f"値が一致しません: {retrieved_value} != {expected_value}"
    
    # 3. データ整合性検証テスト
    validation_result = manager.validate_unified_integrity()
    print(f"データ整合性検証: {validation_result['status']}")
    print(f"   - レコード数: {validation_result.get('statistics', {}).get('total_records', 'N/A')}")
    print(f"   - エラー数: {len(validation_result.get('errors', []))}")
    print(f"   - 警告数: {len(validation_result.get('warnings', []))}")
    
    # 4. キャッシュ統計テスト
    cache_stats = manager.get_cache_stats()
    print(f"キャッシュ統計: {cache_stats}")
    
    # 5. Phase 1データ移行テスト
    phase1_portfolio_values = {
        base_date + timedelta(days=10): 1010000,
        base_date + timedelta(days=11): 1011000,
        base_date + timedelta(days=12): 1012000
    }
    
    phase1_performance_history = {
        'portfolio_value': [1020000, 1021000, 1022000],
        'timestamps': [base_date + timedelta(days=13), base_date + timedelta(days=14), base_date + timedelta(days=15)]
    }
    
    sync_success = manager.sync_with_phase1_data(
        phase1_portfolio_values, 
        [], 
        phase1_performance_history
    )
    
    print(f"Phase 1データ同期: {sync_success}")
    
    # 6. 最終統計
    final_validation = manager.validate_unified_integrity()
    final_records = final_validation.get('statistics', {}).get('total_records', 0)
    print(f"最終データ件数: {final_records}件")
    
    print("Problem 6 Phase 2: PortfolioDataManager拡張機能テスト完了")
    
except Exception as e:
    print(f"テストエラー: {e}")
    import traceback
    traceback.print_exc()