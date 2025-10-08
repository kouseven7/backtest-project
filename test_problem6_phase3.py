# Problem 6 Phase 3: DSSMSBacktester統一マネージャー統合テスト
"""
Phase 3統合検証:
1. 統一マネージャーの初期化確認
2. _sync_portfolio_values()の統一保存機能
3. portfolio_values参照の統一アクセス
4. Phase 1データ移行機能
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'dssms'))

try:
    from dssms_backtester import DSSMSBacktester
    from datetime import datetime, timedelta
    
    print("=== Problem 6 Phase 3: DSSMSBacktester統一マネージャー統合テスト ===")
    
    # 1. DSSMSBacktester初期化と統一マネージャー確認
    test_config = {
        'initial_capital': 1000000,
        'switch_cost_rate': 0.001,
        'min_holding_period_hours': 24,
        'calculation_precision': 6,
        'max_portfolio_history': 10000
    }
    backtester = DSSMSBacktester(test_config)
    
    print(f"DSSMSBacktester初期化: {type(backtester)}")
    print(f"統一マネージャー: {type(backtester.unified_portfolio_manager)}")
    
    # 2. _sync_portfolio_values()統合テスト
    test_date = datetime(2024, 1, 15)
    test_value = 1050000.0
    
    print(f"\\n_sync_portfolio_values()テスト:")
    backtester._sync_portfolio_values(test_date, test_value)
    
    # 統一マネージャーから値取得確認
    unified_value = backtester.unified_portfolio_manager.get_unified_value(test_date, 0.0)
    print(f"   統一マネージャー取得値: {unified_value}")
    
    # レガシー辞書からも取得確認
    legacy_value = backtester.portfolio_values.get(test_date, 0.0)
    print(f"   レガシー辞書取得値: {legacy_value}")
    
    assert abs(unified_value - test_value) < 1e-6, f"統一マネージャー値不一致: {unified_value} != {test_value}"
    assert abs(legacy_value - test_value) < 1e-6, f"レガシー値不一致: {legacy_value} != {test_value}"
    print("   [OK] _sync_portfolio_values()統合成功")
    
    # 3. 複数データ保存テスト
    print(f"\\n複数データ保存テスト:")
    base_date = datetime(2024, 2, 1)
    test_values = [1100000, 1105000, 1102000, 1108000, 1115000]
    
    for i, value in enumerate(test_values):
        date = base_date + timedelta(days=i)
        backtester._sync_portfolio_values(date, value)
    
    print(f"   保存完了: {len(test_values)}件")
    
    # 4. Phase 1データ移行テスト
    print(f"\\nPhase 1データ移行テスト:")
    migration_success = backtester._migrate_phase1_to_unified()
    print(f"   移行結果: {migration_success}")
    
    if migration_success:
        validation_result = backtester.unified_portfolio_manager.validate_unified_integrity()
        total_records = validation_result.get('statistics', {}).get('total_records', 0)
        print(f"   移行後総レコード数: {total_records}")
        print(f"   エラー数: {len(validation_result.get('errors', []))}")
        print(f"   警告数: {len(validation_result.get('warnings', []))}")
    
    # 5. 統一アクセステスト (_make_market_condition内のportfolio_value参照)
    print(f"\\n統一アクセステスト:")
    test_access_date = base_date + timedelta(days=2)
    
    # シミュレーション: _make_market_condition内での統一アクセス
    unified_access_value = backtester.unified_portfolio_manager.get_unified_value(test_access_date, 100000.0)
    expected_value = test_values[2]
    
    print(f"   統一アクセス値: {unified_access_value} (期待値: {expected_value})")
    assert abs(unified_access_value - expected_value) < 1e-6, f"統一アクセス値不一致: {unified_access_value} != {expected_value}"
    print("   [OK] 統一アクセス成功")
    
    # 6. キャッシュ統計
    cache_stats = backtester.unified_portfolio_manager.get_cache_stats()
    print(f"\\n最終キャッシュ統計: {cache_stats}")
    
    print("\\nProblem 6 Phase 3: DSSMSBacktester統一マネージャー統合テスト完了")
    print("27箇所のportfolio_values参照のうち2箇所を統一マネージャーに移行完了")
    
except Exception as e:
    print(f"テストエラー: {e}")
    import traceback
    traceback.print_exc()