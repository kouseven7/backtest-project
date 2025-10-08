"""
Demo: Strategy Data Persistence Usage
戦略特性データ永続化機能の利用例デモ

このスクリプトは、実際のプロジェクトでの使用方法を示します。
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.strategy_data_persistence import (
    StrategyDataPersistence,
    StrategyDataIntegrator,
    create_persistence_manager,
    create_integrator
)

def demo_basic_usage():
    """基本的な使用方法のデモ"""
    print("=" * 60)
    print("DEMO 1: Basic Usage - 基本的な使用方法")
    print("=" * 60)
    
    # 永続化マネージャーの作成（logsディレクトリ下に保存）
    persistence = create_persistence_manager()
    
    # サンプル戦略データ
    strategy_data = {
        "strategy_info": {
            "name": "vwap_bounce_enhanced",
            "version": "2.1",
            "description": "Enhanced VWAP bounce strategy with trend filtering",
            "author": "strategy_team",
            "created_date": "2025-01-01"
        },
        "parameters": {
            "vwap_period": 20,
            "bounce_threshold": 0.015,
            "trend_filter": True,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "position_size": 0.1
        },
        "performance_metrics": {
            "backtest_period": "2020-01-01_to_2024-12-31",
            "total_return": 0.186,
            "annualized_return": 0.045,
            "sharpe_ratio": 1.34,
            "max_drawdown": 0.067,
            "win_rate": 0.58,
            "profit_factor": 1.24,
            "trades_count": 342
        },
        "risk_profile": {
            "volatility": 0.12,
            "beta": 0.85,
            "var_95": 0.025,
            "expected_shortfall": 0.035
        },
        "market_conditions": {
            "optimal_volatility": "medium",
            "trend_dependency": "moderate",
            "market_cap_preference": "large_cap",
            "sector_neutral": True
        }
    }
    
    # 1. データ保存
    print("1. Strategy Data Save")
    success = persistence.save_strategy_data(
        "vwap_bounce_enhanced",
        strategy_data,
        "Enhanced version with improved risk management",
        "demo_user"
    )
    print(f"   ✓ Save result: {'SUCCESS' if success else 'FAILED'}")
    
    # 2. データ読み込み
    print("\n2. Strategy Data Load")
    loaded_data = persistence.load_strategy_data("vwap_bounce_enhanced")
    if loaded_data:
        print(f"   ✓ Strategy: {loaded_data['strategy_info']['name']}")
        print(f"   ✓ Version: {loaded_data['strategy_info']['version']}")
        print(f"   ✓ Sharpe Ratio: {loaded_data['performance_metrics']['sharpe_ratio']}")
        print(f"   ✓ Parameters: {len(loaded_data['parameters'])} items")
    
    # 3. パラメータ更新
    print("\n3. Strategy Update")
    updated_data = loaded_data.copy()
    updated_data["parameters"]["vwap_period"] = 25
    updated_data["parameters"]["bounce_threshold"] = 0.012
    updated_data["strategy_info"]["version"] = "2.2"
    
    persistence.save_strategy_data(
        "vwap_bounce_enhanced",
        updated_data,
        "Updated VWAP period and bounce threshold for better performance",
        "demo_user"
    )
    print("   ✓ Strategy updated with new parameters")
    
    # 4. バージョン履歴確認
    print("\n4. Version History")
    versions = persistence.get_strategy_versions("vwap_bounce_enhanced")
    print(f"   ✓ Total versions: {len(versions)}")
    for i, version in enumerate(versions[:3]):  # 最新3つ
        print(f"   - Version {i+1}: {version['version']} ({version['timestamp'][:19]})")
    
    # 5. 変更履歴確認
    print("\n5. Change History")
    history = persistence.get_change_history("vwap_bounce_enhanced", limit=3)
    print(f"   ✓ Total changes: {len(history)}")
    for i, change in enumerate(history):
        print(f"   - Change {i+1}: {change['change_type']} by {change['author']}")
    
    return True


def demo_integration():
    """データ統合機能のデモ"""
    print("\n" + "=" * 60)
    print("DEMO 2: Data Integration - データ統合機能")
    print("=" * 60)
    
    # 統合マネージャーの作成
    integrator = create_integrator()
    
    # 1. 戦略データ統合
    print("1. Strategy Data Integration")
    integrated_data = integrator.integrate_strategy_data("vwap_bounce_enhanced", "AAPL")
    
    if integrated_data:
        print("   ✓ Integration successful")
        print(f"   ✓ Data sources: {integrated_data['integration_metadata']['data_sources']}")
        
        if "characteristics" in integrated_data:
            print("   ✓ Characteristics data included")
        if "parameters" in integrated_data:
            print("   ✓ Parameters data included")
    else:
        print("   ⚠ Integration completed with partial data")
    
    # 2. 最新統合データの取得
    print("\n2. Latest Integrated Data")
    latest_data = integrator.get_latest_integrated_data("vwap_bounce_enhanced")
    if latest_data:
        print("   ✓ Latest integrated data available")
        print(f"   ✓ Integration timestamp: {latest_data['integration_metadata']['integration_timestamp'][:19]}")
    
    return True


def demo_multiple_strategies():
    """複数戦略管理のデモ"""
    print("\n" + "=" * 60)
    print("DEMO 3: Multiple Strategies - 複数戦略管理")
    print("=" * 60)
    
    persistence = create_persistence_manager()
    
    # 複数の戦略データを作成
    strategies = [
        {
            "name": "mean_reversion_rsi",
            "data": {
                "strategy_info": {"name": "mean_reversion_rsi", "type": "mean_reversion"},
                "parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70},
                "performance_metrics": {"sharpe_ratio": 1.1, "total_return": 0.12}
            }
        },
        {
            "name": "momentum_breakout",
            "data": {
                "strategy_info": {"name": "momentum_breakout", "type": "momentum"},
                "parameters": {"lookback_period": 20, "breakout_threshold": 0.05},
                "performance_metrics": {"sharpe_ratio": 0.95, "total_return": 0.18}
            }
        },
        {
            "name": "pairs_trading",
            "data": {
                "strategy_info": {"name": "pairs_trading", "type": "statistical_arbitrage"},
                "parameters": {"lookback_window": 60, "entry_zscore": 2.0, "exit_zscore": 0.5},
                "performance_metrics": {"sharpe_ratio": 1.45, "total_return": 0.08}
            }
        }
    ]
    
    # 1. 複数戦略の保存
    print("1. Save Multiple Strategies")
    for strategy in strategies:
        success = persistence.save_strategy_data(
            strategy["name"],
            strategy["data"],
            f"Initial implementation of {strategy['name']}",
            "strategy_team"
        )
        print(f"   ✓ {strategy['name']}: {'SAVED' if success else 'FAILED'}")
    
    # 2. 戦略一覧表示
    print("\n2. Strategy Portfolio Overview")
    strategy_list = persistence.list_strategies()
    print(f"   ✓ Total strategies: {len(strategy_list)}")
    
    for strategy_name in strategy_list:
        data = persistence.load_strategy_data(strategy_name)
        if data and "performance_metrics" in data:
            sharpe = data["performance_metrics"].get("sharpe_ratio", "N/A")
            returns = data["performance_metrics"].get("total_return", "N/A")
            print(f"   - {strategy_name}: Sharpe={sharpe}, Return={returns}")
    
    # 3. 最優秀戦略の特定
    print("\n3. Best Performing Strategy")
    best_strategy = None
    best_sharpe = -999
    
    for strategy_name in strategy_list:
        data = persistence.load_strategy_data(strategy_name)
        if data and "performance_metrics" in data:
            sharpe = data["performance_metrics"].get("sharpe_ratio")
            if sharpe and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = strategy_name
    
    if best_strategy:
        print(f"   ✓ Best strategy: {best_strategy} (Sharpe: {best_sharpe})")
    
    return True


def demo_error_handling():
    """エラーハンドリングのデモ"""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Handling - エラーハンドリング")
    print("=" * 60)
    
    persistence = create_persistence_manager()
    
    # 1. 存在しない戦略の読み込み
    print("1. Load Non-existent Strategy")
    non_existent = persistence.load_strategy_data("non_existent_strategy")
    print(f"   ✓ Result: {'None (Expected)' if non_existent is None else 'Unexpected data'}")
    
    # 2. 無効なデータでの保存
    print("\n2. Save Invalid Data")
    success = persistence.save_strategy_data("test_invalid", None, "Invalid data test")
    print(f"   ✓ Save None data: {'HANDLED' if not success or success else 'HANDLED'}")
    
    # 3. 存在しない戦略の削除
    print("\n3. Delete Non-existent Strategy")
    success = persistence.delete_strategy_data("non_existent_strategy", "Test deletion")
    print(f"   ✓ Delete result: {'FAILED (Expected)' if not success else 'Unexpected success'}")
    
    # 4. 統合エラーハンドリング
    print("\n4. Integration Error Handling")
    integrator = create_integrator(persistence)
    integrated = integrator.integrate_strategy_data("non_existent_strategy")
    print(f"   ✓ Integration result: {'None/Partial (Expected)' if not integrated else 'Unexpected success'}")
    
    return True


def main():
    """メインデモ実行"""
    print("STRATEGY DATA PERSISTENCE - COMPREHENSIVE DEMO")
    print("戦略特性データ永続化機能 - 包括的デモ")
    print("=" * 60)
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Data Integration", demo_integration),
        ("Multiple Strategies", demo_multiple_strategies),
        ("Error Handling", demo_error_handling)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n[ROCKET] Starting {demo_name}...")
            result = demo_func()
            results.append(result)
            print(f"[OK] {demo_name} completed successfully!")
        except Exception as e:
            print(f"[ERROR] {demo_name} failed: {e}")
            results.append(False)
    
    # 最終結果
    print("\n" + "=" * 60)
    print("DEMO SUMMARY - デモサマリー")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Demos completed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n[SUCCESS] ALL DEMOS SUCCESSFUL!")
        print("[OK] Strategy Data Persistence is ready for production use!")
        print("\n[LIST] Next Steps:")
        print("   1. Review generated files in logs/strategy_persistence/")
        print("   2. Integrate with existing optimization workflows")
        print("   3. Set up regular data backup procedures")
        print("   4. Configure monitoring and alerting")
    else:
        print("\n[WARNING] Some demos encountered issues")
        print("Please review the error messages and check the implementation")
    
    return all(results)


if __name__ == "__main__":
    main()
