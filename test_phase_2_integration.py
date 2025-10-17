"""
Integration Test: MarketAnalyzer → DynamicStrategySelector
Phase 2 統合テスト
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from main_system.market_analysis.market_analyzer import MarketAnalyzer
from main_system.strategy_selection.dynamic_strategy_selector import DynamicStrategySelector

logger = setup_logger(__name__)


def create_test_data(trend_type='uptrend'):
    """テストデータ生成"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    if trend_type == 'uptrend':
        # 上昇トレンド
        close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    elif trend_type == 'downtrend':
        # 下降トレンド
        close = np.linspace(150, 100, 100) + np.random.normal(0, 2, 100)
    elif trend_type == 'sideways':
        # レンジ相場
        close = 125 + np.random.normal(0, 5, 100)
    else:
        # 通常
        close = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close * 0.99,
        'High': close * 1.02,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    return df


def test_phase_2_integration():
    """Phase 2統合テスト実行"""
    print("=" * 70)
    print("Phase 2 Integration Test: MarketAnalyzer → DynamicStrategySelector")
    print("=" * 70)
    
    # 1. コンポーネント初期化
    print("\n[1] Component Initialization")
    try:
        market_analyzer = MarketAnalyzer()
        print("  ✓ MarketAnalyzer initialized")
    except Exception as e:
        print(f"  ✗ MarketAnalyzer failed: {e}")
        return False
    
    try:
        strategy_selector = DynamicStrategySelector()
        print("  ✓ DynamicStrategySelector initialized")
    except Exception as e:
        print(f"  ✗ DynamicStrategySelector failed: {e}")
        return False
    
    # 2. 複数市場シナリオテスト
    test_scenarios = [
        ('uptrend', 'Test 1: Strong Uptrend Market'),
        ('downtrend', 'Test 2: Downtrend Market'),
        ('sideways', 'Test 3: Sideways/Range Market')
    ]
    
    all_tests_passed = True
    
    for trend_type, test_name in test_scenarios:
        print(f"\n[2] {test_name}")
        print("-" * 70)
        
        # テストデータ生成
        stock_data = create_test_data(trend_type)
        ticker = f"TEST_{trend_type.upper()}"
        
        # 2.1 市場分析実行
        print(f"  [2.1] Market Analysis for {ticker}")
        try:
            market_analysis = market_analyzer.comprehensive_market_analysis(
                stock_data, ticker
            )
            
            regime = market_analysis.get('market_regime', 'unknown')
            confidence = market_analysis.get('confidence_score', 0.0)
            
            print(f"    - Market Regime: {regime}")
            print(f"    - Confidence: {confidence:.2f}")
            print(f"    - Trend Interface Success: {market_analysis.get('trend_interface_success', False)}")
            print(f"    - Perfect Order Success: {market_analysis.get('perfect_order_success', False)}")
            
            if not market_analysis:
                print("    ✗ Market analysis returned empty results")
                all_tests_passed = False
                continue
            
            print("    ✓ Market analysis completed")
            
        except Exception as e:
            print(f"    ✗ Market analysis failed: {e}")
            all_tests_passed = False
            continue
        
        # 2.2 動的戦略選択実行
        print(f"  [2.2] Strategy Selection for {ticker}")
        try:
            selection_results = strategy_selector.select_optimal_strategies(
                market_analysis, stock_data, ticker
            )
            
            selected = selection_results.get('selected_strategies', [])
            weights = selection_results.get('strategy_weights', {})
            sel_confidence = selection_results.get('confidence_level', 0.0)
            
            print(f"    - Selected Strategies: {len(selected)}")
            for strategy in selected:
                weight = weights.get(strategy, 0.0)
                print(f"      * {strategy}: weight={weight:.2f}")
            
            print(f"    - Selection Confidence: {sel_confidence:.2f}")
            print(f"    - Selection Rationale: {selection_results.get('selection_rationale', 'N/A')}")
            
            if not selected:
                print("    ✗ No strategies selected")
                all_tests_passed = False
                continue
            
            # 重み合計チェック
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"    ✗ Weight sum mismatch: {total_weight:.2f} (expected 1.0)")
                all_tests_passed = False
                continue
            
            print("    ✓ Strategy selection completed")
            
        except Exception as e:
            print(f"    ✗ Strategy selection failed: {e}")
            all_tests_passed = False
            continue
        
        # 2.3 統合結果サマリー
        print(f"  [2.3] Integration Summary")
        try:
            summary = strategy_selector.get_selection_summary(selection_results)
            print(f"\n{summary}\n")
        except Exception as e:
            print(f"    ✗ Summary generation failed: {e}")
    
    # 3. 最終結果
    print("\n[3] Test Results")
    print("=" * 70)
    if all_tests_passed:
        print("✓ All integration tests PASSED")
        print("\nPhase 2 Integration Status:")
        print("  ✓ Phase 2.1: MarketAnalyzer - Operational")
        print("  ✓ Phase 2.2: DynamicStrategySelector - Operational")
        print("  ✓ Integration: MarketAnalyzer → DynamicStrategySelector - Working")
        return True
    else:
        print("✗ Some integration tests FAILED")
        print("\nPlease review the error messages above")
        return False


if __name__ == "__main__":
    try:
        success = test_phase_2_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
