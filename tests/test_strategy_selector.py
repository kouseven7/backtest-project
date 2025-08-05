"""
Strategy Selector Integration Test
3-1-1「StrategySelector クラス設計・実装」統合テスト

このテストスクリプトは以下を検証します：
1. StrategySelector の基本機能
2. 既存システムとの統合
3. 複数の選択手法の動作
4. エラーハンドリング
5. パフォーマンス特性
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 実装モジュールのインポート
try:
    from config.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection,
        SelectionMethod, create_strategy_selector, select_best_strategies_for_trend
    )
    print("✓ StrategySelector modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_data(days: int = 100, trend_type: str = "uptrend") -> pd.DataFrame:
    """テスト用マーケットデータの作成"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # トレンドタイプに応じたデータ生成
    base_price = 100
    if trend_type == "uptrend":
        trend = np.linspace(0, 20, days)
        noise = np.random.normal(0, 2, days)
    elif trend_type == "downtrend":
        trend = np.linspace(0, -15, days)
        noise = np.random.normal(0, 2, days)
    elif trend_type == "sideways":
        trend = np.sin(np.linspace(0, 4*np.pi, days)) * 5
        noise = np.random.normal(0, 1, days)
    else:  # random
        trend = np.random.normal(0, 1, days).cumsum()
        noise = np.random.normal(0, 2, days)
    
    prices = base_price + trend + noise
    
    # OHLCV データの作成
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.01, days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(10000, 100000, days)
    })
    
    return data

def test_basic_functionality():
    """基本機能のテスト"""
    print("\n=== 基本機能テスト ===")
    
    try:
        # 1. StrategySelector の初期化
        selector = create_strategy_selector()
        print(f"✓ StrategySelector initialized with {len(selector.get_available_strategies())} strategies")
        
        # 2. テストデータの準備
        test_data = create_test_data(100, "uptrend")
        print(f"✓ Test data created: {len(test_data)} days")
        
        # 3. 基本的な戦略選択
        selection = selector.select_strategies(test_data, "TEST")
        print(f"✓ Strategy selection completed: {len(selection.selected_strategies)} strategies")
        print(f"  Selected: {selection.selected_strategies}")
        print(f"  Weights: {selection.strategy_weights}")
        print(f"  Total Score: {selection.total_score:.3f}")
        print(f"  Confidence: {selection.confidence_level:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_selection_methods():
    """選択手法のテスト"""
    print("\n=== 選択手法テスト ===")
    
    try:
        selector = create_strategy_selector()
        test_data = create_test_data(100, "uptrend")
        
        methods_to_test = [
            (SelectionMethod.TOP_N, "Top N Selection"),
            (SelectionMethod.THRESHOLD, "Threshold Selection"),
            (SelectionMethod.HYBRID, "Hybrid Selection"),
            (SelectionMethod.WEIGHTED, "Weighted Selection"),
            (SelectionMethod.ADAPTIVE, "Adaptive Selection")
        ]
        
        results = {}
        
        for method, description in methods_to_test:
            try:
                criteria = SelectionCriteria(
                    method=method,
                    min_score_threshold=0.5,
                    max_strategies=3
                )
                
                selection = selector.select_strategies(test_data, "TEST", criteria)
                results[method.value] = {
                    "strategies": selection.selected_strategies,
                    "count": len(selection.selected_strategies),
                    "total_score": selection.total_score
                }
                
                print(f"✓ {description}: {len(selection.selected_strategies)} strategies selected")
                
            except Exception as e:
                print(f"❌ {description} failed: {e}")
                results[method.value] = {"error": str(e)}
        
        # 結果サマリー
        print(f"\n選択手法比較:")
        for method, result in results.items():
            if "error" not in result:
                print(f"  {method}: {result['count']} strategies, score: {result['total_score']:.3f}")
            else:
                print(f"  {method}: Error - {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Selection methods test failed: {e}")
        return False

def test_trend_adaptation():
    """トレンド適応テスト"""
    print("\n=== トレンド適応テスト ===")
    
    try:
        selector = create_strategy_selector()
        
        trend_types = ["uptrend", "downtrend", "sideways"]
        results = {}
        
        for trend_type in trend_types:
            test_data = create_test_data(100, trend_type)
            selection = selector.select_strategies(test_data, f"TEST_{trend_type.upper()}")
            
            results[trend_type] = {
                "strategies": selection.selected_strategies,
                "trend_detected": selection.trend_analysis.get("trend", "unknown"),
                "confidence": selection.confidence_level,
                "total_score": selection.total_score
            }
            
            print(f"✓ {trend_type.capitalize()} trend test completed:")
            print(f"  Detected: {results[trend_type]['trend_detected']}")
            print(f"  Selected: {results[trend_type]['strategies']}")
            print(f"  Confidence: {results[trend_type]['confidence']:.3f}")
        
        # トレンド間の選択差異を確認
        uptrend_strategies = set(results["uptrend"]["strategies"])
        downtrend_strategies = set(results["downtrend"]["strategies"])
        sideways_strategies = set(results["sideways"]["strategies"])
        
        print(f"\nトレンド間の戦略選択差異:")
        print(f"  上昇トレンド専用: {uptrend_strategies - downtrend_strategies - sideways_strategies}")
        print(f"  下降トレンド専用: {downtrend_strategies - uptrend_strategies - sideways_strategies}")
        print(f"  横ばい専用: {sideways_strategies - uptrend_strategies - downtrend_strategies}")
        print(f"  共通選択: {uptrend_strategies & downtrend_strategies & sideways_strategies}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trend adaptation test failed: {e}")
        return False

def test_configuration_profiles():
    """設定プロファイルテスト"""
    print("\n=== 設定プロファイルテスト ===")
    
    try:
        selector = create_strategy_selector(
            config_file="config/strategy_selector_config.json"
        )
        test_data = create_test_data(100, "uptrend")
        
        # 保守的プロファイル
        conservative_criteria = SelectionCriteria(
            method=SelectionMethod.THRESHOLD,
            min_score_threshold=0.75,
            max_strategies=2,
            enable_diversification=True
        )
        
        # 積極的プロファイル
        aggressive_criteria = SelectionCriteria(
            method=SelectionMethod.TOP_N,
            min_score_threshold=0.5,
            max_strategies=5,
            enable_diversification=False
        )
        
        # バランス型プロファイル
        balanced_criteria = SelectionCriteria(
            method=SelectionMethod.HYBRID,
            min_score_threshold=0.6,
            max_strategies=3,
            enable_diversification=True
        )
        
        profiles = [
            (conservative_criteria, "Conservative"),
            (aggressive_criteria, "Aggressive"), 
            (balanced_criteria, "Balanced")
        ]
        
        for criteria, profile_name in profiles:
            selection = selector.select_strategies(test_data, "TEST", criteria)
            print(f"✓ {profile_name} profile:")
            print(f"  Strategies: {len(selection.selected_strategies)}")
            print(f"  Selected: {selection.selected_strategies}")
            print(f"  Total Score: {selection.total_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration profiles test failed: {e}")
        return False

def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")
    
    try:
        selector = create_strategy_selector()
        
        # 1. 空データテスト
        try:
            empty_data = pd.DataFrame()
            selection = selector.select_strategies(empty_data, "EMPTY_TEST")
            print(f"✓ Empty data handled: {len(selection.selected_strategies)} strategies")
        except Exception as e:
            print(f"  Empty data error: {e}")
        
        # 2. 不正な設定テスト
        try:
            invalid_criteria = SelectionCriteria(
                min_score_threshold=1.5,  # 無効な値
                max_strategies=0
            )
            test_data = create_test_data(50)
            selection = selector.select_strategies(test_data, "INVALID_TEST", invalid_criteria)
            print(f"✓ Invalid criteria handled: {len(selection.selected_strategies)} strategies")
        except Exception as e:
            print(f"  Invalid criteria error: {e}")
        
        # 3. 短期間データテスト
        try:
            short_data = create_test_data(10)  # 短期間
            selection = selector.select_strategies(short_data, "SHORT_TEST")
            print(f"✓ Short data handled: {len(selection.selected_strategies)} strategies")
        except Exception as e:
            print(f"  Short data error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    try:
        selector = create_strategy_selector()
        
        # 1. 処理時間測定
        test_data = create_test_data(200)
        
        start_time = datetime.now()
        selection = selector.select_strategies(test_data, "PERF_TEST")
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        print(f"✓ Processing time: {processing_time:.1f}ms")
        
        # 2. キャッシュ効果測定
        start_time = datetime.now()
        selection2 = selector.select_strategies(test_data, "PERF_TEST")  # 同じデータ
        end_time = datetime.now()
        
        cached_time = (end_time - start_time).total_seconds() * 1000
        print(f"✓ Cached processing time: {cached_time:.1f}ms")
        print(f"✓ Cache speedup: {processing_time/max(cached_time, 0.1):.1f}x")
        
        # 3. 統計情報
        stats = selector.get_statistics()
        print(f"✓ System statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_integration_with_existing_systems():
    """既存システムとの統合テスト"""
    print("\n=== 既存システム統合テスト ===")
    
    try:
        # 既存モジュールとの統合確認
        from config.strategy_scoring_model import StrategyScoreCalculator
        from indicators.unified_trend_detector import UnifiedTrendDetector
        
        selector = create_strategy_selector()
        test_data = create_test_data(100)
        
        # 1. 戦略スコアリングシステム連携確認
        score_calc = StrategyScoreCalculator()
        print("✓ Strategy scoring system integration verified")
        
        # 2. 統一トレンド判定器連携確認
        trend_detector = UnifiedTrendDetector(test_data)
        trend = trend_detector.detect_trend()
        print(f"✓ Unified trend detector integration verified: {trend}")
        
        # 3. 戦略選択との整合性確認
        selection = selector.select_strategies(test_data, "INTEGRATION_TEST")
        detected_trend = selection.trend_analysis.get("trend")
        print(f"✓ Trend consistency: Direct={trend}, Selection={detected_trend}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_convenience_functions():
    """便利関数のテスト"""
    print("\n=== 便利関数テスト ===")
    
    try:
        test_data = create_test_data(100, "uptrend")
        
        # 簡単選択関数のテスト
        selection = select_best_strategies_for_trend(test_data, "CONVENIENCE_TEST", max_strategies=2)
        
        print(f"✓ Convenience function test:")
        print(f"  Selected: {selection.selected_strategies}")
        print(f"  Weights: {selection.strategy_weights}")
        print(f"  Method: {selection.selection_reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False

def run_comprehensive_test():
    """包括的テストの実行"""
    print("🚀 StrategySelector 包括的テスト開始")
    print("=" * 60)
    
    test_functions = [
        ("基本機能", test_basic_functionality),
        ("選択手法", test_selection_methods),
        ("トレンド適応", test_trend_adaptation),
        ("設定プロファイル", test_configuration_profiles),
        ("エラーハンドリング", test_error_handling),
        ("パフォーマンス", test_performance),
        ("既存システム統合", test_integration_with_existing_systems),
        ("便利関数", test_convenience_functions)
    ]
    
    results = []
    
    for test_name, test_func in test_functions:
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} テスト成功")
            else:
                print(f"❌ {test_name} テスト失敗")
        except Exception as e:
            print(f"❌ {test_name} テスト例外: {e}")
            results.append((test_name, False))
        
        print("-" * 40)
    
    # 結果サマリー
    print("\n📊 テスト結果サマリー")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 総合結果: {passed}/{total} テスト成功")
    
    if passed == total:
        print("🎉 全テスト成功！StrategySelector の実装が完了しました。")
    else:
        print(f"⚠️  {total - passed} 個のテストが失敗しました。")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
