"""
Test script for Trend Performance Calculator
トレンド別パフォーマンス計算器のテストスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np

# パスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis.trend_performance_calculator import (
        TrendPerformanceCalculator,
        run_trend_performance_analysis
    )
    print("✓ trend_performance_calculator のインポート成功")
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
    sys.exit(1)

def test_trend_performance_calculator():
    """トレンド別パフォーマンス計算器のテスト"""
    print("\n=== トレンド別パフォーマンス計算器テスト開始 ===")
    
    # テスト用のバックテスト結果データ
    test_backtest_results = {
        "uptrend": {
            "periods": 3,
            "total_days": 150,
            "trades": [
                {"profit": 100, "取引結果": 100},
                {"profit": -30, "取引結果": -30},
                {"profit": 200, "取引結果": 200},
                {"profit": -50, "取引結果": -50},
                {"profit": 150, "取引結果": 150},
                {"profit": 80, "取引結果": 80},
                {"profit": -20, "取引結果": -20},
                {"profit": 120, "取引結果": 120}
            ]
        },
        "downtrend": {
            "periods": 2,
            "total_days": 80,
            "trades": [
                {"profit": -100, "取引結果": -100},
                {"profit": 50, "取引結果": 50},
                {"profit": -200, "取引結果": -200},
                {"profit": 100, "取引結果": 100},
                {"profit": -80, "取引結果": -80},
                {"profit": 40, "取引結果": 40}
            ]
        },
        "sideways": {
            "periods": 4,
            "total_days": 120,
            "trades": [
                {"profit": 20, "取引結果": 20},
                {"profit": -10, "取引結果": -10},
                {"profit": 30, "取引結果": 30},
                {"profit": -5, "取引結果": -5},
                {"profit": 15, "取引結果": 15},
                {"profit": -8, "取引結果": -8},
                {"profit": 25, "取引結果": 25},
                {"profit": -12, "取引結果": -12}
            ]
        }
    }
    
    try:
        # 1. 計算器の初期化テスト
        print("\n1. 計算器の初期化テスト")
        calculator = TrendPerformanceCalculator(
            output_dir="logs",
            risk_free_rate=0.02,
            trading_days=252
        )
        print("✓ 計算器の初期化成功")
        
        # 2. パフォーマンス指標計算テスト
        print("\n2. パフォーマンス指標計算テスト")
        performance_results = calculator.calculate_trend_performance_metrics(
            test_backtest_results,
            strategy_name="test_vwap_strategy"
        )
        print("✓ パフォーマンス指標計算成功")
        print(f"  - 計算されたトレンド数: {len(performance_results['trend_metrics'])}")
        
        # 3. 結果の構造確認
        print("\n3. 結果構造の確認")
        expected_keys = ["strategy_name", "calculation_timestamp", "trend_metrics", 
                        "comparative_analysis", "overall_summary"]
        for key in expected_keys:
            if key in performance_results:
                print(f"✓ {key} が存在")
            else:
                print(f"✗ {key} が不在")
        
        # 4. 各トレンドの指標確認
        print("\n4. 各トレンドの指標確認")
        for trend_type in ["uptrend", "downtrend", "sideways"]:
            if trend_type in performance_results["trend_metrics"]:
                metrics = performance_results["trend_metrics"][trend_type]
                print(f"  [{trend_type}]")
                print(f"    期間数: {metrics.get('period_count', 0)}")
                print(f"    取引日数: {metrics.get('total_trading_days', 0)}")
                
                if "basic_metrics" in metrics:
                    basic = metrics["basic_metrics"]
                    print(f"    総取引数: {basic.get('total_trades', 0)}")
                    print(f"    総利益: {basic.get('total_profit', 0):.2f}")
                    print(f"    勝率: {basic.get('win_rate', 0):.1f}%")
                
                if "risk_metrics" in metrics:
                    risk = metrics["risk_metrics"]
                    print(f"    Sharpe比: {risk.get('sharpe_ratio', 0):.3f}")
                    print(f"    最大DD: {risk.get('max_drawdown_percent', 0):.2f}%")
        
        # 5. ファイル保存テスト
        print("\n5. ファイル保存テスト")
        saved_filepath = calculator.save_performance_analysis("test_vwap_strategy")
        print(f"✓ 保存成功: {saved_filepath}")
        
        # 6. レポート生成テスト
        print("\n6. レポート生成テスト")
        report = calculator.generate_performance_report("test_vwap_strategy")
        print("✓ レポート生成成功")
        print("--- レポート内容（最初の500文字）---")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # 7. 便利関数のテスト
        print("\n7. 便利関数のテスト")
        convenience_results = run_trend_performance_analysis(
            test_backtest_results,
            strategy_name="convenience_test",
            output_dir="logs",
            save_results=True
        )
        print("✓ 便利関数の実行成功")
        
        print("\n=== すべてのテストが成功しました ===")
        return True
        
    except Exception as e:
        print(f"✗ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_calculations():
    """特定の計算機能のテスト"""
    print("\n=== 特定計算機能のテスト ===")
    
    # シンプルなテストデータ
    simple_trade_data = pd.DataFrame({
        "取引結果": [100, -50, 200, -30, 150, -80, 120, -20]
    })
    
    try:
        from metrics.performance_metrics import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_win_rate,
            calculate_total_profit
        )
        
        print("\n1. 既存performance_metrics関数のテスト")
        returns = simple_trade_data["取引結果"]
        
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        win_rate = calculate_win_rate(simple_trade_data)
        total_profit = calculate_total_profit(simple_trade_data)
        
        print(f"✓ Sharpe比: {sharpe:.3f}")
        print(f"✓ Sortino比: {sortino:.3f}")
        print(f"✓ 勝率: {win_rate:.1f}%")
        print(f"✓ 総利益: {total_profit:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 計算テスト中にエラー: {e}")
        return False

if __name__ == "__main__":
    print("トレンド別パフォーマンス計算器のテストを開始します...")
    
    # 基本テスト
    success1 = test_trend_performance_calculator()
    
    # 特定計算テスト
    success2 = test_specific_calculations()
    
    if success1 and success2:
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n❌ 一部のテストが失敗しました。")
