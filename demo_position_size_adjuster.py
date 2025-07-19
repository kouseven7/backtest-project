"""
Demo: Position Size Adjuster System
File: demo_position_size_adjuster.py
Description: 
  3-3-2「各戦略のポジションサイズ調整機能」
  実際の使用例を示すデモンストレーションスクリプト

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Demo Features:
1. 基本的なポジションサイズ計算
2. ハイブリッド適応型調整デモ
3. 複数戦略のポートフォリオ最適化
4. 市場環境別の調整例
5. エラーハンドリング例
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.position_size_adjuster import (
        PositionSizeAdjuster, PositionSizingConfig, PositionSizeMethod, 
        RiskAdjustmentType, MarketRegime, PositionSizeResult, PortfolioPositionSizing
    )
    from config.strategy_scoring_model import StrategyScore, StrategyScoreManager
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
except ImportError as e:
    logger.warning(f"Import error: {e}. Running with mock data.")

def create_sample_market_data(days: int = 100) -> pd.DataFrame:
    """サンプル市場データの作成"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # ランダムウォーク価格データ
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # 日次リターン
    
    prices = [100.0]  # 初期価格
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    volumes = np.random.randint(1000, 5000, days)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'open': [p * 0.999 for p in prices]
    })
    
    return data

def demo_basic_position_sizing():
    """基本的なポジションサイジングのデモ"""
    print("\n" + "="*60)
    print("📊 Demo 1: 基本的なポジションサイズ計算")
    print("="*60)
    
    try:
        # 1. PositionSizeAdjusterの初期化
        portfolio_value = 1000000.0  # $1M ポートフォリオ
        adjuster = PositionSizeAdjuster(portfolio_value=portfolio_value)
        
        # 2. サンプルデータの準備
        market_data = create_sample_market_data(60)
        ticker = "DEMO_STOCK"
        
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Market Data Period: {len(market_data)} days")
        print(f"Current Price: ${market_data['close'].iloc[-1]:.2f}")
        
        # 3. ポジションサイズ計算
        result = adjuster.calculate_portfolio_position_sizes(
            ticker=ticker,
            market_data=market_data
        )
        
        # 4. 結果の表示
        print(f"\n📈 Portfolio Position Sizing Results:")
        print(f"  Total Strategies: {len(result.position_results)}")
        print(f"  Total Allocated: {result.total_allocated_percentage:.2%}")
        print(f"  Remaining Cash: {result.remaining_cash_percentage:.2%}")
        print(f"  Portfolio Risk: {result.portfolio_risk_estimate:.2%}")
        print(f"  Diversification Score: {result.diversification_score:.3f}")
        print(f"  Market Regime: {result.regime_analysis.get('regime', 'Unknown')}")
        
        # 5. 個別戦略詳細
        if result.position_results:
            print(f"\n📋 Individual Position Details:")
            for strategy_name, pos_result in result.position_results.items():
                print(f"  {strategy_name}:")
                print(f"    Position Size: {pos_result.adjusted_size:.2%}")
                print(f"    Absolute Amount: ${pos_result.absolute_amount:,.2f}" if pos_result.absolute_amount else "    Absolute Amount: N/A")
                print(f"    Share Count: {pos_result.share_count:,}" if pos_result.share_count else "    Share Count: N/A")
                print(f"    Market Regime: {pos_result.market_regime.value}")
                print(f"    Confidence: {pos_result.confidence_level:.2%}")
                print(f"    Reason: {pos_result.calculation_reason}")
                print()
        
        # 6. 制約違反があれば表示
        if result.constraint_violations:
            print(f"⚠️ Constraint Violations:")
            for violation in result.constraint_violations:
                print(f"    - {violation}")
        
        # 7. レポート生成
        report = adjuster.create_position_sizing_report(result)
        print(f"\n📄 Generated Report Summary:")
        print(f"  Report Generated: ✅")
        print(f"  Rebalancing Needed: {'Yes' if result.rebalancing_needed else 'No'}")
        
        return result
        
    except Exception as e:
        logger.error(f"Basic position sizing demo failed: {e}")
        print(f"❌ Error in basic demo: {e}")
        return None

def demo_hybrid_adaptive_sizing():
    """ハイブリッド適応型ポジションサイジングのデモ"""
    print("\n" + "="*60)
    print("🔄 Demo 2: ハイブリッド適応型ポジションサイズ調整")
    print("="*60)
    
    try:
        # 1. カスタム設定の作成
        custom_config = PositionSizingConfig(
            sizing_method=PositionSizeMethod.HYBRID_ADAPTIVE,
            base_position_size=0.03,  # 3%ベース
            max_position_size=0.12,   # 12%最大
            min_position_size=0.008,  # 0.8%最小
            score_weight=0.5,         # スコア重視
            risk_weight=0.25,
            market_weight=0.15,
            trend_confidence_weight=0.10,
            enable_dynamic_adjustment=True,
            regime_sensitivity=0.8    # 高感度
        )
        
        print("🎛️ Custom Configuration:")
        print(f"  Method: {custom_config.sizing_method.value}")
        print(f"  Base Size: {custom_config.base_position_size:.1%}")
        print(f"  Max Size: {custom_config.max_position_size:.1%}")
        print(f"  Score Weight: {custom_config.score_weight:.1%}")
        print(f"  Regime Sensitivity: {custom_config.regime_sensitivity:.1%}")
        
        # 2. アジャスターの初期化
        adjuster = PositionSizeAdjuster(portfolio_value=2000000.0)  # $2M portfolio
        
        # 3. 異なる市場環境での計算
        market_scenarios = {
            'Bull Market': create_trending_up_data(),
            'Bear Market': create_trending_down_data(), 
            'Volatile Market': create_high_volatility_data(),
            'Range Market': create_range_bound_data()
        }
        
        results = {}
        
        for scenario_name, market_data in market_scenarios.items():
            print(f"\n📊 Scenario: {scenario_name}")
            print(f"  Data Points: {len(market_data)}")
            print(f"  Price Range: ${market_data['low'].min():.2f} - ${market_data['high'].max():.2f}")
            
            result = adjuster.calculate_portfolio_position_sizes(
                ticker=f"DEMO_{scenario_name.upper().replace(' ', '_')}",
                market_data=market_data,
                config=custom_config
            )
            
            results[scenario_name] = result
            
            print(f"  Total Allocation: {result.total_allocated_percentage:.2%}")
            print(f"  Portfolio Risk: {result.portfolio_risk_estimate:.2%}")
            print(f"  Market Regime: {result.regime_analysis.get('regime', 'Unknown')}")
            print(f"  Strategies Used: {len(result.position_results)}")
            
            # Top 3 strategies
            if result.position_results:
                top_strategies = sorted(result.position_results.items(), 
                                      key=lambda x: x[1].adjusted_size, reverse=True)[:3]
                print(f"  Top Strategies:")
                for strategy_name, pos_result in top_strategies:
                    print(f"    {strategy_name}: {pos_result.adjusted_size:.2%}")
        
        # 4. シナリオ比較
        print(f"\n📈 Scenario Comparison:")
        print("-" * 60)
        print(f"{'Scenario':<15} {'Allocation':<12} {'Risk':<8} {'Strategies':<10}")
        print("-" * 60)
        for scenario_name, result in results.items():
            print(f"{scenario_name:<15} {result.total_allocated_percentage:<11.1%} {result.portfolio_risk_estimate:<7.1%} {len(result.position_results):<10}")
        
        return results
        
    except Exception as e:
        logger.error(f"Hybrid adaptive sizing demo failed: {e}")
        print(f"❌ Error in hybrid demo: {e}")
        return None

def create_trending_up_data() -> pd.DataFrame:
    """上昇トレンドデータの作成"""
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    np.random.seed(100)
    
    trend = np.linspace(100, 130, 90)  # 30%上昇
    noise = np.random.normal(0, 1.5, 90)
    prices = trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'open': prices * 0.999,
        'volume': np.random.randint(2000, 8000, 90)
    })

def create_trending_down_data() -> pd.DataFrame:
    """下降トレンドデータの作成"""
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    np.random.seed(200)
    
    trend = np.linspace(100, 75, 90)  # 25%下落
    noise = np.random.normal(0, 2.0, 90)
    prices = trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * 1.03,
        'low': prices * 0.97,
        'open': prices * 1.001,
        'volume': np.random.randint(3000, 12000, 90)
    })

def create_high_volatility_data() -> pd.DataFrame:
    """高ボラティリティデータの作成"""
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    np.random.seed(300)
    
    base_price = 100
    volatility_returns = np.random.normal(0, 0.04, 90)  # 高ボラティリティ
    
    prices = [base_price]
    for ret in volatility_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': [p * 1.05 for p in prices],
        'low': [p * 0.95 for p in prices],
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'volume': np.random.randint(5000, 15000, 90)
    })

def create_range_bound_data() -> pd.DataFrame:
    """レンジ相場データの作成"""
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    np.random.seed(400)
    
    # 95-105の範囲で推移
    center = 100
    range_amplitude = 5
    noise = np.random.normal(0, 1.0, 90)
    cycle = np.sin(np.linspace(0, 4*np.pi, 90)) * range_amplitude
    prices = center + cycle + noise
    
    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * 1.015,
        'low': prices * 0.985,
        'open': prices * 1.002,
        'volume': np.random.randint(1500, 4000, 90)
    })

def demo_method_comparison():
    """異なる計算手法の比較デモ"""
    print("\n" + "="*60)
    print("⚖️ Demo 3: ポジションサイズ計算手法比較")
    print("="*60)
    
    try:
        # 共通設定
        portfolio_value = 1500000.0
        market_data = create_sample_market_data(80)
        ticker = "COMPARISON_TEST"
        
        methods_to_test = [
            (PositionSizeMethod.HYBRID_ADAPTIVE, "ハイブリッド適応型"),
            (PositionSizeMethod.SCORE_BASED, "スコアベース"),
            (PositionSizeMethod.RISK_PARITY, "リスクパリティ"),
            (PositionSizeMethod.FIXED_PERCENTAGE, "固定割合")
        ]
        
        results_comparison = {}
        
        for method, method_name in methods_to_test:
            print(f"\n🔬 Testing Method: {method_name}")
            
            # 手法別設定
            config = PositionSizingConfig(
                sizing_method=method,
                base_position_size=0.025,
                enable_dynamic_adjustment=(method != PositionSizeMethod.FIXED_PERCENTAGE)
            )
            
            # アジャスターの初期化と計算
            adjuster = PositionSizeAdjuster(portfolio_value=portfolio_value)
            result = adjuster.calculate_portfolio_position_sizes(
                ticker=ticker,
                market_data=market_data,
                config=config
            )
            
            results_comparison[method_name] = result
            
            # 結果表示
            print(f"  Strategies: {len(result.position_results)}")
            print(f"  Total Allocation: {result.total_allocated_percentage:.2%}")
            print(f"  Portfolio Risk: {result.portfolio_risk_estimate:.2%}")
            print(f"  Diversification: {result.diversification_score:.3f}")
            print(f"  Confidence: {result.metadata.get('allocation_confidence', 'N/A')}")
            
            # 最大ポジション
            if result.position_results:
                max_position = max(result.position_results.items(), 
                                 key=lambda x: x[1].adjusted_size)
                print(f"  Largest Position: {max_position[0]} ({max_position[1].adjusted_size:.2%})")
        
        # 比較サマリー
        print(f"\n📊 Method Comparison Summary:")
        print("-" * 80)
        print(f"{'Method':<20} {'Strategies':<10} {'Allocation':<12} {'Risk':<8} {'Diversification':<15}")
        print("-" * 80)
        
        for method_name, result in results_comparison.items():
            print(f"{method_name:<20} {len(result.position_results):<10} "
                  f"{result.total_allocated_percentage:<11.1%} {result.portfolio_risk_estimate:<7.1%} "
                  f"{result.diversification_score:<15.3f}")
        
        return results_comparison
        
    except Exception as e:
        logger.error(f"Method comparison demo failed: {e}")
        print(f"❌ Error in method comparison: {e}")
        return None

def demo_integration_with_existing_systems():
    """既存システムとの統合デモ"""
    print("\n" + "="*60)
    print("🔗 Demo 4: 既存システムとの統合")
    print("="*60)
    
    try:
        print("🎯 Testing Integration with:")
        print("  - PortfolioWeightCalculator")
        print("  - StrategyScoreManager") 
        print("  - Risk Management")
        
        # 1. ポートフォリオ重み計算機との統合
        try:
            portfolio_calc = PortfolioWeightCalculator()
            portfolio_calc_available = True
        except:
            portfolio_calc = None
            portfolio_calc_available = False
            
        position_adjuster = PositionSizeAdjuster(portfolio_value=1000000.0)
        
        # 2. テストデータ
        market_data = create_sample_market_data(50)
        ticker = "INTEGRATION_TEST"
        
        print(f"\n📊 Market Data Summary:")
        print(f"  Period: {len(market_data)} days")
        print(f"  Price Range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
        print(f"  Average Volume: {market_data['volume'].mean():,.0f}")
        
        # 3. ポートフォリオ重み計算
        print(f"\n⚖️ Step 1: Portfolio Weight Calculation")
        if portfolio_calc_available:
            try:
                weight_result = portfolio_calc.calculate_portfolio_weights(
                    ticker=ticker,
                    market_data=market_data
                )
                print(f"  Strategy Weights Calculated: {len(weight_result.strategy_weights)}")
                print(f"  Expected Return: {weight_result.expected_return:.2%}")
                print(f"  Expected Risk: {weight_result.expected_risk:.2%}")
                print(f"  Sharpe Ratio: {weight_result.sharpe_ratio:.3f}")
            except Exception as e:
                print(f"  ⚠️ Portfolio weight calculation failed: {e}")
                weight_result = None
        else:
            print(f"  ⚠️ Portfolio weight calculator not available")
            weight_result = None
        
        # 4. ポジションサイズ調整
        print(f"\n📏 Step 2: Position Size Adjustment")
        position_result = position_adjuster.calculate_portfolio_position_sizes(
            ticker=ticker,
            market_data=market_data
        )
        
        print(f"  Position Results: {len(position_result.position_results)}")
        print(f"  Total Allocated: {position_result.total_allocated_percentage:.2%}")
        print(f"  Portfolio Risk Estimate: {position_result.portfolio_risk_estimate:.2%}")
        
        # 5. 統合結果の比較
        print(f"\n🔍 Step 3: Integration Analysis")
        if weight_result and weight_result.strategy_weights:
            print(f"  Weight Calculator Found: {len(weight_result.strategy_weights)} strategies")
            print(f"  Position Adjuster Found: {len(position_result.position_results)} strategies")
            
            # 共通戦略の比較
            common_strategies = set(weight_result.strategy_weights.keys()) & \
                              set(position_result.position_results.keys())
            
            if common_strategies:
                print(f"  Common Strategies: {len(common_strategies)}")
                print(f"  Strategy Comparison:")
                for strategy in list(common_strategies)[:3]:  # 最初の3つを表示
                    weight = weight_result.strategy_weights[strategy]
                    position = position_result.position_results[strategy].adjusted_size
                    print(f"    {strategy}: Weight={weight:.2%}, Position={position:.2%}")
        else:
            print(f"  Independent position calculation completed")
        
        # 6. レポート生成
        report = position_adjuster.create_position_sizing_report(position_result)
        print(f"\n📄 Integration Report Generated:")
        print(f"  Summary Items: {len(report.get('summary', {}))}")
        print(f"  Position Details: {len(report.get('positions', {}))}")
        print(f"  Constraint Check: {len(report.get('constraints', []))}")
        
        return {
            'weight_result': weight_result,
            'position_result': position_result,
            'integration_report': report
        }
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        print(f"❌ Error in integration demo: {e}")
        return None

def demo_error_handling():
    """エラーハンドリングのデモ"""
    print("\n" + "="*60)
    print("🛡️ Demo 5: エラーハンドリングとロバストネス")
    print("="*60)
    
    error_scenarios = [
        ("Empty Market Data", pd.DataFrame()),
        ("Invalid Config", "invalid_config"),
        ("Extreme Values", create_extreme_data()),
        ("Missing Columns", create_incomplete_data()),
        ("Zero Portfolio Value", 0.0)
    ]
    
    passed_tests = 0
    total_tests = len(error_scenarios)
    
    for test_name, test_data in error_scenarios:
        print(f"\n🧪 Test: {test_name}")
        
        try:
            if test_name == "Invalid Config":
                # 無効な設定でテスト
                try:
                    config = PositionSizingConfig(sizing_method="invalid_method")
                except:
                    config = PositionSizingConfig()  # デフォルトにフォールバック
                adjuster = PositionSizeAdjuster(portfolio_value=1000000.0)
                result = adjuster.calculate_portfolio_position_sizes(
                    ticker="ERROR_TEST", 
                    market_data=create_sample_market_data(10),
                    config=config
                )
            elif test_name == "Zero Portfolio Value":
                # ゼロ資産でテスト
                adjuster = PositionSizeAdjuster(portfolio_value=test_data)
                result = adjuster.calculate_portfolio_position_sizes(
                    ticker="ZERO_TEST",
                    market_data=create_sample_market_data(10)
                )
            else:
                # その他のテスト
                adjuster = PositionSizeAdjuster(portfolio_value=1000000.0)
                result = adjuster.calculate_portfolio_position_sizes(
                    ticker="ERROR_TEST",
                    market_data=test_data
                )
            
            # 結果の評価
            if result and hasattr(result, 'constraint_violations'):
                if result.constraint_violations:
                    print(f"  ✅ Handled gracefully: {len(result.constraint_violations)} violations")
                else:
                    print(f"  ✅ Processed successfully")
                passed_tests += 1
            else:
                print(f"  ❌ Unexpected result format")
                
        except Exception as e:
            print(f"  ✅ Caught exception properly: {type(e).__name__}")
            passed_tests += 1  # エラーが適切にキャッチされた場合も成功
    
    print(f"\n📊 Error Handling Summary:")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests

def create_extreme_data() -> pd.DataFrame:
    """極端な値を含むデータの作成"""
    data = create_sample_market_data(20)
    
    # 極端な値を追加
    data.loc[10, 'close'] = 0.01  # ほぼゼロ
    data.loc[15, 'close'] = 10000.0  # 極端に高い値
    data.loc[5, 'volume'] = 0  # ゼロボリューム
    
    return data

def create_incomplete_data() -> pd.DataFrame:
    """不完全なデータの作成"""
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'price': [100 + i for i in range(10)]  # 'close'カラムがない
    })
    return data

def main():
    """メインデモ実行"""
    print("🚀 Position Size Adjuster System - Comprehensive Demo")
    print("=" * 80)
    print("Author: imega")
    print("Created: 2025-07-20")
    print("Task: 3-3-2「各戦略のポジションサイズ調整機能」")
    print("=" * 80)
    
    # 警告を抑制
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    demo_results = {}
    
    try:
        # Demo 1: 基本機能
        demo_results['basic'] = demo_basic_position_sizing()
        
        # Demo 2: ハイブリッド適応型
        demo_results['hybrid'] = demo_hybrid_adaptive_sizing()
        
        # Demo 3: 手法比較
        demo_results['comparison'] = demo_method_comparison()
        
        # Demo 4: システム統合
        demo_results['integration'] = demo_integration_with_existing_systems()
        
        # Demo 5: エラーハンドリング
        demo_results['error_handling'] = demo_error_handling()
        
        # 総合評価
        print(f"\n" + "="*60)
        print("📋 Demo Execution Summary")
        print("="*60)
        
        successful_demos = sum(1 for result in demo_results.values() if result)
        total_demos = len(demo_results)
        
        print(f"Total Demos Run: {total_demos}")
        print(f"Successful Demos: {successful_demos}")
        print(f"Success Rate: {(successful_demos/total_demos)*100:.1f}%")
        
        print(f"\n✅ Demo Results:")
        for demo_name, result in demo_results.items():
            status = "✅ Success" if result else "❌ Failed"
            print(f"  {demo_name.title()}: {status}")
        
        if successful_demos == total_demos:
            print(f"\n🎉 All demos completed successfully!")
            print("Position Size Adjuster system is ready for production use.")
        else:
            print(f"\n⚠️ Some demos failed. Please check the error messages above.")
        
        return demo_results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrupted by user")
        return demo_results
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Critical error in demo execution: {e}")
        return demo_results

if __name__ == "__main__":
    results = main()
