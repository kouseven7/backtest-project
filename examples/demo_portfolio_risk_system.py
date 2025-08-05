"""
Demo: Portfolio Risk Management System
File: demo_portfolio_risk_system.py
Description: 3-3-3「ポートフォリオレベルのリスク調整機能」のデモンストレーション

Author: imega
Created: 2025-07-20
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

try:
    from config.portfolio_risk_manager import (
        PortfolioRiskManager, RiskConfiguration, RiskMetricType,
        IntegratedRiskManagementSystem, RiskLimitType
    )
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, AllocationMethod
    )
    from config.position_size_adjuster import (
        PositionSizeAdjuster, PositionSizingConfig, PositionSizeMethod
    )
    from config.signal_integrator import SignalIntegrator, StrategySignal, SignalType
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

def generate_test_data():
    """テスト用データを生成"""
    print("📊 Generating test data...")
    
    # 戦略リターンデータを生成（異なる特性を持つ戦略）
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    strategies_data = {
        'momentum_strategy': {
            'returns': np.random.normal(0.0008, 0.015, 252),  # 高リターン・中リスク
            'weight': 0.3
        },
        'mean_reversion_strategy': {
            'returns': np.random.normal(0.0003, 0.012, 252),  # 中リターン・低リスク
            'weight': 0.25
        },
        'trend_following_strategy': {
            'returns': np.random.normal(0.0010, 0.020, 252),  # 高リターン・高リスク
            'weight': 0.25
        },
        'arbitrage_strategy': {
            'returns': np.random.normal(0.0002, 0.008, 252),  # 低リターン・超低リスク
            'weight': 0.2
        }
    }
    
    # 相関構造を追加（一部戦略に相関を持たせる）
    for i in range(252):
        if i > 0:
            # momentum と trend_following に正の相関
            if np.random.random() < 0.3:
                correlation_factor = np.random.normal(0.6, 0.1)
                strategies_data['trend_following_strategy']['returns'][i] += (
                    strategies_data['momentum_strategy']['returns'][i] * correlation_factor
                )
            
            # 市場ショック時の連動性
            if np.random.random() < 0.05:  # 5%の確率でショック
                shock_factor = np.random.uniform(-0.05, -0.02)
                for strategy in strategies_data:
                    if strategy != 'arbitrage_strategy':  # アービトラージは市場中性
                        strategies_data[strategy]['returns'][i] *= (1 + shock_factor)
    
    # DataFrameに変換
    returns_df = pd.DataFrame({
        strategy: data['returns'] 
        for strategy, data in strategies_data.items()
    }, index=dates)
    
    weights_dict = {
        strategy: data['weight'] 
        for strategy, data in strategies_data.items()
    }
    
    return returns_df, weights_dict

def demo_basic_risk_calculation():
    """基本的なリスク計算のデモ"""
    print("\n🔍 Demo: Basic Risk Calculation")
    print("=" * 50)
    
    # テストデータの生成
    returns_df, weights_dict = generate_test_data()
    
    # リスク設定（厳しめに設定してテスト）
    risk_config = RiskConfiguration(
        var_95_limit=0.03,      # 3%
        var_99_limit=0.05,      # 5%
        max_drawdown_limit=0.10, # 10%
        volatility_limit=0.20,  # 20%
        max_correlation=0.7,    # 70%
        max_single_position=0.35 # 35%
    )
    
    # ダミー依存関係の作成
    try:
        # ダミーのポートフォリオウェイト計算器
        weight_calculator = PortfolioWeightCalculator(None)
        
        # ダミーのポジションサイズ調整器（設定ファイルパスで初期化）
        position_adjuster = PositionSizeAdjuster("dummy_config.json")
        
        signal_integrator = SignalIntegrator()
        
        # リスク管理システムの初期化
        risk_manager = PortfolioRiskManager(
            config=risk_config,
            portfolio_weight_calculator=weight_calculator,
            position_size_adjuster=position_adjuster,
            signal_integrator=signal_integrator
        )
        
        # リスク評価実行
        risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
            returns_df, weights_dict
        )
        
        print(f"Portfolio Risk Assessment Results:")
        print(f"  📋 Total strategies: {len(weights_dict)}")
        print(f"  ⚠️  Needs adjustment: {needs_adjustment}")
        print(f"  📊 Risk metrics calculated: {len(risk_metrics)}")
        
        # 各リスク指標の詳細
        print(f"\n📈 Risk Metrics Details:")
        for metric_name, metric in risk_metrics.items():
            status = "🔴 BREACH" if metric.is_breached else "🟢 OK"
            print(f"  {metric_name:20s}: {metric.current_value:.4f} / {metric.limit_value:.4f} {status}")
            if metric.is_breached:
                print(f"    └─ Severity: {metric.breach_severity:.3f}, Type: {metric.limit_type.value}")
        
        return risk_manager, returns_df, weights_dict, needs_adjustment
        
    except Exception as e:
        print(f"❌ Error in basic risk calculation: {e}")
        return None, returns_df, weights_dict, False

def demo_risk_adjustment():
    """リスク調整のデモ"""
    print("\n⚙️  Demo: Risk Adjustment")
    print("=" * 50)
    
    risk_manager, returns_df, weights_dict, needs_adjustment = demo_basic_risk_calculation()
    
    if not risk_manager:
        print("❌ Risk manager initialization failed")
        return None
    
    # 強制的に調整が必要な状況を作成
    print(f"\n📊 Original portfolio weights:")
    for strategy, weight in weights_dict.items():
        print(f"  {strategy:25s}: {weight:.3f}")
    
    # 集中度を高めてテスト
    test_weights = weights_dict.copy()
    test_weights['momentum_strategy'] = 0.6  # 60%に集中
    test_weights['mean_reversion_strategy'] = 0.15
    test_weights['trend_following_strategy'] = 0.15
    test_weights['arbitrage_strategy'] = 0.10
    
    print(f"\n📊 High concentration test weights:")
    for strategy, weight in test_weights.items():
        print(f"  {strategy:25s}: {weight:.3f}")
    
    # リスク評価
    risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
        returns_df, test_weights
    )
    
    print(f"\n⚠️  High concentration assessment:")
    print(f"  Needs adjustment: {needs_adjustment}")
    
    if needs_adjustment:
        # 調整実行
        adjustment_result = risk_manager.adjust_portfolio_weights(
            returns_df, test_weights, risk_metrics
        )
        
        print(f"\n⚙️  Risk Adjustment Results:")
        print(f"  Timestamp: {adjustment_result.timestamp}")
        print(f"  Actions: {[action.value for action in adjustment_result.adjustment_actions]}")
        print(f"  Effectiveness: {adjustment_result.effectiveness_score:.3f}")
        print(f"  Reason: {adjustment_result.adjustment_reason}")
        
        print(f"\n📊 Weight Changes:")
        weight_changes = adjustment_result.get_weight_changes()
        for strategy, change in weight_changes.items():
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"  {strategy:25s}: {change:+.3f} {direction}")
        
        print(f"\n📊 Final adjusted weights:")
        for strategy, weight in adjustment_result.adjusted_weights.items():
            print(f"  {strategy:25s}: {weight:.3f}")
        
        return adjustment_result
    else:
        print("✅ No adjustment needed")
        return None

def demo_integrated_system():
    """統合システムのデモ"""
    print("\n🔧 Demo: Integrated Risk Management System")
    print("=" * 60)
    
    try:
        # 設定の準備
        risk_config = RiskConfiguration(
            var_95_limit=0.04,
            var_99_limit=0.06,
            max_drawdown_limit=0.12,
            volatility_limit=0.22,
            max_correlation=0.75,
            max_single_position=0.30
        )
        
        weight_config = WeightAllocationConfig(
            allocation_method=AllocationMethod.SCORE_PROPORTIONAL,
            rebalance_frequency=5
        )
        
        # ポジション調整設定（設定ファイルパスで初期化）
        position_adjuster_config_path = "dummy_position_config.json"
        
        # 統合システム初期化
        integrated_system = IntegratedRiskManagementSystem(
            risk_config=risk_config,
            weight_config=weight_config,
            adjustment_config=position_adjuster_config_path
        )
        
        # テストデータ準備
        returns_df, _ = generate_test_data()
        
        # ダミーシグナルデータ
        strategy_signals = {
            'momentum_strategy': StrategySignal(
                strategy_name='momentum_strategy',
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.75,
                timestamp=datetime.now()
            ),
            'mean_reversion_strategy': StrategySignal(
                strategy_name='mean_reversion_strategy',
                signal_type=SignalType.ENTRY_SHORT,
                confidence=0.65,
                timestamp=datetime.now()
            ),
            'trend_following_strategy': StrategySignal(
                strategy_name='trend_following_strategy',
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.85,
                timestamp=datetime.now()
            ),
            'arbitrage_strategy': StrategySignal(
                strategy_name='arbitrage_strategy',
                signal_type=SignalType.ENTRY_LONG,
                confidence=0.90,
                timestamp=datetime.now()
            )
        }
        
        # ダミー市場データ
        market_data = pd.DataFrame({
            'price': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 100000, 100)
        })
        
        print("🚀 Running complete portfolio management flow...")
        
        # 完全なポートフォリオ管理フロー実行
        result = integrated_system.run_complete_portfolio_management(
            returns_data=returns_df,
            strategy_signals=strategy_signals,
            market_data=market_data
        )
        
        print(f"\n📊 Integrated System Results:")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Total Effectiveness: {result.get('total_effectiveness', 0.0):.3f}")
        
        if 'final_weights' in result:
            print(f"\n📊 Final Portfolio Weights:")
            for strategy, weight in result['final_weights'].items():
                print(f"  {strategy:25s}: {weight:.3f}")
        
        # エラーハンドリング
        if result.get('status') == 'error':
            print(f"❌ System Error: {result.get('message', 'Unknown error')}")
        else:
            print("✅ Integrated system completed successfully")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in integrated system demo: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_risk_monitoring():
    """リスクモニタリングのデモ"""
    print("\n📡 Demo: Risk Monitoring & Reporting")
    print("=" * 50)
    
    # 基本デモから結果を取得
    risk_manager, returns_df, weights_dict, _ = demo_basic_risk_calculation()
    
    if not risk_manager:
        print("❌ Risk manager not available for monitoring demo")
        return
    
    # 複数回のリスク評価を実行（履歴を蓄積）
    print("📈 Running multiple risk assessments to build history...")
    
    for i in range(5):
        # ウェイトを少しずつ変更
        test_weights = weights_dict.copy()
        for strategy in test_weights:
            test_weights[strategy] *= (1 + np.random.normal(0, 0.1))
        
        # 正規化
        total_weight = sum(test_weights.values())
        test_weights = {k: v / total_weight for k, v in test_weights.items()}
        
        # リスク評価実行
        risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
            returns_df, test_weights
        )
        
        if needs_adjustment:
            adjustment_result = risk_manager.adjust_portfolio_weights(
                returns_df, test_weights, risk_metrics
            )
    
    # サマリー取得
    summary = risk_manager.get_risk_summary()
    
    print(f"\n📊 Risk Summary:")
    print(f"  Status: {summary.get('status', 'unknown')}")
    print(f"  Total Strategies: {summary.get('total_strategies', 0)}")
    print(f"  Adjustment History Count: {summary.get('adjustment_history_count', 0)}")
    
    # リスク指標サマリー
    if 'risk_metrics' in summary:
        print(f"\n📈 Current Risk Metrics:")
        for metric_name, metric_data in summary['risk_metrics'].items():
            status = "🔴" if metric_data.get('is_breached', False) else "🟢"
            print(f"  {metric_name:20s}: {metric_data.get('current_value', 0):.4f} {status}")
    
    # 制限違反
    if summary.get('breaches'):
        print(f"\n⚠️  Current Breaches:")
        for breach in summary['breaches']:
            print(f"  - {breach['metric']}: Severity {breach['severity']:.3f} ({breach['limit_type']})")
    else:
        print(f"\n✅ No current risk limit breaches")
    
    # 最新調整情報
    if summary.get('last_adjustment'):
        adj = summary['last_adjustment']
        print(f"\n⚙️  Last Adjustment:")
        print(f"  Timestamp: {adj['timestamp']}")
        print(f"  Actions: {adj['actions']}")
        print(f"  Effectiveness: {adj['effectiveness_score']:.3f}")
        print(f"  Reason: {adj['reason']}")
    
    # レポート保存
    report_path = "portfolio_risk_report.json"
    if risk_manager.save_risk_report(report_path):
        print(f"\n💾 Risk report saved to: {report_path}")
    else:
        print(f"\n❌ Failed to save risk report")

def main():
    """メインデモ実行"""
    print("🎯 Portfolio Risk Management System Demo")
    print("=" * 60)
    print("3-3-3「ポートフォリオレベルのリスク調整機能」")
    print("=" * 60)
    
    try:
        # 1. 基本的なリスク計算
        demo_basic_risk_calculation()
        
        # 2. リスク調整
        demo_risk_adjustment()
        
        # 3. 統合システム
        demo_integrated_system()
        
        # 4. リスクモニタリング
        demo_risk_monitoring()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"📊 Check 'portfolio_risk_report.json' for detailed risk analysis")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
