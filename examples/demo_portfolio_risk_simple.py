"""
Demo: Simplified Portfolio Risk Management System
File: demo_portfolio_risk_simple.py
Description: 3-3-3「ポートフォリオレベルのリスク調整機能」の簡易デモンストレーション

Author: imega
Created: 2025-07-20
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

try:
    from config.portfolio_risk_manager import (
        PortfolioRiskManager, RiskConfiguration, 
        PortfolioWeightCalculator, PositionSizeAdjuster, SignalIntegrator
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

def generate_simple_test_data():
    """簡単なテスト用データを生成"""
    print("📊 Generating test data...")
    
    # 戦略リターンデータを生成
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    returns_df = pd.DataFrame({
        'momentum_strategy': np.random.normal(0.0008, 0.015, 100),
        'mean_reversion_strategy': np.random.normal(0.0003, 0.012, 100),
        'trend_following_strategy': np.random.normal(0.0010, 0.020, 100),
        'arbitrage_strategy': np.random.normal(0.0002, 0.008, 100)
    }, index=dates)
    
    weights_dict = {
        'momentum_strategy': 0.3,
        'mean_reversion_strategy': 0.25,
        'trend_following_strategy': 0.25,
        'arbitrage_strategy': 0.2
    }
    
    return returns_df, weights_dict

def demo_basic_risk_calculation():
    """基本的なリスク計算のデモ"""
    print("\n🔍 Demo: Basic Risk Calculation")
    print("=" * 50)
    
    try:
        # テストデータの生成
        returns_df, weights_dict = generate_simple_test_data()
        
        # リスク設定
        risk_config = RiskConfiguration(
            var_95_limit=0.03,      # 3%
            var_99_limit=0.05,      # 5%
            max_drawdown_limit=0.10, # 10%
            volatility_limit=0.20,  # 20%
            max_correlation=0.7,    # 70%
            max_single_position=0.35 # 35%
        )
        
        # ダミー依存関係の作成
        weight_calculator = PortfolioWeightCalculator(None)
        position_adjuster = PositionSizeAdjuster("dummy_config.json")
        signal_integrator = SignalIntegrator()
        
        # リスク管理システムの初期化
        risk_manager = PortfolioRiskManager(
            config=risk_config,
            portfolio_weight_calculator=weight_calculator,
            position_size_adjuster=position_adjuster,
            signal_integrator=signal_integrator
        )
        
        print("✅ Portfolio Risk Manager initialized successfully")
        
        # リスク評価実行
        print("🔄 Running risk assessment...")
        risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
            returns_df, weights_dict
        )
        
        print(f"\n📊 Portfolio Risk Assessment Results:")
        print(f"  📋 Total strategies: {len(weights_dict)}")
        print(f"  ⚠️  Needs adjustment: {needs_adjustment}")
        print(f"  📈 Risk metrics calculated: {len(risk_metrics)}")
        
        # 各リスク指標の詳細
        if risk_metrics:
            print(f"\n📈 Risk Metrics Details:")
            for metric_name, metric in risk_metrics.items():
                status = "🔴 BREACH" if metric.is_breached else "🟢 OK"
                print(f"  {metric_name:20s}: {metric.current_value:.4f} / {metric.limit_value:.4f} {status}")
                if metric.is_breached:
                    print(f"    └─ Severity: {metric.breach_severity:.3f}, Type: {metric.limit_type.value}")
        
        return risk_manager, returns_df, weights_dict, needs_adjustment
        
    except Exception as e:
        print(f"❌ Error in basic risk calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False

def demo_risk_adjustment():
    """リスク調整のデモ"""
    print("\n⚙️  Demo: Risk Adjustment")
    print("=" * 50)
    
    risk_manager, returns_df, weights_dict, needs_adjustment = demo_basic_risk_calculation()
    
    if not risk_manager:
        print("❌ Risk manager initialization failed")
        return None
    
    # 強制的に調整が必要な状況を作成（集中度を高める）
    print(f"\n📊 Creating high concentration scenario...")
    test_weights = {
        'momentum_strategy': 0.6,     # 60%に集中
        'mean_reversion_strategy': 0.15,
        'trend_following_strategy': 0.15,
        'arbitrage_strategy': 0.10
    }
    
    print(f"\n📊 High concentration test weights:")
    for strategy, weight in test_weights.items():
        print(f"  {strategy:25s}: {weight:.3f}")
    
    # リスク評価
    print("🔄 Running high concentration risk assessment...")
    risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
        returns_df, test_weights
    )
    
    print(f"\n⚠️  High concentration assessment:")
    print(f"  Needs adjustment: {needs_adjustment}")
    
    if needs_adjustment:
        print("🔄 Running risk adjustment...")
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

def demo_risk_monitoring():
    """リスクモニタリングのデモ"""
    print("\n📡 Demo: Risk Monitoring & Reporting")
    print("=" * 50)
    
    risk_manager, returns_df, weights_dict, _ = demo_basic_risk_calculation()
    
    if not risk_manager:
        print("❌ Risk manager not available for monitoring demo")
        return
    
    # 複数回のリスク評価を実行（履歴を蓄積）
    print("📈 Building risk assessment history...")
    
    for i in range(3):
        # ウェイトを少しずつ変更
        test_weights = weights_dict.copy()
        for strategy in test_weights:
            test_weights[strategy] *= (1 + np.random.normal(0, 0.05))
        
        # 正規化
        total_weight = sum(test_weights.values())
        if total_weight > 0:
            test_weights = {k: v / total_weight for k, v in test_weights.items()}
        
        # リスク評価実行
        risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
            returns_df, test_weights
        )
        
        if needs_adjustment:
            risk_manager.adjust_portfolio_weights(
                returns_df, test_weights, risk_metrics
            )
        
        print(f"  📊 Assessment {i+1} completed - Adjustment needed: {needs_adjustment}")
    
    # サマリー取得
    print("\n🔄 Generating risk summary...")
    summary = risk_manager.get_risk_summary()
    
    print(f"\n📊 Risk Summary:")
    print(f"  Status: {summary.get('status', 'unknown')}")
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
        print(f"  Actions: {adj['actions']}")
        print(f"  Effectiveness: {adj['effectiveness_score']:.3f}")
        print(f"  Reason: {adj['reason']}")
    
    # レポート保存
    report_path = "portfolio_risk_report.json"
    try:
        if risk_manager.save_risk_report(report_path):
            print(f"\n💾 Risk report saved to: {report_path}")
        else:
            print(f"\n❌ Failed to save risk report")
    except Exception as e:
        print(f"\n⚠️  Report save error: {e}")

def main():
    """メインデモ実行"""
    print("🎯 Portfolio Risk Management System - Simple Demo")
    print("=" * 60)
    print("3-3-3「ポートフォリオレベルのリスク調整機能」")
    print("=" * 60)
    
    try:
        # 1. 基本的なリスク計算
        demo_basic_risk_calculation()
        
        # 2. リスク調整
        demo_risk_adjustment()
        
        # 3. リスクモニタリング
        demo_risk_monitoring()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"📊 Portfolio risk management system is operational")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
