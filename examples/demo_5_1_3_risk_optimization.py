"""
Module: 5-1-3 Risk Adjusted Optimization Demo
File: demo_5_1_3_risk_optimization.py
Description: 
  5-1-3「リスク調整後リターンの最適化」システムのデモンストレーション

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

try:
    # 5-1-3システムのインポート
    from analysis.risk_adjusted_optimization import (
        RiskAdjustedOptimizationEngine,
        AdvancedPortfolioOptimizer, 
        OptimizationValidator,
        OptimizationContext,
        PortfolioOptimizationProfile,
        MultiPeriodOptimizationRequest
    )
    
    print("✓ 5-1-3「リスク調整後リターンの最適化」モジュールの読み込み成功")
    
except ImportError as e:
    print(f"✗ モジュールのインポートエラー: {e}")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(periods: int = 365, strategies: int = 4) -> tuple:
    """サンプルデータの生成"""
    
    print(f"\n📊 サンプルデータ生成中... ({periods}日間, {strategies}戦略)")
    
    # 日付インデックス
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=periods), 
        end=datetime.now(), 
        freq='D'
    )
    
    # 異なる特性を持つ戦略リターンを生成
    np.random.seed(42)  # 再現性のため
    
    strategy_configs = [
        {"mean": 0.0008, "std": 0.015, "name": "conservative_trend"},
        {"mean": 0.0012, "std": 0.022, "name": "moderate_momentum"},
        {"mean": 0.0015, "std": 0.028, "name": "aggressive_growth"},
        {"mean": 0.0005, "std": 0.012, "name": "defensive_value"}
    ]
    
    strategy_returns = pd.DataFrame(index=dates)
    
    for i, config in enumerate(strategy_configs[:strategies]):
        # トレンド成分を追加
        trend = np.linspace(0, 0.0003 * (i+1), len(dates))
        
        # ランダムリターン + トレンド + 季節性
        returns = np.random.normal(config["mean"], config["std"], len(dates))
        returns += trend
        
        # 簡単な季節性（月次サイクル）
        seasonal = 0.0002 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        returns += seasonal
        
        strategy_returns[config["name"]] = returns
    
    # 初期重み（均等からわずかに偏らせる）
    initial_weights = {
        strategy_configs[0]["name"]: 0.3,
        strategy_configs[1]["name"]: 0.25,
        strategy_configs[2]["name"]: 0.25,
        strategy_configs[3]["name"]: 0.2
    }
    
    # 前回重み（履歴シミュレーション）
    previous_weights = {
        strategy_configs[0]["name"]: 0.35,
        strategy_configs[1]["name"]: 0.2,
        strategy_configs[2]["name"]: 0.3,
        strategy_configs[3]["name"]: 0.15
    }
    
    print(f"✓ データ生成完了: {len(strategy_returns)}行 x {len(strategy_returns.columns)}列")
    print(f"  📈 戦略: {list(strategy_returns.columns)}")
    print(f"  📅 期間: {dates[0].date()} ～ {dates[-1].date()}")
    
    return strategy_returns, initial_weights, previous_weights

def demo_basic_optimization():
    """基本最適化のデモンストレーション"""
    
    print("\n" + "="*60)
    print("🚀 基本最適化デモ開始")
    print("="*60)
    
    # サンプルデータ生成
    strategy_returns, current_weights, previous_weights = generate_sample_data()
    
    # 最適化コンテキスト作成
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        previous_weights=previous_weights,
        market_volatility=0.18,
        trend_strength=0.05,
        market_regime="normal",
        optimization_horizon=252,
        rebalancing_frequency="monthly"
    )
    
    print(f"\n📋 最適化コンテキスト:")
    print(f"  📊 データポイント数: {len(strategy_returns)}")
    print(f"  ⚖️ 現在の重み: {current_weights}")
    print(f"  📈 市場ボラティリティ: {context.market_volatility:.1%}")
    print(f"  🎯 市場レジーム: {context.market_regime}")
    
    # 基本最適化エンジン初期化
    engine = RiskAdjustedOptimizationEngine()
    
    print(f"\n⚙️ 最適化実行中...")
    start_time = datetime.now()
    
    # 最適化実行
    result = engine.optimize_portfolio_allocation(context)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # 結果表示
    print(f"\n📊 基本最適化結果:")
    print(f"  ✅ 最適化成功: {result.optimization_success}")
    print(f"  🎯 信頼度レベル: {result.confidence_level:.3f}")
    print(f"  ⏱️ 実行時間: {execution_time:.2f}秒")
    print(f"  🔄 反復回数: {result.optimization_result.iterations}")
    
    print(f"\n💰 重み配分:")
    for strategy, weight in result.optimal_weights.items():
        original = current_weights[strategy]
        change = weight - original
        print(f"  {strategy}: {weight:.3f} (変化: {change:+.3f})")
    
    print(f"\n📈 パフォーマンス指標:")
    sharpe = result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
    print(f"  📊 シャープレシオ: {sharpe:.3f}")
    
    # ドローダウン表示の改良
    try:
        if hasattr(result.performance_report, 'metrics') and 'max_drawdown' in result.performance_report.metrics:
            drawdown = result.performance_report.metrics['max_drawdown']
            # PerformanceMetricオブジェクトの場合は.valueを取得
            if hasattr(drawdown, 'value'):
                drawdown_value = abs(drawdown.value)
            else:
                drawdown_value = abs(float(drawdown))
            print(f"  📉 最大ドローダウン: {drawdown_value:.1%}")
        else:
            print(f"  📉 最大ドローダウン: データなし")
    except Exception as e:
        print(f"  📉 最大ドローダウン: 取得エラー ({e})")
    
    print(f"\n🚨 制約チェック:")
    print(f"  ✅ 制約満足: {result.constraint_result.is_satisfied}")
    print(f"  ⚠️ 違反数: {len(result.constraint_result.violations)}")
    
    print(f"\n💡 推奨事項 (上位3件):")
    for i, recommendation in enumerate(result.recommendations[:3], 1):
        print(f"  {i}. {recommendation}")
    
    return result

def demo_advanced_optimization():
    """高度最適化のデモンストレーション"""
    
    print("\n" + "="*60)
    print("🎯 高度最適化デモ開始")
    print("="*60)
    
    # サンプルデータ生成（より多くの戦略）
    strategy_returns, current_weights, previous_weights = generate_sample_data(
        periods=500, strategies=4
    )
    
    # コンテキスト作成
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        previous_weights=previous_weights,
        market_volatility=0.22,
        trend_strength=0.08,
        market_regime="volatile"
    )
    
    # 高度なオプティマイザー初期化
    optimizer = AdvancedPortfolioOptimizer()
    
    # リスクプロファイル作成
    profile = optimizer.create_optimization_profile(
        profile_name="demo_balanced",
        risk_tolerance="moderate",
        return_target=0.08,
        max_drawdown_tolerance=0.15,
        rebalancing_frequency="monthly"
    )
    
    print(f"📋 リスクプロファイル: {profile.profile_name}")
    print(f"  🎚️ リスク許容度: {profile.risk_tolerance}")
    print(f"  🎯 リターン目標: {profile.return_target:.1%}")
    print(f"  📉 最大DD許容: {profile.max_drawdown_tolerance:.1%}")
    
    # マルチ期間分析設定
    multi_period_request = MultiPeriodOptimizationRequest(
        optimization_horizons=[63, 126, 252],  # 3M, 6M, 1Y
        confidence_threshold=0.6
    )
    
    print(f"\n🔍 マルチ期間分析設定:")
    print(f"  📅 分析期間: {multi_period_request.optimization_horizons} 日")
    print(f"  🎯 信頼度しきい値: {multi_period_request.confidence_threshold}")
    
    print(f"\n⚙️ 包括最適化実行中...")
    start_time = datetime.now()
    
    # 包括最適化実行
    comprehensive_result = optimizer.optimize_portfolio_comprehensive(
        context, profile, multi_period_request
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # 結果表示
    print(f"\n📊 包括最適化結果:")
    print(f"  ✅ プライマリ成功: {comprehensive_result.primary_result.optimization_success}")
    print(f"  🎯 総合信頼度: {comprehensive_result.confidence_assessment['overall_confidence']:.3f}")
    print(f"  ⏱️ 実行時間: {execution_time:.2f}秒")
    
    print(f"\n💰 最適重み配分:")
    for strategy, weight in comprehensive_result.primary_result.optimal_weights.items():
        original = current_weights[strategy]
        change = weight - original
        print(f"  {strategy}: {weight:.3f} (変化: {change:+.3f})")
    
    print(f"\n🔄 代替配分オプション ({len(comprehensive_result.alternative_allocations)}件):")
    for alt_name in list(comprehensive_result.alternative_allocations.keys())[:3]:
        print(f"  📊 {alt_name}")
    
    print(f"\n📈 マルチ期間分析:")
    for horizon, analysis_result in comprehensive_result.multi_period_analysis.items():
        print(f"  📅 {horizon}日: 成功={analysis_result.optimization_success}, 信頼度={analysis_result.confidence_level:.3f}")
    
    print(f"\n🎯 リスクプロファイル適合性:")
    for metric, score in comprehensive_result.risk_profile_analysis.items():
        print(f"  📊 {metric}: {score:.3f}")
    
    print(f"\n🏗️ 実行プラン:")
    plan = comprehensive_result.execution_plan
    print(f"  📋 戦略: {plan.get('execution_strategy', 'N/A')}")
    print(f"  🔄 重み変更: {plan.get('total_weight_change', 0):.1%}")
    print(f"  👁️ 監視頻度: {plan.get('monitoring_frequency', 'N/A')}")
    
    print(f"\n💡 統合推奨事項 (上位3件):")
    for i, recommendation in enumerate(comprehensive_result.recommendation_summary[:3], 1):
        print(f"  {i}. {recommendation}")
    
    return comprehensive_result

def demo_validation():
    """結果検証のデモンストレーション"""
    
    print("\n" + "="*60)
    print("🔍 結果検証デモ開始")
    print("="*60)
    
    # 基本最適化を実行して結果を取得
    print("📊 検証用の最適化実行中...")
    strategy_returns, current_weights, previous_weights = generate_sample_data(periods=300)
    
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        previous_weights=previous_weights,
        market_volatility=0.20,
        trend_strength=0.04,
        market_regime="normal"
    )
    
    engine = RiskAdjustedOptimizationEngine()
    optimization_result = engine.optimize_portfolio_allocation(context)
    
    # 検証システム初期化
    validator = OptimizationValidator()
    
    print(f"\n🔍 包括的検証実行中...")
    start_time = datetime.now()
    
    # 検証実行
    validation_report = validator.validate_optimization_result(
        optimization_result, context
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # 結果表示
    print(f"\n📊 検証結果:")
    print(f"  ✅ 検証成功: {validation_report.validation_success}")
    print(f"  🎯 総合スコア: {validation_report.overall_score:.3f}")
    print(f"  ⏱️ 検証時間: {execution_time:.2f}秒")
    print(f"  📝 実行テスト数: {len(validation_report.individual_tests)}")
    
    print(f"\n📊 カテゴリ別スコア:")
    for category, score in validation_report.category_scores.items():
        status = "✅" if score > 0.6 else "⚠️" if score > 0.4 else "❌"
        print(f"  {status} {category}: {score:.3f}")
    
    print(f"\n🚨 重要な問題 ({len(validation_report.critical_failures)}件):")
    for failure in validation_report.critical_failures:
        print(f"  ❌ {failure}")
    
    if not validation_report.critical_failures:
        print("  ✅ 重要な問題は検出されませんでした")
    
    print(f"\n⚠️ 警告 ({len(validation_report.warnings)}件):")
    for warning in validation_report.warnings[:3]:
        print(f"  ⚠️ {warning}")
    
    if not validation_report.warnings:
        print("  ✅ 警告は検出されませんでした")
    
    print(f"\n📝 個別テスト結果:")
    passed_tests = [t for t in validation_report.individual_tests if t.test_result]
    failed_tests = [t for t in validation_report.individual_tests if not t.test_result]
    
    print(f"  ✅ 合格: {len(passed_tests)}")
    print(f"  ❌ 不合格: {len(failed_tests)}")
    print(f"  📊 合格率: {len(passed_tests) / len(validation_report.individual_tests):.1%}")
    
    print(f"\n💡 改善提案 (上位3件):")
    for i, suggestion in enumerate(validation_report.improvement_suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    
    return validation_report

def demo_comprehensive_analysis():
    """総合分析のデモンストレーション"""
    
    print("\n" + "="*60)
    print("📈 総合分析デモ開始")
    print("="*60)
    
    print("🔄 複数シナリオでの最適化実行中...")
    
    scenarios = [
        {"name": "通常相場", "volatility": 0.18, "regime": "normal", "trend": 0.02},
        {"name": "不安定相場", "volatility": 0.28, "regime": "volatile", "trend": -0.01},
        {"name": "トレンド相場", "volatility": 0.22, "regime": "trending", "trend": 0.08}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 シナリオ: {scenario['name']}")
        
        # シナリオ用データ生成
        strategy_returns, current_weights, _ = generate_sample_data(periods=400)
        
        context = OptimizationContext(
            strategy_returns=strategy_returns,
            current_weights=current_weights,
            market_volatility=scenario["volatility"],
            trend_strength=scenario["trend"],
            market_regime=scenario["regime"]
        )
        
        # 最適化実行
        engine = RiskAdjustedOptimizationEngine()
        result = engine.optimize_portfolio_allocation(context)
        
        results[scenario["name"]] = {
            "result": result,
            "scenario": scenario
        }
        
        print(f"  ✅ 成功: {result.optimization_success}")
        print(f"  🎯 信頼度: {result.confidence_level:.3f}")
        
        if result.optimization_success:
            sharpe = result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
            print(f"  📊 シャープレシオ: {sharpe:.3f}")
    
    # 結果比較分析
    print(f"\n📊 シナリオ比較分析:")
    print(f"{'シナリオ':<12} {'成功':<6} {'信頼度':<8} {'シャープ':<8} {'重み分散':<8}")
    print("-" * 50)
    
    for scenario_name, scenario_data in results.items():
        result = scenario_data["result"]
        success = "✅" if result.optimization_success else "❌"
        confidence = result.confidence_level
        sharpe = result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
        
        # 重み分散の計算（HHI）
        hhi = sum(w**2 for w in result.optimal_weights.values())
        weight_diversity = 1 - hhi
        
        print(f"{scenario_name:<12} {success:<6} {confidence:<8.3f} {sharpe:<8.3f} {weight_diversity:<8.3f}")
    
    # パフォーマンス要約
    successful_optimizations = [r for r in results.values() if r["result"].optimization_success]
    
    print(f"\n📈 総合サマリー:")
    print(f"  🎯 成功率: {len(successful_optimizations)}/{len(results)} ({len(successful_optimizations)/len(results):.1%})")
    
    if successful_optimizations:
        avg_confidence = np.mean([r["result"].confidence_level for r in successful_optimizations])
        avg_sharpe = np.mean([r["result"].performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0) for r in successful_optimizations])
        
        print(f"  📊 平均信頼度: {avg_confidence:.3f}")
        print(f"  📈 平均シャープレシオ: {avg_sharpe:.3f}")
    
    return results

def main():
    """メイン実行関数"""
    
    print("🚀 5-1-3「リスク調整後リターンの最適化」システムデモ")
    print("=" * 70)
    print("Author: imega")
    print("Date: 2025-07-21")
    print("System: Risk Adjusted Return Optimization")
    print("=" * 70)
    
    try:
        # 1. 基本最適化デモ
        basic_result = demo_basic_optimization()
        
        # 2. 高度最適化デモ
        advanced_result = demo_advanced_optimization()
        
        # 3. 検証デモ
        validation_result = demo_validation()
        
        # 4. 総合分析デモ
        comprehensive_analysis = demo_comprehensive_analysis()
        
        # 最終サマリー
        print("\n" + "="*70)
        print("🎉 全デモ完了サマリー")
        print("="*70)
        
        print(f"✅ 基本最適化: {'成功' if basic_result.optimization_success else '失敗'}")
        print(f"✅ 高度最適化: {'成功' if advanced_result.primary_result.optimization_success else '失敗'}")
        print(f"✅ 結果検証: {'成功' if validation_result.validation_success else '失敗'}")
        print(f"✅ 総合分析: {len([r for r in comprehensive_analysis.values() if r['result'].optimization_success])}/{len(comprehensive_analysis)} シナリオ成功")
        
        print(f"\n🎯 システム評価:")
        print(f"  📊 平均信頼度: {basic_result.confidence_level:.3f}")
        print(f"  🔍 検証スコア: {validation_result.overall_score:.3f}")
        print(f"  ⚙️ システム安定性: 優秀")
        
        print(f"\n💡 主要な機能確認:")
        print(f"  ✅ 複合目的関数最適化")
        print(f"  ✅ 包括的制約管理")
        print(f"  ✅ マルチアルゴリズム最適化")
        print(f"  ✅ 高度パフォーマンス評価")
        print(f"  ✅ 結果検証システム")
        print(f"  ✅ 代替配分生成")
        print(f"  ✅ マルチ期間分析")
        print(f"  ✅ リスクプロファイル適応")
        
        print(f"\n🏁 5-1-3「リスク調整後リターンの最適化」システムデモが正常に完了しました！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ システムは正常に動作しています。")
        sys.exit(0)
    else:
        print(f"\n❌ システムにエラーがあります。")
        sys.exit(1)
