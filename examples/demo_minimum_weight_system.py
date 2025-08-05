"""
3-2-2 階層的最小重み設定機能のデモンストレーション

Author: imega
Created: 2025-07-13
Description: 
  最小資金割合設定機能のテストと使用例を示すデモスクリプト
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

from config.portfolio_weight_calculator import (
    PortfolioWeightCalculator, WeightAllocationConfig, PortfolioConstraints,
    AllocationMethod, MinimumWeightRule, MinimumWeightLevel, WeightAdjustmentMethod
)
from config.strategy_scoring_model import StrategyScore

def create_sample_strategy_scores() -> dict:
    """サンプル戦略スコアの作成"""
    strategies = {
        'trend_following_ma': 0.85,
        'momentum_rsi': 0.72,
        'mean_reversion_bollinger': 0.68,
        'breakout_channel': 0.64,
        'volatility_breakout': 0.58,
        'trend_macd': 0.55,
        'momentum_stochastic': 0.45,  # 低スコア戦略
    }
    
    strategy_scores = {}
    for name, score in strategies.items():
        # StrategyScoreオブジェクトを正しく作成
        strategy_score = StrategyScore(
            strategy_name=name,
            ticker="TEST",
            total_score=score,
            component_scores={
                'performance': score * 0.4,
                'risk_adjusted': score * 0.3,
                'trend_adaptation': score * 0.2,
                'robustness': score * 0.1
            },
            trend_fitness=score * 0.8,
            confidence=min(1.0, score + 0.1),
            metadata={'category': 'test'},
            calculated_at=datetime.now()
        )
        
        strategy_scores[name] = strategy_score
    
    return strategy_scores

def create_sample_market_data() -> pd.DataFrame:
    """サンプル市場データの作成"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # ランダムウォークベースの価格データ
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 日次リターン
    prices = 100 * np.exp(np.cumsum(returns))  # 累積価格
    
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    })
    
    return market_data

def demo_basic_functionality():
    """基本機能のデモ"""
    print("="*80)
    print("3-2-2 階層的最小重み設定機能 - 基本デモ")
    print("="*80)
    
    # 1. ポートフォリオ重み計算器の初期化
    calculator = PortfolioWeightCalculator()
    
    # 2. サンプルデータの準備
    strategy_scores = create_sample_strategy_scores()
    market_data = create_sample_market_data()
    
    print(f"\n戦略数: {len(strategy_scores)}")
    for name, score in strategy_scores.items():
        print(f"  {name}: {score.total_score:.3f}")
    
    # 3. 従来の重み計算（3-2-2機能なし）
    print("\n--- 従来の重み計算（階層的最小重み機能無効）---")
    
    config_basic = WeightAllocationConfig(
        method=AllocationMethod.RISK_ADJUSTED,
        constraints=PortfolioConstraints(
            enable_hierarchical_minimum_weights=False,
            min_individual_weight=0.05,
            max_individual_weight=0.4
        )
    )
    
    result_basic = calculator.calculate_portfolio_weights(
        ticker="DEMO",
        market_data=market_data,
        config=config_basic
    )
    
    print("従来の重み配分:")
    for name, weight in result_basic.strategy_weights.items():
        print(f"  {name}: {weight:.3f} ({weight*100:.1f}%)")
    
    return calculator, strategy_scores, market_data

def demo_hierarchical_minimum_weights(calculator, strategy_scores, market_data):
    """階層的最小重み設定のデモ"""
    print("\n--- 3-2-2: 階層的最小重み設定のデモ ---")
    
    # 1. 戦略固有の最小重み設定
    print("\n1. 戦略固有最小重み設定:")
    calculator.add_strategy_minimum_weight("trend_following_ma", 0.15, exclusion_threshold=0.10)
    calculator.add_strategy_minimum_weight("momentum_rsi", 0.12)
    print("  - trend_following_ma: 最小15%, 除外閾値10%")
    print("  - momentum_rsi: 最小12%")
    
    # 2. カテゴリー別最小重み設定
    print("\n2. カテゴリー別最小重み設定:")
    calculator.add_category_minimum_weight("trend_following", 0.08)
    calculator.add_category_minimum_weight("momentum", 0.06)
    print("  - trend_following カテゴリー: 最小8%")
    print("  - momentum カテゴリー: 最小6%")
    
    # 3. デフォルト最小重み設定
    print("\n3. デフォルト最小重み設定:")
    calculator.set_default_minimum_weight(0.04)
    print("  - デフォルト: 最小4%")
    
    # 4. 階層的最小重み機能を有効にした計算
    print("\n4. 階層的最小重み機能適用後の計算:")
    
    config_enhanced = WeightAllocationConfig(
        method=AllocationMethod.RISK_ADJUSTED,
        constraints=PortfolioConstraints(
            enable_hierarchical_minimum_weights=True,
            weight_adjustment_method="proportional",
            enable_conditional_exclusion=True,
            exclusion_score_threshold=0.4,
            min_individual_weight=0.05,
            max_individual_weight=0.4
        )
    )
    
    result_enhanced = calculator.calculate_portfolio_weights(
        ticker="DEMO",
        market_data=market_data,
        config=config_enhanced
    )
    
    print("階層的最小重み適用後の配分:")
    for name, weight in result_enhanced.strategy_weights.items():
        print(f"  {name}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 5. 調整結果の詳細表示
    print("\n5. 調整結果の詳細:")
    metadata = result_enhanced.metadata
    if metadata.get('hierarchical_adjustment_applied'):
        print(f"  - 総調整量: {metadata.get('total_adjustment', 0):.3f}")
        print(f"  - 除外戦略: {metadata.get('excluded_strategies', [])}")
        print(f"  - 適用ルール数: {metadata.get('applied_minimum_weight_rules', 0)}")
        print(f"  - 調整手法: {metadata.get('adjustment_method', 'N/A')}")
        print(f"  - 調整成功: {metadata.get('adjustment_success', False)}")
        print(f"  - 調整理由: {metadata.get('adjustment_reason', 'N/A')}")
    
    # 6. 制約違反のチェック
    if result_enhanced.constraint_violations:
        print("\n6. 制約違反:")
        for violation in result_enhanced.constraint_violations:
            print(f"  - {violation}")
    else:
        print("\n6. 制約違反: なし")
    
    return result_enhanced

def demo_weight_adjustment_methods(calculator, strategy_scores, market_data):
    """重み調整手法の比較デモ"""
    print("\n--- 重み調整手法の比較 ---")
    
    adjustment_methods = ["proportional", "equal", "score_weighted"]
    
    for method in adjustment_methods:
        print(f"\n{method.upper()}調整手法:")
        
        config = WeightAllocationConfig(
            method=AllocationMethod.SCORE_PROPORTIONAL,
            constraints=PortfolioConstraints(
                enable_hierarchical_minimum_weights=True,
                weight_adjustment_method=method,
                min_individual_weight=0.08,  # 高い最小重み設定で調整を誘発
                max_individual_weight=0.3
            )
        )
        
        result = calculator.calculate_portfolio_weights(
            ticker="DEMO",
            market_data=market_data,
            config=config
        )
        
        for name, weight in result.strategy_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        print(f"  総調整量: {result.metadata.get('total_adjustment', 0):.3f}")

def demo_conditional_exclusion(calculator, strategy_scores, market_data):
    """条件付き除外機能のデモ"""
    print("\n--- 条件付き除外機能のデモ ---")
    
    # 除外閾値を設定した計算
    config = WeightAllocationConfig(
        method=AllocationMethod.RISK_ADJUSTED,
        constraints=PortfolioConstraints(
            enable_hierarchical_minimum_weights=True,
            enable_conditional_exclusion=True,
            exclusion_score_threshold=0.5,  # スコア50%以下は除外
            min_individual_weight=0.05,
            max_individual_weight=0.4
        )
    )
    
    result = calculator.calculate_portfolio_weights(
        ticker="DEMO",
        market_data=market_data,
        config=config
    )
    
    print("除外後の戦略配分:")
    for name, weight in result.strategy_weights.items():
        score = strategy_scores[name].total_score
        print(f"  {name}: {weight:.3f} (スコア: {score:.3f})")
    
    excluded = result.metadata.get('excluded_strategies', [])
    if excluded:
        print(f"\n除外された戦略: {excluded}")
        for name in excluded:
            if name in strategy_scores:
                score = strategy_scores[name].total_score
                print(f"  {name}: スコア {score:.3f}")

def demo_performance_comparison():
    """パフォーマンス比較デモ"""
    print("\n--- パフォーマンス比較 ---")
    
    calculator = PortfolioWeightCalculator()
    strategy_scores = create_sample_strategy_scores()
    market_data = create_sample_market_data()
    
    configs = {
        "従来手法": WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(enable_hierarchical_minimum_weights=False)
        ),
        "3-2-2機能": WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(
                enable_hierarchical_minimum_weights=True,
                weight_adjustment_method="proportional"
            )
        )
    }
    
    # 戦略固有ルールの設定
    calculator.add_strategy_minimum_weight("trend_following_ma", 0.12)
    calculator.add_category_minimum_weight("momentum", 0.08)
    
    print("\n配分結果比較:")
    print(f"{'手法':<15} {'期待リターン':<12} {'期待リスク':<12} {'シャープレシオ':<15} {'分散化比率':<12}")
    print("-" * 70)
    
    for name, config in configs.items():
        result = calculator.calculate_portfolio_weights(
            ticker="DEMO",
            market_data=market_data,
            config=config
        )
        
        print(f"{name:<15} {result.expected_return:<12.4f} {result.expected_risk:<12.4f} "
              f"{result.sharpe_ratio:<15.4f} {result.diversification_ratio:<12.4f}")
    
    print("\n重み分布比較:")
    print(f"{'戦略名':<25} {'従来手法':<12} {'3-2-2機能':<12}")
    print("-" * 50)
    
    result_basic = calculator.calculate_portfolio_weights(
        ticker="DEMO", market_data=market_data, config=configs["従来手法"]
    )
    result_enhanced = calculator.calculate_portfolio_weights(
        ticker="DEMO", market_data=market_data, config=configs["3-2-2機能"]
    )
    
    all_strategies = set(result_basic.strategy_weights.keys()) | set(result_enhanced.strategy_weights.keys())
    
    for strategy in sorted(all_strategies):
        basic_weight = result_basic.strategy_weights.get(strategy, 0.0)
        enhanced_weight = result_enhanced.strategy_weights.get(strategy, 0.0)
        print(f"{strategy:<25} {basic_weight:<12.3f} {enhanced_weight:<12.3f}")

def main():
    """メインデモ実行"""
    print("3-2-2 階層的最小重み設定機能 デモンストレーション")
    print("=" * 80)
    
    try:
        # 基本機能デモ
        calculator, strategy_scores, market_data = demo_basic_functionality()
        
        # 階層的最小重み設定デモ
        demo_hierarchical_minimum_weights(calculator, strategy_scores, market_data)
        
        # 重み調整手法比較デモ
        demo_weight_adjustment_methods(calculator, strategy_scores, market_data)
        
        # 条件付き除外デモ
        demo_conditional_exclusion(calculator, strategy_scores, market_data)
        
        # パフォーマンス比較デモ
        demo_performance_comparison()
        
        print("\n" + "="*80)
        print("デモ完了: 3-2-2機能が正常に動作しています")
        print("="*80)
        
    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
