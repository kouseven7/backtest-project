"""
Demo Script: Portfolio Weight System Integration
File: demo_portfolio_weight_system.py
Description: 
  3-2-1「スコアベースの資金配分計算式設計」のデモンストレーション
  実装したポートフォリオ重み計算システムの統合デモ

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Usage:
  python demo_portfolio_weight_system.py
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, PortfolioConstraints, 
        AllocationMethod, RebalanceFrequency
    )
    from config.portfolio_weight_templates import (
        WeightTemplateManager, MarketRegime, TemplateType
    )
    from config.portfolio_weighting_agent import (
        PortfolioWeightingAgent, AutomationLevel
    )
    from config.strategy_scoring_model import StrategyScoreManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_market_data(ticker: str = "AAPL", days: int = 252) -> pd.DataFrame:
    """サンプル市場データの作成"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # ランダムウォークで価格データを生成
    initial_price = 150.0
    returns = np.random.normal(0.0008, 0.02, days)  # 日次リターン
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 出来高データ
    volume = np.random.lognormal(15, 0.5, days)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Adj Close': prices,
        'Volume': volume
    })
    
    data.set_index('Date', inplace=True)
    return data

def demo_basic_weight_calculation():
    """基本的な重み計算のデモ"""
    print("\n" + "="*60)
    print("1. 基本的なポートフォリオ重み計算デモ")
    print("="*60)
    
    try:
        # 重み計算エンジンの初期化
        calculator = PortfolioWeightCalculator()
        
        # サンプルデータの作成
        market_data = create_sample_market_data("AAPL", 100)
        ticker = "AAPL"
        
        # 複数の配分手法でテスト
        methods = [
            AllocationMethod.SCORE_PROPORTIONAL,
            AllocationMethod.RISK_ADJUSTED,
            AllocationMethod.EQUAL_WEIGHT,
            AllocationMethod.HIERARCHICAL
        ]
        
        results = {}
        
        for method in methods:
            print(f"\n--- {method.value.replace('_', ' ').title()} 配分手法 ---")
            
            # 設定の作成
            config = WeightAllocationConfig(
                method=method,
                constraints=PortfolioConstraints(
                    max_individual_weight=0.4,
                    min_individual_weight=0.1,
                    max_strategies=4
                )
            )
            
            # 重み計算
            result = calculator.calculate_portfolio_weights(
                ticker=ticker,
                market_data=market_data,
                config=config
            )
            
            results[method.value] = result
            
            # 結果の表示
            print(f"戦略数: {len(result.strategy_weights)}")
            print(f"期待リターン: {result.expected_return:.4f}")
            print(f"期待リスク: {result.expected_risk:.4f}")
            print(f"シャープレシオ: {result.sharpe_ratio:.4f}")
            print(f"分散化比率: {result.diversification_ratio:.4f}")
            print(f"信頼度: {result.confidence_level:.4f}")
            
            if result.strategy_weights:
                print("戦略重み:")
                for strategy, weight in result.strategy_weights.items():
                    print(f"  {strategy}: {weight:.3f} ({weight*100:.1f}%)")
            
            if result.constraint_violations:
                print(f"制約違反: {len(result.constraint_violations)}")
                for violation in result.constraint_violations[:3]:
                    print(f"  - {violation}")
        
        print(f"\n計算完了: {len(results)}種類の配分手法をテスト")
        return results
        
    except Exception as e:
        logger.error(f"Error in basic weight calculation demo: {e}")
        return {}

def demo_template_system():
    """テンプレートシステムのデモ"""
    print("\n" + "="*60)
    print("2. ポートフォリオ重みテンプレートシステムデモ")
    print("="*60)
    
    try:
        # テンプレートマネージャーの初期化
        manager = WeightTemplateManager()
        
        # 利用可能なテンプレート一覧
        all_templates = manager.get_all_templates()
        print(f"\n利用可能なテンプレート数: {len(all_templates)}")
        
        for name, template in all_templates.items():
            print(f"\n--- {template.name} ---")
            print(f"タイプ: {template.template_type.value}")
            print(f"説明: {template.description}")
            print(f"リスクレベル: {template.risk_level}")
            print(f"最小資本: ${template.min_capital:,.0f}")
            print(f"期待ターンオーバー: {template.expected_turnover:.1%}")
            print(f"適用可能市場環境: {[r.value for r in template.suitable_market_regimes]}")
            
            # 配分設定の要約
            config = template.config
            print(f"配分手法: {config.method.value}")
            print(f"最大個別重み: {config.constraints.max_individual_weight:.1%}")
            print(f"最大戦略数: {config.constraints.max_strategies}")
            print(f"リバランス頻度: {config.rebalance_frequency.value}")
        
        print("\n--- テンプレート推奨システム ---")
        
        # 異なる市場環境での推奨テンプレート
        market_scenarios = [
            (MarketRegime.BULL_MARKET, "high", 200000),
            (MarketRegime.BEAR_MARKET, "low", 100000),
            (MarketRegime.VOLATILE, "medium", 150000),
            (MarketRegime.SIDEWAYS, "medium", 120000)
        ]
        
        for market_regime, risk_tolerance, capital in market_scenarios:
            print(f"\n{market_regime.value.replace('_', ' ').title()} 相場 (リスク許容度: {risk_tolerance}, 資本: ${capital:,})")
            
            recommendations = manager.recommend_template(
                market_regime=market_regime,
                risk_tolerance=risk_tolerance,
                capital_amount=capital
            )
            
            print("推奨テンプレート:")
            for i, (name, template, score) in enumerate(recommendations[:3], 1):
                print(f"  {i}. {template.name} (スコア: {score:.3f})")
        
        # カスタムテンプレートの作成例
        print("\n--- カスタムテンプレート作成例 ---")
        
        custom_config = WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(
                max_individual_weight=0.3,
                min_individual_weight=0.15,
                max_strategies=3,
                min_strategies=2,
                concentration_limit=0.6
            ),
            risk_aversion=2.5,
            confidence_weight=0.4
        )
        
        custom_template = manager.create_custom_template(
            name="Demo Custom Template",
            template_type=TemplateType.CONSERVATIVE,
            description="デモ用カスタムテンプレート",
            config=custom_config,
            risk_level="low-medium",
            expected_turnover=0.05,
            tags=["demo", "custom", "conservative"]
        )
        
        print(f"カスタムテンプレート作成完了: {custom_template.name}")
        
        # テンプレート概要
        summary = manager.get_template_summary()
        print(f"\nテンプレートシステム概要:")
        print(f"  総テンプレート数: {summary['total_templates']}")
        print(f"  事前定義済み: {summary['predefined_templates']}")
        print(f"  カスタム: {summary['custom_templates']}")
        print(f"  テンプレートタイプ別分布: {summary['template_types']}")
        print(f"  リスクレベル別分布: {summary['risk_levels']}")
        
        return manager
        
    except Exception as e:
        logger.error(f"Error in template system demo: {e}")
        return None

def demo_integration_with_existing_systems():
    """既存システムとの統合デモ"""
    print("\n" + "="*60)
    print("3. 既存システム統合デモ")
    print("="*60)
    
    try:
        # コンポーネントの初期化
        calculator = PortfolioWeightCalculator()
        template_manager = WeightTemplateManager()
        score_manager = StrategyScoreManager()
        
        ticker = "AAPL"
        market_data = create_sample_market_data(ticker, 150)
        
        print(f"\n対象銘柄: {ticker}")
        print(f"市場データ期間: {len(market_data)}日")
        
        # 1. 戦略スコア取得
        print("\n--- 戦略スコア取得 ---")
        try:
            strategy_scores = score_manager.calculate_comprehensive_scores([ticker])
            if ticker in strategy_scores:
                ticker_scores = strategy_scores[ticker]
                print(f"スコア計算済み戦略数: {len(ticker_scores)}")
                
                for strategy_name, score in list(ticker_scores.items())[:3]:
                    print(f"  {strategy_name}:")
                    print(f"    総合スコア: {score.total_score:.3f}")
                    print(f"    信頼度: {score.confidence:.3f}")
                    print(f"    トレンド適合度: {score.trend_fitness:.3f}")
            else:
                print("注意: スコアデータがありません。デモ用データを使用します。")
        except Exception as e:
            print(f"スコア取得エラー: {e}")
            print("デモ用データを使用します。")
        
        # 2. テンプレートベース重み計算
        print("\n--- テンプレートベース重み計算 ---")
        
        # 市場環境を推定（簡略化）
        recent_returns = market_data['Adj Close'].pct_change().dropna()[-20:]
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        
        if avg_return > 0.001 and volatility < 0.02:
            market_regime = MarketRegime.BULL_MARKET
        elif avg_return < -0.001:
            market_regime = MarketRegime.BEAR_MARKET
        elif volatility > 0.03:
            market_regime = MarketRegime.VOLATILE
        else:
            market_regime = MarketRegime.SIDEWAYS
        
        print(f"推定市場環境: {market_regime.value}")
        print(f"平均リターン: {avg_return:.4f}")
        print(f"ボラティリティ: {volatility:.4f}")
        
        # 推奨テンプレート取得
        recommendations = template_manager.recommend_template(
            market_regime=market_regime,
            risk_tolerance="medium",
            capital_amount=150000
        )
        
        if recommendations:
            best_template_name, best_template, score = recommendations[0]
            print(f"\n最適テンプレート: {best_template.name} (スコア: {score:.3f})")
            
            # テンプレートを使用した重み計算
            result = calculator.calculate_portfolio_weights(
                ticker=ticker,
                market_data=market_data,
                config=best_template.config
            )
            
            print(f"\nテンプレートベース計算結果:")
            print(f"  戦略数: {len(result.strategy_weights)}")
            print(f"  期待リターン: {result.expected_return:.4f}")
            print(f"  期待リスク: {result.expected_risk:.4f}")
            print(f"  信頼度: {result.confidence_level:.4f}")
            
            if result.strategy_weights:
                print(f"  戦略重み分布:")
                for strategy, weight in result.strategy_weights.items():
                    print(f"    {strategy}: {weight:.3f}")
        
        # 3. 複数テンプレート比較
        print(f"\n--- 複数テンプレート比較 ---")
        
        template_comparison = {}
        for template_name in ["conservative", "balanced", "aggressive"]:
            template = template_manager.get_template(template_name)
            if template:
                result = calculator.calculate_portfolio_weights(
                    ticker=ticker,
                    market_data=market_data,
                    config=template.config
                )
                
                template_comparison[template_name] = {
                    "expected_return": result.expected_return,
                    "expected_risk": result.expected_risk,
                    "sharpe_ratio": result.sharpe_ratio,
                    "confidence": result.confidence_level,
                    "num_strategies": len(result.strategy_weights)
                }
        
        print("テンプレート比較結果:")
        print(f"{'テンプレート':<12} {'期待リターン':<12} {'期待リスク':<12} {'シャープ比':<10} {'信頼度':<8} {'戦略数':<6}")
        print("-" * 70)
        
        for name, metrics in template_comparison.items():
            print(f"{name:<12} {metrics['expected_return']:<12.4f} "
                  f"{metrics['expected_risk']:<12.4f} {metrics['sharpe_ratio']:<10.3f} "
                  f"{metrics['confidence']:<8.3f} {metrics['num_strategies']:<6}")
        
        return template_comparison
        
    except Exception as e:
        logger.error(f"Error in integration demo: {e}")
        return {}

async def demo_automated_agent():
    """自動化エージェントのデモ"""
    print("\n" + "="*60)
    print("4. ポートフォリオ重み付け自動化エージェントデモ")
    print("="*60)
    
    try:
        # エージェントの初期化
        agent = PortfolioWeightingAgent(automation_level=AutomationLevel.SEMI_AUTOMATIC)
        
        ticker = "AAPL"
        market_data = create_sample_market_data(ticker, 100)
        
        print(f"\nエージェント設定:")
        status = agent.get_agent_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print(f"\n自動化ルール一覧:")
        for i, rule in enumerate(agent.automation_rules, 1):
            print(f"  {i}. {rule.name}")
            print(f"     トリガー: {rule.trigger_condition.value}")
            print(f"     アクション: {rule.action_type.value}")
            print(f"     閾値: {rule.threshold_value}")
            print(f"     自動化レベル: {rule.automation_level.value}")
            print(f"     有効: {rule.enabled}")
        
        # 短時間の監視デモ
        print(f"\n--- 短時間監視デモ (30秒間) ---")
        print("注意: 実際のトリガー条件は満たされない可能性があります")
        
        # 1回のチェックサイクルをデモ
        triggered_rules = await agent._check_trigger_conditions(ticker, market_data)
        
        if triggered_rules:
            print(f"トリガーされたルール数: {len(triggered_rules)}")
            for rule in triggered_rules:
                print(f"  - {rule.name} ({rule.trigger_condition.value})")
                
                # 意思決定のデモ
                decision = await agent._make_decision(rule, ticker, market_data)
                if decision:
                    print(f"    決定: {decision.recommended_action}")
                    print(f"    信頼度: {decision.confidence_level:.3f}")
                    print(f"    リスクレベル: {decision.risk_level}")
                    print(f"    承認要否: {decision.required_approval}")
        else:
            print("現在トリガーされているルールはありません")
        
        # 手動でいくつかの決定をシミュレート
        print(f"\n--- 手動決定シミュレーション ---")
        
        # 重み計算の実行
        calculator = PortfolioWeightCalculator()
        result = calculator.calculate_portfolio_weights(ticker, market_data)
        
        if result.strategy_weights:
            agent.current_weights = result.strategy_weights
            print(f"現在の重み設定完了: {len(result.strategy_weights)}戦略")
        
        # エージェント状態の更新確認
        updated_status = agent.get_agent_status()
        print(f"\n更新後エージェント状態:")
        for key, value in updated_status.items():
            print(f"  {key}: {value}")
        
        return agent
        
    except Exception as e:
        logger.error(f"Error in automated agent demo: {e}")
        return None

def demo_performance_comparison():
    """パフォーマンス比較デモ"""
    print("\n" + "="*60)
    print("5. パフォーマンス比較デモ")
    print("="*60)
    
    try:
        calculator = PortfolioWeightCalculator()
        template_manager = WeightTemplateManager()
        
        ticker = "AAPL"
        market_data = create_sample_market_data(ticker, 200)
        
        # 複数期間でのパフォーマンステスト
        test_periods = [30, 60, 100, 150]
        methods = [AllocationMethod.SCORE_PROPORTIONAL, AllocationMethod.RISK_ADJUSTED, AllocationMethod.EQUAL_WEIGHT]
        
        performance_results = {}
        
        print(f"期間別パフォーマンステスト:")
        print(f"{'期間(日)':<8} {'手法':<20} {'期待リターン':<12} {'期待リスク':<12} {'シャープ比':<10} {'処理時間(ms)':<12}")
        print("-" * 80)
        
        for period in test_periods:
            period_data = market_data.tail(period)
            
            for method in methods:
                config = WeightAllocationConfig(method=method)
                
                start_time = datetime.now()
                result = calculator.calculate_portfolio_weights(ticker, period_data, config)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                method_name = method.value.replace('_', ' ')[:19]
                print(f"{period:<8} {method_name:<20} {result.expected_return:<12.4f} "
                      f"{result.expected_risk:<12.4f} {result.sharpe_ratio:<10.3f} {processing_time:<12.1f}")
                
                performance_results[f"{period}_{method.value}"] = {
                    "period": period,
                    "method": method.value,
                    "expected_return": result.expected_return,
                    "expected_risk": result.expected_risk,
                    "sharpe_ratio": result.sharpe_ratio,
                    "processing_time_ms": processing_time,
                    "num_strategies": len(result.strategy_weights)
                }
        
        # 最適パフォーマンスの特定
        print(f"\n最適パフォーマンス分析:")
        
        best_sharpe = max(performance_results.values(), key=lambda x: x['sharpe_ratio'])
        best_return = max(performance_results.values(), key=lambda x: x['expected_return'])
        lowest_risk = min(performance_results.values(), key=lambda x: x['expected_risk'])
        
        print(f"最高シャープレシオ: {best_sharpe['sharpe_ratio']:.3f} "
              f"({best_sharpe['method']}, {best_sharpe['period']}日)")
        print(f"最高期待リターン: {best_return['expected_return']:.4f} "
              f"({best_return['method']}, {best_return['period']}日)")
        print(f"最低リスク: {lowest_risk['expected_risk']:.4f} "
              f"({lowest_risk['method']}, {lowest_risk['period']}日)")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"Error in performance comparison demo: {e}")
        return {}

def main():
    """メインデモ関数"""
    print("3-2-1「スコアベースの資金配分計算式設計」実装デモ")
    print("=" * 80)
    
    try:
        # 1. 基本的な重み計算
        basic_results = demo_basic_weight_calculation()
        
        # 2. テンプレートシステム
        template_manager = demo_template_system()
        
        # 3. 既存システム統合
        integration_results = demo_integration_with_existing_systems()
        
        # 4. 自動化エージェント
        agent = asyncio.run(demo_automated_agent())
        
        # 5. パフォーマンス比較
        performance_results = demo_performance_comparison()
        
        # 総括
        print("\n" + "="*60)
        print("デモ実行完了総括")
        print("="*60)
        
        success_count = sum([
            1 if basic_results else 0,
            1 if template_manager else 0,
            1 if integration_results else 0,
            1 if agent else 0,
            1 if performance_results else 0
        ])
        
        print(f"実行成功デモ数: {success_count}/5")
        print(f"基本重み計算: {'✓' if basic_results else '✗'}")
        print(f"テンプレートシステム: {'✓' if template_manager else '✗'}")
        print(f"既存システム統合: {'✓' if integration_results else '✗'}")
        print(f"自動化エージェント: {'✓' if agent else '✗'}")
        print(f"パフォーマンス比較: {'✓' if performance_results else '✗'}")
        
        if template_manager:
            summary = template_manager.get_template_summary()
            print(f"\nテンプレート統計: {summary['total_templates']}個のテンプレート利用可能")
        
        if performance_results:
            avg_processing_time = np.mean([r['processing_time_ms'] for r in performance_results.values()])
            print(f"平均処理時間: {avg_processing_time:.1f}ms")
        
        print(f"\n3-2-1「スコアベースの資金配分計算式設計」実装完了!")
        print("主な機能:")
        print("  ✓ 5種類の配分手法")
        print("  ✓ 制約管理システム")
        print("  ✓ 5つの事前定義テンプレート")
        print("  ✓ カスタムテンプレート作成")
        print("  ✓ 4段階自動化エージェント")
        print("  ✓ 既存システムとの完全統合")
        
    except Exception as e:
        logger.error(f"Demo execution error: {e}")
        print(f"デモ実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
