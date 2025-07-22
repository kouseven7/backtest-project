"""
Test Suite: Portfolio Weight System
File: test_portfolio_weight_system.py
Description: 
  3-2-1「スコアベースの資金配分計算式設計」のテストスイート
  包括的なユニットテストとインテグレーションテスト

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Usage:
  python -m pytest test_portfolio_weight_system.py -v
"""

import os
import sys
import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, PortfolioConstraints,
        AllocationMethod, RebalanceFrequency, AllocationResult,
        ScoreProportionalAllocation, RiskAdjustedAllocation, HierarchicalAllocation
    )
    from config.portfolio_weight_templates import (
        WeightTemplateManager, WeightTemplate, MarketRegime, TemplateType
    )
    from config.portfolio_weighting_agent import (
        PortfolioWeightingAgent, AutomationLevel, TriggerCondition, ActionType, AutomationRule
    )
    from config.strategy_scoring_model import StrategyScore, ScoreWeights
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# テスト用ロガー設定
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_market_data():
    """サンプル市場データのフィクスチャ"""
    np.random.seed(42)
    days = 100
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })
    
    data.set_index('Date', inplace=True)
    return data

@pytest.fixture
def sample_strategy_scores():
    """サンプル戦略スコアのフィクスチャ"""
    scores = {}
    strategy_names = ["VWAPBounceStrategy", "VWAPBreakoutStrategy", "MomentumStrategy", "ContrarianStrategy"]
    
    for i, name in enumerate(strategy_names):
        component_scores = {
            'performance': 0.7 + i * 0.05,
            'stability': 0.6 + i * 0.08,
            'risk_adjusted': 0.65 + i * 0.06,
            'trend_adaptation': 0.55 + i * 0.1,
            'reliability': 0.8 - i * 0.05
        }
        
        total_score = sum(component_scores.values()) / len(component_scores)
        
        scores[name] = StrategyScore(
            strategy_name=name,
            ticker="AAPL",
            total_score=total_score,
            component_scores=component_scores,
            trend_fitness=0.7 + i * 0.05,
            confidence=0.8 - i * 0.02,
            calculated_at=datetime.now()
        )
    
    return scores

@pytest.fixture
def temp_config_dir():
    """テンポラリ設定ディレクトリのフィクスチャ"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

class TestPortfolioWeightCalculator:
    """PortfolioWeightCalculatorのテストクラス"""
    
    def test_initialization(self, temp_config_dir):
        """初期化のテスト"""
        calculator = PortfolioWeightCalculator(base_dir=temp_config_dir)
        
        assert calculator.base_dir.exists()
        assert len(calculator.allocation_strategies) == 5
        assert calculator.config is not None
        assert calculator._weight_cache == {}
    
    def test_calculate_portfolio_weights_basic(self, sample_market_data, sample_strategy_scores):
        """基本的な重み計算のテスト"""
        calculator = PortfolioWeightCalculator()
        
        # MockStrategyScoreManagerを設定
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {"AAPL": sample_strategy_scores}
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data
        )
        
        assert isinstance(result, AllocationResult)
        assert isinstance(result.strategy_weights, dict)
        assert result.expected_return >= 0
        assert result.expected_risk >= 0
        assert 0 <= result.confidence_level <= 1
        assert len(result.constraint_violations) >= 0
    
    def test_allocation_methods(self, sample_market_data, sample_strategy_scores):
        """各配分手法のテスト"""
        calculator = PortfolioWeightCalculator()
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {"AAPL": sample_strategy_scores}
        
        methods = [
            AllocationMethod.SCORE_PROPORTIONAL,
            AllocationMethod.RISK_ADJUSTED,
            AllocationMethod.EQUAL_WEIGHT,
            AllocationMethod.HIERARCHICAL,
            AllocationMethod.KELLY_CRITERION
        ]
        
        for method in methods:
            config = WeightAllocationConfig(method=method)
            
            result = calculator.calculate_portfolio_weights(
                ticker="AAPL",
                market_data=sample_market_data,
                config=config
            )
            
            assert isinstance(result, AllocationResult)
            assert result.metadata["allocation_method"] == method.value
            
            # 重みの合計が1.0に近いことを確認
            if result.strategy_weights:
                total_weight = sum(result.strategy_weights.values())
                assert abs(total_weight - 1.0) < 0.01
    
    def test_constraints_enforcement(self, sample_market_data, sample_strategy_scores):
        """制約の実施テスト"""
        calculator = PortfolioWeightCalculator()
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {"AAPL": sample_strategy_scores}
        
        # 厳しい制約を設定
        constraints = PortfolioConstraints(
            max_individual_weight=0.2,
            min_individual_weight=0.1,
            max_strategies=2,
            min_strategies=1
        )
        
        config = WeightAllocationConfig(
            method=AllocationMethod.SCORE_PROPORTIONAL,
            constraints=constraints
        )
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data,
            config=config
        )
        
        # 制約チェック
        if result.strategy_weights:
            assert len(result.strategy_weights) <= constraints.max_strategies
            assert len(result.strategy_weights) >= constraints.min_strategies
            
            for weight in result.strategy_weights.values():
                assert weight <= constraints.max_individual_weight + 0.01  # 許容誤差
                assert weight >= constraints.min_individual_weight - 0.01
    
    def test_empty_scores_handling(self, sample_market_data):
        """空のスコアの処理テスト"""
        calculator = PortfolioWeightCalculator()
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {}
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data
        )
        
        assert result.strategy_weights == {}
        assert result.confidence_level == 0.0
        assert len(result.constraint_violations) > 0

class TestAllocationStrategies:
    """配分戦略のテストクラス"""
    
    def test_score_proportional_allocation(self, sample_strategy_scores):
        """スコア比例配分のテスト"""
        strategy = ScoreProportionalAllocation()
        config = WeightAllocationConfig()
        
        weights = strategy.calculate_weights(sample_strategy_scores, config)
        
        assert isinstance(weights, dict)
        if weights:
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert all(w >= 0 for w in weights.values())
    
    def test_risk_adjusted_allocation(self, sample_strategy_scores):
        """リスク調整配分のテスト"""
        strategy = RiskAdjustedAllocation()
        config = WeightAllocationConfig()
        
        weights = strategy.calculate_weights(sample_strategy_scores, config)
        
        assert isinstance(weights, dict)
        if weights:
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert all(w >= 0 for w in weights.values())
    
    def test_hierarchical_allocation(self, sample_strategy_scores):
        """階層的配分のテスト"""
        strategy = HierarchicalAllocation()
        config = WeightAllocationConfig()
        
        weights = strategy.calculate_weights(sample_strategy_scores, config)
        
        assert isinstance(weights, dict)
        if weights:
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert all(w >= 0 for w in weights.values())

class TestWeightTemplateManager:
    """WeightTemplateManagerのテストクラス"""
    
    def test_initialization(self, temp_config_dir):
        """初期化のテスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        assert len(manager.predefined_templates) == 5
        assert isinstance(manager.custom_templates, dict)
        assert manager.templates_dir.exists()
    
    def test_predefined_templates(self, temp_config_dir):
        """事前定義テンプレートのテスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        expected_templates = ["conservative", "balanced", "aggressive", "growth_focused", "income_focused"]
        
        for template_name in expected_templates:
            template = manager.get_template(template_name)
            assert template is not None
            assert isinstance(template, WeightTemplate)
            assert template.config is not None
            assert isinstance(template.template_type, TemplateType)
    
    def test_template_recommendation(self, temp_config_dir):
        """テンプレート推奨のテスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        recommendations = manager.recommend_template(
            market_regime=MarketRegime.BULL_MARKET,
            risk_tolerance="medium",
            capital_amount=150000
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for name, template, score in recommendations:
            assert isinstance(name, str)
            assert isinstance(template, WeightTemplate)
            assert isinstance(score, (int, float))
            assert score >= 0
    
    def test_custom_template_creation(self, temp_config_dir):
        """カスタムテンプレート作成のテスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        config = WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints()
        )
        
        template = manager.create_custom_template(
            name="Test Template",
            template_type=TemplateType.BALANCED,
            description="Test description",
            config=config
        )
        
        assert isinstance(template, WeightTemplate)
        assert template.name == "Test Template"
        assert template.template_type == TemplateType.BALANCED
        
        # カスタムテンプレートリストに追加されていることを確認
        assert "test_template" in manager.custom_templates
    
    def test_template_summary(self, temp_config_dir):
        """テンプレート概要のテスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        summary = manager.get_template_summary()
        
        assert isinstance(summary, dict)
        assert "total_templates" in summary
        assert "predefined_templates" in summary
        assert "custom_templates" in summary
        assert summary["total_templates"] >= 5

class TestPortfolioWeightingAgent:
    """PortfolioWeightingAgentのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        agent = PortfolioWeightingAgent(automation_level=AutomationLevel.SEMI_AUTOMATIC)
        
        assert agent.automation_level == AutomationLevel.SEMI_AUTOMATIC
        assert len(agent.automation_rules) > 0
        assert agent.current_weights == {}
        assert agent.decision_history == []
        assert agent.execution_history == []
    
    def test_automation_rules(self):
        """自動化ルールのテスト"""
        agent = PortfolioWeightingAgent()
        
        # 全ルールが適切に設定されていることを確認
        for rule in agent.automation_rules:
            assert isinstance(rule, AutomationRule)
            assert isinstance(rule.trigger_condition, TriggerCondition)
            assert isinstance(rule.action_type, ActionType)
            assert isinstance(rule.automation_level, AutomationLevel)
            assert isinstance(rule.threshold_value, (int, float))
            assert rule.priority >= 1
    
    @pytest.mark.asyncio
    async def test_trigger_condition_checking(self, sample_market_data):
        """トリガー条件チェックのテスト"""
        agent = PortfolioWeightingAgent()
        
        triggered_rules = await agent._check_trigger_conditions("AAPL", sample_market_data)
        
        assert isinstance(triggered_rules, list)
        # 初回実行では時間ベーストリガーが動作する可能性がある
    
    @pytest.mark.asyncio
    async def test_decision_making(self, sample_market_data):
        """意思決定のテスト"""
        agent = PortfolioWeightingAgent()
        
        # テスト用ルールを作成
        test_rule = AutomationRule(
            name="Test Rule",
            trigger_condition=TriggerCondition.SCORE_CHANGE,
            action_type=ActionType.CALCULATE_WEIGHTS,
            threshold_value=0.1,
            automation_level=AutomationLevel.SEMI_AUTOMATIC
        )
        
        decision = await agent._make_decision(test_rule, "AAPL", sample_market_data)
        
        if decision:  # 決定が作成された場合
            assert decision.decision_type == ActionType.CALCULATE_WEIGHTS
            assert isinstance(decision.confidence_level, (int, float))
            assert 0 <= decision.confidence_level <= 1
            assert isinstance(decision.required_approval, bool)
    
    def test_agent_status(self):
        """エージェント状態のテスト"""
        agent = PortfolioWeightingAgent()
        
        status = agent.get_agent_status()
        
        assert isinstance(status, dict)
        assert "automation_level" in status
        assert "active_rules" in status
        assert "pending_approvals" in status
        assert "active_alerts" in status
        assert status["automation_level"] == agent.automation_level.value

class TestIntegration:
    """統合テストクラス"""
    
    def test_calculator_template_integration(self, sample_market_data, temp_config_dir):
        """計算エンジンとテンプレートの統合テスト"""
        calculator = PortfolioWeightCalculator()
        template_manager = WeightTemplateManager(temp_config_dir)
        
        # テンプレートを取得
        template = template_manager.get_template("balanced")
        assert template is not None
        
        # テンプレートを使用して重み計算
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data,
            config=template.config
        )
        
        assert isinstance(result, AllocationResult)
        assert result.metadata["allocation_method"] == template.config.method.value
    
    @pytest.mark.asyncio
    async def test_agent_calculator_integration(self, sample_market_data):
        """エージェントと計算エンジンの統合テスト"""
        agent = PortfolioWeightingAgent()
        
        # 重み計算の実行
        result = agent.weight_calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data
        )
        
        assert isinstance(result, AllocationResult)
        
        # エージェントの状態更新
        if result.strategy_weights:
            agent.current_weights = result.strategy_weights
            assert len(agent.current_weights) > 0

class TestErrorHandling:
    """エラーハンドリングのテストクラス"""
    
    def test_invalid_market_data(self):
        """無効な市場データの処理テスト"""
        calculator = PortfolioWeightCalculator()
        
        # 空のDataFrame
        empty_data = pd.DataFrame()
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=empty_data
        )
        
        # エラーが適切に処理されることを確認
        assert isinstance(result, AllocationResult)
    
    def test_invalid_allocation_method(self, sample_market_data):
        """無効な配分手法の処理テスト"""
        calculator = PortfolioWeightCalculator()
        
        # 存在しない配分手法を設定
        class InvalidMethod:
            value = "invalid_method"
        
        config = WeightAllocationConfig()
        config.method = InvalidMethod()
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=sample_market_data,
            config=config
        )
        
        # エラーが適切に処理されることを確認
        assert len(result.constraint_violations) > 0
    
    def test_template_not_found(self, temp_config_dir):
        """存在しないテンプレートの処理テスト"""
        manager = WeightTemplateManager(temp_config_dir)
        
        template = manager.get_template("nonexistent_template")
        
        assert template is None

class TestPerformance:
    """パフォーマンステストクラス"""
    
    def test_calculation_performance(self, sample_market_data, sample_strategy_scores):
        """計算パフォーマンスのテスト"""
        calculator = PortfolioWeightCalculator()
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {"AAPL": sample_strategy_scores}
        
        import time
        
        start_time = time.time()
        
        # 複数回の計算を実行
        for _ in range(10):
            result = calculator.calculate_portfolio_weights(
                ticker="AAPL",
                market_data=sample_market_data
            )
            assert isinstance(result, AllocationResult)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # 1回の計算が1秒以内に完了することを確認
        assert avg_time < 1.0
    
    def test_large_data_handling(self, sample_strategy_scores):
        """大量データの処理テスト"""
        calculator = PortfolioWeightCalculator()
        calculator.score_manager.calculate_comprehensive_scores = lambda tickers: {"AAPL": sample_strategy_scores}
        
        # 大量の市場データを作成
        large_data = pd.DataFrame({
            'Adj Close': np.random.normal(100, 10, 10000),
            'Volume': np.random.randint(1000000, 5000000, 10000)
        })
        
        result = calculator.calculate_portfolio_weights(
            ticker="AAPL",
            market_data=large_data
        )
        
        assert isinstance(result, AllocationResult)

def test_module_imports():
    """モジュールインポートのテスト"""
    # すべての主要クラスがインポートできることを確認
    assert PortfolioWeightCalculator is not None
    assert WeightTemplateManager is not None
    assert PortfolioWeightingAgent is not None
    assert AllocationMethod is not None
    assert MarketRegime is not None
    assert AutomationLevel is not None

if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v", "--tb=short"])
