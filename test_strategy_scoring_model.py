"""
Test script for Strategy Scoring Model
戦略スコアリングモデルのテストスクリプト

Usage: python test_strategy_scoring_model.py
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from config.strategy_scoring_model import (
    StrategyScoreCalculator, 
    StrategyScoreReporter, 
    StrategyScoreManager,
    ScoreWeights,
    StrategyScore
)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyScoreModelTester:
    """戦略スコアリングモデルのテストクラス"""
    
    def __init__(self):
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        logger.info("=== Starting Strategy Scoring Model Tests ===")
        
        tests = [
            self.test_score_weights,
            self.test_strategy_score_object,
            self.test_score_calculator_initialization,
            self.test_component_score_calculation,
            self.test_trend_fitness_calculation,
            self.test_total_score_calculation,
            self.test_reporter_functionality,
            self.test_manager_integration,
            self.test_batch_processing,
            self.test_error_handling
        ]
        
        for test in tests:
            try:
                test()
                self.test_results.append((test.__name__, "PASSED", None))
                logger.info(f"✓ {test.__name__} PASSED")
            except Exception as e:
                self.test_results.append((test.__name__, "FAILED", str(e)))
                logger.error(f"✗ {test.__name__} FAILED: {e}")
        
        self.print_test_summary()
        
    def test_score_weights(self):
        """重み設定のテスト"""
        # デフォルト重み
        weights = ScoreWeights()
        total = weights.performance + weights.stability + weights.risk_adjusted + weights.trend_adaptation + weights.reliability
        assert abs(total - 1.0) < 0.001, f"Default weights don't sum to 1.0: {total}"
        
        # カスタム重み（正規化テスト）
        custom_weights = ScoreWeights(performance=0.5, stability=0.3, risk_adjusted=0.1, trend_adaptation=0.05, reliability=0.05)
        custom_total = custom_weights.performance + custom_weights.stability + custom_weights.risk_adjusted + custom_weights.trend_adaptation + custom_weights.reliability
        assert abs(custom_total - 1.0) < 0.001, f"Custom weights don't sum to 1.0: {custom_total}"
        
    def test_strategy_score_object(self):
        """StrategyScoreオブジェクトのテスト"""
        score = StrategyScore(
            strategy_name="test_strategy",
            ticker="AAPL",
            total_score=0.75,
            component_scores={
                'performance': 0.8,
                'stability': 0.7,
                'risk_adjusted': 0.6,
                'reliability': 0.9
            },
            trend_fitness=0.6,
            confidence=0.8,
            metadata={'test': True},
            calculated_at=datetime.now()
        )
        
        # to_dict()メソッドのテスト
        score_dict = score.to_dict()
        assert score_dict['strategy_name'] == "test_strategy"
        assert score_dict['ticker'] == "AAPL"
        assert score_dict['total_score'] == 0.75
        assert 'calculated_at' in score_dict
        
    def test_score_calculator_initialization(self):
        """スコア計算器の初期化テスト"""
        calculator = StrategyScoreCalculator()
        assert calculator.weights is not None
        assert calculator.data_loader is not None
        assert isinstance(calculator._cache, dict)
        
    def test_component_score_calculation(self):
        """コンポーネントスコア計算のテスト"""
        calculator = StrategyScoreCalculator()
        
        # テストデータを作成
        ticker_data = {
            'performance_metrics': {
                'total_return': 25.0,
                'win_rate': 0.65,
                'profit_factor': 1.8,
                'volatility': 0.15,
                'max_drawdown': -0.12,
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.5
            },
            'last_updated': datetime.now().isoformat(),
            'performance_history': list(range(50))  # 50件のデータ
        }
        
        component_scores = calculator._calculate_component_scores(ticker_data)
        
        # すべてのコンポーネントスコアが存在することを確認
        expected_components = ['performance', 'stability', 'risk_adjusted', 'reliability']
        for component in expected_components:
            assert component in component_scores, f"Missing component: {component}"
            assert 0 <= component_scores[component] <= 1, f"Invalid score range for {component}: {component_scores[component]}"
        
    def test_trend_fitness_calculation(self):
        """トレンド適合度計算のテスト"""
        calculator = StrategyScoreCalculator()
        
        ticker_data = {
            'trend_suitability': {
                'uptrend': 0.8,
                'downtrend': 0.3,
                'neutral': 0.6
            }
        }
        
        # トレンドコンテキストありのテスト
        trend_context = {
            'current_trend': 'uptrend',
            'trend_strength': 0.7
        }
        
        trend_fitness = calculator._calculate_trend_fitness(
            "test_strategy", ticker_data, trend_context=trend_context
        )
        
        assert 0 <= trend_fitness <= 1, f"Invalid trend fitness: {trend_fitness}"
        
        # トレンドコンテキストなしのテスト
        trend_fitness_default = calculator._calculate_trend_fitness(
            "test_strategy", ticker_data
        )
        
        assert 0 <= trend_fitness_default <= 1, f"Invalid default trend fitness: {trend_fitness_default}"
        
    def test_total_score_calculation(self):
        """総合スコア計算のテスト"""
        calculator = StrategyScoreCalculator()
        
        component_scores = {
            'performance': 0.8,
            'stability': 0.7,
            'risk_adjusted': 0.6,
            'reliability': 0.9
        }
        
        trend_fitness = 0.75
        
        total_score = calculator._calculate_total_score(component_scores, trend_fitness)
        
        assert 0 <= total_score <= 1, f"Invalid total score: {total_score}"
        
        # 重み付き平均の計算確認
        expected_score = (
            0.8 * calculator.weights.performance +
            0.7 * calculator.weights.stability +
            0.6 * calculator.weights.risk_adjusted +
            0.75 * calculator.weights.trend_adaptation +
            0.9 * calculator.weights.reliability
        )
        
        assert abs(total_score - expected_score) < 0.01, f"Score calculation mismatch: {total_score} vs {expected_score}"
        
    def test_reporter_functionality(self):
        """レポーター機能のテスト"""
        reporter = StrategyScoreReporter()
        
        # テストスコアデータを作成
        test_scores = {
            'strategy1': {
                'AAPL': StrategyScore(
                    strategy_name="strategy1",
                    ticker="AAPL",
                    total_score=0.75,
                    component_scores={'performance': 0.8, 'stability': 0.7},
                    trend_fitness=0.6,
                    confidence=0.8,
                    metadata={},
                    calculated_at=datetime.now()
                )
            }
        }
        
        # レポート生成テスト
        report_path = reporter.generate_score_report(test_scores, "test_report")
        
        assert report_path is not None, "Report generation failed"
        assert os.path.exists(report_path), f"Report file not created: {report_path}"
        
        # 関連ファイルの存在確認
        base_name = report_path.replace('.json', '')
        csv_path = f"{base_name}_summary.csv"
        md_path = f"{base_name}.md"
        
        assert os.path.exists(csv_path), f"CSV summary not created: {csv_path}"
        assert os.path.exists(md_path), f"Markdown report not created: {md_path}"
        
    def test_manager_integration(self):
        """マネージャー統合テスト"""
        manager = StrategyScoreManager()
        
        assert manager.data_loader is not None
        assert manager.calculator is not None
        assert manager.reporter is not None
        
        # get_top_strategiesの基本テスト
        top_strategies = manager.get_top_strategies("AAPL", top_n=3)
        assert isinstance(top_strategies, list)
        
    def test_batch_processing(self):
        """バッチ処理のテスト"""
        calculator = StrategyScoreCalculator()
        
        strategies = ["strategy1", "strategy2"]
        tickers = ["AAPL", "MSFT"]
        
        # バッチ処理実行（データが不足していてもエラーにならないことを確認）
        batch_results = calculator.calculate_batch_scores(strategies, tickers)
        
        assert isinstance(batch_results, dict)
        assert len(batch_results) == len(strategies)
        
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        calculator = StrategyScoreCalculator()
        
        # 存在しない戦略・ティッカーでのスコア計算
        score = calculator.calculate_strategy_score("nonexistent_strategy", "INVALID")
        
        # エラー時にはNoneが返されることを確認
        assert score is None, "Expected None for invalid strategy/ticker combination"
        
        # 不正なデータでのコンポーネントスコア計算
        invalid_data = {'invalid_field': 'invalid_value'}
        component_scores = calculator._calculate_component_scores(invalid_data)
        
        # デフォルト値が返されることを確認
        assert isinstance(component_scores, dict)
        for score_value in component_scores.values():
            assert 0 <= score_value <= 1
            
    def print_test_summary(self):
        """テスト結果のサマリーを出力"""
        logger.info("\n=== Test Summary ===")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r[1] == "PASSED"])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info("\nFailed Tests:")
            for test_name, status, error in self.test_results:
                if status == "FAILED":
                    logger.info(f"  - {test_name}: {error}")
        
        return failed_tests == 0

def main():
    """メイン実行関数"""
    print("Strategy Scoring Model Test Script")
    print("=" * 50)
    
    tester = StrategyScoreModelTester()
    tester.run_all_tests()
    
    return tester.print_test_summary()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
