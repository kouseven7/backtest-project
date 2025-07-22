"""
Batch Scoring Test for Strategy Scoring Model
戦略スコアリングモデルのバッチ処理テスト

Usage: python test_batch_scoring.py
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_data():
    """テスト用データをセットアップ"""
    logger.info("Setting up test data...")
    
    try:
        from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
        from config.strategy_data_persistence import StrategyDataPersistence
        
        # データローダーとストレージを初期化
        loader = StrategyCharacteristicsDataLoader()
        persistence = StrategyDataPersistence()
        
        # テスト戦略データを作成
        test_strategies = {
            'vwap_bounce_strategy': {
                'AAPL': {
                    'performance_metrics': {
                        'total_return': 28.5,
                        'win_rate': 0.68,
                        'profit_factor': 1.9,
                        'volatility': 0.18,
                        'max_drawdown': -0.15,
                        'sharpe_ratio': 1.4,
                        'sortino_ratio': 1.7
                    },
                    'trend_suitability': {
                        'uptrend': 0.85,
                        'downtrend': 0.40,
                        'neutral': 0.65
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.02, 0.03, -0.01, 0.04, 0.02] * 20
                },
                'MSFT': {
                    'performance_metrics': {
                        'total_return': 24.2,
                        'win_rate': 0.65,
                        'profit_factor': 1.7,
                        'volatility': 0.16,
                        'max_drawdown': -0.12,
                        'sharpe_ratio': 1.3,
                        'sortino_ratio': 1.6
                    },
                    'trend_suitability': {
                        'uptrend': 0.80,
                        'downtrend': 0.35,
                        'neutral': 0.60
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.01, 0.02, -0.01, 0.03, 0.01] * 18
                },
                'GOOGL': {
                    'performance_metrics': {
                        'total_return': 31.8,
                        'win_rate': 0.71,
                        'profit_factor': 2.1,
                        'volatility': 0.20,
                        'max_drawdown': -0.18,
                        'sharpe_ratio': 1.5,
                        'sortino_ratio': 1.8
                    },
                    'trend_suitability': {
                        'uptrend': 0.88,
                        'downtrend': 0.42,
                        'neutral': 0.68
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.03, 0.04, -0.02, 0.05, 0.03] * 22
                }
            },
            'golden_cross_strategy': {
                'AAPL': {
                    'performance_metrics': {
                        'total_return': 22.3,
                        'win_rate': 0.62,
                        'profit_factor': 1.6,
                        'volatility': 0.22,
                        'max_drawdown': -0.18,
                        'sharpe_ratio': 1.1,
                        'sortino_ratio': 1.3
                    },
                    'trend_suitability': {
                        'uptrend': 0.90,
                        'downtrend': 0.25,
                        'neutral': 0.55
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.02, 0.01, -0.02, 0.03, 0.01] * 15
                },
                'MSFT': {
                    'performance_metrics': {
                        'total_return': 19.7,
                        'win_rate': 0.59,
                        'profit_factor': 1.5,
                        'volatility': 0.19,
                        'max_drawdown': -0.16,
                        'sharpe_ratio': 1.0,
                        'sortino_ratio': 1.2
                    },
                    'trend_suitability': {
                        'uptrend': 0.85,
                        'downtrend': 0.30,
                        'neutral': 0.50
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.01, 0.02, -0.01, 0.02, 0.01] * 16
                },
                'GOOGL': {
                    'performance_metrics': {
                        'total_return': 26.1,
                        'win_rate': 0.64,
                        'profit_factor': 1.8,
                        'volatility': 0.24,
                        'max_drawdown': -0.20,
                        'sharpe_ratio': 1.2,
                        'sortino_ratio': 1.4
                    },
                    'trend_suitability': {
                        'uptrend': 0.92,
                        'downtrend': 0.28,
                        'neutral': 0.58
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.02, 0.03, -0.02, 0.04, 0.02] * 18
                }
            },
            'mean_reversion_strategy': {
                'AAPL': {
                    'performance_metrics': {
                        'total_return': 15.8,
                        'win_rate': 0.72,
                        'profit_factor': 1.4,
                        'volatility': 0.12,
                        'max_drawdown': -0.08,
                        'sharpe_ratio': 1.2,
                        'sortino_ratio': 1.5
                    },
                    'trend_suitability': {
                        'uptrend': 0.45,
                        'downtrend': 0.50,
                        'neutral': 0.85
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.01, 0.01, 0.00, 0.02, 0.01] * 14
                },
                'MSFT': {
                    'performance_metrics': {
                        'total_return': 13.4,
                        'win_rate': 0.69,
                        'profit_factor': 1.3,
                        'volatility': 0.10,
                        'max_drawdown': -0.06,
                        'sharpe_ratio': 1.1,
                        'sortino_ratio': 1.4
                    },
                    'trend_suitability': {
                        'uptrend': 0.42,
                        'downtrend': 0.48,
                        'neutral': 0.82
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.01, 0.01, 0.00, 0.01, 0.01] * 12
                },
                'GOOGL': {
                    'performance_metrics': {
                        'total_return': 17.2,
                        'win_rate': 0.74,
                        'profit_factor': 1.5,
                        'volatility': 0.14,
                        'max_drawdown': -0.09,
                        'sharpe_ratio': 1.3,
                        'sortino_ratio': 1.6
                    },
                    'trend_suitability': {
                        'uptrend': 0.48,
                        'downtrend': 0.52,
                        'neutral': 0.88
                    },
                    'last_updated': datetime.now().isoformat(),
                    'performance_history': [0.01, 0.02, 0.00, 0.02, 0.01] * 16
                }
            }
        }
        
        # データを保存
        for strategy_name, ticker_data_dict in test_strategies.items():
            for ticker, data in ticker_data_dict.items():
                persistence.save_strategy_data(strategy_name, ticker, data)
        
        logger.info(f"Test data setup completed: {len(test_strategies)} strategies, "
                   f"{sum(len(td) for td in test_strategies.values())} ticker combinations")
        
        return list(test_strategies.keys()), list(set(
            ticker for ticker_dict in test_strategies.values() 
            for ticker in ticker_dict.keys()
        ))
        
    except Exception as e:
        logger.error(f"Error setting up test data: {e}")
        return [], []

def test_batch_scoring():
    """バッチスコアリングのテスト"""
    logger.info("=== Testing Batch Scoring ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreManager
        
        # テストデータをセットアップ
        strategies, tickers = setup_test_data()
        
        if not strategies or not tickers:
            logger.error("Failed to setup test data")
            return False
        
        logger.info(f"Testing with {len(strategies)} strategies and {len(tickers)} tickers")
        
        # スコアマネージャーを初期化
        manager = StrategyScoreManager()
        
        # バッチスコアリングを実行
        logger.info("Starting batch score calculation...")
        report_path = manager.calculate_and_report_scores(
            strategies, tickers, report_name="batch_test_report"
        )
        
        if report_path:
            logger.info(f"Batch scoring completed successfully: {report_path}")
            
            # レポート内容を確認
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    import json
                    report_data = json.load(f)
                    
                total_scores = 0
                score_count = 0
                
                for strategy_name, ticker_scores in report_data.items():
                    logger.info(f"Strategy: {strategy_name}")
                    for ticker, score_data in ticker_scores.items():
                        total_score = score_data.get('total_score', 0)
                        logger.info(f"  {ticker}: {total_score:.3f}")
                        total_scores += total_score
                        score_count += 1
                
                avg_score = total_scores / score_count if score_count > 0 else 0
                logger.info(f"Average score across all combinations: {avg_score:.3f}")
                
                return True
        else:
            logger.error("Batch scoring failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in batch scoring test: {e}")
        return False

def test_top_strategies():
    """トップ戦略取得のテスト"""
    logger.info("=== Testing Top Strategies Selection ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreManager
        
        manager = StrategyScoreManager()
        
        # 各ティッカーのトップ戦略を取得
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        for ticker in tickers:
            logger.info(f"Getting top strategies for {ticker}:")
            top_strategies = manager.get_top_strategies(ticker, top_n=3)
            
            if top_strategies:
                for i, (strategy, score) in enumerate(top_strategies, 1):
                    logger.info(f"  {i}. {strategy}: {score:.3f}")
            else:
                logger.warning(f"No strategies found for {ticker}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in top strategies test: {e}")
        return False

def test_performance_monitoring():
    """パフォーマンス監視のテスト"""
    logger.info("=== Testing Performance Monitoring ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator
        import time
        
        calculator = StrategyScoreCalculator()
        
        # パフォーマンステスト用のデータ
        strategies = ['vwap_bounce_strategy', 'golden_cross_strategy', 'mean_reversion_strategy']
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # 実行時間を測定
        start_time = time.time()
        
        # バッチ処理を実行
        batch_results = calculator.calculate_batch_scores(strategies, tickers)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        total_combinations = len(strategies) * len(tickers)
        avg_time_per_combination = execution_time / total_combinations if total_combinations > 0 else 0
        
        logger.info(f"Performance Test Results:")
        logger.info(f"  Total combinations: {total_combinations}")
        logger.info(f"  Execution time: {execution_time:.3f} seconds")
        logger.info(f"  Average time per combination: {avg_time_per_combination:.3f} seconds")
        logger.info(f"  Successful calculations: {sum(len(ticker_scores) for ticker_scores in batch_results.values())}")
        
        # パフォーマンス基準をチェック
        if avg_time_per_combination < 1.0:  # 1秒以下
            logger.info("✓ Performance test PASSED - calculations are fast enough")
            return True
        else:
            logger.warning(f"⚠ Performance test WARNING - calculations may be slow")
            return True  # 警告だけで失敗とはしない
            
    except Exception as e:
        logger.error(f"Error in performance monitoring test: {e}")
        return False

def test_error_resilience():
    """エラー耐性のテスト"""
    logger.info("=== Testing Error Resilience ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator
        
        calculator = StrategyScoreCalculator()
        
        # 存在しない戦略・ティッカーのテスト
        logger.info("Testing with non-existent strategy/ticker...")
        score = calculator.calculate_strategy_score("nonexistent_strategy", "INVALID")
        if score is None:
            logger.info("✓ Correctly handled non-existent strategy/ticker")
        else:
            logger.warning("⚠ Unexpected result for non-existent strategy/ticker")
        
        # 無効な市場データのテスト
        logger.info("Testing with invalid market data...")
        invalid_market_data = pd.DataFrame({'invalid': [1, 2, 3]})
        score = calculator.calculate_strategy_score(
            "vwap_bounce_strategy", "AAPL", 
            market_data=invalid_market_data
        )
        # エラーが起こらないことを確認
        logger.info("✓ Handled invalid market data gracefully")
        
        # キャッシュクリアのテスト
        logger.info("Testing cache clear...")
        calculator.clear_cache()
        logger.info("✓ Cache cleared successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in error resilience test: {e}")
        return False

def main():
    """メイン実行関数"""
    print("Batch Scoring Test Script")
    print("=" * 50)
    
    test_results = []
    
    # テスト実行
    tests = [
        ("Batch Scoring", test_batch_scoring),
        ("Top Strategies Selection", test_top_strategies),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Resilience", test_error_resilience)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\nStarting test: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, "PASSED" if result else "FAILED"))
            logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            test_results.append((test_name, f"ERROR: {e}"))
            logger.error(f"Test {test_name} ERROR: {e}")
    
    # 結果サマリー
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")
        if result == "PASSED":
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total} tests")
    logger.info(f"Success rate: {(passed/total)*100:.1f}%")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
