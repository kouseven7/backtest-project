"""
Simple Demo for Strategy Scoring Model
戦略スコアリングモデルの簡単なデモスクリプト

Usage: python simple_demo_scoring_model.py
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

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """サンプルデータを作成"""
    logger.info("Creating sample data for demonstration...")
    
    # サンプル戦略データ
    sample_strategies = {
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
                'performance_history': list(range(120))
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
                'performance_history': list(range(100))
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
                'performance_history': list(range(80))
            }
        }
    }
    
    return sample_strategies

def demo_basic_scoring():
    """基本的なスコアリングのデモ"""
    logger.info("=== Basic Scoring Demo ===")
    
    try:
        from config.strategy_scoring_model import (
            StrategyScoreCalculator, 
            ScoreWeights,
            StrategyScore
        )
        
        # カスタム重みでスコア計算器を作成
        calculator = StrategyScoreCalculator()
        logger.info(f"Score calculator initialized with weights: "
                   f"performance={calculator.weights.performance:.2f}, "
                   f"stability={calculator.weights.stability:.2f}")
        
        # サンプルデータを作成
        sample_data = create_sample_data()
        
        # 各戦略のスコアを計算（モック）
        scores = []
        for strategy_name, ticker_data_dict in sample_data.items():
            for ticker, ticker_data in ticker_data_dict.items():
                # コンポーネントスコアを計算
                component_scores = calculator._calculate_component_scores(ticker_data)
                
                # トレンド適合度を計算
                trend_context = {'current_trend': 'uptrend', 'trend_strength': 0.7}
                trend_fitness = calculator._calculate_trend_fitness(
                    strategy_name, ticker_data, trend_context=trend_context
                )
                
                # 信頼度を計算
                confidence = calculator._calculate_confidence(ticker_data, component_scores)
                
                # 総合スコアを計算
                total_score = calculator._calculate_total_score(component_scores, trend_fitness)
                
                # スコアオブジェクトを作成
                score = StrategyScore(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    total_score=total_score,
                    component_scores=component_scores,
                    trend_fitness=trend_fitness,
                    confidence=confidence,
                    metadata={'demo': True},
                    calculated_at=datetime.now()
                )
                
                scores.append(score)
                
                logger.info(f"Strategy: {strategy_name}")
                logger.info(f"  Total Score: {total_score:.3f}")
                logger.info(f"  Performance: {component_scores.get('performance', 0):.3f}")
                logger.info(f"  Stability: {component_scores.get('stability', 0):.3f}")
                logger.info(f"  Risk-Adjusted: {component_scores.get('risk_adjusted', 0):.3f}")
                logger.info(f"  Trend Fitness: {trend_fitness:.3f}")
                logger.info(f"  Confidence: {confidence:.3f}")
                logger.info("")
        
        # 最高スコアの戦略を表示
        best_strategy = max(scores, key=lambda s: s.total_score)
        logger.info(f"Best Strategy: {best_strategy.strategy_name} with score {best_strategy.total_score:.3f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error in basic scoring demo: {e}")
        return []

def demo_weight_customization():
    """重み設定カスタマイズのデモ"""
    logger.info("=== Weight Customization Demo ===")
    
    try:
        from config.strategy_scoring_model import ScoreWeights
        
        # デフォルト重み
        default_weights = ScoreWeights()
        logger.info("Default weights:")
        logger.info(f"  Performance: {default_weights.performance:.2f}")
        logger.info(f"  Stability: {default_weights.stability:.2f}")
        logger.info(f"  Risk-Adjusted: {default_weights.risk_adjusted:.2f}")
        logger.info(f"  Trend Adaptation: {default_weights.trend_adaptation:.2f}")
        logger.info(f"  Reliability: {default_weights.reliability:.2f}")
        
        # カスタム重み（安定性重視）
        stability_focused = ScoreWeights(
            performance=0.25,
            stability=0.40,
            risk_adjusted=0.20,
            trend_adaptation=0.10,
            reliability=0.05
        )
        logger.info("\nStability-focused weights:")
        logger.info(f"  Performance: {stability_focused.performance:.2f}")
        logger.info(f"  Stability: {stability_focused.stability:.2f}")
        logger.info(f"  Risk-Adjusted: {stability_focused.risk_adjusted:.2f}")
        logger.info(f"  Trend Adaptation: {stability_focused.trend_adaptation:.2f}")
        logger.info(f"  Reliability: {stability_focused.reliability:.2f}")
        
        # カスタム重み（パフォーマンス重視）
        performance_focused = ScoreWeights(
            performance=0.50,
            stability=0.15,
            risk_adjusted=0.15,
            trend_adaptation=0.15,
            reliability=0.05
        )
        logger.info("\nPerformance-focused weights:")
        logger.info(f"  Performance: {performance_focused.performance:.2f}")
        logger.info(f"  Stability: {performance_focused.stability:.2f}")
        logger.info(f"  Risk-Adjusted: {performance_focused.risk_adjusted:.2f}")
        logger.info(f"  Trend Adaptation: {performance_focused.trend_adaptation:.2f}")
        logger.info(f"  Reliability: {performance_focused.reliability:.2f}")
        
    except Exception as e:
        logger.error(f"Error in weight customization demo: {e}")

def demo_report_generation():
    """レポート生成のデモ"""
    logger.info("=== Report Generation Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreReporter, StrategyScore
        
        # レポーターを作成
        reporter = StrategyScoreReporter()
        
        # サンプルスコアデータを作成
        sample_scores = {
            'vwap_bounce_strategy': {
                'AAPL': StrategyScore(
                    strategy_name="vwap_bounce_strategy",
                    ticker="AAPL",
                    total_score=0.78,
                    component_scores={
                        'performance': 0.82,
                        'stability': 0.75,
                        'risk_adjusted': 0.70,
                        'reliability': 0.85
                    },
                    trend_fitness=0.80,
                    confidence=0.85,
                    metadata={'demo': True},
                    calculated_at=datetime.now()
                )
            },
            'golden_cross_strategy': {
                'AAPL': StrategyScore(
                    strategy_name="golden_cross_strategy",
                    ticker="AAPL",
                    total_score=0.68,
                    component_scores={
                        'performance': 0.65,
                        'stability': 0.60,
                        'risk_adjusted': 0.72,
                        'reliability': 0.75
                    },
                    trend_fitness=0.85,
                    confidence=0.80,
                    metadata={'demo': True},
                    calculated_at=datetime.now()
                )
            }
        }
        
        # レポートを生成
        report_path = reporter.generate_score_report(sample_scores, "demo_report")
        
        if report_path:
            logger.info(f"Demo report generated: {report_path}")
            
            # JSONファイルの内容を表示（最初の数行）
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')[:10]
                    logger.info("Report preview (first 10 lines):")
                    for line in lines:
                        logger.info(f"  {line}")
            except Exception as e:
                logger.warning(f"Could not preview report: {e}")
        else:
            logger.error("Report generation failed")
            
    except Exception as e:
        logger.error(f"Error in report generation demo: {e}")

def demo_error_handling():
    """エラーハンドリングのデモ"""
    logger.info("=== Error Handling Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator
        
        calculator = StrategyScoreCalculator()
        
        # 不正なデータでのテスト
        logger.info("Testing with invalid data...")
        
        # 空のデータ
        empty_scores = calculator._calculate_component_scores({})
        logger.info(f"Empty data component scores: {empty_scores}")
        
        # 部分的なデータ
        partial_data = {
            'performance_metrics': {
                'total_return': 10.0
                # 他のメトリクスが不足
            }
        }
        partial_scores = calculator._calculate_component_scores(partial_data)
        logger.info(f"Partial data component scores: {partial_scores}")
        
        # 不正な値
        invalid_data = {
            'performance_metrics': {
                'total_return': 'invalid',
                'win_rate': -1.0,  # 不正な値
                'volatility': None
            }
        }
        invalid_scores = calculator._calculate_component_scores(invalid_data)
        logger.info(f"Invalid data component scores: {invalid_scores}")
        
        logger.info("Error handling working correctly - no crashes!")
        
    except Exception as e:
        logger.error(f"Error in error handling demo: {e}")

def main():
    """メイン実行関数"""
    print("Strategy Scoring Model Demo")
    print("=" * 50)
    
    try:
        # 基本的なスコアリングのデモ
        scores = demo_basic_scoring()
        
        # 重み設定カスタマイズのデモ
        demo_weight_customization()
        
        # レポート生成のデモ
        demo_report_generation()
        
        # エラーハンドリングのデモ
        demo_error_handling()
        
        logger.info("=== Demo Completed Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
