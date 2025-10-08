"""
Real-world Usage Demo for Strategy Scoring Model
戦略スコアリングモデルの実運用シナリオデモ

このスクリプトは実際の運用を想定した使用例を示します。
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

def demo_daily_strategy_evaluation():
    """日次戦略評価のデモ"""
    logger.info("=== Daily Strategy Evaluation Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreManager
        
        # スコアマネージャーを初期化
        manager = StrategyScoreManager()
        
        # 評価対象の戦略とティッカー
        strategies = ["vwap_bounce_strategy", "golden_cross_strategy", "mean_reversion_strategy"]
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        logger.info(f"Evaluating {len(strategies)} strategies across {len(tickers)} tickers")
        
        # 日次レポートを生成
        today = datetime.now().strftime("%Y%m%d")
        report_path = manager.calculate_and_report_scores(
            strategies, tickers, 
            report_name=f"daily_strategy_evaluation_{today}"
        )
        
        if report_path:
            logger.info(f"Daily evaluation report generated: {report_path}")
            
            # 各ティッカーのトップ戦略を表示
            for ticker in tickers:
                top_strategies = manager.get_top_strategies(ticker, top_n=3)
                logger.info(f"Top strategies for {ticker}:")
                for i, (strategy, score) in enumerate(top_strategies, 1):
                    logger.info(f"  {i}. {strategy}: {score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in daily strategy evaluation: {e}")
        return False

def demo_strategy_comparison():
    """戦略比較分析のデモ"""
    logger.info("=== Strategy Comparison Analysis Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator, ScoreWeights
        
        # 異なる重み設定での比較
        weight_configs = {
            "balanced": ScoreWeights(),  # デフォルト
            "performance_focused": ScoreWeights(
                performance=0.50, stability=0.15, risk_adjusted=0.15, 
                trend_adaptation=0.15, reliability=0.05
            ),
            "stability_focused": ScoreWeights(
                performance=0.25, stability=0.40, risk_adjusted=0.20, 
                trend_adaptation=0.10, reliability=0.05
            ),
            "risk_focused": ScoreWeights(
                performance=0.25, stability=0.20, risk_adjusted=0.40, 
                trend_adaptation=0.10, reliability=0.05
            )
        }
        
        test_strategy = "vwap_bounce_strategy"
        test_ticker = "AAPL"
        
        logger.info(f"Comparing scoring approaches for {test_strategy} on {test_ticker}:")
        
        for config_name, weights in weight_configs.items():
            calculator = StrategyScoreCalculator()
            calculator.weights = weights
            
            # 実際にはここでスコア計算（データが不足のためシミュレート）
            logger.info(f"{config_name.upper()} approach:")
            logger.info(f"  Performance weight: {weights.performance:.2f}")
            logger.info(f"  Stability weight: {weights.stability:.2f}")
            logger.info(f"  Risk-adjusted weight: {weights.risk_adjusted:.2f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in strategy comparison: {e}")
        return False

def demo_portfolio_optimization():
    """ポートフォリオ最適化シナリオのデモ"""
    logger.info("=== Portfolio Optimization Scenario Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreManager
        
        manager = StrategyScoreManager()
        
        # ポートフォリオ対象ティッカー
        portfolio_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        available_strategies = ["vwap_bounce_strategy", "golden_cross_strategy", "mean_reversion_strategy"]
        
        logger.info("Portfolio optimization scenario:")
        logger.info(f"Target tickers: {portfolio_tickers}")
        logger.info(f"Available strategies: {available_strategies}")
        
        # 各ティッカーに最適な戦略を選択
        portfolio_allocation = {}
        
        for ticker in portfolio_tickers:
            top_strategies = manager.get_top_strategies(ticker, top_n=1)
            if top_strategies:
                best_strategy, score = top_strategies[0]
                portfolio_allocation[ticker] = {
                    "strategy": best_strategy,
                    "score": score,
                    "confidence": "High" if score > 0.7 else "Medium" if score > 0.5 else "Low"
                }
            else:
                portfolio_allocation[ticker] = {
                    "strategy": "No recommendation",
                    "score": 0.0,
                    "confidence": "None"
                }
        
        logger.info("\nPortfolio allocation recommendations:")
        for ticker, allocation in portfolio_allocation.items():
            logger.info(f"{ticker}: {allocation['strategy']} "
                       f"(Score: {allocation['score']:.3f}, "
                       f"Confidence: {allocation['confidence']})")
        
        # ポートフォリオ統計
        valid_scores = [alloc['score'] for alloc in portfolio_allocation.values() if alloc['score'] > 0]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            
            logger.info(f"\nPortfolio Statistics:")
            logger.info(f"Average Score: {avg_score:.3f}")
            logger.info(f"Score Range: {min_score:.3f} - {max_score:.3f}")
            logger.info(f"Coverage: {len(valid_scores)}/{len(portfolio_tickers)} tickers")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return False

def demo_risk_monitoring():
    """リスク監視シナリオのデモ"""
    logger.info("=== Risk Monitoring Scenario Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator
        
        calculator = StrategyScoreCalculator()
        
        # リスク閾値の設定
        risk_thresholds = {
            "min_total_score": 0.6,
            "min_stability_score": 0.5,
            "min_confidence": 0.7,
            "max_risk_concentration": 0.4  # 単一戦略の最大割合
        }
        
        logger.info("Risk monitoring thresholds:")
        for threshold, value in risk_thresholds.items():
            logger.info(f"  {threshold}: {value}")
        
        # モック戦略スコアでリスク評価
        mock_scores = {
            "strategy_a": {"total_score": 0.75, "stability": 0.60, "confidence": 0.80},
            "strategy_b": {"total_score": 0.45, "stability": 0.40, "confidence": 0.50},  # リスク
            "strategy_c": {"total_score": 0.65, "stability": 0.70, "confidence": 0.75}
        }
        
        logger.info("\nRisk assessment results:")
        
        risk_alerts = []
        for strategy, scores in mock_scores.items():
            alerts = []
            
            if scores["total_score"] < risk_thresholds["min_total_score"]:
                alerts.append(f"Low total score: {scores['total_score']:.3f}")
            
            if scores["stability"] < risk_thresholds["min_stability_score"]:
                alerts.append(f"Low stability: {scores['stability']:.3f}")
            
            if scores["confidence"] < risk_thresholds["min_confidence"]:
                alerts.append(f"Low confidence: {scores['confidence']:.3f}")
            
            if alerts:
                risk_alerts.extend([(strategy, alert) for alert in alerts])
                logger.warning(f"[WARNING]  {strategy}: {', '.join(alerts)}")
            else:
                logger.info(f"[OK] {strategy}: All risk checks passed")
        
        if risk_alerts:
            logger.warning(f"\nTotal risk alerts: {len(risk_alerts)}")
            logger.warning("Recommended actions: Review strategy parameters, increase monitoring frequency")
        else:
            logger.info("\n[OK] No risk alerts - Portfolio within acceptable risk levels")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in risk monitoring: {e}")
        return False

def demo_performance_tracking():
    """パフォーマンス追跡のデモ"""
    logger.info("=== Performance Tracking Demo ===")
    
    try:
        from config.strategy_scoring_model import StrategyScoreCalculator
        
        calculator = StrategyScoreCalculator()
        
        # 模擬的な過去データ
        historical_performance = {
            "2025-07-01": {"vwap_bounce": 0.72, "golden_cross": 0.68, "mean_reversion": 0.55},
            "2025-07-02": {"vwap_bounce": 0.74, "golden_cross": 0.65, "mean_reversion": 0.58},
            "2025-07-03": {"vwap_bounce": 0.71, "golden_cross": 0.70, "mean_reversion": 0.60},
            "2025-07-04": {"vwap_bounce": 0.75, "golden_cross": 0.67, "mean_reversion": 0.57},
            "2025-07-05": {"vwap_bounce": 0.73, "golden_cross": 0.69, "mean_reversion": 0.59}
        }
        
        logger.info("Historical performance tracking:")
        
        # 各戦略のトレンド分析
        for strategy in ["vwap_bounce", "golden_cross", "mean_reversion"]:
            scores = [daily_scores[strategy] for daily_scores in historical_performance.values()]
            
            avg_score = np.mean(scores)
            trend = "[UP] Improving" if scores[-1] > scores[0] else "[DOWN] Declining" if scores[-1] < scores[0] else "➡️ Stable"
            volatility = np.std(scores)
            
            logger.info(f"{strategy}:")
            logger.info(f"  Average Score: {avg_score:.3f}")
            logger.info(f"  Trend: {trend}")
            logger.info(f"  Volatility: {volatility:.3f}")
            logger.info(f"  Latest Score: {scores[-1]:.3f}")
        
        # キャッシュ統計の表示
        cache_stats = calculator.get_cache_stats() if hasattr(calculator, 'get_cache_stats') else {}
        if cache_stats:
            logger.info(f"\nCache Performance:")
            logger.info(f"  Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
            logger.info(f"  Cache Size: {cache_stats.get('cache_size', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in performance tracking: {e}")
        return False

def main():
    """メイン実行関数"""
    print("Strategy Scoring Model - Real-world Usage Demo")
    print("=" * 60)
    
    scenarios = [
        ("Daily Strategy Evaluation", demo_daily_strategy_evaluation),
        ("Strategy Comparison Analysis", demo_strategy_comparison),
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Risk Monitoring", demo_risk_monitoring),
        ("Performance Tracking", demo_performance_tracking)
    ]
    
    results = []
    
    for scenario_name, scenario_func in scenarios:
        logger.info(f"\n{'='*20} {scenario_name} {'='*20}")
        try:
            success = scenario_func()
            results.append((scenario_name, "SUCCESS" if success else "FAILED"))
        except Exception as e:
            logger.error(f"Scenario {scenario_name} failed: {e}")
            results.append((scenario_name, "ERROR"))
    
    # 結果サマリー
    logger.info("\n" + "="*60)
    logger.info("SCENARIO EXECUTION SUMMARY")
    logger.info("="*60)
    
    success_count = 0
    for scenario_name, result in results:
        status_icon = "[OK]" if result == "SUCCESS" else "[ERROR]"
        logger.info(f"{status_icon} {scenario_name}: {result}")
        if result == "SUCCESS":
            success_count += 1
    
    logger.info(f"\nSuccess Rate: {success_count}/{len(results)} scenarios")
    
    # 実運用への推奨事項
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS FOR PRODUCTION")
    logger.info("="*60)
    logger.info("1. Set up automated daily scoring with real market data")
    logger.info("2. Configure risk alert thresholds based on your risk tolerance")
    logger.info("3. Implement database storage for historical score tracking")
    logger.info("4. Set up monitoring dashboard for real-time score visualization")
    logger.info("5. Schedule regular model calibration based on performance feedback")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
