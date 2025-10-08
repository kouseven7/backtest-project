"""
A→B市場分類システムのテストデモ
Enhanced Walkforward System with Market Classification
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import logging

# パスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_market_classification_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用のマーケットデータを生成"""
    logger.info("Creating test market data...")
    
    # 日付範囲
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    
    data = {}
    np.random.seed(42)  # 再現性のため
    
    for symbol in symbols:
        # 基本価格データの生成
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, len(date_range)):
            # トレンドとボラティリティを期間によって変更
            if i < len(date_range) * 0.2:  # 2020年前半：下落トレンド（COVID）
                trend = -0.0005
                volatility = 0.03
            elif i < len(date_range) * 0.4:  # 2020年後半：回復トレンド
                trend = 0.002
                volatility = 0.025
            elif i < len(date_range) * 0.6:  # 2021年：強気トレンド
                trend = 0.001
                volatility = 0.02
            elif i < len(date_range) * 0.8:  # 2022年：弱気トレンド
                trend = -0.0008
                volatility = 0.028
            else:  # 2023年：横ばいトレンド
                trend = 0.0002
                volatility = 0.018
            
            # 価格変動の計算
            daily_return = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # OHLCV データの生成
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, len(closes))))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, len(closes))))
        volumes = np.random.randint(1000000, 10000000, len(closes))
        
        symbol_data = pd.DataFrame({
            'Open': closes * (1 + np.random.normal(0, 0.005, len(closes))),
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=date_range)
        
        data[symbol] = symbol_data
    
    # マルチインデックスDataFrameの作成
    multi_data = pd.concat(data, axis=1)
    
    logger.info(f"Created test data for {len(symbols)} symbols from {start_date} to {end_date}")
    return multi_data


def test_market_classification():
    """市場分類システムのテスト"""
    logger.info("=== Testing Market Classification System ===")
    
    try:
        from src.analysis.market_classification.market_classifier import MarketClassifier
        from src.analysis.market_classification.classification_analyzer import ClassificationAnalyzer
        
        # テストデータの生成
        test_data = create_test_data()
        
        # 市場分類器の作成
        classifier = MarketClassifier(
            lookback_periods=20,
            volatility_threshold=0.02,
            trend_threshold=0.001,
            confidence_threshold=0.6
        )
        
        # 分析器の作成
        analyzer = ClassificationAnalyzer()
        
        # 各シンボルの分類実行
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        
        for symbol in symbols:
            logger.info(f"Classifying market for {symbol}...")
            
            # シンボルデータの抽出
            symbol_data = test_data[symbol].dropna()
            
            # 複数期間での分類テスト
            periods = [
                ('2020-03-01', '2020-06-01', 'COVID Crash'),
                ('2020-07-01', '2020-12-01', 'Recovery'),
                ('2021-01-01', '2021-12-01', 'Bull Market'),
                ('2022-01-01', '2022-12-01', 'Bear Market'),
                ('2023-01-01', '2023-12-01', 'Sideways')
            ]
            
            for start_date, end_date, period_name in periods:
                period_data = symbol_data[start_date:end_date]
                
                if not period_data.empty:
                    # 分類実行
                    result = classifier.classify(period_data, symbol, mode="hybrid")
                    analyzer.add_result(result)
                    
                    logger.info(f"  {period_name}: {result.simple_condition.value} "
                              f"({result.detailed_condition.value}) - "
                              f"Confidence: {result.confidence:.3f}")
        
        # 分析結果の生成
        logger.info("Generating classification analysis...")
        summary = analyzer.get_distribution_summary()
        
        logger.info(f"Total Classifications: {summary.get('total_classifications', 0)}")
        logger.info("Simple Classification Distribution:")
        for condition, count in summary.get('simple_distribution', {}).items():
            logger.info(f"  {condition}: {count}")
        
        # レポート生成
        report_content = analyzer.generate_classification_report()
        
        # 結果の保存
        output_dir = "output/market_classification_test"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # レポート保存
        report_path = os.path.join(output_dir, f'classification_report_{timestamp}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # CSV エクスポート
        csv_path = os.path.join(output_dir, f'classification_results_{timestamp}.csv')
        analyzer.export_to_csv(csv_path)
        
        logger.info(f"Classification test completed. Results saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Market classification test failed: {e}")
        return False


def test_enhanced_walkforward():
    """拡張ウォークフォワードシステムのテスト"""
    logger.info("=== Testing Enhanced Walkforward System ===")
    
    try:
        from src.analysis.enhanced_walkforward.enhanced_walkforward_executor import EnhancedWalkforwardExecutor
        from src.analysis.enhanced_walkforward.classification_integration import ClassificationIntegration
        
        # テストデータの生成
        test_data = create_test_data()
        
        # 設定ファイルのパス
        config_path = "src/analysis/walkforward_config.json"
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Creating minimal config.")
            
            # 最小限の設定を作成
            minimal_config = {
                "test_scenarios": [
                    {
                        "name": "COVID_crash",
                        "start_date": "2020-03-01",
                        "end_date": "2020-06-01",
                        "market_condition": "volatile"
                    },
                    {
                        "name": "recovery",
                        "start_date": "2020-07-01", 
                        "end_date": "2020-12-01",
                        "market_condition": "recovery"
                    },
                    {
                        "name": "bull_market",
                        "start_date": "2021-01-01",
                        "end_date": "2021-12-01", 
                        "market_condition": "trending_bull"
                    }
                ],
                "symbols": ["AAPL", "MSFT", "GOOGL", "SPY"],
                "strategies": [
                    {"name": "VWAPBreakoutStrategy"},
                    {"name": "VWAPBounceStrategy"},
                    {"name": "MomentumInvestingStrategy"}
                ],
                "walkforward_config": {
                    "training_window_days": 252,
                    "testing_window_days": 63,
                    "step_size_days": 21
                }
            }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_config, f, indent=2)
        
        # 拡張ウォークフォワードエグゼキューターの作成
        enhanced_config = {
            "market_classification": {
                "enabled": True,
                "mode": "hybrid",
                "lookback_periods": 20,
                "confidence_threshold": 0.6
            }
        }
        
        executor = EnhancedWalkforwardExecutor(config_path, enhanced_config)
        
        # 実行
        logger.info("Executing enhanced walkforward with market classification...")
        results = executor.execute_enhanced_walkforward(test_data, mode="hybrid")
        
        # 結果の確認
        if 'error' not in results:
            logger.info("Enhanced walkforward execution successful!")
            
            # 結果の保存
            output_dir = "output/enhanced_walkforward_test"
            saved_files = executor.save_results(output_dir)
            
            logger.info("Results saved:")
            for file_type, file_path in saved_files.items():
                logger.info(f"  {file_type}: {file_path}")
            
            # 現在の市場推奨の取得
            current_recommendations = executor.get_strategy_recommendations_for_current_market(
                test_data.tail(60), ["AAPL", "SPY"]
            )
            
            logger.info("Current market recommendations:")
            market_summary = current_recommendations.get('market_summary', {})
            logger.info(f"  Dominant condition: {market_summary.get('dominant_condition', 'unknown')}")
            logger.info(f"  Consensus level: {market_summary.get('consensus_level', 0):.2f}")
            
        else:
            logger.error(f"Enhanced walkforward execution failed: {results.get('error')}")
            return False
        
        logger.info("Enhanced walkforward test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced walkforward test failed: {e}")
        return False


def test_integration():
    """統合テスト"""
    logger.info("=== Testing System Integration ===")
    
    try:
        from src.analysis.enhanced_walkforward.classification_integration import ClassificationIntegration
        
        # テストデータ
        test_data = create_test_data()
        
        # 設定ファイルパス
        config_path = "src/analysis/walkforward_config.json"
        
        # 統合システムの作成
        integration = ClassificationIntegration(config_path)
        
        # 基本設定の拡張テスト
        base_config = {
            "test_scenarios": [{"name": "test", "start_date": "2020-01-01", "end_date": "2020-12-31"}],
            "symbols": ["AAPL", "SPY"]
        }
        
        enhanced_config = integration.enhance_walkforward_config(base_config)
        
        logger.info("Enhanced config created successfully")
        logger.info(f"Market classification enabled: {enhanced_config.get('market_classification', {}).get('enabled', False)}")
        
        # 期間分類テスト
        classification_result = integration.classify_market_for_period(
            test_data['AAPL'], 'AAPL', '2020-03-01', '2020-06-01'
        )
        
        logger.info(f"Classification result for AAPL (COVID period):")
        logger.info(f"  Simple: {classification_result.simple_condition.value}")
        logger.info(f"  Detailed: {classification_result.detailed_condition.value}")
        logger.info(f"  Confidence: {classification_result.confidence:.3f}")
        
        # 戦略推奨テスト
        recommendations = integration.get_strategy_recommendations(classification_result)
        
        logger.info("Strategy recommendations:")
        logger.info(f"  Primary: {recommendations.get('primary_strategies', [])}")
        logger.info(f"  Secondary: {recommendations.get('secondary_strategies', [])}")
        
        logger.info("Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def main():
    """メイン実行関数"""
    logger.info("Starting A→B Market Classification System Demo")
    logger.info("=" * 60)
    
    test_results = {
        'market_classification': False,
        'enhanced_walkforward': False,
        'integration': False
    }
    
    # 1. 市場分類システムのテスト
    test_results['market_classification'] = test_market_classification()
    
    # 2. 統合テスト
    test_results['integration'] = test_integration()
    
    # 3. 拡張ウォークフォワードシステムのテスト
    test_results['enhanced_walkforward'] = test_enhanced_walkforward()
    
    # 結果サマリー
    logger.info("=" * 60)
    logger.info("Demo Results Summary:")
    
    for test_name, result in test_results.items():
        status = "[OK] PASSED" if result else "[ERROR] FAILED"
        logger.info(f"  {test_name}: {status}")
    
    overall_success = all(test_results.values())
    
    if overall_success:
        logger.info("[SUCCESS] All tests passed! A→B Market Classification System is working correctly.")
    else:
        logger.error("[WARNING]  Some tests failed. Please check the logs for details.")
    
    logger.info("Demo completed.")
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
