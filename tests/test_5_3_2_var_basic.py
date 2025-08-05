"""
簡単なテスト実行スクリプト
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_5_3_2_var_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_test_data():
    """テストデータ作成"""
    try:
        logger.info("Creating test data...")
        
        # 基本設定
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        n_days = 200
        
        # リターンデータ生成
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        
        returns_data = pd.DataFrame(index=dates, columns=symbols)
        
        # 各銘柄のリターン生成
        for symbol in symbols:
            returns_data[symbol] = np.random.normal(0.001, 0.02, n_days)
        
        # ポートフォリオ重み
        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        
        logger.info(f"Test data created: {len(returns_data)} days, {len(symbols)} symbols")
        return returns_data, weights
        
    except Exception as e:
        logger.error(f"Test data creation error: {e}")
        raise

def test_basic_var_calculation():
    """基本VaR計算テスト"""
    try:
        logger.info("Testing basic VaR calculation...")
        
        # データ準備
        returns_data, weights = create_test_data()
        
        # VaR設定インポート
        from config.portfolio_var_calculator.advanced_var_engine import (
            VaRCalculationConfig, AdvancedVaREngine
        )
        
        # VaRエンジン初期化
        config = VaRCalculationConfig(
            confidence_levels=[0.95, 0.99],
            lookback_period=100,
            monte_carlo_simulations=1000
        )
        
        engine = AdvancedVaREngine(config)
        
        # VaR計算
        result = engine.calculate_comprehensive_var(returns_data, weights)
        
        # 結果表示
        logger.info(f"VaR 95%: {result.get_var_95():.4f}")
        logger.info(f"VaR 99%: {result.get_var_99():.4f}")
        logger.info(f"Market regime: {result.market_regime}")
        logger.info(f"Calculation method: {result.calculation_method}")
        
        logger.info("Basic VaR calculation test: PASSED ✓")
        return result
        
    except Exception as e:
        logger.error(f"Basic VaR calculation test error: {e}")
        raise

def test_hybrid_var_calculator():
    """ハイブリッドVaR計算テスト"""
    try:
        logger.info("Testing hybrid VaR calculator...")
        
        # データ準備
        returns_data, weights = create_test_data()
        
        # ハイブリッド計算器インポート
        from config.portfolio_var_calculator.advanced_var_engine import (
            VaRCalculationConfig, AdvancedVaREngine
        )
        from config.portfolio_var_calculator.hybrid_var_calculator import HybridVaRCalculator
        
        # エンジン初期化
        config = VaRCalculationConfig(
            confidence_levels=[0.95, 0.99],
            lookback_period=100
        )
        engine = AdvancedVaREngine(config)
        
        # ハイブリッド計算器
        hybrid_calc = HybridVaRCalculator(engine)
        
        # 計算実行
        result = hybrid_calc.calculate_hybrid_var(returns_data, weights)
        
        # 結果表示
        logger.info(f"Hybrid VaR 95%: {result.get_var_95():.4f}")
        logger.info(f"Hybrid VaR 99%: {result.get_var_99():.4f}")
        logger.info(f"Selected method: {result.calculation_method}")
        
        logger.info("Hybrid VaR calculator test: PASSED ✓")
        return result
        
    except Exception as e:
        logger.error(f"Hybrid VaR calculator test error: {e}")
        raise

def test_backtesting_engine():
    """バックテスティングエンジンテスト"""
    try:
        logger.info("Testing backtesting engine...")
        
        # より長期データの準備
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        n_days = 400
        
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        
        extended_data = pd.DataFrame(index=dates, columns=symbols)
        for symbol in symbols:
            extended_data[symbol] = np.random.normal(0.001, 0.02, n_days)
        
        # 重み履歴
        weights_history = {}
        for i, date in enumerate(dates[::30]):  # 月次重み更新
            weights_history[date] = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        
        # バックテストエンジンインポート
        from config.portfolio_var_calculator.advanced_var_engine import (
            VaRCalculationConfig, AdvancedVaREngine
        )
        from config.portfolio_var_calculator.var_backtesting_engine import (
            VaRBacktestingEngine, BacktestConfig
        )
        
        # エンジン初期化
        var_config = VaRCalculationConfig(confidence_levels=[0.95, 0.99])
        var_engine = AdvancedVaREngine(var_config)
        
        backtest_config = BacktestConfig(
            lookback_window=100,
            rolling_window=30,
            min_observations=50
        )
        
        backtest_engine = VaRBacktestingEngine(var_engine, None, backtest_config)
        
        # バックテスト実行
        start_date = dates[150]  # 十分な履歴を確保
        result = backtest_engine.run_comprehensive_backtest(
            extended_data, weights_history, start_date
        )
        
        # 結果表示
        logger.info(f"Backtest observations: {result.total_observations}")
        logger.info(f"VaR 95% violations: {result.var_95_violations} ({result.var_95_violation_rate:.2%})")
        logger.info(f"VaR 99% violations: {result.var_99_violations} ({result.var_99_violation_rate:.2%})")
        logger.info(f"Model accuracy: {result.model_accuracy_score:.3f}")
        logger.info(f"Calibration quality: {result.calibration_quality}")
        
        logger.info("Backtesting engine test: PASSED ✓")
        return result
        
    except Exception as e:
        logger.error(f"Backtesting engine test error: {e}")
        raise

def main():
    """メインテスト関数"""
    try:
        logger.info("=== 5-3-2 ポートフォリオVaR計算システム 基本機能テスト開始 ===")
        
        # 1. 基本VaR計算テスト
        basic_result = test_basic_var_calculation()
        
        # 2. ハイブリッドVaR計算テスト
        hybrid_result = test_hybrid_var_calculator()
        
        # 3. バックテスティングエンジンテスト
        backtest_result = test_backtesting_engine()
        
        # テストサマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info("✓ 基本VaR計算エンジン: 正常動作")
        logger.info("✓ ハイブリッドVaR計算器: 正常動作") 
        logger.info("✓ バックテスティングエンジン: 正常動作")
        
        logger.info("5-3-2 ポートフォリオVaR計算システム 基本機能実装完了")
        
        return {
            'basic_result': basic_result,
            'hybrid_result': hybrid_result,
            'backtest_result': backtest_result,
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        logger.error(f"Main test error: {e}")
        return {'status': 'FAILED', 'error': str(e)}

if __name__ == "__main__":
    result = main()
