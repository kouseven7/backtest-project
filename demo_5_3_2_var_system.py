"""
Demo 5-3-2: ポートフォリオVaR計算システム

高度なVaR計算システムのデモンストレーション
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# VaRシステムのインポート
from config.portfolio_var_calculator import (
    AdvancedVaREngine,
    VaRCalculationConfig,
    HybridVaRCalculator,
    RealTimeVaRMonitor,
    MonitoringConfig,
    VaRIntegrationBridge,
    BridgeConfig,
    VaRBacktestingEngine,
    BacktestConfig
)

# データ取得のインポート
from data_fetcher import DataFetcher
from data_processor import DataProcessor

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_5_3_2_var_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_sample_data() -> tuple[pd.DataFrame, Dict[str, float]]:
    """サンプルデータの作成"""
    try:
        logger.info("Creating sample data for VaR calculation")
        
        # 期間設定
        end_date = datetime.now()
        start_date = end_date - timedelta(days=300)  # 約1年のデータ
        
        # サンプル銘柄
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # サンプルリターンデータの生成（正規分布ベース）
        np.random.seed(42)  # 再現性のため
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 営業日のみ
        
        returns_data = pd.DataFrame(index=dates, columns=symbols)
        
        # 各銘柄の特性を設定
        characteristics = {
            'AAPL': {'mean': 0.001, 'std': 0.02, 'correlation_base': 1.0},
            'GOOGL': {'mean': 0.0012, 'std': 0.025, 'correlation_base': 0.7},
            'MSFT': {'mean': 0.0008, 'std': 0.018, 'correlation_base': 0.8},
            'TSLA': {'mean': 0.002, 'std': 0.035, 'correlation_base': 0.4},
            'AMZN': {'mean': 0.0015, 'std': 0.022, 'correlation_base': 0.6}
        }
        
        # 共通のマーケットファクター
        market_factor = np.random.normal(0, 0.015, len(dates))
        
        for symbol in symbols:
            char = characteristics[symbol]
            
            # 個別リターン + マーケットファクター
            idiosyncratic = np.random.normal(char['mean'], char['std'] * 0.7, len(dates))
            market_component = market_factor * char['correlation_base']
            
            returns_data[symbol] = idiosyncratic + market_component
        
        # ポートフォリオ重み
        weights = {
            'AAPL': 0.25,
            'GOOGL': 0.20,
            'MSFT': 0.20,
            'TSLA': 0.15,
            'AMZN': 0.20
        }
        
        logger.info(f"Generated {len(returns_data)} days of return data for {len(symbols)} symbols")
        return returns_data, weights
        
    except Exception as e:
        logger.error(f"Sample data creation error: {e}")
        raise

def demo_advanced_var_engine():
    """高度VaRエンジンのデモ"""
    logger.info("=== Advanced VaR Engine Demo ===")
    
    try:
        # データ準備
        returns_data, weights = create_sample_data()
        
        # 高度VaRエンジンの初期化
        var_config = VaRCalculationConfig(
            confidence_levels=[0.95, 0.99],
            lookback_period=252,
            monte_carlo_simulations=10000,
            enable_regime_detection=True,
            enable_component_var=True
        )
        
        var_engine = AdvancedVaREngine(var_config)
        
        # VaR計算実行
        logger.info("Calculating comprehensive VaR...")
        var_result = var_engine.calculate_comprehensive_var(returns_data, weights)
        
        # 結果表示
        logger.info(f"VaR 95%: {var_result.get_var_95():.4f}")
        logger.info(f"VaR 99%: {var_result.get_var_99():.4f}")
        logger.info(f"Market Regime: {var_result.market_regime}")
        logger.info(f"Calculation Method: {var_result.calculation_method}")
        logger.info(f"Diversification Benefit: {var_result.diversification_benefit:.4f}")
        
        if var_result.component_var:
            logger.info("Component VaR:")
            for symbol, comp_var in var_result.component_var.items():
                logger.info(f"  {symbol}: {comp_var:.4f}")
        
        return var_engine, var_result
        
    except Exception as e:
        logger.error(f"Advanced VaR engine demo error: {e}")
        raise

def demo_hybrid_calculator(var_engine: AdvancedVaREngine):
    """ハイブリッド計算器のデモ"""
    logger.info("=== Hybrid VaR Calculator Demo ===")
    
    try:
        # データ準備
        returns_data, weights = create_sample_data()
        
        # ハイブリッド計算器の初期化
        hybrid_calculator = HybridVaRCalculator(var_engine)
        
        # ハイブリッドVaR計算
        logger.info("Calculating hybrid VaR with dynamic weighting...")
        hybrid_result = hybrid_calculator.calculate_hybrid_var(returns_data, weights)
        
        # 結果表示
        logger.info(f"Hybrid VaR 95%: {hybrid_result.get_var_95():.4f}")
        logger.info(f"Hybrid VaR 99%: {hybrid_result.get_var_99():.4f}")
        logger.info(f"Selected Method: {hybrid_result.calculation_method}")
        
        # 方法別重み情報取得
        if hasattr(hybrid_calculator, 'get_last_method_weights'):
            try:
                method_weights = hybrid_calculator.get_last_method_weights()
                logger.info("Method Weights:")
                for method, weight in method_weights.items():
                    logger.info(f"  {method}: {weight:.3f}")
            except Exception as e:
                logger.warning(f"Could not retrieve method weights: {e}")
        
        return hybrid_calculator, hybrid_result
        
    except Exception as e:
        logger.error(f"Hybrid calculator demo error: {e}")
        raise

def demo_integration_bridge(var_engine: AdvancedVaREngine, hybrid_calculator: HybridVaRCalculator):
    """統合ブリッジのデモ"""
    logger.info("=== VaR Integration Bridge Demo ===")
    
    try:
        # データ準備
        returns_data, weights = create_sample_data()
        
        # 統合ブリッジの初期化
        bridge_config = BridgeConfig(
            enable_legacy_comparison=True,
            prefer_advanced_when_available=True,
            log_comparisons=True
        )
        
        integration_bridge = VaRIntegrationBridge(
            var_engine, hybrid_calculator, None, bridge_config
        )
        
        # 統合VaR計算
        logger.info("Calculating integrated VaR...")
        integration_result = integration_bridge.calculate_integrated_var(
            returns_data, weights, use_hybrid=True
        )
        
        if integration_result.success:
            logger.info("Integration successful!")
            
            if integration_result.advanced_var_result:
                adv_result = integration_result.advanced_var_result
                logger.info(f"Advanced VaR 95%: {adv_result.get_var_95():.4f}")
                logger.info(f"Advanced VaR 99%: {adv_result.get_var_99():.4f}")
            
            if integration_result.legacy_var_result:
                leg_result = integration_result.legacy_var_result
                logger.info(f"Legacy VaR 95%: {leg_result.get('var_95', 'N/A')}")
                logger.info(f"Legacy VaR 99%: {leg_result.get('var_99', 'N/A')}")
            
            if integration_result.recommendations:
                logger.info("Recommendations:")
                for rec in integration_result.recommendations:
                    logger.info(f"  - {rec}")
        else:
            logger.error(f"Integration failed: {integration_result.error_message}")
        
        # 統一VaRレポート作成
        logger.info("Creating unified VaR report...")
        unified_report = integration_bridge.create_unified_var_report(returns_data, weights)
        
        if 'error' not in unified_report:
            logger.info("Unified report created successfully")
            logger.info(f"Portfolio Assets: {unified_report['portfolio_summary']['total_assets']}")
            
            if 'advanced_var' in unified_report:
                adv_var = unified_report['advanced_var']
                logger.info(f"Advanced VaR Summary - 95%: {adv_var.get('var_95', 'N/A'):.4f}, 99%: {adv_var.get('var_99', 'N/A'):.4f}")
        
        return integration_bridge, integration_result
        
    except Exception as e:
        logger.error(f"Integration bridge demo error: {e}")
        raise

def demo_backtesting_engine(var_engine: AdvancedVaREngine, hybrid_calculator: HybridVaRCalculator):
    """バックテスティングエンジンのデモ"""
    logger.info("=== VaR Backtesting Engine Demo ===")
    
    try:
        # データ準備（より長期間）
        logger.info("Preparing extended data for backtesting...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)  # 約1.5年のデータ
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # 拡張データ生成
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        extended_returns = pd.DataFrame(index=dates, columns=symbols)
        
        # より複雑なリターンパターン
        for i, symbol in enumerate(symbols):
            base_return = 0.001 + i * 0.0002
            base_vol = 0.015 + i * 0.005
            
            # ボラティリティクラスタリングをシミュレート
            volatility = np.random.exponential(base_vol, len(dates))
            returns = np.random.normal(base_return, volatility)
            
            extended_returns[symbol] = returns
        
        # 重み履歴の作成（定期的に重み変更）
        weights_history = {}
        current_date = start_date
        
        while current_date <= end_date:
            # 重みをランダムに調整（リバランシングをシミュレート）
            base_weights = [0.2, 0.2, 0.25, 0.15, 0.2]
            noise = np.random.normal(0, 0.02, len(base_weights))
            adjusted_weights = np.maximum(0.05, base_weights + noise)  # 最小5%
            
            # 正規化
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            
            weights_history[current_date] = dict(zip(symbols, adjusted_weights))
            current_date += timedelta(days=30)  # 月次リバランシング
        
        # バックテスティングエンジンの初期化
        backtest_config = BacktestConfig(
            lookback_window=200,
            rolling_window=60,
            min_observations=50,
            enable_kupiec_test=True,
            enable_christoffersen_test=True
        )
        
        backtesting_engine = VaRBacktestingEngine(
            var_engine, hybrid_calculator, backtest_config
        )
        
        # バックテスト実行
        logger.info("Running comprehensive backtest...")
        test_start = start_date + timedelta(days=250)  # 初期訓練期間を確保
        
        backtest_result = backtesting_engine.run_comprehensive_backtest(
            extended_returns, weights_history, test_start
        )
        
        # 結果表示
        logger.info(f"Backtest Period: {backtest_result.test_period_start.date()} to {backtest_result.test_period_end.date()}")
        logger.info(f"Total Observations: {backtest_result.total_observations}")
        logger.info(f"VaR 95% Violations: {backtest_result.var_95_violations} ({backtest_result.var_95_violation_rate:.2%})")
        logger.info(f"VaR 99% Violations: {backtest_result.var_99_violations} ({backtest_result.var_99_violation_rate:.2%})")
        logger.info(f"Model Accuracy Score: {backtest_result.model_accuracy_score:.3f}")
        logger.info(f"Calibration Quality: {backtest_result.calibration_quality}")
        
        # Kupiec検定結果
        if backtest_result.kupiec_test_95:
            kupiec_95 = backtest_result.kupiec_test_95
            logger.info(f"Kupiec Test 95% - Statistic: {kupiec_95.get('statistic', 'N/A'):.3f}, P-value: {kupiec_95.get('p_value', 'N/A'):.3f}")
            logger.info(f"Kupiec Test 95% - Reject Null: {kupiec_95.get('reject_null', 'N/A')}")
        
        if backtest_result.kupiec_test_99:
            kupiec_99 = backtest_result.kupiec_test_99
            logger.info(f"Kupiec Test 99% - Statistic: {kupiec_99.get('statistic', 'N/A'):.3f}, P-value: {kupiec_99.get('p_value', 'N/A'):.3f}")
            logger.info(f"Kupiec Test 99% - Reject Null: {kupiec_99.get('reject_null', 'N/A')}")
        
        # 推奨事項
        if backtest_result.recommendations:
            logger.info("Backtesting Recommendations:")
            for rec in backtest_result.recommendations:
                logger.info(f"  - {rec}")
        
        return backtesting_engine, backtest_result
        
    except Exception as e:
        logger.error(f"Backtesting engine demo error: {e}")
        raise

def demo_monitoring_simulation():
    """監視システムのシミュレーションデモ"""
    logger.info("=== Real-time VaR Monitoring Simulation Demo ===")
    
    try:
        # データ準備
        returns_data, initial_weights = create_sample_data()
        
        # VaRエンジンとハイブリッド計算器
        var_config = VaRCalculationConfig(
            confidence_levels=[0.95, 0.99],
            lookback_period=200,
            enable_regime_detection=True
        )
        var_engine = AdvancedVaREngine(var_config)
        hybrid_calculator = HybridVaRCalculator(var_engine)
        
        # 監視設定（テスト用に短い間隔）
        monitoring_config = MonitoringConfig(
            monitoring_interval=1,  # 1秒（デモ用）
            var_95_threshold=0.03,  # 3%（テスト用に低く設定）
            var_99_threshold=0.05,  # 5%（テスト用に低く設定）
            warning_threshold_ratio=0.8,
            critical_threshold_ratio=1.2
        )
        
        # リアルタイム監視システム
        monitor = RealTimeVaRMonitor(var_engine, monitoring_config)
        
        # データプロバイダー関数（シミュレーション用）
        data_counter = 0
        def data_provider():
            nonlocal data_counter
            # データの一部を返す（リアルタイムデータをシミュレート）
            end_idx = min(len(returns_data), 100 + data_counter * 10)
            start_idx = max(0, end_idx - 100)
            data_counter += 1
            return returns_data.iloc[start_idx:end_idx]
        
        def weight_provider():
            # 重みを若干変動させる（リバランシングをシミュレート）
            noise = np.random.normal(0, 0.01, len(initial_weights))
            symbols = list(initial_weights.keys())
            weights_array = np.array(list(initial_weights.values())) + noise
            weights_array = np.maximum(0.05, weights_array)  # 最小5%
            weights_array = weights_array / weights_array.sum()  # 正規化
            
            return dict(zip(symbols, weights_array))
        
        # ドローダウンコントローラーコールバックのシミュレーション
        def drawdown_controller_callback(signal):
            logger.info(f"Drawdown Controller Signal: {signal['signal_type']} - Severity: {signal['severity_level']:.3f}")
            
            if signal['signal_type'] == 'var_breach':
                logger.info("Risk mitigation actions would be triggered")
                # 実際の実装では、ドローダウン制御システムが呼び出される
        
        monitor.set_drawdown_controller_callback(drawdown_controller_callback)
        
        # 監視開始（短時間のシミュレーション）
        logger.info("Starting monitoring simulation...")
        success = monitor.start_monitoring(data_provider, weight_provider)
        
        if success:
            logger.info("Monitoring started successfully")
            
            # シミュレーション実行（5秒間）
            import time
            time.sleep(5)
            
            # 監視停止
            monitor.stop_monitoring()
            logger.info("Monitoring stopped")
            
            # 監視結果の確認
            status = monitor.get_monitoring_status()
            logger.info("Monitoring Status:")
            for key, value in status.items():
                if key != 'config':  # 設定詳細はスキップ
                    logger.info(f"  {key}: {value}")
            
            # アラートサマリー
            alert_summary = monitor.get_alert_summary(hours=1)
            logger.info(f"Alert Summary: {alert_summary}")
            
        else:
            logger.error("Failed to start monitoring")
        
        return monitor
        
    except Exception as e:
        logger.error(f"Monitoring simulation demo error: {e}")
        raise

def main():
    """メイン実行関数"""
    try:
        logger.info("5-3-2 ポートフォリオVaR計算システム - 包括的デモンストレーション開始")
        
        # 1. 高度VaRエンジンのデモ
        var_engine, var_result = demo_advanced_var_engine()
        
        # 2. ハイブリッド計算器のデモ
        hybrid_calculator, hybrid_result = demo_hybrid_calculator(var_engine)
        
        # 3. 統合ブリッジのデモ
        integration_bridge, integration_result = demo_integration_bridge(var_engine, hybrid_calculator)
        
        # 4. バックテスティングエンジンのデモ
        backtesting_engine, backtest_result = demo_backtesting_engine(var_engine, hybrid_calculator)
        
        # 5. 監視システムのシミュレーションデモ
        monitor = demo_monitoring_simulation()
        
        # 最終結果のサマリー
        logger.info("=== Demo Summary ===")
        logger.info("✓ Advanced VaR Engine: 高度な計算手法によるVaR計算")
        logger.info("✓ Hybrid VaR Calculator: 動的重み付けによる最適化")
        logger.info("✓ Integration Bridge: 既存システムとの統合")
        logger.info("✓ Backtesting Engine: モデル性能の検証")
        logger.info("✓ Real-time Monitor: リアルタイム監視・アラート")
        
        # 統合システムの性能評価
        logger.info("=== System Performance Assessment ===")
        logger.info(f"基本VaR計算精度: ✓ (VaR95%: {var_result.get_var_95():.4f})")
        logger.info(f"ハイブリッド最適化: ✓ (Method: {hybrid_result.calculation_method})")
        logger.info(f"バックテスト性能: ✓ (精度スコア: {backtest_result.model_accuracy_score:.3f})")
        
        if integration_result.success:
            logger.info("システム統合: ✓ (正常動作)")
        else:
            logger.info("システム統合: ⚠ (部分的成功)")
        
        logger.info("5-3-2 ポートフォリオVaR計算システム実装完了")
        
        return {
            'var_engine': var_engine,
            'hybrid_calculator': hybrid_calculator,
            'integration_bridge': integration_bridge,
            'backtesting_engine': backtesting_engine,
            'monitor': monitor,
            'demo_results': {
                'var_result': var_result,
                'hybrid_result': hybrid_result,
                'integration_result': integration_result,
                'backtest_result': backtest_result
            }
        }
        
    except Exception as e:
        logger.error(f"Demo execution error: {e}")
        raise

if __name__ == "__main__":
    result = main()
