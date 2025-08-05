"""
5-2-2 システム簡易検証スクリプト

各コンポーネントの初期化をステップバイステップでテストします
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_individual_components():
    """個別コンポーネントのテスト"""
    
    try:
        # 1. パッケージのインポートテスト
        logger.info("=== パッケージインポートテスト ===")
        sys.path.append("config")
        
        from trend_precision_adjustment import (
            TrendPrecisionTracker,
            ParameterAdjuster,
            ConfidenceCalibrator,
            TrendPrecisionCorrectionEngine,
            EnhancedTrendDetector,
            AdaptiveLearningEngine,
            TrendPrecisionBatchProcessor
        )
        
        logger.info("✓ すべてのモジュールをインポート成功")
        
        # 2. 設定の読み込み
        logger.info("=== 設定読み込みテスト ===")
        with open("config/trend_precision_config/precision_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("✓ メイン設定読み込み成功")
        
        # 3. 個別初期化テスト
        logger.info("=== 個別初期化テスト ===")
        
        # TrendPrecisionTracker
        tracker_config = config.get("tracking", {})
        tracker = TrendPrecisionTracker(tracker_config)
        logger.info("✓ TrendPrecisionTracker OK")
        
        # ParameterAdjuster
        param_config = config.get("parameter_adjustment", {})
        parameter_adjuster = ParameterAdjuster(param_config)
        logger.info("✓ ParameterAdjuster OK")
        
        # ConfidenceCalibrator
        calib_config = config.get("confidence_calibration", {})
        confidence_calibrator = ConfidenceCalibrator(calib_config)
        logger.info("✓ ConfidenceCalibrator OK")
        
        # TrendPrecisionCorrectionEngine
        correction_config = config.get("correction_engine", {})
        correction_engine = TrendPrecisionCorrectionEngine(correction_config)
        logger.info("✓ TrendPrecisionCorrectionEngine OK")
        
        # AdaptiveLearningEngine
        learning_config = config.get("adaptive_learning", {})
        adaptive_learning = AdaptiveLearningEngine(learning_config)
        logger.info("✓ AdaptiveLearningEngine OK")
        
        # TrendPrecisionBatchProcessor
        batch_config = config.get("batch_processing", {})
        batch_processor = TrendPrecisionBatchProcessor(batch_config)
        logger.info("✓ TrendPrecisionBatchProcessor OK")
        
        # EnhancedTrendDetector（ダミー検出器を使用）
        class DummyDetector:
            def __init__(self):
                self.strategy_name = "dummy_strategy"
            def detect_trend(self, ticker="UNKNOWN"):
                return ("up", 0.6)
        
        base_detector = DummyDetector()
        enhanced_detector = EnhancedTrendDetector(
            base_detector,
            correction_engine,
            enable_correction=True,
            precision_tracker=tracker
        )
        logger.info("✓ EnhancedTrendDetector OK")
        
        logger.info("=== 全コンポーネント初期化成功 ===")
        
        # 4. 簡易機能テスト
        logger.info("=== 簡易機能テスト ===")
        
        # サンプルデータ生成
        sample_records = tracker.generate_sample_data(10)
        logger.info(f"✓ サンプルデータ生成: {len(sample_records)}件")
        
        # 精度統計
        summary = tracker.get_recent_performance_summary(7)
        logger.info(f"✓ パフォーマンスサマリー取得: {type(summary)}")
        
        # 学習ステータス
        learning_status = adaptive_learning.get_learning_status()
        logger.info(f"✓ 学習ステータス: {learning_status.get('algorithm', 'unknown')}")
        
        # バッチステータス
        batch_status = batch_processor.get_batch_status()
        logger.info(f"✓ バッチステータス: 日次={batch_status.get('daily_batch_enabled', False)}")
        
        logger.info("=== 5-2-2 システム検証完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_individual_components()
    if success:
        print("\n🎉 5-2-2 システム検証成功！")
        print("すべてのコンポーネントが正常に動作しています。")
    else:
        print("\n❌ 5-2-2 システム検証失敗")
        print("一部のコンポーネントに問題があります。")
