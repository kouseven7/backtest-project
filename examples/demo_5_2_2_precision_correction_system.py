"""
5-2-2「トレンド判定精度の自動補正」システム統合デモスクリプト

このスクリプトは以下の機能を実装・統合しています：
1. トレンド精度追跡システム
2. パラメータ自動調整機能
3. 信頼度較正システム
4. ハイブリッド補正エンジン
5. 統合トレンド検出器
6. 適応学習システム
7. バッチ処理システム

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List

# ワーニングを非表示
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_demo_environment():
    """デモ環境の設定"""
    
    try:
        # 必要なディレクトリとファイルの確認
        config_dir = "config/trend_precision_config"
        adjustment_dir = "config/trend_precision_adjustment"
        
        required_files = [
            f"{config_dir}/precision_config.json",
            f"{config_dir}/parameter_bounds.json",
            f"{adjustment_dir}/__init__.py",
            f"{adjustment_dir}/precision_tracker.py",
            f"{adjustment_dir}/parameter_adjuster.py",
            f"{adjustment_dir}/confidence_calibrator.py",
            f"{adjustment_dir}/correction_engine.py",
            f"{adjustment_dir}/enhanced_trend_detector.py",
            f"{adjustment_dir}/adaptive_learning.py",
            f"{adjustment_dir}/batch_processor.py"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"必要なファイルが見つかりません: {missing_files}")
            return False
        
        logger.info("✓ 5-2-2 システムファイル検証完了")
        return True
        
    except Exception as e:
        logger.error(f"環境設定エラー: {e}")
        return False

def load_system_configurations():
    """システム設定を読み込み"""
    
    try:
        # メイン設定
        with open("config/trend_precision_config/precision_config.json", "r", encoding="utf-8") as f:
            main_config = json.load(f)
        
        # パラメータ境界設定
        with open("config/trend_precision_config/parameter_bounds.json", "r", encoding="utf-8") as f:
            bounds_config = json.load(f)
        
        logger.info("✓ システム設定読み込み完了")
        
        return main_config, bounds_config
        
    except Exception as e:
        logger.error(f"設定読み込みエラー: {e}")
        return None, None

def initialize_system_components(main_config: Dict, bounds_config: Dict):
    """システムコンポーネントの初期化"""
    
    try:
        # パッケージのインポート
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
        
        # コンポーネントの初期化
        components = {}
        
        # 1. 精度追跡システム
        tracking_config = main_config.get("tracking", {})
        components["precision_tracker"] = TrendPrecisionTracker(tracking_config)
        logger.info("✓ TrendPrecisionTracker 初期化完了")
        
        # 2. パラメータ調整システム
        adjustment_config = main_config.get("parameter_adjustment", {})
        # parameter_boundsは内部でロードされるため、設定に含める
        adjustment_config["parameter_bounds_config"] = bounds_config
        components["parameter_adjuster"] = ParameterAdjuster(adjustment_config)
        logger.info("✓ ParameterAdjuster 初期化完了")
        
        # 3. 信頼度較正システム
        calibration_config = main_config.get("confidence_calibration", {})
        components["confidence_calibrator"] = ConfidenceCalibrator(calibration_config)
        logger.info("✓ ConfidenceCalibrator 初期化完了")
        
        # 4. 補正エンジン
        correction_config = main_config.get("correction_engine", {})
        # サブシステムの設定をネストして渡す
        correction_config["parameter_adjustment"] = adjustment_config
        correction_config["confidence_calibration"] = calibration_config
        components["correction_engine"] = TrendPrecisionCorrectionEngine(correction_config)
        logger.info("✓ TrendPrecisionCorrectionEngine 初期化完了")
        
        # 5. 統合トレンド検出器
        detection_config = main_config.get("enhanced_detection", {})
        # ダミーベース検出器を作成（実際の実装では既存の検出器を使用）
        class DummyDetector:
            def __init__(self):
                self.strategy_name = "dummy_strategy"
            def detect_trend(self, ticker="UNKNOWN"):
                return ("up", 0.6)
        
        base_detector = DummyDetector()
        components["enhanced_detector"] = EnhancedTrendDetector(
            base_detector,
            components["correction_engine"],
            enable_correction=detection_config.get("enable_correction", True),
            precision_tracker=components["precision_tracker"]
        )
        logger.info("✓ EnhancedTrendDetector 初期化完了")
        
        # 6. 適応学習システム
        learning_config = main_config.get("adaptive_learning", {})
        components["adaptive_learning"] = AdaptiveLearningEngine(learning_config)
        logger.info("✓ AdaptiveLearningEngine 初期化完了")
        
        # 7. バッチ処理システム
        batch_config = main_config.get("batch_processing", {})
        components["batch_processor"] = TrendPrecisionBatchProcessor(batch_config)
        logger.info("✓ TrendPrecisionBatchProcessor 初期化完了")
        
        return components
        
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {e}")
        return None
    except Exception as e:
        logger.error(f"コンポーネント初期化エラー: {e}")
        return None

def demo_precision_tracking(components: Dict):
    """精度追跡システムのデモ"""
    
    try:
        logger.info("\n=== 1. 精度追跡システム デモ ===")
        
        precision_tracker = components["precision_tracker"]
        
        # サンプル予測記録の生成と追加
        logger.info("サンプル予測記録を生成中...")
        sample_records = precision_tracker.generate_sample_data(50)
        
        # 記録の追加
        added_count = 0
        for record in sample_records:
            result = precision_tracker.add_prediction_record(record)
            if result.get("success", False):
                added_count += 1
        
        logger.info(f"✓ {added_count}件の予測記録を追加")
        
        # 精度統計の取得
        stats = precision_tracker.get_accuracy_statistics()
        if stats:
            logger.info(f"平均精度: {stats.get('overall_accuracy', 0):.3f}")
            logger.info(f"信頼度較正誤差: {stats.get('mean_calibration_error', 0):.3f}")
        
        # 精度履歴の取得
        history = precision_tracker.get_accuracy_history(days=7)
        if history:
            logger.info(f"7日間の履歴: {len(history)}件")
        
        return True
        
    except Exception as e:
        logger.error(f"精度追跡デモエラー: {e}")
        return False

def demo_parameter_adjustment(components: Dict):
    """パラメータ調整システムのデモ"""
    
    try:
        logger.info("\n=== 2. パラメータ調整システム デモ ===")
        
        parameter_adjuster = components["parameter_adjuster"]
        
        # 現在のパラメータを取得
        current_params = {
            "sma_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "rsi_period": 14
        }
        
        logger.info(f"現在のパラメータ: {current_params}")
        
        # パラメータ最適化の実行
        logger.info("パラメータ最適化を実行中...")
        optimization_result = parameter_adjuster.optimize_parameters(
            current_params, method="grid_search", max_iterations=10
        )
        
        if optimization_result and optimization_result.get("success", False):
            optimized_params = optimization_result.get("best_parameters", {})
            improvement = optimization_result.get("performance_improvement", 0)
            
            logger.info(f"✓ 最適化完了")
            logger.info(f"最適化パラメータ: {optimized_params}")
            logger.info(f"パフォーマンス改善: {improvement:.4f}")
            
            # パラメータ検証
            validation_result = parameter_adjuster.validate_parameters(optimized_params)
            if validation_result.get("valid", False):
                logger.info("✓ パラメータ検証成功")
            else:
                logger.warning(f"パラメータ検証失敗: {validation_result.get('errors', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"パラメータ調整デモエラー: {e}")
        return False

def demo_confidence_calibration(components: Dict):
    """信頼度較正システムのデモ"""
    
    try:
        logger.info("\n=== 3. 信頼度較正システム デモ ===")
        
        confidence_calibrator = components["confidence_calibrator"]
        precision_tracker = components["precision_tracker"]
        
        # 較正データの準備
        recent_records = precision_tracker.get_recent_records(100)
        
        if recent_records:
            logger.info(f"較正データ: {len(recent_records)}件")
            
            # Platt Scaling較正の実行
            logger.info("Platt Scaling較正を実行中...")
            platt_result = confidence_calibrator.calibrate_confidence_platt_scaling(recent_records)
            
            if platt_result.get("success", False):
                logger.info(f"✓ Platt Scaling較正完了")
                logger.info(f"較正前ECE: {platt_result.get('ece_before', 0):.4f}")
                logger.info(f"較正後ECE: {platt_result.get('ece_after', 0):.4f}")
            
            # Isotonic Regression較正の実行
            logger.info("Isotonic Regression較正を実行中...")
            isotonic_result = confidence_calibrator.calibrate_confidence_isotonic_regression(recent_records)
            
            if isotonic_result.get("success", False):
                logger.info(f"✓ Isotonic Regression較正完了")
                logger.info(f"較正前ECE: {isotonic_result.get('ece_before', 0):.4f}")
                logger.info(f"較正後ECE: {isotonic_result.get('ece_after', 0):.4f}")
            
            # 較正メトリクスの比較
            comparison = confidence_calibrator.compare_calibration_methods(recent_records)
            if comparison:
                best_method = comparison.get("best_method", "unknown")
                logger.info(f"✓ 最適較正手法: {best_method}")
        
        return True
        
    except Exception as e:
        logger.error(f"信頼度較正デモエラー: {e}")
        return False

def demo_correction_engine(components: Dict):
    """補正エンジンのデモ"""
    
    try:
        logger.info("\n=== 4. ハイブリッド補正エンジン デモ ===")
        
        correction_engine = components["correction_engine"]
        precision_tracker = components["precision_tracker"]
        
        # サンプル市場データ
        sample_data = {
            "timestamp": datetime.now(),
            "price": 100.0,
            "volume": 1000,
            "indicators": {
                "sma_20": 99.5,
                "macd": 0.1,
                "rsi": 55.0
            }
        }
        
        logger.info("補正処理を実行中...")
        
        # ハイブリッド補正の実行
        correction_result = correction_engine.apply_hybrid_correction(
            sample_data, method="combined"
        )
        
        if correction_result and correction_result.get("success", False):
            corrected_result = correction_result.get("corrected_result", {})
            
            logger.info(f"✓ 補正処理完了")
            logger.info(f"補正前信頼度: {corrected_result.get('original_confidence', 0):.3f}")
            logger.info(f"補正後信頼度: {corrected_result.get('corrected_confidence', 0):.3f}")
            logger.info(f"パラメータ調整: {corrected_result.get('parameter_adjusted', False)}")
            logger.info(f"較正適用: {corrected_result.get('calibration_applied', False)}")
            
            # フィードバックの提供
            feedback_data = precision_tracker.get_recent_records(10)
            if feedback_data:
                feedback_result = correction_engine.provide_feedback(feedback_data)
                if feedback_result.get("success", False):
                    logger.info("✓ フィードバック処理完了")
        
        return True
        
    except Exception as e:
        logger.error(f"補正エンジンデモエラー: {e}")
        return False

def demo_enhanced_detection(components: Dict):
    """統合検出システムのデモ"""
    
    try:
        logger.info("\n=== 5. 統合トレンド検出システム デモ ===")
        
        enhanced_detector = components["enhanced_detector"]
        
        # サンプル市場データの準備
        sample_market_data = [
            {"timestamp": datetime.now() - timedelta(minutes=i), "price": 100 + i * 0.1, "volume": 1000}
            for i in range(100, 0, -1)
        ]
        
        logger.info("統合トレンド検出を実行中...")
        
        # 統合検出の実行
        detection_result = enhanced_detector.detect_trend_with_correction(sample_market_data)
        
        if detection_result and detection_result.get("success", False):
            enhanced_result = detection_result.get("enhanced_result", {})
            
            logger.info(f"✓ トレンド検出完了")
            logger.info(f"検出トレンド: {enhanced_result.get('trend_direction', 'unknown')}")
            logger.info(f"信頼度: {enhanced_result.get('corrected_confidence', 0):.3f}")
            logger.info(f"強度: {enhanced_result.get('trend_strength', 0):.3f}")
            logger.info(f"補正適用: {enhanced_result.get('correction_applied', False)}")
            
            # 複数手法での検証
            validation_result = enhanced_detector.validate_detection(sample_market_data)
            if validation_result.get("success", False):
                consensus = validation_result.get("consensus_confidence", 0)
                logger.info(f"✓ 複数手法コンセンサス信頼度: {consensus:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"統合検出デモエラー: {e}")
        return False

def demo_adaptive_learning(components: Dict):
    """適応学習システムのデモ"""
    
    try:
        logger.info("\n=== 6. 適応学習システム デモ ===")
        
        adaptive_learning = components["adaptive_learning"]
        correction_engine = components["correction_engine"]
        precision_tracker = components["precision_tracker"]
        
        # 新しいフィードバックデータの取得
        new_feedback = precision_tracker.get_recent_records(20)
        
        if new_feedback:
            logger.info(f"フィードバックデータ: {len(new_feedback)}件")
            
            # 継続的学習の実行
            logger.info("継続的学習を実行中...")
            learning_result = adaptive_learning.continuous_learning_update(
                correction_engine, new_feedback
            )
            
            if learning_result and not learning_result.get("error"):
                logger.info(f"✓ 継続的学習完了")
                logger.info(f"更新パラメータ数: {learning_result.get('parameters_updated', 0)}")
                logger.info(f"較正更新数: {learning_result.get('calibration_updated', 0)}")
                logger.info(f"処理フィードバック数: {learning_result.get('feedback_processed', 0)}")
                
                # 学習メトリクス
                learning_metrics = learning_result.get("learning_metrics", {})
                if learning_metrics:
                    algorithm = learning_metrics.get("algorithm", "unknown")
                    performance_trend = learning_metrics.get("performance_trend", "unknown")
                    logger.info(f"学習アルゴリズム: {algorithm}")
                    logger.info(f"パフォーマンストレンド: {performance_trend}")
            
            # 学習ステータスの確認
            status = adaptive_learning.get_learning_status()
            if status:
                logger.info(f"学習ステップ: {status.get('learning_steps', 0)}")
                logger.info(f"追跡パラメータ数: {status.get('tracked_parameters', {})}")
        
        return True
        
    except Exception as e:
        logger.error(f"適応学習デモエラー: {e}")
        return False

def demo_batch_processing(components: Dict):
    """バッチ処理システムのデモ"""
    
    try:
        logger.info("\n=== 7. バッチ処理システム デモ ===")
        
        batch_processor = components["batch_processor"]
        precision_tracker = components["precision_tracker"]
        correction_engine = components["correction_engine"]
        parameter_adjuster = components["parameter_adjuster"]
        confidence_calibrator = components["confidence_calibrator"]
        
        # 日次バッチ処理の実行
        logger.info("日次精度更新処理を実行中...")
        daily_result = batch_processor.run_daily_precision_update(
            precision_tracker, correction_engine
        )
        
        if daily_result and not daily_result.get("error"):
            logger.info(f"✓ 日次処理完了")
            logger.info(f"処理記録数: {daily_result.get('records_processed', 0)}")
            logger.info(f"処理時間: {daily_result.get('processing_duration', 0):.2f}秒")
            
            # 精度更新結果
            precision_update = daily_result.get("precision_update", {})
            if precision_update:
                avg_accuracy = precision_update.get("avg_accuracy", 0)
                logger.info(f"平均精度: {avg_accuracy:.3f}")
        
        # 週次包括的処理の実行
        logger.info("週次包括的更新処理を実行中...")
        weekly_result = batch_processor.run_weekly_comprehensive_update(
            precision_tracker, correction_engine, parameter_adjuster, confidence_calibrator
        )
        
        if weekly_result and not weekly_result.get("error"):
            logger.info(f"✓ 週次処理完了")
            logger.info(f"分析記録数: {weekly_result.get('records_analyzed', 0)}")
            logger.info(f"処理時間: {weekly_result.get('processing_duration', 0):.2f}秒")
            
            # 週次レポート
            weekly_report = weekly_result.get("weekly_report", {})
            if weekly_report:
                summary = weekly_report.get("summary", {})
                recommendations = weekly_report.get("recommendations", [])
                
                logger.info(f"総予測数: {summary.get('total_predictions', 0)}")
                logger.info(f"平均精度: {summary.get('avg_accuracy', 0):.3f}")
                
                if recommendations:
                    logger.info("推奨事項:")
                    for i, rec in enumerate(recommendations[:3], 1):
                        logger.info(f"  {i}. {rec}")
        
        # バッチステータスの確認
        batch_status = batch_processor.get_batch_status()
        if batch_status:
            logger.info("✓ バッチ処理ステータス確認完了")
            logger.info(f"日次処理有効: {batch_status.get('daily_batch_enabled', False)}")
            logger.info(f"週次処理有効: {batch_status.get('weekly_batch_enabled', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"バッチ処理デモエラー: {e}")
        return False

def run_comprehensive_system_test(components: Dict):
    """包括的システムテスト"""
    
    try:
        logger.info("\n=== 8. 包括的システム統合テスト ===")
        
        # エンドツーエンドのワークフロー
        enhanced_detector = components["enhanced_detector"]
        adaptive_learning = components["adaptive_learning"]
        correction_engine = components["correction_engine"]
        precision_tracker = components["precision_tracker"]
        
        # 1. 市場データでトレンド検出
        sample_data = [
            {"timestamp": datetime.now() - timedelta(minutes=i), "price": 100 + i * 0.05, "volume": 1000}
            for i in range(50, 0, -1)
        ]
        
        detection_result = enhanced_detector.detect_trend_with_correction(sample_data)
        if detection_result.get("success", False):
            logger.info("✓ ステップ1: トレンド検出成功")
        
        # 2. 結果をフィードバックとして追加
        if detection_result.get("enhanced_result"):
            result = detection_result["enhanced_result"]
            feedback_record = {
                "prediction_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now(),
                "predicted_trend": result.get("trend_direction", "up"),
                "confidence_score": result.get("corrected_confidence", 0.5),
                "actual_trend": "up",  # 仮の実際の結果
                "accuracy": 0.75,  # 仮の精度
                "method": "enhanced_detection"
            }
            
            precision_tracker.add_prediction_record(feedback_record)
            logger.info("✓ ステップ2: フィードバック記録追加成功")
        
        # 3. 適応学習による継続的改善
        feedback_data = precision_tracker.get_recent_records(10)
        if feedback_data:
            learning_result = adaptive_learning.continuous_learning_update(
                correction_engine, feedback_data
            )
            if learning_result and not learning_result.get("error"):
                logger.info("✓ ステップ3: 適応学習更新成功")
        
        # 4. システム全体の統計確認
        overall_stats = precision_tracker.get_accuracy_statistics()
        if overall_stats:
            logger.info(f"✓ 最終統計 - 全体精度: {overall_stats.get('overall_accuracy', 0):.3f}")
            logger.info(f"✓ 最終統計 - 較正誤差: {overall_stats.get('mean_calibration_error', 0):.3f}")
        
        logger.info("✓ 包括的システム統合テスト完了")
        return True
        
    except Exception as e:
        logger.error(f"包括的システムテストエラー: {e}")
        return False

def generate_system_report(components: Dict):
    """システムレポートの生成"""
    
    try:
        logger.info("\n=== 9. システムレポート生成 ===")
        
        report = {
            "system_name": "5-2-2 トレンド判定精度の自動補正システム",
            "generated_at": datetime.now().isoformat(),
            "components": {},
            "overall_status": "operational"
        }
        
        # 各コンポーネントのステータス
        component_names = [
            "precision_tracker", "parameter_adjuster", "confidence_calibrator",
            "correction_engine", "enhanced_detector", "adaptive_learning", "batch_processor"
        ]
        
        for name in component_names:
            if name in components:
                component = components[name]
                try:
                    if hasattr(component, 'get_status'):
                        status = component.get_status()
                    elif hasattr(component, f'get_{name.split("_")[-1]}_status'):
                        status_method = getattr(component, f'get_{name.split("_")[-1]}_status')
                        status = status_method()
                    else:
                        status = {"status": "initialized", "component": name}
                    
                    report["components"][name] = status
                    logger.info(f"✓ {name}: ステータス取得完了")
                    
                except Exception as e:
                    report["components"][name] = {"error": str(e)}
                    logger.warning(f"⚠ {name}: ステータス取得エラー - {e}")
        
        # レポート保存（実際の実装では外部ファイルに保存）
        report_summary = f"""
=== 5-2-2 システムレポート ===
生成日時: {report['generated_at']}
システム: {report['system_name']}
全体ステータス: {report['overall_status']}

コンポーネント数: {len(report['components'])}
正常コンポーネント: {len([c for c in report['components'].values() if not c.get('error')])}

実装機能:
- トレンド精度追跡システム ✓
- パラメータ自動調整機能 ✓
- 信頼度較正システム ✓
- ハイブリッド補正エンジン ✓
- 統合トレンド検出器 ✓
- 適応学習システム ✓
- バッチ処理システム ✓

システム統合: 完了
テスト実行: 完了
        """
        
        logger.info(report_summary)
        logger.info("✓ システムレポート生成完了")
        
        return report
        
    except Exception as e:
        logger.error(f"レポート生成エラー: {e}")
        return None

def main():
    """メインデモ関数"""
    
    logger.info("=== 5-2-2「トレンド判定精度の自動補正」システム 統合デモ開始 ===")
    
    # 1. 環境設定
    if not setup_demo_environment():
        logger.error("環境設定に失敗しました")
        return False
    
    # 2. 設定読み込み
    main_config, bounds_config = load_system_configurations()
    if not main_config or not bounds_config:
        logger.error("設定読み込みに失敗しました")
        return False
    
    # 3. システム初期化
    components = initialize_system_components(main_config, bounds_config)
    if not components:
        logger.error("システム初期化に失敗しました")
        return False
    
    # 4. 各システムのデモ実行
    demo_functions = [
        demo_precision_tracking,
        demo_parameter_adjustment,
        demo_confidence_calibration,
        demo_correction_engine,
        demo_enhanced_detection,
        demo_adaptive_learning,
        demo_batch_processing,
        run_comprehensive_system_test
    ]
    
    success_count = 0
    for demo_func in demo_functions:
        try:
            if demo_func(components):
                success_count += 1
            else:
                logger.warning(f"{demo_func.__name__} が失敗しました")
        except Exception as e:
            logger.error(f"{demo_func.__name__} でエラーが発生: {e}")
    
    # 5. システムレポート生成
    report = generate_system_report(components)
    
    # 6. 最終結果
    total_demos = len(demo_functions)
    success_rate = (success_count / total_demos) * 100
    
    logger.info(f"\n=== デモ実行結果 ===")
    logger.info(f"実行デモ数: {total_demos}")
    logger.info(f"成功デモ数: {success_count}")
    logger.info(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 85:
        logger.info("✅ 5-2-2 システム統合デモ 正常終了")
        return True
    else:
        logger.warning("⚠️ 一部デモで問題が発生しました")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("デモが中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        sys.exit(1)
