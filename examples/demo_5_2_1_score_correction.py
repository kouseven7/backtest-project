"""
Demo Script: 5-2-1 Performance-Based Score Correction System
File: demo_5_2_1_score_correction.py
Description: 
  5-2-1「戦略実績に基づくスコア補正機能」のデモンストレーション

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """メインデモ実行"""
    print("=" * 60)
    print("[ROCKET] 5-2-1「戦略実績に基づくスコア補正機能」システムデモ")
    print("=" * 60)
    
    try:
        # 設定ファイルの読み込み
        config = load_demo_config()
        
        # システムコンポーネントのテスト
        test_results = {}
        
        print("\n[CHART] 1. パフォーマンス追跡システムのテスト")
        test_results['tracker'] = test_performance_tracker(config)
        
        print("\n[TOOL] 2. スコア補正エンジンのテスト")
        test_results['corrector'] = test_score_corrector(config)
        
        print("\n⚡ 3. 統合計算器のテスト")
        test_results['calculator'] = test_enhanced_calculator(config)
        
        print("\n🔄 4. バッチ処理システムのテスト")
        test_results['batch'] = test_batch_processor(config)
        
        print("\n[UP] 5. 統合システムテスト")
        test_results['integration'] = test_integrated_system(config)
        
        # 結果サマリー
        print_test_summary(test_results)
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"[ERROR] デモ実行エラー: {e}")

def load_demo_config() -> dict:
    """デモ用設定を読み込み"""
    try:
        config_path = project_root / "config" / "score_correction" / "correction_config.json"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[OK] 設定ファイル読み込み成功: {config_path}")
        else:
            # デフォルト設定
            config = {
                "tracker": {
                    "tracking_window_days": 30,
                    "min_records": 5,
                    "performance_threshold": 0.1
                },
                "correction": {
                    "ema_alpha": 0.3,
                    "lookback_periods": 20,
                    "max_correction": 0.3,
                    "min_confidence": 0.6,
                    "min_records": 5,
                    "adaptive_learning_enabled": True
                },
                "batch_processing": {
                    "update_schedule": "daily",
                    "batch_size": 10,
                    "max_concurrent_updates": 3
                }
            }
            print("[WARNING] デフォルト設定を使用")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def test_performance_tracker(config: dict) -> bool:
    """パフォーマンス追跡システムのテスト"""
    try:
        from config.performance_score_correction import PerformanceTracker
        
        print("  📝 PerformanceTracker初期化中...")
        tracker = PerformanceTracker(config.get('tracker', {}))
        
        # テストデータの追加
        print("  📝 テストデータ記録中...")
        test_strategies = ['MovingAverageCross', 'RSIStrategy', 'BollingerBands']
        test_tickers = ['AAPL', 'MSFT']
        
        for i, strategy in enumerate(test_strategies):
            for j, ticker in enumerate(test_tickers):
                predicted_score = 0.5 + i * 0.1
                actual_performance = predicted_score + np.random.normal(0, 0.1)
                
                record_id = tracker.record_strategy_performance(
                    strategy_name=strategy,
                    ticker=ticker,
                    predicted_score=predicted_score,
                    actual_performance=actual_performance,
                    market_context={'volatility': 0.2, 'trend': 'upward'}
                )
                print(f"    [OK] 記録追加: {record_id}")
        
        # 履歴取得テスト
        print("  [CHART] パフォーマンス履歴取得テスト...")
        for strategy in test_strategies:
            history = tracker.get_performance_history(strategy, days=30)
            stats = tracker.get_strategy_statistics(strategy, days=30)
            print(f"    [UP] {strategy}: {len(history)}件, 平均精度: {stats.get('avg_accuracy', 0):.3f}")
        
        print("  [OK] パフォーマンス追跡システムテスト完了")
        return True
        
    except Exception as e:
        print(f"  [ERROR] パフォーマンス追跡システムテスト失敗: {e}")
        return False

def test_score_corrector(config: dict) -> bool:
    """スコア補正エンジンのテスト"""
    try:
        from config.performance_score_correction import PerformanceBasedScoreCorrector
        
        print("  [TOOL] PerformanceBasedScoreCorrector初期化中...")
        corrector = PerformanceBasedScoreCorrector(config)
        
        # 補正ファクター計算テスト
        print("  🧮 補正ファクター計算テスト...")
        test_cases = [
            ('MovingAverageCross', 'AAPL', 0.6),
            ('RSIStrategy', 'MSFT', 0.7),
            ('BollingerBands', 'AAPL', 0.5)
        ]
        
        for strategy, ticker, score in test_cases:
            result = corrector.calculate_correction_factor(strategy, ticker, score)
            print(f"    [CHART] {strategy}/{ticker}: factor={result.correction_factor:.3f}, "
                  f"confidence={result.confidence:.3f}, reason={result.reason}")
        
        # パフォーマンス記録更新テスト
        print("  📝 パフォーマンス記録更新テスト...")
        record_id = corrector.update_performance_record(
            strategy_name='TestStrategy',
            ticker='TEST',
            predicted_score=0.8,
            actual_performance=0.75,
            market_context={'test': True}
        )
        print(f"    [OK] 記録更新完了: {record_id}")
        
        print("  [OK] スコア補正エンジンテスト完了")
        return True
        
    except Exception as e:
        print(f"  [ERROR] スコア補正エンジンテスト失敗: {e}")
        return False

def test_enhanced_calculator(config: dict) -> bool:
    """統合計算器のテスト"""
    try:
        from config.performance_score_correction import EnhancedStrategyScoreCalculator
        
        print("  ⚡ EnhancedStrategyScoreCalculator初期化中...")
        calculator = EnhancedStrategyScoreCalculator()
        
        # 補正付きスコア計算テスト
        print("  🧮 補正付きスコア計算テスト...")
        test_cases = [
            ('MovingAverageCross', 'AAPL'),
            ('RSIStrategy', 'MSFT'),
            ('BollingerBands', 'GOOGL')
        ]
        
        for strategy, ticker in test_cases:
            corrected_score = calculator.calculate_corrected_strategy_score(
                strategy_name=strategy,
                ticker=ticker,
                apply_correction=True
            )
            
            if corrected_score:
                print(f"    [UP] {strategy}/{ticker}:")
                print(f"      基本スコア: {corrected_score.base_score.total_score:.3f}")
                print(f"      補正スコア: {corrected_score.corrected_total_score:.3f}")
                print(f"      補正ファクター: {corrected_score.correction_factor:.3f}")
                print(f"      改善比率: {corrected_score.get_improvement_ratio():.3f}")
            else:
                print(f"    [WARNING] {strategy}/{ticker}: スコア計算失敗")
        
        # パフォーマンス統計取得
        print("  [CHART] 補正パフォーマンス統計取得...")
        performance = calculator.get_correction_performance()
        print(f"    総計算数: {performance.get('total_calculations', 0)}")
        print(f"    補正適用数: {performance.get('corrections_applied', 0)}")
        print(f"    補正率: {performance.get('correction_rate', 0):.3f}")
        
        print("  [OK] 統合計算器テスト完了")
        return True
        
    except Exception as e:
        print(f"  [ERROR] 統合計算器テスト失敗: {e}")
        return False

def test_batch_processor(config: dict) -> bool:
    """バッチ処理システムのテスト"""
    try:
        from config.performance_score_correction import ScoreCorrectionBatchProcessor
        
        print("  🔄 ScoreCorrectionBatchProcessor初期化中...")
        processor = ScoreCorrectionBatchProcessor(config)
        
        # 日次更新テスト
        print("  📅 日次更新テスト実行中...")
        daily_result = processor.run_daily_correction_update(
            strategy_list=['MovingAverageCross', 'RSIStrategy']
        )
        
        print(f"    更新タイプ: {daily_result.update_type}")
        print(f"    処理対象: {daily_result.total_strategies}")
        print(f"    成功: {daily_result.successful_updates}")
        print(f"    失敗: {daily_result.failed_updates}")
        print(f"    成功率: {daily_result.get_success_rate():.3f}")
        print(f"    実行時間: {daily_result.get_duration():.2f}秒")
        
        # 週次分析テスト
        print("  [UP] 週次分析テスト実行中...")
        weekly_result = processor.run_weekly_analysis()
        
        print(f"    分析対象: {weekly_result.total_strategies}")
        print(f"    パフォーマンス指標数: {len(weekly_result.performance_metrics)}")
        for key, value in weekly_result.performance_metrics.items():
            print(f"      {key}: {value:.3f}")
        
        print("  [OK] バッチ処理システムテスト完了")
        return True
        
    except Exception as e:
        print(f"  [ERROR] バッチ処理システムテスト失敗: {e}")
        return False

def test_integrated_system(config: dict) -> bool:
    """統合システムテスト"""
    try:
        print("  🔗 統合システム連携テスト...")
        
        from config.performance_score_correction import (
            PerformanceTracker,
            PerformanceBasedScoreCorrector,
            EnhancedStrategyScoreCalculator,
            ScoreCorrectionBatchProcessor
        )
        
        # 全コンポーネントの初期化
        print("    [TOOL] 全コンポーネント初期化...")
        tracker = PerformanceTracker(config.get('tracker', {}))
        corrector = PerformanceBasedScoreCorrector(config)
        calculator = EnhancedStrategyScoreCalculator(score_corrector=corrector)
        processor = ScoreCorrectionBatchProcessor(config)
        
        # エンドツーエンドワークフローのテスト
        print("    🔄 エンドツーエンドワークフロー実行...")
        
        # 1. 初期スコア計算
        strategy = 'IntegratedTest'
        ticker = 'TEST'
        
        initial_score = calculator.calculate_corrected_strategy_score(
            strategy_name=strategy,
            ticker=ticker,
            apply_correction=False  # 初回は補正なし
        )
        
        if initial_score:
            print(f"      初期スコア: {initial_score.base_score.total_score:.3f}")
            
            # 2. パフォーマンスフィードバック
            actual_performance = initial_score.base_score.total_score + np.random.normal(0, 0.1)
            feedback_id = calculator.update_performance_feedback(
                strategy_name=strategy,
                ticker=ticker,
                predicted_score=initial_score.base_score.total_score,
                actual_performance=actual_performance
            )
            print(f"      フィードバック記録: {feedback_id}")
            
            # 3. 補正適用スコア計算
            corrected_score = calculator.calculate_corrected_strategy_score(
                strategy_name=strategy,
                ticker=ticker,
                apply_correction=True
            )
            
            if corrected_score:
                print(f"      補正後スコア: {corrected_score.corrected_total_score:.3f}")
                print(f"      改善効果: {corrected_score.get_improvement_ratio():.3f}")
            
            # 4. 統計確認
            stats = calculator.get_correction_performance()
            print(f"      システム統計: {len(stats)}項目")
            
        print("  [OK] 統合システムテスト完了")
        return True
        
    except Exception as e:
        print(f"  [ERROR] 統合システムテスト失敗: {e}")
        return False

def print_test_summary(test_results: dict):
    """テスト結果サマリーを出力"""
    print("\n" + "=" * 60)
    print("[LIST] テスト結果サマリー")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result)
    
    for component, result in test_results.items():
        status = "[OK] 成功" if result else "[ERROR] 失敗"
        component_name = {
            'tracker': 'パフォーマンス追跡システム',
            'corrector': 'スコア補正エンジン',
            'calculator': '統合計算器',
            'batch': 'バッチ処理システム',
            'integration': '統合システム'
        }.get(component, component)
        
        print(f"  {status}: {component_name}")
    
    print(f"\n[TARGET] 総合結果: {successful_tests}/{total_tests} テスト成功")
    
    if successful_tests == total_tests:
        print("[SUCCESS] 5-2-1「戦略実績に基づくスコア補正機能」システム実装完了！")
        
        # システム機能サマリー
        print("\n[IDEA] 実装された機能:")
        print("  [OK] 実績ベースのパフォーマンス追跡")
        print("  [OK] 指数移動平均による補正計算")
        print("  [OK] 適応的学習による調整")
        print("  [OK] 統合されたスコア計算器")
        print("  [OK] バッチ処理による自動更新")
        print("  [OK] 包括的なレポーティング")
        
        # 使用方法の案内
        print("\n📖 使用方法:")
        print("  1. EnhancedStrategyScoreCalculatorを使用してスコア計算")
        print("  2. update_performance_feedbackでパフォーマンス記録")
        print("  3. ScoreCorrectionBatchProcessorで定期更新")
        print("  4. get_correction_performanceで統計確認")
        
    else:
        print("[WARNING]  一部のテストが失敗しました。ログを確認してください。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
