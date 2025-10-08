"""
2-2-2「トレンド移行期の特別処理ルール」
統合テストスクリプト

Module: Trend Transition Processing Rules Test
Description: 
  トレンド移行期検出と特別処理ルールの統合テスト
  段階的テストによる動作確認

Author: imega
Created: 2025-07-13
Modified: 2025-07-13
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trend_transition_system():
    """2-2-2 トレンド移行期特別処理ルール - 統合テスト"""
    
    print("[ROCKET] 2-2-2「トレンド移行期の特別処理ルール」統合テスト開始")
    print("=" * 80)
    
    test_results = {
        'module_imports': False,
        'data_creation': False,
        'transition_detection': False,
        'rule_management': False,
        'position_adjustment': False,
        'integration_test': False,
        'performance_test': False,
        'error_handling': False
    }
    
    try:
        # テスト1: モジュールインポート
        print("\n📦 テスト1: モジュールインポート")
        try:
            from indicators.trend_transition_detector import (
                TrendTransitionDetector, 
                TransitionDetectionResult,
                detect_trend_transition
            )
            from config.trend_transition_manager import (
                TrendTransitionManager,
                TransitionRule,
                PositionAdjustment,
                TransitionManagementResult,
                manage_trend_transition
            )
            print("✓ 全モジュールのインポートに成功")
            test_results['module_imports'] = True
        except Exception as e:
            print(f"✗ インポートエラー: {e}")
            return test_results
        
        # テスト2: テストデータ作成
        print("\n[CHART] テスト2: テストデータ作成")
        try:
            # 通常トレンドデータ
            normal_data = create_normal_trend_data()
            
            # 移行期データ（高ボラティリティ）
            transition_data = create_transition_period_data()
            
            # ポジションデータ
            sample_positions = {
                'AAPL': 100.0,
                'GOOGL': 75.0,
                'MSFT': 50.0
            }
            
            print(f"✓ 通常データ: {len(normal_data)}行")
            print(f"✓ 移行期データ: {len(transition_data)}行")
            print(f"✓ サンプルポジション: {len(sample_positions)}銘柄")
            test_results['data_creation'] = True
        except Exception as e:
            print(f"✗ データ作成エラー: {e}")
            return test_results
        
        # テスト3: 移行期検出テスト
        print("\n[SEARCH] テスト3: 移行期検出テスト")
        try:
            detector = TrendTransitionDetector(detection_sensitivity="medium")
            
            # 通常期検出
            normal_result = detector.detect_transition(normal_data, "TestStrategy")
            print(f"✓ 通常期検出: {normal_result.transition_type}")
            print(f"  - 移行期判定: {normal_result.is_transition_period}")
            print(f"  - リスクレベル: {normal_result.risk_level}")
            print(f"  - 信頼度: {normal_result.confidence_score:.3f}")
            
            # 移行期検出
            transition_result = detector.detect_transition(transition_data, "TestStrategy")
            print(f"✓ 移行期検出: {transition_result.transition_type}")
            print(f"  - 移行期判定: {transition_result.is_transition_period}")
            print(f"  - リスクレベル: {transition_result.risk_level}")
            print(f"  - 信頼度: {transition_result.confidence_score:.3f}")
            print(f"  - 使用指標: {transition_result.indicators_used}")
            
            test_results['transition_detection'] = True
        except Exception as e:
            print(f"✗ 移行期検出エラー: {e}")
            return test_results
        
        # テスト4: ルール管理テスト
        print("\n[LIST] テスト4: ルール管理テスト")
        try:
            manager = TrendTransitionManager(detection_sensitivity="medium")
            
            # 通常期管理
            normal_management = manager.manage_transition(
                normal_data, "TestStrategy", sample_positions
            )
            print(f"✓ 通常期管理:")
            print(f"  - エントリー許可: {normal_management.entry_allowed}")
            print(f"  - 制限数: {len(normal_management.entry_restrictions)}")
            print(f"  - ポジション調整数: {len(normal_management.position_adjustments)}")
            
            # 移行期管理
            transition_management = manager.manage_transition(
                transition_data, "TestStrategy", sample_positions
            )
            print(f"✓ 移行期管理:")
            print(f"  - エントリー許可: {transition_management.entry_allowed}")
            print(f"  - 制限理由: {transition_management.entry_restrictions}")
            print(f"  - ポジション調整数: {len(transition_management.position_adjustments)}")
            print(f"  - アクティブルール: {transition_management.active_rules}")
            
            test_results['rule_management'] = True
        except Exception as e:
            print(f"✗ ルール管理エラー: {e}")
            return test_results
        
        # テスト5: ポジション調整テスト
        print("\n⚖️ テスト5: ポジション調整テスト")
        try:
            if transition_management.position_adjustments:
                for adjustment in transition_management.position_adjustments:
                    print(f"✓ ポジション調整: {adjustment.strategy_name}")
                    print(f"  - 現在サイズ: {adjustment.current_position_size}")
                    print(f"  - 推奨サイズ: {adjustment.recommended_size}")
                    print(f"  - 調整率: {adjustment.adjustment_ratio:.1%}")
                    print(f"  - 緊急度: {adjustment.urgency}")
                    print(f"  - 理由: {adjustment.reason}")
            else:
                print("✓ ポジション調整不要（正常動作）")
            
            test_results['position_adjustment'] = True
        except Exception as e:
            print(f"✗ ポジション調整エラー: {e}")
            return test_results
        
        # テスト6: 統合動作テスト
        print("\n🔗 テスト6: 統合動作テスト")
        try:
            # 複数戦略での統合テスト
            strategies = ["VWAP_Bounce", "VWAP_Breakout", "Momentum", "Breakout"]
            integration_results = {}
            
            for strategy in strategies:
                result = manage_trend_transition(
                    transition_data, strategy, sample_positions
                )
                integration_results[strategy] = {
                    'entry_allowed': result.entry_allowed,
                    'adjustments': len(result.position_adjustments),
                    'confidence_adj': result.confidence_adjustment
                }
            
            print("✓ 複数戦略統合テスト:")
            for strategy, result in integration_results.items():
                print(f"  - {strategy}: エントリー={result['entry_allowed']}, "
                      f"調整={result['adjustments']}, 信頼度調整={result['confidence_adj']:.3f}")
            
            test_results['integration_test'] = True
        except Exception as e:
            print(f"✗ 統合動作エラー: {e}")
            return test_results
        
        # テスト7: パフォーマンステスト
        print("\n⚡ テスト7: パフォーマンステスト")
        try:
            import time
            
            start_time = time.time()
            for _ in range(10):
                result = manage_trend_transition(transition_data, "TestStrategy", sample_positions)
            execution_time = time.time() - start_time
            
            print(f"✓ パフォーマンステスト: 10回実行 {execution_time:.3f}秒")
            print(f"  - 平均実行時間: {execution_time/10:.3f}秒/回")
            
            if execution_time < 1.0:
                print("✓ パフォーマンス良好")
                test_results['performance_test'] = True
            else:
                print("[WARNING] パフォーマンス要改善")
                test_results['performance_test'] = False
                
        except Exception as e:
            print(f"✗ パフォーマンステストエラー: {e}")
            return test_results
        
        # テスト8: エラーハンドリングテスト
        print("\n🛡️ テスト8: エラーハンドリングテスト")
        try:
            # 不正データでのテスト
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
            error_result = manage_trend_transition(invalid_data, "TestStrategy")
            
            print(f"✓ 不正データ処理: {error_result.transition_detection.transition_type}")
            
            # 空データでのテスト
            empty_data = pd.DataFrame()
            empty_result = manage_trend_transition(empty_data, "TestStrategy")
            
            print(f"✓ 空データ処理: {empty_result.transition_detection.transition_type}")
            
            test_results['error_handling'] = True
        except Exception as e:
            print(f"✗ エラーハンドリングテストエラー: {e}")
            return test_results
        
        # 最終結果
        print("\n" + "=" * 80)
        print("[CHART] テスト結果サマリー")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "[OK] PASS" if result else "[ERROR] FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n[TARGET] 総合結果: {passed_tests}/{total_tests} テストパス ({passed_tests/total_tests*100:.0f}%)")
        
        if passed_tests == total_tests:
            print("[SUCCESS] 2-2-2「トレンド移行期の特別処理ルール」実装完了！")
            print("次の実装項目: 2-2-3「信頼度閾値に基づく意思決定ロジック」")
        else:
            print("[WARNING] 一部テストに失敗しました。実装を確認してください。")
            
        return test_results
        
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        logger.exception("Test execution failed")
        return test_results

def create_normal_trend_data() -> pd.DataFrame:
    """通常トレンドデータ作成"""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    
    # 安定した上昇トレンド
    base_price = 100
    trend = np.linspace(0, 10, 60)  # 10%の上昇
    noise = np.random.normal(0, 0.5, 60)  # 低ノイズ
    
    prices = base_price + trend + noise
    volumes = np.random.randint(1000000, 2000000, 60)
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Open': np.roll(prices, 1)
    })

def create_transition_period_data() -> pd.DataFrame:
    """移行期データ作成（高ボラティリティ）"""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    
    # 不安定な価格動き
    base_price = 100
    
    # 前半は上昇、後半は横ばい（移行期）
    trend_part1 = np.linspace(0, 8, 30)
    trend_part2 = np.full(30, 8) + np.random.normal(0, 2, 30)
    trend = np.concatenate([trend_part1, trend_part2])
    
    # 高ボラティリティノイズ
    noise = np.random.normal(0, 2.0, 60)  # 高ノイズ
    
    prices = base_price + trend + noise
    volumes = np.random.randint(2000000, 5000000, 60)  # 高出来高
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.05,
        'Low': prices * 0.95,
        'Open': np.roll(prices, 1)
    })

if __name__ == "__main__":
    test_trend_transition_system()
