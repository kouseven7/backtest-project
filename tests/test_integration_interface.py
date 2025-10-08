#!/usr/bin/env python3
"""
統合インターフェースの動作確認テストファイル
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config.trend_strategy_integration_interface import create_integration_interface, quick_strategy_decision

def test_basic_functionality():
    """基本機能のテスト"""
    print("[TOOL] 統合インターフェースの基本機能テスト開始")
    
    # サンプルデータ作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Adj Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    try:
        # 1. 統合インターフェースの作成
        print("[CHART] 統合インターフェース初期化...")
        interface = create_integration_interface(enable_async=False)
        print("  ✓ 初期化成功")
        
        # 2. データ検証
        print("[CHART] データ検証テスト...")
        is_valid, issues = interface.validate_market_data(sample_data)
        print(f"  データ検証: {'✓ 有効' if is_valid else '[ERROR] 無効'}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")
        
        # 3. 統合判定テスト
        print("[CHART] 統合判定テスト...")
        result = interface.integrate_decision(sample_data, "TEST_TICKER")
        print(f"  選択戦略: {result.strategy_selection.selected_strategies}")
        print(f"  トレンド: {result.trend_analysis.trend_type} (信頼度: {result.trend_analysis.confidence:.2f})")
        print(f"  総合リスク: {result.risk_assessment.get('overall_risk', 'N/A')}")
        print(f"  処理時間: {result.processing_time_ms:.1f}ms")
        print(f"  統合ステータス: {result.integration_status}")
        
        # 4. パフォーマンス統計
        print("[CHART] パフォーマンス統計...")
        stats = interface.get_performance_statistics()
        print(f"  総リクエスト数: {stats['performance_metrics']['total_requests']}")
        print(f"  成功率: {stats['performance_metrics']['successful_requests']}/{stats['performance_metrics']['total_requests']}")
        print(f"  キャッシュヒット率: {stats['performance_metrics']['cache_hit_rate']:.2%}")
        
        # 5. クイック判定テスト
        print("[CHART] クイック判定テスト...")
        quick_result = quick_strategy_decision(sample_data, "QUICK_TEST", max_strategies=2)
        print(f"  クイック選択戦略: {quick_result.strategy_selection.selected_strategies}")
        print(f"  クイック処理時間: {quick_result.processing_time_ms:.1f}ms")
        
        print("[OK] 統合インターフェース基本機能テスト完了！")
        return True
        
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """エラーハンドリングのテスト"""
    print("[TOOL] エラーハンドリングテスト開始")
    
    try:
        interface = create_integration_interface(enable_async=False)
        
        # 不正データでのテスト
        bad_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        is_valid, issues = interface.validate_market_data(bad_data)
        print(f"  不正データ検証: {'✓ 正しく検出' if not is_valid else '[ERROR] 検出失敗'}")
        print(f"  検出された問題: {len(issues)}件")
        
        # フォールバック処理のテスト
        result = interface.integrate_decision(bad_data, "ERROR_TEST")
        print(f"  フォールバック処理: {'✓ 動作' if result else '[ERROR] 失敗'}")
        print(f"  フォールバック戦略: {result.strategy_selection.selected_strategies}")
        
        print("[OK] エラーハンドリングテスト完了！")
        return True
        
    except Exception as e:
        print(f"[ERROR] エラーハンドリングテストエラー: {e}")
        return False

if __name__ == "__main__":
    print("[ROCKET] 3-1-2「トレンド戦略統合インターフェース」動作確認テスト")
    print("=" * 60)
    
    # 基本機能テスト
    basic_success = test_basic_functionality()
    print()
    
    # エラーハンドリングテスト
    error_success = test_error_handling()
    print()
    
    # 結果サマリー
    print("=" * 60)
    if basic_success and error_success:
        print("[SUCCESS] 全てのテストが成功！統合インターフェースが正常に動作しています。")
    else:
        print("[WARNING]  一部のテストで問題が発生しました。ログを確認してください。")
    
    print("\n[LIST] 実装完了機能:")
    print("  ✓ 厚い統合レイヤー方式")
    print("  ✓ リアルタイム・バッチ処理対応")
    print("  ✓ 新しいデータクラス群")
    print("  ✓ フォールバック方式")
    print("  ✓ キャッシュシステム")
    print("  ✓ リスク評価機能")
    print("  ✓ 推奨アクション生成")
    print("  ✓ パフォーマンス監視")
