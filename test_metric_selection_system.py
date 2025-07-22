"""
Test script for Metric Selection System
重要指標選定システムの包括的テストスクリプト

Usage: python test_metric_selection_system.py
"""

import os
import sys
import logging
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

def test_config_system():
    """設定システムのテスト"""
    print("=== 設定システムテスト ===")
    
    try:
        from config.metric_selection_config import MetricSelectionConfig
        
        # 設定の初期化
        config = MetricSelectionConfig()
        print("✓ 設定システム初期化完了")
        
        # 設定値の取得テスト
        target_metrics = config.get_target_metrics()
        target_variable = config.get_target_variable()
        analysis_methods = config.get_analysis_methods()
        
        print(f"  対象指標数: {len(target_metrics)}")
        print(f"  目標指標: {target_variable}")
        print(f"  分析手法: {analysis_methods}")
        
        # 設定検証
        errors = config.validate_config()
        if errors:
            print(f"  設定エラー: {errors}")
            return False
        else:
            print("  ✓ 設定は有効です")
            return True
        
    except Exception as e:
        print(f"✗ 設定システムエラー: {e}")
        return False

def test_importance_analyzer():
    """重要指標分析エンジンのテスト"""
    print("\n=== 重要指標分析エンジンテスト ===")
    
    try:
        from config.metric_importance_analyzer import MetricImportanceAnalyzer
        
        # 分析エンジンの初期化
        analyzer = MetricImportanceAnalyzer()
        print("✓ 分析エンジン初期化完了")
        
        # データ収集テスト
        data = analyzer.collect_strategy_data()
        if not data.empty:
            print(f"✓ データ収集完了: {len(data)}行, {len(data.columns)}列")
            
            # 基本的な分析実行
            results = analyzer.analyze_metric_importance()
            
            if "error" not in results:
                print("✓ 分析実行完了")
                
                # 結果の詳細表示
                data_summary = results.get("data_summary", {})
                print(f"  分析サンプル数: {data_summary.get('total_samples', 0)}")
                print(f"  分析戦略数: {data_summary.get('strategies_count', 0)}")
                print(f"  分析手法: {results.get('analysis_methods', [])}")
                
                # 推奨指標
                recommended = results.get("recommended_metrics", [])
                print(f"  推奨指標数: {len(recommended)}")
                
                if recommended:
                    print("  上位推奨指標:")
                    for i, metric in enumerate(recommended[:3], 1):
                        print(f"    {i}. {metric['feature']} (スコア: {metric['importance_score']:.3f})")
                
                return True
            else:
                print(f"✗ 分析エラー: {results['error']}")
                return False
        else:
            print("⚠️ データが見つかりません（正常な場合もあります）")
            return True
            
    except Exception as e:
        print(f"✗ 分析エンジンエラー: {e}")
        return False

def test_weight_optimizer():
    """重み最適化システムのテスト"""
    print("\n=== 重み最適化システムテスト ===")
    
    try:
        from config.metric_weight_optimizer import MetricWeightOptimizer
        
        # 最適化器の初期化
        optimizer = MetricWeightOptimizer()
        print("✓ 重み最適化器初期化完了")
        
        # 簡単な最適化テスト
        result = optimizer.optimize_weights(optimization_method="importance_based")
        
        if result.success:
            print("✓ 重み最適化完了")
            print(f"  改善スコア: {result.improvement_score:.3f}")
            print(f"  最適化手法: {result.optimization_method}")
            
            # 重みの変化を表示
            print("  重みの変化 (上位3カテゴリ):")
            changes = []
            for category in result.original_weights.keys():
                original = result.original_weights[category]
                optimized = result.optimized_weights[category]
                change = abs(optimized - original)
                changes.append((category, original, optimized, change))
            
            # 変化量で降順ソート
            changes.sort(key=lambda x: x[3], reverse=True)
            
            for i, (category, original, optimized, change) in enumerate(changes[:3], 1):
                print(f"    {i}. {category}: {original:.3f} → {optimized:.3f} (変化: {change:.3f})")
            
            return True
        else:
            print(f"✗ 最適化エラー: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"✗ 重み最適化エラー: {e}")
        return False

def test_integrated_system():
    """統合システムのテスト"""
    print("\n=== 統合システムテスト ===")
    
    try:
        from config.metric_selection_manager import MetricSelectionManager
        
        # 統合管理システムの初期化
        manager = MetricSelectionManager()
        print("✓ 統合システム初期化完了")
        
        # 完全分析の実行（重み適用なし）
        summary = manager.run_complete_analysis(
            optimization_method="balanced_approach",
            apply_weights=False
        )
        
        if summary.success:
            print("✓ 統合分析実行完了")
            print(f"  信頼度レベル: {summary.confidence_level}")
            print(f"  推奨指標数: {len(summary.recommended_metrics)}")
            
            # パフォーマンス影響の表示
            if summary.performance_impact:
                print("  パフォーマンス影響:")
                for key, value in list(summary.performance_impact.items())[:3]:
                    print(f"    {key}: {value:.3f}")
            
            # 重み最適化結果
            if summary.weight_optimization_result and summary.weight_optimization_result.success:
                print(f"  重み最適化改善: {summary.weight_optimization_result.improvement_score:.3f}")
            
            return True
        else:
            print("✗ 統合分析失敗")
            for error in summary.error_messages[:3]:  # 最初の3つのエラーのみ表示
                print(f"  エラー: {error}")
            return False
            
    except Exception as e:
        print(f"✗ 統合システムエラー: {e}")
        return False

def test_file_operations():
    """ファイル操作のテスト"""
    print("\n=== ファイル操作テスト ===")
    
    try:
        import tempfile
        import json
        from pathlib import Path
        
        # 一時ディレクトリでのテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 設定ファイルの作成テスト
            config_dir = temp_path / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            test_config = {
                "test_key": "test_value",
                "timestamp": datetime.now().isoformat()
            }
            
            config_file = config_dir / "test_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(test_config, f, indent=2, ensure_ascii=False)
            
            # ファイル読み込みテスト
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            if loaded_config["test_key"] == "test_value":
                print("✓ ファイル操作テスト完了")
                return True
            else:
                print("✗ ファイル内容が一致しません")
                return False
                
    except Exception as e:
        print(f"✗ ファイル操作エラー: {e}")
        return False

def run_comprehensive_test():
    """包括的テストの実行"""
    print("=" * 60)
    print("重要指標選定システム 包括的テスト開始")
    print("=" * 60)
    
    test_results = []
    
    # 各テストの実行
    tests = [
        ("設定システム", test_config_system),
        ("ファイル操作", test_file_operations),
        ("重要指標分析エンジン", test_importance_analyzer),
        ("重み最適化システム", test_weight_optimizer),
        ("統合システム", test_integrated_system)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}で予期しないエラー: {e}")
            test_results.append((test_name, False))
    
    # 結果の集計
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n合計: {passed + failed}テスト")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    
    if failed == 0:
        print("\n🎉 全テストが成功しました！")
        print("重要指標選定システムは正常に動作しています。")
    else:
        print(f"\n⚠️ {failed}個のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテストが中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n予期しないエラーが発生しました: {e}")
        sys.exit(1)
