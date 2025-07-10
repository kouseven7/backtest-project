"""
重要指標選定システム - 実運用デモンストレーション
Metric Selection System - Production Usage Demo

このスクリプトは実運用での使用方法をデモンストレーションします。
"""

import os
import sys
from datetime import datetime

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

def simple_analysis_demo():
    """シンプルな分析デモ"""
    print("🚀 重要指標選定システム - シンプル実行デモ")
    print("=" * 60)
    
    try:
        from config.metric_selection_manager import MetricSelectionManager
        
        # 1. マネージャー初期化
        print("📋 システム初期化中...")
        manager = MetricSelectionManager()
        print("   ✓ 初期化完了")
        
        # 2. 重要指標分析実行
        print("\n📊 重要指標分析実行中...")
        result = manager.run_complete_analysis(
            target_metric="sharpe_ratio",
            optimization_method="balanced_approach"
        )
        print("   ✓ 分析完了")
        
        # 3. 結果表示
        print(f"\n📈 分析結果:")
        print(f"   信頼度レベル: {result.confidence_level}")
        print(f"   推奨指標数: {len(result.recommended_metrics)}")
        
        if result.weight_optimization_result:
            print(f"   重み改善スコア: {result.weight_optimization_result.improvement_score:.3f}")
        
        # 4. 上位推奨指標
        print(f"\n🏆 上位推奨指標 (TOP-3):")
        for i, metric in enumerate(result.recommended_metrics[:3], 1):
            print(f"   {i}. {metric['feature']} (重要度: {metric['importance_score']:.3f})")
        
        # 5. 最適化された重み
        if result.weight_optimization_result and result.weight_optimization_result.optimized_weights:
            print(f"\n⚖️ 最適化された重み:")
            weights = result.weight_optimization_result.optimized_weights
            for category, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"   {category}: {weight:.3f}")
        
        print(f"\n✅ システムが正常に動作しています!")
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_latest_report():
    """最新のレポートファイルを表示"""
    print("\n📄 最新分析レポートの確認")
    print("-" * 40)
    
    try:
        import glob
        report_pattern = "logs/metric_selection_system/reports/metric_selection_report_*.md"
        reports = glob.glob(report_pattern)
        
        if reports:
            latest_report = max(reports, key=os.path.getctime)
            print(f"   最新レポート: {latest_report}")
            
            # レポートの一部を表示
            with open(latest_report, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("\n📋 レポートサマリー:")
                for line in lines[:15]:  # 最初の15行を表示
                    if line.strip():
                        print(f"   {line.rstrip()}")
        else:
            print("   レポートファイルが見つかりません")
    
    except Exception as e:
        print(f"   レポート表示エラー: {e}")

def check_system_status():
    """システム状態の確認"""
    print("\n🔍 システム状態確認")
    print("-" * 40)
    
    # 必要なディレクトリの確認
    required_dirs = [
        "logs/metric_importance",
        "logs/metric_weight_optimization", 
        "logs/metric_selection_system",
        "logs/strategy_characteristics"
    ]
    
    print("   必要ディレクトリ:")
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"   {status} {dir_path}")
    
    # 設定ファイルの確認
    config_files = [
        "config/scoring_weights.json",
        "config/metric_selection_config.py",
        "config/strategy_scoring_model.py"
    ]
    
    print("\n   重要ファイル:")
    for file_path in config_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"   {status} {file_path}")

def main():
    """メイン関数"""
    print("🔧 重要指標選定システム - 実運用デモ")
    print("=" * 60)
    print("実行メニュー:")
    print("1. システム状態確認")
    print("2. シンプル分析実行")
    print("3. 最新レポート表示")
    print("4. 全体実行（推奨）")
    
    try:
        choice = input("\n選択 (1-4): ").strip()
        
        if choice == "1":
            check_system_status()
        elif choice == "2":
            simple_analysis_demo()
        elif choice == "3":
            show_latest_report()
        elif choice == "4":
            print("\n📋 全体実行を開始します...")
            check_system_status()
            success = simple_analysis_demo()
            if success:
                show_latest_report()
        else:
            print("無効な選択です。全体実行を行います。")
            check_system_status()
            success = simple_analysis_demo()
            if success:
                show_latest_report()
        
        print(f"\n🎯 実行完了 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 実行が中断されました。")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    main()
