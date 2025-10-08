"""
Demo Script for 4-3-2 Dashboard System
戦略比率とパフォーマンスのリアルタイム表示 デモンストレーション

このスクリプトは4-3-2システムの動作確認用デモです。
"""

import os
import sys
import time
from pathlib import Path
import logging

# プロジェクトパス設定
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """4-3-2 ダッシュボード システム デモ実行"""
    logger.info("=== 4-3-2 ダッシュボードシステム デモ開始 ===")
    
    try:
        # 1. データコレクター デモ
        demo_data_collector()
        
        # 2. チャート生成 デモ
        demo_chart_generator()
        
        # 3. 設定管理 デモ
        demo_config_manager()
        
        # 4. 統合ダッシュボード デモ
        demo_integrated_dashboard()
        
        logger.info("=== 4-3-2 ダッシュボードシステム デモ完了 ===")
        
    except Exception as e:
        logger.error(f"デモ実行中にエラー: {e}")
        return False
    
    return True

def demo_data_collector():
    """データコレクター デモ"""
    logger.info("--- データコレクター デモ開始 ---")
    
    try:
        from visualization.performance_data_collector import PerformanceDataCollector
        
        # データコレクター作成
        collector = PerformanceDataCollector()
        logger.info("データコレクター作成成功")
        
        # スナップショット収集テスト
        snapshot = collector.collect_current_snapshot("USDJPY")
        
        if snapshot:
            logger.info(f"スナップショット収集成功: {snapshot.timestamp}")
            logger.info(f"戦略数: {len(snapshot.strategy_allocations)}")
            logger.info(f"アラート数: {len(snapshot.alerts)}")
        else:
            logger.warning("スナップショット収集に失敗")
        
    except Exception as e:
        logger.error(f"データコレクター デモエラー: {e}")

def demo_chart_generator():
    """チャート生成 デモ"""
    logger.info("--- チャート生成 デモ開始 ---")
    
    try:
        from visualization.dashboard_chart_generator import DashboardChartGenerator
        from visualization.performance_data_collector import PerformanceDataCollector
        
        # チャート生成器作成
        chart_generator = DashboardChartGenerator()
        logger.info("チャート生成器作成成功")
        
        # テストデータ収集
        collector = PerformanceDataCollector()
        snapshot = collector.collect_current_snapshot("USDJPY")
        
        if snapshot:
            # ダッシュボードチャート生成
            chart_path = chart_generator.generate_performance_dashboard(
                snapshot, [snapshot]
            )
            
            if chart_path:
                logger.info(f"ダッシュボードチャート生成成功: {chart_path}")
                
                # サマリー生成テスト
                summary = chart_generator.generate_simple_summary(snapshot)
                logger.info(f"サマリー生成成功: {len(summary)} 文字")
            else:
                logger.warning("チャート生成に失敗")
        
    except Exception as e:
        logger.error(f"チャート生成 デモエラー: {e}")

def demo_config_manager():
    """設定管理 デモ"""
    logger.info("--- 設定管理 デモ開始 ---")
    
    try:
        from visualization.dashboard_config import DashboardConfig
        
        # デフォルト設定作成
        config = DashboardConfig()
        logger.info("デフォルト設定作成成功")
        logger.info(f"更新間隔: {config.update_interval_minutes}分")
        logger.info(f"チャート幅: {config.chart_width}px")
        
        # 設定保存テスト
        test_config_path = "logs/test_dashboard_config.json"
        success = config.save_to_file(test_config_path)
        
        if success:
            logger.info(f"設定ファイル保存成功: {test_config_path}")
            
            # 設定読み込みテスト
            loaded_config = DashboardConfig.load_from_file(test_config_path)
            if loaded_config:
                logger.info("設定ファイル読み込み成功")
            else:
                logger.warning("設定ファイル読み込み失敗")
        else:
            logger.warning("設定ファイル保存失敗")
        
    except Exception as e:
        logger.error(f"設定管理 デモエラー: {e}")

def demo_integrated_dashboard():
    """統合ダッシュボード デモ"""
    logger.info("--- 統合ダッシュボード デモ開始 ---")
    
    try:
        from visualization.strategy_performance_dashboard import StrategyPerformanceDashboard
        
        # ダッシュボード作成
        dashboard = StrategyPerformanceDashboard("USDJPY")
        logger.info("統合ダッシュボード作成成功")
        
        # 状態確認
        status = dashboard.get_status()
        logger.info(f"ダッシュボード状態: {status}")
        
        # 手動更新テスト
        logger.info("手動更新テスト実行中...")
        update_success = dashboard.manual_update()
        
        if update_success:
            logger.info("手動更新成功")
            
            # レポート生成テスト
            report_path = dashboard.generate_dashboard_report()
            if report_path:
                logger.info(f"ダッシュボードレポート生成成功: {report_path}")
            else:
                logger.warning("ダッシュボードレポート生成失敗")
        else:
            logger.warning("手動更新失敗")
        
        # クリーンアップ
        dashboard.stop_dashboard()
        logger.info("ダッシュボード停止完了")
        
    except Exception as e:
        logger.error(f"統合ダッシュボード デモエラー: {e}")

def test_components_individually():
    """コンポーネント個別テスト"""
    logger.info("=== コンポーネント個別テスト開始 ===")
    
    results = {}
    
    # 1. データ収集テスト
    try:
        from visualization.performance_data_collector import PerformanceDataCollector
        collector = PerformanceDataCollector()
        snapshot = collector.collect_current_snapshot("USDJPY")
        results['data_collector'] = snapshot is not None
        logger.info(f"データ収集テスト: {'成功' if results['data_collector'] else '失敗'}")
    except Exception as e:
        results['data_collector'] = False
        logger.error(f"データ収集テストエラー: {e}")
    
    # 2. チャート生成テスト
    try:
        from visualization.dashboard_chart_generator import DashboardChartGenerator
        chart_gen = DashboardChartGenerator()
        results['chart_generator'] = True
        logger.info("チャート生成テスト: 成功")
    except Exception as e:
        results['chart_generator'] = False
        logger.error(f"チャート生成テストエラー: {e}")
    
    # 3. 設定管理テスト
    try:
        from visualization.dashboard_config import DashboardConfig
        config = DashboardConfig()
        results['config_manager'] = True
        logger.info("設定管理テスト: 成功")
    except Exception as e:
        results['config_manager'] = False
        logger.error(f"設定管理テストエラー: {e}")
    
    # 4. メインダッシュボードテスト
    try:
        from visualization.strategy_performance_dashboard import StrategyPerformanceDashboard
        dashboard = StrategyPerformanceDashboard("USDJPY")
        results['main_dashboard'] = True
        logger.info("メインダッシュボードテスト: 成功")
    except Exception as e:
        results['main_dashboard'] = False
        logger.error(f"メインダッシュボードテストエラー: {e}")
    
    # テスト結果サマリー
    logger.info("=== テスト結果サマリー ===")
    success_count = sum(results.values())
    total_count = len(results)
    
    for component, success in results.items():
        status = "[OK] 成功" if success else "[ERROR] 失敗"
        logger.info(f"{component}: {status}")
    
    logger.info(f"総合結果: {success_count}/{total_count} 成功")
    return success_count == total_count

if __name__ == "__main__":
    # コマンドライン引数処理
    import argparse
    parser = argparse.ArgumentParser(description='4-3-2 Dashboard Demo')
    parser.add_argument('--component-test', action='store_true', 
                       help='Run individual component tests')
    args = parser.parse_args()
    
    if args.component_test:
        success = test_components_individually()
    else:
        success = main()
    
    if success:
        logger.info("[SUCCESS] デモ実行完了 - 全て正常")
        sys.exit(0)
    else:
        logger.error("[ERROR] デモ実行失敗")
        sys.exit(1)
