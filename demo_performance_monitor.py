"""
パフォーマンス監視システム デモ実行スクリプト
短期間隔でのテスト実行
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from performance_monitor import PerformanceMonitor


async def demo_performance_monitoring():
    """デモ実行（短期間隔）"""
    print("[ROCKET] パフォーマンス監視デモ開始")
    
    # 短期間隔設定でテスト実行
    monitor = PerformanceMonitor()
    monitor.config['monitoring_settings']['update_interval_seconds'] = 30  # 30秒間隔
    
    print("[CHART] 設定確認:")
    print(f"  監視間隔: {monitor.config['monitoring_settings']['update_interval_seconds']}秒")
    print(f"  出力ディレクトリ: {monitor.output_dir}")
    print(f"  アラートルール: {len(monitor.performance_alert_manager.alert_rules)}件")
    
    try:
        # 60秒間のデモ実行
        print("\n⏱️  60秒間のデモ監視を開始...")
        print("   Ctrl+C で早期停止可能")
        
        # タイムアウト付きで実行
        await asyncio.wait_for(monitor.start_monitoring(), timeout=60)
        
    except asyncio.TimeoutError:
        print("\n[OK] デモ完了（60秒経過）")
        monitor.stop_monitoring()
    except KeyboardInterrupt:
        print("\n[WARNING]  ユーザーによる停止")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"\n[ERROR] エラー発生: {e}")
        monitor.stop_monitoring()
    
    # 結果確認
    print("\n[LIST] 監視結果:")
    status = monitor.get_status()
    print(f"  履歴データ件数: {status['history_count']}")
    print(f"  最後の更新: {status['last_update']}")
    
    # 出力ファイル確認
    output_files = list(monitor.output_dir.glob("*.json"))
    if output_files:
        print(f"  出力ファイル: {len(output_files)}件")
        for file_path in output_files[-3:]:  # 最新3件表示
            print(f"    {file_path.name}")
    
    print("\n[SUCCESS] デモ完了")


if __name__ == "__main__":
    asyncio.run(demo_performance_monitoring())
