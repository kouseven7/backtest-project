#!/usr/bin/env python3
"""
直接テスト: 自動削除機能統合動作確認

Author: GitHub Copilot Agent
Created: 2025-10-06
"""

import sys
from pathlib import Path

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fallback_monitoring_system import FallbackMonitor

def test_direct_cleanup():
    """直接自動削除機能テスト"""
    print("[TEST] 直接自動削除機能テスト")
    
    monitor = FallbackMonitor()
    
    # 1. auto_cleanup属性の確認
    print(f"auto_cleanup属性存在: {hasattr(monitor, 'auto_cleanup')}")
    print(f"auto_cleanup値: {monitor.auto_cleanup is not None}")
    
    # 2. 直接auto_cleanup機能呼び出し
    if monitor.auto_cleanup is not None:
        try:
            print("\n🧹 直接自動削除実行...")
            cleanup_results = monitor.auto_cleanup.implement_auto_cleanup()
            
            print("[OK] 直接実行成功:")
            print(f"  ステータス: {cleanup_results.get('overall_status', 'unknown')}")
            print(f"  実行時間: {cleanup_results.get('execution_duration', 0):.2f}秒")
            
            cleanup_data = cleanup_results.get('cleanup_results', {})
            print(f"  処理ファイル数: {cleanup_data.get('total_files_processed', 0)}")
            print(f"  削除ファイル数: {cleanup_data.get('total_files_deleted', 0)}")
            print(f"  解放容量: {cleanup_data.get('total_space_freed', 0)}MB")
            
            # 3. 統計情報取得
            print("\n[CHART] 統計情報取得...")
            stats = monitor.auto_cleanup.get_cleanup_statistics()
            
            for dir_name, dir_stats in stats.get('directories', {}).items():
                print(f"  {dir_name}: {dir_stats['total_files']}ファイル ({dir_stats['total_size_mb']}MB)")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 直接実行エラー: {e}")
            return False
    else:
        print("[ERROR] auto_cleanup機能が利用できません")
        return False

if __name__ == "__main__":
    success = test_direct_cleanup()
    sys.exit(0 if success else 1)