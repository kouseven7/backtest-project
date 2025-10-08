#!/usr/bin/env python3
"""
自動削除機能統合テスト

FallbackMonitorクラスの自動削除機能統合の動作検証を行う

Author: GitHub Copilot Agent
Created: 2025-10-06
Task: Auto cleanup integration test
"""

import sys
from pathlib import Path

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fallback_monitoring_system import FallbackMonitor

def test_integrated_auto_cleanup():
    """統合自動削除機能テスト"""
    print("[TEST] 自動削除機能統合テスト開始")
    
    try:
        # FallbackMonitor初期化
        monitor = FallbackMonitor()
        
        # 1. 自動削除機能の利用可能性確認
        print(f"\n[TOOL] 自動削除機能利用可能: {monitor.auto_cleanup is not None}")
        
        # 2. 自動削除統計情報取得
        print("\n[CHART] 自動削除統計情報:")
        cleanup_stats = monitor.get_auto_cleanup_statistics()
        
        if cleanup_stats.get('auto_cleanup_available', True):
            for dir_name, dir_stats in cleanup_stats.get('directories', {}).items():
                print(f"  {dir_name}: {dir_stats['total_files']}ファイル ({dir_stats['total_size_mb']}MB)")
        else:
            print(f"  {cleanup_stats.get('message', '不明なエラー')}")
        
        # 3. 自動削除機能付き週次レポート生成テスト
        print("\n[LIST] 自動削除機能付き週次レポート生成テスト...")
        enhanced_report = monitor.generate_weekly_report_with_cleanup()
        
        # 4. 結果検証
        print("\n[OK] テスト結果:")
        print(f"  統合ステータス: {enhanced_report.get('integration_status', 'unknown')}")
        print(f"  自動削除有効: {enhanced_report.get('auto_cleanup_enabled', False)}")
        
        if enhanced_report.get('cleanup_results'):
            cleanup_results = enhanced_report['cleanup_results']
            print(f"  処理ファイル数: {cleanup_results.get('cleanup_results', {}).get('total_files_processed', 0)}")
            print(f"  削除ファイル数: {cleanup_results.get('cleanup_results', {}).get('total_files_deleted', 0)}")
            print(f"  解放容量: {cleanup_results.get('cleanup_results', {}).get('total_space_freed', 0)}MB")
            print(f"  実行時間: {cleanup_results.get('execution_duration', 0):.2f}秒")
        
        # 5. ディレクトリ構造確認
        print("\n📁 生成ディレクトリ構造:")
        reports_dir = monitor.reports_dir
        for subdir in ['weekly', 'charts', 'dashboard', 'backup', 'cleanup_logs']:
            subdir_path = reports_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob('*')))
                print(f"  {subdir}/: {file_count}ファイル")
            else:
                print(f"  {subdir}/: ディレクトリ未作成")
        
        print("\n[SUCCESS] 自動削除機能統合テスト成功")
        return True
        
    except Exception as e:
        print(f"[ERROR] 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    return test_integrated_auto_cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)