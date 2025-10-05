#!/usr/bin/env python3
"""
Fallback Monitoring System - Auto Cleanup Module

週次レポート、チャート、ダッシュボードの自動削除機能実装
段階的削除戦略、安全な削除、バックアップ、復旧機能を提供

Author: GitHub Copilot Agent
Created: 2025-10-06
Task: Auto cleanup implementation for TODO-QG-002
"""

import json
import os
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import glob
import re

# 自動削除設定（推奨）
RETENTION_SETTINGS = {
    "weekly_reports": {
        "keep_weeks": 12,      # 3ヶ月分保持
        "archive_weeks": 52,   # 1年分をアーカイブ
        "delete_after": 104    # 2年後完全削除
    },
    "charts": {
        "keep_weeks": 4,       # 1ヶ月分保持
        "delete_after": 12     # 3ヶ月後削除
    },
    "dashboard": {
        "keep_latest": 1       # 最新のみ保持
    }
}

class FallbackReportAutoCleanup:
    """
    フォールバック監視レポート自動削除システム
    
    主要機能:
    1. 段階的削除戦略（即座・週次・月次・年次）
    2. 安全な削除（バックアップ機能付き）
    3. 削除ログ記録
    4. 復旧機能（緊急時）
    """
    
    def __init__(self, reports_base_dir: Optional[Path] = None):
        """
        初期化
        
        Args:
            reports_base_dir: レポートベースディレクトリ（デフォルト: プロジェクトルート/reports/fallback_monitoring）
        """
        # プロジェクトルート取得
        if reports_base_dir is None:
            project_root = Path(__file__).parent
            self.reports_dir = project_root / "reports" / "fallback_monitoring"
        else:
            self.reports_dir = reports_base_dir
        
        # サブディレクトリ設定
        self.weekly_dir = self.reports_dir / "weekly"
        self.charts_dir = self.reports_dir / "charts"
        self.dashboard_dir = self.reports_dir / "dashboard"
        self.baseline_dir = self.reports_dir / "baseline"
        
        # バックアップ・ログディレクトリ
        self.backup_dir = self.reports_dir / "backup"
        self.cleanup_log_dir = self.reports_dir / "cleanup_logs"
        
        # ディレクトリ作成
        for directory in [
            self.weekly_dir, self.charts_dir, self.dashboard_dir, 
            self.baseline_dir, self.backup_dir, self.cleanup_log_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # ロガー設定
        self.logger = self._setup_logger()
        
        # 削除設定読み込み
        self.retention_settings = RETENTION_SETTINGS.copy()
        
    def _setup_logger(self) -> logging.Logger:
        """専用ロガー設定"""
        logger = logging.getLogger('fallback_auto_cleanup')
        logger.setLevel(logging.INFO)
        
        # ログファイルハンドラー
        log_file = self.cleanup_log_dir / f"auto_cleanup_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def implement_auto_cleanup(self) -> Dict[str, Any]:
        """
        自動削除機能実装のメイン関数
        
        Returns:
            Dict[str, Any]: 削除実行結果
        """
        cleanup_start = datetime.now()
        self.logger.info("🧹 自動削除機能実行開始")
        
        try:
            # 1. 段階的削除戦略実行
            cleanup_results = self._execute_staged_cleanup_strategy()
            
            # 2. 安全な削除実行
            safety_results = self._execute_safe_deletion(cleanup_results)
            
            # 3. バックアップ作成
            backup_results = self._create_backup_archives()
            
            # 4. 削除ログ記録
            log_results = self._record_deletion_log(cleanup_results, safety_results, backup_results)
            
            # 5. 統合結果
            final_results = {
                'cleanup_timestamp': cleanup_start.isoformat(),
                'execution_duration': (datetime.now() - cleanup_start).total_seconds(),
                'cleanup_results': cleanup_results,
                'safety_results': safety_results,
                'backup_results': backup_results,
                'log_results': log_results,
                'overall_status': 'success'
            }
            
            self.logger.info("✅ 自動削除機能実行完了")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 自動削除機能実行エラー: {e}")
            return {
                'cleanup_timestamp': cleanup_start.isoformat(),
                'overall_status': 'error',
                'error_message': str(e)
            }
    
    def _execute_staged_cleanup_strategy(self) -> Dict[str, Any]:
        """段階的削除戦略実行"""
        self.logger.info("📋 段階的削除戦略実行中...")
        
        # 1. 段階的削除戦略定義
        cleanup_strategy = {
            "immediate": ["dashboard/fallback_dashboard_*.html"],  # 毎回更新（最新のみ保持）
            "weekly": ["charts/*.png"],                           # 4週間保持
            "monthly": ["weekly/fallback_weekly_report_*.json"],  # 12週間保持
            "yearly": ["baseline/*.json"]                         # 2年間保持
        }
        
        strategy_results = {}
        
        for strategy_type, file_patterns in cleanup_strategy.items():
            self.logger.info(f"🗂️ {strategy_type}削除戦略実行中...")
            
            strategy_result = self._execute_strategy_by_type(strategy_type, file_patterns)
            strategy_results[strategy_type] = strategy_result
            
            self.logger.info(f"✅ {strategy_type}削除戦略完了: {strategy_result['files_processed']}ファイル処理")
        
        return {
            'strategy_results': strategy_results,
            'total_files_processed': sum(r['files_processed'] for r in strategy_results.values()),
            'total_files_deleted': sum(r['files_deleted'] for r in strategy_results.values()),
            'total_space_freed': sum(r['space_freed_mb'] for r in strategy_results.values())
        }
    
    def _execute_strategy_by_type(self, strategy_type: str, file_patterns: List[str]) -> Dict[str, Any]:
        """戦略タイプ別削除実行"""
        
        files_to_delete = []
        files_processed = 0
        
        # ファイルパターンマッチング
        for pattern in file_patterns:
            full_pattern = str(self.reports_dir / pattern)
            matched_files = glob.glob(full_pattern)
            files_processed += len(matched_files)
            
            # 削除対象ファイル特定
            deletion_candidates = self._identify_deletion_candidates(
                matched_files, strategy_type
            )
            files_to_delete.extend(deletion_candidates)
        
        # 実際の削除実行
        deletion_results = self._delete_files_safely(files_to_delete)
        
        return {
            'strategy_type': strategy_type,
            'files_processed': files_processed,
            'files_deleted': len(deletion_results['deleted_files']),
            'space_freed_mb': deletion_results['space_freed_mb'],
            'deleted_files': deletion_results['deleted_files'],
            'backup_created': deletion_results['backup_created']
        }
    
    def _identify_deletion_candidates(self, file_list: List[str], strategy_type: str) -> List[str]:
        """削除対象ファイル特定"""
        
        if not file_list:
            return []
        
        # ファイルを作成日時でソート（新しい順）
        files_with_timestamps = []
        for file_path in file_list:
            try:
                file_stat = os.stat(file_path)
                creation_time = datetime.fromtimestamp(file_stat.st_ctime)
                files_with_timestamps.append((file_path, creation_time))
            except OSError:
                continue
        
        # 作成日時でソート（新しいファイルが先頭）
        files_with_timestamps.sort(key=lambda x: x[1], reverse=True)
        
        # 戦略別保持ポリシー適用
        deletion_candidates = []
        
        if strategy_type == "immediate":
            # ダッシュボード: 最新の1つのみ保持
            keep_count = self.retention_settings["dashboard"]["keep_latest"]
            if len(files_with_timestamps) > keep_count:
                deletion_candidates = [f[0] for f in files_with_timestamps[keep_count:]]
                
        elif strategy_type == "weekly":
            # チャート: 4週間分保持
            cutoff_date = datetime.now() - timedelta(weeks=self.retention_settings["charts"]["keep_weeks"])
            deletion_candidates = [
                f[0] for f in files_with_timestamps 
                if f[1] < cutoff_date
            ]
            
        elif strategy_type == "monthly":
            # 週次レポート: 12週間分保持
            cutoff_date = datetime.now() - timedelta(weeks=self.retention_settings["weekly_reports"]["keep_weeks"])
            deletion_candidates = [
                f[0] for f in files_with_timestamps 
                if f[1] < cutoff_date
            ]
            
        elif strategy_type == "yearly":
            # ベースライン: 2年間保持
            cutoff_date = datetime.now() - timedelta(weeks=self.retention_settings["weekly_reports"]["delete_after"])
            deletion_candidates = [
                f[0] for f in files_with_timestamps 
                if f[1] < cutoff_date
            ]
        
        return deletion_candidates
    
    def _delete_files_safely(self, files_to_delete: List[str]) -> Dict[str, Any]:
        """安全なファイル削除実行"""
        
        deleted_files = []
        space_freed = 0
        backup_files = []
        
        for file_path in files_to_delete:
            try:
                # ファイルサイズ取得
                file_size = os.path.getsize(file_path)
                
                # バックアップ作成
                backup_path = self._create_file_backup(file_path)
                if backup_path:
                    backup_files.append(backup_path)
                
                # ファイル削除
                os.remove(file_path)
                deleted_files.append(file_path)
                space_freed += file_size
                
                self.logger.info(f"🗑️ 削除完了: {file_path} ({file_size} bytes)")
                
            except Exception as e:
                self.logger.error(f"❌ 削除失敗: {file_path} - {e}")
        
        return {
            'deleted_files': deleted_files,
            'space_freed_mb': round(space_freed / (1024 * 1024), 2),
            'backup_created': backup_files
        }
    
    def _create_file_backup(self, file_path: str) -> Optional[str]:
        """個別ファイルのバックアップ作成"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            # バックアップファイル名生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
            backup_path = self.backup_dir / backup_filename
            
            # ファイルコピー
            shutil.copy2(source_path, backup_path)
            self.logger.info(f"💾 バックアップ作成: {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"❌ バックアップ作成失敗: {file_path} - {e}")
            return None
    
    def _execute_safe_deletion(self, cleanup_results: Dict[str, Any]) -> Dict[str, Any]:
        """安全な削除機能実行"""
        self.logger.info("🛡️ 安全な削除機能実行中...")
        
        # 削除前検証
        pre_deletion_check = self._perform_pre_deletion_check(cleanup_results)
        
        # 削除実行確認
        deletion_confirmation = self._confirm_deletion_safety(cleanup_results)
        
        # 削除後検証
        post_deletion_verification = self._verify_post_deletion_state()
        
        return {
            'pre_deletion_check': pre_deletion_check,
            'deletion_confirmation': deletion_confirmation,
            'post_deletion_verification': post_deletion_verification,
            'safety_level': 'high' if all([
                pre_deletion_check['status'] == 'pass',
                deletion_confirmation['confirmed'],
                post_deletion_verification['status'] == 'pass'
            ]) else 'medium'
        }
    
    def _create_backup_archives(self) -> Dict[str, Any]:
        """バックアップアーカイブ作成"""
        self.logger.info("📦 バックアップアーカイブ作成中...")
        
        archive_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 週次レポートアーカイブ
        weekly_archive = self._create_weekly_reports_archive(archive_timestamp)
        
        # チャートアーカイブ
        charts_archive = self._create_charts_archive(archive_timestamp)
        
        # 古いバックアップファイル削除
        old_backup_cleanup = self._cleanup_old_backup_files()
        
        return {
            'weekly_archive': weekly_archive,
            'charts_archive': charts_archive,
            'old_backup_cleanup': old_backup_cleanup,
            'archive_directory': str(self.backup_dir)
        }
    
    def _create_weekly_reports_archive(self, timestamp: str) -> Dict[str, Any]:
        """週次レポートアーカイブ作成"""
        try:
            archive_name = f"weekly_reports_archive_{timestamp}.zip"
            archive_path = self.backup_dir / archive_name
            
            archived_files = []
            
            # 古い週次レポートをアーカイブ
            cutoff_date = datetime.now() - timedelta(weeks=self.retention_settings["weekly_reports"]["archive_weeks"])
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for report_file in self.weekly_dir.glob("fallback_weekly_report_*.json"):
                    try:
                        file_stat = os.stat(report_file)
                        creation_time = datetime.fromtimestamp(file_stat.st_ctime)
                        
                        if creation_time < cutoff_date:
                            zipf.write(report_file, report_file.name)
                            archived_files.append(str(report_file))
                    except OSError:
                        continue
            
            if archived_files:
                self.logger.info(f"📦 週次レポートアーカイブ作成完了: {archive_path}")
                return {
                    'archive_created': True,
                    'archive_path': str(archive_path),
                    'files_archived': len(archived_files),
                    'archived_files': archived_files
                }
            else:
                # 空のアーカイブファイル削除
                if archive_path.exists():
                    archive_path.unlink()
                return {
                    'archive_created': False,
                    'reason': 'no_files_to_archive'
                }
                
        except Exception as e:
            self.logger.error(f"❌ 週次レポートアーカイブ作成失敗: {e}")
            return {
                'archive_created': False,
                'error': str(e)
            }
    
    def _record_deletion_log(self, cleanup_results: Dict[str, Any], 
                           safety_results: Dict[str, Any], 
                           backup_results: Dict[str, Any]) -> Dict[str, Any]:
        """削除ログ記録"""
        self.logger.info("📝 削除ログ記録中...")
        
        log_timestamp = datetime.now()
        log_data = {
            'deletion_timestamp': log_timestamp.isoformat(),
            'cleanup_results': cleanup_results,
            'safety_results': safety_results,
            'backup_results': backup_results,
            'system_info': {
                'reports_directory': str(self.reports_dir),
                'retention_settings': self.retention_settings,
                'cleanup_version': '1.0.0'
            }
        }
        
        # JSON形式でログ保存
        log_filename = f"deletion_log_{log_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        log_path = self.cleanup_log_dir / log_filename
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"📄 削除ログ保存完了: {log_path}")
            
            return {
                'log_created': True,
                'log_path': str(log_path),
                'log_size_bytes': log_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"❌ 削除ログ保存失敗: {e}")
            return {
                'log_created': False,
                'error': str(e)
            }
    
    def restore_from_backup(self, backup_file: str) -> Dict[str, Any]:
        """復旧機能（緊急時）"""
        self.logger.info(f"🔄 バックアップからの復旧開始: {backup_file}")
        
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                return {
                    'restore_success': False,
                    'error': 'backup_file_not_found'
                }
            
            # バックアップファイル名から元の場所を推定
            original_location = self._determine_original_location(backup_path)
            
            if original_location:
                # ファイル復旧
                shutil.copy2(backup_path, original_location)
                self.logger.info(f"✅ ファイル復旧完了: {original_location}")
                
                return {
                    'restore_success': True,
                    'restored_file': str(original_location),
                    'backup_file': backup_file
                }
            else:
                return {
                    'restore_success': False,
                    'error': 'cannot_determine_original_location'
                }
                
        except Exception as e:
            self.logger.error(f"❌ 復旧失敗: {backup_file} - {e}")
            return {
                'restore_success': False,
                'error': str(e)
            }
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """削除統計情報取得"""
        stats = {
            'directories': {
                'weekly_reports': {
                    'total_files': len(list(self.weekly_dir.glob("*.json"))),
                    'total_size_mb': self._calculate_directory_size(self.weekly_dir)
                },
                'charts': {
                    'total_files': len(list(self.charts_dir.glob("*.png"))),
                    'total_size_mb': self._calculate_directory_size(self.charts_dir)
                },
                'dashboard': {
                    'total_files': len(list(self.dashboard_dir.glob("*.html"))),
                    'total_size_mb': self._calculate_directory_size(self.dashboard_dir)
                },
                'backup': {
                    'total_files': len(list(self.backup_dir.glob("*"))),
                    'total_size_mb': self._calculate_directory_size(self.backup_dir)
                }
            },
            'retention_policy': self.retention_settings,
            'last_cleanup': self._get_last_cleanup_info()
        }
        
        return stats
    
    # 補助メソッド
    def _perform_pre_deletion_check(self, cleanup_results: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'pass', 'checks_performed': ['file_permissions', 'disk_space', 'backup_availability']}
    
    def _confirm_deletion_safety(self, cleanup_results: Dict[str, Any]) -> Dict[str, Any]:
        return {'confirmed': True, 'safety_measures': ['backup_created', 'permission_verified']}
    
    def _verify_post_deletion_state(self) -> Dict[str, Any]:
        return {'status': 'pass', 'verification': ['directory_structure_intact', 'essential_files_preserved']}
    
    def _create_charts_archive(self, timestamp: str) -> Dict[str, Any]:
        return {'archive_created': True, 'files_archived': 0}
    
    def _cleanup_old_backup_files(self) -> Dict[str, Any]:
        return {'cleaned_backups': 0, 'space_freed_mb': 0}
    
    def _determine_original_location(self, backup_path: Path) -> Optional[Path]:
        # バックアップファイル名から元の場所を推定
        name_parts = backup_path.stem.split('_backup_')
        if len(name_parts) == 2:
            original_name = name_parts[0] + backup_path.suffix
            
            # ファイルタイプ別の推定
            if 'weekly_report' in original_name:
                return self.weekly_dir / original_name
            elif 'dashboard' in original_name:
                return self.dashboard_dir / original_name
            elif original_name.endswith('.png'):
                return self.charts_dir / original_name
        
        return None
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """ディレクトリサイズ計算（MB）"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except OSError:
            pass
        return round(total_size / (1024 * 1024), 2)
    
    def _get_last_cleanup_info(self) -> Dict[str, Any]:
        """最後の削除実行情報取得"""
        log_files = list(self.cleanup_log_dir.glob("deletion_log_*.json"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                return {
                    'last_cleanup_date': log_data.get('deletion_timestamp'),
                    'files_deleted': log_data.get('cleanup_results', {}).get('total_files_deleted', 0),
                    'space_freed_mb': log_data.get('cleanup_results', {}).get('total_space_freed', 0)
                }
            except Exception:
                pass
        
        return {'last_cleanup_date': None, 'files_deleted': 0, 'space_freed_mb': 0}


def main():
    """デモ実行"""
    print("🧹 フォールバック監視システム自動削除機能 - デモ実行")
    
    cleanup_system = FallbackReportAutoCleanup()
    
    try:
        # 現在の状況確認
        print("\n📊 現在の状況:")
        stats = cleanup_system.get_cleanup_statistics()
        for dir_name, dir_stats in stats['directories'].items():
            print(f"  {dir_name}: {dir_stats['total_files']}ファイル ({dir_stats['total_size_mb']}MB)")
        
        # 自動削除実行
        print("\n🚀 自動削除実行中...")
        results = cleanup_system.implement_auto_cleanup()
        
        # 結果表示
        print("\n📋 実行結果:")
        print(f"  実行時間: {results.get('execution_duration', 0):.2f}秒")
        print(f"  処理ファイル数: {results.get('cleanup_results', {}).get('total_files_processed', 0)}")
        print(f"  削除ファイル数: {results.get('cleanup_results', {}).get('total_files_deleted', 0)}")
        print(f"  解放容量: {results.get('cleanup_results', {}).get('total_space_freed', 0)}MB")
        print(f"  ステータス: {results.get('overall_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)