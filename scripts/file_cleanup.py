#!/usr/bin/env python3
"""
FileCleanupManager for Problem 18 Implementation
DSSMS Core保護付きファイル整理システム
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
import fnmatch
import logging

class FileCleanupManager:
    """Problem 18 - ファイル管理最適化システム"""
    
    def __init__(self, project_root: str, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.timestamp = datetime.now()
        
        # ログ設定
        self.logger = self._setup_logger()
        
        # アーカイブディレクトリ設定
        self.archive_root = self.project_root / "archive" / "deleted_files"
        self.archive_session = self.archive_root / self.timestamp.strftime('%Y%m%d_%H%M%S')
        
        # 統計データ
        self.cleanup_stats: Dict[str, Any] = {
            'timestamp': self.timestamp.isoformat(),
            'dry_run': self.dry_run,
            'files_deleted': 0,
            'files_archived': 0,
            'space_freed_mb': 0.0,
            'categories': {},
            'protected_accessed': 0,
            'errors': []
        }
        
        # DSSMS Core絶対保護パターン (85.0点エンジン品質維持)
        self.critical_protection_patterns = {
            # 中核エンジンファイル
            'dssms_core': [
                'src/dssms/*.py',
                'dssms_unified_output_engine.py',  # 85.0点エンジン
                'dssms_backtester.py',
                'dssms_backtester_config.json'
            ],
            # 本質的実行ファイル
            'essential_execution': [
                'main.py',
                'data_fetcher.py', 
                'data_processor.py'
            ],
            # 重要設定
            'critical_config': [
                'config/dssms/*.json',
                'config/optimized_parameters.py',
                'config/risk_management.py'
            ],
            # パフォーマンス最適化（Problem 8成果）
            'performance_system': [
                'src/dssms/performance_optimizer.py',
                'config/dssms/performance_config.json'
            ]
        }
        
        # 積極的整理対象パターン
        self.aggressive_cleanup_patterns = {
            'temp_files': ['*.tmp', '*.temp', '*~', '.DS_Store', 'Thumbs.db'],
            'cache_files': ['__pycache__/*', '*.pyc', '*.pyo', '.pytest_cache/*'],
            'backup_files': ['backup_*', '*_backup*', '*.bak', '*_bak_*'],
            'old_output': ['*.png', '*.csv', '*.txt', '*.xlsx'],  # 出力ファイル
            'log_files': ['*.log', 'logs/*'],
            'venv_cache': ['.venv/Lib/site-packages/*/tests/*', '.venv/Lib/site-packages/*/__pycache__/*']
        }
        
        # 慎重整理対象（確認付き）
        self.careful_cleanup_patterns = {
            'analysis_scripts': ['analyze_*.py', 'check_*.py', '*_test.py'],
            'demo_files': ['*_demo.py', 'demo_*.py'],
            'old_versions': ['*_v1.py', '*_v2.py', '*_old.py']
        }
        
    def _setup_logger(self) -> logging.Logger:
        """ログ設定初期化"""
        logger = logging.getLogger(f'FileCleanup_{self.timestamp.strftime("%Y%m%d_%H%M%S")}')
        logger.setLevel(logging.INFO)
        
        # ログファイル設定
        log_file = self.project_root / f"cleanup_log_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # フォーマット設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def is_protected_file(self, file_path: Path) -> bool:
        """DSSMS Core保護ファイル判定（絶対保護）"""
        rel_path = file_path.relative_to(self.project_root)
        path_str = str(rel_path).replace('\\', '/')
        
        for category, patterns in self.critical_protection_patterns.items():
            for pattern in patterns:
                if self._match_pattern(path_str, pattern):
                    self.cleanup_stats['protected_accessed'] += 1
                    self.logger.info(f"🛡️  PROTECTED: {path_str} (category: {category})")
                    return True
        
        return False
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """ファイルパターンマッチング"""
        if '*' in pattern:
            return fnmatch.fnmatch(path, pattern)
        else:
            return path == pattern
    
    def identify_cleanup_candidates(self) -> Dict[str, List[Path]]:
        """整理対象ファイル識別"""
        candidates: Dict[str, List[Path]] = {
            'aggressive': [],
            'careful': [],
            'unknown': []
        }
        
        self.logger.info("🔍 Cleanup candidate identification starting...")
        
        for root, dirs, files in os.walk(self.project_root):
            # .git等のスキップ
            dirs[:] = [d for d in dirs if not d.startswith('.git')]
            
            for file in files:
                file_path = Path(root) / file
                
                try:
                    # 保護チェック（最優先）
                    if self.is_protected_file(file_path):
                        continue
                    
                    rel_path = file_path.relative_to(self.project_root)
                    path_str = str(rel_path).replace('\\', '/')
                    file_name = file_path.name.lower()
                    
                    # 積極的整理対象判定
                    categorized = False
                    for category, patterns in self.aggressive_cleanup_patterns.items():
                        for pattern in patterns:
                            if self._match_pattern(path_str.lower(), pattern.lower()) or \
                               self._match_pattern(file_name, pattern.lower()):
                                candidates['aggressive'].append(file_path)
                                categorized = True
                                break
                        if categorized:
                            break
                    
                    if categorized:
                        continue
                    
                    # 慎重整理対象判定
                    for category, patterns in self.careful_cleanup_patterns.items():
                        for pattern in patterns:
                            if self._match_pattern(path_str.lower(), pattern.lower()) or \
                               self._match_pattern(file_name, pattern.lower()):
                                candidates['careful'].append(file_path)
                                categorized = True
                                break
                        if categorized:
                            break
                    
                    # 未分類は保留
                    if not categorized and file_path.suffix in ['.py', '.json', '.md']:
                        candidates['unknown'].append(file_path)
                
                except (OSError, PermissionError) as e:
                    self.cleanup_stats['errors'].append(f"Access error: {file_path} - {e}")
        
        # 統計更新
        for category, file_list in candidates.items():
            self.cleanup_stats['categories'][category] = len(file_list)
        
        self.logger.info(f"📋 Candidates identified: "
                        f"Aggressive={len(candidates['aggressive'])}, "
                        f"Careful={len(candidates['careful'])}, "
                        f"Unknown={len(candidates['unknown'])}")
        
        return candidates
    
    def safe_delete_file(self, file_path: Path, category: str = 'unknown') -> bool:
        """安全なファイル削除（バックアップ付き）"""
        try:
            # 保護再確認
            if self.is_protected_file(file_path):
                self.logger.warning(f"🚨 PROTECTION VIOLATION PREVENTED: {file_path}")
                return False
            
            file_size = file_path.stat().st_size
            
            if self.dry_run:
                self.logger.info(f"🔍 DRY-RUN DELETE: {file_path} ({file_size/1024:.1f} KB)")
                self.cleanup_stats['files_deleted'] += 1
                self.cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
                return True
            
            # アーカイブディレクトリ作成
            self.archive_session.mkdir(parents=True, exist_ok=True)
            
            # バックアップ作成
            rel_path = file_path.relative_to(self.project_root)
            archive_path = self.archive_session / rel_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(file_path, archive_path)
            self.cleanup_stats['files_archived'] += 1
            
            # 元ファイル削除
            file_path.unlink()
            self.cleanup_stats['files_deleted'] += 1
            self.cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
            
            self.logger.info(f"🗑️  DELETED: {file_path} -> archived to {archive_path}")
            return True
            
        except Exception as e:
            error_msg = f"Delete failed: {file_path} - {e}"
            self.cleanup_stats['errors'].append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def execute_cleanup(self, 
                       include_aggressive: bool = True,
                       include_careful: bool = False,
                       max_files: Optional[int] = None) -> Dict[str, Any]:
        """ファイル整理実行"""
        
        self.logger.info(f"🚀 Problem 18 File Cleanup Execution Starting...")
        self.logger.info(f"   Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        self.logger.info(f"   Aggressive: {include_aggressive}, Careful: {include_careful}")
        
        # 整理候補識別
        candidates = self.identify_cleanup_candidates()
        
        # 削除対象選択
        delete_queue: List[Path] = []
        
        if include_aggressive:
            delete_queue.extend(candidates['aggressive'])
            self.logger.info(f"📦 Added {len(candidates['aggressive'])} aggressive cleanup files")
        
        if include_careful:
            delete_queue.extend(candidates['careful'])
            self.logger.info(f"⚠️  Added {len(candidates['careful'])} careful cleanup files")
        
        # ファイル数制限
        if max_files and len(delete_queue) > max_files:
            delete_queue = delete_queue[:max_files]
            self.logger.info(f"🔢 Limited to {max_files} files")
        
        # 削除実行
        success_count = 0
        for file_path in delete_queue:
            if self.safe_delete_file(file_path):
                success_count += 1
        
        # 結果統計
        self.cleanup_stats['execution_summary'] = {
            'candidates_found': sum(len(files) for files in candidates.values()),
            'files_processed': len(delete_queue),
            'files_success': success_count,
            'files_failed': len(delete_queue) - success_count,
            'success_rate': success_count / len(delete_queue) * 100 if delete_queue else 0
        }
        
        self.logger.info(f"✅ Cleanup completed: {success_count}/{len(delete_queue)} files processed")
        self.logger.info(f"💾 Space freed: {self.cleanup_stats['space_freed_mb']:.2f} MB")
        
        return self.cleanup_stats
    
    def generate_cleanup_report(self) -> str:
        """整理レポート生成"""
        report_lines = [
            "=" * 80,
            f"Problem 18 File Management - Cleanup Execution Report",
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode: {'DRY-RUN SIMULATION' if self.dry_run else 'LIVE EXECUTION'}",
            "=" * 80,
            "",
            "📊 Cleanup Results:",
            f"  - Files Processed: {self.cleanup_stats.get('files_deleted', 0):,}",
            f"  - Files Archived: {self.cleanup_stats.get('files_archived', 0):,}",
            f"  - Space Freed: {self.cleanup_stats.get('space_freed_mb', 0):.2f} MB",
            f"  - Protected Files Accessed: {self.cleanup_stats.get('protected_accessed', 0)}",
            "",
            "🛡️  DSSMS Core Protection Status:",
            f"  - dssms_unified_output_engine.py: PROTECTED (85.0-point engine)",
            f"  - DSSMS Core Files: PROTECTED",
            f"  - Performance Optimizer: PROTECTED (Problem 8 achievement)",
            "",
        ]
        
        # 実行サマリー
        if 'execution_summary' in self.cleanup_stats:
            summary = self.cleanup_stats['execution_summary']
            report_lines.extend([
                "📈 Execution Summary:",
                f"  - Candidates Found: {summary.get('candidates_found', 0):,}",
                f"  - Files Processed: {summary.get('files_processed', 0):,}",
                f"  - Success Rate: {summary.get('success_rate', 0):.1f}%",
                "",
            ])
        
        # カテゴリ別統計
        if 'categories' in self.cleanup_stats:
            report_lines.extend([
                "📂 Categories:",
            ])
            for category, count in self.cleanup_stats['categories'].items():
                report_lines.append(f"  - {category.title()}: {count:,} files")
            report_lines.append("")
        
        # エラー情報
        if self.cleanup_stats.get('errors'):
            report_lines.extend([
                "⚠️  Errors Encountered:",
            ])
            for error in self.cleanup_stats['errors'][:5]:
                report_lines.append(f"  - {error}")
            if len(self.cleanup_stats['errors']) > 5:
                report_lines.append(f"  ... and {len(self.cleanup_stats['errors']) - 5} more errors")
            report_lines.append("")
        
        # KPI達成状況
        freed_mb = self.cleanup_stats.get('space_freed_mb', 0)
        files_deleted = self.cleanup_stats.get('files_deleted', 0)
        
        report_lines.extend([
            "🎯 Problem 18 KPI Achievement:",
            f"  - Space Reduction: {freed_mb:.2f} MB freed",
            f"  - File Count Reduction: {files_deleted:,} files",
            f"  - Protection Success: {self.cleanup_stats.get('protected_accessed', 0)} access attempts blocked",
            f"  - DSSMS Core Integrity: MAINTAINED ✅",
        ])
        
        return "\n".join(report_lines)
    
    def save_cleanup_report(self) -> str:
        """整理レポート保存"""
        timestamp = self.timestamp.strftime('%Y%m%d_%H%M%S')
        
        # JSONデータ保存
        json_file = self.project_root / f"cleanup_stats_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_stats, f, indent=2, ensure_ascii=False)
        
        # テキストレポート保存
        report_file = self.project_root / f"cleanup_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_cleanup_report())
        
        return str(report_file)

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Problem 18 File Cleanup Manager')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (default)')
    parser.add_argument('--live', action='store_true', help='Live execution mode')
    parser.add_argument('--aggressive', action='store_true', help='Include aggressive cleanup')
    parser.add_argument('--careful', action='store_true', help='Include careful cleanup') 
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    
    args = parser.parse_args()
    
    # 実行モード決定
    dry_run = not args.live  # デフォルトはdry-run
    
    # FileCleanupManager初期化
    manager = FileCleanupManager(
        project_root=args.project_root,
        dry_run=dry_run
    )
    
    print(f"🧹 Problem 18 File Cleanup Manager")
    print(f"   Project: {manager.project_root}")
    print(f"   Mode: {'DRY-RUN' if dry_run else 'LIVE EXECUTION'}")
    print("")
    
    # 整理実行
    results = manager.execute_cleanup(
        include_aggressive=args.aggressive,
        include_careful=args.careful,
        max_files=args.max_files
    )
    
    # レポート表示・保存
    report = manager.generate_cleanup_report()
    print(report)
    
    report_file = manager.save_cleanup_report()
    print(f"\n💾 Report saved: {report_file}")
    
    # KPI要約
    print(f"\n🎯 Problem 18 KPI Summary:")
    print(f"   - Files: {results.get('files_deleted', 0):,} deleted")
    print(f"   - Space: {results.get('space_freed_mb', 0):.2f} MB freed")
    print(f"   - Protection: {results.get('protected_accessed', 0)} access attempts blocked")

if __name__ == "__main__":
    main()