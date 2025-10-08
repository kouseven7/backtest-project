#!/usr/bin/env python3
"""
File Baseline Analyzer for Problem 18 Implementation
プロジェクトファイル現状調査とベースライン測定
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
from typing import Dict, List, Any

class FileBaselineAnalyzer:
    """プロジェクトファイルのベースライン調査分析"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_time = datetime.now()
        self.baseline_data: Dict[str, Any] = {
            'timestamp': self.analysis_time.isoformat(),
            'project_root': str(self.project_root),
            'total_files': 0,
            'total_size_mb': 0.0,
            'file_categories': {},
            'directory_structure': {},
            'protection_files': [],
            'cleanup_candidates': []
        }
        
        # DSSMS Core保護パターン（Problem 18設計）
        self.protected_patterns = {
            'dssms_core': [
                'src/dssms/*.py',
                'dssms_unified_output_engine.py',
                'dssms_backtester.py',
                'config/dssms/*.json'
            ],
            'essential_scripts': [
                'main.py',
                'data_fetcher.py',
                'data_processor.py'
            ],
            'config_files': [
                'config/*.py',
                'config/*.json'
            ]
        }
        
        # 整理対象候補パターン
        self.cleanup_patterns = {
            'temp_files': ['*.tmp', '*.temp', '*~', '.DS_Store'],
            'cache_files': ['__pycache__/*', '*.pyc', '*.pyo'],
            'backup_files': ['backup_*', '*_backup*', '*.bak'],
            'log_files': ['*.log', 'logs/*'],
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'output_files': ['*.png', '*.csv', '*.txt', '*.xlsx'],
            'test_files': ['*_test.py', 'test_*.py', 'conftest.py']
        }
    
    def analyze_project_files(self):
        """プロジェクト全体のファイル分析実行"""
        print(f"[SEARCH] Project 18 ファイルベースライン調査開始: {self.project_root}")
        
        # ディレクトリ走査
        for root, dirs, files in os.walk(self.project_root):
            # .git等の除外
            dirs[:] = [d for d in dirs if not d.startswith('.git')]
            
            rel_root = Path(root).relative_to(self.project_root)
            dir_info = {
                'files': len(files),
                'total_size': 0,
                'file_types': defaultdict(int)
            }
            
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    self.baseline_data['total_files'] += 1
                    self.baseline_data['total_size_mb'] += file_size / (1024 * 1024)
                    dir_info['total_size'] += file_size
                    
                    # 拡張子別分類
                    ext = file_path.suffix.lower()
                    dir_info['file_types'][ext] += 1
                    
                    # カテゴリ分類
                    self._categorize_file(file_path, rel_root)
                    
                except (OSError, PermissionError):
                    print(f"[WARNING]  ファイルアクセス失敗: {file_path}")
            
            # ディレクトリ情報保存
            if str(rel_root) != '.':
                self.baseline_data['directory_structure'][str(rel_root)] = dir_info
    
    def _categorize_file(self, file_path: Path, rel_root: Path):
        """ファイルのカテゴリ分類と保護/整理判定"""
        rel_path = rel_root / file_path.name
        file_info = {
            'path': str(rel_path),
            'size': file_path.stat().st_size,
            'ext': file_path.suffix.lower()
        }
        
        # DSSMS Core保護判定
        if self._is_protected_file(rel_path):
            self.baseline_data['protection_files'].append(file_info)
            category = 'protected'
        # 整理対象判定
        elif self._is_cleanup_candidate(rel_path):
            self.baseline_data['cleanup_candidates'].append(file_info)
            category = 'cleanup_candidate'
        else:
            category = 'standard'
        
        # カテゴリ統計更新
        if category not in self.baseline_data['file_categories']:
            self.baseline_data['file_categories'][category] = {
                'count': 0,
                'total_size': 0,
                'extensions': defaultdict(int)
            }
        
        self.baseline_data['file_categories'][category]['count'] += 1
        self.baseline_data['file_categories'][category]['total_size'] += file_info['size']
        self.baseline_data['file_categories'][category]['extensions'][file_info['ext']] += 1
    
    def _is_protected_file(self, file_path: Path) -> bool:
        """DSSMS Core保護対象ファイル判定"""
        path_str = str(file_path).replace('\\', '/')
        
        for category, patterns in self.protected_patterns.items():
            for pattern in patterns:
                # 簡易パターンマッチング
                if '*' in pattern:
                    pattern_parts = pattern.split('*')
                    if len(pattern_parts) == 2:
                        if path_str.startswith(pattern_parts[0]) and path_str.endswith(pattern_parts[1]):
                            return True
                else:
                    if path_str == pattern:
                        return True
        
        return False
    
    def _is_cleanup_candidate(self, file_path: Path) -> bool:
        """整理対象候補ファイル判定"""
        file_name = file_path.name.lower()
        path_str = str(file_path).replace('\\', '/').lower()
        
        for category, patterns in self.cleanup_patterns.items():
            for pattern in patterns:
                if '*' in pattern:
                    pattern_parts = pattern.split('*')
                    if len(pattern_parts) == 2:
                        if file_name.startswith(pattern_parts[0]) and file_name.endswith(pattern_parts[1]):
                            return True
                    elif pattern.endswith('/*'):
                        if path_str.startswith(pattern[:-2]):
                            return True
                else:
                    if file_name == pattern:
                        return True
        
        return False
    
    def generate_baseline_report(self) -> str:
        """ベースライン分析レポート生成"""
        report_lines = [
            "=" * 80,
            f"Problem 18 File Management - Baseline Analysis Report",
            f"Generated: {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "[CHART] Project Overview:",
            f"  - Total Files: {self.baseline_data['total_files']:,}",
            f"  - Total Size: {self.baseline_data['total_size_mb']:.2f} MB",
            f"  - Project Root: {self.baseline_data['project_root']}",
            "",
            "🛡️  DSSMS Core Protected Files:",
            f"  - Count: {len(self.baseline_data['protection_files'])}",
        ]
        
        # 保護ファイル詳細
        for file_info in self.baseline_data['protection_files'][:10]:
            size_kb = file_info['size'] / 1024
            report_lines.append(f"    {file_info['path']} ({size_kb:.1f} KB)")
        
        if len(self.baseline_data['protection_files']) > 10:
            report_lines.append(f"    ... and {len(self.baseline_data['protection_files']) - 10} more protected files")
        
        report_lines.extend([
            "",
            "🧹 Cleanup Candidate Files:",
            f"  - Count: {len(self.baseline_data['cleanup_candidates'])}",
        ])
        
        # 整理候補詳細
        cleanup_size_mb = sum(f['size'] for f in self.baseline_data['cleanup_candidates']) / (1024 * 1024)
        report_lines.append(f"  - Total Size: {cleanup_size_mb:.2f} MB")
        
        # カテゴリ別統計
        report_lines.extend([
            "",
            "📁 File Categories:",
        ])
        
        for category, stats in self.baseline_data['file_categories'].items():
            size_mb = stats['total_size'] / (1024 * 1024)
            report_lines.append(f"  - {category.title()}: {stats['count']} files ({size_mb:.2f} MB)")
        
        # ディレクトリ統計（上位10個）
        report_lines.extend([
            "",
            "📂 Directory Analysis (Top 10 by file count):",
        ])
        
        dir_stats = sorted(
            self.baseline_data['directory_structure'].items(),
            key=lambda x: x[1]['files'],
            reverse=True
        )[:10]
        
        for dir_path, stats in dir_stats:
            size_mb = stats['total_size'] / (1024 * 1024)
            report_lines.append(f"  - {dir_path}: {stats['files']} files ({size_mb:.2f} MB)")
        
        return "\n".join(report_lines)
    
    def save_baseline_data(self) -> str:
        """ベースラインデータのJSON保存"""
        timestamp = self.analysis_time.strftime('%Y%m%d_%H%M%S')
        output_file = self.project_root / f"file_baseline_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.baseline_data, f, indent=2, ensure_ascii=False)
        
        return str(output_file)

def main():
    """メイン実行関数"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
    
    analyzer = FileBaselineAnalyzer(project_root)
    
    print("[ROCKET] Problem 18 File Baseline Analysis Starting...")
    analyzer.analyze_project_files()
    
    # レポート生成・表示
    report = analyzer.generate_baseline_report()
    print(report)
    
    # データ保存
    json_file = analyzer.save_baseline_data()
    print(f"\n💾 Baseline data saved: {json_file}")
    
    # KPI計算
    cleanup_count = len(analyzer.baseline_data['cleanup_candidates'])
    cleanup_size_mb = sum(f['size'] for f in analyzer.baseline_data['cleanup_candidates']) / (1024 * 1024)
    
    print(f"\n[UP] Problem 18 KPI Baseline:")
    print(f"  - Cleanup Potential: {cleanup_count} files ({cleanup_size_mb:.2f} MB)")
    print(f"  - Protected Files: {len(analyzer.baseline_data['protection_files'])} files")
    print(f"  - Cleanup Ratio: {cleanup_count/analyzer.baseline_data['total_files']*100:.1f}%")

if __name__ == "__main__":
    main()