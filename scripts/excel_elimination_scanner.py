"""
Phase 2.5: Excel出力完全撲滅スキャナー
集中的Excel出力検出・削除・アーカイブツール

作成日: 2025年10月8日
目的: プロジェクト全体のExcel出力を一括検出・削除・アーカイブ
特徴: バックテスト基本理念遵守チェック、自動バックアップ、詳細報告書生成
"""

import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import sys

# ロガー設定
logger = logging.getLogger(__name__)

class ExcelEliminationScanner:
    """Excel出力を集中的に検出・削除するスキャナー"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # scriptsディレクトリから実行されることを想定
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent  # scripts/ から ../
        else:
            self.project_root = project_root
            
        self.archive_dir = self.project_root / "archived_excel_outputs"
        self.scan_results: List[Dict[str, Any]] = []
        self.elimination_log: List[Dict[str, Any]] = []
        
        # アーカイブディレクトリ作成
        self.archive_dir.mkdir(exist_ok=True)
        
        # Excel出力パターン定義（バックテスト基本理念考慮版）
        self.excel_patterns = {
            'pandas_to_excel': r'\.to_excel\s*\(',
            'openpyxl_save': r'\.save\s*\(\s*[\'"][^\'"]*.xlsx[\'"]',
            'xlsxwriter_workbook': r'xlsxwriter\.Workbook\s*\(',
            'excel_file_creation': r'[\'"][^\'"]*.xlsx[\'"].*=.*open|with.*open.*[\'"][^\'"]*.xlsx[\'"]',
            'excel_output_paths': r'output.*\.xlsx|excel_output|excel_results',
            'excel_writer_patterns': r'ExcelWriter|pd\.ExcelWriter',
            'workbook_patterns': r'openpyxl\.Workbook|load_workbook.*save'
        }
        
        # 保護対象（読み取り専用）
        self.protected_patterns = {
            'read_excel': r'pd\.read_excel|pandas\.read_excel',
            'config_files': r'config/.*\.xlsx|stock_list\.xlsx',
            'input_data': r'input.*\.xlsx',
            'load_workbook_read': r'load_workbook.*mode.*r'
        }
        
        logger.info(f"Excel撲滅スキャナー初期化完了: {self.project_root}")
        logger.info(f"アーカイブディレクトリ: {self.archive_dir}")
    
    def scan_project_for_excel_outputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """プロジェクト全体のExcel出力スキャン"""
        
        logger.info("🔍 Excel出力スキャン開始...")
        
        scan_results: Dict[str, List[Dict[str, Any]]] = {
            'violations': [],
            'protected': [],
            'suspicious': []
        }
        
        # Python ファイルをスキャン
        python_files = list(self.project_root.rglob("*.py"))
        logger.info(f"📂 スキャン対象Pythonファイル数: {len(python_files)}")
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            violations = self._scan_file_for_excel_violations(py_file)
            if violations:
                scan_results['violations'].extend(violations)
        
        # 設定ファイル・その他をスキャン
        config_files = list(self.project_root.rglob("*.json"))
        logger.info(f"📄 スキャン対象設定ファイル数: {len(config_files)}")
        
        for config_file in config_files:
            violations = self._scan_config_for_excel_paths(config_file)
            if violations:
                scan_results['suspicious'].extend(violations)
        
        logger.info(f"📊 スキャン完了: 違反{len(scan_results['violations'])}件, "
                   f"疑わしい{len(scan_results['suspicious'])}件, "
                   f"保護対象{len(scan_results['protected'])}件")
        
        # バックテスト基本理念への影響チェック
        self._check_backtest_principle_impact(scan_results['violations'])
        
        return scan_results
    
    def _scan_file_for_excel_violations(self, file_path: Path) -> List[Dict[str, Any]]:
        """単一ファイルのExcel出力違反検出"""
        violations: List[Dict[str, Any]] = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # 保護対象チェック（スキップ）
                if self._is_protected_excel_operation(line):
                    continue
                
                # 違反パターンチェック
                for pattern_name, pattern in self.excel_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append({
                            'file': file_path,
                            'line_number': line_num,
                            'line_content': line.strip(),
                            'violation_type': pattern_name,
                            'severity': self._determine_severity(pattern_name, line),
                            'backtest_impact': self._assess_backtest_impact(line, file_path)
                        })
        
        except Exception as e:
            logger.warning(f"ファイル読み取りエラー {file_path}: {e}")
        
        return violations
    
    def _scan_config_for_excel_paths(self, config_path: Path) -> List[Dict[str, Any]]:
        """設定ファイルのExcelパス検出"""
        violations: List[Dict[str, Any]] = []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # .xlsx パス検出
            xlsx_paths = re.findall(r'[\'"][^\'"]*\.xlsx[\'"]', content)
            for xlsx_path in xlsx_paths:
                if 'output' in xlsx_path.lower() or 'result' in xlsx_path.lower():
                    violations.append({
                        'file': config_path,
                        'line_number': 0,
                        'line_content': xlsx_path,
                        'violation_type': 'config_excel_path',
                        'severity': 'MEDIUM',
                        'backtest_impact': 'CONFIG'
                    })
        
        except Exception as e:
            logger.warning(f"設定ファイル読み取りエラー {config_path}: {e}")
        
        return violations
    
    def eliminate_excel_outputs_batch(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Excel出力の一括削除・アーカイブ"""
        
        logger.info("🗑️ Excel出力一括削除開始...")
        
        elimination_stats = {
            'files_processed': 0,
            'lines_commented': 0,
            'files_archived': 0,
            'errors': 0,
            'backtest_impacted': 0
        }
        
        # ファイル別にグループ化
        files_to_process: Dict[Path, List[Dict[str, Any]]] = {}
        for violation in violations:
            file_path = violation['file']
            if file_path not in files_to_process:
                files_to_process[file_path] = []
            files_to_process[file_path].append(violation)
        
        # ファイルごとに処理
        for file_path, file_violations in files_to_process.items():
            try:
                result = self._eliminate_file_excel_outputs(file_path, file_violations)
                elimination_stats['files_processed'] += 1
                elimination_stats['lines_commented'] += result['lines_modified']
                
                if result['archived']:
                    elimination_stats['files_archived'] += 1
                
                # バックテスト影響チェック
                if any(v.get('backtest_impact') not in ['NONE', 'CONFIG'] for v in file_violations):
                    elimination_stats['backtest_impacted'] += 1
                
            except Exception as e:
                logger.error(f"ファイル処理エラー {file_path}: {e}")
                elimination_stats['errors'] += 1
        
        logger.info(f"✅ Excel出力削除完了: {elimination_stats}")
        return elimination_stats
    
    def _eliminate_file_excel_outputs(self, file_path: Path, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """単一ファイルのExcel出力削除"""
        
        # 元ファイルのバックアップ
        backup_path = self.archive_dir / f"{file_path.name}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        
        # ファイル読み取り
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 違反行をコメントアウト
        modified_lines = 0
        for violation in violations:
            line_num = violation['line_number'] - 1  # 0-indexed
            if 0 <= line_num < len(lines):
                original_line: str = lines[line_num]
                
                # バックテスト基本理念への影響を考慮したコメント
                impact_note = ""
                if violation.get('backtest_impact') == 'HIGH':
                    impact_note = " # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected"
                elif violation.get('backtest_impact') == 'MEDIUM':
                    impact_note = " # BACKTEST_IMPACT: Trading data output affected"
                
                # コメントアウト + TODO追加
                commented_line = (
                    f"# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08){impact_note}\n"
                    f"# ORIGINAL: {original_line.strip()}\n"
                )
                lines[line_num] = commented_line
                modified_lines += 1
                
                # ログ記録
                self.elimination_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'file': str(file_path),
                    'line': violation['line_number'],
                    'original': original_line.strip(),
                    'violation_type': violation['violation_type'],
                    'backtest_impact': violation.get('backtest_impact', 'UNKNOWN'),
                    'action': 'COMMENTED_OUT'
                })
        
        # 修正版ファイル書き戻し
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"📝 {file_path}: {modified_lines}行をコメントアウト")
        
        return {
            'lines_modified': modified_lines,
            'backup_path': backup_path,
            'archived': True
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """スキャン対象外ファイルの判定"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            '.venv',
            'env',
            '.pytest_cache',
            'archived_excel_outputs',  # アーカイブディレクトリ
            'excel_elimination_scanner.py'  # 自分自身
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _is_protected_excel_operation(self, line: str) -> bool:
        """保護対象Excel操作の判定"""
        return any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in self.protected_patterns.values()
        )
    
    def _determine_severity(self, pattern_name: str, line: str) -> str:
        """違反の重要度判定"""
        if 'to_excel' in pattern_name or 'save' in pattern_name:
            return 'HIGH'
        elif 'workbook' in pattern_name.lower():
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_backtest_impact(self, line: str, file_path: Path) -> str:
        """バックテスト基本理念への影響評価"""
        # バックテスト関連キーワードチェック
        backtest_keywords = [
            'Entry_Signal', 'Exit_Signal', 'backtest', 'trades', 
            'performance', 'portfolio', 'strategy'
        ]
        
        high_impact_files = [
            'main.py', 'dssms', 'strategy', 'backtest'
        ]
        
        # ファイル名チェック
        if any(keyword in str(file_path).lower() for keyword in high_impact_files):
            if any(keyword in line for keyword in backtest_keywords):
                return 'HIGH'
            else:
                return 'MEDIUM'
        
        # 行内容チェック
        if any(keyword in line for keyword in backtest_keywords):
            return 'MEDIUM'
        
        return 'NONE'
    
    def _check_backtest_principle_impact(self, violations: List[Dict[str, Any]]):
        """バックテスト基本理念への影響チェック"""
        high_impact_count = sum(1 for v in violations if v.get('backtest_impact') == 'HIGH')
        medium_impact_count = sum(1 for v in violations if v.get('backtest_impact') == 'MEDIUM')
        
        if high_impact_count > 0:
            logger.warning(f"⚠️ バックテスト基本理念への高影響違反: {high_impact_count}件")
        
        if medium_impact_count > 0:
            logger.info(f"ℹ️ バックテスト基本理念への中影響違反: {medium_impact_count}件")
    
    def generate_elimination_report(self) -> str:
        """Excel削除報告書生成"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 統計情報
        total_files = len(set(log['file'] for log in self.elimination_log))
        total_lines = len(self.elimination_log)
        backtest_high_impact = sum(1 for log in self.elimination_log if log.get('backtest_impact') == 'HIGH')
        backtest_medium_impact = sum(1 for log in self.elimination_log if log.get('backtest_impact') == 'MEDIUM')
        
        report = f"""
==========================================
Excel出力完全撲滅報告書
実行日時: {timestamp}
==========================================

📊 撲滅統計:
- 処理ファイル数: {total_files}
- 削除コード行数: {total_lines}
- アーカイブファイル数: {len(list(self.archive_dir.glob('*.backup.*')))}

⚠️ バックテスト基本理念への影響:
- 高影響(Entry_Signal/Exit_Signal関連): {backtest_high_impact}件
- 中影響(取引データ関連): {backtest_medium_impact}件
- 低影響・なし: {total_lines - backtest_high_impact - backtest_medium_impact}件

🗑️ 削除詳細:
"""
        
        # ファイル別削除詳細
        files_processed: Dict[str, List[Dict[str, Any]]] = {}
        for log_entry in self.elimination_log:
            file_name = log_entry['file']
            if file_name not in files_processed:
                files_processed[file_name] = []
            files_processed[file_name].append(log_entry)
        
        for file_name, logs in files_processed.items():
            report += f"""
📁 ファイル: {file_name}
削除行数: {len(logs)}行
"""
            for log in logs[:3]:  # 最初の3行のみ表示
                report += f"  L{log['line']}: {log['violation_type']} - {log['original'][:50]}...\n"
            
            if len(logs) > 3:
                report += f"  ... 他 {len(logs) - 3} 行\n"
            report += "---\n"
        
        report += f"""

✅ 次回フェーズ:
1. 新形式出力テスト実行
2. CSV+JSON+TXT+YAML動作確認
3. バックテスト基本理念遵守確認

🔧 バックテスト基本理念対応:
- 高影響違反は統一出力エンジンで代替実装推奨
- Entry_Signal/Exit_Signal出力の完整性確保必須
- 取引履歴データの新形式移行必須

🔒 アーカイブ場所: {self.archive_dir}
復元方法: backup ファイルから手動復元可能

==========================================
Excel出力完全廃棄完了 ✨
==========================================
"""
        
        return report


def execute_excel_elimination_phase():
    """Phase 2.5: Excel出力完全撲滅の実行"""
    
    print("🚀 Phase 2.5: Excel出力完全撲滅開始")
    print("=" * 50)
    
    try:
        scanner = ExcelEliminationScanner()
        
        # 1. スキャン実行
        print("🔍 プロジェクト全体スキャン中...")
        scan_results = scanner.scan_project_for_excel_outputs()
        
        print(f"\n📋 検出結果:")
        print(f"  - Excel出力違反: {len(scan_results['violations'])}件")
        print(f"  - 疑わしい設定: {len(scan_results['suspicious'])}件")
        print(f"  - 保護対象: {len(scan_results['protected'])}件")
        
        if not scan_results['violations'] and not scan_results['suspicious']:
            print("\n✅ Excel出力違反なし - 撲滅完了済み")
            return
        
        # 2. 検出内容の詳細表示
        all_violations = scan_results['violations'] + scan_results['suspicious']
        
        print(f"\n🔍 検出されたExcel出力 (上位{min(10, len(all_violations))}件):")
        for i, violation in enumerate(all_violations[:10], 1):
            file_name = Path(violation['file']).name
            impact = violation.get('backtest_impact', 'UNKNOWN')
            print(f"  {i}. {file_name}:{violation['line_number']} - {violation['violation_type']} (影響:{impact})")
        
        if len(all_violations) > 10:
            print(f"  ... 他 {len(all_violations) - 10} 件")
        
        # 3. 削除実行の確認
        print(f"\n⚠️ 重要: {len(all_violations)}件のExcel出力を検出しました")
        print("以下の操作が実行されます:")
        print("  1. 違反コードのコメントアウト")
        print("  2. 元ファイルの自動バックアップ")
        print("  3. TODO タグ付与")
        print("  4. 詳細報告書生成")
        
        confirm = input(f"\n続行しますか？ (y/N): ")
        if confirm.lower() != 'y':
            print("❌ 処理をキャンセルしました")
            return
        
        # 4. 一括削除実行
        print("\n🗑️ Excel出力削除実行中...")
        elimination_stats = scanner.eliminate_excel_outputs_batch(all_violations)
        
        # 5. 報告書生成
        print("📝 報告書生成中...")
        report = scanner.generate_elimination_report()
        
        # 報告書保存
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / f"excel_elimination_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 結果表示
        print(f"\n✅ Excel出力完全撲滅完了!")
        print("=" * 50)
        print(f"📊 処理結果:")
        print(f"  - 処理ファイル数: {elimination_stats['files_processed']}")
        print(f"  - 削除行数: {elimination_stats['lines_commented']}")
        print(f"  - アーカイブファイル数: {elimination_stats['files_archived']}")
        print(f"  - バックテスト影響ファイル数: {elimination_stats.get('backtest_impacted', 0)}")
        print(f"  - エラー数: {elimination_stats['errors']}")
        print(f"\n📄 詳細報告書: {report_file}")
        print(f"📦 アーカイブ場所: {scanner.archive_dir}")
        
        # 次のステップガイド
        print(f"\n🚀 次のステップ:")
        print("  1. 新形式出力エンジンでの代替実装")
        print("  2. バックテスト基本理念遵守確認")
        print("  3. CSV+JSON+TXT+YAML形式での動作テスト")
        
    except Exception as e:
        logger.error(f"Excel撲滅処理エラー: {e}")
        print(f"❌ エラーが発生しました: {e}")
        print("詳細はログを確認してください")
        raise


if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/excel_elimination.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    execute_excel_elimination_phase()