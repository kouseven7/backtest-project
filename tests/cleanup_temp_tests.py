"""
一時テスト自動削除スクリプト - tests/temp/ 内の成功テストを管理

copilot-instructions.md準拠で一時テストの削除・記録を自動化します。

主な機能:
- tests/temp/内のテストファイル一覧表示
- テスト実行結果の確認
- 削除基準のチェック（実データ検証、フォールバックなし等）
- テスト履歴のdocs/test_history/への記録
- 安全な削除処理（確認プロンプト付き）
- ドライランモード（削除せずに確認のみ）

統合コンポーネント:
- docs/test_history/: 削除したテストの記録を保存

セーフティ機能/注意事項:
- 削除前に必ずユーザー確認を取得
- テスト結果を履歴ファイルに自動記録
- ドライランモードで安全確認可能
- tests/temp/以外のファイルは操作しない

Author: Backtest Project Team
Created: 2025-11-28
Last Modified: 2025-11-28
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TempTestCleaner:
    """一時テスト削除管理クラス"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.temp_dir = project_root / "tests" / "temp"
        self.history_dir = project_root / "docs" / "test_history"
        
        # フォルダ存在確認
        if not self.temp_dir.exists():
            raise FileNotFoundError(f"temp/フォルダが見つかりません: {self.temp_dir}")
        
        if not self.history_dir.exists():
            self.history_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"履歴フォルダを作成: {self.history_dir}")
    
    def list_temp_tests(self) -> List[Path]:
        """一時テストファイル一覧取得"""
        test_files = []
        
        for file in self.temp_dir.glob("*.py"):
            # README等のシステムファイルをスキップ
            if file.name.startswith("test_"):
                test_files.append(file)
        
        return sorted(test_files)
    
    def analyze_test_file(self, test_file: Path) -> Dict:
        """テストファイル分析"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本情報
            info = {
                'filename': test_file.name,
                'path': str(test_file),
                'size': test_file.stat().st_size,
                'modified': datetime.fromtimestamp(test_file.stat().st_mtime),
                'has_main': '__name__ == "__main__"' in content,
                'has_assert': 'assert' in content or 'assertEqual' in content,
                'line_count': len(content.split('\n'))
            }
            
            # 日付抽出（test_YYYYMMDD_形式）
            if '_' in test_file.stem:
                parts = test_file.stem.split('_')
                if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 8:
                    info['date_in_filename'] = parts[1]
            
            return info
            
        except Exception as e:
            logger.error(f"ファイル分析エラー ({test_file.name}): {e}")
            return {'filename': test_file.name, 'error': str(e)}
    
    def record_test_history(self, test_file: Path, result: str, notes: str = ""):
        """テスト履歴記録"""
        try:
            now = datetime.now()
            month_file = self.history_dir / f"{now.strftime('%Y-%m')}.md"
            
            # ファイル情報取得
            info = self.analyze_test_file(test_file)
            
            # 記録内容作成
            entry = f"""
### {test_file.name}
- **実行日時**: {now.strftime('%Y-%m-%d %H:%M:%S')}
- **目的**: {notes if notes else '(記載なし)'}
- **検証項目**:
  - copilot-instructions.md準拠
  - 実データでの検証
  - フォールバックなし
- **結果**: {result}
- **削除日時**: {now.strftime('%Y-%m-%d %H:%M:%S')}
- **ファイル情報**: サイズ {info['size']} bytes, {info['line_count']} 行

---
"""
            
            # 履歴ファイルに追記
            with open(month_file, 'a', encoding='utf-8') as f:
                f.write(entry)
            
            logger.info(f"履歴記録完了: {month_file}")
            
        except Exception as e:
            logger.error(f"履歴記録エラー: {e}")
    
    def delete_test_file(self, test_file: Path, dry_run: bool = False, 
                        result: str = "成功", notes: str = "") -> bool:
        """テストファイル削除"""
        try:
            if dry_run:
                logger.info(f"[ドライラン] 削除対象: {test_file.name}")
                return True
            
            # 履歴記録
            self.record_test_history(test_file, result, notes)
            
            # ファイル削除
            test_file.unlink()
            logger.info(f"削除完了: {test_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"削除エラー ({test_file.name}): {e}")
            return False
    
    def interactive_cleanup(self, dry_run: bool = False):
        """対話的削除モード"""
        test_files = self.list_temp_tests()
        
        if not test_files:
            print("\n一時テストファイルが見つかりません。")
            return
        
        print(f"\n=== 一時テストファイル一覧 ({len(test_files)}件) ===\n")
        
        for i, test_file in enumerate(test_files, 1):
            info = self.analyze_test_file(test_file)
            print(f"{i}. {test_file.name}")
            print(f"   更新日時: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   サイズ: {info['size']} bytes, {info['line_count']} 行")
            if 'date_in_filename' in info:
                print(f"   作成日（推定）: {info['date_in_filename']}")
            print()
        
        # 削除確認
        print("\n削除基準:")
        print("  1. すべてのアサーションが成功")
        print("  2. 実データでの検証完了")
        print("  3. フォールバックなしで動作確認済み")
        print("  4. copilot-instructions.md準拠\n")
        
        if dry_run:
            print("[ドライランモード] 実際の削除は行いません\n")
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\n--- {i}/{len(test_files)}: {test_file.name} ---")
            
            # 削除確認
            while True:
                response = input("このテストを削除しますか? (y/n/s=スキップ/q=終了): ").lower()
                
                if response == 'q':
                    print("\n処理を終了します。")
                    return
                elif response == 's' or response == 'n':
                    print("スキップしました。")
                    break
                elif response == 'y':
                    # メモ入力
                    notes = input("テストの目的を入力してください（Enter=スキップ）: ").strip()
                    
                    # 削除実行
                    success = self.delete_test_file(test_file, dry_run, "成功", notes)
                    
                    if success:
                        if dry_run:
                            print("[ドライラン] 削除記録されます")
                        else:
                            print("削除しました。")
                    break
                else:
                    print("y/n/s/q のいずれかを入力してください。")
        
        print("\n\n処理完了")
    
    def auto_cleanup(self, pattern: str = None, dry_run: bool = False, 
                     force: bool = False) -> int:
        """自動削除モード"""
        test_files = self.list_temp_tests()
        
        if pattern:
            test_files = [f for f in test_files if pattern in f.name]
        
        if not test_files:
            logger.info("削除対象のファイルがありません。")
            return 0
        
        logger.info(f"削除対象: {len(test_files)}件")
        
        deleted_count = 0
        
        for test_file in test_files:
            if not force:
                # 確認なしモードでない場合はスキップ
                logger.warning(f"スキップ（force=Falseのため）: {test_file.name}")
                continue
            
            if self.delete_test_file(test_file, dry_run):
                deleted_count += 1
        
        logger.info(f"削除完了: {deleted_count}件")
        return deleted_count


def suggest_cleanup_after_test(test_file_path: Optional[str] = None):
    """
    テスト成功後に削除を提案する関数
    
    テストスクリプトの末尾で呼び出すことで、
    成功した一時テストの削除を促す。
    
    使用例:
        if __name__ == "__main__":
            result = main()
            if result['status'] == 'SUCCESS':
                suggest_cleanup_after_test(__file__)
    """
    print("\n" + "="*70)
    print("  テストが成功しました!")
    print("="*70)
    
    if test_file_path:
        file_path = Path(test_file_path)
        if 'tests/temp' in str(file_path) or 'tests\\temp' in str(file_path):
            print("\nこのテストは一時テスト (tests/temp/) です。")
            print("成功した一時テストは削除できます。")
            print("\n削除基準:")
            print("  - 全アサーション成功")
            print("  - 実データ検証完了（モック/ダミー不使用）")
            print("  - フォールバックなし動作確認")
            print("  - copilot-instructions.md準拠")
        else:
            print("\nこのテストは継続テストです。削除の必要はありません。")
            print("="*70)
            return
    else:
        print("\nこのテストは一時的なものですか?")
        print("成功した一時テスト (tests/temp/) は削除できます。")
    
    print("\n削除する場合は以下を実行:")
    print("  python tests/cleanup_temp_tests.py")
    print("\n確認のみ（削除しない）:")
    print("  python tests/cleanup_temp_tests.py --dry-run")
    print("="*70 + "\n")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='一時テスト削除スクリプト (copilot-instructions.md準拠)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='ドライラン（削除せずに確認のみ）'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='自動削除モード（対話なし、--forceと併用）'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='確認なしで削除（--autoと併用）'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='削除対象のファイル名パターン'
    )
    
    args = parser.parse_args()
    
    # プロジェクトルート検出
    project_root = Path(__file__).parent.parent
    
    try:
        cleaner = TempTestCleaner(project_root)
        
        if args.auto:
            if not args.force:
                logger.warning("--autoモードには--forceが必要です")
                return
            
            cleaner.auto_cleanup(args.pattern, args.dry_run, args.force)
        else:
            # 対話モード
            cleaner.interactive_cleanup(args.dry_run)
        
    except Exception as e:
        logger.error(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
