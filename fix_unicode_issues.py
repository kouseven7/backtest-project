"""
Unicode問題修正スクリプト
- 絵文字・特殊文字を ASCII 互換文字に置換
- Windows PowerShell 互換性確保
"""
import os
import re
from typing import Dict, List

class UnicodeFixEngine:
    """Unicode文字修正エンジン"""
    
    def __init__(self):
        # 絵文字→ASCII置換マップ
        self.emoji_replacements = {
            '[TARGET]': '[TARGET]',
            '[TOOL]': '[TOOL]', 
            '[LIST]': '[LIST]',
            '[ALERT]': '[ALERT]',
            '[WARNING]': '[WARNING]',
            '[OK]': '[OK]',
            '[ERROR]': '[ERROR]',
            '[CHART]': '[CHART]',
            '[MONEY]': '[MONEY]',
            '[SEARCH]': '[SEARCH]',
            '[ROCKET]': '[ROCKET]',
            '[UP]': '[UP]',
            '[DOWN]': '[DOWN]',
            '[SUCCESS]': '[SUCCESS]',
            '[FIRE]': '[FIRE]',
            '[IDEA]': '[IDEA]',
            '[TEST]': '[TEST]',
            '[FINISH]': '[FINISH]'
        }
        
        self.files_to_fix = []
        self.fixed_count = 0
    
    def scan_unicode_issues(self, directory: str = ".") -> List[str]:
        """Unicode問題のあるファイルをスキャン"""
        
        problem_files = []
        
        for root, dirs, files in os.walk(directory):
            # .git, __pycache__ 等をスキップ
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.csv')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 絵文字・特殊文字の検出
                        has_emoji = any(emoji in content for emoji in self.emoji_replacements.keys())
                        
                        if has_emoji:
                            problem_files.append(file_path)
                            
                    except Exception as e:
                        print(f"[WARNING] ファイル読み取りエラー: {file_path} - {e}")
        
        return problem_files
    
    def fix_file_unicode(self, file_path: str) -> bool:
        """個別ファイルのUnicode問題を修正"""
        
        try:
            # ファイル読み取り
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 絵文字置換
            for emoji, replacement in self.emoji_replacements.items():
                content = content.replace(emoji, replacement)
            
            # 変更があった場合のみ保存
            if content != original_content:
                # バックアップ作成
                backup_path = f"{file_path}.unicode_backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 修正版保存
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixed_count += 1
                print(f"[OK] 修正完了: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] 修正失敗: {file_path} - {e}")
            return False
    
    def execute_fix(self) -> Dict[str, any]:
        """Unicode修正の実行"""
        
        print("[SEARCH] Unicode問題のスキャン開始...")
        
        # 問題ファイルの検出
        problem_files = self.scan_unicode_issues()
        
        if not problem_files:
            return {
                'status': 'no_issues',
                'message': 'Unicode問題は検出されませんでした',
                'fixed_files': 0
            }
        
        print(f"[ALERT] {len(problem_files)}個のファイルでUnicode問題を検出")
        
        # 修正実行
        for file_path in problem_files:
            self.fix_file_unicode(file_path)
        
        return {
            'status': 'completed',
            'message': f'{self.fixed_count}個のファイルを修正しました',
            'fixed_files': self.fixed_count,
            'problem_files': len(problem_files)
        }

def fix_main_py_unicode():
    """main.pyのUnicode問題を特別に修正"""
    
    main_files = ['main.py', 'src/main.py']
    
    # 絵文字→ASCII置換マップ
    emoji_map = {
        '[TARGET]': '[TARGET]',
        '[TOOL]': '[TOOL]', 
        '[LIST]': '[LIST]',
        '[ALERT]': '[ALERT]',
        '[WARNING]': '[WARNING]',
        '[OK]': '[OK]',
        '[ERROR]': '[ERROR]',
        '[CHART]': '[CHART]',
        '[MONEY]': '[MONEY]',
        '[SEARCH]': '[SEARCH]',
        '[ROCKET]': '[ROCKET]',
        '': '[UP]',
        '[DOWN]': '[DOWN]',
        '[SUCCESS]': '[SUCCESS]',
        '[FIRE]': '[FIRE]',
        '�': '[IDEA]',
        '[TEST]': '[TEST]',
        '[FINISH]': '[FINISH]'
    }
    
    for main_file in main_files:
        if os.path.exists(main_file):
            print(f"[TOOL] {main_file} のUnicode修正...")
            
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 絵文字を順次置換
                for emoji, replacement in emoji_map.items():
                    content = content.replace(emoji, replacement)
                
                # 修正版保存
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"[OK] {main_file} Unicode修正完了")
                
            except Exception as e:
                print(f"[ERROR] {main_file} 修正失敗: {e}")

if __name__ == "__main__":
    print("=== Unicode問題修正開始 ===")
    
    # メイン修正エンジン実行
    fixer = UnicodeFixEngine()
    result = fixer.execute_fix()
    
    # main.py特別修正
    fix_main_py_unicode()
    
    print(f"\n=== 修正結果 ===")
    print(f"ステータス: {result['status']}")
    print(f"メッセージ: {result['message']}")
    
    if result['status'] == 'completed':
        print(f"修正完了ファイル数: {result['fixed_files']}")
        print(f"問題検出ファイル数: {result['problem_files']}")
        
        print(f"\n[TOOL] 次のステップ:")
        print(f"1. python main.py でテスト実行")
        print(f"2. 出力が正常表示されることを確認")
        print(f"3. バックアップファイル(.unicode_backup)の削除")
    
    print("=== Unicode修正完了 ===")