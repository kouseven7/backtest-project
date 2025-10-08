"""
安全なUnicode修正スクリプト - 最小限の修正のみ実行
"""
import os
import re

def fix_main_files_safely():
    """main.pyファイルのみを安全に修正"""
    
    # 絵文字→ASCII置換マップ（厳選）
    emoji_replacements = {
        '🎯': '[TARGET]',
        '🔧': '[TOOL]',
        '📋': '[LIST]',
        '🚨': '[ALERT]',
        '⚠️': '[WARNING]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '📊': '[CHART]',
        '💰': '[MONEY]',
        '🔍': '[SEARCH]',
        '🚀': '[ROCKET]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '🎉': '[SUCCESS]',
        '🔥': '[FIRE]',
        '💡': '[IDEA]',
        '🧪': '[TEST]',
        '🏁': '[FINISH]'
    }
    
    main_files = ['main.py', 'src/main.py']
    
    for main_file in main_files:
        if os.path.exists(main_file):
            print(f"[TOOL] {main_file} の安全なUnicode修正...")
            
            try:
                # ファイル読み取り
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 絵文字のみを置換（文字列の境界を確認）
                for emoji, replacement in emoji_replacements.items():
                    if emoji in content:
                        content = content.replace(emoji, replacement)
                        print(f"    置換: {emoji} -> {replacement}")
                
                # 変更があった場合のみ保存
                if content != original_content:
                    # バックアップ作成
                    backup_path = f"{main_file}.safe_backup"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    # 修正版保存
                    with open(main_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"[OK] {main_file} 修正完了")
                else:
                    print(f"[INFO] {main_file} 修正不要")
                
            except Exception as e:
                print(f"[ERROR] {main_file} 修正失敗: {e}")

if __name__ == "__main__":
    print("=== 安全なUnicode修正開始 ===")
    fix_main_files_safely()
    print("=== 修正完了 ===")