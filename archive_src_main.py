#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/main.py アーカイブ処理
無意味な重複ファイルを整理し、システムをシンプルにする
"""
import os
import shutil
from datetime import datetime

def archive_src_main():
    """src/main.pyをアーカイブして整理"""
    
    print("=== 無意味な重複ファイル整理 ===")
    
    if not os.path.exists("src/main.py"):
        print("[INFO] src/main.py は既に存在しません")
        return True
    
    # アーカイブディレクトリ作成
    archive_dir = "archived_duplicate_files"
    os.makedirs(archive_dir, exist_ok=True)
    
    # タイムスタンプ付きでアーカイブ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"{archive_dir}/main_src_archived_{timestamp}.py"
    
    # アーカイブ理由をREADMEに記録
    readme_path = f"{archive_dir}/README.md"
    readme_content = f"""# アーカイブされた重複ファイル

## src/main.py ({timestamp})

### アーカイブ理由
- ルートmain.pyと機能重複
- パス問題により正常動作しない
- 使用箇所なし（他ファイルからの参照なし）
- 保守の必要性なし
- 混乱の原因のみ

### 対処方針
**システムはルートmain.pyのみで動作します**

### 注意
ただし、現在のルートmain.pyにも以下の重大な問題があります：
- 110件すべてが6438.89で強制決済（戦略機能していない）
- 利益計算ロジック破綻
- 出力データの品質不良

これらの問題解決が優先課題です。
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # アーカイブ実行
    shutil.move("src/main.py", archive_path)
    
    print(f"[OK] src/main.py をアーカイブしました: {archive_path}")
    print(f"[OK] アーカイブ理由をREADMEに記録: {readme_path}")
    print("[WARNING] しかし、ルートmain.pyにも重大な問題があります")
    print("         - 全110件が6438.89で強制決済")
    print("         - 戦略ロジックが機能していない")
    print("         - 出力データが不正")
    print("[NEXT] 次は実際の問題（強制決済・計算ロジック）の修正が必要")
    
    return True

def verify_system_state():
    """システム状態の確認"""
    print("\n=== システム状態確認 ===")
    
    # ルートmain.pyの存在確認
    if os.path.exists("main.py"):
        print("[OK] ルートmain.py 存在確認済み")
    else:
        print("[ERROR] ルートmain.py が存在しません")
        return False
    
    # src/main.pyの不存在確認
    if not os.path.exists("src/main.py"):
        print("[OK] src/main.py 正常にアーカイブ済み")
    else:
        print("[WARNING] src/main.py がまだ存在します")
    
    # アーカイブ確認
    archive_dir = "archived_duplicate_files"
    if os.path.exists(archive_dir):
        archived_files = [f for f in os.listdir(archive_dir) if f.startswith("main_src_archived")]
        print(f"[OK] アーカイブファイル数: {len(archived_files)}")
    
    return True

if __name__ == "__main__":
    # アーカイブ実行
    success = archive_src_main()
    
    if success:
        # システム状態確認
        verify_system_state()
        
        print("\n=== 整理完了 ===")
        print("[OK] システムは単一main.pyで動作します")
        print("[CRITICAL] ただし、main.pyの実際の問題修正が急務です")
    else:
        print("[ERROR] アーカイブ処理失敗")