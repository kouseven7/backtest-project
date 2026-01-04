"""
DSSMS統合システムのimport不一致問題を修正

問題: dssms_integrated_main.pyで間違ったクラス名をインポートしているため、
DSS Core V3が利用不可能とみなされ、フォールバックも正常に動作していない。

修正: DSSBacktesterV3 → DSSBacktesterV3 (実際のクラス名に修正)
"""

import os
import sys

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath('.'))

def fix_import_issue():
    """DSSMS統合メインのimport修正"""
    
    file_path = "src/dssms/dssms_integrated_main.py"
    
    try:
        # ファイル読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 間違ったimportを修正
        old_import = "from src.dssms.dssms_backtester_v3 import DSSBacktesterV3"
        new_import = "from src.dssms.dssms_backtester_v3 import DSSBacktesterV3"
        
        # クラス名をもう一度確認
        if "DSSBacktesterV3" in content:
            print("現在のimport文を確認中...")
            
            # 実際のクラス名確認のため dssms_backtester_v3.pyを読み込み
            with open("src/dssms/dssms_backtester_v3.py", 'r', encoding='utf-8') as f:
                backtester_content = f.read()
            
            if "class DSSBacktesterV3:" in backtester_content:
                print("✓ 正しいクラス名: DSSBacktesterV3")
                actual_class_name = "DSSBacktesterV3"
            else:
                print("✗ クラス名を再確認する必要があります")
                # 実際のクラス名を探索
                import re
                class_matches = re.findall(r'class\s+(\w*DSSBacktester\w*):', backtester_content)
                if class_matches:
                    actual_class_name = class_matches[0]
                    print(f"✓ 発見された実際のクラス名: {actual_class_name}")
                else:
                    print("✗ DSSBacktesterクラスが見つかりません")
                    return False
            
            # import文を修正
            updated_content = content.replace(
                "from src.dssms.dssms_backtester_v3 import DSSBacktesterV3",
                f"from src.dssms.dssms_backtester_v3 import {actual_class_name}"
            )
            
            # クラス参照も修正
            if actual_class_name != "DSSBacktesterV3":
                updated_content = updated_content.replace(
                    "DSSBacktesterV3()",
                    f"{actual_class_name}()"
                )
            
            # ファイル書き込み
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ {file_path} のimport文を修正しました")
            return True
            
    except Exception as e:
        print(f"✗ 修正中にエラー: {e}")
        return False

def verify_import():
    """修正後のimportを検証"""
    try:
        from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
        print("✓ DSS Core V3 import成功")
        
        # インスタンス作成テスト
        dss = DSSBacktesterV3()
        print("✓ DSS Core V3 インスタンス作成成功")
        
        return True
        
    except Exception as e:
        print(f"✗ DSS Core V3 import失敗: {e}")
        return False

if __name__ == "__main__":
    print("DSSMS統合システム import修正開始")
    
    # Step 1: Import修正
    if fix_import_issue():
        print("\nStep 1: Import修正完了")
        
        # Step 2: 検証
        print("\nStep 2: Import検証開始")
        if verify_import():
            print("✅ 修正完了 - DSS Core V3が正常に利用可能になりました")
        else:
            print("❌ 検証失敗 - さらなる調査が必要です")
    else:
        print("❌ Import修正に失敗しました")