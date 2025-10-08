#!/usr/bin/env python3
"""
DSSMSバックテスターExcel出力機能 - 簡易デモ

既存のDSSMSテストスクリプトを活用してExcel出力機能をテストする
"""

import sys
import os
from pathlib import Path
import logging

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def run_excel_output_demo():
    """既存のDSSMSデモスクリプトを実行してExcel出力をテスト"""
    print("DSSMSバックテスターExcel出力デモ")
    print("=" * 50)
    
    try:
        # 既存のDSSMSデモスクリプトが存在するか確認
        demo_scripts = [
            "demo_balanced_dssms.py",
            "demo_dssms_enhanced_reporting.py",
            "demo_dssms_phase1.py",
            "demo_dssms_phase2.py"
        ]
        
        available_demo = None
        for script in demo_scripts:
            if os.path.exists(script):
                available_demo = script
                break
        
        if available_demo:
            print(f"使用可能なデモスクリプト: {available_demo}")
            
            # Pythonコマンドで実行
            import subprocess
            result = subprocess.run([
                sys.executable, available_demo
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ デモ実行成功")
                print("\n=== 実行ログ ===")
                print(result.stdout)
                if result.stderr:
                    print("\n=== 警告/エラー ===")
                    print(result.stderr)
            else:
                print(f"❌ デモ実行失敗 (終了コード: {result.returncode})")
                print("\n=== エラー出力 ===")
                print(result.stderr)
                
        else:
            print("❌ 利用可能なDSSMSデモスクリプトが見つかりません")
            print("以下のファイルを確認してください:")
            for script in demo_scripts:
                print(f"  - {script}")
    
    except subprocess.TimeoutExpired:
        print("❌ デモ実行がタイムアウトしました（5分）")
        
    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")

if __name__ == "__main__":
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: run_excel_output_demo()
    
    print("\n" + "=" * 50)
    print("デモ完了")
    print("\nExcel出力ファイルは以下のディレクトリに保存されます:")
    print("  - backtest_results/dssms_results/")
    print("  - test_output/dssms_excel_test/")
