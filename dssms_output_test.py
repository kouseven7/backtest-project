"""
DSSMS出力統合検証デモ

2026-01-08: 10ファイル統合が正しく実行されることを検証
短期間テストでファイル生成を確認

Author: Backtest Project Team
Created: 2026-01-08
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.append(project_root)

# DSSMS統合システムをインポート
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def run_demo():
    """DSSMS短期テストを実行"""
    print("=" * 60)
    print("DSSMS出力統合検証デモ")
    print("=" * 60)
    print(f"開始時刻: {datetime.now()}")
    
    try:
        # DSSMS統合システム初期化
        dssms_system = DSSMSIntegratedBacktester()
        print("[✓] DSSMSシステム初期化完了")
        
        # 短期間での実行（5営業日程度）
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        print(f"[実行] バックテスト期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print("=" * 40)
        
        # バックテスト実行
        results = dssms_system.run_dynamic_backtest(
            start_date=start_date,
            end_date=end_date
        )
        
        print("=" * 40)
        print("[✓] バックテスト実行完了")
        
        # 結果確認
        if results:
            print(f"結果タイプ: {type(results)}")
            if isinstance(results, dict):
                print(f"結果キー: {list(results.keys())}")
        
        print(f"完了時刻: {datetime.now()}")
        print("=" * 60)
        print("出力フォルダを確認してください:")
        print("output/dssms_integration/ 配下のdssms_{timestamp}フォルダ")
        print("10ファイルが生成されているはずです")
        
    except Exception as e:
        print(f"[エラー] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()