"""
ウォームアップ期間エントリー防止機能の検証スクリプト

Issue調査報告20260210修正の検証:
- all_transactions.csvのentry_dateが全て2024-01-01以降か確認
- system信頼性が改善されたか確認
"""

import pandas as pd
from pathlib import Path
import datetime

def verify_warmup_fix():
    """ウォームアップ期間エントリー防止機能の検証"""
    
    # 最新のDSSMS出力フォルダを取得
    output_dir = Path("output/dssms_integration")
    dssms_folders = sorted(output_dir.glob("dssms_*"), reverse=True)
    
    if not dssms_folders:
        print("❌ DSSMSフォルダが見つかりません")
        return
    
    latest_folder = dssms_folders[0]
    print(f"📁 検証対象: {latest_folder}")
    print()
    
    # all_transactions.csvを読み込み
    csv_path = latest_folder / "all_transactions.csv"
    if not csv_path.exists():
        print(f"❌ {csv_path}が見つかりません")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📊 取引件数: {len(df)}件")
    print()
    
    # entry_dateを日付型に変換
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    
    # ウォームアップ期間（2024-01-01より前）のエントリーを確認
    trading_start_date = datetime.datetime(2024, 1, 1)
    warmup_entries = df[df['entry_date'] < trading_start_date]
    normal_entries = df[df['entry_date'] >= trading_start_date]
    
    print("=" * 60)
    print("🎯 検証結果")
    print("=" * 60)
    print(f"ウォームアップ期間エントリー: {len(warmup_entries)}件")
    print(f"通常期間エントリー: {len(normal_entries)}件")
    print()
    
    if len(warmup_entries) > 0:
        print("❌ ウォームアップ期間エントリーが存在します:")
        print(warmup_entries[['symbol', 'entry_date', 'entry_price', 'strategy_name']])
    else:
        print("✅ ウォームアップ期間エントリーは0件（修正成功）")
    
    print()
    
    # 通常期間エントリーの詳細
    if len(normal_entries) > 0:
        print("通常期間エントリー詳細:")
        print(normal_entries[['symbol', 'entry_date', 'exit_date', 'pnl', 'strategy_name']])
    
    print()
    
    # system信頼性を確認
    summary_path = latest_folder / "summary.txt"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = f.read()
            if "システム信頼性" in summary:
                # システム信頼性の行を抽出
                for line in summary.split('\n'):
                    if "システム信頼性" in line:
                        print(f"📈 {line.strip()}")
                        # 信頼性の数値を抽出
                        import re
                        match = re.search(r'(\d+\.\d+)%', line)
                        if match:
                            reliability = float(match.group(1))
                            print()
                            if reliability >= 50.0:
                                print(f"✅ システム信頼性{reliability}%（目標50%以上達成）")
                            else:
                                print(f"⚠️ システム信頼性{reliability}%（目標50%未達）")
    
    print()
    print("=" * 60)
    print("🎯 検証完了")
    print("=" * 60)

if __name__ == "__main__":
    verify_warmup_fix()
