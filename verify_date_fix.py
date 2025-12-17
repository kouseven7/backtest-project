"""
日付逆行検証スクリプト（2025-12-17修正後）

修正内容: comprehensive_reporter.py Line 479-487でタイムスタンプソート実装
検証目的: dssms_trades.csvに日付逆行が残っていないことを確認
"""
import pandas as pd
import sys

csv_path = 'output/dssms_integration/dssms_20251217_002959/dssms_trades.csv'

try:
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV読取成功: {csv_path}")
    print(f"[INFO] 総取引数: {len(df)}")
    
    # 日付文字列をdatetimeに変換
    df['entry_datetime'] = pd.to_datetime(df['entry_date'])
    df['exit_datetime'] = pd.to_datetime(df['exit_date'])
    df['days_diff'] = (df['exit_datetime'] - df['entry_datetime']).dt.days
    
    # 日付逆行チェック（days_diff < 0）
    reversed = df[df['days_diff'] < 0]
    
    print(f"\n[RESULT] 日付逆行エラー数: {len(reversed)}")
    
    if len(reversed) > 0:
        print("\n[ERROR] 以下の取引で日付逆行が検出されました:\n")
        print(reversed[['entry_date', 'exit_date', 'days_diff', 'strategy']].to_string(index=False))
        sys.exit(1)
    else:
        print("\n[SUCCESS] 全取引で日付順序が正常（entry_date <= exit_date）")
        print(f"\n[DETAIL] holding_period_days分布:")
        print(df['holding_period_days'].describe())
        sys.exit(0)

except FileNotFoundError:
    print(f"[ERROR] ファイルが見つかりません: {csv_path}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] 予期しないエラー: {e}")
    sys.exit(1)
