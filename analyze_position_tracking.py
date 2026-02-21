"""
ポジション数追跡分析スクリプト

目的: 日次のポジション保有数変化を追跡してmax_positions=2制約違反を検出
"""
import re

log_path = "output/dssms_integration/dssms_20260215_083410/comprehensive_report.txt"

try:
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 日次取引履歴から保有数推定
    print("=" * 80)
    print("日次ポジション管理分析")
    print("=" * 80)
    
    # all_transactions.csvから時系列分析
    import pandas as pd
    csv_path = "output/dssms_integration/dssms_20260215_083410/all_transactions.csv"
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    # 日付範囲作成
    date_range = pd.date_range(start=df['entry_date'].min(), end=df['exit_date'].max(), freq='D')
    
    position_counts = {}
    for date in date_range:
        holding = 0
        symbols_held = []
        for _, row in df.iterrows():
            if row['entry_date'] <= date <= row['exit_date']:
                holding += 1
                symbols_held.append(row['symbol'])
        position_counts[date] = (holding, symbols_held)
    
    # max_positions超過日を抽出
    violations = []
    for date, (count, symbols) in position_counts.items():
        if count > 2:
            violations.append((date, count, symbols))
    
    print(f"\nmax_positions=2 違反検出結果:")
    print(f"違反日数: {len(violations)}日")
    print(f"\n違反詳細:")
    for date, count, symbols in violations[:10]:  # 最初の10件表示
        print(f"  {date.strftime('%Y-%m-%d')}: {count}銘柄保有 {symbols}")
    
    # 保有数分布
    from collections import Counter
    count_dist = Counter([count for count, _ in position_counts.values()])
    print(f"\n保有数分布:")
    for count in sorted(count_dist.keys()):
        print(f"  {count}銘柄: {count_dist[count]}日")
    
except FileNotFoundError as e:
    print(f"ファイル未検出: {e}")
except Exception as e:
    print(f"エラー: {e}")
