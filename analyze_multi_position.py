import pandas as pd
from collections import defaultdict

# all_transactions.csvを読み込み
df = pd.read_csv('output/dssms_integration/dssms_20260215_005537/all_transactions.csv')
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])

print('=' * 60)
print('複数銘柄保有対応 詳細分析レポート')
print('=' * 60)

print('\n=== 1. 取引一覧 ===')
for idx, row in df.iterrows():
    print(f'{idx+1}. {row["symbol"]}: {row["entry_date"].strftime("%Y-%m-%d")} ~ {row["exit_date"].strftime("%Y-%m-%d")} '
          f'({row["holding_period_days"]}日), PnL={row["pnl"]:+,.0f}円 ({row["return_pct"]*100:+.2f}%)')

print(f'\n全{len(df)}銘柄, 合計PnL={df["pnl"].sum():+,.0f}円')

print('\n=== 2. 同時保有期間の検証 ===')
# 各日付で保有している銘柄を確認
holdings_by_date = defaultdict(list)

for idx, row in df.iterrows():
    entry = row['entry_date']
    exit = row['exit_date']
    symbol = row['symbol']
    
    # 保有期間中の全日付をチェック（営業日ベース）
    date_range = pd.date_range(start=entry, end=exit, freq='D')
    for date in date_range:
        holdings_by_date[date].append(symbol)

# 2銘柄以上保有している日を抽出
multi_holdings = {date: symbols for date, symbols in holdings_by_date.items() if len(symbols) >= 2}

if multi_holdings:
    print(f'\n2銘柄以上同時保有した日数: {len(multi_holdings)}日')
    print('\n最初の10日:')
    for i, (date, symbols) in enumerate(sorted(multi_holdings.items())[:10]):
        print(f'  {date.strftime("%Y-%m-%d")}: {symbols} ({len(symbols)}銘柄)')
    
    if len(multi_holdings) > 20:
        print(f'  ... (中略 {len(multi_holdings)-20}日)')
    
    print('\n最後の10日:')
    for date, symbols in sorted(multi_holdings.items())[-10:]:
        print(f'  {date.strftime("%Y-%m-%d")}: {symbols} ({len(symbols)}銘柄)')
else:
    print('2銘柄同時保有は確認されませんでした')

# 最大同時保有数
max_holdings = max(len(symbols) for symbols in holdings_by_date.values())
print(f'\n最大同時保有銘柄数: {max_holdings}銘柄')

# 保有銘柄数の分布
holdings_count = defaultdict(int)
for symbols in holdings_by_date.values():
    holdings_count[len(symbols)] += 1

print('\n=== 3. 保有銘柄数の分布 ===')
for count in sorted(holdings_count.keys()):
    print(f'{count}銘柄保有: {holdings_count[count]}日')

print('\n=== 4. max_positions=2の実動作確認 ===')
if max_holdings <= 2:
    print(f'OK: 最大同時保有数は{max_holdings}銘柄で、max_positions=2を超えていません')
else:
    print(f'NG: 最大同時保有数が{max_holdings}銘柄で、max_positions=2を超えています')

print('\n=== 5. 強制決済の確認 ===')
forced_exits = df[df['is_forced_exit'] == True]
print(f'強制決済された銘柄: {len(forced_exits)}件 / {len(df)}件')
for idx, row in forced_exits.iterrows():
    print(f'  - {row["symbol"]}: {row["exit_date"].strftime("%Y-%m-%d")}に強制決済')

print('\n=== 6. CSV完全性の確認 ===')
checks = [
    ('exit_date', df['exit_date'].notna().all()),
    ('exit_price', df['exit_price'].notna().all()),
    ('pnl', df['pnl'].notna().all()),
    ('return_pct', df['return_pct'].notna().all()),
]

all_ok = True
for field, is_ok in checks:
    status = 'OK' if is_ok else 'NG'
    print(f'{field}: {status}')
    if not is_ok:
        all_ok = False

if all_ok:
    print('\nall_transactions.csv完全性: OK')
else:
    print('\nall_transactions.csv完全性: NG - 空の値が存在します')
