"""
Task 4: エントリーシグナル発生レベル分析

目的: エントリー条件緩和後、どのレベルでエントリーが発生したかを分析
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# all_transactions.csvから取引データ読み込み
df = pd.read_csv("output/dssms_integration/dssms_20260116_133050/all_transactions.csv")

# 未決済取引を除外
df = df[df['exit_date'].notna()].copy()

print(f"分析対象取引数: {len(df)}件")
print("\n=== エントリーシグナル発生レベル分析開始 ===\n")

results = []

for idx, row in df.iterrows():
    symbol = row['symbol']
    entry_date_str = row['entry_date']
    entry_price = row['entry_price']
    
    # タイムスタンプ文字列をdatetime変換
    entry_date = pd.to_datetime(entry_date_str)
    
    # MA計算に必要な期間を取得（entry_date - 90日 ~ entry_date + 1日）
    start_date = entry_date - timedelta(days=90)
    end_date = entry_date + timedelta(days=5)
    
    try:
        # yfinanceで株価データ取得
        ticker_data = yf.download(f"{symbol}.T", start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if ticker_data.empty or len(ticker_data) < 30:
            print(f"[SKIP] {symbol} @ {entry_date.date()}: データ不足")
            continue
        
        # MA計算
        ticker_data['MA5'] = ticker_data['Close'].rolling(window=5, min_periods=5).mean()
        ticker_data['MA25'] = ticker_data['Close'].rolling(window=25, min_periods=25).mean()
        
        # エントリー日のデータ取得
        if entry_date not in ticker_data.index:
            # 最も近い営業日を探す
            nearest_date = ticker_data.index[ticker_data.index <= entry_date].max()
            if pd.isna(nearest_date):
                print(f"[SKIP] {symbol} @ {entry_date.date()}: エントリー日データなし")
                continue
            entry_date = nearest_date
        
        entry_row = ticker_data.loc[entry_date]
        
        # 前日データ取得（シフト1日）
        entry_idx = ticker_data.index.get_loc(entry_date)
        if entry_idx == 0:
            print(f"[SKIP] {symbol} @ {entry_date.date()}: 前日データなし")
            continue
        
        prev_date = ticker_data.index[entry_idx - 1]
        prev_row = ticker_data.loc[prev_date]
        
        # 5MA, 25MA, 前日5MA, 前日25MA取得（iloc[]でスカラー値に変換）
        try:
            entry_idx_loc = ticker_data.index.get_loc(entry_date)
            prev_idx_loc = ticker_data.index.get_loc(prev_date)
            
            short_sma = ticker_data.iloc[entry_idx_loc]['MA5']
            long_sma = ticker_data.iloc[entry_idx_loc]['MA25']
            prev_short_sma = ticker_data.iloc[prev_idx_loc]['MA5']
            prev_long_sma = ticker_data.iloc[prev_idx_loc]['MA25']
            
            # None/NaN値チェック（Seriesではなくスカラー値として処理）
            if any(pd.isna(v) for v in [short_sma, long_sma, prev_short_sma, prev_long_sma]):
                print(f"[SKIP] {symbol} @ {entry_date.date()}: MA値にNaNあり")
                continue
        except Exception as e:
            print(f"[SKIP] {symbol} @ {entry_date.date()}: MA値取得エラー - {str(e)}")
            continue
        
        # GC条件判定
        # golden_cross: short_sma > long_sma and prev_short_sma <= prev_long_sma
        # uptrend_continuation: short_sma > long_sma and short_sma > prev_short_sma and long_sma > prev_long_sma
        
        golden_cross = (short_sma > long_sma) and (prev_short_sma <= prev_long_sma)
        uptrend_continuation = (short_sma > long_sma) and (short_sma > prev_short_sma) and (long_sma > prev_long_sma)
        
        # エントリー理由判定
        if golden_cross and uptrend_continuation:
            entry_reason = "GC+継続"
        elif golden_cross:
            entry_reason = "GC"
        elif uptrend_continuation:
            entry_reason = "継続"
        else:
            entry_reason = "不明"
        
        # トレンド状態判定（簡易版: MA5 > MA25 = uptrend）
        if short_sma > long_sma:
            trend_state = "uptrend"
        elif short_sma < long_sma:
            trend_state = "downtrend"
        else:
            trend_state = "sideways"
        
        results.append({
            "symbol": symbol,
            "entry_date": entry_date.date(),
            "entry_price": entry_price,
            "MA5": short_sma,
            "MA25": long_sma,
            "prev_MA5": prev_short_sma,
            "prev_MA25": prev_long_sma,
            "MA5_change": short_sma - prev_short_sma,
            "MA25_change": long_sma - prev_long_sma,
            "entry_reason": entry_reason,
            "trend_state": trend_state,
            "pnl": row['pnl'],
            "return_pct": row['return_pct']
        })
        
        print(f"[OK] {symbol} @ {entry_date.date()}: {entry_reason}")
        
    except Exception as e:
        print(f"[ERROR] {symbol} @ {entry_date.date()}: {str(e)}")
        continue

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

print(f"\n=== 分析完了 ===")
print(f"成功: {len(results_df)}件")

# 統計サマリー
print("\n=== エントリー理由別集計 ===")
print(results_df['entry_reason'].value_counts())

print("\n=== トレンド状態別集計 ===")
print(results_df['trend_state'].value_counts())

# エントリー理由別の平均損益
print("\n=== エントリー理由別 平均損益 ===")
print(results_df.groupby('entry_reason')['pnl'].mean())

print("\n=== エントリー理由別 平均リターン ===")
print(results_df.groupby('entry_reason')['return_pct'].mean())

# CSVに保存
output_path = "output/dssms_integration/dssms_20260116_133050/entry_signal_analysis.csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[OUTPUT] {output_path} に保存完了")

# 詳細レポート表示（先頭10件）
print("\n=== 詳細レポート（先頭10件） ===")
print(results_df.head(10).to_string())
