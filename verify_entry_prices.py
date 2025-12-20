"""
エントリー価格検証スクリプト - ルックアヘッドバイアス調査

2025-01-06のエントリー価格が終値か始値かを検証

Author: Backtest Project Team
Created: 2025-12-20
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    yfinanceから市場データを取得
    
    Args:
        symbol: 銘柄コード（例: "8053.T"）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
    
    Returns:
        pd.DataFrame: 市場データ（Open, High, Low, Close, Adj Close, Volume）
    """
    print(f"\n{'='*80}")
    print(f"銘柄データ取得: {symbol}")
    print(f"期間: {start_date} 〜 {end_date}")
    print(f"{'='*80}")
    
    try:
        # copilot-instructions.md準拠: auto_adjust=False必須
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if data.empty:
            print(f"⚠️ 警告: {symbol}のデータが取得できませんでした")
            return None
        
        print(f"\n取得データ件数: {len(data)}行")
        print(f"カラム: {data.columns.tolist()}")
        print(f"\n取得データ:")
        print(data.to_string())
        
        return data
    
    except Exception as e:
        print(f"❌ エラー: {symbol}のデータ取得中にエラーが発生しました: {e}")
        return None


def load_transaction_data(csv_path: str) -> pd.DataFrame:
    """
    取引履歴CSVをロード
    
    Args:
        csv_path: CSVファイルパス
    
    Returns:
        pd.DataFrame: 取引履歴
    """
    print(f"\n{'='*80}")
    print(f"取引履歴ロード: {csv_path}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n取引履歴件数: {len(df)}行")
        return df
    
    except Exception as e:
        print(f"❌ エラー: 取引履歴のロード中にエラーが発生しました: {e}")
        return None


def compare_prices(symbol_jp: str, entry_date: str, entry_price: float, 
                   market_data: pd.DataFrame) -> dict:
    """
    エントリー価格と市場データを比較
    
    Args:
        symbol_jp: 日本の銘柄コード（例: "8053"）
        entry_date: エントリー日（YYYY-MM-DD）
        entry_price: エントリー価格
        market_data: 市場データ
    
    Returns:
        dict: 比較結果
    """
    print(f"\n{'='*80}")
    print(f"価格比較: {symbol_jp} on {entry_date}")
    print(f"エントリー価格: {entry_price:.4f}円")
    print(f"{'='*80}")
    
    # エントリー日のデータを取得
    entry_date_dt = pd.to_datetime(entry_date).tz_localize('Asia/Tokyo')
    
    # エントリー日のデータ
    if entry_date_dt in market_data.index:
        entry_day_data = market_data.loc[entry_date_dt]
        print(f"\n【{entry_date}のデータ】")
        print(f"  始値 (Open):      {entry_day_data['Open']:.4f}円")
        print(f"  高値 (High):      {entry_day_data['High']:.4f}円")
        print(f"  安値 (Low):       {entry_day_data['Low']:.4f}円")
        print(f"  終値 (Close):     {entry_day_data['Close']:.4f}円")
        print(f"  調整後終値 (Adj Close): {entry_day_data['Adj Close']:.4f}円")
        
        # 差分計算
        diff_open = entry_price - entry_day_data['Open']
        diff_close = entry_price - entry_day_data['Close']
        diff_adj_close = entry_price - entry_day_data['Adj Close']
        
        print(f"\n【エントリー価格との差分】")
        print(f"  vs 始値:         {diff_open:+.4f}円 ({(diff_open/entry_day_data['Open']*100):+.4f}%)")
        print(f"  vs 終値:         {diff_close:+.4f}円 ({(diff_close/entry_day_data['Close']*100):+.4f}%)")
        print(f"  vs 調整後終値:   {diff_adj_close:+.4f}円 ({(diff_adj_close/entry_day_data['Adj Close']*100):+.4f}%)")
        
        # 翌日データ
        next_day_dt = entry_date_dt + pd.Timedelta(days=1)
        next_day_data = None
        
        # 翌営業日を探す（最大5日先まで）
        for i in range(1, 6):
            check_date = entry_date_dt + pd.Timedelta(days=i)
            if check_date in market_data.index:
                next_day_data = market_data.loc[check_date]
                next_day_dt = check_date
                break
        
        if next_day_data is not None:
            print(f"\n【翌営業日 {next_day_dt.strftime('%Y-%m-%d')} のデータ】")
            print(f"  始値 (Open):      {next_day_data['Open']:.4f}円")
            
            diff_next_open = entry_price - next_day_data['Open']
            print(f"\n【エントリー価格 vs 翌日始値】")
            print(f"  差分:            {diff_next_open:+.4f}円 ({(diff_next_open/next_day_data['Open']*100):+.4f}%)")
        
        # 判定
        print(f"\n{'='*80}")
        print(f"【判定結果】")
        print(f"{'='*80}")
        
        # 最も近い価格を特定
        abs_diffs = {
            '当日始値': abs(diff_open),
            '当日終値': abs(diff_close),
            '当日調整後終値': abs(diff_adj_close),
        }
        
        if next_day_data is not None:
            abs_diffs['翌日始値'] = abs(diff_next_open)
        
        closest = min(abs_diffs, key=abs_diffs.get)
        closest_diff = abs_diffs[closest]
        
        print(f"エントリー価格に最も近い: {closest}")
        print(f"差分: {closest_diff:.4f}円 ({(closest_diff/entry_price*100):.4f}%)")
        
        # ルックアヘッドバイアス判定
        if closest == '当日終値' or closest == '当日調整後終値':
            print(f"\n⚠️ 【ルックアヘッドバイアスの疑い】")
            print(f"   エントリー価格が当日終値に最も近いため、")
            print(f"   終値確定後にエントリー判断している可能性があります。")
            bias_detected = True
        elif closest == '翌日始値':
            print(f"\n✅ 【ルックアヘッドバイアスなし】")
            print(f"   エントリー価格が翌日始値に最も近いため、")
            print(f"   正しい実装の可能性があります。")
            bias_detected = False
        else:
            print(f"\n❓ 【判定保留】")
            print(f"   エントリー価格が当日始値に最も近いですが、")
            print(f"   始値でのエントリーは一般的ではありません。")
            bias_detected = None
        
        return {
            'symbol': symbol_jp,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'day_open': entry_day_data['Open'],
            'day_close': entry_day_data['Close'],
            'day_adj_close': entry_day_data['Adj Close'],
            'next_day_open': next_day_data['Open'] if next_day_data is not None else None,
            'closest_match': closest,
            'closest_diff': closest_diff,
            'bias_detected': bias_detected
        }
    
    else:
        print(f"⚠️ 警告: {entry_date}のデータが市場データに存在しません")
        return None


def main():
    """メイン実行"""
    print(f"\n{'#'*80}")
    print(f"# エントリー価格検証 - ルックアヘッドバイアス調査")
    print(f"# 日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    # 設定
    target_date = "2025-01-06"
    start_date = "2025-01-03"
    end_date = "2025-01-10"
    
    # 既存取引データから該当レコードを抽出
    csv_path = r"output\dssms_integration\dssms_20251219_233627\dssms_all_transactions.csv"
    transactions = load_transaction_data(csv_path)
    
    if transactions is None:
        print("❌ 取引履歴のロードに失敗しました")
        return
    
    # 2025-01-06のデータを抽出
    target_transactions = transactions[
        transactions['entry_date'].str.contains(target_date)
    ].head(3)  # 最初の3件を確認
    
    print(f"\n{'='*80}")
    print(f"2025-01-06のエントリー（最初の3件）:")
    print(f"{'='*80}")
    print(target_transactions[['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price']].to_string(index=False))
    
    # 各銘柄について検証
    results = []
    
    for idx, row in target_transactions.iterrows():
        symbol_jp = row['symbol']
        symbol_yf = f"{symbol_jp}.T"
        entry_price = row['entry_price']
        
        # 市場データ取得
        market_data = fetch_market_data(symbol_yf, start_date, end_date)
        
        if market_data is not None:
            # 価格比較
            result = compare_prices(symbol_jp, target_date, entry_price, market_data)
            if result:
                results.append(result)
    
    # 総合判定
    print(f"\n{'#'*80}")
    print(f"# 総合判定")
    print(f"{'#'*80}")
    
    if results:
        bias_count = sum(1 for r in results if r['bias_detected'] is True)
        no_bias_count = sum(1 for r in results if r['bias_detected'] is False)
        unclear_count = sum(1 for r in results if r['bias_detected'] is None)
        
        print(f"\n検証件数: {len(results)}件")
        print(f"  ルックアヘッドバイアスあり: {bias_count}件")
        print(f"  バイアスなし: {no_bias_count}件")
        print(f"  判定保留: {unclear_count}件")
        
        if bias_count > 0:
            print(f"\n⚠️ 【重大な問題】")
            print(f"   {bias_count}件でルックアヘッドバイアスが検出されました。")
            print(f"   エントリー価格が当日終値に近いため、")
            print(f"   終値確定後にエントリー判断している可能性が高いです。")
        elif no_bias_count == len(results):
            print(f"\n✅ 【問題なし】")
            print(f"   全てのエントリー価格が翌日始値に近いため、")
            print(f"   正しい実装の可能性があります。")
        else:
            print(f"\n❓ 【要追加調査】")
            print(f"   結果が混在しているため、追加の調査が必要です。")
    
    print(f"\n{'#'*80}")
    print(f"# 検証完了")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
