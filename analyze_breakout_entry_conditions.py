"""
Cycle 14: BreakoutStrategy volume_threshold=1.0 取引0件問題調査スクリプト

目的: 2025-01-15～2025-01-31期間中、各銘柄でBreakoutStrategy（volume_threshold=1.0）の
      エントリー条件を満たす日がなぜ0件なのか、日毎に条件チェックして原因特定

調査項目:
1. 価格ブレイクアウト条件: current_price > previous_high * 1.01
2. 出来高条件: current_volume > previous_volume * 1.0
3. 両方満たす日の有無

Author: Cycle 14 Investigation
Created: 2026-01-10
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def check_breakout_conditions(symbol, start_date, end_date):
    """
    指定銘柄・期間でBreakoutStrategy条件チェック
    
    Args:
        symbol: 銘柄コード（例: "8233.T"）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
    
    Returns:
        DataFrame: 日毎の条件チェック結果
    """
    # データ取得（ウォームアップ期間含む）
    start_with_warmup = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"\n{'='*80}")
    print(f"銘柄: {symbol} | 期間: {start_date} ~ {end_date}")
    print(f"{'='*80}")
    
    try:
        data = yf.download(symbol, start=start_with_warmup, end=end_date, auto_adjust=False, progress=False)
        
        if data.empty:
            print(f"[エラー] データ取得失敗")
            return None
        
        if len(data) < 2:
            print(f"[エラー] データ不十分（{len(data)}日）")
            return None
            
        # BreakoutStrategy条件チェック
        results = []
        
        for idx in range(1, len(data)):
            current_date = data.index[idx]
            
            if current_date < pd.to_datetime(start_date):
                continue  # ウォームアップ期間はスキップ
            
            try:
                current_price = float(data['Adj Close'].iloc[idx])
                previous_high = float(data['High'].iloc[idx - 1])
                current_volume = float(data['Volume'].iloc[idx])
                previous_volume = float(data['Volume'].iloc[idx - 1])
            except (KeyError, IndexError, TypeError) as e:
                print(f"[警告] 日付{current_date}: データ取得失敗 - {e}")
                continue
            
            # 条件1: 価格ブレイクアウト（前日高値 * 1.01を上抜け）
            breakout_threshold = previous_high * 1.01
            price_breakout = current_price > breakout_threshold
            
            # 条件2: 出来高増加（前日比1.0倍以上 = 前日と同等以上）
            volume_threshold = previous_volume * 1.0
            volume_increase = current_volume > volume_threshold
            
            # 両方満たすかチェック
            entry_signal = price_breakout and volume_increase
            
            results.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'current_price': f"{current_price:.2f}",
                'prev_high': f"{previous_high:.2f}",
                'breakout_threshold': f"{breakout_threshold:.2f}",
                'price_breakout': '✓' if price_breakout else '✗',
                'current_volume': f"{current_volume:,.0f}",
                'prev_volume': f"{previous_volume:,.0f}",
                'volume_increase': '✓' if volume_increase else '✗',
                'entry_signal': '🔥 ENTRY' if entry_signal else '-'
            })
        
        df = pd.DataFrame(results)
        
        # サマリー表示
        entry_count = (df['entry_signal'] == '🔥 ENTRY').sum()
        price_ok_count = (df['price_breakout'] == '✓').sum()
        volume_ok_count = (df['volume_increase'] == '✓').sum()
        
        print(f"\n[サマリー]")
        print(f"  - 全取引日数: {len(df)}日")
        print(f"  - 価格ブレイクアウト満たす日: {price_ok_count}日")
        print(f"  - 出来高条件満たす日: {volume_ok_count}日")
        print(f"  - 両方満たす日（エントリー可能）: {entry_count}日")
        
        if entry_count == 0:
            print(f"\n[問題特定]")
            if price_ok_count == 0:
                print(f"  → 価格ブレイクアウト条件が厳しすぎる（前日高値*1.01を一度も上抜けず）")
            if volume_ok_count == 0:
                print(f"  → 出来高条件が満たされない（常に前日以下）")
            elif price_ok_count > 0 and volume_ok_count > 0:
                print(f"  → 両条件を同時に満たす日がない（タイミング不一致）")
        
        # 詳細表示
        print(f"\n[日毎詳細]")
        print(df.to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"[エラー] {symbol}: {str(e)}")
        return None


def main():
    """メイン調査関数"""
    # テスト期間
    start_date = "2025-01-15"
    end_date = "2025-01-31"
    
    # DSSMS選択銘柄（ログから抽出）
    symbols = [
        "8233.T",  # 2025-01-30選択
        "6723.T",  # 2025-01-31選択
        "8604.T",  # 他の高スコア銘柄
        "8411.T",
        "8331.T"
    ]
    
    print("="*80)
    print("Cycle 14: BreakoutStrategy volume_threshold=1.0 エントリー条件調査")
    print("="*80)
    print(f"期間: {start_date} ~ {end_date}")
    print(f"調査銘柄: {len(symbols)}銘柄")
    print(f"条件1: current_price > previous_high * 1.01（価格ブレイクアウト）")
    print(f"条件2: current_volume > previous_volume * 1.0（出来高増加）")
    print("="*80)
    
    all_results = {}
    for symbol in symbols:
        result = check_breakout_conditions(symbol, start_date, end_date)
        if result is not None:
            all_results[symbol] = result
    
    # 総合サマリー
    print("\n" + "="*80)
    print("【総合結果】")
    print("="*80)
    
    total_entry_days = 0
    for symbol, df in all_results.items():
        entry_count = (df['entry_signal'] == '🔥 ENTRY').sum()
        total_entry_days += entry_count
        print(f"{symbol}: エントリー可能日={entry_count}日")
    
    print(f"\n全銘柄合計: {total_entry_days}日")
    
    if total_entry_days == 0:
        print("\n【推奨対策】")
        print("1. breakout_buffer を 0.01 → 0.005 に緩和（前日高値*1.005）")
        print("2. volume_threshold を 1.0 → 0.9 に緩和（前日の90%でOK）")
        print("3. look_back を 1 → 2 に変更（2日前の高値でブレイクアウト判定）")


if __name__ == "__main__":
    main()
