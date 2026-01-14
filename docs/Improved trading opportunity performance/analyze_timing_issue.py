"""
銘柄スイッチとエントリータイミングの関係分析

目的:
- 銘柄スイッチのタイミングとGCシグナルの関係を確認
- なぜエントリーできなかったかを特定

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("銘柄スイッチとエントリータイミングの関係分析")
    print("="*80)
    
    # データ読み込み
    base_path = project_root / "output" / "dssms_integration" / "dssms_20260111_232522"
    
    # 銘柄スイッチ履歴
    switch_df = pd.read_csv(base_path / "dssms_switch_history.csv")
    switch_df['switch_date'] = pd.to_datetime(switch_df['switch_date'])
    
    # トランザクション履歴
    trans_df = pd.read_csv(base_path / "all_transactions.csv")
    trans_df['entry_date'] = pd.to_datetime(trans_df['entry_date'])
    trans_df['exit_date'] = pd.to_datetime(trans_df['exit_date'])
    
    # GCシグナル情報（前回の分析結果から）
    gc_signals = {
        4506: [
            '2025-01-21', '2025-03-21', '2025-04-24', 
            '2025-06-30', '2025-07-25', '2025-08-12', '2025-11-17'
        ],
        5202: [
            '2024-09-02', '2024-10-18', '2025-01-30', 
            '2025-03-12', '2025-04-11', '2025-07-22', 
            '2025-08-14', '2025-10-16'
        ],
        1662: [
            '2024-10-07', '2024-10-30', '2025-03-13', 
            '2025-04-22', '2025-07-01', '2025-08-13', 
            '2025-09-11', '2025-10-30'
        ],
        5713: [
            '2025-01-09', '2025-01-21', '2025-03-21', 
            '2025-04-24', '2025-06-30', '2025-07-25', 
            '2025-08-12', '2025-11-17'
        ],
        8604: [
            '2025-01-21', '2025-03-26', '2025-05-09', '2025-10-29'
        ]
    }
    
    # 各GCシグナルをDataFrameに変換
    gc_signal_list = []
    for ticker, dates in gc_signals.items():
        for date_str in dates:
            gc_signal_list.append({
                'ticker': ticker,
                'signal_date': pd.to_datetime(date_str)
            })
    
    gc_df = pd.DataFrame(gc_signal_list)
    
    print(f"\nデータ確認:")
    print(f"  銘柄スイッチ: {len(switch_df)}回")
    print(f"  トランザクション: {len(trans_df)}回")
    print(f"  GCシグナル: {len(gc_df)}回")
    
    # 分析1: GCシグナル発生時の銘柄保有状況
    print(f"\n{'='*80}")
    print("分析1: GCシグナル発生時の銘柄保有状況")
    print(f"{'='*80}")
    
    missed_opportunities = []
    
    for idx, gc_row in gc_df.iterrows():
        ticker = gc_row['ticker']
        signal_date = gc_row['signal_date']
        
        # このシグナル日に該当銘柄を保有していたか確認
        # switch_historyから該当日の銘柄を確認
        prior_switches = switch_df[switch_df['switch_date'] <= signal_date]
        if len(prior_switches) > 0:
            latest_switch = prior_switches.iloc[-1]
            current_ticker = int(latest_switch['to_symbol'])
            
            # 実際のエントリーがあったか確認
            actual_entry = trans_df[
                (trans_df['symbol'] == ticker) &
                (trans_df['entry_date'] == signal_date)
            ]
            
            if current_ticker == ticker and len(actual_entry) == 0:
                # 保有していたのにエントリーしなかった
                missed_opportunities.append({
                    'ticker': ticker,
                    'signal_date': signal_date,
                    'current_ticker': current_ticker,
                    'reason': '保有中だが未エントリー'
                })
            elif current_ticker != ticker:
                # 別の銘柄を保有していた
                missed_opportunities.append({
                    'ticker': ticker,
                    'signal_date': signal_date,
                    'current_ticker': current_ticker,
                    'reason': f'別銘柄保有({current_ticker})'
                })
    
    print(f"\nGCシグナル総数: {len(gc_df)}回")
    print(f"実際のエントリー: {len(trans_df)}回")
    print(f"逃した機会: {len(missed_opportunities)}回")
    
    # 理由別の集計
    missed_df = pd.DataFrame(missed_opportunities)
    if len(missed_df) > 0:
        print(f"\n逃した理由の内訳:")
        reason_counts = missed_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}回")
        
        # 別銘柄保有の詳細
        other_ticker_cases = missed_df[missed_df['reason'].str.contains('別銘柄保有')]
        if len(other_ticker_cases) > 0:
            print(f"\n別銘柄保有の詳細（最初の10件）:")
            for idx, row in other_ticker_cases.head(10).iterrows():
                print(f"  {row['signal_date'].strftime('%Y-%m-%d')}: {row['ticker']}のGCシグナル発生 → {row['current_ticker']}を保有中")
    
    # 分析2: 銘柄スイッチ直後のGCシグナル
    print(f"\n{'='*80}")
    print("分析2: 銘柄スイッチ直後のGCシグナル（エントリー前にスイッチ）")
    print(f"{'='*80}")
    
    switch_after_gc = []
    
    for idx, gc_row in gc_df.iterrows():
        ticker = gc_row['ticker']
        signal_date = gc_row['signal_date']
        
        # GCシグナルの翌日以降に銘柄スイッチがあったか確認
        next_switches = switch_df[
            (switch_df['switch_date'] > signal_date) &
            (switch_df['switch_date'] <= signal_date + timedelta(days=3))  # 3日以内
        ]
        
        if len(next_switches) > 0:
            for _, switch_row in next_switches.iterrows():
                if int(switch_row['from_symbol']) == ticker:
                    switch_after_gc.append({
                        'ticker': ticker,
                        'signal_date': signal_date,
                        'switch_date': switch_row['switch_date'],
                        'days_after_signal': (switch_row['switch_date'] - signal_date).days
                    })
    
    if len(switch_after_gc) > 0:
        print(f"\nGCシグナル後3日以内にスイッチされた回数: {len(switch_after_gc)}回")
        switch_after_df = pd.DataFrame(switch_after_gc)
        print(f"\nスイッチまでの平均日数: {switch_after_df['days_after_signal'].mean():.1f}日")
        
        print(f"\n詳細（最初の10件）:")
        for idx, row in switch_after_df.head(10).iterrows():
            print(f"  {row['signal_date'].strftime('%Y-%m-%d')}: {row['ticker']}のGCシグナル発生")
            print(f"    → {row['switch_date'].strftime('%Y-%m-%d')}（{row['days_after_signal']}日後）にスイッチ")
    
    # サマリー
    print(f"\n{'='*80}")
    print("エントリー機会損失の原因サマリー")
    print(f"{'='*80}")
    
    print(f"\n1. GCシグナル総数: {len(gc_df)}回")
    print(f"2. 実際のエントリー: {len(trans_df)}回 ({len(trans_df)/len(gc_df)*100:.1f}%)")
    print(f"3. 逃した機会: {len(missed_opportunities)}回 ({len(missed_opportunities)/len(gc_df)*100:.1f}%)")
    
    if len(missed_df) > 0:
        other_ticker_count = len(missed_df[missed_df['reason'].str.contains('別銘柄保有')])
        print(f"   - 別銘柄保有中: {other_ticker_count}回 ({other_ticker_count/len(missed_opportunities)*100:.1f}%)")
        
        no_entry_count = len(missed_df[missed_df['reason'] == '保有中だが未エントリー'])
        print(f"   - 保有中だが未エントリー: {no_entry_count}回 ({no_entry_count/len(missed_opportunities)*100:.1f}%)")
    
    print(f"\n4. GCシグナル後3日以内にスイッチ: {len(switch_after_gc)}回")
    
    print(f"\n主要原因:")
    print(f"  [1] 銘柄スイッチが頻回（85回、平均3.9日に1回）")
    print(f"  [2] GCシグナル発生時に別銘柄を保有")
    print(f"  [3] GCシグナル発生後すぐに銘柄スイッチされエントリー機会を逃す")
    
    print(f"\n{'='*80}")
    print("調査完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
