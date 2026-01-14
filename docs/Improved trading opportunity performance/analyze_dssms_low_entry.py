"""
DSSMS低エントリー回数調査スクリプト

目的:
- 実際にGCシグナルが何回発生したか確認
- 各銘柄でのエントリー機会を分析
- 戦略選択の偏りを確認

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_parameters_and_data
from strategies.gc_strategy_signal import GCStrategy

def analyze_gc_signals_for_ticker(ticker: str, start_date: str, end_date: str):
    """特定銘柄のGCシグナル発生回数を確認"""
    print(f"\n{'='*80}")
    print(f"分析対象: {ticker} ({start_date} ~ {end_date})")
    print(f"{'='*80}")
    
    try:
        # データ取得（ウォームアップ期間含む）
        _, _, _, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            warmup_days=150
        )
        
        if stock_data is None or len(stock_data) == 0:
            print(f"[SKIP] {ticker}: データ取得失敗")
            return None
        
        print(f"データ期間: {stock_data.index[0]} ~ {stock_data.index[-1]} ({len(stock_data)}日)")
        
        # GC戦略インスタンス作成（GCStrategyの正しい引数に修正）
        gc_strategy = GCStrategy(
            data=stock_data,
            params=None,
            price_column="Adj Close",
            ticker=ticker
        )
        
        # GCシグナル確認
        gc_signals = stock_data[stock_data['GC_Signal'] == 1]
        signal_count = len(gc_signals)
        
        print(f"GCシグナル発生回数: {signal_count}回")
        
        if signal_count > 0:
            print(f"\nGCシグナル発生日:")
            for date in gc_signals.index[:10]:  # 最初の10件のみ表示
                print(f"  - {date.strftime('%Y-%m-%d')}")
            if signal_count > 10:
                print(f"  ... 他 {signal_count - 10}件")
        
        # トレンドフィルター確認
        trend_filter = gc_strategy.params.get("trend_filter_enabled", False)
        print(f"\nトレンドフィルター: {'有効' if trend_filter else '無効'}")
        
        return {
            'ticker': ticker,
            'data_period': f"{stock_data.index[0]} ~ {stock_data.index[-1]}",
            'data_days': len(stock_data),
            'gc_signal_count': signal_count,
            'trend_filter_enabled': trend_filter
        }
        
    except Exception as e:
        print(f"[ERROR] {ticker} 分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_dssms_switch_history():
    """DSSMS銘柄切替履歴を分析"""
    print(f"\n{'='*80}")
    print("DSSMS銘柄切替履歴分析")
    print(f"{'='*80}")
    
    switch_file = project_root / "output" / "dssms_integration" / "dssms_20260111_232522" / "dssms_switch_history.csv"
    
    if not switch_file.exists():
        print(f"[ERROR] 銘柄切替履歴が見つかりません: {switch_file}")
        return
    
    switch_df = pd.read_csv(switch_file)
    print(f"\n総切替回数: {len(switch_df)}回")
    print(f"期間: {switch_df['switch_date'].iloc[0]} ~ {switch_df['switch_date'].iloc[-1]}")
    
    # 銘柄別の選択回数
    print(f"\n銘柄別選択回数:")
    symbol_counts = switch_df['to_symbol'].value_counts()
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count}回")
    
    # 平均保有期間
    switch_df['switch_date'] = pd.to_datetime(switch_df['switch_date'])
    switch_df_sorted = switch_df.sort_values('switch_date')
    holding_periods = switch_df_sorted['switch_date'].diff().dt.days
    avg_holding = holding_periods.mean()
    
    print(f"\n平均保有期間: {avg_holding:.1f}日")
    print(f"最短保有期間: {holding_periods.min():.0f}日")
    print(f"最長保有期間: {holding_periods.max():.0f}日")
    
    # 1日で切り替わった回数
    one_day_switches = len(holding_periods[holding_periods == 1])
    print(f"\n1日で切替: {one_day_switches}回 ({one_day_switches/len(switch_df)*100:.1f}%)")
    
    return switch_df


def analyze_transactions():
    """実際のトランザクション分析"""
    print(f"\n{'='*80}")
    print("トランザクション分析")
    print(f"{'='*80}")
    
    trans_file = project_root / "output" / "dssms_integration" / "dssms_20260111_232522" / "all_transactions.csv"
    
    if not trans_file.exists():
        print(f"[ERROR] トランザクションファイルが見つかりません: {trans_file}")
        return
    
    trans_df = pd.read_csv(trans_file)
    print(f"\n総取引回数: {len(trans_df)}回")
    
    # 戦略別取引回数
    print(f"\n戦略別取引回数:")
    strategy_counts = trans_df['strategy_name'].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}回")
    
    # 銘柄別取引回数
    print(f"\n銘柄別取引回数:")
    symbol_counts = trans_df['symbol'].value_counts()
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count}回")
    
    # 強制決済の回数
    forced_exits = trans_df[trans_df['is_forced_exit'] == True]
    print(f"\n強制決済: {len(forced_exits)}回 ({len(forced_exits)/len(trans_df)*100:.1f}%)")
    
    # 平均保有期間
    avg_holding = trans_df['holding_period_days'].mean()
    print(f"\n平均保有期間: {avg_holding:.1f}日")
    
    return trans_df


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("DSSMS低エントリー回数調査スクリプト実行開始")
    print("="*80)
    
    # 1. DSSMS切替履歴分析
    switch_df = analyze_dssms_switch_history()
    
    # 2. トランザクション分析
    trans_df = analyze_transactions()
    
    # 3. 主要銘柄のGCシグナル分析
    print(f"\n{'='*80}")
    print("主要銘柄のGCシグナル分析")
    print(f"{'='*80}")
    
    # 取引が発生した銘柄を確認
    if trans_df is not None:
        traded_symbols = trans_df['symbol'].unique()
        print(f"\n取引が発生した銘柄: {list(traded_symbols)}")
    
    # 最も選択された銘柄のGCシグナル分析
    if switch_df is not None:
        top_symbols = switch_df['to_symbol'].value_counts().head(5).index.tolist()
        print(f"\n最も選択された上位5銘柄: {top_symbols}")
        
        results = []
        for symbol in top_symbols:
            result = analyze_gc_signals_for_ticker(
                ticker=symbol,
                start_date="2025-01-01",
                end_date="2025-11-30"
            )
            if result:
                results.append(result)
        
        # サマリー出力
        if results:
            print(f"\n{'='*80}")
            print("GCシグナル発生サマリー")
            print(f"{'='*80}")
            
            df_summary = pd.DataFrame(results)
            print(df_summary.to_string(index=False))
            
            # CSVに保存
            output_file = project_root / "docs" / "Improved trading opportunity performance" / "gc_signal_analysis_summary.csv"
            df_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nサマリーを保存しました: {output_file}")
    
    print(f"\n{'='*80}")
    print("調査完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
