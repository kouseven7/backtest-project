"""
contrarian_strategy修正後のバックテストテスト

ルックアヘッドバイアス修正後の結果を記録します。
修正前の結果と比較してバグ修正の効果を確認します。

主な機能:
- contrarian_strategy.pyの修正後バックテスト実行
- 取引回数、勝率、総利益の記録
- 修正前との比較分析
- CSV/JSON形式での結果保存

統合コンポーネント:
- strategies/contrarian_strategy.py: テスト対象戦略（修正済み）
- データ取得: yfinanceまたはキャッシュ

セーフティ機能/注意事項:
- 実際のバックテストを実行（推測なし）
- 結果をファイルに保存して比較可能
- エラー時は詳細を報告

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import yfinance as yf
from strategies.contrarian_strategy import ContrarianStrategy
from datetime import datetime
import json

def test_contrarian_after_fix():
    """修正後のcontrarian_strategyをテスト"""
    
    print("="*80)
    print("contrarian_strategy 修正後バックテスト")
    print("="*80)
    
    # テストデータ取得
    ticker = "7203.T"  # トヨタ自動車
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print(f"\nデータ取得中: {ticker} ({start_date} - {end_date})")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(data) == 0:
        print("エラー: データ取得失敗")
        return
    
    # MultiIndexの場合はフラット化
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"取得データ行数: {len(data)}")
    print(f"カラム: {data.columns.tolist()}")
    
    # 戦略パラメータ（修正前と同じ）
    params = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "gap_threshold": 0.02,
        "stop_loss": 0.05,
        "take_profit": 0.10,
        "trailing_stop": 0.03,
        "max_holding_period": 10,
        "trend_filter_enabled": False,
        "allowed_trends": ["downtrend", "neutral"]
    }
    
    # 戦略初期化（Closeカラムを使用）
    print("\n戦略初期化中...")
    strategy = ContrarianStrategy(data, params, price_column="Close")
    
    # バックテスト実行
    print("\nバックテスト実行中...")
    results = strategy.backtest()
    
    # 結果集計
    entry_signals = (results['Entry_Signal'] == 1).sum()
    exit_signals = (results['Exit_Signal'] == -1).sum()
    
    # エントリーとイグジットのペアを計算
    entries = results[results['Entry_Signal'] == 1].index.tolist()
    exits = results[results['Exit_Signal'] == -1].index.tolist()
    
    total_trades = min(len(entries), len(exits))
    
    # 簡易的な利益計算（エントリー価格とイグジット価格の差）
    if total_trades > 0:
        profits = []
        for i in range(total_trades):
            entry_price = results.loc[entries[i], 'Close']
            exit_price = results.loc[exits[i], 'Close']
            profit = exit_price - entry_price
            profits.append(profit)
        
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])
        win_rate = (winning_trades / total_trades * 100)
        total_profit = sum(profits)
        avg_profit = total_profit / total_trades
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        total_profit = 0
        avg_profit = 0
    
    # 結果表示
    print("\n" + "="*80)
    print("バックテスト結果（修正後）:")
    print("="*80)
    print(f"総取引回数: {total_trades}")
    print(f"勝ちトレード: {winning_trades}")
    print(f"負けトレード: {losing_trades}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"総利益: {total_profit:.2f}円")
    print(f"平均利益: {avg_profit:.2f}円")
    
    # 修正前の結果を読み込んで比較
    try:
        with open(r"c:\Users\imega\Documents\my_backtest_project\contrarian_before_fix_result.json", 'r', encoding='utf-8') as f:
            before_data = json.load(f)
        
        print("\n" + "="*80)
        print("修正前後の比較:")
        print("="*80)
        print(f"取引回数: {before_data['total_trades']} -> {total_trades} (変化: {total_trades - before_data['total_trades']})")
        print(f"勝率: {before_data['win_rate']:.2f}% -> {win_rate:.2f}% (変化: {win_rate - before_data['win_rate']:.2f}%)")
        print(f"総利益: {before_data['total_profit']:.2f}円 -> {total_profit:.2f}円 (変化: {total_profit - before_data['total_profit']:.2f}円)")
        print(f"平均利益: {before_data['avg_profit']:.2f}円 -> {avg_profit:.2f}円 (変化: {avg_profit - before_data['avg_profit']:.2f}円)")
    except FileNotFoundError:
        print("\n修正前の結果ファイルが見つかりません")
    
    # 結果をファイルに保存
    output_data = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "period": f"{start_date} to {end_date}",
        "status": "after_fix",
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate": float(win_rate),
        "total_profit": float(total_profit),
        "avg_profit": float(avg_profit),
        "parameters": params
    }
    
    # JSON保存
    json_path = r"c:\Users\imega\Documents\my_backtest_project\contrarian_after_fix_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n結果をJSONで保存しました: {json_path}")
    
    # CSV保存
    if total_trades > 0:
        csv_path = r"c:\Users\imega\Documents\my_backtest_project\contrarian_after_fix_trades.csv"
        results.to_csv(csv_path, index=True, encoding='utf-8-sig')
        print(f"取引詳細をCSVで保存しました: {csv_path}")
    
    return output_data

if __name__ == "__main__":
    try:
        result = test_contrarian_after_fix()
        if result:
            print("\n修正後テスト完了")
    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
