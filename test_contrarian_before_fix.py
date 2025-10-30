"""
contrarian_strategy修正前のバックテストテスト

ルックアヘッドバイアス修正前の結果を記録します。
修正後の結果と比較するための基準データを取得します。

主な機能:
- contrarian_strategy.pyの現状でのバックテスト実行
- 取引回数、勝率、総利益の記録
- CSV形式での結果保存

統合コンポーネント:
- strategies/contrarian_strategy.py: テスト対象戦略
- データ取得: yfinanceまたはキャッシュ

セーフティ機能/注意事項:
- 実際のバックテストを実行（推測なし）
- 結果をファイルに保存して後で比較可能
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

def test_contrarian_before_fix():
    """修正前のcontrarian_strategyをテスト"""
    
    print("="*80)
    print("contrarian_strategy 修正前バックテスト")
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
    
    # 戦略パラメータ
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
    print("バックテスト結果（修正前）:")
    print("="*80)
    print(f"総取引回数: {total_trades}")
    print(f"勝ちトレード: {winning_trades}")
    print(f"負けトレード: {losing_trades}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"総利益: {total_profit:.2f}円")
    print(f"平均利益: {avg_profit:.2f}円")
    
    # 結果をファイルに保存
    output_data = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "period": f"{start_date} to {end_date}",
        "status": "before_fix",
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate": float(win_rate),
        "total_profit": float(total_profit),
        "avg_profit": float(avg_profit),
        "parameters": params
    }
    
    # JSON保存
    json_path = r"c:\Users\imega\Documents\my_backtest_project\contrarian_before_fix_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n結果をJSONで保存しました: {json_path}")
    
    # CSV保存
    if total_trades > 0:
        csv_path = r"c:\Users\imega\Documents\my_backtest_project\contrarian_before_fix_trades.csv"
        results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"取引詳細をCSVで保存しました: {csv_path}")
    
    return output_data

if __name__ == "__main__":
    try:
        result = test_contrarian_before_fix()
        if result:
            print("\n修正前テスト完了")
    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
