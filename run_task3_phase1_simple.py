"""
Task 3 Phase 1簡易実行 - トレーリングストップ最適化（7月単一銘柄テスト）

7月最多取引銘柄（8411）で5パターンのtrailing_stop_pct/take_profitをテスト。
高速実行（5分）で傾向を把握し、Phase 2（11ヶ月DSSMS）へ。

Author: Backtest Project Team
Created: 2026-01-16
"""
import sys
import os
from datetime import datetime
import pandas as pd

# プロジェクトルート追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from strategies.gc_strategy_signal import GCStrategy
from data_fetcher import get_parameters_and_data

# テストパラメータ（5パターン）
test_patterns = [
    {"name": "A", "trailing_stop_pct": 0.03, "take_profit": 0.03},
    {"name": "B", "trailing_stop_pct": 0.05, "take_profit": 0.05},
    {"name": "C", "trailing_stop_pct": 0.10, "take_profit": 0.10},
    {"name": "D", "trailing_stop_pct": 0.15, "take_profit": 0.15},
    {"name": "E", "trailing_stop_pct": 0.20, "take_profit": 0.20},
]

# 共通パラメータ
common_params = {
    "short_window": 5,
    "long_window": 25,
    "stop_loss": 0.03,  # 固定
    "max_hold_days": 300,  # 実質無効化
    "exit_on_death_cross": True,
    "trend_filter_enabled": False,
}

# テスト期間・銘柄
test_start = "2025-07-01"
test_end = "2025-07-31"
test_ticker = "8411.T"  # 7月最多取引銘柄

print("=" * 80)
print("Task 3 Phase 1: トレーリングストップ最適化（7月単一銘柄テスト）")
print("=" * 80)
print(f"テスト期間: {test_start} ~ {test_end}")
print(f"テスト銘柄: {test_ticker}")
print(f"パターン数: {len(test_patterns)}")
print("")

# 結果保存用
results_summary = []

# 各パターンでバックテスト実行
for pattern in test_patterns:
    print("-" * 80)
    print(f"パターン{pattern['name']}: trailing_stop_pct={pattern['trailing_stop_pct']*100}%, take_profit={pattern['take_profit']*100}%")
    print("-" * 80)
    
    # パラメータ統合
    params = {**common_params, **pattern}
    
    try:
        # データ取得（2025-01-01から取得し、7月のみバックテスト）
        ticker, start_date_str, end_date_str, stock_data, index_data = get_parameters_and_data(
            ticker=test_ticker,
            start_date="2025-01-01",
            end_date=test_end,
            warmup_days=150
        )
        
        # GC戦略初期化
        strategy = GCStrategy(stock_data, params=params, ticker=ticker)
        strategy.initialize_strategy()
        
        # 7月のみバックテスト実行（pd.Timestamp化）
        backtest_results = strategy.backtest(
            trading_start_date=pd.Timestamp(test_start),
            trading_end_date=pd.Timestamp(test_end)
        )
        
        # デバッグ：backtest_resultsの詳細確認
        print(f"  [DEBUG] backtest_results type: {type(backtest_results)}")
        if backtest_results is not None:
            print(f"  [DEBUG] backtest_results shape: {backtest_results.shape}")
            print(f"  [DEBUG] backtest_results columns: {backtest_results.columns.tolist()}")
            if len(backtest_results) > 0:
                print(f"  [DEBUG] First 3 rows:\n{backtest_results.head(3)}")
        
        # 結果抽出
        if backtest_results is not None and len(backtest_results) > 0:
            total_trades = len(backtest_results[backtest_results['exit_date'].notna()])
            if total_trades > 0:
                total_pnl = backtest_results['pnl'].sum()
                avg_return = backtest_results['return_pct'].mean()
                win_rate = (backtest_results['pnl'] > 0).sum() / total_trades
                
                print(f"\n[結果]")
                print(f"  総取引数: {total_trades}件")
                print(f"  総損益: {total_pnl:,.0f}円")
                print(f"  平均リターン: {avg_return*100:.2f}%")
                print(f"  勝率: {win_rate*100:.1f}%")
                
                results_summary.append({
                    "pattern": pattern['name'],
                    "trailing_stop_pct": pattern['trailing_stop_pct'],
                    "take_profit": pattern['take_profit'],
                    "total_trades": total_trades,
                    "total_pnl": total_pnl,
                    "avg_return_pct": avg_return * 100,
                    "win_rate": win_rate * 100,
                })
            else:
                print(f"\n[警告] 決済済み取引なし")
                results_summary.append({
                    "pattern": pattern['name'],
                    "trailing_stop_pct": pattern['trailing_stop_pct'],
                    "take_profit": pattern['take_profit'],
                    "total_trades": 0,
                    "total_pnl": 0,
                    "avg_return_pct": 0,
                    "win_rate": 0,
                })
        else:
            print(f"\n[警告] 取引なし")
            results_summary.append({
                "pattern": pattern['name'],
                "trailing_stop_pct": pattern['trailing_stop_pct'],
                "take_profit": pattern['take_profit'],
                "total_trades": 0,
                "total_pnl": 0,
                "avg_return_pct": 0,
                "win_rate": 0,
            })
    
    except Exception as e:
        print(f"\n[エラー] {e}")
        results_summary.append({
            "pattern": pattern['name'],
            "trailing_stop_pct": pattern['trailing_stop_pct'],
            "take_profit": pattern['take_profit'],
            "total_trades": 0,
            "total_pnl": 0,
            "avg_return_pct": 0,
            "win_rate": 0,
        })

# 結果比較
print("\n" + "=" * 80)
print("Phase 1結果サマリー（7月・8411単一銘柄）")
print("=" * 80)

df_results = pd.DataFrame(results_summary)
df_results = df_results.sort_values('total_pnl', ascending=False)

print(df_results.to_string(index=False))

# 最適パターン特定
if len(df_results) > 0 and df_results['total_pnl'].max() > 0:
    best_pattern = df_results.iloc[0]
    print("\n" + "=" * 80)
    print(f"最適パターン: {best_pattern['pattern']}")
    print(f"  trailing_stop_pct: {best_pattern['trailing_stop_pct']*100}%")
    print(f"  take_profit: {best_pattern['take_profit']*100}%")
    print(f"  総損益: {best_pattern['total_pnl']:,.0f}円")
    print(f"  勝率: {best_pattern['win_rate']:.1f}%")
    print("=" * 80)
else:
    print("\n[警告] 全パターンで取引なしまたは損失")

# CSV出力
output_path = "output/task3_phase1_results_simple.csv"
df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[OUTPUT] {output_path} 生成完了")
