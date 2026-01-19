"""
2024年バックテスト計算検証スクリプト

目的: output/dssms_integration/dssms_20260119_161822/all_transactions.csvから
      手計算で総損益を検証し、comprehensive_report.txtの矛盾を解明

Author: Backtest Project Team
Created: 2026-01-19
"""

import pandas as pd
from pathlib import Path

def verify_2024_calculation():
    """2024年バックテスト結果の計算を検証"""
    
    # ファイルパス
    csv_path = Path("output/dssms_integration/dssms_20260119_161822/all_transactions.csv")
    
    if not csv_path.exists():
        print(f"エラー: ファイルが見つかりません - {csv_path}")
        return
    
    # CSVを読み込み
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("2024年バックテスト計算検証")
    print("=" * 80)
    print(f"ファイル: {csv_path}")
    print(f"総取引数: {len(df)}\n")
    
    # 基本統計
    print("1. 基本統計")
    print("-" * 40)
    print(f"初期資金: ¥1,000,000（想定）")
    print(f"取引数（全て）: {len(df)}")
    
    # 決済済み取引と未決済取引を分離
    completed_trades = df[df['exit_date'].notna()]
    open_trades = df[df['exit_date'].isna()]
    
    print(f"決済済み取引: {len(completed_trades)}")
    print(f"未決済取引: {len(open_trades)}\n")
    
    # 2. 決済済み取引の損益計算
    print("2. 決済済み取引の損益")
    print("-" * 40)
    
    if len(completed_trades) > 0:
        total_pnl_completed = completed_trades['pnl'].sum()
        winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
        break_even_trades = len(completed_trades[completed_trades['pnl'] == 0])
        
        print(f"決済済み取引のPnL合計: ¥{total_pnl_completed:,.2f}")
        print(f"勝ちトレード: {winning_trades}件")
        print(f"負けトレード: {losing_trades}件")
        print(f"ブレークイーブン: {break_even_trades}件")
        
        if len(completed_trades) > 0:
            win_rate = (winning_trades / len(completed_trades)) * 100
            print(f"勝率: {win_rate:.2f}%")
        
        # 総利益・総損失の詳細
        profits = completed_trades[completed_trades['pnl'] > 0]['pnl']
        losses = completed_trades[completed_trades['pnl'] < 0]['pnl']
        
        total_profit = profits.sum() if len(profits) > 0 else 0
        total_loss = losses.sum() if len(losses) > 0 else 0
        
        print(f"\n総利益: ¥{total_profit:,.2f}")
        print(f"総損失: ¥{total_loss:,.2f}")
        print(f"純利益: ¥{total_pnl_completed:,.2f}")
    else:
        total_pnl_completed = 0
        print("決済済み取引なし")
    
    # 3. 未決済取引の詳細
    print("\n3. 未決済取引の詳細")
    print("-" * 40)
    
    if len(open_trades) > 0:
        for idx, trade in open_trades.iterrows():
            print(f"銘柄: {trade['symbol']}")
            print(f"エントリー日: {trade['entry_date']}")
            print(f"エントリー価格: ¥{trade['entry_price']:,.2f}")
            print(f"株数: {trade['shares']:.0f}")
            print(f"ポジション価値: ¥{trade['position_value']:,.2f}")
            print(f"PnL（CSV記載）: ¥{trade['pnl']:,.2f}")
            print()
        
        # 未決済ポジションの評価
        print("未決済ポジションの評価:")
        print("※終値データがないため、含み損益は不明")
        print("※CSV上のPnLは0.0（未決済のため）")
    else:
        print("未決済取引なし")
    
    # 4. 最終資本の計算（2パターン）
    print("\n4. 最終資本の計算")
    print("-" * 40)
    
    initial_capital = 1000000
    
    # パターンA: 決済済み取引のみ
    final_capital_a = initial_capital + total_pnl_completed
    return_a = ((final_capital_a - initial_capital) / initial_capital) * 100
    
    print(f"パターンA（決済済み取引のみ）:")
    print(f"  最終資本: ¥{final_capital_a:,.2f}")
    print(f"  総リターン: {return_a:+.2f}%")
    
    # パターンB: 未決済ポジションの評価を含む（仮定）
    if len(open_trades) > 0:
        # 未決済ポジションの価値を初期投資として差し引き
        open_position_cost = open_trades['position_value'].sum()
        
        print(f"\nパターンB（未決済ポジション考慮）:")
        print(f"  決済済みPnL: ¥{total_pnl_completed:,.2f}")
        print(f"  未決済ポジション投入資金: ¥{open_position_cost:,.2f}")
        print(f"  残現金: ¥{initial_capital + total_pnl_completed - open_position_cost:,.2f}")
        print(f"  ※未決済ポジションの含み損益は不明（終値データなし）")
    
    # 5. comprehensive_report.txtとの比較
    print("\n5. comprehensive_report.txtの報告値との比較")
    print("-" * 40)
    print("報告値:")
    print("  最終ポートフォリオ値: ¥946,596.375")
    print("  総リターン: -5.34%")
    print("  純利益: ¥52,335")
    print("  総利益: ¥184,856")
    print("  総損失: ¥-132,521")
    print()
    print("CSV手計算値:")
    print(f"  純利益（決済済み）: ¥{total_pnl_completed:,.2f}")
    print(f"  総利益（決済済み）: ¥{total_profit:,.2f}" if len(completed_trades) > 0 else "  総利益: N/A")
    print(f"  総損失（決済済み）: ¥{total_loss:,.2f}" if len(completed_trades) > 0 else "  総損失: N/A")
    
    # 矛盾点の指摘
    print("\n矛盾点:")
    if len(completed_trades) > 0:
        pnl_diff = abs(total_pnl_completed - 52335)
        profit_diff = abs(total_profit - 184856)
        loss_diff = abs(total_loss - (-132521))
        
        if pnl_diff > 1:
            print(f"  ❌ 純利益が不一致: CSV={total_pnl_completed:,.2f} vs 報告=52,335 (差: {pnl_diff:,.2f})")
        else:
            print(f"  ✅ 純利益は一致: ¥{total_pnl_completed:,.2f}")
        
        if profit_diff > 1:
            print(f"  ❌ 総利益が不一致: CSV={total_profit:,.2f} vs 報告=184,856 (差: {profit_diff:,.2f})")
        else:
            print(f"  ✅ 総利益は一致: ¥{total_profit:,.2f}")
        
        if loss_diff > 1:
            print(f"  ❌ 総損失が不一致: CSV={total_loss:,.2f} vs 報告=-132,521 (差: {loss_diff:,.2f})")
        else:
            print(f"  ✅ 総損失は一致: ¥{total_loss:,.2f}")
        
        # 最終資本の矛盾
        if abs(final_capital_a - 946596.375) > 1:
            capital_diff = final_capital_a - 946596.375
            print(f"\n  ❗ 最終資本の不一致:")
            print(f"     CSV計算: ¥{final_capital_a:,.2f}")
            print(f"     報告値: ¥946,596.375")
            print(f"     差額: ¥{capital_diff:,.2f}")
            print(f"\n  🔍 推測される原因:")
            
            if len(open_trades) > 0:
                print(f"     1. 未決済ポジション（{len(open_trades)}件）の含み損が{abs(capital_diff):,.2f}円")
                print(f"     2. 未決済ポジションの終値評価が含まれていない可能性")
                print(f"     3. 未決済ポジション投入資金: ¥{open_trades['position_value'].sum():,.2f}")
            else:
                print(f"     1. 計算ロジックのバグ")
                print(f"     2. 初期資本の設定ミス")
                print(f"     3. 手数料・スリッページの考慮")
    
    print("\n" + "=" * 80)
    print("検証完了")
    print("=" * 80)

if __name__ == "__main__":
    verify_2024_calculation()
