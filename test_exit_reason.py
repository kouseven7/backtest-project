"""
Task 1検証スクリプト: Exit_Reason列の動作確認

1銘柄（トヨタ自動車 7203.T）でバックテストを実行し、
Exit_Reason列が正しく記録されることを確認します。

Author: Backtest Project Team
Created: 2026-01-25
"""

import sys
from pathlib import Path

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from data_fetcher import get_parameters_and_data
from strategies.gc_strategy_signal import GCStrategy

def test_exit_reason():
    """Exit_Reason列の動作確認"""
    
    print("=" * 80)
    print("Task 1検証: Exit_Reason列の動作確認")
    print("=" * 80)
    
    # テストパラメータ
    ticker = "7203.T"  # トヨタ自動車
    start_date = "2023-01-01"
    end_date = "2024-12-31"  # 2年間
    
    print(f"\n銘柄: {ticker}")
    print(f"期間: {start_date} ~ {end_date}")
    
    # データ取得
    print("\n[1/3] データ取得中...")
    _, _, _, stock_data, _ = get_parameters_and_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        warmup_days=150
    )
    
    print(f"  データ件数: {len(stock_data)}行")
    
    # GC戦略でバックテスト実行
    print("\n[2/3] バックテスト実行中...")
    strategy_params = {
        'short_window': 5,
        'long_window': 25,
        'stop_loss': 0.05,          # 損切5%
        'trailing_stop_pct': 0.10,  # トレーリング10%
        'take_profit': None         # 利確なし
    }
    
    strategy = GCStrategy(stock_data, params=strategy_params, ticker=ticker)
    results_df = strategy.backtest()
    
    print(f"  バックテスト完了: {len(results_df)}行")
    
    # Exit_Reason列の存在確認
    print("\n[3/3] Exit_Reason列の検証...")
    
    if 'Exit_Reason' not in results_df.columns:
        print("FAIL: Exit_Reason列が存在しません")
        print(f"  存在する列: {list(results_df.columns)}")
        return False
    
    print("SUCCESS: Exit_Reason列が存在します")
    
    # エグジット理由の集計
    exit_trades = results_df[results_df['Exit_Signal'] == -1]
    
    if len(exit_trades) == 0:
        print("FAIL: エグジット取引が0件です")
        return False
    
    print(f"\n総エグジット数: {len(exit_trades)}件")
    print("\nエグジット理由の内訳:")
    
    reason_counts = exit_trades['Exit_Reason'].value_counts()
    for reason, count in reason_counts.items():
        ratio = count / len(exit_trades) * 100
        print(f"  {reason}: {count}件 ({ratio:.1f}%)")
    
    # ペイオフレシオ計算
    if 'Profit_Loss' in results_df.columns:
        winning_trades = exit_trades[exit_trades['Profit_Loss'] > 0]
        losing_trades = exit_trades[exit_trades['Profit_Loss'] <= 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades['Profit_Loss'].mean()
            avg_loss = abs(losing_trades['Profit_Loss'].mean())
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            print(f"\nペイオフレシオ: {payoff_ratio:.2f}")
            print(f"  平均利益: {avg_win:.2f}")
            print(f"  平均損失: {avg_loss:.2f}")
    
    # サンプルデータ表示
    print("\nサンプル取引（最初の5件）:")
    sample_columns = ['Exit_Signal', 'Exit_Reason', 'Profit_Loss']
    available_columns = [col for col in sample_columns if col in exit_trades.columns]
    print(exit_trades[available_columns].head().to_string())
    
    # 成功判定
    success = (
        'Exit_Reason' in results_df.columns and
        len(exit_trades) > 0 and
        'none' not in reason_counts.index  # 'none'以外の理由が存在する
    )
    
    print("\n" + "=" * 80)
    if success:
        print("SUCCESS: Task 1検証成功")
        print("  - Exit_Reason列が記録されています")
        print(f"  - エグジット理由が{len(reason_counts)}種類検出されました")
    else:
        print("FAIL: Task 1検証失敗")
        if 'none' in reason_counts.index:
            print("  - Exit_Reasonが'none'のままの取引があります")
    print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = test_exit_reason()
    sys.exit(0 if success else 1)
