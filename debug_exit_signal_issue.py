"""
Exit_Signal問題のデバッグスクリプト

Exit_Signalが141回記録される問題の原因を特定するため、
実際のデータフレームを詳細に調査します。
"""

import sys
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.gc_strategy_signal import GCStrategy

def main():
    print("=" * 80)
    print("Exit_Signal問題デバッグスクリプト")
    print("=" * 80)
    
    # データ取得
    print("\n[1] データ取得...")
    data_feed = YFinanceDataFeed()
    stock_data = data_feed.get_stock_data(
        ticker="8306.T",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    print(f"  データ行数: {len(stock_data)}")
    
    # 戦略初期化
    print("\n[2] GCStrategy初期化...")
    params = {
        "short_window": 5,
        "long_window": 25,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.03,
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True,
        "trend_filter_enabled": False
    }
    
    strategy = GCStrategy(stock_data, params)
    
    # バックテスト実行
    print("\n[3] バックテスト実行...")
    result = strategy.backtest()
    
    # Entry_Signal分析
    print("\n[4] Entry_Signal分析:")
    entry_signals = result[result['Entry_Signal'] == 1]
    print(f"  Entry_Signal == 1: {len(entry_signals)} 回")
    print("\n  エントリー日付:")
    for idx, row in entry_signals.iterrows():
        print(f"    {idx}: Price={row['Adj Close']:.2f}")
    
    # Exit_Signal分析
    print("\n[5] Exit_Signal分析:")
    exit_signals = result[result['Exit_Signal'] == -1]
    print(f"  Exit_Signal == -1: {len(exit_signals)} 回")
    
    # 最初の10件のExit_Signal
    print("\n  最初の10件のExit_Signal日付:")
    for idx, row in exit_signals.head(10).iterrows():
        print(f"    {idx}: Price={row['Adj Close']:.2f}, Position={row['Position']}")
    
    # Position列の分析
    print("\n[6] Position列の分析:")
    position_transitions = result[result['Position'].diff() != 0]
    print(f"  Positionが変化した回数: {len(position_transitions)}")
    print("\n  Position変化:")
    for idx, row in position_transitions.head(20).iterrows():
        print(f"    {idx}: Position={row['Position']}, Entry={row['Entry_Signal']}, Exit={row['Exit_Signal']}")
    
    # エントリーとイグジットのペア確認
    print("\n[7] エントリー/イグジットペアの確認:")
    entry_count = (result['Entry_Signal'] == 1).sum()
    exit_count = (result['Exit_Signal'] == -1).sum()
    print(f"  エントリー: {entry_count} 回")
    print(f"  イグジット: {exit_count} 回")
    print(f"  差分: {exit_count - entry_count} 回")
    
    # 問題の特定
    print("\n[8] 問題の特定:")
    
    # Position==1の期間中にExit_Signal==-1が複数あるか確認
    position_periods = []
    in_position = False
    entry_idx = None
    
    for idx in range(len(result)):
        if not in_position and result['Entry_Signal'].iloc[idx] == 1:
            in_position = True
            entry_idx = idx
            position_periods.append({
                'entry_idx': entry_idx,
                'entry_date': result.index[idx],
                'exit_count': 0
            })
        elif in_position and result['Exit_Signal'].iloc[idx] == -1:
            position_periods[-1]['exit_count'] += 1
            if result['Position'].iloc[idx] == 0:
                in_position = False
    
    print(f"\n  ポジション期間: {len(position_periods)}")
    for i, period in enumerate(position_periods, 1):
        print(f"  期間#{i}: entry_date={period['entry_date']}, Exit_Signal回数={period['exit_count']}")
    
    # 結果をCSVに保存
    print("\n[9] 結果をCSVに保存...")
    output_file = "tests/results/debug_exit_signals.csv"
    result.to_csv(output_file)
    print(f"  保存先: {output_file}")
    
    print("\n" + "=" * 80)
    print("デバッグ完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
