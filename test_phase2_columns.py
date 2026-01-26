"""
Phase 2+3カラム追加テストスクリプト

1銘柄のみで高速検証し、Phase 2+3のカラムが正常に生成されるか確認。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.validate_exit_simple_v2 import run_single_backtest, calculate_performance_metrics
import pandas as pd

# テスト対象: 7203.T（トヨタ）、1組み合わせのみ
ticker = "7203.T"
start_date = "2020-01-01"
end_date = "2025-12-31"
exit_params = {
    'stop_loss_pct': 0.03,
    'trailing_stop_pct': 0.10,
    'take_profit_pct': None
}

print("=" * 80)
print("Phase 2+3カラム追加テスト開始")
print(f"銘柄: {ticker}")
print(f"パラメータ: {exit_params}")
print("=" * 80)

# バックテスト実行
metrics, trade_details = run_single_backtest(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    exit_params=exit_params,
    warmup_days=150
)

if trade_details is not None and len(trade_details) > 0:
    print(f"\n取引履歴件数: {len(trade_details)}")
    print(f"カラム数: {len(trade_details.columns)}")
    print(f"\n全カラム一覧:")
    for col in trade_details.columns:
        print(f"  - {col}")
    
    # Phase 2カラムチェック
    phase2_columns = ['r_multiple', 'entry_volume', 'avg_volume_20d', 'volume_ratio', 'exit_gap_pct', 'highest_price_during_hold']
    print(f"\nPhase 2カラム存在確認:")
    for col in phase2_columns:
        exists = col in trade_details.columns
        status = "OK" if exists else "NG"
        print(f"  [{status}] {col}")
    
    # Phase 3カラムチェック
    phase3_columns = ['exit_atr', 'max_gap_during_hold', 'trailing_activated', 'trailing_trigger_price', 'entry_trend_strength', 'sma_distance_pct']
    print(f"\nPhase 3カラム存在確認:")
    for col in phase3_columns:
        exists = col in trade_details.columns
        status = "OK" if exists else "NG"
        print(f"  [{status}] {col}")
    
    # サンプルデータ表示
    print(f"\n最初の3行（Phase 3カラムのみ）:")
    if all(col in trade_details.columns for col in phase3_columns):
        print(trade_details[phase3_columns].head(3).to_string())
    else:
        print("Phase 3カラムが不足しています")
    
    # 統計サマリー
    print(f"\nPhase 2カラム統計:")
    for col in phase2_columns:
        if col in trade_details.columns:
            non_null = trade_details[col].notna().sum()
            print(f"  {col}: 非NULL件数 {non_null}/{len(trade_details)}")
    
    print(f"\nPhase 3カラム統計:")
    for col in phase3_columns:
        if col in trade_details.columns:
            non_null = trade_details[col].notna().sum()
            print(f"  {col}: 非NULL件数 {non_null}/{len(trade_details)}")
    
    print("\n" + "=" * 80)
    print("Phase 2+3カラム追加テスト完了")
    print("=" * 80)
else:
    print("エラー: 取引履歴が取得できませんでした")
