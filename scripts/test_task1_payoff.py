"""
Task 1検証: ペイオフレシオ・エグジット理由テスト（1銘柄）
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validate_exit_simple_v2 import run_single_backtest

# 1銘柄テスト: 7203.T（トヨタ自動車）
ticker = '7203.T'
params = {
    'stop_loss_pct': 0.05,
    'trailing_stop_pct': 0.10,
    'take_profit_pct': None
}

print(f"Task 1検証: {ticker}でペイオフレシオ計算テスト")
print(f"パラメータ: {params}")
print("-" * 60)

metrics, df = run_single_backtest(
    ticker=ticker,
    start_date='2020-01-01',
    end_date='2025-12-31',
    exit_params=params,
    warmup_days=150
)

# 結果表示
print(f"PF: {metrics['profit_factor']:.2f}")
print(f"ペイオフレシオ: {metrics.get('payoff_ratio', 'N/A')}")
print(f"平均利益: {metrics.get('avg_win', 0):.2f}")
print(f"平均損失: {metrics.get('avg_loss', 0):.2f}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"取引数: {metrics['num_trades']}")
print("\nエグジット理由:")
print(f"  損切: {metrics.get('stop_loss_count', 0)}")
print(f"  トレーリングストップ: {metrics.get('trailing_stop_count', 0)}")
print(f"  デッドクロス: {metrics.get('dead_cross_count', 0)}")
print(f"  強制決済: {metrics.get('force_close_count', 0)}")
print(f"  その他: {metrics.get('other_count', 0)}")
print("-" * 60)
print("Task 1検証完了")
