import pandas as pd
from metrics.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_expectancy,
    calculate_max_consecutive_losses,
    calculate_max_consecutive_wins,
    calculate_avg_consecutive_losses,
    calculate_avg_consecutive_wins,
    calculate_max_drawdown_during_losses,
    calculate_total_trades,
    calculate_win_rate,
    calculate_total_profit,
    calculate_average_profit,
    calculate_max_profit,
    calculate_max_loss,
    calculate_max_drawdown,
    calculate_max_drawdown_amount,
    calculate_risk_return_ratio
)

def add_performance_metrics(trade_results: dict) -> dict:
    """
    バックテスト結果にパフォーマンス指標を追加します。

    Parameters:
        trade_results (dict): バックテスト結果の辞書。

    Returns:
        dict: パフォーマンス指標が追加されたバックテスト結果の辞書。
    """
    trade_history = trade_results.get("取引履歴", pd.DataFrame())
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())

    if not trade_history.empty:
        daily_returns = pnl_summary["日次損益"] / 1000000  # 総資産を基準にリターンを計算
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        sortino_ratio = calculate_sortino_ratio(daily_returns)
        expectancy = calculate_expectancy(trade_history)
        max_consecutive_losses = calculate_max_consecutive_losses(trade_history)
        max_consecutive_wins = calculate_max_consecutive_wins(trade_history)
        avg_consecutive_losses = calculate_avg_consecutive_losses(trade_history)
        avg_consecutive_wins = calculate_avg_consecutive_wins(trade_history)
        max_drawdown_during_losses = calculate_max_drawdown_during_losses(trade_history)

        # 新しい指標を計算
        total_trades = calculate_total_trades(trade_history)
        win_rate = calculate_win_rate(trade_history)
        total_profit = calculate_total_profit(trade_history)
        average_profit = calculate_average_profit(trade_history)
        max_profit = calculate_max_profit(trade_history)
        max_loss = calculate_max_loss(trade_history)
        max_drawdown = calculate_max_drawdown(pnl_summary["累積損益"])
        max_drawdown_amount = calculate_max_drawdown_amount(pnl_summary["累積損益"])
        risk_return_ratio = calculate_risk_return_ratio(total_profit, max_drawdown)

        performance_metrics = pd.DataFrame({
            "指標": [
                "シャープレシオ", "ソルティノレシオ", "期待値", "最大連敗数", "最大連勝数",
                "平均連敗数", "平均連勝数", "連敗時の最大ドローダウン", "総取引数", "勝率",
                "損益合計", "平均損益", "最大利益", "最大損失", "最大ドローダウン(%)", "最大ドローダウン(金額)", "リスクリターン比率"
            ],
            "値": [
                f"{sharpe_ratio:.2f}",
                f"{sortino_ratio:.2f}",
                f"{expectancy:.2f}円",
                max_consecutive_losses,
                max_consecutive_wins,
                f"{avg_consecutive_losses:.2f}",
                f"{avg_consecutive_wins:.2f}",
                f"{max_drawdown_during_losses:.2f}円",
                total_trades,
                f"{win_rate:.2f}%",
                f"{total_profit:.2f}円",
                f"{average_profit:.2f}円",
                f"{max_profit:.2f}円",
                f"{max_loss:.2f}円",
                f"{max_drawdown:.2f}%",
                f"{max_drawdown_amount:.2f}円",
                f"{risk_return_ratio:.2f}"
            ]
        })
    else:
        performance_metrics = pd.DataFrame({
            "指標": [
                "シャープレシオ", "ソルティノレシオ", "期待値", "最大連敗数", "最大連勝数",
                "平均連敗数", "平均連勝数", "連敗時の最大ドローダウン", "総取引数", "勝率",
                "損益合計", "平均損益", "最大利益", "最大損失", "最大ドローダウン(%)", "最大ドローダウン(金額)", "リスクリターン比率"
            ],
            "値": ["0", "0", "0円", "0", "0", "0", "0", "0円", "0", "0%", "0円", "0円", "0円", "0円", "0%", "0"]
        })

    trade_results["パフォーマンス指標"] = performance_metrics
    return trade_results