"""
Module: Performance Metrics
File: performance_metrics.py
Description: 
  バックテスト結果に基づいてパフォーマンス指標（シャープレシオ、ソルティレシオ、期待値など）を計算するモジュールです。

Author: imega
Created: 2025-04-29

Dependencies:
  - pandas
  - numpy
"""

import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    シャープレシオを計算する。

    Parameters:
        returns (pd.Series): 日次リターンのシリーズ。
        risk_free_rate (float): 無リスク利子率（デフォルトは0.0）。
        trading_days (int): 年間の取引日数（デフォルトは252）。

    Returns:
        float: シャープレシオ。
    """
    excess_returns = returns - (risk_free_rate / trading_days)
    return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    ソルティノレシオを計算する。

    Parameters:
        returns (pd.Series): 日次リターンのシリーズ。
        risk_free_rate (float): 無リスク利子率（デフォルトは0.0）。
        trading_days (int): 年間の取引日数（デフォルトは252）。

    Returns:
        float: ソルティノレシオ。
    """
    excess_returns = returns - (risk_free_rate / trading_days)
    downside_risk = np.sqrt((excess_returns[excess_returns < 0] ** 2).mean())
    return np.sqrt(trading_days) * excess_returns.mean() / downside_risk

def calculate_expectancy(trade_results: pd.DataFrame) -> float:
    """
    期待値を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 期待値。
    """
    total_trades = len(trade_results)
    if total_trades == 0:
        return 0.0
    total_profit = trade_results['取引結果'].sum()
    return total_profit / total_trades

def calculate_max_consecutive_losses(trade_results: pd.DataFrame) -> int:
    """
    最大連敗数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        int: 最大連敗数。
    """
    losses = (trade_results['取引結果'] < 0).astype(int)
    return (losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)).max()

def calculate_max_consecutive_wins(trade_results: pd.DataFrame) -> int:
    """
    最大連勝数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        int: 最大連勝数。
    """
    wins = (trade_results['取引結果'] > 0).astype(int)
    return (wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)).max()

def calculate_avg_consecutive_losses(trade_results: pd.DataFrame) -> float:
    """
    平均連敗数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 平均連敗数。
    """
    losses = (trade_results['取引結果'] < 0).astype(int)
    streaks = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
    return streaks[streaks > 0].mean()

def calculate_avg_consecutive_wins(trade_results: pd.DataFrame) -> float:
    """
    平均連勝数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 平均連勝数。
    """
    wins = (trade_results['取引結果'] > 0).astype(int)
    streaks = wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)
    return streaks[streaks > 0].mean()

def calculate_max_drawdown_during_losses(trade_results: pd.DataFrame) -> float:
    """
    連敗時の最大ドローダウン（金額）を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 連敗時の最大ドローダウン（金額）。
    """
    losses = trade_results['取引結果'].copy()
    losses[losses > 0] = 0  # 利益を0にする
    cumulative_losses = losses.cumsum()
    return cumulative_losses.min()

def calculate_max_drawdown_amount(cumulative_pnl: pd.Series) -> float:
    """
    最大ドローダウン（金額）を計算する。

    Parameters:
        cumulative_pnl (pd.Series): 累積損益の時系列データ。

    Returns:
        float: 最大ドローダウン（金額）。
    """
    peak = cumulative_pnl.cummax()
    drawdown = peak - cumulative_pnl
    return drawdown.max()

def calculate_total_trades(trade_results: pd.DataFrame) -> int:
    """総取引数を計算する。"""
    return len(trade_results)

def calculate_win_rate(trade_results: pd.DataFrame) -> float:
    """勝率を計算する。"""
    total_trades = len(trade_results)
    if total_trades == 0:
        return 0.0
    win_trades = len(trade_results[trade_results['取引結果'] > 0])
    return (win_trades / total_trades) * 100

def calculate_total_profit(trade_results: pd.DataFrame) -> float:
    """損益合計を計算する。"""
    return trade_results['取引結果'].sum()

def calculate_average_profit(trade_results: pd.DataFrame) -> float:
    """平均損益を計算する。"""
    total_trades = len(trade_results)
    if total_trades == 0:
        return 0.0
    return trade_results['取引結果'].mean()

def calculate_max_profit(trade_results: pd.DataFrame) -> float:
    """最大利益を計算する。"""
    return trade_results['取引結果'].max()

def calculate_max_loss(trade_results: pd.DataFrame) -> float:
    """最大損失を計算する。"""
    return trade_results['取引結果'].min()

def calculate_max_drawdown(cumulative_pnl: pd.Series) -> float:
    """最大ドローダウン（％）を計算する。"""
    peak = cumulative_pnl.cummax()
    drawdown = (peak - cumulative_pnl) / peak * 100
    return drawdown.max()

def calculate_risk_return_ratio(total_profit: float, max_drawdown: float) -> float:
    """リスクリターン比率を計算する。"""
    if max_drawdown == 0:
        return float('inf')
    return total_profit / max_drawdown

if __name__ == "__main__":
    # テスト用のダミーデータ
    trade_data = pd.DataFrame({
        '取引結果': [100, -50, -200, 300, -100, -50, -50, 200, -300, -100]
    })

    max_losses = calculate_max_consecutive_losses(trade_data)
    max_wins = calculate_max_consecutive_wins(trade_data)
    avg_losses = calculate_avg_consecutive_losses(trade_data)
    avg_wins = calculate_avg_consecutive_wins(trade_data)
    max_drawdown = calculate_max_drawdown_during_losses(trade_data)

    print(f"最大連敗数: {max_losses}")
    print(f"最大連勝数: {max_wins}")
    print(f"平均連敗数: {avg_losses}")
    print(f"平均連勝数: {avg_wins}")
    print(f"連敗時の最大ドローダウン: {max_drawdown}")