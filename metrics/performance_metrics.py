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
    if returns.empty or returns.isna().all():
        return 0.0
    
    # 取引のあった日（リターンが0でない日）のみを対象に計算
    returns = returns[returns != 0]
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / trading_days)
    mean_return = excess_returns.mean()
    std = excess_returns.std()
    
    # 平均がマイナスで標準偏差が0やNaNの場合は最低値を返す
    if np.isnan(mean_return) or np.isnan(std) or std == 0:
        return 0.0
    
    # 年率換算したシャープレシオを返す
    return np.sqrt(trading_days) * mean_return / std

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
    if returns.empty or returns.isna().all():
        return 0.0
    
    # 取引のあった日（リターンが0でない日）のみを対象に計算
    returns = returns[returns != 0]
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / trading_days)
    mean_return = excess_returns.mean()
    
    # 下方リスク（負のリターンの標準偏差）を計算
    neg_returns = excess_returns[excess_returns < 0]
    if len(neg_returns) == 0:
        # 負のリターンがない場合は最大値を返す
        return 5.0  # 十分に大きな値
    
    downside_risk = np.sqrt((neg_returns ** 2).mean())
    
    # 平均がマイナスや、分母が0やNaNの場合は0を返す
    if np.isnan(mean_return) or np.isnan(downside_risk) or downside_risk == 0:
        return 0.0
    
    # 年率換算したソルティノレシオを返す
    return np.sqrt(trading_days) * mean_return / downside_risk

def calculate_expectancy(trade_results: pd.DataFrame) -> float:
    """
    期待値を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 期待値。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
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
        int: 最大連敗数。'取引結果'カラムが存在しない場合は0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0
        
    losses = (trade_results['取引結果'] < 0).astype(int)
    return (losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)).max()

def calculate_max_consecutive_wins(trade_results: pd.DataFrame) -> int:
    """
    最大連勝数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        int: 最大連勝数。'取引結果'カラムが存在しない場合は0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0
        
    wins = (trade_results['取引結果'] > 0).astype(int)
    return (wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)).max()

def calculate_avg_consecutive_losses(trade_results: pd.DataFrame) -> float:
    """
    平均連敗数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 平均連敗数。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    losses = (trade_results['取引結果'] < 0).astype(int)
    streaks = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
    return streaks[streaks > 0].mean() if len(streaks[streaks > 0]) > 0 else 0.0

def calculate_avg_consecutive_wins(trade_results: pd.DataFrame) -> float:
    """
    平均連勝数を計算する。

    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']

    Returns:
        float: 平均連勝数。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    wins = (trade_results['取引結果'] > 0).astype(int)
    streaks = wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)
    return streaks[streaks > 0].mean() if len(streaks[streaks > 0]) > 0 else 0.0

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
    """総取引数を計算する."""
    return len(trade_results)

def calculate_win_rate(trade_results: pd.DataFrame) -> float:
    """
    勝率を計算する.
    
    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                      必須カラム: ['取引結果']
                                      
    Returns:
        float: 勝率（百分率）。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    total_trades = len(trade_results)
    if total_trades == 0:
        return 0.0
    win_trades = len(trade_results[trade_results['取引結果'] > 0])
    return (win_trades / total_trades) * 100

def calculate_total_profit(trade_results: pd.DataFrame) -> float:
    """
    損益合計を計算する.
    
    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                     必須カラム: ['取引結果']
                                      
    Returns:
        float: 損益合計。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    return trade_results['取引結果'].sum()

def calculate_average_profit(trade_results: pd.DataFrame) -> float:
    """
    平均損益を計算する.
    
    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                     必須カラム: ['取引結果']
                                      
    Returns:
        float: 平均損益。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    total_trades = len(trade_results)
    if total_trades == 0:
        return 0.0
    return trade_results['取引結果'].mean()

def calculate_max_profit(trade_results: pd.DataFrame) -> float:
    """
    最大利益を計算する.
    
    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                     必須カラム: ['取引結果']
                                      
    Returns:
        float: 最大利益。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    # 取引結果がない場合は0.0を返す
    if len(trade_results) == 0:
        return 0.0
        
    return trade_results['取引結果'].max()

def calculate_max_loss(trade_results: pd.DataFrame) -> float:
    """
    最大損失を計算する.
    
    Parameters:
        trade_results (pd.DataFrame): 取引結果を含むデータフレーム。
                                     必須カラム: ['取引結果']
                                      
    Returns:
        float: 最大損失（マイナスの値）。'取引結果'カラムが存在しない場合は0.0を返す。
    """
    # '取引結果'カラムが存在するか確認
    if '取引結果' not in trade_results.columns:
        return 0.0
        
    # 取引結果がない場合は0.0を返す
    if len(trade_results) == 0:
        return 0.0
        
    return trade_results['取引結果'].min()

def calculate_max_drawdown(cumulative_pnl: pd.Series) -> float:
    """
    最大ドローダウン（％）を計算する.
    
    Parameters:
        cumulative_pnl (pd.Series): 累積損益のシリーズ
                                      
    Returns:
        float: 最大ドローダウン（％）。累積損益が空またはすべて0の場合は0.0を返す。
    """
    # 累積損益が空の場合は0.0を返す
    if cumulative_pnl is None or cumulative_pnl.empty:
        return 0.0
    
    # 累積損益がすべて0の場合は0.0を返す
    if (cumulative_pnl == 0).all():
        return 0.0
        
    peak = cumulative_pnl.cummax()
    
    # peakが0の場合は分母が0になるのを防ぐ
    valid_indices = peak != 0
    if not valid_indices.any():
        return 0.0
        
    drawdown = (peak[valid_indices] - cumulative_pnl[valid_indices]) / peak[valid_indices] * 100
    return drawdown.max() if not drawdown.empty else 0.0

def calculate_risk_return_ratio(total_profit: float, max_drawdown: float) -> float:
    """リスクリターン比率を計算する."""
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