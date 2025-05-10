"""
最適化で使用する様々な目的関数を提供します。
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union
from metrics.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,  # この関数が必要
    calculate_risk_return_ratio,
    calculate_expectancy
)

def sharpe_ratio_objective(trade_results: Dict) -> float:
    """
    シャープレシオを最大化する目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: シャープレシオ（高いほど良い）
    """
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty or "日次損益" not in pnl_summary.columns:
        return -np.inf
    
    daily_returns = pnl_summary["日次損益"]
    return calculate_sharpe_ratio(daily_returns)

def sortino_ratio_objective(trade_results: Dict) -> float:
    """
    ソルティノレシオを最大化する目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: ソルティノレシオ（高いほど良い）
    """
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty or "日次損益" not in pnl_summary.columns:
        return -np.inf
    
    daily_returns = pnl_summary["日次損益"]
    return calculate_sortino_ratio(daily_returns)

def risk_adjusted_return_objective(trade_results: Dict) -> float:
    """
    リスク調整後リターンを最大化する複合目的関数
    シャープレシオ - (最大ドローダウン * 2)
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: リスク調整後リターン（高いほど良い）
    """
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty or "日次損益" not in pnl_summary.columns or "累積損益" not in pnl_summary.columns:
        return -np.inf
    
    daily_returns = pnl_summary["日次損益"]
    cumulative_returns = pnl_summary["累積損益"]
    
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    
    # max_ddは百分率なので、0.01を掛けて調整
    # ドローダウンに対するペナルティを2倍に設定
    return sharpe - (max_dd * 0.01 * 2)

def win_rate_expectancy_objective(trade_results: Dict) -> float:
    """
    勝率と期待値を組み合わせた目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: 勝率と期待値の組み合わせスコア（高いほど良い）
    """
    trades = trade_results.get("取引履歴", pd.DataFrame())
    if trades.empty:
        return -np.inf
    
    win_rate = calculate_win_rate(trades)
    expectancy = calculate_expectancy(trades)
    
    # 勝率（0～1）と期待値を組み合わせる
    # 期待値は通常小さい値なので10倍して重みを付ける
    return win_rate + (expectancy * 10)

def win_rate_objective(trade_results: Dict) -> float:
    """
    勝率を最大化する目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: 勝率（高いほど良い）
    """
    trades = trade_results.get("取引履歴", pd.DataFrame())
    if trades.empty:
        return 0.0
        
    win_rate = calculate_win_rate(trades)
    return win_rate

class CompositeObjective:
    """
    複数の目的関数を重み付けして組み合わせるクラス
    """
    def __init__(self, objectives_with_weights: List[tuple]):
        """
        複合目的関数の初期化
        
        Parameters:
            objectives_with_weights (List[tuple]): (目的関数, 重み) のリスト
        """
        self.objectives_with_weights = objectives_with_weights
        
    def __call__(self, trade_results: Dict) -> float:
        """
        複合目的関数の評価
        
        Parameters:
            trade_results (Dict): バックテスト結果
            
        Returns:
            float: 複合評価スコア
        """
        score = 0.0
        for objective, weight in self.objectives_with_weights:
            objective_score = objective(trade_results)
            score += objective_score * weight
            
        return score

def create_custom_objective(objectives_config: List[Dict[str, Union[str, float]]]) -> Callable:
    """
    設定に基づいてカスタム目的関数を作成する
    
    Parameters:
        objectives_config (List[Dict]): 目的関数の設定リスト
            例: [{"name": "sharpe_ratio", "weight": 1.0}, {"name": "max_drawdown", "weight": -0.5}]
            
    Returns:
        Callable: 複合目的関数
    """
    objective_map = {
        "sharpe_ratio": sharpe_ratio_objective,
        "sortino_ratio": sortino_ratio_objective,
        "risk_adjusted_return": risk_adjusted_return_objective,
        "win_rate_expectancy": win_rate_expectancy_objective,
        "win_rate": win_rate_objective  # ← この行を追加
    }
    
    objectives_with_weights = []
    for config in objectives_config:
        name = config["name"]
        weight = config.get("weight", 1.0)
        
        if name in objective_map:
            objectives_with_weights.append((objective_map[name], weight))
        else:
            raise ValueError(f"未知の目的関数名: {name}")
    
    return CompositeObjective(objectives_with_weights)