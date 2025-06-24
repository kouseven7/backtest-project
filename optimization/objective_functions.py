"""
最適化で使用する様々な目的関数を提供します。
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Callable, Any, Optional, Union, TypeVar

# 型ヒントのためのTypeVar
T = TypeVar('T')
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
    logger = logging.getLogger(__name__)
    logger.info("シャープレシオ目的関数が呼ばれました")
    logger.info(f"trade_results keys: {list(trade_results.keys())}")
    
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty:
        logger.error("損益推移データが空です")
        return -np.inf
    
    if "日次損益" not in pnl_summary.columns:
        logger.error("「日次損益」カラムが見つかりません")
        logger.info(f"利用可能なカラム: {list(pnl_summary.columns)}")
        return -np.inf
    
    daily_returns = pnl_summary["日次損益"]
    logger.info(f"日次損益データのサンプル: {daily_returns.head()}")
    logger.info(f"日次損益の統計: ゼロ以外: {len(daily_returns[daily_returns != 0])}, 全体: {len(daily_returns)}")
    
    sharpe = calculate_sharpe_ratio(daily_returns)
    logger.info(f"計算されたシャープレシオ: {sharpe}")
    return sharpe

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

def expectancy_objective(trade_results: Dict) -> float:
    """
    期待値（1トレードあたりの平均損益）を最大化する目的関数

    Parameters:
        trade_results (Dict): バックテスト結果

    Returns:
        float: 期待値（高いほど良い）
    """
    logger = logging.getLogger(__name__)
    logger.info("期待値目的関数が呼ばれました")
    logger.info(f"trade_results keys: {list(trade_results.keys())}")
    
    trades = trade_results.get("取引履歴", pd.DataFrame())
    if trades.empty:
        logger.error("取引履歴が空です")
        return -np.inf
    
    logger.info(f"取引数: {len(trades)}, カラム: {list(trades.columns)}")
    
    if '取引結果' not in trades.columns:
        logger.error("「取引結果」カラムが見つかりません")
        return -np.inf
        
    # NaNや無限値をチェックして置換
    if trades['取引結果'].isna().any() or trades['取引結果'].isin([np.inf, -np.inf]).any():
        logger.warning("取引結果にNaNまたは無限値が含まれています。置換します。")
        trades['取引結果'] = trades['取引結果'].fillna(0).replace([np.inf, -np.inf], 0)
    
    total_trades = len(trades)
    if total_trades == 0:
        logger.error("取引データがありません")
        return -np.inf
        
    try:
        total_profit = trades['取引結果'].sum()
        expectancy = total_profit / total_trades
        
        logger.info(f"合計損益: {total_profit}, 期待値: {expectancy}")
        
        # 大きな負の値や無限値のチェック
        if np.isnan(expectancy) or np.isinf(expectancy):
            logger.warning(f"期待値計算でNaNまたは無限値が発生しました: {expectancy}")
            # 取引自体はあるが期待値が無効な場合は0を返す（-infではなく）
            if total_trades > 0:
                return 0.0
            return -np.inf
        
        return expectancy
    except Exception as e:
        logger.error(f"期待値計算でエラー発生: {str(e)}")
        return -np.inf

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
        logger = logging.getLogger(__name__)
        logger.info("CompositeObjective.__call__ が実行されました")
        
        # 取引データがあるか確認
        if "取引履歴" in trade_results:
            trades = trade_results.get("取引履歴", pd.DataFrame())
            if not trades.empty and '取引結果' in trades.columns:
                trade_count = len(trades)
                total_profit = trades['取引結果'].sum() if not trades['取引結果'].isna().all() else 0
                logger.info(f"取引データ: {trade_count}件, 合計損益: {total_profit}")
        
        score = 0.0
        all_inf = True  # すべての目的関数が-infを返したかのフラグ
        
        for i, (objective, weight) in enumerate(self.objectives_with_weights):
            try:
                objective_name = objective.__name__ if hasattr(objective, "__name__") else f"目的関数{i+1}"
                logger.info(f"{objective_name}を評価中...")
                
                objective_score = objective(trade_results)
                
                if np.isnan(objective_score):
                    logger.warning(f"{objective_name}がNaNを返しました。0として処理します。")
                    objective_score = 0.0
                elif np.isinf(objective_score) and objective_score < 0:
                    logger.warning(f"{objective_name}が-infを返しました。")
                    # -infはそのまま伝播させるが、全部が-infでなければ全体は-infにはならない
                else:
                    all_inf = False  # 少なくとも1つの関数が有効なスコアを返した
                    
                logger.info(f"{objective_name}のスコア: {objective_score}, 重み: {weight}, 寄与: {objective_score * weight}")
                score += weight * objective_score
                
            except Exception as e:                # エラーが発生した場合はログに記録
                logger.error(f"{objective_name}評価中にエラー: {str(e)}")
                # 1つの目的関数のエラーで全体を無効にしないが、ログは残す
                continue
        
        # 取引があるのに全ての目的関数が-infを返した場合は、フォールバックとして0を返す
        if all_inf and score == -np.inf:
            trades = trade_results.get("取引履歴", pd.DataFrame())
            if not trades.empty and len(trades) > 0:
                logger.warning("すべての目的関数が-infを返しましたが、取引がある場合は0を返します")
                return 0.0
        
        logger.info(f"最終スコア: {score}")
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
    logger = logging.getLogger(__name__)
    logger.info(f"カスタム目的関数を作成します: {objectives_config}")
    
    objective_map = {
        "sharpe_ratio": sharpe_ratio_objective,
        "sortino_ratio": sortino_ratio_objective,
        "risk_adjusted_return": risk_adjusted_return_objective,
        "win_rate_expectancy": win_rate_expectancy_objective,
        "win_rate": win_rate_objective,
        "expectancy": expectancy_objective,  # ← 追加
    }
    
    objectives_with_weights = []
    for config in objectives_config:
        name = config["name"]
        weight = config.get("weight", 1.0)
        
        if name in objective_map:
            logger.info(f"目的関数を追加: {name}, 重み: {weight}")
            objectives_with_weights.append((objective_map[name], weight))
        else:
            logger.error(f"未知の目的関数名: {name}")
            raise ValueError(f"未知の目的関数名: {name}")
    
    # 作成された複合目的関数
    composite = CompositeObjective(objectives_with_weights)
    
    # デバッグ用に追加のラッパー関数を作成
    def debug_objective(trade_results: Dict) -> float:
        logger.info("複合目的関数が呼ばれました")
        
        # 個々の目的関数のスコアを記録
        individual_scores = []
        for objective, weight in objectives_with_weights:
            name = next((k for k, v in objective_map.items() if v == objective), "unknown")
            try:
                score = objective(trade_results)
                logger.info(f"目的関数 {name}: スコア={score}, 重み={weight}")
                individual_scores.append((name, score, weight))
            except Exception as e:
                logger.error(f"目的関数 {name} でエラー: {e}")
                individual_scores.append((name, -np.inf, weight))
        
        # 複合スコアを計算
        total_score = composite(trade_results)
        logger.info(f"複合スコア: {total_score}")
        
        # trade_resultsの簡単な確認
        if isinstance(trade_results, dict):
            if "取引履歴" in trade_results:
                trades_df = trade_results["取引履歴"]
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    logger.info(f"取引総数: {len(trades_df)}, 総損益: {trades_df['取引結果'].sum()}")
                else:
                    logger.warning("取引履歴が空またはデータフレームでない")
            else:
                logger.warning("取引履歴が見つからない")
                
        return total_score
    
    return debug_objective