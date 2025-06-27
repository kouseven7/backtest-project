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
        logger.warning("損益推移データが空です。0を返します")
        return 0.0  # -infではなく0を返す
    
    if "日次損益" not in pnl_summary.columns:
        logger.warning("「日次損益」カラムが見つかりません。0を返します")
        logger.info(f"利用可能なカラム: {list(pnl_summary.columns)}")
        return 0.0  # -infではなく0を返す
    
    daily_returns = pnl_summary["日次損益"]
    logger.info(f"日次損益データのサンプル: {daily_returns.head()}")
    logger.info(f"日次損益の統計: ゼロ以外: {len(daily_returns[daily_returns != 0])}, 全体: {len(daily_returns)}")
    
    if len(daily_returns) == 0 or daily_returns.isna().all():
        logger.warning("日次損益データが無効です。0を返します")
        return 0.0
    
    try:
        sharpe = calculate_sharpe_ratio(daily_returns)
        logger.info(f"計算されたシャープレシオ: {sharpe}")
        
        # NaNや無限値をチェック
        if np.isnan(sharpe) or np.isinf(sharpe):
            logger.warning(f"シャープレシオが無効な値です: {sharpe}。0を返します")
            return 0.0
            
        return sharpe
    except Exception as e:
        logger.error(f"シャープレシオ計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す

def sortino_ratio_objective(trade_results: Dict) -> float:
    """
    ソルティノレシオを最大化する目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: ソルティノレシオ（高いほど良い）
    """
    logger = logging.getLogger(__name__)
    logger.info("ソルティノレシオ目的関数が呼ばれました")
    
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty or "日次損益" not in pnl_summary.columns:
        logger.warning("損益推移データが空または日次損益カラムがありません。0を返します")
        return 0.0  # -infではなく0を返す
    
    daily_returns = pnl_summary["日次損益"]
    
    if len(daily_returns) == 0 or daily_returns.isna().all():
        logger.warning("日次損益データが無効です。0を返します")
        return 0.0
    
    try:
        sortino = calculate_sortino_ratio(daily_returns)
        logger.info(f"計算されたソルティノレシオ: {sortino}")
        
        # NaNや無限値をチェック
        if np.isnan(sortino) or np.isinf(sortino):
            logger.warning(f"ソルティノレシオが無効な値です: {sortino}。0を返します")
            return 0.0
            
        return sortino
    except Exception as e:
        logger.error(f"ソルティノレシオ計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す

def risk_adjusted_return_objective(trade_results: Dict) -> float:
    """
    リスク調整後リターンを最大化する複合目的関数
    シャープレシオ - (最大ドローダウン * 2)
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: リスク調整後リターン（高いほど良い）
    """
    logger = logging.getLogger(__name__)
    logger.info("リスク調整後リターン目的関数が呼ばれました")
    
    pnl_summary = trade_results.get("損益推移", pd.DataFrame())
    if pnl_summary.empty or "日次損益" not in pnl_summary.columns or "累積損益" not in pnl_summary.columns:
        logger.warning("損益推移データが空または必要なカラムがありません。0を返します")
        return 0.0  # -infではなく0を返す
    
    daily_returns = pnl_summary["日次損益"]
    cumulative_returns = pnl_summary["累積損益"]
    
    try:
        sharpe = calculate_sharpe_ratio(daily_returns)
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        # 計算値のチェック
        if np.isnan(sharpe) or np.isinf(sharpe):
            logger.warning(f"シャープレシオが無効な値です: {sharpe}。0を返します")
            sharpe = 0.0
            
        if np.isnan(max_dd) or np.isinf(max_dd):
            logger.warning(f"最大ドローダウンが無効な値です: {max_dd}。0を返します")
            max_dd = 0.0
        
        # max_ddは百分率なので、0.01を掛けて調整
        # ドローダウンに対するペナルティを2倍に設定
        score = sharpe - (max_dd * 0.01 * 2)
        logger.info(f"リスク調整後リターン: {score} (シャープ={sharpe}, 最大ドローダウン={max_dd}%)")
        return score
    except Exception as e:
        logger.error(f"リスク調整後リターン計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す

def win_rate_expectancy_objective(trade_results: Dict) -> float:
    """
    勝率と期待値を組み合わせた目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: 勝率と期待値の組み合わせスコア（高いほど良い）
    """
    logger = logging.getLogger(__name__)
    logger.info("勝率と期待値の組み合わせ目的関数が呼ばれました")
    
    trades = trade_results.get("取引履歴", pd.DataFrame())
    if trades.empty:
        logger.warning("取引履歴が空です。0を返します")
        return 0.0  # -infではなく0を返す
    
    try:
        win_rate = calculate_win_rate(trades)
        expectancy = calculate_expectancy(trades)
        
        # NaNや無限値をチェック
        if np.isnan(win_rate) or np.isinf(win_rate):
            logger.warning(f"勝率が無効な値です: {win_rate}。0を使用します")
            win_rate = 0.0
            
        if np.isnan(expectancy) or np.isinf(expectancy):
            logger.warning(f"期待値が無効な値です: {expectancy}。0を使用します")
            expectancy = 0.0
        
        # 勝率（0～1）と期待値を組み合わせる
        # 期待値は通常小さい値なので10倍して重みを付ける
        score = win_rate + (expectancy * 10)
        logger.info(f"勝率と期待値の組み合わせスコア: {score} (勝率={win_rate}, 期待値={expectancy})")
        return score
    except Exception as e:
        logger.error(f"勝率と期待値の組み合わせ計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す

def win_rate_objective(trade_results: Dict) -> float:
    """
    勝率を最大化する目的関数
    
    Parameters:
        trade_results (Dict): バックテスト結果
        
    Returns:
        float: 勝率（高いほど良い）
    """
    logger = logging.getLogger(__name__)
    logger.info("勝率目的関数が呼ばれました")
    
    trades = trade_results.get("取引履歴", pd.DataFrame())
    if trades.empty:
        logger.warning("取引履歴が空です。0を返します")
        return 0.0
    
    try:    
        win_rate = calculate_win_rate(trades)
        
        # NaNや無限値をチェック
        if np.isnan(win_rate) or np.isinf(win_rate):
            logger.warning(f"勝率が無効な値です: {win_rate}。0を返します")
            return 0.0
            
        logger.info(f"計算された勝率: {win_rate}%")
        return win_rate
    except Exception as e:
        logger.error(f"勝率計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す

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
        logger.warning("取引履歴が空です、0を返します")
        return 0.0  # -infではなく0を返す
    
    logger.info(f"取引数: {len(trades)}, カラム: {list(trades.columns)}")
    
    if '取引結果' not in trades.columns:
        logger.warning("「取引結果」カラムが見つかりません。0を返します")
        return 0.0  # -infではなく0を返す
        
    # NaNや無限値をチェックして置換
    trades_copy = trades.copy()
    if trades_copy['取引結果'].isna().any() or trades_copy['取引結果'].isin([np.inf, -np.inf]).any():
        logger.warning("取引結果にNaNまたは無限値が含まれています。置換します。")
        trades_copy['取引結果'] = trades_copy['取引結果'].fillna(0).replace([np.inf, -np.inf], 0)
    
    total_trades = len(trades_copy)
    if total_trades == 0:
        logger.warning("取引データがありません。0を返します")
        return 0.0  # -infではなく0を返す
        
    try:
        total_profit = trades_copy['取引結果'].sum()
        expectancy = total_profit / total_trades if total_trades > 0 else 0.0
        
        logger.info(f"合計損益: {total_profit}, 期待値: {expectancy}")
        
        # 大きな負の値や無限値のチェック
        if np.isnan(expectancy) or np.isinf(expectancy):
            logger.warning(f"期待値計算でNaNまたは無限値が発生しました: {expectancy}")
            return 0.0  # 常に0を返す（-infではなく）
        
        return expectancy
    except Exception as e:
        logger.error(f"期待値計算でエラー発生: {str(e)}")
        return 0.0  # エラー時も0を返す（-infではなく）

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
        valid_scores = 0  # 有効なスコアの数
        
        for i, (objective, weight) in enumerate(self.objectives_with_weights):
            try:
                objective_name = objective.__name__ if hasattr(objective, "__name__") else f"目的関数{i+1}"
                logger.info(f"{objective_name}を評価中...")
                
                objective_score = objective(trade_results)
                
                # スコアの品質チェックと正規化
                if np.isnan(objective_score):
                    logger.warning(f"{objective_name}がNaNを返しました。0として処理します。")
                    objective_score = 0.0
                elif np.isinf(objective_score):
                    if objective_score < 0:
                        logger.warning(f"{objective_name}が-infを返しました。0として処理します。")
                        objective_score = 0.0  # -infは0に変換
                    else:
                        logger.warning(f"{objective_name}が+infを返しました。大きな正の値として処理します。")
                        objective_score = 1000.0  # +infは大きな値に変換
                elif objective_score < -1000:
                    # 極端に小さな値は制限する
                    logger.warning(f"{objective_name}が極端に小さな値を返しました: {objective_score}。制限値に置き換えます。")
                    objective_score = -1000.0
                elif objective_score > 1000:
                    # 極端に大きな値は制限する
                    logger.warning(f"{objective_name}が極端に大きな値を返しました: {objective_score}。制限値に置き換えます。")
                    objective_score = 1000.0
                
                # 有効なスコアをカウント (非ゼロのスコア)
                if abs(objective_score) > 1e-10:  # 実質的にゼロでなければ
                    valid_scores += 1
                    
                logger.info(f"{objective_name}のスコア: {objective_score}, 重み: {weight}, 寄与: {objective_score * weight}")
                score += weight * objective_score
                
            except Exception as e:                # エラーが発生した場合はログに記録
                logger.error(f"{objective_name}評価中にエラー: {str(e)}")
                # 1つの目的関数のエラーで全体を無効にしないが、ログは残す
                continue
        
        # スコアの後処理
        if np.isnan(score) or np.isinf(score):
            logger.warning(f"複合スコアが無効な値です: {score}。0を返します")
            score = 0.0
        
        # 取引が全くないか極少ない場合、スコアは意味を持たないため調整
        trades = trade_results.get('取引履歴', pd.DataFrame())
        if trades.empty or len(trades) < 3:  # 3件未満の取引は信頼性が低い
            logger.warning(f"取引数が少なすぎます({0 if trades.empty else len(trades)}件)。スコアを下方修正します。")
            # スコアがプラスなら小さく、マイナスならさらに低く
            if score > 0:
                score *= 0.1  # 正のスコアは90%減少
            else:
                score -= 100  # 負のスコアはさらにペナルティ
            
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