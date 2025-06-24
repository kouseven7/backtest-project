"""
目的関数のデバッグ用モジュール
最適化中に `inf` が返されている原因を特定するために使用します
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_expectancy

# ロガーの設定
logger = logging.getLogger(__name__)

def diagnose_objective_function(trade_results: Dict[str, Any]) -> None:
    """
    目的関数の診断を行い、なぜ `-inf` が返されるのかを調査します。
    
    Parameters:
        trade_results: バックテスト結果の辞書
    """
    logger.info("目的関数の診断を開始します")
    
    # 1. trade_resultsのキーの確認
    keys = list(trade_results.keys())
    logger.info(f"trade_resultsのキー: {keys}")
    
    # 2. '取引履歴'と'損益推移'の確認
    if '取引履歴' in trade_results:
        trades = trade_results['取引履歴']
        logger.info(f"取引履歴のサイズ: {len(trades)}")
        logger.info(f"取引履歴のカラム: {list(trades.columns)}")
        logger.info(f"取引履歴の例: \n{trades.head()}")
        
        # 取引結果の合計と平均
        if '取引結果' in trades.columns:
            total_profit = trades['取引結果'].sum()
            avg_profit = trades['取引結果'].mean()
            win_count = len(trades[trades['取引結果'] > 0])
            win_rate = win_count / len(trades) * 100 if len(trades) > 0 else 0
            
            logger.info(f"取引合計損益: {total_profit}")
            logger.info(f"取引平均損益: {avg_profit}")
            logger.info(f"勝率: {win_rate:.2f}%")
    else:
        logger.warning("取引履歴が見つかりません")
    
    if '損益推移' in trade_results:
        pnl = trade_results['損益推移']
        logger.info(f"損益推移のサイズ: {len(pnl)}")
        logger.info(f"損益推移のカラム: {list(pnl.columns)}")
        
        # 日次損益の確認
        if '日次損益' in pnl.columns:
            daily_returns = pnl['日次損益']
            non_zero_returns = daily_returns[daily_returns != 0]
            logger.info(f"日次損益の統計: 要素数={len(daily_returns)}, ゼロ以外={len(non_zero_returns)}")
            logger.info(f"日次損益の例: \n{daily_returns.head()}")
            
            # 日次損益に問題がないか確認
            if daily_returns.isna().any():
                logger.warning(f"日次損益にNaN値が含まれています: {daily_returns.isna().sum()}個")
            
            if daily_returns.isin([np.inf, -np.inf]).any():
                logger.warning(f"日次損益に無限大値が含まれています")
                
            # シャープレシオを試しに計算
            try:
                sharpe = calculate_sharpe_ratio(daily_returns)
                logger.info(f"シャープレシオ: {sharpe}")
            except Exception as e:
                logger.error(f"シャープレシオの計算でエラー: {e}")
            
            # ソルティノレシオを試しに計算
            try:
                sortino = calculate_sortino_ratio(daily_returns)
                logger.info(f"ソルティノレシオ: {sortino}")
            except Exception as e:
                logger.error(f"ソルティノレシオの計算でエラー: {e}")
        else:
            logger.warning("損益推移に「日次損益」カラムが見つかりません")
    else:
        logger.warning("損益推移が見つかりません")
    
    logger.info("目的関数の診断を完了しました")
    
def fix_trade_results(trade_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    目的関数の評価のために trade_results を修正します。
    必要に応じてデータの問題を修正します。
    
    Parameters:
        trade_results: バックテスト結果の辞書
        
    Returns:
        修正済みのバックテスト結果
    """
    fixed_results = trade_results.copy()
    
    # 取引履歴の修正
    if '取引履歴' in fixed_results and isinstance(fixed_results['取引履歴'], pd.DataFrame):
        trades = fixed_results['取引履歴']
        
        # NaN値の対処
        if '取引結果' in trades.columns and trades['取引結果'].isna().any():
            logger.warning(f"取引履歴の「取引結果」にNaN値を0に置き換えました")
            trades['取引結果'] = trades['取引結果'].fillna(0)
        
        fixed_results['取引履歴'] = trades
    
    # 損益推移の修正
    if '損益推移' in fixed_results and isinstance(fixed_results['損益推移'], pd.DataFrame):
        pnl = fixed_results['損益推移']
        
        # 日次損益の修正
        if '日次損益' in pnl.columns:
            # NaNをゼロに置換
            if pnl['日次損益'].isna().any():
                logger.warning(f"損益推移の「日次損益」のNaN値を0に置き換えました")
                pnl['日次損益'] = pnl['日次損益'].fillna(0)
            
            # 無限大値を大きな有限値に置換
            inf_mask = pnl['日次損益'].isin([np.inf, -np.inf])
            if inf_mask.any():
                logger.warning(f"損益推移の「日次損益」の無限大値を置き換えました")
                pnl.loc[inf_mask & (pnl['日次損益'] > 0), '日次損益'] = 1000000  # 大きな正の数
                pnl.loc[inf_mask & (pnl['日次損益'] < 0), '日次損益'] = -1000000  # 大きな負の数
        
        fixed_results['損益推移'] = pnl
    
    return fixed_results
