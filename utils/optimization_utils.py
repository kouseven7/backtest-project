"""
最適化プロセスのためのユーティリティ関数を提供するモジュール

このモジュールには、最適化プロセスの品質向上と可視化のための
様々なユーティリティ関数が含まれています。
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import json


def safe_score_calculation(func: Callable) -> Callable:
    """
    目的関数を安全に実行するためのデコレータ
    - 例外をキャッチし、適切なデフォルト値を返す
    - NaN/inf値を適切に処理
    - ロギングを強化
    
    Parameters:
        func (Callable): 装飾する目的関数
        
    Returns:
        Callable: 装飾された関数
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            # 元の関数を実行
            result = func(*args, **kwargs)
            
            # inf/NaN値のチェック
            if pd.isna(result) or np.isinf(result):
                logger.warning(f"目的関数 {func.__name__} が無効な値を返しました: {result}。0.0に変換します。")
                return 0.0
            
            # 極端な値のキャップ
            if result > 1e6:
                logger.warning(f"目的関数 {func.__name__} が極端に大きい値を返しました: {result}。1e6にキャップします。")
                return 1e6
            elif result < -1e6:
                logger.warning(f"目的関数 {func.__name__} が極端に小さい値を返しました: {result}。-1e6にキャップします。")
                return -1e6
                
            return result
        except Exception as e:
            logger.exception(f"目的関数 {func.__name__} の実行中にエラーが発生しました: {str(e)}")
            return 0.0
    
    return wrapper


def validate_optimization_results(results_df: pd.DataFrame, 
                                  param_grid: Dict[str, List[Any]],
                                  min_trades: int = 10) -> pd.DataFrame:
    """
    最適化結果の検証と要約を行う
    
    Parameters:
        results_df (pd.DataFrame): 最適化結果のDataFrame
        param_grid (Dict[str, List[Any]]): 使用されたパラメータグリッド
        min_trades (int): 最低限必要な取引数
        
    Returns:
        pd.DataFrame: 検証済みの最適化結果
    """
    logger = logging.getLogger(__name__)
    
    if results_df.empty:
        logger.warning("最適化結果が空です。")
        return results_df
        
    # 無効なスコア（-inf）の数をカウント
    invalid_scores = results_df[results_df['score'] == -np.inf].shape[0]
    total_scores = results_df.shape[0]
    
    if invalid_scores > 0:
        logger.warning(f"無効なスコアの割合: {invalid_scores}/{total_scores} ({invalid_scores/total_scores:.1%})")
    
    # 取引数が少ないレコードを特定（'trades'列がある場合）
    if 'trades' in results_df.columns:
        low_trade_count = results_df[results_df['trades'] < min_trades].shape[0]
        if low_trade_count > 0:
            logger.warning(f"取引数が少ない結果の割合: {low_trade_count}/{total_scores} ({low_trade_count/total_scores:.1%})")
    
    # パラメータの探索範囲の境界にある最適値を検出
    edge_params = []
    for param, values in param_grid.items():
        if param in results_df.columns:
            best_row = results_df.nlargest(1, 'score')
            if best_row[param].iloc[0] in [min(values), max(values)]:
                edge_params.append(param)
    
    if edge_params:
        logger.warning(f"境界上にある最適パラメータ: {', '.join(edge_params)}")
        logger.warning("これらのパラメータの探索範囲を拡大することを検討してください。")
    
    return results_df


def create_optimization_visualizations(results_df: pd.DataFrame, 
                                      output_dir: str,
                                      title_prefix: str = "最適化結果") -> None:
    """
    最適化結果の可視化を生成する
    
    Parameters:
        results_df (pd.DataFrame): 最適化結果のDataFrame
        output_dir (str): 出力ディレクトリのパス
        title_prefix (str): グラフタイトルのプレフィックス
    """
    if results_df.empty:
        return
        
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # スコア分布のヒストグラム
    plt.figure(figsize=(10, 6))
    valid_scores = results_df[results_df['score'] > -np.inf]['score']
    if not valid_scores.empty:
        sns.histplot(valid_scores, kde=True)
        plt.title(f"{title_prefix} - スコア分布")
        plt.xlabel("スコア")
        plt.ylabel("頻度")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"score_distribution_{timestamp}.png"))
        plt.close()
    
    # パラメータごとのスコア箱ひげ図（最大5つのパラメータまで）
    param_cols = [col for col in results_df.columns if col != 'score' and col != 'error']
    for param in param_cols[:5]:  # 最大5つのパラメータのみ可視化
        if len(results_df[param].unique()) <= 10:  # 値の種類が10以下の場合のみ可視化
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=param, y='score', data=results_df[results_df['score'] > -np.inf])
            plt.title(f"{title_prefix} - パラメータ {param} の影響")
            plt.xlabel(param)
            plt.ylabel("スコア")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"param_impact_{param}_{timestamp}.png"))
            plt.close()
    
    # 上位パラメータ組み合わせ表
    top_results = results_df.nlargest(10, 'score')
    # 結果をJSONとして保存
    top_results.to_json(os.path.join(output_dir, f"top_params_{timestamp}.json"), orient="records")
    
    
def export_strategy_performance_summary(trade_results: Dict[str, Any], 
                                       params: Dict[str, Any],
                                       output_dir: str,
                                       strategy_name: str) -> str:
    """
    戦略のパフォーマンス要約をエクスポート
    
    Parameters:
        trade_results (Dict): トレード結果
        params (Dict): 使用されたパラメータ
        output_dir (str): 出力ディレクトリ
        strategy_name (str): 戦略名
        
    Returns:
        str: 出力ファイルのパス
    """
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{strategy_name}_summary_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {strategy_name} パフォーマンス要約\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # パラメータ情報
        f.write("## 使用パラメータ\n\n")
        f.write("```python\n")
        for key, value in params.items():
            f.write(f"{key} = {value}\n")
        f.write("```\n\n")
        
        # トレード統計
        f.write("## トレード統計\n\n")
        
        trade_stats = trade_results.get("取引統計", {})
        if trade_stats:
            f.write("| 指標 | 値 |\n")
            f.write("|------|----|\n")
            for key, value in trade_stats.items():
                f.write(f"| {key} | {value} |\n")
        
        # 月次パフォーマンス
        monthly_performance = trade_results.get("月次パフォーマンス")
        if isinstance(monthly_performance, pd.DataFrame) and not monthly_performance.empty:
            f.write("\n## 月次パフォーマンス\n\n")
            f.write(monthly_performance.to_markdown())
            
        # 取引記録
        trades = trade_results.get("取引履歴")
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            f.write("\n## 取引サマリー\n\n")
            f.write(f"取引数: {len(trades)}\n\n")
            
            if len(trades) > 0:
                # 結果別の取引数
                win_count = sum(trades["損益率"] > 0)
                loss_count = sum(trades["損益率"] <= 0)
                f.write(f"勝ちトレード: {win_count} ({win_count/len(trades):.1%})\n")
                f.write(f"負けトレード: {loss_count} ({loss_count/len(trades):.1%})\n\n")
                
                # 平均利益率と平均損失率
                avg_win = trades.loc[trades["損益率"] > 0, "損益率"].mean() if win_count > 0 else 0
                avg_loss = trades.loc[trades["損益率"] <= 0, "損益率"].mean() if loss_count > 0 else 0
                f.write(f"平均利益率: {avg_win:.2%}\n")
                f.write(f"平均損失率: {avg_loss:.2%}\n")
    
    return output_file
