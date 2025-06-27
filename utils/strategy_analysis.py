"""
戦略分析とレポート作成のためのユーティリティ

戦略のパフォーマンス分析、リスク評価、レポートなどに関わる
機能を提供するモジュール
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging


def generate_performance_plots(trade_results: Dict, output_dir: str, strategy_name: str) -> List[str]:
    """
    バックテスト結果からパフォーマンスグラフを生成する
    
    Parameters:
        trade_results (Dict): バックテスト結果のディクショナリ
        output_dir (str): 出力ディレクトリのパス
        strategy_name (str): 戦略名
        
    Returns:
        List[str]: 生成されたファイルのパスリスト
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files = []
    
    try:
        # 損益推移グラフ
        pnl_summary = trade_results.get("損益推移")
        if isinstance(pnl_summary, pd.DataFrame) and not pnl_summary.empty and "累積損益" in pnl_summary.columns:
            equity_file = os.path.join(output_dir, f"{strategy_name}_equity_curve_{timestamp}.png")
            
            plt.figure(figsize=(12, 6))
            plt.plot(pnl_summary.index, pnl_summary["累積損益"], label="累積損益")
            
            # ドローダウン計算と表示
            if len(pnl_summary) > 0:
                rolling_max = pnl_summary["累積損益"].cummax()
                drawdown = (pnl_summary["累積損益"] - rolling_max) / rolling_max * 100
                plt.fill_between(pnl_summary.index, 0, drawdown, alpha=0.3, color='red')
            
            plt.title(f"{strategy_name} - 資本曲線", fontsize=14)
            plt.xlabel("日付", fontsize=12)
            plt.ylabel("累積損益 (%)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(equity_file, dpi=100)
            plt.close()
            
            generated_files.append(equity_file)
            logger.info(f"資本曲線グラフを生成しました: {equity_file}")
            
        # 取引履歴分析
        trades = trade_results.get("取引履歴")
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            # 取引結果の分布図
            pnl_dist_file = os.path.join(output_dir, f"{strategy_name}_pnl_distribution_{timestamp}.png")
            
            plt.figure(figsize=(10, 6))
            sns.histplot(trades["損益率"] * 100, kde=True, bins=20)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f"{strategy_name} - 取引損益分布", fontsize=14)
            plt.xlabel("損益率 (%)", fontsize=12)
            plt.ylabel("頻度", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(pnl_dist_file, dpi=100)
            plt.close()
            
            generated_files.append(pnl_dist_file)
            logger.info(f"損益分布グラフを生成しました: {pnl_dist_file}")
            
            # 保有期間分析
            if "保有期間" in trades.columns:
                holding_file = os.path.join(output_dir, f"{strategy_name}_holding_periods_{timestamp}.png")
                
                plt.figure(figsize=(10, 6))
                sns.histplot(trades["保有期間"], bins=20)
                plt.title(f"{strategy_name} - 保有期間分布", fontsize=14)
                plt.xlabel("保有期間 (日数)", fontsize=12)
                plt.ylabel("頻度", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(holding_file, dpi=100)
                plt.close()
                
                generated_files.append(holding_file)
                logger.info(f"保有期間分布グラフを生成しました: {holding_file}")
                
            # 月次パフォーマンス
            monthly_performance = trade_results.get("月次パフォーマンス")
            if isinstance(monthly_performance, pd.DataFrame) and not monthly_performance.empty:
                monthly_file = os.path.join(output_dir, f"{strategy_name}_monthly_returns_{timestamp}.png")
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x=monthly_performance.index, y=monthly_performance["月間リターン"])
                plt.title(f"{strategy_name} - 月次リターン", fontsize=14)
                plt.xticks(rotation=45)
                plt.xlabel("月", fontsize=12)
                plt.ylabel("リターン (%)", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(monthly_file, dpi=100)
                plt.close()
                
                generated_files.append(monthly_file)
                logger.info(f"月次リターングラフを生成しました: {monthly_file}")
                
        return generated_files
        
    except Exception as e:
        logger.exception(f"パフォーマンスグラフ生成中にエラーが発生しました: {str(e)}")
        return generated_files


def analyze_sensitivity(param_name: str, 
                        param_values: List[Any], 
                        all_results: pd.DataFrame,
                        output_dir: str,
                        strategy_name: str) -> Optional[str]:
    """
    特定のパラメータに対する感度分析を実行
    
    Parameters:
        param_name (str): 分析するパラメータ名
        param_values (List[Any]): パラメータの値リスト
        all_results (pd.DataFrame): 全最適化結果
        output_dir (str): 出力ディレクトリ
        strategy_name (str): 戦略名
        
    Returns:
        Optional[str]: 生成されたファイルのパス（失敗時はNone）
    """
    logger = logging.getLogger(__name__)
    
    try:
        if param_name not in all_results.columns:
            logger.warning(f"パラメータ '{param_name}' は結果に含まれていません")
            return None
            
        # このパラメータのみが変化した結果をフィルタ
        filtered_results = all_results.copy()
        
        # パラメータに対するスコアの変化をプロット
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{strategy_name}_{param_name}_sensitivity_{timestamp}.png")
        
        plt.figure(figsize=(10, 6))
        
        # 箱ひげ図を描画
        sns.boxplot(x=param_name, y='score', data=filtered_results)
        
        # 平均値の折れ線グラフ追加
        means = filtered_results.groupby(param_name)['score'].mean()
        plt.plot(range(len(means)), means.values, 'ro-', linewidth=2)
        
        plt.title(f"{strategy_name} - パラメータ {param_name} の感度分析", fontsize=14)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel("スコア", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        plt.close()
        
        logger.info(f"感度分析グラフを生成しました: {output_file}")
        return output_file
        
    except Exception as e:
        logger.exception(f"感度分析中にエラーが発生しました: {str(e)}")
        return None


def create_parameter_impact_summary(results_df: pd.DataFrame, 
                                   param_grid: Dict[str, List[Any]],
                                   output_file: str) -> None:
    """
    パラメータの影響度合いを要約したレポートを作成
    
    Parameters:
        results_df (pd.DataFrame): 最適化結果
        param_grid (Dict[str, List[Any]]): パラメータグリッド
        output_file (str): 出力ファイルパス
    """
    logger = logging.getLogger(__name__)
    
    try:
        if results_df.empty:
            logger.warning("結果が空のため、パラメータ影響度の要約を作成できません")
            return
            
        # 各パラメータの影響度を計算
        param_impact = {}
        
        for param in param_grid.keys():
            if param in results_df.columns:
                # パラメータごとのスコア平均を計算
                impact_data = results_df.groupby(param)['score'].agg(['mean', 'std', 'count']).reset_index()
                
                # 最大と最小の差を計算
                max_mean = impact_data['mean'].max()
                min_mean = impact_data['mean'].min()
                impact_score = max_mean - min_mean
                
                param_impact[param] = {
                    'impact_score': impact_score,
                    'best_value': impact_data.loc[impact_data['mean'].idxmax(), param],
                    'worst_value': impact_data.loc[impact_data['mean'].idxmin(), param],
                    'std_dev': impact_data['std'].mean()
                }
        
        # 影響度でソート
        sorted_impact = sorted(param_impact.items(), key=lambda x: x[1]['impact_score'], reverse=True)
        
        # レポート作成
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# パラメータ影響度分析\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## パラメータの影響度ランキング\n\n")
            f.write("| パラメータ | 影響度スコア | 最適値 | 最悪値 | 標準偏差 |\n")
            f.write("|------------|--------------|--------|--------|----------|\n")
            
            for param, data in sorted_impact:
                f.write(f"| {param} | {data['impact_score']:.4f} | {data['best_value']} | {data['worst_value']} | {data['std_dev']:.4f} |\n")
            
            f.write("\n## 分析サマリー\n\n")
            
            # トップ3の影響度の高いパラメータに関する詳細情報
            f.write("### 最も影響度の高いパラメータ\n\n")
            for i, (param, data) in enumerate(sorted_impact[:3], 1):
                if i <= len(sorted_impact):
                    f.write(f"**{i}. {param}**\n\n")
                    f.write(f"- 影響度スコア: {data['impact_score']:.4f}\n")
                    f.write(f"- 最適値: {data['best_value']}\n")
                    f.write(f"- 最悪値: {data['worst_value']}\n")
                    f.write(f"- このパラメータは全体のパフォーマンスに大きな影響を与えています。\n\n")
            
            # 最適パラメータセットの要約
            best_params = results_df.loc[results_df['score'].idxmax()]
            f.write("## 最適パラメータセット\n\n")
            f.write("```python\n")
            for param in param_grid.keys():
                if param in best_params.index:
                    f.write(f"{param} = {best_params[param]}\n")
            f.write("```\n\n")
            
            f.write("## 最適化の推奨事項\n\n")
            
            # 影響度の高いパラメータに関する推奨事項
            for param, data in sorted_impact[:3]:
                # パラメータが境界にある場合は探索範囲の拡大を推奨
                if data['best_value'] in [min(param_grid[param]), max(param_grid[param])]:
                    f.write(f"- **{param}**: 探索範囲の拡大を推奨します。最適値が現在の範囲の境界にあります。\n")
                else:
                    f.write(f"- **{param}**: 最適値 {data['best_value']} の周辺でより細かい探索を検討してください。\n")
        
        logger.info(f"パラメータ影響度分析レポートを生成しました: {output_file}")
        
    except Exception as e:
        logger.exception(f"パラメータ影響度分析中にエラーが発生しました: {str(e)}")
