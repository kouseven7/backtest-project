"""
改善された最適化ワークフローの実行スクリプト
- より高度なロギング
- パラメータ影響度分析
- 可視化とレポート生成
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロガーの設定
log_filepath = r"C:\Users\imega\Documents\my_backtest_project\logs\advanced_optimization.log"
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("advanced_optimization")

# ユーティリティのインポート
from utils.optimization_utils import (
    safe_score_calculation,
    validate_optimization_results,
    create_optimization_visualizations,
    export_strategy_performance_summary
)
from utils.strategy_analysis import (
    generate_performance_plots,
    analyze_sensitivity,
    create_parameter_impact_summary
)

def run_advanced_optimization(strategy_name: str = "VWAP_Breakout", data_years: int = 3):
    """
    改善された最適化ワークフローを実行
    
    Parameters:
        strategy_name (str): 最適化する戦略の名前
        data_years (int): 使用するデータの年数
    """
    try:
        logger.info("=" * 80)
        logger.info(f"高度な最適化ワークフローを開始: {strategy_name}")
        logger.info("=" * 80)
        
        # ステップ1: データ取得と準備
        logger.info("■ ステップ1: データ取得と準備")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # より長期のデータを使用（指定された年数に基づく）
        logger.info(f"取得データ期間: {stock_data.index[0]} 〜 {stock_data.index[-1]} ({len(stock_data)}日)")
        logger.info(f"{data_years}年分のデータを準備中...")
        
        # 必要に応じてデータ期間を調整
        if data_years > 0 and len(stock_data) > data_years * 252:  # 約252取引日/年
            logger.info(f"データを最新の{data_years}年分に制限します")
            data_days = data_years * 252
            stock_data = stock_data.iloc[-data_days:]
            index_data = index_data.iloc[-data_days:]
        
        logger.info(f"使用データ期間: {stock_data.index[0]} 〜 {stock_data.index[-1]} ({len(stock_data)}日)")
        
        # ステップ2: 最適化設定
        logger.info("■ ステップ2: 最適化設定")
        
        # 戦略に応じたインポート
        if strategy_name == "VWAP_Breakout":
            from optimization.configs import vwap_breakout_optimization
            from strategies.VWAP_Breakout import VWAPBreakoutStrategy
            
            # パラメータグリッド取得
            param_grid = vwap_breakout_optimization.get_param_grid()
            strategy_class = VWAPBreakoutStrategy
            
            # 目的関数の改善版を生成
            from optimization.objective_functions import create_custom_objective, sharpe_ratio_objective
            from functools import partial
            
            # デコレータで強化したバージョン
            @safe_score_calculation
            def enhanced_objective(trade_results):
                # 複合目的関数を使用（シャープレシオ + 勝率 + 期待値）
                return create_custom_objective(
                    [sharpe_ratio_objective, "win_rate", "expectancy"],
                    [0.5, 0.3, 0.2]
                )(trade_results)
            
            objective_function = enhanced_objective
            
        else:
            logger.error(f"未知の戦略名: {strategy_name}")
            return
        
        # パラメータグリッドのサマリーをログに出力
        logger.info("パラメータグリッド:")
        for param, values in param_grid.items():
            logger.info(f"- {param}: {values} ({len(values)}通り)")
        
        # ステップ3: 最適化の実行
        logger.info("■ ステップ3: 最適化実行")
        from optimization.parameter_optimizer import ParameterOptimizer
        
        # 最適化ディレクトリの設定
        output_dir = os.path.join(r"C:\Users\imega\Documents\my_backtest_project\backtest_results", 
                                 f"optimization_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 最適化の実行
        optimizer = ParameterOptimizer(
            data=stock_data,
            strategy_class=strategy_class,
            param_grid=param_grid,
            objective_function=objective_function,
            output_dir=output_dir,
            strategy_kwargs={"index_data": index_data}
        )
        
        # CPU負荷とメモリ使用量を抑えるためにn_jobs=2で実行
        results_df = optimizer.parallel_grid_search(n_jobs=2)
        
        # ステップ4: 結果の検証と分析
        logger.info("■ ステップ4: 結果の検証と分析")
        
        # 結果の検証
        validated_results = validate_optimization_results(
            results_df=results_df,
            param_grid=param_grid,
            min_trades=10
        )
        
        # 最適化結果の可視化
        logger.info("最適化結果の可視化を生成中...")
        create_optimization_visualizations(
            results_df=validated_results,
            output_dir=output_dir,
            title_prefix=f"{strategy_name} 最適化"
        )
        
        # パラメータごとの感度分析
        logger.info("パラメータ感度分析を実行中...")
        for param_name in param_grid.keys():
            analyze_sensitivity(
                param_name=param_name,
                param_values=param_grid[param_name],
                all_results=validated_results,
                output_dir=output_dir,
                strategy_name=strategy_name
            )
        
        # パラメータ影響度サマリー
        impact_file = os.path.join(output_dir, f"{strategy_name}_parameter_impact.md")
        create_parameter_impact_summary(
            results_df=validated_results,
            param_grid=param_grid,
            output_file=impact_file
        )
        
        # ステップ5: 最適パラメータによるバックテスト
        logger.info("■ ステップ5: 最適パラメータによるバックテスト")
        
        # 最適パラメータの取得
        if not validated_results.empty:
            best_params = validated_results.loc[validated_results['score'].idxmax()].drop(['score'])
            best_params = best_params.to_dict()
            
            logger.info(f"最適パラメータ: {best_params}")
            
            # 最適パラメータでの戦略インスタンス作成
            strategy = strategy_class(stock_data, **best_params, index_data=index_data)
            
            # バックテスト実行
            logger.info("最適パラメータでバックテストを実行...")
            backtest_results = strategy.backtest()
            
            # 結果のエクスポート
            export_file = export_strategy_performance_summary(
                trade_results=backtest_results,
                params=best_params,
                output_dir=output_dir,
                strategy_name=strategy_name
            )
            
            logger.info(f"パフォーマンスサマリーを保存: {export_file}")
            
            # パフォーマンスプロットの生成
            plot_files = generate_performance_plots(
                trade_results=backtest_results,
                output_dir=output_dir,
                strategy_name=strategy_name
            )
            
            logger.info(f"生成されたパフォーマンスグラフ: {len(plot_files)}個")
            
        else:
            logger.warning("有効な最適化結果が見つかりません。バックテストをスキップします。")
        
        # 結果の保存
        result_path = os.path.join(output_dir, f"{strategy_name}_optimization_results.csv")
        validated_results.to_csv(result_path, index=False, encoding='utf-8')
        logger.info(f"最適化結果を保存: {result_path}")
        
        logger.info("=" * 80)
        logger.info("高度な最適化ワークフロー完了")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception(f"最適化プロセス中にエラーが発生: {str(e)}")


if __name__ == "__main__":
    # コマンドライン引数の処理
    import argparse
    
    parser = argparse.ArgumentParser(description="高度な最適化ワークフローを実行")
    parser.add_argument("--strategy", type=str, default="VWAP_Breakout",
                       help="最適化する戦略の名前")
    parser.add_argument("--years", type=int, default=3,
                       help="使用するデータの年数")
    
    args = parser.parse_args()
    
    run_advanced_optimization(
        strategy_name=args.strategy,
        data_years=args.years
    )
