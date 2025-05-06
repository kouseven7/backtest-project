"""
戦略のパラメータ最適化を実行するスクリプト
"""
import pandas as pd
import argparse
import time
import logging
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.gc_strategy_signal import GCStrategy
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from walk_forward.train_test_split import split_data_for_walk_forward
from optimization.parameter_optimizer import ParameterOptimizer
from optimization.parallel_optimizer import ParallelParameterOptimizer
from optimization.objective_functions import (
    sharpe_ratio_objective,
    risk_adjusted_return_objective,
    create_custom_objective
)
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

def optimize_breakout_strategy(data, use_parallel=False):
    """
    ブレイクアウト戦略の最適化を実行
    
    Parameters:
        data (pd.DataFrame): 株価データ
        use_parallel (bool): 並列処理を使用するかどうか
        
    Returns:
        pd.DataFrame: 最適化結果
    """
    # 最適化するパラメータの定義
    param_grid = {
        "volume_threshold": [1.1, 1.2, 1.3, 1.4, 1.5],
        "take_profit": [0.02, 0.03, 0.04, 0.05],
        "look_back": [1, 2, 3]
    }
    
    # 交差検証用のデータ分割
    train_size = 252  # 1年
    test_size = 63    # 3ヶ月
    splits = split_data_for_walk_forward(data, train_size, test_size)
    
    # 目的関数の設定
    objectives_config = [
        {"name": "sharpe_ratio", "weight": 1.0},
        {"name": "risk_adjusted_return", "weight": 0.5}
    ]
    custom_objective = create_custom_objective(objectives_config)
    
    # 最適化クラスの選択
    if use_parallel:
        optimizer = ParallelParameterOptimizer(
            data=data,
            strategy_class=BreakoutStrategy,
            param_grid=param_grid,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir="backtest_results/optimization",
            n_jobs=-1  # 全CPUコアを使用
        )
        results = optimizer.parallel_grid_search()
    else:
        optimizer = ParameterOptimizer(
            data=data,
            strategy_class=BreakoutStrategy,
            param_grid=param_grid,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir="backtest_results/optimization"
        )
        results = optimizer.grid_search()
    
    # 結果の保存
    filename = f"breakout_optimization_{time.strftime('%Y%m%d')}"
    optimizer.save_results(filename=filename, format="excel")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='戦略パラメータの最適化')
    parser.add_argument('--strategy', type=str, default='breakout',
                        choices=['breakout', 'vwap_breakout', 'momentum', 'gc'],
                        help='最適化する戦略')
    parser.add_argument('--parallel', action='store_true',
                        help='並列処理を使用する')
    args = parser.parse_args()
    
    # データ取得
    logger.info("株価データを取得中...")
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    
    # データ前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    # 選択した戦略の最適化を実行
    strategy_map = {
        'breakout': optimize_breakout_strategy,
        # 他の戦略の最適化関数も追加可能
    }
    
    if args.strategy in strategy_map:
        logger.info(f"{args.strategy}戦略の最適化を開始...")
        optimize_func = strategy_map[args.strategy]
        results = optimize_func(stock_data, use_parallel=args.parallel)
        
        # 最適なパラメータを表示
        if not results.empty:
            best_params = results.iloc[0].to_dict()
            best_score = best_params.pop("score", None)
            logger.info(f"最適化完了: 最良スコア = {best_score}")
            logger.info(f"最適パラメータ: {best_params}")
        else:
            logger.warning("最適化に失敗しました。結果が空です。")
    else:
        logger.error(f"未実装の戦略: {args.strategy}")
        logger.info(f"実装済み戦略: {list(strategy_map.keys())}")

if __name__ == "__main__":
    main()