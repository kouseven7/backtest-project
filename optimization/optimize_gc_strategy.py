"""
GC戦略の最適化を実行するモジュール
"""
import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from strategies.gc_strategy_signal import GCStrategy
from optimization.parameter_optimizer import ParameterOptimizer, ParallelParameterOptimizer
from optimization.configs.gc_strategy_optimization import PARAM_GRID, OBJECTIVES_CONFIG
from optimization.objective_functions import create_custom_objective
from walk_forward.train_test_split import split_data_for_walk_forward
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

def optimize_gc_strategy(data, use_parallel=False):
    """
    GC戦略の最適化を実行する
    
    Parameters:
        data (pd.DataFrame): 最適化に使用する株価データ
        use_parallel (bool): 並列処理を使用するかどうか
        
    Returns:
        pd.DataFrame: 最適化結果
    """
    logger.info("GC戦略の最適化を開始します。")
    
    # データの前処理（必要に応じて）
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"日付インデックスへの変換に失敗しました: {e}")
            raise ValueError("データのインデックスを日付型に変換できません")
    
    # 必要な指標が計算されていることを確認
    if not any(col.startswith('SMA_') for col in data.columns):
        logger.info("移動平均を計算します")
        for window in [5, 10, 15, 20, 25, 50, 100, 200]:
            data[f'SMA_{window}'] = data['Adj Close'].rolling(window=window).mean()
            # EMAも計算
            data[f'EMA_{window}'] = data['Adj Close'].ewm(span=window, adjust=False).mean()
    
    # ウォークフォワード分割
    train_size = 252  # 約1年
    test_size = 63    # 約3ヶ月
    splits = split_data_for_walk_forward(data, train_size, test_size)
    logger.info(f"ウォークフォワード分割: {len(splits)}分割")
    
    # 目的関数の設定
    custom_objective = create_custom_objective(OBJECTIVES_CONFIG)
    
    # 出力ディレクトリの設定
    output_dir = os.path.join("backtest_results", "optimization", "gc_strategy")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 最適化の実行
    if use_parallel:
        logger.info("並列処理を使用して最適化を実行します")
        optimizer = ParallelParameterOptimizer(
            data=data,
            strategy_class=GCStrategy,
            param_grid=PARAM_GRID,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir=output_dir,
            n_jobs=-1  # 使用可能なすべてのコアを使用
        )
        results = optimizer.parallel_grid_search()
    else:
        logger.info("シングルスレッドで最適化を実行します")
        optimizer = ParameterOptimizer(
            data=data,
            strategy_class=GCStrategy,
            param_grid=PARAM_GRID,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir=output_dir
        )
        results = optimizer.grid_search()
    
    # 結果の保存
    filename = f"gc_strategy_results_{timestamp}"
    optimizer.save_results(filename=filename, format="excel")
    logger.info(f"最適化結果を保存しました: {filename}")
    
    if not results.empty:
        best_params = results.iloc[0].to_dict()
        best_score = best_params.pop("score", None)
        logger.info(f"最適化完了: 最良スコア = {best_score}")
        logger.info(f"最適パラメータ: {best_params}")
    else:
        logger.warning("最適化に失敗しました。結果が空です。")
    
    return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GC戦略の最適化')
    parser.add_argument('--ticker', type=str, default=None, help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default=None, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--parallel', action='store_true', help='並列処理を使用する')
    args = parser.parse_args()
    
    try:
        # データの取得
        logger.info("株価データを取得中...")
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(
            ticker=args.ticker, start_date=args.start, end_date=args.end
        )
        
        # データの前処理
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"最適化対象: {ticker}, 期間: {start_date} から {end_date}")
        
        # GC戦略の最適化を実行
        results = optimize_gc_strategy(stock_data, use_parallel=args.parallel)
        
        # 最適化結果をコンソールに表示
        if not results.empty:
            print("\n*** 最適化結果 (上位5件) ***")
            pd.set_option('display.max_columns', None)
            print(results.head(5))
            
            # 最適パラメータを表示
            best_params = results.iloc[0].to_dict()
            score = best_params.pop("score", None)
            print(f"\n最良スコア: {score}")
            print("最適パラメータ:")
            for param, value in best_params.items():
                if param not in ["score", "train_score", "test_score"]:
                    print(f"  {param}: {value}")
        
    except Exception as e:
        logger.exception(f"最適化実行中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()