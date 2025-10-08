"""
OpeningGap戦略の最適化を実行するモジュール
"""
import sys
import os
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from strategies.Opening_Gap import OpeningGapStrategy
from optimization.parameter_optimizer import ParameterOptimizer
from optimization.configs.opening_gap_optimization import PARAM_GRID, OBJECTIVES_CONFIG
from optimization.objective_functions import create_custom_objective
from walk_forward.train_test_split import split_data_for_walk_forward
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from config.logger_config import setup_logger
from optimization.parallel_optimizer import ParallelParameterOptimizer
from metrics.performance_metrics_util import PerformanceMetricsCalculator

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

class OpeningGapParameterOptimizer(ParameterOptimizer):
    def __init__(self, data, dow_data, strategy_class, param_grid, objective_function, cv_splits=None, output_dir="backtest_results"):
        super().__init__(data, strategy_class, param_grid, objective_function, cv_splits, output_dir)
        self.dow_data = dow_data

    def _run_backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        try:
            # OpeningGapStrategyはdow_dataが必要
            strategy = self.strategy_class(data, self.dow_data, params=params)
            result_data = strategy.backtest()
            from trade_simulation import simulate_trades
            trade_results = simulate_trades(result_data, "最適化中")
            score = self.objective_function(trade_results)
            return score
        except Exception as e:
            logger.error(f"バックテスト実行中にエラー: {str(e)}")
            raise

class OpeningGapParallelParameterOptimizer(ParallelParameterOptimizer):
    def __init__(self, data, dow_data, strategy_class, param_grid, objective_function, cv_splits=None, output_dir="backtest_results", n_jobs=-1, verbose=10):
        super().__init__(data, strategy_class, param_grid, objective_function, cv_splits, output_dir, n_jobs, verbose)
        self.dow_data = dow_data

    def _run_backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        try:
            strategy = self.strategy_class(data, self.dow_data, params=params)
            result_data = strategy.backtest()
            from trade_simulation import simulate_trades
            trade_results = simulate_trades(result_data, "最適化中")
            score = self.objective_function(trade_results)
            return score
        except Exception as e:
            logger.error(f"バックテスト実行中にエラー: {str(e)}")
            raise

def optimize_opening_gap_strategy(data, dow_data=None, use_parallel=False):
    """
    OpeningGap戦略の最適化を実行
    
    Parameters:
        data (pd.DataFrame): 最適化に使用する株価データ
        dow_data (pd.DataFrame): ダウ平均データ（フィルタリング用）
        use_parallel (bool): 並列処理を使用するかどうか
        
    Returns:
        pd.DataFrame: 最適化結果
    """
    logger.info("OpeningGap戦略の最適化を開始します。")
    
    # データの前処理（必要に応じて）
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"日付インデックスへの変換に失敗しました: {e}")
            raise ValueError("データのインデックスを日付型に変換できません")
    
    # 必要な指標が計算されているか確認
    if 'ATR' not in data.columns:
        logger.info("ATR指標を計算します")
        from indicators.volatility_indicators import calculate_atr
        data['ATR'] = calculate_atr(data, 'Adj Close')
    
    # 出来高変化率の計算
    if 'Volume_Change' not in data.columns and 'Volume' in data.columns:
        logger.info("出来高変化率を計算します")
        data['Volume_Change'] = data['Volume'].pct_change()
    
    # ウォークフォワード分割
    train_size = 252  # 約1年
    test_size = 63    # 約3ヶ月
    splits = split_data_for_walk_forward(data, train_size, test_size)
    logger.info(f"ウォークフォワード分割: {len(splits)}分割")
    
    # 目的関数の設定
    custom_objective = create_custom_objective(OBJECTIVES_CONFIG)
    
    # 出力ディレクトリの設定
    output_dir = os.path.join("backtest_results", "optimization", "opening_gap_strategy")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 最適化の実行
    if use_parallel:
        logger.info("並列処理を使用して最適化を実行します")
        try:
            from optimization.parameter_optimizer import ParallelParameterOptimizer
            optimizer = OpeningGapParallelParameterOptimizer(
                data=data,
                dow_data=dow_data,
                strategy_class=OpeningGapStrategy,
                param_grid=PARAM_GRID,
                objective_function=custom_objective,
                cv_splits=splits,
                output_dir=output_dir,
                n_jobs=-1
            )
            results = optimizer.parallel_grid_search()
        except ImportError:
            logger.warning("ParallelParameterOptimizerが実装されていないため、通常処理を使用します")
            optimizer = OpeningGapParameterOptimizer(
                data=data,
                dow_data=dow_data,
                strategy_class=OpeningGapStrategy,
                param_grid=PARAM_GRID,
                objective_function=custom_objective,
                cv_splits=splits,
                output_dir=output_dir
            )
            results = optimizer.grid_search()
    else:
        logger.info("シングルスレッドで最適化を実行します")
        optimizer = OpeningGapParameterOptimizer(
            data=data,
            dow_data=dow_data,
            strategy_class=OpeningGapStrategy,
            param_grid=PARAM_GRID,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir=output_dir
        )
        results = optimizer.grid_search()
    
    # パフォーマンス指標の計算・保存
    if not results.empty:
        best_params = results.iloc[0].to_dict()
        strategy = OpeningGapStrategy(data, dow_data, params=best_params)
        result_data = strategy.backtest()
        from trade_simulation import simulate_trades
        trade_results = simulate_trades(result_data, "最適化後評価")
        metrics = PerformanceMetricsCalculator.calculate_all(trade_results["取引履歴"])
        metrics_path = os.path.join(output_dir, f"performance_metrics_{timestamp}.xlsx")
        pd.DataFrame([metrics]).to_excel(metrics_path, index=False)
        logger.info(f"パフォーマンス指標を保存しました: {metrics_path}")

        # ★ここで最適化パラメータを半自動システム用に保存
        from config.optimized_parameters import OptimizedParameterManager
        manager = OptimizedParameterManager()
        # strategy_nameは必ず"opening_gap"で保存
        manager.save_optimized_params(
            strategy_name="opening_gap",
            ticker="7203.T",  # 必要に応じてargs.ticker等に変更
            params=best_params,
            metrics=metrics,
            optimization_date=datetime.now().strftime("%Y-%m-%d"),
            status="pending_review"
        )
    
    # 結果の保存
    filename = f"opening_gap_strategy_results_{timestamp}"
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
    parser = argparse.ArgumentParser(description='OpeningGap戦略の最適化')
    parser.add_argument('--ticker', type=str, default=None, help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default=None, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--parallel', action='store_true', help='並列処理を使用する')
    parser.add_argument('--no-dow', action='store_true', help='ダウデータを使用しない')
    args = parser.parse_args()
    
    try:
        # データの取得
        logger.info("株価データを取得中...")
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(
            ticker=args.ticker, start_date=args.start, end_date=args.end
        )
        
        # ダウデータの取得
        dow_data = None
        if not args.no_dow:
            logger.info("ダウ平均データを取得中...")
            try:
                # ダウ平均データの取得ロジック（実装に応じて調整）
                import yfinance as yf
                dow_data = yf.download("^DJI", start=start_date, end=end_date)
                logger.info(f"ダウ平均データ: {len(dow_data)}行 取得完了")
            except Exception as e:
                logger.warning(f"ダウ平均データの取得に失敗しました: {e}")
        
        # データの前処理
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"最適化対象: {ticker}, 期間: {start_date} から {end_date}")
        
        # OpeningGap戦略の最適化を実行
        results = optimize_opening_gap_strategy(stock_data, dow_data, use_parallel=args.parallel)
        
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