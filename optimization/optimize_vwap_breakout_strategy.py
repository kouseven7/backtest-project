"""
VWAPブレイクアウト戦略の最適化を実行するモジュール
"""
import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from optimization.parameter_optimizer import ParameterOptimizer
from optimization.configs.vwap_breakout_optimization import PARAM_GRID, OBJECTIVES_CONFIG
from optimization.objective_functions import create_custom_objective
from walk_forward.train_test_split import split_data_for_walk_forward
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from config.logger_config import setup_logger
from metrics.performance_metrics_util import PerformanceMetricsCalculator
from trade_simulation import simulate_trades

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

def optimize_vwap_breakout_strategy(data, index_data=None, use_parallel=False):
    """
    VWAPブレイクアウト戦略の最適化を実行
    
    Parameters:
        data (pd.DataFrame): 最適化に使用する株価データ
        index_data (pd.DataFrame): 市場インデックスデータ
        use_parallel (bool): 並列処理を使用するかどうか
        
    Returns:
        pd.DataFrame: 最適化結果
    """
    logger.info("VWAPブレイクアウト戦略の最適化を開始します。")
    
    # データの前処理（必要に応じて）
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            logger.error(f"日付インデックスへの変換に失敗しました: {e}")
            raise ValueError("データのインデックスを日付型に変換できません")
    
    # インデックスデータも同様に確認
    if index_data is not None and not isinstance(index_data.index, pd.DatetimeIndex):
        # インデックスが "Ticker" など日付でない場合の対処
        if index_data.index[0] == 'Ticker' or 'Date' in index_data.columns:
            logger.info("インデックスデータにヘッダー行が含まれています。インデックスをリセットして再設定します。")
            index_data = index_data.reset_index(drop=True)
            if 'Date' in index_data.columns:
                index_data = index_data.set_index('Date')
        try:
            index_data.index = pd.to_datetime(index_data.index)
        except Exception as e:
            logger.error(f"インデックスデータの日付変換に失敗しました: {e}")
            raise ValueError("インデックスデータのインデックスを日付型に変換できません")
    
    # 必要な指標が計算されているか確認
    if 'VWAP' not in data.columns:
        logger.info("VWAP指標を計算します")
        from indicators.basic_indicators import calculate_vwap
        data['VWAP'] = calculate_vwap(data, price_column='Adj Close', volume_column='Volume')
    
    # 必要な指標を計算
    required_indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
    for indicator in required_indicators:
        if indicator not in data.columns:
            logger.info(f"{indicator}指標を計算します")
            # 既存のcompute_indicators関数を使用するか、個別に計算
            if indicator.startswith('SMA_'):
                period = int(indicator.split('_')[1])
                from indicators.basic_indicators import calculate_sma
                data[indicator] = calculate_sma(data, 'Adj Close', period)
            elif indicator == 'RSI':
                from indicators.basic_indicators import calculate_rsi
                data['RSI'] = calculate_rsi(data['Adj Close'], 14)
            elif indicator in ['MACD', 'Signal_Line'] and 'MACD' not in data.columns:
                from indicators.momentum_indicators import calculate_macd
                data['MACD'], data['Signal_Line'] = calculate_macd(data, 'Adj Close')
    
    # インデックスデータの処理
    if index_data is not None:
        for sma in ['SMA_20', 'SMA_50']:
            if sma not in index_data.columns:
                period = int(sma.split('_')[1])
                from indicators.basic_indicators import calculate_sma
                index_data[sma] = calculate_sma(index_data, 'Adj Close', period)
    else:
        logger.warning("インデックスデータが提供されていないため、市場トレンドフィルターは無効化されます")
        # 市場データがない場合はダミーデータを作成（最適化用）
        dates = data.index
        index_data = pd.DataFrame({
            'Adj Close': np.ones(len(dates)) * 100,  # ダミー値
            'SMA_20': np.ones(len(dates)) * 95,
            'SMA_50': np.ones(len(dates)) * 90
        }, index=dates)
    
    # ウォークフォワード分割
    train_size = 252  # 約1年
    test_size = 63    # 約3ヶ月
    splits = split_data_for_walk_forward(data, train_size, test_size)
    logger.info(f"ウォークフォワード分割: {len(splits)}分割")
    
    # 目的関数の設定
    # "profit_factor"が未定義なので削除または修正
    filtered_objectives = [obj for obj in OBJECTIVES_CONFIG if obj["name"] != "profit_factor"]
    if len(filtered_objectives) != len(OBJECTIVES_CONFIG):
        logger.warning("profit_factorが目的関数に定義されていませんが存在しないため削除しました")
    
    custom_objective = create_custom_objective(filtered_objectives)
    
    # 出力ディレクトリの設定
    output_dir = os.path.join("backtest_results", "optimization", "vwap_breakout_strategy")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 最適化の実行
    if use_parallel:
        logger.info("並列処理を使用して最適化を実行します")
        try:
            from optimization.parallel_optimizer import ParallelParameterOptimizer
            optimizer = ParallelParameterOptimizer(
                data=data,
                strategy_class=VWAPBreakoutStrategy,
                param_grid=PARAM_GRID,
                objective_function=custom_objective,
                cv_splits=splits,
                output_dir=output_dir,
                n_jobs=-1,  # 使用可能なすべてのコアを使用
                index_data=index_data  # ここでindex_dataを渡す
            )
            results = optimizer.parallel_grid_search()
        except ImportError:
            logger.warning("ParallelParameterOptimizerが実装されていないため、通常処理を使用します")
            optimizer = ParameterOptimizer(
                data=data,
                strategy_class=VWAPBreakoutStrategy,
                param_grid=PARAM_GRID,
                objective_function=custom_objective,
                cv_splits=splits,
                output_dir=output_dir,
                strategy_kwargs={'index_data': index_data}
            )
            results = optimizer.grid_search()
    else:
        logger.info("シングルスレッドで最適化を実行します")
        optimizer = ParameterOptimizer(
            data=data,
            strategy_class=VWAPBreakoutStrategy,
            param_grid=PARAM_GRID,
            objective_function=custom_objective,
            cv_splits=splits,
            output_dir=output_dir,
            strategy_kwargs={'index_data': index_data}
        )
        results = optimizer.grid_search()
    
    # パフォーマンス指標の計算・保存
    if not results.empty:
        best_params = results.iloc[0].to_dict()
        # ここで再インポートしていた問題を修正 (VWAPBreakoutStrategyはすでにインポート済み)
        strategy = VWAPBreakoutStrategy(data, index_data=index_data, params=best_params)
        result_data = strategy.backtest()
        # トレードシミュレーションを実行
        trade_results = simulate_trades(result_data, "最適化後評価")
        metrics = PerformanceMetricsCalculator.calculate_all(trade_results["取引履歴"])
        metrics_path = os.path.join(output_dir, f"performance_metrics_{timestamp}.xlsx")
        pd.DataFrame([metrics]).to_excel(metrics_path, index=False)
        logger.info(f"パフォーマンス指標を保存しました: {metrics_path}")
    
    # 結果の保存
    filename = f"vwap_breakout_strategy_results_{timestamp}"
    optimizer.save_results(filename=filename, format="excel")
    logger.info(f"最適化結果を保存しました: {filename}")
    
    if not results.empty:
        best_params = results.iloc[0].to_dict()
        best_score = best_params.pop("score", None)
        logger.info(f"最適化完了: 最良スコア = {best_score}")
        logger.info(f"最適パラメータ: {best_params}")
        
        # ウォークフォワード分析の詳細結果
        if hasattr(optimizer, 'walk_forward_results') and optimizer.walk_forward_results:
            wf_output_path = os.path.join(output_dir, f"walk_forward_details_{timestamp}.xlsx")
            with pd.ExcelWriter(wf_output_path) as writer:
                for i, wf_result in enumerate(optimizer.walk_forward_results):
                    if isinstance(wf_result, pd.DataFrame):
                        wf_result.to_excel(writer, sheet_name=f"Window_{i+1}")
            logger.info(f"ウォークフォワード分析詳細結果を保存しました: {wf_output_path}")
    else:
        logger.warning("最適化に失敗しました。結果が空です。")
    
    return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='VWAPブレイクアウト戦略の最適化')
    parser.add_argument('--ticker', type=str, default=None, help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default=None, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--parallel', action='store_true', help='並列処理を使用する')
    parser.add_argument('--no-index', action='store_true', help='市場インデックスデータを使用しない')
    parser.add_argument('--walk-forward', action='store_true', help='ウォークフォワード分析の詳細結果を表示')
    args = parser.parse_args()
    
    try:
        # データの取得
        logger.info("株価データを取得中...")
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()  # 引数なしで呼び出し
        # コマンドライン引数で上書き
        if args.ticker is not None:
            ticker = args.ticker
        if args.start is not None:
            start_date = args.start
        if args.end is not None:
            end_date = args.end
        # 必要ならstock_data/index_dataを再取得するロジックを追加してもよい
        if args.no_index:
            logger.info("市場インデックスデータを使用しないオプションが指定されました")
            index_data = None
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"最適化対象: {ticker}, 期間: {start_date} から {end_date}")
        
        # VWAPブレイクアウト戦略の最適化を実行
        results = optimize_vwap_breakout_strategy(stock_data, index_data, use_parallel=args.parallel)
        
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
            
            # ウォークフォワード分析の結果を表示
            if args.walk_forward and hasattr(results, 'walk_forward_details'):
                print("\n*** ウォークフォワード分析結果 ***")
                print(results.walk_forward_details)
        
    except Exception as e:
        logger.exception(f"最適化実行中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()