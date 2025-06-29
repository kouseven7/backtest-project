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
from optimization.configs.vwap_breakout_optimization_reduced import PARAM_GRID, OBJECTIVES_CONFIG
from optimization.objective_functions import create_custom_objective
from walk_forward.train_test_split import split_data_for_walk_forward
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from config.logger_config import setup_logger
from metrics.performance_metrics_util import PerformanceMetricsCalculator

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
    
    # ウォークフォワード分割（より長いトレーニング期間と複数の検証期間）
    train_size = 504  # 約2年（より堅牢なトレーニング）
    test_size = 63    # 約3ヶ月
    
    # データが十分あるか確認
    if len(data) < train_size + test_size:
        logger.warning(f"データ不足: 利用可能データ {len(data)}日、必要データ {train_size + test_size}日")
        # データが少なくても最低限のトレーニングができるよう調整
        if len(data) > 252 + 21:  # 少なくとも1年+1ヶ月
            train_size = len(data) - 21
            test_size = 21
            logger.info(f"データ期間を調整: トレーニング {train_size}日、テスト {test_size}日")
        else:
            # 極端に少ない場合は80:20で分割
            train_size = int(len(data) * 0.8)
            test_size = len(data) - train_size
            logger.info(f"最小構成で分割: トレーニング {train_size}日、テスト {test_size}日")
    
    # 複数の期間でテスト（より堅牢な評価）
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
        results = optimizer.grid_search()        # パフォーマンス指標の計算・保存
    if not results.empty:
        best_params = results.iloc[0].to_dict()
        # PARAM_GRIDのキーのみを抽出してパラメータを渡す
        param_keys = set(PARAM_GRID.keys())
        def filter_params(params):
            return {k: v for k, v in params.items() if k in param_keys}

        filtered_params = filter_params(best_params)
        strategy = VWAPBreakoutStrategy(data, index_data=index_data, params=filtered_params)
        result_data = strategy.backtest()
        from trade_simulation import simulate_trades
        trade_results = simulate_trades(result_data, "最適化後評価")
        
        # 取引数の確認と警告（取引数が少なすぎると統計的に意味がない）
        trade_count = len(trade_results["取引履歴"])
        if trade_count < 20:
            logger.warning(f"取引数が少なすぎます ({trade_count}件)。パラメータを調整して取引機会を増やしてください。")
        else:
            logger.info(f"取引総数: {trade_count}件")
            
        # 勝率の確認
        if '取引結果' in trade_results["取引履歴"].columns:
            win_trades = len(trade_results["取引履歴"][trade_results["取引履歴"]['取引結果'] > 0])
            win_rate = win_trades / trade_count if trade_count > 0 else 0
            logger.info(f"勝率: {win_rate:.2%} ({win_trades}/{trade_count})")
            
            # 平均リターンの確認
            avg_return = trade_results["取引履歴"]['取引結果'].mean()
            logger.info(f"平均リターン: {avg_return:.4f}")
            
            # 利益・損失の比率
            profit_sum = trade_results["取引履歴"][trade_results["取引履歴"]['取引結果'] > 0]['取引結果'].sum()
            loss_sum = abs(trade_results["取引履歴"][trade_results["取引履歴"]['取引結果'] < 0]['取引結果'].sum())
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            logger.info(f"利益・損失比率: {profit_factor:.2f}")
            
            # リスク調整後リターンの確認
            if '累積損益' in trade_results["損益推移"].columns:
                from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown
                returns = trade_results["損益推移"]['累積損益'].pct_change().fillna(0)
                sharpe = calculate_sharpe_ratio(returns)
                sortino = calculate_sortino_ratio(returns)
                max_dd = calculate_max_drawdown(trade_results["損益推移"]['累積損益'])
                logger.info(f"シャープレシオ: {sharpe:.4f}")
                logger.info(f"ソルティノレシオ: {sortino:.4f}")
                logger.info(f"最大ドローダウン: {max_dd:.4%}")
        
        # 従来の結果表示も残す（デバッグ用）
        print("\n=== 取引履歴['取引結果']の先頭10件 ===")
        print(trade_results["取引履歴"]["取引結果"].head(10) if '取引結果' in trade_results["取引履歴"].columns else "No results")
        print("\n=== 取引履歴['取引結果']の記述統計 ===")
        print(trade_results["取引履歴"]["取引結果"].describe() if '取引結果' in trade_results["取引履歴"].columns else "No results")
        print("\n=== 損益推移['日次損益']の先頭20件 ===")
        print(trade_results["損益推移"]["日次損益"].head(20) if '日次損益' in trade_results["損益推移"].columns else "No results")
        print("\n=== 損益推移['日次損益']の記述統計 ===")
        print(trade_results["損益推移"]["日次損益"].describe() if '日次損益' in trade_results["損益推移"].columns else "No results")
        
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