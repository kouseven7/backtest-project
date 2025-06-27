"""
バックテスト・最適化システムの改善版実行スクリプト

このスクリプトは、VWAPブレイクアウト戦略の最適化を改善された方法で実行し、
詳細な分析とダッシュボード生成を行います。
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse
import webbrowser
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # GUIが不要なバックエンドを使用

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロガーの設定
log_filename = f"enhanced_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(r"C:\Users\imega\Documents\my_backtest_project\logs", log_filename)
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("enhanced_workflow")


def setup_directories():
    """必要なディレクトリを設定"""
    base_dir = r"C:\Users\imega\Documents\my_backtest_project"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリ
    output_dir = os.path.join(base_dir, "backtest_results", f"enhanced_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析結果ディレクトリ
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 最適化結果ディレクトリ
    optimization_dir = os.path.join(output_dir, "optimization")
    os.makedirs(optimization_dir, exist_ok=True)
    
    # グラフディレクトリ
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "analysis_dir": analysis_dir,
        "optimization_dir": optimization_dir,
        "graphs_dir": graphs_dir,
        "timestamp": timestamp
    }


def run_enhanced_workflow(strategy: str = "VWAP_Breakout", data_years: int = 3,
                         parallel: bool = True, jobs: int = 2, 
                         open_dashboard: bool = True):
    """
    改善された最適化ワークフローを実行
    
    Parameters:
        strategy (str): 最適化する戦略名
        data_years (int): 使用するデータの年数
        parallel (bool): 並列処理を使用するかどうか
        jobs (int): 並列処理で使用するCPUコア数
        open_dashboard (bool): ダッシュボードをブラウザで開くかどうか
    """
    start_time = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info(f"改善された最適化ワークフローを開始: {strategy}")
        logger.info("=" * 80)
        
        # ディレクトリ設定
        dirs = setup_directories()
        
        # データ取得
        logger.info("■ ステップ1: データ取得と準備")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # データ期間の調整
        logger.info(f"取得データ期間: {stock_data.index[0]} 〜 {stock_data.index[-1]} ({len(stock_data)}日)")
        
        if data_years > 0 and len(stock_data) > data_years * 252:  # 約252取引日/年
            logger.info(f"データを最新の{data_years}年分に制限します")
            data_days = data_years * 252
            stock_data = stock_data.iloc[-data_days:]
            if index_data is not None and len(index_data) > data_days:
                index_data = index_data.iloc[-data_days:]
        
        logger.info(f"使用データ期間: {stock_data.index[0]} 〜 {stock_data.index[-1]} ({len(stock_data)}日)")
        
        # 戦略固有の処理
        if strategy == "VWAP_Breakout":
            # 戦略とパラメータグリッドのインポート
            logger.info("■ ステップ2: 戦略とパラメータグリッドの準備")
            from strategies.VWAP_Breakout import VWAPBreakoutStrategy
            from optimization.configs import vwap_breakout_optimization
            param_grid = vwap_breakout_optimization.get_param_grid()
            strategy_class = VWAPBreakoutStrategy
            
            # 強化された目的関数の準備
            from utils.optimization_utils import safe_score_calculation
            from optimization.objective_functions import create_custom_objective
            
            @safe_score_calculation
            def enhanced_objective(trade_results):
                """強化された複合目的関数"""
                # シャープレシオ、勝率、期待値の組み合わせ
                return create_custom_objective(
                    ["sharpe_ratio", "win_rate", "expectancy", "risk_return_ratio"],
                    [0.4, 0.3, 0.2, 0.1]  # 重み付け
                )(trade_results)
                
            objective_function = enhanced_objective
            
        else:
            logger.error(f"未対応の戦略: {strategy}")
            return
        
        # パラメータグリッド情報をログ出力
        logger.info("パラメータグリッド:")
        total_combinations = 1
        for param, values in param_grid.items():
            total_combinations *= len(values)
            logger.info(f"- {param}: {values} ({len(values)}通り)")
        logger.info(f"パラメータ組み合わせ総数: {total_combinations}")
        
        # 最適化の実行
        logger.info("■ ステップ3: 最適化の実行")
        from optimization.parameter_optimizer import ParameterOptimizer
        
        optimizer = ParameterOptimizer(
            data=stock_data,
            strategy_class=strategy_class,
            param_grid=param_grid,
            objective_function=objective_function,
            output_dir=dirs["optimization_dir"],
            strategy_kwargs={"index_data": index_data}
        )
        
        # 並列または通常の最適化を実行
        if parallel:
            logger.info(f"並列処理で最適化を実行 (n_jobs={jobs})...")
            results_df = optimizer.parallel_grid_search(n_jobs=jobs)
        else:
            logger.info("シングルプロセスで最適化を実行...")
            results_df = optimizer.grid_search()
        
        # 結果の保存
        results_csv = os.path.join(dirs["optimization_dir"], f"{strategy}_results_{dirs['timestamp']}.csv")
        results_df.to_csv(results_csv, index=False, encoding='utf-8')
        logger.info(f"最適化結果を保存: {results_csv}")
        
        # 結果の検証
        logger.info("■ ステップ4: 最適化結果の検証と分析")
        from utils.optimization_utils import validate_optimization_results
        
        validated_results = validate_optimization_results(
            results_df=results_df,
            param_grid=param_grid,
            min_trades=10
        )
        
        # パラメータ影響度の分析
        logger.info("パラメータ影響度を分析中...")
        from utils.strategy_analysis import create_parameter_impact_summary
        
        impact_file = os.path.join(dirs["analysis_dir"], f"{strategy}_parameter_impact.md")
        create_parameter_impact_summary(
            results_df=validated_results,
            param_grid=param_grid,
            output_file=impact_file
        )
        logger.info(f"パラメータ影響度分析を保存: {impact_file}")
        
        # ダッシュボード生成
        logger.info("■ ステップ5: 最適化ダッシュボードの生成")
        from utils.create_optimization_dashboard import create_optimization_dashboard
        
        dashboard_file = create_optimization_dashboard(
            results_file=results_csv,
            output_dir=dirs["output_dir"]
        )
        
        if dashboard_file and open_dashboard:
            logger.info(f"ダッシュボードをブラウザで開きます: {dashboard_file}")
            webbrowser.open('file://' + os.path.abspath(dashboard_file))
        
        # 最適パラメータでのバックテスト
        logger.info("■ ステップ6: 最適パラメータでのバックテスト")
        
        if not validated_results.empty:
            best_params = validated_results.loc[validated_results['score'].idxmax()].drop(['score'])
            if 'error' in best_params:
                best_params = best_params.drop(['error'])
            best_params_dict = best_params.to_dict()
            
            logger.info(f"最適パラメータ: {best_params_dict}")
            
            # 最適パラメータでの戦略インスタンス作成
            strategy_instance = strategy_class(stock_data, **best_params_dict, index_data=index_data)
            
            # バックテスト実行
            logger.info("最適パラメータでバックテストを実行中...")
            backtest_results = strategy_instance.backtest()
            
            # トレード分析の実行
            logger.info("■ ステップ7: トレード結果の詳細分析")
            from utils.trade_analyzer import TradeAnalyzer
            
            analyzer = TradeAnalyzer(
                trade_results=backtest_results,
                strategy_name=strategy,
                parameters=best_params_dict
            )
            
            analysis_results = analyzer.analyze_all(dirs["analysis_dir"])
            
            # パフォーマンスプロットの生成
            logger.info("パフォーマンスプロットを生成中...")
            from utils.strategy_analysis import generate_performance_plots
            
            plot_files = generate_performance_plots(
                trade_results=backtest_results,
                output_dir=dirs["graphs_dir"],
                strategy_name=strategy
            )
            logger.info(f"{len(plot_files)}個のパフォーマンスグラフを生成しました")
            
            # パフォーマンスサマリーのエクスポート
            from utils.optimization_utils import export_strategy_performance_summary
            
            summary_file = export_strategy_performance_summary(
                trade_results=backtest_results,
                params=best_params_dict,
                output_dir=dirs["output_dir"],
                strategy_name=strategy
            )
            logger.info(f"パフォーマンスサマリーを保存: {summary_file}")
            
        else:
            logger.warning("有効な最適化結果がありません。バックテストをスキップします。")
        
        # ワークフロー完了
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"改善された最適化ワークフローが完了しました。処理時間: {total_time:.1f}秒")
        logger.info(f"すべての結果は以下のディレクトリに保存されています: {dirs['output_dir']}")
        logger.info("=" * 80)
        
        return dirs["output_dir"]
    
    except Exception as e:
        logger.exception(f"改善された最適化ワークフロー中にエラーが発生: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改善された最適化ワークフローを実行します')
    parser.add_argument('--strategy', type=str, default="VWAP_Breakout",
                       help='最適化する戦略の名前')
    parser.add_argument('--years', type=int, default=3,
                       help='使用するデータの年数')
    parser.add_argument('--no-parallel', action='store_true',
                       help='並列処理を無効にする')
    parser.add_argument('--jobs', type=int, default=2,
                       help='並列処理で使用するCPUコア数')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='ダッシュボードを自動で開かない')
    
    args = parser.parse_args()
    
    run_enhanced_workflow(
        strategy=args.strategy,
        data_years=args.years,
        parallel=not args.no_parallel,
        jobs=args.jobs,
        open_dashboard=not args.no_dashboard
    )
