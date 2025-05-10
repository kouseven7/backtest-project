"""
戦略のパラメータ最適化を実行するスクリプト
"""
import argparse
import logging
from datetime import datetime

# 戦略クラスのインポート
from strategies.Breakout import BreakoutStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy  # 追加インポート

# 最適化モジュールのインポート
from optimization.optimize_breakout_strategy import optimize_breakout_strategy
from optimization.optimize_gc_strategy import optimize_gc_strategy
from optimization.optimize_contrarian_strategy import optimize_contrarian_strategy  # 追加
from optimization.optimize_momentum_strategy import optimize_momentum_strategy  # 追加インポート
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

def main():
    parser = argparse.ArgumentParser(description='戦略パラメータの最適化')
    parser.add_argument('--strategy', type=str, default='breakout',
                        choices=['breakout', 'vwap_breakout', 'momentum', 'gc', 'contrarian'],
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
        'contrarian': optimize_contrarian_strategy,
        'gc': optimize_gc_strategy,
        'momentum': optimize_momentum_strategy,  # この行を追加
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