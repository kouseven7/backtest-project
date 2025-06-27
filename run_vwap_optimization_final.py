"""
VWAPブレイクアウト戦略の最適化実行スクリプト（最終版）
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロガーの設定
log_filepath = r"C:\Users\imega\Documents\my_backtest_project\logs\vwap_final_optimization.log"
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_final")

def run_optimization():
    """
    VWAPブレイクアウト戦略の最終最適化実行
    - 改善されたパラメータグリッド
    - 改善された目的関数
    - より長い期間のデータ
    """
    try:
        # ステップ1: データ取得と準備
        logger.info("■ ステップ1: データ取得と準備")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # 最長のデータを使用
        logger.info(f"取得データ期間: {stock_data.index[0]} 〜 {stock_data.index[-1]} ({len(stock_data)}日)")
        logger.info(f"最適化データを準備中...")
        
        # ステップ2: 最適化の設定
        logger.info("■ ステップ2: 最適化設定")
        from optimization.configs import vwap_breakout_optimization
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        from optimization.objective_functions import create_custom_objective
        from optimization.parameter_optimizer import ParameterOptimizer
        
        # パラメータグリッドと目的関数
        param_grid = vwap_breakout_optimization.PARAM_GRID
        objectives_config = vwap_breakout_optimization.OBJECTIVES_CONFIG
        
        # 改良したCompositeObjective関数
        composite_objective = create_custom_objective(objectives_config)
        
        # ログ出力
        param_combinations = 1
        for param, values in param_grid.items():
            param_combinations *= len(values)
            logger.info(f"パラメータ {param}: {values}")
        logger.info(f"総組み合わせ数: {param_combinations}")
        
        # 目的関数の内容
        logger.info("目的関数設定:")
        for obj in objectives_config:
            logger.info(f"  {obj['name']} (重み: {obj['weight']})")
        
        # ステップ3: 最適化の実行
        logger.info("■ ステップ3: 最適化実行")
        
        # 最適化実行
        optimizer = ParameterOptimizer(
            data=stock_data,
            strategy_class=VWAPBreakoutStrategy,
            param_grid=param_grid,
            objective_function=composite_objective,
            strategy_kwargs={"index_data": index_data},
            output_dir=r"C:\Users\imega\Documents\my_backtest_project\backtest_results\optimization"
        )
        
        # 最適化実行（標準）
        logger.info("最適化を実行中... (これには時間がかかります)")
        results = optimizer.grid_search()
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = optimizer.save_results(f"vwap_optimization_{timestamp}", format="excel")
        logger.info(f"最適化結果を保存しました: {result_file}")
        
        # 結果の表示
        if not results.empty:
            logger.info(f"最適化結果: {len(results)}件")
            
            # 上位5件
            top_5 = results.head(5)
            logger.info("===上位5パラメータセット===")
            for i, row in top_5.iterrows():
                params = {k: v for k, v in row.items() if k != 'score'}
                logger.info(f"順位{i+1}: スコア={row['score']:.4f}")
                for key, value in params.items():
                    logger.info(f"  {key}: {value}")
        else:
            logger.error("最適化結果が空です")
        
        logger.info("■ 最適化完了")
        
    except Exception as e:
        logger.error(f"最適化プロセス中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

def run_best_backtest():
    """
    最適なパラメータでバックテスト実行
    """
    logger.info("■ 最適パラメータでのバックテスト実行")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # 最適パラメータ（実行結果から取得した値に置き換えてください）
        best_params = {
            "stop_loss": 0.05,
            "take_profit": 0.12,
            "sma_short": 8,
            "sma_long": 20,
            "volume_threshold": 1.2,
            "confirmation_bars": 1,
            "breakout_min_percent": 0.003,
            "trailing_stop": 0.05,
            "trailing_start_threshold": 0.04,
            "max_holding_period": 15,
            "market_filter_method": "none",
            "rsi_filter_enabled": False,
            "atr_filter_enabled": False,
            "partial_exit_enabled": True,
            "partial_exit_threshold": 0.07,
            "partial_exit_portion": 0.5,
            "rsi_period": 14,
            "rsi_lower": 30,
            "rsi_upper": 70,
            "volume_increase_mode": "average"
        }
        
        # 戦略を実行
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        strategy = VWAPBreakoutStrategy(stock_data, index_data, params=best_params)
        strategy.initialize_strategy()
        result_data = strategy.backtest()
        
        # 取引データ分析
        trade_count = (result_data['Entry_Signal'] == 1).sum()
        exit_count = (result_data['Exit_Signal'] == -1).sum()
        logger.info(f"バックテスト結果: エントリー数={trade_count}, イグジット数={exit_count}")
        
        # トレードシミュレーション
        from trade_simulation import simulate_trades
        trade_results = simulate_trades(result_data, ticker)
        
        # 取引履歴の詳細確認
        trades_df = trade_results.get('取引履歴', pd.DataFrame())
        if not trades_df.empty:
            logger.info(f"取引数: {len(trades_df)}")
            logger.info(f"勝ち取引: {(trades_df['取引結果'] > 0).sum()} 件, 負け取引: {(trades_df['取引結果'] < 0).sum()} 件")
            win_rate = (trades_df['取引結果'] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0
            logger.info(f"勝率: {win_rate:.2f}%")
            logger.info(f"平均損益: {trades_df['取引結果'].mean():.2f}円")
            logger.info(f"合計損益: {trades_df['取引結果'].sum():.2f}円")
            
            # 各種メトリクス計算
            if '損益推移' in trade_results:
                pnl_df = trade_results['損益推移']
                if not pnl_df.empty and '日次損益' in pnl_df.columns:
                    from metrics.performance_metrics import (
                        calculate_sharpe_ratio, calculate_sortino_ratio, 
                        calculate_max_drawdown, calculate_risk_return_ratio
                    )
                    daily_returns = pnl_df['日次損益']
                    sharpe = calculate_sharpe_ratio(daily_returns)
                    sortino = calculate_sortino_ratio(daily_returns)
                    max_dd = calculate_max_drawdown(pnl_df['累積損益']) if '累積損益' in pnl_df.columns else 0
                    
                    logger.info(f"シャープレシオ: {sharpe:.4f}")
                    logger.info(f"ソルティノレシオ: {sortino:.4f}")
                    logger.info(f"最大ドローダウン: {max_dd:.2f}%")
        else:
            logger.warning("取引がありません！")
        
        logger.info("■ バックテスト完了")
        
    except Exception as e:
        logger.error(f"最適パラメータでのバックテスト中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 利用可能なメモリ量を表示
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"システムメモリ: 合計={mem.total/(1024**3):.1f}GB, 利用可能={mem.available/(1024**3):.1f}GB")
    
    # コマンドライン引数によって挙動を変える
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_best_backtest()
    else:
        run_optimization()
