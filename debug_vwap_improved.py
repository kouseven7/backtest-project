"""
VWAPブレイクアウト戦略の最適化デバッグスクリプト（より長いテスト期間）
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# より詳細なロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_debug_improved.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_debug")

def run_debug():
    """
    VWAPブレイクアウト戦略の最適化問題デバッグ（より多くのデータを使用）
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # より多くのデータを使用（最後の600日）
        test_data = stock_data.iloc[-600:].copy()
        test_index = index_data.iloc[-600:].copy() if index_data is not None else None
        
        logger.info(f"テストデータサイズ: {len(test_data)} 日分 ({test_data.index[0]} 〜 {test_data.index[-1]})")
        
        # ステップ2: 改善されたVWAP_Breakout戦略でバックテスト
        logger.info("■ ステップ2: 改善されたVWAP_Breakout戦略でバックテスト")
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        
        # パラメータを調整してより多くの取引を生成
        improved_params = {
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "sma_short": 10,
            "sma_long": 20,
            "volume_threshold": 1.2,  # 閾値を下げて取引回数を増やす
            "confirmation_bars": 1,
            "breakout_min_percent": 0.003,  # 閾値を下げて取引回数を増やす
            "trailing_stop": 0.05,
            "trailing_start_threshold": 0.03,
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
        strategy = VWAPBreakoutStrategy(test_data, test_index, params=improved_params)
        strategy.initialize_strategy()
        result_data = strategy.backtest()
        
        # 取引状況確認
        trade_count = (result_data['Entry_Signal'] == 1).sum()
        exit_count = (result_data['Exit_Signal'] == -1).sum()
        logger.info(f"バックテスト結果: エントリー数={trade_count}, イグジット数={exit_count}")
        
        # ステップ3: トレードシミュレーションと目的関数テスト
        logger.info("■ ステップ3: トレードシミュレーション")
        from trade_simulation import simulate_trades
        trade_results = simulate_trades(result_data, ticker)
        
        # 取引履歴の詳細確認
        trades_df = trade_results.get('取引履歴', pd.DataFrame())
        if not trades_df.empty:
            logger.info(f"取引数: {len(trades_df)}")
            logger.info(f"勝ち取引: {(trades_df['取引結果'] > 0).sum()} 件, 負け取引: {(trades_df['取引結果'] < 0).sum()} 件")
            logger.info(f"勝率: {(trades_df['取引結果'] > 0).sum() / len(trades_df) * 100:.2f}%")
            logger.info(f"平均損益: {trades_df['取引結果'].mean():.2f}円")
            logger.info(f"合計損益: {trades_df['取引結果'].sum():.2f}円")
        else:
            logger.warning("取引がありません！")
        
        # ステップ4: CompositeObjective関数の機能確認
        logger.info("■ ステップ4: CompositeObjective関数のテスト")
        from optimization.objective_functions import create_custom_objective
        
        # 複合目的関数をテスト
        objectives_config = [
            {"name": "sharpe_ratio", "weight": 1.0},
            {"name": "win_rate", "weight": 0.5},
            {"name": "expectancy", "weight": 0.5}
        ]
        
        composite_objective = create_custom_objective(objectives_config)
        try:
            composite_score = composite_objective(trade_results)
            logger.info(f"複合目的関数スコア: {composite_score}")
        except Exception as e:
            logger.error(f"複合目的関数でエラー: {e}")
        
        # ステップ5: ミニ最適化テスト（小さいグリッドで）
        logger.info("■ ステップ5: ミニ最適化テスト")
        try:
            # 小さなパラメータグリッドで最適化をテスト
            mini_param_grid = {
                "stop_loss": [0.03, 0.05],
                "take_profit": [0.08, 0.12],
                "volume_threshold": [1.2, 1.5]
            }
            
            from optimization.parameter_optimizer import ParameterOptimizer
            
            optimizer = ParameterOptimizer(
                data=test_data,
                strategy_class=VWAPBreakoutStrategy,
                param_grid=mini_param_grid,
                objective_function=composite_objective,
                strategy_kwargs={"index_data": test_index}
            )
            
            # 最適化実行
            results = optimizer.grid_search()
            
            # 結果確認
            if not results.empty:
                logger.info(f"最適化結果: {len(results)}件")
                best_params = results.iloc[0].to_dict()
                logger.info(f"最良スコア: {best_params['score']}")
                logger.info(f"最適パラメータ: {best_params}")
            else:
                logger.error("最適化結果が空です")
                
        except Exception as e:
            logger.error(f"最適化でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        logger.info("■ デバッグ完了")
        
    except Exception as e:
        logger.error(f"全体エラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_debug()
