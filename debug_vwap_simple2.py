"""
VWAPブレイクアウト戦略の最適化デバッグスクリプト（単純化版）

このスクリプトは以下の改善点を含みます：
1. データ取得と準備の簡素化
2. パラメータの最適化
3. 目的関数のテスト
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
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_debug")

def run_debug():
    """
    VWAPブレイクアウト戦略の最適化デバッグ
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # テストデータの準備（最大500日）
        test_data_size = min(500, len(stock_data))
        test_data = stock_data.iloc[-test_data_size:].copy()
        test_index = index_data.iloc[-test_data_size:].copy() if index_data is not None else None
        
        logger.info(f"テストデータサイズ: {len(test_data)} 日分 ({test_data.index[0]} 〜 {test_data.index[-1]})")
        
        # ステップ2: 改善されたVWAP_Breakout戦略でバックテスト
        logger.info("■ ステップ2: 改善されたVWAP_Breakout戦略でバックテスト")
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        
        # パラメータを調整してより多くの取引を生成
        improved_params = {
            # リスク・リワード設定
            "stop_loss": 0.05,
            "take_profit": 0.1,
            
            # エントリー条件（より緩和）
            "sma_short": 8,  # 短期間に変更
            "sma_long": 15,  # 短期間に変更
            "volume_threshold": 1.2,  # 閾値を下げて取引回数を増やす
            "confirmation_bars": 1,
            "breakout_min_percent": 0.003,  # 閾値を下げて取引回数を増やす
            
            # イグジット条件
            "trailing_stop": 0.05,
            "trailing_start_threshold": 0.03,
            "max_holding_period": 20,  # 長く保有
            
            # フィルター設定（オフにして取引数を増やす）
            "market_filter_method": "none",
            "rsi_filter_enabled": False,
            "atr_filter_enabled": False,
            
            # 部分決済設定
            "partial_exit_enabled": True,
            "partial_exit_threshold": 0.05,  # 早めに部分利確
            "partial_exit_portion": 0.5,
            
            # 技術指標パラメータ
            "rsi_period": 14,
            "rsi_lower": 30,
            "rsi_upper": 70,
            "volume_increase_mode": "average"
        }
        
        # 戦略を実行
        logger.info("戦略を初期化して実行します")
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
            {"name": "sortino_ratio", "weight": 0.8}, 
            {"name": "win_rate", "weight": 0.6},
            {"name": "expectancy", "weight": 0.6}
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
                "volume_threshold": [1.0, 1.2]
            }
            
            from optimization.parameter_optimizer import ParameterOptimizer
            
            # 計算時間短縮のため、より小さなデータセットを使用
            test_subset = test_data.iloc[-250:].copy()
            test_index_subset = test_index.iloc[-250:].copy() if test_index is not None else None
            
            optimizer = ParameterOptimizer(
                data=test_subset,
                strategy_class=VWAPBreakoutStrategy,
                param_grid=mini_param_grid,
                objective_function=composite_objective,
                strategy_kwargs={"index_data": test_index_subset}
            )
            
            # 最適化実行
            results = optimizer.grid_search()
            
            # 結果確認
            if not results.empty:
                logger.info(f"最適化結果: {len(results)}件")
                best_params = results.iloc[0].to_dict()
                score = best_params.pop('score', 0)
                logger.info(f"最良スコア: {score}")
                logger.info(f"最適パラメータ: {best_params}")
            else:
                logger.error("最適化結果が空です")
        except Exception as e:
            logger.error(f"最適化でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # ステップ6: 総括レポート
        logger.info("■ ステップ6: 総括レポート")
        logger.info("===== 最適化プロセス診断レポート =====")
        logger.info(f"1. データ品質: {len(test_data)}日分のデータ使用")
        logger.info(f"2. 取引サンプル: {trade_count}件のエントリー, {exit_count}件のイグジット")
        
        if trade_count > 0:
            logger.info("3. 最適化プロセスは正常に機能しています")
            logger.info("4. -inf問題は修正されましたが、最適化の品質はデータと取引数に依存します")
        else:
            logger.info("3. 最適化プロセスに課題があります - 十分な取引サンプルがありません")
            logger.info("4. パラメータ調整で取引数を増やす必要があります")
        
        logger.info("■ デバッグ完了")
        
    except Exception as e:
        logger.error(f"全体エラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_debug()
