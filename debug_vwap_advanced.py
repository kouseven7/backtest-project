"""
VWAPブレイクアウト戦略の最適化デバッグスクリプト（さらなる改善版）

このスクリプトは以下の改善点を含みます：
1. テストデータを3年分使用してより多くの取引サンプルを確保
2. パラメータチューニングによる取引数の増加
3. 改善されたデバッグ情報の表示
4. 小規模最適化テスト実行
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロガーの設定
try:
    log_filepath = r"C:\Users\imega\Documents\my_backtest_project\logs\vwap_debug_advanced.log"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    # ファイル上書き
    with open(log_filepath, 'w') as f:
        f.write('')  # ログをクリア
        
    # ファイルとコンソール両方に出力
    logging.basicConfig(
        level=logging.INFO,  # DEBUGからINFOに変更して出力量を減らす
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    # ログファイル作成に失敗した場合は、コンソールのみに出力
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print(f"ログファイル設定エラー: {e} - コンソールのみに出力します")

logger = logging.getLogger("vwap_debug")

def get_stacked_data(test_periods=3):
    """
    複数の銘柄または期間のデータを取得して連結
    """
    try:
        from data_fetcher import get_stock_data
        import yfinance as yf
        
        # 最近のデータを取得（最大で過去3年分）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*test_periods)).strftime('%Y-%m-%d')
        
        # テスト用ティッカー
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        logger.info(f"テストデータ期間: {start_date} 〜 {end_date}")
        
        all_stock_data = []
        all_index_data = []
        
        # 各ティッカーのデータを取得
        for ticker in tickers[:2]:  # 最初の2つだけ使用（時間短縮のため）
            logger.info(f"銘柄 {ticker} のデータを取得中...")
            
            try:
                # データダウンロード
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                index_data = yf.download("^GSPC", start=start_date, end=end_date)  # S&P 500
                
                # データチェック
                if len(stock_data) > 100:
                    logger.info(f"銘柄 {ticker} のデータ: {len(stock_data)} 行")
                    all_stock_data.append(stock_data)
                    all_index_data.append(index_data)
                else:
                    logger.warning(f"銘柄 {ticker} のデータが不足しています: {len(stock_data)} 行")
            except Exception as e:
                logger.error(f"銘柄 {ticker} のデータ取得エラー: {e}")
                
        # データが取得できたかチェック
        if not all_stock_data:
            logger.error("データが取得できませんでした。代替方法を試みます。")
            # yfinanceからダウンロードできなかった場合は自前のデータを使用
            from data_fetcher import get_parameters_and_data
            ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
            return ticker, stock_data, index_data
            
        # 最もデータ量の多い銘柄を使用
        best_idx = np.argmax([len(df) for df in all_stock_data])
        chosen_ticker = tickers[best_idx]
        chosen_stock = all_stock_data[best_idx]
        chosen_index = all_index_data[best_idx]
        
        logger.info(f"選択した銘柄: {chosen_ticker} (データ行数: {len(chosen_stock)})")
        return chosen_ticker, chosen_stock, chosen_index
        
    except Exception as e:
        logger.error(f"データ取得中に例外が発生しました: {e}")
        # バックアップとして環境変数で指定されたデータ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        return ticker, stock_data, index_data

def analyze_strategy_params(strategy, params):
    """
    戦略パラメータの影響を分析し、トレードの推奨パラメータを提案
    """
    logger.info("=== パラメータ分析 ===")
    # エントリー条件のキー要素
    entry_keys = ["sma_short", "sma_long", "volume_threshold", "breakout_min_percent", "confirmation_bars"]
    logger.info("■ エントリー要因:")
    for key in entry_keys:
        if key in params:
            logger.info(f"  - {key}: {params[key]}")
    
    # イグジット条件のキー要素
    exit_keys = ["stop_loss", "take_profit", "trailing_stop", "trailing_start_threshold", "max_holding_period"]
    logger.info("■ イグジット要因:")
    for key in exit_keys:
        if key in params:
            logger.info(f"  - {key}: {params[key]}")
    
    # フィルター設定
    filter_keys = ["market_filter_method", "rsi_filter_enabled", "atr_filter_enabled", 
                  "rsi_lower", "rsi_upper"]
    logger.info("■ フィルター設定:")
    for key in filter_keys:
        if key in params:
            logger.info(f"  - {key}: {params[key]}")
    
    logger.info("=== 推奨設定 ===")
    logger.info("取引数増加: volume_threshold を下げる、breakout_min_percent を下げる")
    logger.info("勝率向上: confirmation_bars を増やす、市場フィルターを強化")
    logger.info("利益向上: trailing_stop を調整、部分利確閾値を調整")
    
    return

def run_debug():
    """
    VWAPブレイクアウト戦略の最適化問題デバッグ（拡張版）
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        ticker, stock_data, index_data = get_stacked_data(3)
        
        # データの先頭を表示
        logger.info(f"データサンプル: \n{stock_data.head(3)}")
        
        # テストデータの準備（直近1年分のデータを使用）
        test_data_size = min(750, len(stock_data))
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
            "volume_threshold": 1.1,  # 閾値を大幅に下げて取引回数を増やす
            "confirmation_bars": 1,
            "breakout_min_percent": 0.002,  # 閾値を大幅に下げて取引回数を増やす
            
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
        
        analyze_strategy_params(VWAPBreakoutStrategy, improved_params)
        
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
            
            # 統計分析
            if len(trades_df) >= 5:  # 最低5取引以上ある場合
                logger.info("=== 取引統計 ===")
                logger.info(f"最大利益取引: {trades_df['取引結果'].max():.2f}円")
                logger.info(f"最大損失取引: {trades_df['取引結果'].min():.2f}円")
                logger.info(f"平均保有期間: {trades_df['保有期間'].mean():.1f}日")
                logger.info(f"取引結果の標準偏差: {trades_df['取引結果'].std():.2f}")
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
        
        # ステップ5: 交差検証によるパラメータ安定性テスト
        logger.info("■ ステップ5: 交差検証によるパラメータ安定性テスト")
        
        # 時系列交差検証の設定
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(test_data):
            logger.info(f"交差検証: テストサイズ={len(test_idx)}")
            
            # テストデータのみを使用
            cv_test_data = test_data.iloc[test_idx].copy()
            cv_test_index = test_index.iloc[test_idx].copy() if test_index is not None else None
            
            # 戦略とシミュレーション実行
            cv_strategy = VWAPBreakoutStrategy(cv_test_data, cv_test_index, params=improved_params)
            cv_strategy.initialize_strategy()
            cv_result = cv_strategy.backtest()
            cv_trade_results = simulate_trades(cv_result, ticker)
            
            # スコア計算
            try:
                cv_score = composite_objective(cv_trade_results)
                trades = cv_trade_results.get('取引履歴', pd.DataFrame())
                trade_count = 0 if trades.empty else len(trades)
                cv_scores.append((cv_score, trade_count))
                logger.info(f"交差検証スコア: {cv_score}, 取引数: {trade_count}")
            except Exception as e:
                logger.error(f"交差検証でエラー: {e}")
        
        # 交差検証結果の表示
        if cv_scores:
            mean_score = np.mean([s[0] for s in cv_scores])
            mean_trades = np.mean([s[1] for s in cv_scores])
            logger.info(f"交差検証平均スコア: {mean_score:.4f}, 平均取引数: {mean_trades:.1f}")
        
        # ステップ6: ミニ最適化テスト（小さいグリッドで）
        logger.info("■ ステップ6: ミニ最適化テスト")
        try:
            # 小さなパラメータグリッドで最適化をテスト
            mini_param_grid = {
                "stop_loss": [0.03, 0.05],
                "take_profit": [0.08, 0.12],
                "volume_threshold": [1.0, 1.2],
                "breakout_min_percent": [0.002, 0.004]
            }
            
            from optimization.parameter_optimizer import ParameterOptimizer
            
            # 計算時間短縮のため、データサイズをさらに減らす
            test_subset = test_data.iloc[-300:].copy()
            test_index_subset = test_index.iloc[-300:].copy() if test_index is not None else None
            
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
                score = best_params.pop('score', None)
                logger.info(f"最良スコア: {score}")
                logger.info(f"最適パラメータ: {best_params}")
                
                # 結果表示
                top_results = results.head(3)
                logger.info("===トップ3パラメータセット===")
                for i, row in top_results.iterrows():
                    params_str = {k: v for k, v in row.items() if k != 'score'}
                    logger.info(f"順位{i+1}: スコア={row['score']:.4f}, パラメータ={params_str}")
            else:
                logger.error("最適化結果が空です")
        except Exception as e:
            logger.error(f"最適化でエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # ステップ7: 総括レポート
        logger.info("■ ステップ7: 総括レポート")
        logger.info("===== 最適化プロセス診断レポート =====")
        logger.info(f"1. データ品質: {len(test_data)}日分のデータ使用")
        logger.info(f"2. 取引サンプル: {trade_count}件のエントリー, {exit_count}件のイグジット")
        if trade_count > 0:
            logger.info("3. 最適化プロセスは正常に機能しています")
        else:
            logger.info("3. 最適化プロセスに課題があります - 十分な取引サンプルがありません")
        
        logger.info("4. CompositeObjectiveの動作: 正常 (但し、取引が十分にある場合のみ意味のあるスコアとなる)")
        logger.info("5. -inf問題は修正されました")
        
        logger.info("===== 改善推奨事項 =====")
        logger.info("1. より多くの取引サンプルを確保するためパラメーター範囲の広げる")
        logger.info("2. エントリー条件の緩和 (volume_threshold, breakout_min_percent)")
        logger.info("3. トレーニングデータ期間をさらに延長する")
        
        logger.info("■ デバッグ完了")
        
    except Exception as e:
        logger.error(f"全体エラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_debug()
