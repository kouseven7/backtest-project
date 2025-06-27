"""
VWAPブレイクアウト戦略のデバッグ用シンプルスクリプト
一部の機能を確認することで最適化プロセスのデバッグが行えます
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# より詳細なロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_debug")

def run_debug():
    """
    VWAPブレイクアウト戦略の動作を確認する
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        if stock_data is None:
            logger.error("データ取得に失敗しました")
            return
            
        logger.info(f"取得したデータ: {ticker}, {start_date}～{end_date}")
        logger.info(f"データ形状: {stock_data.shape}")
        logger.info(f"最初の5日分のデータ: \n{stock_data.head()}")
        
        # ステップ2: パラメータ確認
        logger.info("\n■ ステップ2: パラメータ確認")
        from optimization.configs.vwap_breakout_optimization import PARAM_GRID
        
        # 組み合わせ数を計算
        combinations = 1
        for param, values in PARAM_GRID.items():
            combinations *= len(values)
            logger.info(f"パラメータ '{param}': 値の数={len(values)}, 値={values}")
        
        logger.info(f"パラメータ組み合わせ総数: {combinations:,}")
        
        # ステップ3: バックテスト動作確認
        logger.info("\n■ ステップ3: バックテスト動作確認")
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        
        # デフォルトパラメータで戦略インスタンスを作成
        strategy = VWAPBreakoutStrategy(stock_data, index_data)
        strategy.initialize_strategy()
        result = strategy.backtest()
        
        # 結果確認
        trade_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        logger.info(f"バックテスト結果: エントリー数={trade_count}, イグジット数={exit_count}")
        
        # 最適化スモールバッチテスト
        logger.info("\n■ ステップ4: 最適化ミニテスト")
        
        try:
            from optimization.optimize_vwap_breakout_strategy import optimize_vwap_breakout_strategy
            # データの一部だけを使用してテスト
            test_data = stock_data.iloc[-200:].copy()
            test_index = index_data.iloc[-200:].copy() if index_data is not None else None
            mini_result = optimize_vwap_breakout_strategy(test_data, test_index, use_parallel=False)
            logger.info(f"ミニ最適化テスト成功: {len(mini_result)}件の結果")
        except Exception as e:
            logger.error(f"最適化テスト中にエラーが発生: {e}")
            logger.error(traceback.format_exc())
        
        logger.info("デバッグ完了")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    run_debug()
        stock_data['VWAP'] = calculate_vwap(stock_data, price_column='Adj Close', volume_column='Volume')
    
    # テスト用のパラメータ（ログでプラス収益が出ていたもの）
    params = {
        "stop_loss": 0.03,
        "take_profit": 0.1,
        "sma_short": 10,
        "sma_long": 30,
        "volume_threshold": 1.2,
        "confirmation_bars": 1,
        "breakout_min_percent": 0.003,
        "trailing_stop": 0.05,
        "trailing_start_threshold": 0.03,
        "max_holding_period": 10,
        "market_filter_method": "none",
        "rsi_filter_enabled": False,
        "atr_filter_enabled": False,
        "partial_exit_enabled": False,
        "rsi_period": 14,
        "volume_increase_mode": "simple"
    }
    
    # 戦略インスタンスを作成
    logger.info("戦略のバックテスト実行中...")
    strategy = VWAPBreakoutStrategy(stock_data, index_data=index_data, params=params)
    
    # バックテスト実行
    result_data = strategy.backtest()
      # トレード数と損益を確認
    # カラム名を確認して適切なカラムを選択
    available_columns = ['Position']
    for col in ['Entry_Signal', 'Exit_Signal', 'Entry_Price', 'Exit_Price']:
        if col in result_data.columns:
            available_columns.append(col)
    
    signals = result_data[available_columns]
    logger.info(f"利用可能なカラム: {result_data.columns.tolist()}")
    logger.info(f"シグナル統計:\n{signals.describe()}")
    
    # トレードシミュレーション
    logger.info("トレードシミュレーション実行中...")
    trade_results = simulate_trades(result_data, ticker)
    
    # トレード結果の診断
    logger.info("トレード結果の診断中...")
    diagnose_objective_function(trade_results)
    
    # 目的関数の個別テスト
    logger.info("目的関数の個別テスト中...")
    try:
        sharpe = sharpe_ratio_objective(trade_results)
        logger.info(f"シャープレシオ目的関数の結果: {sharpe}")
    except Exception as e:
        logger.error(f"シャープレシオ計算でエラー: {e}")
    
    try:
        sortino = sortino_ratio_objective(trade_results)
        logger.info(f"ソルティノレシオ目的関数の結果: {sortino}")
    except Exception as e:
        logger.error(f"ソルティノレシオ計算でエラー: {e}")
    
    try:
        expectancy = expectancy_objective(trade_results)
        logger.info(f"期待値目的関数の結果: {expectancy}")
    except Exception as e:
        logger.error(f"期待値計算でエラー: {e}")
    
    # 修正版のトレード結果でも試す
    logger.info("修正版のトレード結果でテスト中...")
    fixed_results = fix_trade_results(trade_results)
    
    try:
        sharpe = sharpe_ratio_objective(fixed_results)
        logger.info(f"修正後のシャープレシオ目的関数の結果: {sharpe}")
    except Exception as e:
        logger.error(f"修正後のシャープレシオ計算でエラー: {e}")
    
    # トレード履歴の基本統計を出力
    if '取引履歴' in trade_results and not trade_results['取引履歴'].empty:
        trades = trade_results['取引履歴']
        logger.info(f"取引総数: {len(trades)}")
        logger.info(f"取引合計損益: {trades['取引結果'].sum()}")
        logger.info(f"平均利益: {trades['取引結果'].mean()}")
        logger.info(f"勝率: {len(trades[trades['取引結果'] > 0]) / len(trades) * 100:.2f}%")
    
    # 損益推移の基本統計を出力
    if '損益推移' in trade_results and not trade_results['損益推移'].empty:
        pnl = trade_results['損益推移']
        if '日次損益' in pnl.columns:
            daily_returns = pnl['日次損益']
            logger.info(f"日次損益サンプル: \n{daily_returns.head()}")
            logger.info(f"日次損益統計: \n{daily_returns.describe()}")
            
            # ゼロ値の割合を確認
            zero_count = (daily_returns == 0).sum()
            logger.info(f"日次損益でゼロの日: {zero_count}/{len(daily_returns)} ({zero_count/len(daily_returns)*100:.2f}%)")
            
            # 等値やNaN, 無限大値がないか確認
            unique_values = daily_returns.nunique()
            logger.info(f"日次損益のユニーク値数: {unique_values}")
            
            has_nan = daily_returns.isna().any()
            logger.info(f"日次損益にNaN値: {has_nan}")
            
            has_inf = daily_returns.isin([np.inf, -np.inf]).any()
            logger.info(f"日次損益に無限大値: {has_inf}")

if __name__ == "__main__":
    diagnose_vwap_breakout_optimization()
