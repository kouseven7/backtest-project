#main.py

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")
from config.file_utils import resolve_excel_file

from config.logger_config import setup_logger

# Excel設定ファイルのパスを、解決関数で取得
config_file = resolve_excel_file(r"C:\Users\imega\Documents\my_backtest_project\config\backtest_config.xlsx")
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\backtest.log")


def get_parameters_and_data():
    """
    Excel設定ファイルからパラメータ取得と市場データ取得（キャッシュ利用）を行います。
    Returns:
        ticker (str), start_date (str), end_date (str), stock_data (pd.DataFrame)
    """
    from config.error_handling import read_excel_parameters, fetch_stock_data
    # ※ Excel の実際のカラム名に合わせてください（ここでは「銘柄」と仮定）
    config_df = read_excel_parameters(config_file, "銘柄設定")
    ticker = config_df["銘柄"].iloc[0]
    start_date = config_df["開始日"].iloc[0].strftime('%Y-%m-%d')
    end_date = config_df["終了日"].iloc[0].strftime('%Y-%m-%d')
    logger.info(f"パラメータ取得: {ticker}, {start_date}, {end_date}")

    # キャッシュを無視して常に新規にデータ取得する
    from config.cache_manager import get_cache_filepath, save_cache
    cache_filepath = get_cache_filepath(ticker, start_date, end_date)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    save_cache(stock_data, cache_filepath)
    
    # 'Adj Close' がない場合は 'Close' を代用
    if 'Adj Close' not in stock_data.columns:
        logger.warning(f"'{ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を代用します。")
        stock_data['Adj Close'] = stock_data['Close']
    
    # カラムが MultiIndex になっている場合はフラット化
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    return ticker, start_date, end_date, stock_data


def preprocess_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    前処理として、日次リターンと累積リターン、ボラティリティを計算します。
    """
    import preprocessing.returns as returns
    stock_data = returns.add_returns(stock_data, price_column="Adj Close")
    import preprocessing.volatility as volatility
    stock_data = volatility.add_volatility(stock_data)
    logger.info("前処理（リターン、ボラティリティ計算）完了")
    return stock_data


def compute_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    基本インジケーター、ボリンジャーバンド、出来高関連指標を計算して追加します。
    """
    from indicators.basic_indicators import add_basic_indicators
    stock_data = add_basic_indicators(stock_data, price_column="Adj Close")
    from indicators.bollinger_atr import bollinger_atr
    stock_data = bollinger_atr(stock_data, price_column="Adj Close")
    from indicators.volume_indicators import add_volume_indicators
    stock_data = add_volume_indicators(stock_data, price_column="Adj Close")
    logger.info("インジケーター計算完了")
    return stock_data


def apply_strategy(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    戦略パラメータを用いてシグナルを生成します。
    """
    from strategies.gc_strategy_signal import GCStrategy
    # 戦略パラメータの例。必要に応じて strategy_parameter_manager から取得してください。
    params = {"短期移動平均": 5, "長期移動平均": 25, "利益確定％": 5, "損切割合％": -3}
    strategy = GCStrategy(stock_data, params, price_column="Adj Close")
    result_data = strategy.generate_signals()
    logger.info("戦略適用（シグナル生成）完了")
    return result_data


def simulate_and_save(result_data: pd.DataFrame, ticker: str):
    """
    バックテストシミュレーションを実行し、結果をExcelに出力します。
    """
    import trade_simulation
    # simulate_trades 関数には result_data と ticker を渡して、取引履歴に銘柄情報を追加します。
    trade_results = trade_simulation.simulate_trades(result_data, ticker)
    logger.info("バックテスト（トレードシミュレーション）完了")
    
    output_file = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\backtest_results.xlsx"
    from output.excel_result_exporter import ensure_workbook_exists, add_pnl_chart, create_pivot_from_trade_history, save_backtest_results
    ensure_workbook_exists(output_file)
    # trade_results のキーを「取引履歴」と「損益推移」にしている場合
    save_backtest_results(trade_results, output_file)
    logger.info("バックテスト結果をExcelに出力完了")
    add_pnl_chart(output_file, sheet_name="損益推移", chart_title="累積損益推移")
    create_pivot_from_trade_history(output_file, trade_sheet="取引履歴", pivot_sheet="Pivot_取引履歴")
    logger.info("Excelのチャート、ピボットテーブル追加完了")
    return trade_results


def main():
    try:
        ticker, start_date, end_date, stock_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        result_data = apply_strategy(stock_data)
        simulate_and_save(result_data, ticker)
        logger.info("全体のバックテスト処理が正常に完了しました。")
    except Exception as e:
        logger.exception("バックテスト実行中にエラーが発生しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()
