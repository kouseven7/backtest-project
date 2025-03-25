# ファイル: trade_simulation.py
import pandas as pd
import logging
from config.logger_config import setup_logger
from config.file_utils import resolve_excel_file
from strategies_modules.strategy_parameter_manager import StrategyParameterManager
from data_fetcher import fetch_yahoo_data  # 適切なデータ取得モジュール
from strategies.gc_strategy_signal import GCStrategy
from output.excel_result_exporter import save_backtest_results

logger = setup_logger(__name__)

# Excel設定ファイルのパスを、解決関数で取得
config_file = resolve_excel_file(r"C:\Users\imega\Documents\my_backtest_project\config\backtest_config.xlsx")


def run_trade_simulation():
    """
    トレードシミュレーションを実行するメイン関数。
    1. Excelから戦略パラメータを取得
    2. 市場データを取得
    3. 戦略クラスをインスタンス化してシグナルを生成
    4. シミュレーションを実行してトレード履歴とパフォーマンスを計算
    5. 結果をExcelに出力する
    """
    try:
        # 1. 戦略パラメータの取得（例: GC戦略）
        param_manager = StrategyParameterManager(config_file)
        gc_params = param_manager.get_params("GC戦略")
        logger.info("GC戦略のパラメータを取得しました。")

        # 2. 市場データの取得（例: 銘柄設定シートから取得）
        stock_params = pd.read_excel(config_file, sheet_name="銘柄設定")
        ticker = stock_params["銘柄コード"].iloc[0]
        start_date = stock_params["開始日"].iloc[0]
        end_date = stock_params["終了日"].iloc[0]
        logger.info(f"市場データの取得対象: {ticker} {start_date}〜{end_date}")
        
        # 市場データの取得
        stock_data = fetch_yahoo_data(ticker, start_date, end_date)
        
        # 3. 戦略クラスのインスタンス化とシグナル生成
        # ここでは、"Adj Close" を指定します
        strategy = GCStrategy(stock_data, gc_params, price_column="Adj Close")
        result_data = strategy.generate_signals()
        logger.info("シグナルの生成が完了しました。")
        
        # 4. トレードシミュレーションの実行
        trade_history, performance_summary = simulate_trades(result_data)
        
        # 5. 結果の保存
        results = {
            "取引履歴": trade_history,
            "損益推移": performance_summary,
        }
        output_file = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\backtest_results.xlsx"
        save_backtest_results(results, output_file)
        logger.info("トレードシミュレーションの結果を保存しました。")
    
    except Exception as e:
        logger.exception("トレードシミュレーションの実行中にエラーが発生しました。")
        raise

def simulate_trades(data: pd.DataFrame, ticker: str) -> dict:
    """
    シンプルなトレードシミュレーションの例。
    
    Parameters:
        data (pd.DataFrame): シグナルが含まれた株価データ
        ticker (str): 対象の銘柄（例："8306.T"）
        
    Returns:
        dict: 取引履歴（"trade_history"）および累積損益（"pnl"）を含む辞書
    """
    # 取引履歴 DataFrame に「銘柄」カラムを追加
    trade_history = pd.DataFrame(columns=["日付", "銘柄", "エントリー", "イグジット", "取引結果"])
    cumulative_pnl = []
    cum_profit = 0
    
    for idx in range(len(data)):
        entry_signal = data["Entry_Signal"].iloc[idx]
        exit_signal = data["Exit_Signal"].iloc[idx]
        date = data.index[idx]
        
        # エントリー処理（例）：エントリーシグナルが 1 なら新しい行を追加
        if entry_signal == 1:
            entry_price = data["Close"].iloc[idx]
            trade_history.loc[len(trade_history)] = [date, ticker, entry_price, None, None]
        
        # イグジット処理（例）：シグナルが -1 かつ直前の行のイグジットが未入力の場合
        if exit_signal == -1 and not trade_history.empty and pd.isna(trade_history["イグジット"].iloc[-1]):
            exit_price = data["Close"].iloc[idx]
            trade_history.at[trade_history.index[-1], "イグジット"] = exit_price
            profit = exit_price - trade_history["エントリー"].iloc[-1]
            trade_history.at[trade_history.index[-1], "取引結果"] = profit
            cum_profit += profit
        
        cumulative_pnl.append(cum_profit)
    
    performance_summary = pd.DataFrame({
        "日付": data.index,
        "累積損益": cumulative_pnl
    })
    
    return {"取引履歴": trade_history, "損益推移": performance_summary}



if __name__ == "__main__":
    run_trade_simulation()
