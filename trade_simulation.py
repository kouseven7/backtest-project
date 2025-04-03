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
    """
    try:
        # 1. 戦略パラメータの取得（例: GC戦略）
        param_manager = StrategyParameterManager(config_file)
        gc_params = param_manager.get_params("GC戦略")
        logger.info("GC戦略のパラメータを取得しました。")

        # 2. 市場データの取得（例: 銘柄設定シートから取得）
        stock_params = pd.read_excel(config_file, sheet_name="銘柄設定")
        ticker = stock_params["銘柄"].iloc[0]
        start_date = stock_params["開始日"].iloc[0]
        end_date = stock_params["終了日"].iloc[0]
        logger.info(f"市場データの取得対象: {ticker} {start_date}〜{end_date}")
        
        # 市場データの取得
        stock_data = fetch_yahoo_data(ticker, start_date, end_date)
        
        # 3. 戦略クラスのインスタンス化とシグナル生成
        strategy = GCStrategy(stock_data, gc_params, price_column="Adj Close")
        # generate_signals() メソッドがなく、代わりに backtest() メソッドを使用
        result_data = strategy.backtest()
        logger.info("シグナルの生成が完了しました。")
        
        # 4. トレードシミュレーションの実行
        trade_results = simulate_trades(result_data, ticker)
        
        # 5. 結果の保存
        output_file = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\backtest_results.xlsx"
        save_backtest_results(trade_results, output_file)
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
    from config.risk_management import RiskManagement
    
    # デバッグ：シグナルの数を出力
    entry_count = data["Entry_Signal"].sum() if "Entry_Signal" in data.columns else 0
    exit_count = (data["Exit_Signal"] == -1).sum() if "Exit_Signal" in data.columns else 0
    logger.info(f"シグナル確認: エントリー: {entry_count}回, イグジット: {exit_count}回")
    
    # 取引履歴 DataFrame に「銘柄」カラムを追加
    trade_history = pd.DataFrame(columns=["日付", "銘柄", "戦略", "エントリー", "イグジット", "取引結果", "取引量", "手数料", "リスク状態"])
    cumulative_pnl = []
    cum_profit = 0
    
    # リスク管理システムの初期化（シミュレーション用）
    risk_manager = RiskManagement(total_assets=1000000)  # 総資産100万円
    
    # データにリスク管理状態のメタデータがある場合、それを使用
    risk_mgr_state = data.attrs.get('risk_management', {})
    
    # 複数ポジションを追跡するための辞書
    # {取引ID: [エントリー日, エントリー価格, 戦略名, ポジションサイズ, エントリーインデックス]}
    active_positions = {}
    next_trade_id = 1
    
    # エントリー日とイグジット日のペアを記録
    entry_exit_pairs = []
    
    # まず、全てのエントリーとイグジットのペアを見つける
    in_position = False
    entry_idx = -1
    entry_date = None
    
    for idx in range(len(data)):
        date = data.index[idx]
        entry_signal = data["Entry_Signal"].iloc[idx] if "Entry_Signal" in data.columns else 0
        exit_signal = data["Exit_Signal"].iloc[idx] if "Exit_Signal" in data.columns else 0
        
        # エントリーシグナルがあり、ポジションを持っていない場合
        if entry_signal == 1 and not in_position:
            in_position = True
            entry_idx = idx
            entry_date = date
        
        # イグジットシグナルがあり、ポジションを持っている場合
        if exit_signal == -1 and in_position:
            entry_exit_pairs.append((entry_idx, entry_date, idx, date))
            in_position = False
    
    # 最後にポジションが残っている場合、最終日にクローズ
    if in_position and entry_idx >= 0:
        final_idx = len(data) - 1
        final_date = data.index[final_idx]
        entry_exit_pairs.append((entry_idx, entry_date, final_idx, final_date))
    
    logger.info(f"取引ペア数: {len(entry_exit_pairs)}")
    
    # 各エントリー・イグジットのペアについて取引を処理
    for entry_idx, entry_date, exit_idx, exit_date in entry_exit_pairs:
        strategy_name = data["Strategy"].iloc[entry_idx] if "Strategy" in data.columns and data["Strategy"].iloc[entry_idx] != "" else "デフォルト戦略"
        position_size = data["Position_Size"].iloc[entry_idx] if "Position_Size" in data.columns else 1
        
        # エントリー価格を取得
        entry_price = data["Close"].iloc[entry_idx] if "Close" in data.columns else data["Adj Close"].iloc[entry_idx]
        
        # イグジット価格を取得
        exit_price = data["Close"].iloc[exit_idx] if "Close" in data.columns else data["Adj Close"].iloc[exit_idx]
        
        # 損益計算
        profit = exit_price - entry_price
        
        # 取引量と手数料
        trade_amount = 100000 * position_size  # 10万円 × ポジションサイズ
        fee = trade_amount * 0.001  # 0.1%手数料
        profit_after_fee = (profit / entry_price) * trade_amount - fee
        
        # リスク管理の状態をJSON文字列に変換して保存
        risk_state_str = str({})  # 簡略化のため空の辞書を使用
        
        # 取引履歴に追加
        trade_history.loc[len(trade_history)] = [
            entry_date, 
            ticker, 
            strategy_name,
            entry_price, 
            exit_price, 
            profit_after_fee, 
            trade_amount,
            fee,
            risk_state_str
        ]
        
        cum_profit += profit_after_fee
        logger.debug(f"取引: エントリー {entry_date} @ {entry_price}, イグジット {exit_date} @ {exit_price}, 損益: {profit_after_fee:.2f}円")
        
    logger.info(f"合計取引数: {len(trade_history)}件, 合計損益: {cum_profit:.2f}円")
    
    # 損益推移の計算
    if len(trade_history) > 0:
        # 損益合計を再計算
        trade_history["取引結果"] = trade_history["取引結果"].astype(float)
        total_profit = trade_history["取引結果"].sum()
        
        # 損益推移を作成
        dates = data.index
        daily_pnl = pd.Series(0.0, index=dates)
        
        # 取引履歴から各日の損益を計算
        for _, row in trade_history.iterrows():
            exit_date = row["日付"]  # エントリー日を基準に計上
            if exit_date in daily_pnl.index:
                daily_pnl[exit_date] += row["取引結果"]
        
        # 累積損益の計算
        cumulative_pnl = daily_pnl.cumsum()
        
        # 損益推移データフレームの作成
        performance_summary = pd.DataFrame({
            "日付": cumulative_pnl.index,
            "日次損益": daily_pnl.values,
            "累積損益": cumulative_pnl.values
        })
        
        # 基本的なパフォーマンス指標を追加
        win_trades = len(trade_history[trade_history["取引結果"] > 0])
        loss_trades = len(trade_history[trade_history["取引結果"] <= 0])
        total_trades = len(trade_history)
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        
        # リスク管理統計も追加
        max_drawdown = 0
        drawdown_series = []
        peak = 0
        
        for value in cumulative_pnl.values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            drawdown_series.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # リスクリターン比率
        risk_return_ratio = total_profit / max_drawdown if max_drawdown > 0 else float('inf')
        
        performance_metrics = pd.DataFrame({
            "指標": [
                "総取引数", "勝率", "損益合計", "平均損益", "最大利益", "最大損失",
                "最大ドローダウン(%)", "リスクリターン比率"
            ],
            "値": [
                total_trades,
                f"{win_rate:.2f}%",
                f"{total_profit:.2f}円",
                f"{(total_profit/total_trades if total_trades > 0 else 0):.2f}円",
                f"{trade_history['取引結果'].max() if not trade_history.empty else 0:.2f}円",
                f"{trade_history['取引結果'].min() if not trade_history.empty else 0:.2f}円",
                f"{max_drawdown:.2f}%",
                f"{risk_return_ratio:.2f}"
            ]
        })
    else:
        # 取引がない場合のダミーデータ
        performance_summary = pd.DataFrame({
            "日付": data.index,
            "日次損益": [0] * len(data),
            "累積損益": [0] * len(data)
        })
        performance_metrics = pd.DataFrame({
            "指標": ["総取引数", "勝率", "損益合計", "最大ドローダウン(%)", "リスクリターン比率"],
            "値": ["0", "0%", "0円", "0%", "0"]
        })
    
    # リスク管理情報を追加
    risk_summary = pd.DataFrame({
        "リスク管理設定": [
            "総資産",
            "最大許容ドローダウン",
            "1回の取引あたりの最大損失",
            "同日での最大連敗数",
            "最大ポジション数"
        ],
        "値": [
            f"{risk_manager.total_assets:,}円",
            f"{risk_manager.max_drawdown * 100:.1f}%",
            f"{risk_manager.max_loss_per_trade * 100:.1f}%",
            f"{risk_manager.max_daily_losses}回",
            f"{risk_manager.max_total_positions}ポジション"
        ]
    })
    
    return {
        "取引履歴": trade_history, 
        "損益推移": performance_summary,
        "パフォーマンス指標": performance_metrics,
        "リスク管理設定": risk_summary
    }


if __name__ == "__main__":
    run_trade_simulation()
