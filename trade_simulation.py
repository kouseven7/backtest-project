"""
Module: Trade Simulation
File: trade_simulation.py
Description: 
  トレードシミュレーションを実行し、バックテスト結果を生成するモジュールです。
  戦略のパラメータ管理、データ取得、シグナル生成、結果保存を含みます。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - config.logger_config
  - config.file_utils
  - strategies_modules.strategy_parameter_manager
  - data_fetcher
  - strategies.gc_strategy_signal
  - output.excel_result_exporter
"""

# ファイル: trade_simulation.py
import pandas as pd
import logging
from config.logger_config import setup_logger
from config.file_utils import resolve_excel_file
from strategies_modules.strategy_parameter_manager import StrategyParameterManager
from data_fetcher import fetch_yahoo_data  # 適切なデータ取得モジュール
from strategies.gc_strategy_signal import GCStrategy
from output.excel_result_exporter import save_backtest_results
from metrics.performance_metrics import (
    calculate_total_trades,
    calculate_win_rate,
    calculate_total_profit,
    calculate_average_profit,
    calculate_max_profit,
    calculate_max_loss,
    calculate_max_drawdown,
    calculate_risk_return_ratio
)

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
    
    # エントリー日とイグジット日のペアを記録
    entry_exit_pairs = []
      # シミュレーションループ
    in_position = False
    entry_idx = -1
    entry_date = None
    strategy_name = ""
    
    # データフレームにStrategyカラムがない場合は、戦略名を引数から取得
    has_strategy_column = "Strategy" in data.columns
    default_strategy = ticker
    
    for idx in range(len(data)):
        date = data.index[idx]
        entry_signal = data["Entry_Signal"].iloc[idx] if "Entry_Signal" in data.columns else 0
        exit_signal = data["Exit_Signal"].iloc[idx] if "Exit_Signal" in data.columns else 0
        
        # エントリーシグナルがあり、ポジションを持っていない場合
        if entry_signal == 1 and not in_position:
            in_position = True
            entry_idx = idx
            entry_date = date
            # Strategyカラムがあれば値を使用、なければデフォルト値
            strategy_name = data["Strategy"].iloc[idx] if has_strategy_column else default_strategy
        
        # イグジットシグナルがあり、ポジションを持っている場合
        if exit_signal == -1 and in_position:
            entry_exit_pairs.append((entry_idx, entry_date, idx, date, strategy_name))
            in_position = False
    
    # 最後にポジションが残っている場合、最終日にクローズ
    if in_position and entry_idx >= 0:
        final_idx = len(data) - 1
        final_date = data.index[final_idx]
        entry_exit_pairs.append((entry_idx, entry_date, final_idx, final_date, strategy_name))
    
    logger.info(f"取引ペア数: {len(entry_exit_pairs)}")
      # 各エントリー・イグジットのペアについて取引を処理
    for entry_idx, entry_date, exit_idx, exit_date, strategy_name in entry_exit_pairs:        # NaNチェックを追加して、無効なインデックスを避ける
        if pd.isna(entry_idx) or pd.isna(exit_idx):
            logger.warning(f"無効なエントリー/イグジットペア: entry_idx={entry_idx}, exit_idx={exit_idx}")
            continue
            
        try:
            # インデックスを整数に変換
            entry_idx = int(entry_idx)
            exit_idx = int(exit_idx)
            
            # ポジションサイズと部分利確の取得
            position_size = data["Position_Size"].iloc[entry_idx] if "Position_Size" in data.columns else 1
            # 部分利確があれば反映
            partial_exit = data["Partial_Exit"].iloc[exit_idx] if "Partial_Exit" in data.columns else 0

            # 価格の取得
            price_column = "Adj Close"  # デフォルト価格カラム
            entry_price = data[price_column].iloc[entry_idx]
            exit_price = data[price_column].iloc[exit_idx]
        except (TypeError, ValueError) as e:
            logger.error(f"エントリー/イグジットインデックス変換エラー: {e}, entry_idx={entry_idx}, exit_idx={exit_idx}")
            continue
              # NaNチェックを追加
        if pd.isna(entry_price) or pd.isna(exit_price):
            logger.warning(f"無効な価格: entry_price={entry_price}, exit_price={exit_price}")
            continue
            
        try:
            # NaNチェックを追加
            if pd.isna(position_size):
                position_size = 1  # デフォルト値
            if pd.isna(partial_exit):
                partial_exit = 0  # デフォルト値
                
            # 損益計算（部分利確・ポジションサイズ考慮）
            # 部分利確が0なら全量、0.3なら70%分の損益
            effective_position = float(position_size) * (1.0 - float(partial_exit))
            profit = float(exit_price - entry_price) * effective_position

            # エントリー価格が0または無効な場合のチェック
            if entry_price <= 0 or pd.isna(entry_price):
                logger.warning(f"無効なエントリー価格: entry_price={entry_price}, idx={entry_idx}")
                # 安全なデフォルト値を設定
                entry_price = exit_price if exit_price > 0 else 1.0

            # 取引量と手数料
            trade_amount = 100000 * effective_position  # 10万円 × 実効ポジションサイズ
            fee = trade_amount * 0.001  # 0.1%手数料
            profit_after_fee = (profit / float(entry_price)) * trade_amount - fee
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"損益計算エラー: {e}, entry_price={entry_price}, exit_price={exit_price}, position_size={position_size}, partial_exit={partial_exit}")
            continue

        # リスク管理の状態をJSON文字列に変換して保存
        risk_state_str = str({})  # 簡略化のため空の辞書を使用

        # 取引履歴に追加
        trade_history.loc[len(trade_history)] = [
            exit_date, 
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
        
        # パフォーマンス指標を計算
        total_trades = calculate_total_trades(trade_history)
        win_rate = calculate_win_rate(trade_history)
        total_profit = calculate_total_profit(trade_history)
        average_profit = calculate_average_profit(trade_history)
        max_profit = calculate_max_profit(trade_history)
        max_loss = calculate_max_loss(trade_history)
        max_drawdown = calculate_max_drawdown(cumulative_pnl)
        risk_return_ratio = calculate_risk_return_ratio(total_profit, max_drawdown)

        # パフォーマンス指標をデータフレームに追加
        performance_metrics = pd.DataFrame({
            "指標": [
                "総取引数", "勝率", "損益合計", "平均損益", "最大利益", "最大損失",
                "最大ドローダウン(%)", "リスクリターン比率"
            ],
            "値": [
                total_trades,
                f"{win_rate:.2f}%",
                f"{total_profit:.2f}円",
                f"{average_profit:.2f}円",
                f"{max_profit:.2f}円",
                f"{max_loss:.2f}円",
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
