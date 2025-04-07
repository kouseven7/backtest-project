#main.py

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from config.risk_management import RiskManagement
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\backtest.log")

# リスク管理の初期化
risk_manager = RiskManagement(total_assets=1000000)  # 総資産100万円

def get_parameters_and_data():
    """
    Excel設定ファイルからパラメータ取得と市場データ取得（キャッシュ利用）を行います。
    Returns:
        ticker (str), start_date (str), end_date (str), stock_data (pd.DataFrame), index_data (pd.DataFrame)
    """
    from config.error_handling import read_excel_parameters, fetch_stock_data
    from config.cache_manager import get_cache_filepath, save_cache
    import os
    
    # 設定ファイルからパラメータを取得 - .xlsx と .xlsm の両方を試す
    config_base_path = r"C:\Users\imega\Documents\my_backtest_project\config"
    config_file_xlsx = os.path.join(config_base_path, "backtest_config.xlsx")
    config_file_xlsm = os.path.join(config_base_path, "backtest_config.xlsm")
    config_csv = os.path.join(config_base_path, "config.csv")
    
    # 最初に Excel ファイルを試す
    try:
        if os.path.exists(config_file_xlsx):
            config_df = read_excel_parameters(config_file_xlsx, "銘柄設定")
            logger.info(f"設定ファイル読み込み: {config_file_xlsx}")
        elif os.path.exists(config_file_xlsm):
            config_df = read_excel_parameters(config_file_xlsm, "銘柄設定")
            logger.info(f"設定ファイル読み込み: {config_file_xlsm}")
        else:
            # Excel ファイルが見つからない場合は CSV にフォールバック
            logger.warning(f"Excel設定ファイルが見つからないため、CSVファイルを使用します: {config_csv}")
            config_df = pd.read_csv(config_csv)
            
        # 銘柄情報の取得
        if "銘柄" in config_df.columns:
            ticker = config_df["銘柄"].iloc[0]
        elif "ticker" in config_df.columns:
            ticker = config_df["ticker"].iloc[0]
        else:
            # デフォルト値を設定
            ticker = "9101.T"  # デフォルト銘柄
            logger.warning(f"銘柄情報が見つからないため、デフォルト値を使用します: {ticker}")
            
        # 日付情報の取得
        if "開始日" in config_df.columns and "終了日" in config_df.columns:
            start_date = config_df["開始日"].iloc[0]
            end_date = config_df["終了日"].iloc[0]
            # 日付が datetime オブジェクトの場合は文字列に変換
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
        elif "start_date" in config_df.columns and "end_date" in config_df.columns:
            start_date = config_df["start_date"].iloc[0]
            end_date = config_df["end_date"].iloc[0]
        else:
            # デフォルト値を設定
            start_date = "2023-01-01"
            end_date = "2023-12-31"
            logger.warning(f"日付情報が見つからないため、デフォルト値を使用します: {start_date} ~ {end_date}")
            
        logger.info(f"パラメータ取得: {ticker}, {start_date}, {end_date}")
        
    except Exception as e:
        # すべての方法が失敗した場合はデフォルト値を使用
        logger.error(f"設定ファイルの読み込みに失敗しました: {str(e)}")
        ticker = "9101.T"  # 日本郵船をデフォルトに設定
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        logger.warning(f"デフォルト値を使用します: {ticker}, {start_date}, {end_date}")

    # データ取得
    try:
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
            
    except Exception as e:
        # データ取得に失敗した場合は、キャッシュから既存のデータを探す
        logger.error(f"データ取得に失敗しました: {str(e)}")
        cache_dir = r"C:\Users\imega\Documents\my_backtest_project\data_cache"
        cache_files = os.listdir(cache_dir)
        
        # 指定された銘柄のキャッシュを探す
        matching_files = [f for f in cache_files if f.startswith(ticker)]
        if matching_files:
            # 最新のファイルを使用
            latest_file = sorted(matching_files)[-1]
            cache_path = os.path.join(cache_dir, latest_file)
            logger.info(f"キャッシュファイルを使用します: {cache_path}")
            stock_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"銘柄 {ticker} のデータが見つかりませんでした。")

    # 市場インデックスデータの取得（日本株の場合は日経平均、米国株の場合はS&P 500）
    try:
        index_ticker = "^N225" if ticker.endswith(".T") else "^GSPC"  # 日本株なら日経平均、それ以外ならS&P 500
        index_cache_filepath = get_cache_filepath(index_ticker, start_date, end_date)
        
        # キャッシュからインデックスデータを探す
        if (os.path.exists(index_cache_filepath)):
            logger.info(f"インデックス {index_ticker} のキャッシュを使用します")
            index_data = pd.read_csv(index_cache_filepath, index_col=0, parse_dates=True)
        else:
            logger.info(f"インデックス {index_ticker} のデータを取得します")
            index_data = fetch_stock_data(index_ticker, start_date, end_date)
            save_cache(index_data, index_cache_filepath)
        
        # 'Adj Close' がない場合は 'Close' を代用
        if 'Adj Close' not in index_data.columns:
            logger.warning(f"'{index_ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を代用します。")
            index_data['Adj Close'] = index_data['Close']
        
        # カラムが MultiIndex になっている場合はフラット化
        if isinstance(index_data.columns, pd.MultiIndex):
            index_data.columns = index_data.columns.get_level_values(0)
    except Exception as e:
        logger.error(f"インデックスデータの取得に失敗しました: {str(e)}")
        # インデックスデータがなくても処理を続行するため、None を設定
        index_data = None
        logger.warning("インデックスデータなしで処理を続行します。一部の戦略が正常に動作しない可能性があります。")

    return ticker, start_date, end_date, stock_data, index_data


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


def apply_strategies(stock_data: pd.DataFrame, index_data: pd.DataFrame = None):
    """
    複数の戦略を適用し、シグナルを生成します。
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame, optional): 市場インデックスデータ
        
    Returns:
        pd.DataFrame: シグナルを追加した株価データ
    """
    # VWAPBreakoutStrategy は index_data が必要なので、なければダミーデータを作成
    if index_data is None:
        logger.warning("インデックスデータがないため、VWAPBreakoutStrategyにはダミーデータを使用します。")
        index_data = stock_data.copy()  # 最低限のデータとして、同じデータを使用
    
    # テストモード関連のコードを削除し、常に標準の戦略セットを使用
    strategies = {
        "VWAP Breakout.py": VWAPBreakoutStrategy(stock_data, index_data),
        "Momentum Investing.py": MomentumInvestingStrategy(stock_data),
        "Breakout.py": BreakoutStrategy(stock_data)
    }

    # Entry_Signal と Exit_Signal カラムを追加（初期値は0）
    if 'Entry_Signal' not in stock_data.columns:
        stock_data['Entry_Signal'] = 0
    if 'Exit_Signal' not in stock_data.columns:
        stock_data['Exit_Signal'] = 0
    if 'Strategy' not in stock_data.columns:
        stock_data['Strategy'] = ""
    if 'Position_Size' not in stock_data.columns:
        stock_data['Position_Size'] = 0
    
    # 戦略シグナル分析用の列を追加
    for strat_name in strategies.keys():
        col_name = f"Signal_{strat_name.replace('.py', '').replace(' ', '_')}"
        stock_data[col_name] = 0
    
    # シグナル統計用の辞書
    signal_stats = {strat_name: 0 for strat_name in strategies.keys()}
    
    # バックテストのために、すべての日付に対してシグナルを生成
    logger.info("バックテスト用にすべての日付に対してシグナルを生成します")
    
    # ウォームアップ期間（最初の30日間はインジケーターが安定しないので除外）
    warmup_period = 30
    
    # バックテスト用のリスク管理状態を初期化
    risk_mgr_state = {}  # {日付: {戦略名: ポジションサイズ}} の形式で保存
    current_positions = {}  # 現在保有中のポジション {戦略名: [エントリー日, エントリー価格]}
    
    # シグナル分析用のログファイル
    signal_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'signal_analysis.log')
    signal_logger = setup_logger('signal_analysis', log_file=signal_log_path, level=logging.INFO)
    signal_logger.info(f"日付,{'、'.join(strategies.keys())},選択された戦略")
    
    for i in range(warmup_period, len(stock_data)):
        current_date = stock_data.index[i]
        
        # 前日のリスク管理状態をコピー
        if i > warmup_period:
            prev_date = stock_data.index[i-1]
            if prev_date in risk_mgr_state:
                # 前日の状態を今日の初期状態として設定
                risk_mgr_state[current_date] = risk_mgr_state[prev_date].copy()
            else:
                risk_mgr_state[current_date] = {}
        else:
            risk_mgr_state[current_date] = {}
        
        # 現在日のリスク管理状態を取得
        current_risk_state = risk_mgr_state[current_date]
        
        # 現在のポジション数をリスク管理システムに反映
        risk_manager.active_trades = current_risk_state
        
        # その日付までのデータで各戦略のシグナルを生成
        signals = {}
        
        for strategy_name, strategy in strategies.items():
            # 適切なインデックスで各戦略のシグナルを生成
            entry_signal = strategy.generate_entry_signal(idx=i)
            signals[strategy_name] = entry_signal
            
            # シグナルをデータフレームに記録
            col_name = f"Signal_{strategy_name.replace('.py', '').replace(' ', '_')}"
            stock_data.at[current_date, col_name] = entry_signal
            
            # シグナルが1の場合、統計を更新
            if entry_signal == 1:
                signal_stats[strategy_name] += 1
        
        # 戦略の優先順位設定を削除したため、シグナルの強さに基づいてソート（1のシグナルを優先）
        # 同じシグナル強度の場合は、ランダム性を与えるためにlistを使ってシャッフル
        prioritized_strategies = sorted(signals.keys(), key=lambda x: (-signals[x], hash(x) % 100))
        
        # 信号分析用ログ出力
        signal_values = [str(signals[s]) for s in strategies.keys()]
        selected_strategy = "なし"
        
        # 各戦略のシグナルに基づいてポジション管理
        for strategy_name in prioritized_strategies:
            entry_signal = signals[strategy_name]
            
            # エントリーシグナルが発生した場合
            if entry_signal == 1:
                # リスク管理: ポジションサイズを確認
                if risk_manager.check_position_size(strategy_name):
                    stock_data.at[current_date, 'Entry_Signal'] = 1
                    stock_data.at[current_date, 'Strategy'] = strategy_name
                    selected_strategy = strategy_name
                    
                    # ポジションサイズを1に設定（1単元の取引）
                    position_size = 1
                    stock_data.at[current_date, 'Position_Size'] = position_size
                    
                    # リスク管理システムの状態を更新
                    risk_manager.update_position(strategy_name, position_size)
                    current_risk_state[strategy_name] = current_risk_state.get(strategy_name, 0) + position_size
                    
                    # 現在のポジション情報を記録
                    entry_price = stock_data.at[current_date, 'Close'] if 'Close' in stock_data.columns else stock_data.at[current_date, 'Adj Close']
                    current_positions[strategy_name] = [current_date, entry_price]
                    
                    logger.debug(f"{current_date}: {strategy_name} からのエントリーシグナル - ポジションサイズ: {position_size}")
                    
                    # 選択理由のログ
                    competing_signals = [s for s, v in signals.items() if v == 1]
                    if len(competing_signals) > 1:
                        logger.info(f"{current_date}: 複数の戦略が同時にシグナルを出しました: {competing_signals}, 選択された戦略: {strategy_name}")
                    break  # 1つの戦略が選ばれたら終了
                else:
                    logger.debug(f"{current_date}: {strategy_name} のシグナルは検出されましたが、ポジションサイズの制限によりスキップされました")
        
        # シグナル分析ログに記録
        signal_logger.info(f"{current_date},{','.join(signal_values)},{selected_strategy}")
        
        # Exit シグナルの生成と処理
        for strategy_name, position_info in list(current_positions.items()):
            entry_date, entry_price = position_info
            holding_days = (current_date - entry_date).days
            
            # 1. 保有期間による決済（3日間保持後）
            if holding_days >= 3:
                stock_data.at[current_date, 'Exit_Signal'] = -1
                # ポジションクローズ
                if strategy_name in current_risk_state:
                    del current_risk_state[strategy_name]
                del current_positions[strategy_name]
                logger.debug(f"{current_date}: {strategy_name} の保有期間（3日）経過によるイグジットシグナル")
                continue
            
            # 2. 損切りロジック - 2%以上の下落
            current_price = stock_data.at[current_date, 'Close'] if 'Close' in stock_data.columns else stock_data.at[current_date, 'Adj Close']
            price_change = (current_price - entry_price) / entry_price
            
            if price_change < -0.02:
                stock_data.at[current_date, 'Exit_Signal'] = -1
                # ポジションクローズ
                if strategy_name in current_risk_state:
                    del current_risk_state[strategy_name]
                del current_positions[strategy_name]
                logger.debug(f"{current_date}: {strategy_name} の2%以上の下落によるイグジットシグナル（損切り）")
                continue
    
    # ログ出力
    entry_count = stock_data['Entry_Signal'].sum()
    exit_count = (stock_data['Exit_Signal'] == -1).sum()
    logger.info(f"バックテスト期間中の総エントリー回数: {entry_count}回")
    logger.info(f"バックテスト期間中の総イグジット回数: {exit_count}回")
    
    # 戦略ごとのシグナル統計
    logger.info("各戦略のシグナル生成回数:")
    for strategy_name, count in signal_stats.items():
        logger.info(f"  {strategy_name}: {count}回")
    
    # 実際に選択された戦略の分布
    strategy_dist = stock_data[stock_data['Entry_Signal'] == 1]['Strategy'].value_counts()
    logger.info("実際に選択された戦略の分布:")
    for strategy_name, count in strategy_dist.items():
        logger.info(f"  {strategy_name}: {count}回 ({count/entry_count*100:.1f}%)")
    
    # 戦略間の競合状況分析
    logger.info("戦略間の競合状況分析:")
    conflict_days = 0
    for i in range(warmup_period, len(stock_data)):
        current_date = stock_data.index[i]
        competing_signals = sum(1 for col in stock_data.columns if col.startswith('Signal_') and stock_data.at[current_date, col] == 1)
        if competing_signals > 1:
            conflict_days += 1
    
    if entry_count > 0:
        logger.info(f"  複数戦略が同時にシグナルを出した日数: {conflict_days}日")
        logger.info(f"  選択された戦略が他の戦略より優先された比率: {conflict_days/entry_count*100:.1f}%")
    
    # リスク管理情報をメタデータとして追加
    stock_data.attrs['risk_management'] = risk_mgr_state
    stock_data.attrs['signal_stats'] = signal_stats
    
    return stock_data


def simulate_and_save(result_data: pd.DataFrame, ticker: str):
    """
    バックテストシミュレーションを実行し、結果をExcelに出力します。
    """
    import trade_simulation as trade_simulation
    # simulate_trades 関数には result_data と ticker を渡して、取引履歴に銘柄情報を追加します。
    trade_results = trade_simulation.simulate_trades(result_data, ticker)
    logger.info("バックテスト（トレードシミュレーション）完了")
    
    output_dir = r"C:\Users\imega\Documents\my_backtest_project\backtest_results"
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリを作成しました: {output_dir}")
    
    # 実行日時をファイル名に含める
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"backtest_results_{now}.xlsx")
    
    try:
        from output.excel_result_exporter import ensure_workbook_exists, add_pnl_chart, create_pivot_from_trade_history, save_backtest_results
        
        # 新しいExcelファイルを作成
        ensure_workbook_exists(output_file)
        
        # 結果を保存
        save_backtest_results(trade_results, output_file)
        logger.info(f"バックテスト結果をExcelに出力完了: {output_file}")
        
        # チャートとピボットテーブルを追加
        add_pnl_chart(output_file, sheet_name="損益推移", chart_title="累積損益推移")
        create_pivot_from_trade_history(output_file, trade_sheet="取引履歴", pivot_sheet="Pivot_取引履歴")
        logger.info("Excelのチャート、ピボットテーブル追加完了")
        
    except PermissionError as e:
        logger.warning(f"Excelファイルにアクセスできません: {e}")
        # アクセスできない場合はCSVに出力
        csv_dir = os.path.join(output_dir, "csv_backup")
        os.makedirs(csv_dir, exist_ok=True)
        
        for sheet_name, df in trade_results.items():
            if isinstance(df, pd.DataFrame):
                csv_file = os.path.join(csv_dir, f"backtest_{sheet_name}_{now}.csv")
                df.to_csv(csv_file)
                logger.info(f"代替として結果をCSVに出力しました: {csv_file}")
        
        # 元のExcelファイルを別名で保存してみる
        alt_output_file = os.path.join(output_dir, f"backtest_results_alt_{now}.xlsx")
        try:
            ensure_workbook_exists(alt_output_file)
            save_backtest_results(trade_results, alt_output_file)
            logger.info(f"代替Excelファイルに結果を出力しました: {alt_output_file}")
            output_file = alt_output_file  # 成功した場合は新しいファイル名を使用
        except Exception as ex:
            logger.error(f"代替Excelファイルの作成にも失敗しました: {ex}")
            # CSVファイルだけで十分とする
    
    return trade_results


def main():
    try:
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        stock_data = apply_strategies(stock_data, index_data)
        
        # バックテスト結果をExcelに出力
        backtest_results = simulate_and_save(stock_data, ticker)
        
        # シグナル分析データも出力
        signal_columns = [col for col in stock_data.columns if col.startswith('Signal_')]
        if signal_columns:
            signal_analysis_df = stock_data[['Entry_Signal', 'Strategy'] + signal_columns]
            signal_analysis_df = signal_analysis_df[signal_analysis_df.index >= stock_data.index[30]]  # ウォームアップ期間を除外
            
            # シグナル分析ファイルの出力
            output_dir = r"C:\Users\imega\Documents\my_backtest_project\analysis_results"
            os.makedirs(output_dir, exist_ok=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            signal_file = os.path.join(output_dir, f"signal_analysis_{ticker}_{now}.csv")
            signal_analysis_df.to_csv(signal_file)
            logger.info(f"シグナル分析データをCSVに出力しました: {signal_file}")
        
        logger.info(f"バックテスト結果をExcelに出力しました: {backtest_results}")
        
        logger.info("全体のバックテスト処理が正常に完了しました。")
        
    except Exception as e:
        logger.exception("バックテスト実行中にエラーが発生しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()