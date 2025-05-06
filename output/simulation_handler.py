import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def simulate_and_save(result_data: pd.DataFrame, ticker: str, splits=None):
    """
    バックテストシミュレーションを実行し、結果をExcelに出力します。
    
    Parameters:
        result_data (pd.DataFrame): シミュレーション対象のデータフレーム
        ticker (str): 銘柄コード
        splits (list, optional): ウォークフォワード分割情報
        
    Returns:
        dict: シミュレーション結果の辞書
    """
    import trade_simulation as trade_simulation
    from output.performance_calculator import add_performance_metrics
    
    trade_results = trade_simulation.simulate_trades(result_data, ticker)

    # パフォーマンス指標を追加
    trade_results = add_performance_metrics(trade_results)
    logger.info("バックテスト（トレードシミュレーション）完了")
    
    output_dir = r"C:\Users\imega\Documents\my_backtest_project\backtest_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリを作成しました: {output_dir}")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"backtest_results_{now}.xlsx")
    
    try:
        from output.excel_result_exporter import ensure_workbook_exists, add_pnl_chart, create_pivot_from_trade_history, save_backtest_results, save_splits_to_excel
        
        ensure_workbook_exists(output_file)
        
        save_backtest_results(trade_results, output_file)
        logger.info(f"バックテスト結果をExcelに出力完了: {output_file}")

        # ウォークフォワードの分割情報を同ファイルに書き込み
        if splits:
            save_splits_to_excel(splits, output_file, sheet_name="分割期間")
            logger.info("ウォークフォワードの分割期間情報を同ファイルに出力しました。")

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