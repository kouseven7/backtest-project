"""
Module: excel_result_exporter
Description:
    バックテスト結果をExcelファイルに出力するための関数群を提供します。
    - Workbookの存在確認および新規作成
    - 複数シートへのDataFrame出力（タイムゾーン付き日時の処理含む）
    - 損益推移グラフの追加
    - 取引履歴からピボットテーブルの作成
"""

import os
import logging
import pandas as pd
import openpyxl
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import Dict

from config.logger_config import setup_logger

logger = setup_logger(__name__)

def ensure_workbook_exists(workbook_path: str) -> None:
    """
    指定したパスにExcelファイルが存在しない場合、必要なシートとヘッダーを含む新規Workbookを作成して保存します。
    シート名は「損益推移」と「取引履歴」に統一されます。
    
    Parameters:
        workbook_path (str): 作成するExcelファイルのパス
    """
    # 出力先ディレクトリの存在確認と作成
    output_dir = os.path.dirname(workbook_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリを作成しました: {output_dir}")
    
    if not os.path.exists(workbook_path):
        logger.info(f"Workbookが存在しません。新規に作成します: {workbook_path}")
        wb = Workbook()
        # 最初のシートを「損益推移」として設定
        ws_pnl = wb.active
        ws_pnl.title = "損益推移"
        ws_pnl.append(["日付", "累積損益"])
        # 新たに「取引履歴」シートを作成
        ws_trade = wb.create_sheet(title="取引履歴")
        ws_trade.append(["日付", "銘柄", "エントリー", "イグジット", "取引結果"])
        wb.save(workbook_path)
        logger.info("新規Workbookを作成しました。")
    else:
        logger.info("Workbookは既に存在します。")


def _remove_timezone_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame内の各セルに対して、もしタイムゾーン付きdatetimeであればタイムゾーン情報を除去します.
    
    Returns:
        pd.DataFrame: タイムゾーン情報を除去したDataFrame
    """
    def remove_tz(x):
        try:
            # datetime型でtzinfoが存在する場合、tz_localize(None)を適用
            if hasattr(x, 'tzinfo') and x.tzinfo is not None:
                return x.tz_localize(None)
        except Exception:
            pass
        return x
    return df.applymap(remove_tz)

def save_backtest_results(results: Dict[str, pd.DataFrame], output_file: str) -> None:
    """
    バックテスト結果をExcelファイルに保存する関数.
    
    Parameters:
        results (Dict[str, pd.DataFrame]): キーがシート名、値が出力すべきDataFrameの辞書.
            例:
                {
                    "取引履歴": trade_history_df,
                    "損益推移": pnl_df,
                    "最大ドローダウン": drawdown_df,
                    "全体結果": summary_df
                }
        output_file (str): 出力先のExcelファイルの絶対パス.
    """
    try:
        # 出力先ディレクトリの存在確認と作成
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"出力ディレクトリを作成しました: {output_dir}")
        
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for sheet_name, df in results.items():
                # タイムゾーン付きdatetimeの情報を解除
                df = _remove_timezone_from_df(df.copy())
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"シート '{sheet_name}' の出力に成功しました。")
        logger.info(f"バックテスト結果をExcelファイルに保存しました: {output_file}")
    except Exception as e:
        logger.exception("バックテスト結果のExcelファイル保存中にエラーが発生しました。")
        raise

def add_pnl_chart(workbook_path: str, sheet_name: str = "損益推移", chart_title: str = "累積損益推移") -> None:
    """
    指定したExcelファイルの指定シートに、累積損益の折れ線グラフを追加します.
    シート内は、A列に「日付」、B列に「累積損益」が存在する前提です.
    
    Parameters:
        workbook_path (str): Excelファイルのパス
        sheet_name (str): グラフを追加するシート名（デフォルトは「損益推移」）
        chart_title (str): グラフのタイトル
    """
    try:
        wb = openpyxl.load_workbook(workbook_path)
        ws = wb[sheet_name]
        max_row = ws.max_row
        
        chart = LineChart()
        chart.title = chart_title
        chart.style = 10
        chart.y_axis.title = '累積損益'
        chart.x_axis.title = '日付'
        
        data = Reference(ws, min_col=2, min_row=1, max_row=max_row)
        cats = Reference(ws, min_col=1, min_row=2, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        
        ws.add_chart(chart, "D2")
        wb.save(workbook_path)
        logger.info(f"チャート '{chart_title}' をシート '{sheet_name}' に追加し、保存しました。")
    except Exception as e:
        logger.exception("チャートの追加中にエラーが発生しました。")
        raise

def create_pivot_from_trade_history(
    workbook_path: str, 
    trade_sheet: str = "取引履歴", 
    pivot_sheet: str = "Pivot_取引履歴"
) -> None:
    """
    指定したExcelファイルの取引履歴シートからデータを読み込み、
    銘柄ごとの取引数と取引結果合計を算出したピボットテーブルを新規シートに出力します.
    
    Parameters:
        workbook_path (str): Excelファイルのパス
        trade_sheet (str): 取引履歴が格納されているシート名（デフォルトは「取引履歴」）
        pivot_sheet (str): ピボットテーブルを出力するシート名（デフォルトは「Pivot_取引履歴」）
    """
    try:
        df = pd.read_excel(workbook_path, sheet_name=trade_sheet)
        pivot = df.pivot_table(index="銘柄", aggfunc={"取引結果": ["count", "sum"]})
        pivot.columns = ["取引数", "取引結果合計"]
        pivot.reset_index(inplace=True)
        
        wb = openpyxl.load_workbook(workbook_path)
        if pivot_sheet in wb.sheetnames:
            ws = wb[pivot_sheet]
            for row in ws.iter_rows():
                for cell in row:
                    cell.value = None
        else:
            ws = wb.create_sheet(title=pivot_sheet)
        
        for r_idx, row in enumerate(dataframe_to_rows(pivot, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx, column=c_idx, value=value)
        
        wb.save(workbook_path)
        logger.info(f"ピボットテーブルをシート '{pivot_sheet}' に作成し、保存しました。")
    except Exception as e:
        logger.exception("ピボットテーブルの作成中にエラーが発生しました。")
        raise
