"""
Module: Strategy Parameter Manager
File: strategy_parameter_manager.py
Description: 
  Excelファイルから戦略のパラメータを一元管理するためのモジュールです。
  各戦略シートからパラメータを抽出し、辞書形式で管理します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - config.logger_config
  - config.file_utils
"""

# ファイル: strategy_modules/strategy_parameter_manager.py
import pandas as pd
from config.logger_config import setup_logger
from config.file_utils import resolve_excel_file

# Excel設定ファイルのパスを、解決関数で取得
config_file = resolve_excel_file(r"C:\Users\imega\Documents\my_backtest_project\config\backtest_config.xlsx")


class StrategyParameterManager:
    """
    Excelファイルから各戦略のパラメータを一元管理するクラス。
    各戦略シートから、ヘッダー行をパラメータ名、2行目を値として取得し、
    全戦略のパラメータを内部辞書として保持します。
    """
    def __init__(self, excel_file, strategy_suffix="戦略"):
        """
        初期化時にExcelファイルと戦略シートのサフィックスを指定し、
        全戦略のパラメータをロードします。
        """
        self.excel_file = excel_file
        self.strategy_suffix = strategy_suffix
        self.strategy_params = self.load_all_strategy_params()

    def load_strategy_params(self, sheet_name):
        """
        指定されたシートから、ヘッダー行とその次のデータ行を基にパラメータを
        辞書形式で取得します。
        """
        df = pd.read_excel(self.excel_file, sheet_name=sheet_name, header=0)
        if df.empty:
            raise ValueError(f"{sheet_name}シートが空です。")
        params = df.iloc[0].to_dict()
        return params

    def load_all_strategy_params(self):
        """
        Excelファイル内の全シートをチェックし、シート名が指定されたサフィックスで終わる
        戦略シートからパラメータを抽出して、一元管理用の辞書として返します。
        """
        xls = pd.ExcelFile(self.excel_file)
        strategy_params = {}
        for sheet in xls.sheet_names:
            if sheet.endswith(self.strategy_suffix):
                params = self.load_strategy_params(sheet)
                strategy_params[sheet] = params
        return strategy_params

    def get_params(self, strategy_name=None):
        """
        戦略名が指定されればその戦略のパラメータ辞書を、指定がなければ全戦略のパラメータ辞書を返します。
        """
        if strategy_name:
            return self.strategy_params.get(strategy_name, None)
        return self.strategy_params
