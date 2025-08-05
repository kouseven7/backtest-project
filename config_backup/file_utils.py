"""
Module: File Utilities
File: file_utils.py
Description: 
  Excelファイルのパス解決を行うためのユーティリティモジュールです。
  マクロ有効ファイル（.xlsm）と通常のExcelファイル（.xlsx）の両方をサポートします。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - os
"""

# config/file_utils.py
#Excelのマクロ有効ファイルも普通のExcelファイルも認識するためのモジュール
import os

def resolve_excel_file(path: str) -> str:
    """
    指定されたExcel設定ファイルのパスについて、ファイルが存在しない場合、
    拡張子を変えて存在するか確認します。
    
    例:
      - path が "backtest_config.xlsx" で存在しなければ、
        "backtest_config.xlsm" を確認し、存在すればそのパスを返す。
      - 逆の場合も同様。
    """
    if os.path.exists(path):
        return path
    else:
        alternative = None
        if path.lower().endswith(".xlsx"):
            alternative = path[:-5] + ".xlsm"
        elif path.lower().endswith(".xlsm"):
            alternative = path[:-5] + ".xlsx"
        if alternative and os.path.exists(alternative):
            return alternative
    # どちらも存在しなければ元のパスを返す
    return path
