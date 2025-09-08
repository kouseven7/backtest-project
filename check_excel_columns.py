#!/usr/bin/env python3
"""
Excelファイルの列名確認ツール
"""

import pandas as pd
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_excel_columns():
    """Excelファイルの列名を確認"""
    excel_file = "backtest_results/dssms_results/dssms_unified_backtest_20250908_160009.xlsx"
    
    try:
        with pd.ExcelFile(excel_file) as excel_data:
            # 取引履歴シートの列名を確認
            if '取引履歴' in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name='取引履歴')
                logger.info(f"取引履歴シートの列名: {list(df.columns)}")
                logger.info(f"データ行数: {len(df)}")
                
                # 最初の数行を表示
                logger.info("最初の5行:")
                print(df.head())
                
            # 戦略別統計シートも確認
            if '戦略別統計' in excel_data.sheet_names:
                df_strategy = pd.read_excel(excel_data, sheet_name='戦略別統計')
                logger.info(f"戦略別統計シートの列名: {list(df_strategy.columns)}")
                print("\n戦略別統計:")
                print(df_strategy)
                
    except Exception as e:
        logger.error(f"エラー: {e}")

if __name__ == "__main__":
    check_excel_columns()
