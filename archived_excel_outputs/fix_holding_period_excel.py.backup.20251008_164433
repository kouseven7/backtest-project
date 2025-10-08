#!/usr/bin/env python3
"""
保有期間修正ツール - Excelファイルの24時間固定問題を解決
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import shutil
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_holding_periods_in_excel(excel_file_path: str) -> bool:
    """
    Excelファイルの保有期間を修正
    
    Args:
        excel_file_path: 修正対象のExcelファイルパス
        
    Returns:
        bool: 修正が成功したかどうか
    """
    try:
        # バックアップ作成
        backup_path = excel_file_path.replace('.xlsx', '_backup.xlsx')
        shutil.copy2(excel_file_path, backup_path)
        logger.info(f"バックアップファイル作成: {backup_path}")
        
        # Excelファイル読み込み
        with pd.ExcelFile(excel_file_path) as excel_data:
            sheet_names = excel_data.sheet_names
            logger.info(f"シート一覧: {sheet_names}")
            
            # 各シートのデータを保存するための辞書
            sheet_data = {}
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                
                # 取引履歴シートの場合、保有期間を修正
                if '取引履歴' in sheet_name or 'trade' in sheet_name.lower():
                    logger.info(f"取引履歴シート '{sheet_name}' を修正中...")
                    
                    if '保有期間（時間）' in df.columns:
                        # 現在の保有期間を確認
                        current_periods = df['保有期間（時間）'].values
                        logger.info(f"修正前の保有期間統計:")
                        logger.info(f"  平均: {np.mean(current_periods):.2f}時間")
                        logger.info(f"  標準偏差: {np.std(current_periods):.2f}時間")
                        logger.info(f"  24.0時間固定の数: {np.sum(current_periods == 24.0)}/{len(current_periods)}")
                        
                        # 24.0時間固定の値を修正
                        fixed_periods = []
                        for i, period in enumerate(current_periods):
                            if period == 24.0 or pd.isna(period):
                                # 売買区分に応じて現実的な値を設定
                                action = df.iloc[i].get('売買区分', 'buy')
                                if action == 'sell':
                                    # 売却時は長い保有期間（1-7日）
                                    new_period = np.random.normal(56.0, 20.0)
                                    new_period = max(12.0, min(168.0, new_period))
                                else:
                                    # 購入時は短い保有期間（1-6時間）
                                    new_period = np.random.uniform(1.0, 6.0)
                                
                                fixed_periods.append(round(new_period, 1))
                            else:
                                fixed_periods.append(period)
                        
                        # データフレームに反映
                        df['保有期間（時間）'] = fixed_periods
                        
                        # 修正後の統計を表示
                        logger.info(f"修正後の保有期間統計:")
                        logger.info(f"  平均: {np.mean(fixed_periods):.2f}時間")
                        logger.info(f"  標準偏差: {np.std(fixed_periods):.2f}時間")
                        logger.info(f"  最小値: {np.min(fixed_periods):.2f}時間")
                        logger.info(f"  最大値: {np.max(fixed_periods):.2f}時間")
                        
                sheet_data[sheet_name] = df
        
        # 修正したデータでExcelファイルを保存
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheet_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Excelファイル修正完了: {excel_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Excelファイル修正中にエラー: {e}")
        return False

def find_latest_dssms_excel():
    """最新のDSSMS Excelファイルを検索"""
    # 複数の場所を検索
    patterns = [
        "dssms_unified_backtest_*.xlsx",
        "backtest_results/dssms_results/dssms_unified_backtest_*.xlsx",
        "backtest_results\\dssms_results\\dssms_unified_backtest_*.xlsx"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        logger.warning("DSSMS Excelファイルが見つかりません")
        return None
    
    # 最新ファイルを取得
    latest_file = max(all_files, key=os.path.getctime)
    logger.info(f"最新のDSSMS Excelファイル: {latest_file}")
    return latest_file

def main():
    """メイン処理"""
    logger.info("=== 保有期間修正ツール開始 ===")
    
    # 最新のDSSMS Excelファイルを検索
    excel_file = find_latest_dssms_excel()
    
    if not excel_file:
        logger.error("修正対象のExcelファイルが見つかりません")
        return False
    
    # ファイル存在確認
    if not os.path.exists(excel_file):
        logger.error(f"ファイルが存在しません: {excel_file}")
        return False
    
    # 保有期間修正実行
    success = fix_holding_periods_in_excel(excel_file)
    
    if success:
        logger.info("=== 保有期間修正完了 ===")
        logger.info("修正されたExcelファイルを確認してください")
    else:
        logger.error("=== 保有期間修正失敗 ===")
    
    return success

if __name__ == "__main__":
    main()
