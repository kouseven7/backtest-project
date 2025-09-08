#!/usr/bin/env python3
"""
切替分析シート問題調査ツール
"""

import pandas as pd
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_switch_analysis_sheet():
    """切替分析シートの問題を調査"""
    excel_file = "backtest_results/dssms_results/dssms_unified_backtest_20250908_160009.xlsx"
    
    try:
        with pd.ExcelFile(excel_file) as excel_data:
            logger.info(f"利用可能シート: {excel_data.sheet_names}")
            
            if '切替分析' in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name='切替分析')
                logger.info(f"切替分析シート: {len(df)}行")
                logger.info(f"列名: {list(df.columns)}")
                
                # 最初の10行を表示
                logger.info("最初の10行:")
                print(df.head(10))
                
                # 切替後パフォーマンスと成功判定の関係を確認
                if '切替後パフォーマンス' in df.columns and '成功判定' in df.columns:
                    performance_col = '切替後パフォーマンス'
                    success_col = '成功判定'
                    
                    logger.info(f"\n=== 切替後パフォーマンスと成功判定の分析 ===")
                    
                    # パフォーマンスの統計
                    perf_values = df[performance_col].dropna()
                    logger.info(f"パフォーマンス統計:")
                    logger.info(f"  平均: {perf_values.mean():.2f}%")
                    logger.info(f"  最小値: {perf_values.min():.2f}%")
                    logger.info(f"  最大値: {perf_values.max():.2f}%")
                    
                    # 成功判定の分布
                    success_counts = df[success_col].value_counts()
                    logger.info(f"成功判定の分布:")
                    for value, count in success_counts.items():
                        logger.info(f"  {value}: {count}件")
                    
                    # 問題のケースを特定
                    logger.info(f"\n=== 問題ケースの特定 ===")
                    positive_perf = df[df[performance_col] > 0]
                    if len(positive_perf) > 0:
                        logger.info(f"プラスパフォーマンス件数: {len(positive_perf)}")
                        positive_but_failed = positive_perf[positive_perf[success_col] == '失敗']
                        logger.info(f"プラスなのに失敗判定: {len(positive_but_failed)}件")
                        
                        if len(positive_but_failed) > 0:
                            logger.info("プラスなのに失敗と判定されているケース:")
                            print(positive_but_failed[[performance_col, success_col]].head())
                    
                    # 13%前後のケースを確認
                    around_13_perf = df[(df[performance_col] >= 10) & (df[performance_col] <= 16)]
                    if len(around_13_perf) > 0:
                        logger.info(f"\n13%前後のパフォーマンス: {len(around_13_perf)}件")
                        print(around_13_perf[[performance_col, success_col]])
                    
                else:
                    logger.warning("切替後パフォーマンスまたは成功判定列が見つかりません")
            else:
                logger.warning("切替分析シートが見つかりません")
                
    except Exception as e:
        logger.error(f"エラー: {e}")

if __name__ == "__main__":
    analyze_switch_analysis_sheet()
