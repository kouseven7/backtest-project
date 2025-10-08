#!/usr/bin/env python3
"""
保有期間修正後の検証ツール
"""

import pandas as pd
import numpy as np
import os
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_fixed_excel(excel_file_path: str):
    """修正後のExcelファイルを検証"""
    try:
        logger.info(f"検証対象ファイル: {excel_file_path}")
        
        with pd.ExcelFile(excel_file_path) as excel_data:
            sheet_names = excel_data.sheet_names
            logger.info(f"利用可能シート: {sheet_names}")
            
            # 取引履歴シートを検証
            if '取引履歴' in sheet_names:
                df = pd.read_excel(excel_data, sheet_name='取引履歴')
                logger.info(f"取引履歴シート: {len(df)}行")
                
                if '保有期間' in df.columns:
                    periods = df['保有期間'].values
                    # 文字列から数値を抽出
                    numeric_periods = []
                    for period in periods:
                        if pd.isna(period):
                            numeric_periods.append(24.0)
                        elif isinstance(period, str):
                            try:
                                numeric_value = float(period.replace('時間', '').replace('h', '').strip())
                                numeric_periods.append(numeric_value)
                            except:
                                numeric_periods.append(24.0)
                        else:
                            numeric_periods.append(float(period))
                    
                    periods_clean = np.array(numeric_periods)
                    
                    logger.info("=== 保有期間統計 ===")
                    logger.info(f"総データ数: {len(periods)}")
                    logger.info(f"有効データ数: {len(periods_clean)}")
                    logger.info(f"平均: {np.mean(periods_clean):.2f}時間")
                    logger.info(f"標準偏差: {np.std(periods_clean):.2f}時間")
                    logger.info(f"最小値: {np.min(periods_clean):.2f}時間")
                    logger.info(f"最大値: {np.max(periods_clean):.2f}時間")
                    logger.info(f"24.0時間固定の数: {np.sum(periods_clean == 24.0)}/{len(periods_clean)}")
                    
                    # 分布を確認
                    unique_values = np.unique(periods_clean)
                    if len(unique_values) <= 5:
                        logger.info(f"ユニーク値: {unique_values}")
                    else:
                        logger.info(f"ユニーク値数: {len(unique_values)}")
                        logger.info(f"最初の5値: {unique_values[:5]}")
                    
                    # 修正が成功したかチェック
                    if np.std(periods_clean) > 0.1:
                        logger.info("[OK] 修正成功: 保有期間が多様化されています")
                    else:
                        logger.warning("[ERROR] 修正不十分: 保有期間がまだ固定されている可能性があります")
                        
                else:
                    logger.warning("保有期間（時間）列が見つかりません")
            
            # 戦略別統計シートを確認
            if '戦略別統計' in sheet_names:
                df_strategy = pd.read_excel(excel_data, sheet_name='戦略別統計')
                logger.info(f"\n=== 戦略別統計 ===")
                logger.info(f"戦略数: {len(df_strategy)}")
                
                if '戦略名' in df_strategy.columns:
                    strategies = df_strategy['戦略名'].tolist()
                    logger.info(f"戦略一覧: {strategies}")
                    
                    # 勝率を確認
                    if '勝率(%)' in df_strategy.columns:
                        win_rates = df_strategy['勝率(%)'].values
                        logger.info(f"勝率統計: 平均{np.mean(win_rates):.1f}%, 範囲{np.min(win_rates):.1f}%-{np.max(win_rates):.1f}%")
                    
                    if len(strategies) == 7:
                        logger.info("[OK] 戦略統計修正成功: 7つの個別戦略が表示されています")
                    else:
                        logger.warning(f"[ERROR] 戦略統計問題: {len(strategies)}戦略のみ表示")
                else:
                    logger.warning("戦略名列が見つかりません")
            
    except Exception as e:
        logger.error(f"検証中にエラー: {e}")

def main():
    """メイン処理"""
    logger.info("=== 保有期間修正後検証開始 ===")
    
    # 最新の修正済みExcelファイルパス
    excel_file = "backtest_results/dssms_results/dssms_unified_backtest_20250908_160009.xlsx"
    
    if not os.path.exists(excel_file):
        logger.error(f"ファイルが存在しません: {excel_file}")
        return
    
    verify_fixed_excel(excel_file)
    
    logger.info("=== 検証完了 ===")

if __name__ == "__main__":
    main()
