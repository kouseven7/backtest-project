#!/usr/bin/env python3
"""
修正後の切替分析シートを検証するスクリプト
"""

import pandas as pd
import os
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_excel_file():
    """最新のDSSMS Excelファイルを特定"""
    results_dir = "backtest_results/dssms_results"
    
    if not os.path.exists(results_dir):
        logger.error(f"結果ディレクトリが見つかりません: {results_dir}")
        return None
    
    excel_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx') and 'unified_backtest' in f and not f.startswith('~$')]
    
    if not excel_files:
        logger.error("Excelファイルが見つかりません")
        return None
    
    # 最新のファイルを選択（ファイル名のタイムスタンプで判断）
    latest_file = sorted(excel_files)[-1]
    full_path = os.path.join(results_dir, latest_file)
    
    logger.info(f"最新のExcelファイル: {full_path}")
    return full_path

def analyze_switch_analysis_sheet(excel_path):
    """切替分析シートを詳細分析"""
    
    try:
        # 切替分析シートを読み込み
        df = pd.read_excel(excel_path, sheet_name='切替分析')
        
        logger.info(f"切替分析シート読み込み成功: {len(df)}行")
        
        # 列名を確認
        logger.info(f"列名: {list(df.columns)}")
        
        # 最初の5行を表示
        logger.info("最初の5行のデータ:")
        for i, row in df.head().iterrows():
            logger.info(f"Row {i+1}: {dict(row)}")
        
        # パフォーマンス列の分析
        if '切替後パフォーマンス' in df.columns:
            performance_col = df['切替後パフォーマンス']
            
            logger.info("パフォーマンス列の分析:")
            logger.info(f"  データ型: {performance_col.dtype}")
            logger.info(f"  サンプル値: {list(performance_col.head())}")
            
            # 数値変換を試行
            try:
                numeric_performance = pd.to_numeric(performance_col, errors='coerce')
                logger.info(f"  数値変換成功: {numeric_performance.notna().sum()}/{len(numeric_performance)}件")
                
                if numeric_performance.notna().sum() > 0:
                    logger.info(f"  最小値: {numeric_performance.min():.6f}")
                    logger.info(f"  最大値: {numeric_performance.max():.6f}")
                    logger.info(f"  平均値: {numeric_performance.mean():.6f}")
                    
                    # 正の値の数
                    positive_count = (numeric_performance > 0).sum()
                    logger.info(f"  正の値: {positive_count}件")
            except Exception as e:
                logger.error(f"数値変換エラー: {e}")
        
        # 成功判定列の分析
        if '成功判定' in df.columns:
            success_col = df['成功判定']
            
            logger.info("成功判定列の分析:")
            success_counts = success_col.value_counts()
            logger.info(f"  成功判定分布: {dict(success_counts)}")
            
            # パフォーマンスと成功判定の対応確認
            if '切替後パフォーマンス' in df.columns:
                logger.info("パフォーマンスと成功判定の対応確認:")
                
                # 具体例をいくつか表示
                for i in range(min(10, len(df))):
                    perf = df.iloc[i]['切替後パフォーマンス']
                    success = df.iloc[i]['成功判定']
                    
                    try:
                        perf_num = float(perf) if perf is not None else 0.0
                        expected_success = "成功" if perf_num > 0 else "失敗"
                        match = "[OK]" if success == expected_success else "[ERROR]"
                        
                        logger.info(f"  Row {i+1}: Performance={perf} ({perf_num:.6f}) -> 判定={success}, 期待={expected_success} {match}")
                    except (ValueError, TypeError):
                        logger.info(f"  Row {i+1}: Performance={perf} (変換不可) -> 判定={success}")
        
        return True
        
    except Exception as e:
        logger.error(f"切替分析シート分析エラー: {e}")
        return False

def compare_with_issue_description():
    """問題の説明と比較"""
    
    logger.info("=== 修正前後の比較 ===")
    logger.info("修正前の問題:")
    logger.info("- 13%のパフォーマンスが '失敗' と表示")
    logger.info("- パフォーマンス値が文字列として連結")
    logger.info("- 成功判定ロジックの不整合")
    
    logger.info("修正内容:")
    logger.info("- profit_loss_at_switch ベースの成功判定")
    logger.info("- 数値型の適切な処理")
    logger.info("- 型安全なExcel出力")

def main():
    """メイン実行関数"""
    
    logger.info("=== 修正後切替分析シート検証開始 ===")
    
    # 最新のExcelファイルを特定
    excel_path = find_latest_excel_file()
    
    if not excel_path:
        logger.error("Excelファイルが見つかりません")
        return
    
    # 切替分析シートを分析
    success = analyze_switch_analysis_sheet(excel_path)
    
    if success:
        logger.info("[OK] 切替分析シートの検証が完了しました")
        
        # 問題との比較
        compare_with_issue_description()
        
        logger.info(f"\\n詳細確認用: {excel_path}")
        logger.info("切替分析シートを直接開いて以下を確認してください:")
        logger.info("1. パフォーマンス値が個別の数値として表示されているか")
        logger.info("2. 正のパフォーマンス値に対して '成功' が表示されているか")
        logger.info("3. 負のパフォーマンス値に対して '失敗' が表示されているか")
    else:
        logger.error("[ERROR] 切替分析シートの検証に失敗しました")

if __name__ == "__main__":
    main()
