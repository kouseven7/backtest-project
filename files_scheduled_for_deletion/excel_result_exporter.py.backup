"""
バックテストやシミュレーション結果をExcelに出力するためのモジュール
"""
import os
import pandas as pd
from typing import Dict, List, Any, Union, Optional
import logging
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\output.log")

def save_backtest_results(backtest_results: Dict[str, Any], 
                          output_path: str, 
                          filename: Optional[str] = None) -> str:
    """
    バックテスト結果をExcelファイルに保存します。
    
    Parameters:
        backtest_results (Dict[str, Any]): バックテストの結果データ
        output_path (str): 出力先ディレクトリのパス
        filename (str, optional): 出力ファイル名（拡張子なし）
    
    Returns:
        str: 保存したファイルのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # ファイル名が指定されていない場合はデフォルト名を使用
    if filename is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_result_{timestamp}"
    
    # 拡張子がない場合は.xlsxを追加
    if not filename.endswith(".xlsx"):
        filename = f"{filename}.xlsx"
    
    # 完全なファイルパスを構築
    filepath = os.path.join(output_path, filename)
    
    try:
        with pd.ExcelWriter(filepath) as writer:
            # 取引結果をシートに保存
            if "取引履歴" in backtest_results and isinstance(backtest_results["取引履歴"], pd.DataFrame):
                backtest_results["取引履歴"].to_excel(writer, sheet_name="取引履歴", index=False)
            
            # パフォーマンス指標をシートに保存
            if "パフォーマンス指標" in backtest_results and isinstance(backtest_results["パフォーマンス指標"], dict):
                # 辞書をDataFrameに変換
                metrics_df = pd.DataFrame.from_dict(backtest_results["パフォーマンス指標"], 
                                                  orient="index", 
                                                  columns=["値"])
                metrics_df.to_excel(writer, sheet_name="パフォーマンス指標")
            
            # 損益推移をシートに保存
            if "損益推移" in backtest_results and isinstance(backtest_results["損益推移"], pd.DataFrame):
                backtest_results["損益推移"].to_excel(writer, sheet_name="損益推移")
            
            # ポジション履歴をシートに保存
            if "ポジション履歴" in backtest_results and isinstance(backtest_results["ポジション履歴"], pd.DataFrame):
                backtest_results["ポジション履歴"].to_excel(writer, sheet_name="ポジション履歴")
            
            # 月次パフォーマンスをシートに保存
            if "月次パフォーマンス" in backtest_results and isinstance(backtest_results["月次パフォーマンス"], pd.DataFrame):
                backtest_results["月次パフォーマンス"].to_excel(writer, sheet_name="月次パフォーマンス")
            
            # 年次パフォーマンスをシートに保存
            if "年次パフォーマンス" in backtest_results and isinstance(backtest_results["年次パフォーマンス"], pd.DataFrame):
                backtest_results["年次パフォーマンス"].to_excel(writer, sheet_name="年次パフォーマンス")
            
            # その他のデータフレームも保存
            for key, value in backtest_results.items():
                if isinstance(value, pd.DataFrame) and key not in ["取引履歴", "損益推移", "ポジション履歴", "月次パフォーマンス", "年次パフォーマンス"]:
                    try:
                        value.to_excel(writer, sheet_name=key)
                    except Exception as e:
                        logger.warning(f"'{key}'シートの保存中にエラーが発生しました: {e}")
        
        logger.info(f"バックテスト結果を保存しました: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"バックテスト結果の保存中にエラーが発生しました: {e}")
        return ""

def save_splits_to_excel(splits: List[tuple], 
                         output_path: str, 
                         filename: str = "train_test_splits.xlsx") -> str:
    """
    ウォークフォワード分析のためのトレーニングとテストのデータ分割情報をExcelファイルに保存します。
    
    Parameters:
        splits (List[tuple]): 分割情報のリスト[(train_indices, test_indices), ...]
        output_path (str): 出力先ディレクトリのパス
        filename (str): 出力ファイル名
    
    Returns:
        str: 保存したファイルのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # 拡張子がない場合は.xlsxを追加
    if not filename.endswith(".xlsx"):
        filename = f"{filename}.xlsx"
    
    # 完全なファイルパスを構築
    filepath = os.path.join(output_path, filename)
    
    try:
        with pd.ExcelWriter(filepath) as writer:
            for i, (train_indices, test_indices) in enumerate(splits):
                # トレーニングインデックスをDataFrameに変換
                train_df = pd.DataFrame({"トレーニングインデックス": train_indices})
                test_df = pd.DataFrame({"テストインデックス": test_indices})
                
                # 横に結合
                combined_df = pd.concat([train_df, test_df], axis=1)
                
                # シートに保存
                combined_df.to_excel(writer, sheet_name=f"Window_{i+1}", index=False)
        
        logger.info(f"分割情報を保存しました: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"分割情報の保存中にエラーが発生しました: {e}")
        return ""

def save_optimization_results(results: pd.DataFrame,
                             output_path: str,
                             filename: Optional[str] = None) -> str:
    """
    最適化の結果をExcelファイルに保存します。
    
    Parameters:
        results (pd.DataFrame): 最適化結果のDataFrame
        output_path (str): 出力先ディレクトリのパス
        filename (str, optional): 出力ファイル名（拡張子なし）
    
    Returns:
        str: 保存したファイルのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # ファイル名が指定されていない場合はデフォルト名を使用
    if filename is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}"
    
    # 拡張子がない場合は.xlsxを追加
    if not filename.endswith(".xlsx"):
        filename = f"{filename}.xlsx"
    
    # 完全なファイルパスを構築
    filepath = os.path.join(output_path, filename)
    
    try:
        # 結果をソート（スコアの降順）
        if "score" in results.columns:
            results = results.sort_values("score", ascending=False)
        
        # Excelに保存
        results.to_excel(filepath, index=False, sheet_name="最適化結果")
        
        logger.info(f"最適化結果を保存しました: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"最適化結果の保存中にエラーが発生しました: {e}")
        return ""
