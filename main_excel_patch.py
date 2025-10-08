"""
main.pyでの新Excel出力モジュール統合パッチ
File: main_excel_patch.py
Description: 
  main.pyの結果を新しいsimple_excel_exporter.pyで処理するためのパッチモジュール。
  既存のmain.pyを変更せずに新しいExcel出力モジュールを利用できます。

Author: imega
Created: 2025-07-30

Usage:
    # main.pyの最後に以下を追加:
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: from main_excel_patch import apply_new_excel_output
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: apply_new_excel_output(stock_data, ticker)
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ロガーの設定
from config.logger_config import setup_logger
logger = setup_logger(__name__)

# 新しいExcel出力モジュールをインポート
from output.simple_excel_exporter import save_backtest_results_simple

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def apply_new_excel_output(stock_data: pd.DataFrame, ticker: str,
                          output_filename: str = None) -> str:
    """
    新しいExcel出力モジュールを使用してバックテスト結果を出力する
    
    Parameters:
        stock_data (pd.DataFrame): バックテスト結果を含む株価データ
        ticker (str): 銘柄コード
        output_filename (str): 出力ファイル名（省略時は自動生成）
        
    Returns:
        str: 出力ファイルパス
    """
    try:
        logger.info(f"新Excel出力モジュールによる結果保存開始: {ticker}")
        
        # ファイル名の生成
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: output_filename = f"improved_backtest_{ticker}_{timestamp}.xlsx"
        
        # 新しいモジュールで出力
        output_path = save_backtest_results_simple(
            stock_data=stock_data,
            ticker=ticker,
            output_dir=None,  # デフォルトディレクトリ使用
            filename=output_filename
        )
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"[OK] 新Excel出力完了: {output_path}")
            logger.info(f"ファイルサイズ: {file_size:,} bytes")
            
            # 出力内容の要約
            summarize_output(output_path, ticker)
            
            return output_path
        else:
            logger.error("[ERROR] 新Excel出力に失敗しました")
            return ""
            
    except Exception as e:
        logger.error(f"新Excel出力エラー: {e}")
        import traceback
        traceback.print_exc()
        return ""

def summarize_output(output_path: str, ticker: str):
    """出力されたExcelファイルの内容を要約する"""
    try:
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"=== {ticker} Excel出力サマリー ===")
        logger.info(f"総シート数: {len(sheet_names)}")
        
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(output_path, sheet_name=sheet_name)
                
                if sheet_name == '取引履歴' and not df.empty:
                    total_trades = len(df)
                    total_profit = df['取引結果'].sum() if '取引結果' in df.columns else 0
                    logger.info(f"  [CHART] 取引履歴: {total_trades} 件, 総損益: {total_profit:,.0f}円")
                    
                elif sheet_name == 'パフォーマンス指標' and not df.empty:
                    if '指標' in df.columns and '値' in df.columns:
                        win_rate_row = df[df['指標'] == '勝率']
                        if not win_rate_row.empty:
                            win_rate = win_rate_row['値'].iloc[0]
                            logger.info(f"  [UP] 勝率: {win_rate}")
                            
                elif sheet_name == '損益推移' and not df.empty:
                    if '累積損益' in df.columns:
                        final_pnl = df['累積損益'].iloc[-1]
                        logger.info(f"  [MONEY] 最終累積損益: {final_pnl:,.0f}円")
                        
                else:
                    logger.info(f"  📄 {sheet_name}: {len(df)} 行")
                    
            except Exception as e:
                logger.warning(f"シート '{sheet_name}' の要約エラー: {e}")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.warning(f"出力要約エラー: {e}")

def patch_main_with_new_excel(main_module_path: str = "main.py"):
    """
    main.pyファイルに新Excel出力モジュールの呼び出しを追加する（実験的）
    
    Parameters:
        main_module_path (str): main.pyのパス
    """
    try:
        logger.info("main.pyへの新Excel出力パッチ適用開始")
        
        # main.pyファイルを読み込み
        if not os.path.exists(main_module_path):
            logger.error(f"main.pyが見つかりません: {main_module_path}")
            return False
        
        with open(main_module_path, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # 既にパッチが適用されているかチェック
        if "main_excel_patch" in main_content:
            logger.info("main.pyには既に新Excel出力パッチが適用されています")
            return True
        
        # パッチコードを準備
        patch_code = """
# === 新Excel出力モジュール統合パッチ ===
try:
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: from main_excel_patch import apply_new_excel_output
    logger.info("新Excel出力モジュールでも結果を保存します...")
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: new_excel_path = apply_new_excel_output(stock_data, ticker)
    if new_excel_path:
        logger.info(f"新Excel出力完了: {new_excel_path}")
    else:
        logger.warning("新Excel出力に失敗しました")
except Exception as e:
    logger.warning(f"新Excel出力パッチエラー: {e}")
# === パッチ終了 ===
"""
        
        # main.pyの最後に追加
        patched_content = main_content + patch_code
        
        # バックアップを作成
        backup_path = main_module_path + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(main_content)
        
        logger.info(f"main.pyのバックアップを作成: {backup_path}")
        
        # パッチを適用
        with open(main_module_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        
        logger.info("[OK] main.pyへの新Excel出力パッチ適用完了")
        return True
        
    except Exception as e:
        logger.error(f"main.pyパッチ適用エラー: {e}")
        return False

def demo_patch_application():
    """パッチ適用のデモ"""
    print("=" * 60)
    print("新Excel出力モジュール - main.py統合パッチデモ")
    print("=" * 60)
    
    # サンプルデータでテスト
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: from demo_simple_excel_output import create_sample_data
    sample_data = create_sample_data()
    ticker = "PATCH_DEMO"
    
    # 新Excel出力を実行
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: output_path = apply_new_excel_output(sample_data, ticker)
    
    if output_path:
        print(f"\n[OK] パッチデモ成功: {output_path}")
    else:
        print("\n[ERROR] パッチデモ失敗")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    demo_patch_application()
