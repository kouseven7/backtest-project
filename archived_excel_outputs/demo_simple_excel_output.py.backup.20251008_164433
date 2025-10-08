"""
新Excel出力モジュールのデモ実行スクリプト
File: demo_simple_excel_output.py
Description: 
  新しいsimple_excel_exporter.pyを使用したデモ実行スクリプト。
  正確な取引履歴計算と適切なExcel出力を確認できます。

Author: imega
Created: 2025-07-30

Usage:
    python demo_simple_excel_output.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ロガーの設定
from config.logger_config import setup_logger
logger = setup_logger(__name__)

# 新しいExcel出力モジュールをインポート
from output.simple_excel_exporter import SimpleExcelExporter, save_backtest_results_simple

def create_sample_data() -> pd.DataFrame:
    """サンプルデータを作成（テスト用）"""
    logger.info("サンプルデータ作成開始")
    
    # 30日間のサンプルデータ
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # 基本価格データ（ランダムウォーク）
    np.random.seed(42)
    base_price = 1000
    price_changes = np.random.normal(0, 20, len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    # OHLC価格を生成
    opens = prices + np.random.normal(0, 5, len(dates))
    highs = np.maximum(opens, prices) + np.random.exponential(10, len(dates))
    lows = np.minimum(opens, prices) - np.random.exponential(10, len(dates))
    volumes = np.random.randint(100000, 1000000, len(dates))
    
    # データフレーム作成
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Adj Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # シンプルなエントリー・エグジットシグナルを生成
    data['Entry_Signal'] = 0
    data['Exit_Signal'] = 0
    data['Strategy'] = 'None'
    
    # 5日ごとにエントリー、その3日後にエグジット
    entry_days = [5, 10, 15, 20, 25]
    
    for entry_day in entry_days:
        if entry_day < len(data):
            data.loc[data.index[entry_day], 'Entry_Signal'] = 1
            data.loc[data.index[entry_day], 'Strategy'] = 'TestStrategy'
            
            # エグジットは3日後
            exit_day = entry_day + 3
            if exit_day < len(data):
                data.loc[data.index[exit_day], 'Exit_Signal'] = -1
                data.loc[data.index[exit_day], 'Strategy'] = 'TestStrategy'
    
    # デバッグ情報を追加
    entry_count = (data['Entry_Signal'] == 1).sum()
    exit_count = (data['Exit_Signal'] == -1).sum()
    logger.info(f"実際のエントリーシグナル: {entry_count} 件")
    logger.info(f"実際のエグジットシグナル: {exit_count} 件")
    
    logger.info(f"サンプルデータ作成完了: {len(data)} 日分, エントリー {len(entry_days)} 回")
    return data

def demo_simple_excel_export():
    """新しいExcel出力モジュールのデモ実行"""
    logger.info("新Excel出力モジュールデモ開始")
    
    try:
        # 1. サンプルデータの作成
        stock_data = create_sample_data()
        ticker = "TEST_DEMO"
        
        # 2. 新しいExcel出力モジュールでエクスポート
        logger.info("新Excel出力モジュールでバックテスト結果を出力中...")
        
        output_path = save_backtest_results_simple(
            stock_data=stock_data,
            ticker=ticker,
            output_dir=None,  # デフォルト: backtest_results/improved_results/
            filename=None     # 自動生成
        )
        
        if output_path and os.path.exists(output_path):
            logger.info(f"✅ Excel出力成功: {output_path}")
            
            # ファイルサイズの確認
            file_size = os.path.getsize(output_path)
            logger.info(f"ファイルサイズ: {file_size:,} bytes")
            
            # 出力結果の簡易検証
            verify_excel_output(output_path)
            
        else:
            logger.error("❌ Excel出力に失敗しました")
            
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()

def verify_excel_output(output_path: str):
    """Excel出力結果の簡易検証"""
    try:
        logger.info("Excel出力結果の検証開始")
        
        # Excelファイルのシート一覧を確認
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"出力シート数: {len(sheet_names)}")
        logger.info(f"シート名: {sheet_names}")
        
        # 各シートの概要を確認
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(output_path, sheet_name=sheet_name)
                logger.info(f"シート '{sheet_name}': {len(df)} 行 x {len(df.columns)} 列")
                
                # 重要なシートの内容を詳細確認
                if sheet_name == '取引履歴' and not df.empty:
                    total_trades = len(df)
                    total_profit = df['取引結果'].sum() if '取引結果' in df.columns else 0
                    logger.info(f"  - 総取引数: {total_trades}")
                    logger.info(f"  - 総損益: {total_profit:,.0f}円")
                    
                elif sheet_name == 'パフォーマンス指標' and not df.empty:
                    if '指標' in df.columns and '値' in df.columns:
                        win_rate_row = df[df['指標'] == '勝率']
                        if not win_rate_row.empty:
                            win_rate = win_rate_row['値'].iloc[0]
                            logger.info(f"  - 勝率: {win_rate}")
                            
                        drawdown_row = df[df['指標'] == '最大ドローダウン(%)']
                        if not drawdown_row.empty:
                            drawdown = drawdown_row['値'].iloc[0]
                            logger.info(f"  - 最大ドローダウン: {drawdown}")
                
            except Exception as e:
                logger.warning(f"シート '{sheet_name}' の読み込みエラー: {e}")
        
        logger.info("✅ Excel出力結果の検証完了")
        
    except Exception as e:
        logger.error(f"Excel出力結果の検証エラー: {e}")

def compare_with_old_module():
    """既存モジュールとの比較デモ（オプション）"""
    logger.info("既存モジュールとの比較開始")
    
    try:
        # サンプルデータの作成
        stock_data = create_sample_data()
        ticker = "COMPARE_TEST"
        
        # 新しいモジュールで出力
        new_output = save_backtest_results_simple(stock_data, ticker)
        
        # 既存モジュールで出力（エラーが出る可能性あり）
        try:
            from output.excel_result_exporter import save_backtest_results
            old_output = save_backtest_results(stock_data, ticker)
            
            if new_output and old_output:
                logger.info(f"新モジュール出力: {new_output}")
                logger.info(f"既存モジュール出力: {old_output}")
                logger.info("✅ 両方の出力が完了しました")
            else:
                logger.warning("一方または両方の出力に問題がありました")
                
        except Exception as e:
            logger.warning(f"既存モジュール出力エラー: {e}")
            logger.info("新モジュールのみでの出力を確認してください")
            
    except Exception as e:
        logger.error(f"比較デモエラー: {e}")

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("新Excel出力モジュール（simple_excel_exporter.py）デモ")
    print("=" * 60)
    
    # デモ実行
    demo_simple_excel_export()
    
    print("\n" + "=" * 60)
    print("デモ実行完了")
    print("=" * 60)
    
    # 出力ディレクトリの確認
    output_dir = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\improved_results"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        excel_files = [f for f in files if f.endswith('.xlsx')]
        print(f"\n出力ディレクトリ: {output_dir}")
        print(f"Excelファイル数: {len(excel_files)}")
        if excel_files:
            print("最新のファイル:")
            for file in sorted(excel_files)[-3:]:  # 最新3ファイル
                print(f"  - {file}")

if __name__ == "__main__":
    main()
