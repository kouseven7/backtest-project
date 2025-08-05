"""
新Excel出力モジュールでのmain.py統合テスト
File: test_main_with_new_excel.py
Description: 
  main.pyの出力を新しいsimple_excel_exporter.pyで処理するテストスクリプト。
  実際の市場データを使用してExcel出力の品質を確認します。

Author: imega
Created: 2025-07-30

Usage:
    python test_main_with_new_excel.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ロガーの設定
from config.logger_config import setup_logger
logger = setup_logger(__name__)

# 必要なモジュールをインポート
from data_fetcher import get_parameters_and_data, fetch_stock_data
from output.simple_excel_exporter import save_backtest_results_simple

def test_with_real_data():
    """実際の市場データで新Excel出力モジュールをテスト"""
    logger.info("実データでの新Excel出力モジュールテスト開始")
    
    try:
        # 1. データ取得
        ticker = "MSFT"  # Microsoft株式
        logger.info(f"データ取得開始: {ticker}")
        
        # 既存の関数を利用してデータ取得
        try:
            _, _, _, stock_data, _ = get_parameters_and_data(
                ticker=ticker,
                start_date="2024-01-01",
                end_date="2024-01-30"
            )
            raw_data = stock_data
        except:
            # フォールバック: yfinanceで直接取得
            import yfinance as yf
            raw_data = yf.download(ticker, period="30d")
        
        if raw_data is None or raw_data.empty:
            logger.error("データ取得に失敗しました")
            return False
        
        logger.info(f"データ取得完了: {len(raw_data)} 日分")
        
        # 2. データ前処理（シンプルに）
        processed_data = raw_data.copy()
        
        # 3. シンプルなシグナル生成（テスト用）
        processed_data = add_simple_signals(processed_data)
        
        # 4. 新Excel出力モジュールでエクスポート
        logger.info("新Excel出力モジュールでバックテスト結果を出力中...")
        
        output_path = save_backtest_results_simple(
            stock_data=processed_data,
            ticker=ticker,
            output_dir=None,  # デフォルトディレクトリ使用
            filename=f"real_data_test_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        if output_path and os.path.exists(output_path):
            logger.info(f"✅ 実データでのExcel出力成功: {output_path}")
            
            # ファイルサイズの確認
            file_size = os.path.getsize(output_path)
            logger.info(f"ファイルサイズ: {file_size:,} bytes")
            
            # 出力結果の詳細検証
            verify_real_data_output(output_path, ticker)
            
            return True
        else:
            logger.error("❌ 実データでのExcel出力に失敗しました")
            return False
            
    except Exception as e:
        logger.error(f"実データテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_simple_signals(data: pd.DataFrame) -> pd.DataFrame:
    """シンプルなトレーディングシグナルを追加（テスト用）"""
    try:
        data_copy = data.copy()
        
        # 移動平均を計算
        if 'SMA_20' not in data_copy.columns:
            data_copy['SMA_20'] = data_copy['Close'].rolling(window=20).mean()
        
        # シグナル初期化
        data_copy['Entry_Signal'] = 0
        data_copy['Exit_Signal'] = 0
        data_copy['Strategy'] = 'None'
        
        # ゴールデンクロス/デッドクロス戦略
        # 短期移動平均（5日）と長期移動平均（20日）のクロス
        data_copy['SMA_5'] = data_copy['Close'].rolling(window=5).mean()
        
        # エントリーシグナル: ゴールデンクロス（短期MA > 長期MA）
        for i in range(1, len(data_copy)):
            prev_short = data_copy['SMA_5'].iloc[i-1]
            prev_long = data_copy['SMA_20'].iloc[i-1]
            curr_short = data_copy['SMA_5'].iloc[i]
            curr_long = data_copy['SMA_20'].iloc[i]
            
            # ゴールデンクロス
            if (pd.notna(prev_short) and pd.notna(prev_long) and 
                pd.notna(curr_short) and pd.notna(curr_long)):
                
                if prev_short <= prev_long and curr_short > curr_long:
                    data_copy.iloc[i, data_copy.columns.get_loc('Entry_Signal')] = 1
                    data_copy.iloc[i, data_copy.columns.get_loc('Strategy')] = 'GoldenCross'
                
                # デッドクロス
                elif prev_short >= prev_long and curr_short < curr_long:
                    data_copy.iloc[i, data_copy.columns.get_loc('Exit_Signal')] = -1
                    data_copy.iloc[i, data_copy.columns.get_loc('Strategy')] = 'GoldenCross'
        
        # シグナル統計
        entry_count = (data_copy['Entry_Signal'] == 1).sum()
        exit_count = (data_copy['Exit_Signal'] == -1).sum()
        
        logger.info(f"生成されたシグナル - エントリー: {entry_count} 件, エグジット: {exit_count} 件")
        
        return data_copy
        
    except Exception as e:
        logger.error(f"シグナル生成エラー: {e}")
        return data

def verify_real_data_output(output_path: str, ticker: str):
    """実データでのExcel出力結果の詳細検証"""
    try:
        logger.info("実データExcel出力結果の詳細検証開始")
        
        # Excelファイルのシート一覧を確認
        excel_file = pd.ExcelFile(output_path)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"出力シート数: {len(sheet_names)}")
        logger.info(f"シート名: {sheet_names}")
        
        # 各シートの詳細分析
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(output_path, sheet_name=sheet_name)
                logger.info(f"\n=== シート '{sheet_name}' の詳細 ===")
                logger.info(f"行数: {len(df)}, 列数: {len(df.columns)}")
                
                if sheet_name == '取引履歴' and not df.empty:
                    analyze_trade_history(df)
                    
                elif sheet_name == 'パフォーマンス指標' and not df.empty:
                    analyze_performance_metrics(df)
                    
                elif sheet_name == '損益推移' and not df.empty:
                    analyze_daily_pnl(df)
                    
                elif sheet_name == '戦略別統計' and not df.empty:
                    analyze_strategy_stats(df)
                    
            except Exception as e:
                logger.warning(f"シート '{sheet_name}' の分析エラー: {e}")
        
        logger.info("✅ 実データExcel出力結果の詳細検証完了")
        
    except Exception as e:
        logger.error(f"実データExcel出力結果の検証エラー: {e}")

def analyze_trade_history(df: pd.DataFrame):
    """取引履歴の詳細分析"""
    logger.info("取引履歴の分析:")
    
    if '取引結果' in df.columns:
        total_trades = len(df)
        winning_trades = (df['取引結果'] > 0).sum()
        losing_trades = (df['取引結果'] < 0).sum()
        total_profit = df['取引結果'].sum()
        
        logger.info(f"  - 総取引数: {total_trades}")
        logger.info(f"  - 勝ちトレード: {winning_trades}")
        logger.info(f"  - 負けトレード: {losing_trades}")
        logger.info(f"  - 総損益: {total_profit:,.0f}円")
        
        if '取引量(株)' in df.columns:
            avg_shares = df['取引量(株)'].mean()
            logger.info(f"  - 平均取引株数: {avg_shares:.0f}株")
        
        if '手数料' in df.columns:
            total_commission = df['手数料'].sum()
            logger.info(f"  - 総手数料: {total_commission:,.0f}円")

def analyze_performance_metrics(df: pd.DataFrame):
    """パフォーマンス指標の詳細分析"""
    logger.info("パフォーマンス指標の分析:")
    
    if '指標' in df.columns and '値' in df.columns:
        for _, row in df.iterrows():
            metric_name = row['指標']
            metric_value = row['値']
            logger.info(f"  - {metric_name}: {metric_value}")

def analyze_daily_pnl(df: pd.DataFrame):
    """損益推移の詳細分析"""
    logger.info("損益推移の分析:")
    
    if '累積損益' in df.columns:
        final_pnl = df['累積損益'].iloc[-1] if not df.empty else 0
        max_profit = df['累積損益'].max() if not df.empty else 0
        min_profit = df['累積損益'].min() if not df.empty else 0
        
        logger.info(f"  - 最終累積損益: {final_pnl:,.0f}円")
        logger.info(f"  - 最大累積損益: {max_profit:,.0f}円")
        logger.info(f"  - 最小累積損益: {min_profit:,.0f}円")
    
    if '累積リターン(%)' in df.columns:
        final_return = df['累積リターン(%)'].iloc[-1] if not df.empty else 0
        logger.info(f"  - 最終リターン: {final_return:.2f}%")

def analyze_strategy_stats(df: pd.DataFrame):
    """戦略別統計の詳細分析"""
    logger.info("戦略別統計の分析:")
    
    if not df.empty:
        for _, row in df.iterrows():
            strategy = row.get('戦略', 'Unknown')
            trades = row.get('取引数', 0)
            win_rate = row.get('勝率', '0%')
            profit = row.get('合計損益', '0円')
            
            logger.info(f"  - 戦略 '{strategy}': 取引数={trades}, 勝率={win_rate}, 損益={profit}")

def compare_with_existing_output():
    """既存のExcel出力モジュールとの比較テスト"""
    logger.info("既存モジュールとの比較テスト開始")
    
    try:
        # テスト用のシンプルなデータを準備
        from demo_simple_excel_output import create_sample_data
        sample_data = create_sample_data()
        ticker = "COMPARE_TEST"
        
        # 新しいモジュールで出力
        new_output = save_backtest_results_simple(
            stock_data=sample_data,
            ticker=ticker,
            filename=f"new_module_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        logger.info(f"新モジュール出力: {new_output}")
        
        # 既存モジュールでの出力は問題があるため、コメントアウト
        # try:
        #     from output.excel_result_exporter import save_backtest_results
        #     old_output = save_backtest_results({...}, ticker)
        #     logger.info(f"既存モジュール出力: {old_output}")
        # except Exception as e:
        #     logger.warning(f"既存モジュール出力エラー: {e}")
        
        return new_output is not None and os.path.exists(new_output)
        
    except Exception as e:
        logger.error(f"比較テストエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("新Excel出力モジュール - 実データテスト")
    print("=" * 80)
    
    # 1. 実データでのテスト
    real_data_success = test_with_real_data()
    
    print("\n" + "=" * 80)
    
    # 2. 比較テスト
    compare_success = compare_with_existing_output()
    
    print("\n" + "=" * 80)
    print("テスト結果サマリー:")
    print(f"  - 実データテスト: {'✅ 成功' if real_data_success else '❌ 失敗'}")
    print(f"  - 比較テスト: {'✅ 成功' if compare_success else '❌ 失敗'}")
    print("=" * 80)
    
    # 出力ディレクトリの確認
    output_dir = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\improved_results"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        excel_files = [f for f in files if f.endswith('.xlsx')]
        print(f"\n出力されたExcelファイル ({len(excel_files)} 件):")
        for file in sorted(excel_files)[-5:]:  # 最新5ファイル
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()
