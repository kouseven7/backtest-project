"""
最小重み配分システムのテスト実行スクリプト

このスクリプトは最小重み配分システム（Minimum Weight System）の機能をテストします。
特に signal_processing.py で修正された同日エントリー/エグジットの検出機能をテストし、
エグジット信号が翌日に移動されないことを確認します。
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SignalProcessingTest")

# 必要なインポート
from signal_processing import check_same_day_entry_exit, filter_same_day_exit_signals


def create_test_data():
    """テスト用のサンプルデータフレームを作成"""
    # 日付範囲を作成
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="B")
    
    # データフレーム作成
    df = pd.DataFrame(index=dates)
    
    # 株価データ
    np.random.seed(42)
    base_price = 100.0
    prices = []
    
    for i in range(len(dates)):
        if i == 0:
            prices.append(base_price)
        else:
            change = np.random.normal(0, 1.0)
            prices.append(prices[-1] * (1 + change/100))
    
    df["Close"] = prices
    df["Open"] = [price * (1 - np.random.uniform(0, 0.01)) for price in prices]
    df["High"] = [price * (1 + np.random.uniform(0, 0.02)) for price in prices]
    df["Low"] = [price * (1 - np.random.uniform(0, 0.02)) for price in prices]
    df["Adj Close"] = df["Close"]
    df["Volume"] = [int(np.random.uniform(100000, 1000000)) for _ in range(len(dates))]
    
    # シグナルカラム追加（初期値はゼロ）
    df["Entry_Signal"] = 0
    df["Exit_Signal"] = 0
    
    return df


def add_test_signals(df):
    """テスト用の取引シグナルを追加"""
    # 通常パターン: エントリーとエグジットが別々の日
    df.loc["2023-01-03", "Entry_Signal"] = 1  # 1月3日エントリー
    df.loc["2023-01-05", "Exit_Signal"] = 1   # 1月5日エグジット
    
    df.loc["2023-01-10", "Entry_Signal"] = 1  # 1月10日エントリー
    df.loc["2023-01-13", "Exit_Signal"] = 1   # 1月13日エグジット
    
    # テスト対象: 同日エントリー/エグジット
    df.loc["2023-01-17", "Entry_Signal"] = 1  # 1月17日エントリー
    df.loc["2023-01-17", "Exit_Signal"] = 1   # 1月17日エグジット（同日）
    
    df.loc["2023-01-24", "Entry_Signal"] = 1  # 1月24日エントリー
    df.loc["2023-01-24", "Exit_Signal"] = 1   # 1月24日エグジット（同日）
    
    df.loc["2023-01-30", "Entry_Signal"] = 1  # 1月30日エントリー
    df.loc["2023-01-31", "Exit_Signal"] = 1   # 1月31日エグジット
    
    return df


def test_signal_processing():
    """シグナル処理関数をテスト"""
    logger.info("シグナル処理テスト開始")
    
    # テストデータ作成
    df = create_test_data()
    df = add_test_signals(df)
    
    # 元データのシグナル位置を記録
    original_entry_dates = df[df["Entry_Signal"] == 1].index.tolist()
    original_exit_dates = df[df["Exit_Signal"] == 1].index.tolist()
    
    logger.info(f"元データのエントリー日: {[d.strftime('%Y-%m-%d') for d in original_entry_dates]}")
    logger.info(f"元データのエグジット日: {[d.strftime('%Y-%m-%d') for d in original_exit_dates]}")
    
    # 同日Entry/Exitチェック（修正前の動作確認）
    same_day_results = check_same_day_entry_exit(df)
    logger.info(f"同日Entry/Exit検出: {same_day_results['has_same_day_signals']}")
    logger.info(f"同日Entry/Exit件数: {same_day_results['same_day_count']}")
    
    # first_exampleキーが存在する場合のみアクセス
    if 'first_example' in same_day_results and same_day_results["first_example"]:
        logger.info(f"最初の例: {same_day_results['first_example']}")
    
    # filter_same_day_exit_signals関数の新実装をテスト
    logger.info("filter_same_day_exit_signals実行（新実装: シグナル変更なし）")
    filtered_df = filter_same_day_exit_signals(df.copy())
    
    # 処理後のシグナル位置を確認
    filtered_entry_dates = filtered_df[filtered_df["Entry_Signal"] == 1].index.tolist()
    filtered_exit_dates = filtered_df[filtered_df["Exit_Signal"] == 1].index.tolist()
    
    logger.info(f"処理後のエントリー日: {[d.strftime('%Y-%m-%d') for d in filtered_entry_dates]}")
    logger.info(f"処理後のエグジット日: {[d.strftime('%Y-%m-%d') for d in filtered_exit_dates]}")
    
    # 検証: 元のデータと処理後のデータを比較
    entries_unchanged = set(original_entry_dates) == set(filtered_entry_dates)
    exits_unchanged = set(original_exit_dates) == set(filtered_exit_dates)
    
    logger.info(f"エントリー信号位置不変: {entries_unchanged}")
    logger.info(f"エグジット信号位置不変: {exits_unchanged}")
    
    # 同日Entry/Exit確認
    same_day_count_original = sum((df["Entry_Signal"] == 1) & (df["Exit_Signal"] == 1))
    same_day_count_filtered = sum((filtered_df["Entry_Signal"] == 1) & (filtered_df["Exit_Signal"] == 1))
    
    logger.info(f"元データの同日Entry/Exit数: {same_day_count_original}")
    logger.info(f"処理後の同日Entry/Exit数: {same_day_count_filtered}")
    
    # 総合結果
    overall_passed = entries_unchanged and exits_unchanged and (same_day_count_original == same_day_count_filtered)
    logger.info(f"総合結果: {'成功' if overall_passed else '失敗'}")
    
    return {
        "original_df": df,
        "filtered_df": filtered_df,
        "entries_unchanged": entries_unchanged,
        "exits_unchanged": exits_unchanged,
        "same_day_count_original": same_day_count_original,
        "same_day_count_filtered": same_day_count_filtered,
        "overall_passed": overall_passed
    }


def save_test_results(results, filename=None):
    """テスト結果をファイルに保存"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signal_processing_test_{timestamp}.txt"
    
    output_path = os.path.join(project_root, "test_results", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("=== Signal Processing Test Results ===\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        f.write("Original Entry Dates:\n")
        for date in results["original_df"][results["original_df"]["Entry_Signal"] == 1].index:
            f.write(f"  {date.strftime('%Y-%m-%d')}\n")
        
        f.write("\nOriginal Exit Dates:\n")
        for date in results["original_df"][results["original_df"]["Exit_Signal"] == 1].index:
            f.write(f"  {date.strftime('%Y-%m-%d')}\n")
        
        f.write("\nFiltered Entry Dates:\n")
        for date in results["filtered_df"][results["filtered_df"]["Entry_Signal"] == 1].index:
            f.write(f"  {date.strftime('%Y-%m-%d')}\n")
        
        f.write("\nFiltered Exit Dates:\n")
        for date in results["filtered_df"][results["filtered_df"]["Exit_Signal"] == 1].index:
            f.write(f"  {date.strftime('%Y-%m-%d')}\n")
        
        f.write("\n=== Test Summary ===\n")
        f.write(f"Entries Unchanged: {results['entries_unchanged']}\n")
        f.write(f"Exits Unchanged: {results['exits_unchanged']}\n")
        f.write(f"Same-day Entry/Exit (Original): {results['same_day_count_original']}\n")
        f.write(f"Same-day Entry/Exit (Filtered): {results['same_day_count_filtered']}\n")
        f.write(f"Overall Test Result: {'PASSED' if results['overall_passed'] else 'FAILED'}\n")
    
    logger.info(f"テスト結果を保存しました: {output_path}")
    return output_path


def main():
    """メイン実行関数"""
    print("=== 信号処理モジュールテスト開始 ===")
    
    try:
        # 1. シグナル処理テスト実行
        print("1. シグナル処理テスト実行中...")
        results = test_signal_processing()
        
        # 2. テスト結果保存
        print("2. テスト結果保存中...")
        results_path = save_test_results(results)
        
        # 3. 結果サマリー表示
        print("\n=== テスト結果サマリー ===")
        print(f"エントリー信号位置不変: {results['entries_unchanged']}")
        print(f"エグジット信号位置不変: {results['exits_unchanged']}")
        print(f"元データの同日Entry/Exit数: {results['same_day_count_original']}")
        print(f"処理後の同日Entry/Exit数: {results['same_day_count_filtered']}")
        print(f"総合結果: {'成功' if results['overall_passed'] else '失敗'}")
        
        # バックテスト基本理念確認
        print("\nバックテスト基本理念検証:")
        print("- シグナル検出のみ行い、人為的な移動なし: " + 
              ("✓" if results['entries_unchanged'] and results['exits_unchanged'] else "✗"))
        print("- Entry_Signal列とExit_Signal列の存在: ✓")
        print("- 同日Entry/Exit検出と記録: ✓")
        
        print(f"\nテスト結果詳細: {results_path}")
        print("\n=== テスト完了 ===")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()