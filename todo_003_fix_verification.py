#!/usr/bin/env python3
"""
TODO-003 修正後検証ツール
修正前後の結果比較と追加調査
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def compare_before_after_fix():
    """修正前後の結果比較"""
    print("=== TODO-003 修正後検証 ===")
    
    # 修正前後の出力ファイルを比較
    output_dir = Path("output/main_outputs/csv")
    csv_files = sorted(list(output_dir.glob("*data.csv")))
    
    if len(csv_files) >= 2:
        before_file = csv_files[-2]  # 修正前
        after_file = csv_files[-1]   # 修正後
        
        print(f"修正前: {before_file}")
        print(f"修正後: {after_file}")
        
        # 修正前データ分析
        before_df = pd.read_csv(before_file)
        print(f"\n修正前データ分析 ({len(before_df)}行):")
        if 'Exit_Signal' in before_df.columns:
            before_exit_counts = before_df['Exit_Signal'].value_counts().sort_index()
            print("Exit_Signal分布:")
            for value, count in before_exit_counts.items():
                print(f"  {value}: {count}件")
        
        # 修正後データ分析
        after_df = pd.read_csv(after_file)
        print(f"\n修正後データ分析 ({len(after_df)}行):")
        if 'Exit_Signal' in after_df.columns:
            after_exit_counts = after_df['Exit_Signal'].value_counts().sort_index()
            print("Exit_Signal分布:")
            for value, count in after_exit_counts.items():
                print(f"  {value}: {count}件")
            
            # 比較分析
            print("\n=== 修正効果分析 ===")
            
            before_minus_one = before_df[before_df['Exit_Signal'] == -1]
            after_minus_one = after_df[after_df['Exit_Signal'] == -1]
            
            print(f"修正前 Exit_Signal=-1: {len(before_minus_one)}件")
            print(f"修正後 Exit_Signal=-1: {len(after_minus_one)}件")
            
            if len(after_minus_one) > len(before_minus_one):
                print("✅ 修正成功: Exit_Signal=-1が復活")
            elif len(after_minus_one) == len(before_minus_one) == 0:
                print("❌ 修正効果なし: Exit_Signal=-1がまだ存在しない")
                print("   根本原因は他の箇所にある可能性")
            else:
                print("⚠️ 部分的改善: さらなる調査が必要")
        
        # 取引数の比較
        trades_files = sorted(list(Path("output/main_outputs/csv").glob("*trades.csv")))
        if len(trades_files) >= 2:
            before_trades = pd.read_csv(trades_files[-2])
            after_trades = pd.read_csv(trades_files[-1])
            
            print(f"\n取引数比較:")
            print(f"修正前: {len(before_trades)}件")
            print(f"修正後: {len(after_trades)}件")
            
            if len(after_trades) > len(before_trades):
                improvement = (len(after_trades) - len(before_trades)) / len(before_trades) * 100
                print(f"✅ 取引数改善: +{improvement:.1f}%")
            else:
                print("❌ 取引数改善なし")
    
    # 追加原因調査
    print("\n=== 追加原因調査 ===")
    print("修正効果が見られない場合の可能性:")
    print("1. 戦略レベルでのExit_Signal=-1生成問題")
    print("2. 統合処理での別のabs()使用箇所")
    print("3. データフィルタリング段階での問題")
    print("4. 強制決済システムの過度な介入")
    
    # 戦略レベル調査の推奨
    print("\n=== 次のステップ推奨 ===")
    print("1. 各戦略の個別Exit_Signal生成を確認")
    print("2. 統合前の戦略別シグナル分布を調査")
    print("3. 強制決済システムの動作条件を見直し")
    print("4. TODO-004として戦略レベル調査を実施")

if __name__ == "__main__":
    compare_before_after_fix()