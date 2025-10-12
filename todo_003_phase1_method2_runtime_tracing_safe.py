#!/usr/bin/env python3
"""
TODO-003 Phase 1-2: Runtime Tracing (安全版)
Exit_Signal変換処理の実行時特定と分析

重要調査プロンプト: Exit_Signal: -1→1 変換処理の特定と修正
二重調査手法: Phase 1-2 Runtime Tracing (safe implementation)
"""

import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_existing_output():
    """生成された出力ファイルからExit_Signal変換パターンを分析"""
    print("=== TODO-003 Phase 1-2: Runtime Tracing (安全版) ===")
    print("既存出力ファイルからExit_Signal変換パターンを分析")
    
    # 最新の出力ディレクトリを探す
    output_dir = Path("output/main_outputs")
    if not output_dir.exists():
        print("ERROR: 出力ディレクトリが見つかりません")
        return
    
    # 最新のCSVファイルを特定
    csv_files = list(output_dir.glob("**/*data.csv"))
    if not csv_files:
        print("ERROR: データCSVファイルが見つかりません")
        return
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"分析対象: {latest_csv}")
    
    # CSVデータ読み込み
    try:
        df = pd.read_csv(latest_csv)
        print(f"データ読み込み完了: {len(df)}行, {len(df.columns)}列")
        
        # Exit_Signal列の分析
        if 'Exit_Signal' in df.columns:
            print("\n=== Exit_Signal カラム分析 ===")
            exit_values = df['Exit_Signal'].value_counts().sort_index()
            print("Exit_Signal値の分布:")
            for value, count in exit_values.items():
                print(f"  {value}: {count}件")
            
            # -1から1への変換を検出
            exit_minus_one = df[df['Exit_Signal'] == -1]
            exit_plus_one = df[df['Exit_Signal'] == 1]
            
            print(f"\nExit_Signal = -1 の行数: {len(exit_minus_one)}")
            print(f"Exit_Signal = 1 の行数: {len(exit_plus_one)}")
            
            if len(exit_minus_one) == 0 and len(exit_plus_one) > 0:
                print("❌ 重大発見: Exit_Signal = -1 が完全に失われています!")
                print("   これはabs()変換またはExit_Signal = 1設定による可能性が高い")
            elif len(exit_minus_one) > 0:
                print("✅ Exit_Signal = -1 が保持されています")
        else:
            print("ERROR: Exit_Signalカラムが見つかりません")
        
        # Entry_Signal列の分析（比較用）
        if 'Entry_Signal' in df.columns:
            print("\n=== Entry_Signal カラム分析（比較用） ===")
            entry_values = df['Entry_Signal'].value_counts().sort_index()
            print("Entry_Signal値の分布:")
            for value, count in entry_values.items():
                print(f"  {value}: {count}件")
        
        # 戦略別分析
        strategy_columns = [col for col in df.columns if col.endswith('_Entry_Signal') or col.endswith('_Exit_Signal')]
        if strategy_columns:
            print(f"\n=== 戦略別Signal分析 ===")
            print(f"検出された戦略Signal列: {len(strategy_columns)}列")
            
            for col in sorted(strategy_columns):
                if col.endswith('_Exit_Signal'):
                    values = df[col].value_counts().sort_index()
                    print(f"\n{col}:")
                    for value, count in values.items():
                        if pd.notna(value):
                            print(f"  {value}: {count}件")
        
        # 取引CSVファイルも分析
        trades_files = list(output_dir.glob("**/*trades.csv"))
        if trades_files:
            latest_trades = max(trades_files, key=lambda x: x.stat().st_mtime)
            print(f"\n=== 取引データ分析: {latest_trades} ===")
            
            trades_df = pd.read_csv(latest_trades)
            print(f"取引データ: {len(trades_df)}行")
            
            if 'Signal_Type' in trades_df.columns:
                signal_types = trades_df['Signal_Type'].value_counts()
                print("Signal_Type分布:")
                for signal_type, count in signal_types.items():
                    print(f"  {signal_type}: {count}件")
            
            if 'Signal_Value' in trades_df.columns:
                signal_values = trades_df['Signal_Value'].value_counts().sort_index()
                print("Signal_Value分布:")
                for value, count in signal_values.items():
                    print(f"  {value}: {count}件")
                    
                # -1から1変換の検出
                if -1 not in signal_values and 1 in signal_values:
                    print("❌ 取引データでも-1→1変換を検出!")
        
        # JSONファイルからメタデータ分析
        json_files = list(output_dir.glob("**/*complete.json"))
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"\n=== JSON メタデータ分析: {latest_json} ===")
            
            with open(latest_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if 'strategy_results' in json_data:
                for strategy_name, results in json_data['strategy_results'].items():
                    if 'signal_statistics' in results:
                        print(f"\n戦略 {strategy_name}:")
                        signal_stats = results['signal_statistics']
                        if 'exit_signals' in signal_stats:
                            exit_stats = signal_stats['exit_signals']
                            print(f"  Exit Signal統計: {exit_stats}")
        
        return {
            'exit_signal_analysis': exit_values.to_dict() if 'Exit_Signal' in df.columns else {},
            'entry_signal_analysis': entry_values.to_dict() if 'Entry_Signal' in df.columns else {},
            'strategy_columns': len(strategy_columns),
            'data_rows': len(df),
            'potential_conversion_detected': len(exit_minus_one) == 0 and len(exit_plus_one) > 0 if 'Exit_Signal' in df.columns else False
        }
        
    except Exception as e:
        print(f"ERROR: ファイル分析エラー: {e}")
        return None

def generate_phase1_summary():
    """Phase 1-1とPhase 1-2の結果を統合"""
    print("\n=== TODO-003 Phase 1 統合サマリー ===")
    
    # Phase 1-1 Static Analysis結果
    print("Phase 1-1 Static Analysis結果:")
    print("- 60 Exit_Signal関連行")
    print("- 7 abs()変換行（疑い）")
    print("- 18 怪しい処理行")
    print("- Lines 549, 968, 1084: abs(Exit_Signal)")
    
    # Phase 1-2 Runtime Tracing結果
    runtime_results = analyze_existing_output()
    
    if runtime_results:
        print("\nPhase 1-2 Runtime Tracing結果:")
        print(f"- データ行数: {runtime_results['data_rows']}")
        print(f"- 戦略Signal列数: {runtime_results['strategy_columns']}")
        print(f"- Exit_Signal分析: {runtime_results['exit_signal_analysis']}")
        print(f"- Entry_Signal分析: {runtime_results['entry_signal_analysis']}")
        
        if runtime_results['potential_conversion_detected']:
            print("❌ 変換検出: Exit_Signal = -1 が完全に失われている")
        else:
            print("✅ Exit_Signal = -1 が保持されている")
    
    # Phase 1 結論
    print("\n=== Phase 1 結論 ===")
    print("Static Analysis + Runtime Tracingにより以下を確認:")
    print("1. main.pyの Lines 549, 968, 1084 でabs(Exit_Signal)を使用")
    print("2. 実行時出力でExit_Signal = -1の有無を検証")
    print("3. 両方の証拠からExit_Signal変換の実態を特定")
    
    return runtime_results

if __name__ == "__main__":
    print("TODO-003 Phase 1-2: Exit_Signal変換処理の実行時特定")
    print("重要調査プロンプト対応: 二重調査手法による確実な特定")
    
    # 安全な実行時分析
    results = generate_phase1_summary()
    
    # 調査結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"todo_003_phase1_runtime_analysis_safe_{timestamp}.json"
    
    if results:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_timestamp': timestamp,
                'phase': 'TODO-003 Phase 1-2',
                'method': 'Runtime Tracing (Safe)',
                'results': results,
                'status': 'completed'
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n調査結果保存: {results_file}")
    
    print("\nTODO-003 Phase 1-2 完了")