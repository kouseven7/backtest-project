#!/usr/bin/env python3
"""
TODO-003 Phase 1 調査手法2: 実行時トレーシング調査
デバッグ版main.pyでExit_Signal値の変化追跡
- 戦略出力→統合処理→unified_exporter入力の各段階確認
- 実際の数値: -1→1変換が発生する具体的なタイミング特定
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
import time

print("=" * 80)
print("🔍 TODO-003 Phase 1-2: 実行時トレーシング調査")
print("=" * 80)

def create_debug_main_py():
    """デバッグ版main.pyを作成してExit_Signal変化を追跡"""
    
    # 元のmain.pyを読み取り
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"❌ main.py読み取りエラー: {e}")
        return None
    
    # デバッグコードを挿入するポイントを特定
    debug_insertions = []
    
    # Phase 1-1で特定された疑わしい行にデバッグコード挿入
    suspicious_lines = [
        (549, "exit_signals = abs(strategy_result['Exit_Signal']).sum()"),
        (968, "exit_signals = result_data[result_data['Exit_Signal'].abs() == 1]"),
        (1084, "exit_signals = stock_data[stock_data['Exit_Signal'].abs() == 1]"),
        (464, "integrated_data.loc[exit_idx, 'Exit_Signal'] = 1"),
        (503, "integrated_data.loc[exit_idx, 'Exit_Signal'] = 1"),
        (522, "integrated_data.loc[exit_idx, 'Exit_Signal'] = 1"),
        (612, "integrated_data.loc[final_positions_mask, 'Exit_Signal'] = 1")
    ]
    
    lines = original_content.split('\n')
    
    # デバッグコードを追加
    debug_code_template = '''
# DEBUG: TODO-003 Exit_Signal変化追跡
print(f"[DEBUG-{line_num}] Exit_Signal変化追跡")
if 'Exit_Signal' in locals() and hasattr(Exit_Signal, 'sum'):
    print(f"  Exit_Signal合計: {{Exit_Signal.sum()}}")
if 'strategy_result' in locals() and 'Exit_Signal' in strategy_result.columns:
    exit_neg_ones = (strategy_result['Exit_Signal'] == -1).sum()
    exit_pos_ones = (strategy_result['Exit_Signal'] == 1).sum()
    exit_zeros = (strategy_result['Exit_Signal'] == 0).sum()
    print(f"  strategy_result Exit_Signal: -1={exit_neg_ones}, 1={exit_pos_ones}, 0={exit_zeros}")
if 'integrated_data' in locals() and 'Exit_Signal' in integrated_data.columns:
    int_neg_ones = (integrated_data['Exit_Signal'] == -1).sum()
    int_pos_ones = (integrated_data['Exit_Signal'] == 1).sum()
    int_zeros = (integrated_data['Exit_Signal'] == 0).sum()
    print(f"  integrated_data Exit_Signal: -1={int_neg_ones}, 1={int_pos_ones}, 0={int_zeros}")
if 'stock_data' in locals() and 'Exit_Signal' in stock_data.columns:
    stock_neg_ones = (stock_data['Exit_Signal'] == -1).sum()
    stock_pos_ones = (stock_data['Exit_Signal'] == 1).sum()
    stock_zeros = (stock_data['Exit_Signal'] == 0).sum()
    print(f"  stock_data Exit_Signal: -1={stock_neg_ones}, 1={stock_pos_ones}, 0={stock_zeros}")
'''
    
    # デバッグコードを挿入
    debug_lines = []
    for i, line in enumerate(lines):
        debug_lines.append(line)
        
        # 疑わしい行の後にデバッグコードを挿入
        for line_num, target_line in suspicious_lines:
            if i + 1 == line_num:  # 1-based line numbers
                debug_code = debug_code_template.replace('{line_num}', str(line_num))
                debug_lines.extend(debug_code.split('\n'))
                break
    
    debug_content = '\n'.join(debug_lines)
    
    # デバッグ版main.pyを保存
    debug_filename = 'main_debug_todo003.py'
    try:
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write(debug_content)
        print(f"✅ デバッグ版main.py作成完了: {debug_filename}")
        return debug_filename
    except Exception as e:
        print(f"❌ デバッグ版main.py作成エラー: {e}")
        return None

def run_debug_main_and_trace():
    """デバッグ版main.pyを実行してExit_Signal変化を追跡"""
    
    debug_filename = create_debug_main_py()
    if not debug_filename:
        return None
    
    print(f"\n🔍 **デバッグ版main.py実行開始**")
    print(f"実行ファイル: {debug_filename}")
    
    try:
        # デバッグ版main.pyを実行
        result = subprocess.run(
            [sys.executable, debug_filename],
            capture_output=True,
            text=True,
            timeout=300  # 5分タイムアウト
        )
        
        print(f"✅ デバッグ版実行完了")
        print(f"Return code: {result.returncode}")
        
        # 出力からExit_Signal変化を抽出
        stdout_lines = result.stdout.split('\n')
        stderr_lines = result.stderr.split('\n')
        
        debug_traces = []
        
        # DEBUGログを抽出
        for line in stdout_lines:
            if '[DEBUG-' in line:
                debug_traces.append(line)
        
        print(f"\n📊 **Exit_Signal変化追跡結果**")
        print(f"Debug trace数: {len(debug_traces)}")
        
        for trace in debug_traces:
            print(f"  {trace}")
        
        # エラーログも確認
        error_lines = [line for line in stderr_lines if line.strip()]
        if error_lines:
            print(f"\n⚠️ **エラーログ（最初の5行）**:")
            for line in error_lines[:5]:
                print(f"  {line}")
        
        return {
            'debug_traces': debug_traces,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ デバッグ版実行タイムアウト（5分）")
        return None
    except Exception as e:
        print(f"❌ デバッグ版実行エラー: {e}")
        return None

def analyze_exit_signal_conversion_pattern():
    """Exit_Signal変換パターンを分析"""
    
    print(f"\n🔍 **Exit_Signal変換パターン分析**")
    
    # 最新の出力ファイルを確認
    output_files = []
    for filename in os.listdir('.'):
        if filename.endswith('_data.csv') and '7203.T' in filename:
            output_files.append(filename)
    
    if not output_files:
        print(f"❌ 出力ファイルが見つかりません")
        return None
    
    # 最新のファイルを選択
    latest_file = max(output_files, key=os.path.getmtime)
    print(f"📂 分析対象ファイル: {latest_file}")
    
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(latest_file)
        
        print(f"✅ データ読み込み完了: {len(df)}行")
        print(f"カラム: {list(df.columns)}")
        
        # Exit_Signal分析
        if 'Exit_Signal' in df.columns:
            exit_signal_counts = df['Exit_Signal'].value_counts().sort_index()
            print(f"\n📊 **Exit_Signal値分布**:")
            for value, count in exit_signal_counts.items():
                print(f"  Exit_Signal = {value}: {count}件")
            
            # -1 vs 1 の分析
            negative_ones = (df['Exit_Signal'] == -1).sum()
            positive_ones = (df['Exit_Signal'] == 1).sum()
            zeros = (df['Exit_Signal'] == 0).sum()
            
            print(f"\n🚨 **TODO-001/TODO-002結果との比較**:")
            print(f"  Exit_Signal = -1: {negative_ones}件（期待値: >0）")
            print(f"  Exit_Signal = +1: {positive_ones}件（異常値: >0なら問題）")
            print(f"  Exit_Signal =  0: {zeros}件")
            
            if positive_ones > 0 and negative_ones == 0:
                print(f"  🚨 **変換確認**: -1 → 1 変換が発生している")
            elif negative_ones > 0 and positive_ones == 0:
                print(f"  ✅ **正常**: Exit_Signal = -1 が保持されている")
            elif negative_ones > 0 and positive_ones > 0:
                print(f"  ⚠️ **混在**: -1 と 1 が両方存在（部分的変換）")
            else:
                print(f"  ℹ️ **エグジットなし**: Exit_Signal変更なし")
        
        # Entry_Signalとの同時発生確認
        if 'Entry_Signal' in df.columns and 'Exit_Signal' in df.columns:
            simultaneous = ((df['Entry_Signal'] == 1) & (df['Exit_Signal'] == 1)).sum()
            print(f"\n🔍 **同時発生パターン確認**:")
            print(f"  Entry_Signal=1 & Exit_Signal=1: {simultaneous}行")
            
            if simultaneous > 0:
                print(f"  🚨 **TODO-001パターン確認**: 同時発生が存在")
                # 具体的なインデックス確認
                simultaneous_indices = df[(df['Entry_Signal'] == 1) & (df['Exit_Signal'] == 1)].index.tolist()
                print(f"  同時発生インデックス（最初の10件）: {simultaneous_indices[:10]}")
        
        return {
            'exit_signal_counts': exit_signal_counts.to_dict(),
            'negative_ones': negative_ones,
            'positive_ones': positive_ones,
            'simultaneous_count': simultaneous if 'Entry_Signal' in df.columns else 0
        }
        
    except Exception as e:
        print(f"❌ データ分析エラー: {e}")
        return None

# 実行
print("🔍 実行時トレーシング調査を開始")

# Step 1: デバッグ版main.py実行
trace_result = run_debug_main_and_trace()

# Step 2: 出力ファイル分析
pattern_result = analyze_exit_signal_conversion_pattern()

print(f"\n" + "=" * 80)
print(f"📊 **TODO-003 Phase 1-2 実行時トレーシング結果**")
print(f"=" * 80)

if trace_result:
    print(f"Debug traces取得: {len(trace_result['debug_traces'])}件")
    if trace_result['returncode'] == 0:
        print(f"✅ デバッグ版実行成功")
    else:
        print(f"⚠️ デバッグ版実行エラー（code: {trace_result['returncode']}）")
else:
    print(f"❌ Debug traces取得失敗")

if pattern_result:
    print(f"Exit_Signal分析完了:")
    print(f"  -1: {pattern_result['negative_ones']}件")
    print(f"  +1: {pattern_result['positive_ones']}件") 
    print(f"  同時発生: {pattern_result['simultaneous_count']}件")
    
    # 重要な判定
    if pattern_result['positive_ones'] > 0 and pattern_result['negative_ones'] == 0:
        print(f"🚨 **Phase 1重要発見**: Exit_Signal: -1→1変換を確認")
    elif pattern_result['simultaneous_count'] > 0:
        print(f"🚨 **Phase 1重要発見**: TODO-001パターン（同時発生）を確認")
else:
    print(f"❌ Exit_Signal分析失敗")

print(f"\n✅ TODO-003 Phase 1-2 実行時トレーシング完了")
print(f"📋 次: Phase 1結果統合と Phase 2準備")

print("=" * 80)