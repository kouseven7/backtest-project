#!/usr/bin/env python3
"""
unified_exporter入力データトレーサー
main.pyからunified_exporterに渡されるDataFrameの内容を詳細記録
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

# パス設定
sys.path.append('.')

def trace_dataframe_content(df, trace_point_name):
    """DataFrameの内容を詳細トレース"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[TRACE {timestamp}] {trace_point_name}")
    print("="*60)
    
    if df is None:
        print("❌ DataFrame is None")
        return
    
    if df.empty:
        print("❌ DataFrame is empty")
        return
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Entry_Signal分析
    if 'Entry_Signal' in df.columns:
        entry_counts = df['Entry_Signal'].value_counts().sort_index()
        print(f"Entry_Signal distribution: {entry_counts.to_dict()}")
        entry_ones = df[df['Entry_Signal'] == 1]
        print(f"Entry_Signal=1: {len(entry_ones)}件")
        if len(entry_ones) > 0:
            print(f"Entry_Signal=1 indices: {entry_ones.index.tolist()[:10]}...")
    else:
        print("❌ Entry_Signal column not found")
    
    # Exit_Signal分析（重要）
    if 'Exit_Signal' in df.columns:
        exit_counts = df['Exit_Signal'].value_counts().sort_index()
        print(f"Exit_Signal distribution: {exit_counts.to_dict()}")
        
        exit_minus_ones = df[df['Exit_Signal'] == -1]
        exit_ones = df[df['Exit_Signal'] == 1]
        exit_zeros = df[df['Exit_Signal'] == 0]
        
        print(f"Exit_Signal=-1: {len(exit_minus_ones)}件")
        print(f"Exit_Signal=1: {len(exit_ones)}件")
        print(f"Exit_Signal=0: {len(exit_zeros)}件")
        
        if len(exit_minus_ones) > 0:
            print(f"Exit_Signal=-1 indices: {exit_minus_ones.index.tolist()[:10]}...")
            print("Exit_Signal=-1 samples:")
            for idx in exit_minus_ones.index[:3]:
                row = df.loc[idx]
                print(f"  Index {idx}: Close={row.get('Close', 'N/A')}, Exit_Signal={row['Exit_Signal']}")
        
        if len(exit_ones) > 0:
            print(f"Exit_Signal=1 indices: {exit_ones.index.tolist()[:10]}...")
        
        # 同一行でのEntry/Exit同時発生チェック
        if 'Entry_Signal' in df.columns:
            simultaneous = df[(df['Entry_Signal'] == 1) & (df['Exit_Signal'] != 0)]
            if len(simultaneous) > 0:
                print(f"🚨 同一行Entry/Exit同時発生: {len(simultaneous)}件")
                print(f"同時発生indices: {simultaneous.index.tolist()[:10]}...")
            else:
                print("✅ 同一行Entry/Exit同時発生なし")
    else:
        print("❌ Exit_Signal column not found")
    
    # Strategy列分析
    if 'Strategy' in df.columns:
        strategy_counts = df['Strategy'].value_counts()
        print(f"Strategy distribution: {strategy_counts.to_dict()}")
        nan_strategies = df[df['Strategy'].isna()]
        print(f"Strategy=NaN: {len(nan_strategies)}件")
    
    print("="*60)
    
    # トレースデータを保存
    trace_data = {
        'timestamp': timestamp,
        'trace_point': trace_point_name,
        'shape': df.shape,
        'columns': list(df.columns),
        'entry_signal_dist': df['Entry_Signal'].value_counts().to_dict() if 'Entry_Signal' in df.columns else {},
        'exit_signal_dist': df['Exit_Signal'].value_counts().to_dict() if 'Exit_Signal' in df.columns else {},
        'exit_minus_one_count': len(df[df['Exit_Signal'] == -1]) if 'Exit_Signal' in df.columns else 0,
        'simultaneous_entry_exit': len(df[(df['Entry_Signal'] == 1) & (df['Exit_Signal'] != 0)]) if 'Entry_Signal' in df.columns and 'Exit_Signal' in df.columns else 0
    }
    
    trace_filename = f"unified_exporter_input_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(trace_filename, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)
    
    return trace_data

# main.pyの重要な処理をフック
def hook_main_processing():
    """main.pyの重要処理をフック"""
    print("unified_exporter入力データトレーサー開始")
    
    # 実際のmain.pyを実行しながらトレース
    import main
    
if __name__ == "__main__":
    hook_main_processing()
