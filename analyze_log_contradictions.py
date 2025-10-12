#!/usr/bin/env python3
"""
ログ矛盾分析ツール
目的: main.pyのログと実際の調査結果の矛盾を詳細分析

調査対象の矛盾:
1. エントリー数: ログ62件 vs 調査結果81件 
2. エグジット数: ログ62件 vs 調査結果0件
3. フォールバック警告の重複
4. 統合システム実行失敗の重複ログ
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def analyze_output_files():
    """
    最新の出力ファイルを分析してログと実際のデータの差を確認
    """
    print("=" * 60)
    print("🔍 出力ファイル分析開始")
    print("=" * 60)
    
    # 出力ディレクトリのパス
    output_dir = Path(r"C:\Users\imega\Documents\my_backtest_project\output\main_outputs")
    
    # 最新のファイルを探す
    csv_dir = output_dir / 'csv'
    json_dir = output_dir / 'json'
    txt_dir = output_dir / 'txt'
    
    latest_files = {}
    
    # 最新のCSVファイル（取引データ）
    if csv_dir.exists():
        trades_files = list(csv_dir.glob("*trades*.csv"))
        data_files = list(csv_dir.glob("*data*.csv"))
        
        if trades_files:
            latest_files['trades_csv'] = max(trades_files, key=lambda x: x.stat().st_mtime)
        if data_files:
            latest_files['data_csv'] = max(data_files, key=lambda x: x.stat().st_mtime)
    
    # 最新のJSONファイル
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))
        if json_files:
            latest_files['json'] = max(json_files, key=lambda x: x.stat().st_mtime)
    
    # 最新のTXTファイル
    if txt_dir.exists():
        txt_files = list(txt_dir.glob("*.txt"))
        if txt_files:
            latest_files['txt'] = max(txt_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 発見された最新ファイル: {len(latest_files)}個")
    for file_type, file_path in latest_files.items():
        print(f"  - {file_type}: {file_path.name}")
    
    return latest_files

def analyze_trades_csv(trades_csv_path):
    """
    取引CSVファイルの詳細分析
    """
    print(f"\n📊 取引CSVファイル分析: {trades_csv_path.name}")
    print("-" * 50)
    
    try:
        trades_df = pd.read_csv(trades_csv_path)
        
        print(f"取引データ行数: {len(trades_df)}")
        print(f"取引データ列数: {len(trades_df.columns)}")
        print(f"列名: {list(trades_df.columns)}")
        
        # Entry/Exit分析
        if 'Type' in trades_df.columns:
            entry_count = (trades_df['Type'] == 'entry').sum() if 'Type' in trades_df.columns else 0
            exit_count = (trades_df['Type'] == 'exit').sum() if 'Type' in trades_df.columns else 0
            
            print(f"📈 エントリー取引数: {entry_count}")
            print(f"📉 エグジット取引数: {exit_count}")
            print(f"🎯 総取引数: {len(trades_df)}")
        
        # 戦略別統計
        if 'Strategy' in trades_df.columns:
            strategy_counts = trades_df['Strategy'].value_counts()
            print(f"\n📋 戦略別取引数:")
            for strategy, count in strategy_counts.items():
                print(f"  - {strategy}: {count}件")
        
        # サンプルデータ表示
        print(f"\n📝 サンプルデータ（最初の5行）:")
        print(trades_df.head().to_string())
        
        return {
            'total_trades': len(trades_df),
            'entry_count': entry_count if 'Type' in trades_df.columns else 'N/A',
            'exit_count': exit_count if 'Type' in trades_df.columns else 'N/A',
            'strategies': strategy_counts.to_dict() if 'Strategy' in trades_df.columns else {}
        }
        
    except Exception as e:
        print(f"❌ CSVファイル読み込みエラー: {e}")
        return None

def analyze_data_csv(data_csv_path):
    """
    データCSVファイルの詳細分析（Entry_Signal/Exit_Signal列）
    """
    print(f"\n📊 データCSVファイル分析: {data_csv_path.name}")
    print("-" * 50)
    
    try:
        data_df = pd.read_csv(data_csv_path)
        
        print(f"データ行数: {len(data_df)}")
        print(f"データ列数: {len(data_df.columns)}")
        
        # Entry_Signal/Exit_Signal分析
        signal_analysis = {}
        
        if 'Entry_Signal' in data_df.columns:
            entry_signals = (data_df['Entry_Signal'] == 1).sum()
            signal_analysis['entry_signals'] = entry_signals
            print(f"📈 Entry_Signal=1の数: {entry_signals}")
        
        if 'Exit_Signal' in data_df.columns:
            exit_signals = (data_df['Exit_Signal'] == 1).sum()
            signal_analysis['exit_signals'] = exit_signals
            print(f"📉 Exit_Signal=1の数: {exit_signals}")
        
        # Strategy列の分析
        if 'Strategy' in data_df.columns:
            active_strategies = data_df[data_df['Strategy'] != '']['Strategy'].value_counts()
            print(f"\n📋 アクティブ戦略分布:")
            for strategy, count in active_strategies.items():
                print(f"  - {strategy}: {count}回")
        
        return signal_analysis
        
    except Exception as e:
        print(f"❌ データCSVファイル読み込みエラー: {e}")
        return None

def analyze_json_metadata(json_path):
    """
    JSONファイルのメタデータ分析
    """
    print(f"\n📊 JSONファイル分析: {json_path.name}")
    print("-" * 50)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 基本統計
        if 'trading_statistics' in json_data:
            stats = json_data['trading_statistics']
            print(f"📊 JSON内取引統計:")
            print(f"  - 総取引数: {stats.get('total_trades', 'N/A')}")
            print(f"  - エントリー数: {stats.get('entry_count', 'N/A')}")
            print(f"  - エグジット数: {stats.get('exit_count', 'N/A')}")
        
        # パフォーマンス統計
        if 'performance_metrics' in json_data:
            perf = json_data['performance_metrics']
            print(f"💰 パフォーマンス統計:")
            print(f"  - 総損益: ¥{perf.get('total_pnl', 'N/A')}")
            print(f"  - 勝率: {perf.get('win_rate', 'N/A')}%")
        
        return json_data
        
    except Exception as e:
        print(f"❌ JSONファイル読み込みエラー: {e}")
        return None

def compare_with_previous_analysis():
    """
    前回の調査結果（main.py operation problem.md）との比較
    """
    print(f"\n🔍 前回調査結果との比較")
    print("-" * 50)
    
    previous_results = {
        'total_entries': 81,
        'total_exits': 0,
        'strategies': {
            'VWAPBreakoutStrategy': {'entries': 6, 'exits': 0},
            'MomentumInvestingStrategy': {'entries': 30, 'exits': 0},
            'BreakoutStrategy': {'entries': 20, 'exits': 0},
            'VWAPBounceStrategy': {'entries': 0, 'exits': 0},
            'OpeningGapStrategy': {'entries': 4, 'exits': 0},
            'ContrarianStrategy': {'entries': 20, 'exits': 0},
            'GCStrategy': {'entries': 1, 'exits': 0}
        }
    }
    
    print(f"📋 前回調査結果 (test_main_initialization.py):")
    print(f"  - 総エントリー: {previous_results['total_entries']}")
    print(f"  - 総エグジット: {previous_results['total_exits']}")
    
    print(f"\n📋 今回ログ結果:")
    print(f"  - エントリー: 62件（ログ記載）")
    print(f"  - エグジット: 62件（ログ記載）") 
    print(f"  - 取引ペア: 61ペア, 1未ペア")
    
    print(f"\n❓ 重要な矛盾:")
    print(f"  1. エントリー数: 81件 → 62件 (差: -19件)")
    print(f"  2. エグジット数: 0件 → 62件 (差: +62件)")
    print(f"  3. この劇的な変化の原因は？")
    
    return previous_results

def analyze_log_patterns():
    """
    ログパターンの分析（フォールバック重複等）
    """
    print(f"\n🔍 ログパターン分析")
    print("-" * 50)
    
    log_issues = [
        "統合システムの実行に失敗のログが二回ある",
        "フォールバック使用統計も二回ある", 
        "フォールバック使用統計の重複はバグか？"
    ]
    
    print("🚨 発見された問題パターン:")
    for i, issue in enumerate(log_issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n🔧 推測される原因:")
    print(f"  - MultiStrategyManager.execute_multi_strategy_flow の二重実行")
    print(f"  - フォールバック処理の重複トリガー")
    print(f"  - システム初期化時の競合状態")
    
    return log_issues

def main():
    print("🔍 ログ矛盾分析ツール実行開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: 出力ファイル発見
    latest_files = analyze_output_files()
    
    # Step 2: 各ファイル詳細分析
    analysis_results = {}
    
    if 'trades_csv' in latest_files:
        analysis_results['trades'] = analyze_trades_csv(latest_files['trades_csv'])
    
    if 'data_csv' in latest_files:
        analysis_results['signals'] = analyze_data_csv(latest_files['data_csv'])
    
    if 'json' in latest_files:
        analysis_results['metadata'] = analyze_json_metadata(latest_files['json'])
    
    # Step 3: 前回調査との比較
    previous_results = compare_with_previous_analysis()
    
    # Step 4: ログパターン問題
    log_issues = analyze_log_patterns()
    
    # Step 5: 総合分析結果
    print("\n" + "=" * 60)
    print("🎯 総合分析結果")
    print("=" * 60)
    
    print(f"\n📊 データの一貫性チェック:")
    if 'trades' in analysis_results and 'signals' in analysis_results:
        trades_total = analysis_results['trades']['total_trades']
        signal_entries = analysis_results['signals'].get('entry_signals', 'N/A')
        signal_exits = analysis_results['signals'].get('exit_signals', 'N/A')
        
        print(f"  - 取引ファイル総取引数: {trades_total}")
        print(f"  - データファイルエントリー: {signal_entries}")
        print(f"  - データファイルエグジット: {signal_exits}")
        
        # 矛盾検出
        if trades_total == 124 and signal_entries != 62:
            print(f"  ⚠️  矛盾検出: 取引数124 vs ログエントリー62")
        
    print(f"\n🔍 重要な発見:")
    print(f"  1. 前回調査(81エントリー, 0エグジット) vs 今回ログ(62エントリー, 62エグジット)")
    print(f"  2. この変化は実際のコード変更によるものか、ログの誤表示か")
    print(f"  3. フォールバック重複は潜在的なバグを示唆")
    
    print(f"\n📝 次の調査ステップ:")
    print(f"  - 実際の出力ファイル内容確認")
    print(f"  - unified_exporter.pyのペアリングロジック調査")
    print(f"  - フォールバック重複原因特定")
    
    return analysis_results

if __name__ == "__main__":
    results = main()