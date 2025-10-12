#!/usr/bin/env python3
"""
TODO-007 Phase 2: フォールバック処理と統合システム競合調査

緊急優先調査: 戦略レベル591件のExit_Signal=-1がmain.py統合処理で0件に消失する原因解明
MultiStrategyManagerとフォールバック処理の競合状態を詳細分析
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import subprocess
import re

# パス設定
sys.path.append('.')

def analyze_main_py_multistrategy_manager():
    """main.pyのMultiStrategyManager処理を詳細分析"""
    print("=== MultiStrategyManager処理分析開始 ===")
    
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # MultiStrategyManager関連の処理を検索
        multistrategy_patterns = [
            r'MultiStrategyManager',
            r'strategy_manager',
            r'from.*strategy_manager',
            r'統合システム',
            r'フォールバック',
            r'fallback',
            r'backup.*system'
        ]
        
        multistrategy_matches = []
        for i, pattern in enumerate(multistrategy_patterns):
            matches = re.finditer(pattern, main_content, re.IGNORECASE)
            for match in matches:
                lines = main_content[:match.start()].count('\n') + 1
                context_start = max(0, match.start() - 100)
                context_end = min(len(main_content), match.end() + 100)
                context = main_content[context_start:context_end]
                
                multistrategy_matches.append({
                    'pattern': pattern,
                    'line': lines,
                    'match': match.group(),
                    'context': context
                })
        
        print(f"MultiStrategyManager関連パターン発見: {len(multistrategy_matches)}件")
        
        for match in multistrategy_matches[:5]:  # 最初の5件を表示
            print(f"  Line {match['line']}: {match['match']} - {match['pattern']}")
        
        return multistrategy_matches
        
    except Exception as e:
        print(f"❌ main.py分析エラー: {e}")
        return []

def analyze_fallback_processing():
    """フォールバック処理の詳細分析"""
    print("\n=== フォールバック処理分析開始 ===")
    
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # フォールバック処理関連パターン
        fallback_patterns = [
            r'try:.*?except.*?:.*?fallback',
            r'backup.*system',
            r'emergency.*mode',
            r'統合システム.*失敗',
            r'個別戦略.*実行',
            r'_execute_individual_strategy',
            r'strategy\.backtest\(\)'
        ]
        
        fallback_matches = []
        for pattern in fallback_patterns:
            matches = re.finditer(pattern, main_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                lines = main_content[:match.start()].count('\n') + 1
                fallback_matches.append({
                    'pattern': pattern,
                    'line': lines,
                    'match': match.group()[:100] + "..." if len(match.group()) > 100 else match.group()
                })
        
        print(f"フォールバック処理パターン発見: {len(fallback_matches)}件")
        
        for match in fallback_matches:
            print(f"  Line {match['line']}: {match['match']}")
        
        return fallback_matches
        
    except Exception as e:
        print(f"❌ フォールバック処理分析エラー: {e}")
        return []

def analyze_signal_integration_processing():
    """シグナル統合処理の詳細分析"""
    print("\n=== シグナル統合処理分析開始 ===")
    
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # シグナル統合関連パターン
        signal_patterns = [
            r'Exit_Signal.*=.*-1',
            r'Exit_Signal.*=.*1',
            r'Exit_Signal.*abs\(',
            r'signal.*integration',
            r'シグナル.*統合',
            r'strategy_result.*Exit_Signal',
            r'result_data.*Exit_Signal',
            r'stock_data.*Exit_Signal'
        ]
        
        signal_matches = []
        for pattern in signal_patterns:
            matches = re.finditer(pattern, main_content, re.IGNORECASE)
            for match in matches:
                lines = main_content[:match.start()].count('\n') + 1
                context_start = max(0, match.start() - 50)
                context_end = min(len(main_content), match.end() + 50)
                context = main_content[context_start:context_end]
                
                signal_matches.append({
                    'pattern': pattern,
                    'line': lines,
                    'match': match.group(),
                    'context': context
                })
        
        print(f"シグナル統合処理パターン発見: {len(signal_matches)}件")
        
        for match in signal_matches:
            print(f"  Line {match['line']}: {match['match']}")
            print(f"    Context: {repr(match['context'])}")
        
        return signal_matches
        
    except Exception as e:
        print(f"❌ シグナル統合処理分析エラー: {e}")
        return []

def analyze_strategy_result_aggregation():
    """戦略結果集約処理の分析"""
    print("\n=== 戦略結果集約処理分析開始 ===")
    
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # 戦略結果集約関連パターン
        aggregation_patterns = [
            r'all_results.*append',
            r'strategy_results.*\[',
            r'combined.*result',
            r'merge.*strategy',
            r'concat.*strategy',
            r'aggregate.*signal',
            r'統合.*結果'
        ]
        
        aggregation_matches = []
        for pattern in aggregation_patterns:
            matches = re.finditer(pattern, main_content, re.IGNORECASE)
            for match in matches:
                lines = main_content[:match.start()].count('\n') + 1
                context_start = max(0, match.start() - 100)
                context_end = min(len(main_content), match.end() + 100)
                context = main_content[context_start:context_end]
                
                aggregation_matches.append({
                    'pattern': pattern,
                    'line': lines,
                    'match': match.group(),
                    'context': context
                })
        
        print(f"戦略結果集約処理パターン発見: {len(aggregation_matches)}件")
        
        for match in aggregation_matches:
            print(f"  Line {match['line']}: {match['match']}")
        
        return aggregation_matches
        
    except Exception as e:
        print(f"❌ 戦略結果集約処理分析エラー: {e}")
        return []

def run_main_py_with_debug_tracing():
    """main.pyをデバッグトレーシング付きで実行"""
    print("\n=== main.pyデバッグトレーシング実行開始 ===")
    
    try:
        # デバッグ用main.pyの一時コピー作成
        debug_main_content = """
import sys
import os
import traceback
from datetime import datetime

# デバッグ用ログ関数
def debug_log(message, data=None):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {message}")
    if data is not None:
        if hasattr(data, 'shape'):
            print(f"  Data shape: {data.shape}")
        if hasattr(data, 'Exit_Signal'):
            exit_signal_counts = data['Exit_Signal'].value_counts().sort_index()
            print(f"  Exit_Signal distribution: {exit_signal_counts.to_dict()}")
        if hasattr(data, 'columns'):
            print(f"  Columns: {list(data.columns)}")

# 元のmain.pyをインポート
sys.path.append('.')

# 戦略実行をフック
original_execute_individual_strategy = None

def debug_execute_individual_strategy(*args, **kwargs):
    debug_log("戦略実行開始", kwargs.get('strategy_name', 'unknown'))
    result = original_execute_individual_strategy(*args, **kwargs)
    debug_log(f"戦略実行完了: {kwargs.get('strategy_name', 'unknown')}", result)
    return result

# main.pyの主要関数をデバッグフック
if True:
    print("デバッグトレーシング：main.py実行開始")
    exec(open('main.py').read())
"""
        
        # デバッグmain.pyの実行
        result = subprocess.run(
            [sys.executable, '-c', debug_main_content],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"実行終了コード: {result.returncode}")
        
        if result.stdout:
            print("=== 標準出力 ===")
            stdout_lines = result.stdout.split('\n')
            # Exit_Signal関連のデバッグ情報を抽出
            exit_signal_lines = [line for line in stdout_lines if 'Exit_Signal' in line or 'DEBUG' in line]
            for line in exit_signal_lines[-20:]:  # 最後の20行
                print(line)
        
        if result.stderr:
            print("=== エラー出力 ===")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-10:]:  # 最後の10行
                print(line)
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"❌ デバッグトレーシング実行エラー: {e}")
        return None

def analyze_data_flow_consistency():
    """データフロー整合性の分析"""
    print("\n=== データフロー整合性分析開始 ===")
    
    # Phase 1の結果ファイルを確認
    phase1_files = list(Path('.').glob('todo_007_phase1_individual_strategy_analysis_*.json'))
    
    if not phase1_files:
        print("❌ Phase 1結果ファイルが見つかりません")
        return None
    
    latest_phase1_file = max(phase1_files, key=lambda x: x.stat().st_mtime)
    print(f"Phase 1結果ファイル: {latest_phase1_file}")
    
    try:
        with open(latest_phase1_file, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
        
        # Phase 1データの分析
        total_strategies = phase1_data.get('total_strategies_tested', 0)
        successful_strategies = phase1_data.get('successful_strategies', 0)
        total_exit_minus_one = phase1_data.get('total_exit_minus_one_generated', 0)
        
        print(f"Phase 1結果サマリー:")
        print(f"  調査戦略数: {total_strategies}")
        print(f"  成功戦略数: {successful_strategies}")
        print(f"  Exit_Signal=-1総数: {total_exit_minus_one}")
        
        # 戦略別詳細
        strategy_results = phase1_data.get('strategy_results', [])
        print(f"\n戦略別Exit_Signal=-1生成:")
        for result in strategy_results:
            strategy_name = result.get('strategy_name', 'unknown')
            exit_minus_one_count = result.get('exit_minus_one_count', 0)
            entry_signals = result.get('entry_signals', 0)
            print(f"  {strategy_name}: Entry={entry_signals}, Exit(-1)={exit_minus_one_count}")
        
        return phase1_data
        
    except Exception as e:
        print(f"❌ Phase 1データ分析エラー: {e}")
        return None

def main():
    """TODO-007 Phase 2: フォールバック処理と統合システム競合調査"""
    print("=== TODO-007 Phase 2: フォールバック処理と統合システム競合調査 ===")
    print("緊急優先調査: 戦略レベル591件 → main.py統合処理0件の消失原因解明")
    
    # Phase 2調査項目
    investigations = [
        ("MultiStrategyManager処理分析", analyze_main_py_multistrategy_manager),
        ("フォールバック処理分析", analyze_fallback_processing),
        ("シグナル統合処理分析", analyze_signal_integration_processing),
        ("戦略結果集約処理分析", analyze_strategy_result_aggregation),
        ("データフロー整合性分析", analyze_data_flow_consistency),
        ("main.pyデバッグトレーシング実行", run_main_py_with_debug_tracing)
    ]
    
    phase2_results = {
        'analysis_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'phase': 'TODO-007 Phase 2',
        'investigation_type': 'Fallback Processing and System Integration Competition Analysis',
        'investigations': {}
    }
    
    for name, func in investigations:
        print(f"\n{'='*60}")
        print(f"調査項目: {name}")
        print(f"{'='*60}")
        
        try:
            result = func()
            phase2_results['investigations'][name] = {
                'status': 'completed',
                'result': result
            }
            print(f"✅ {name} 完了")
        except Exception as e:
            print(f"❌ {name} エラー: {e}")
            phase2_results['investigations'][name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Phase 2結果サマリー
    print("\n" + "="*80)
    print("TODO-007 Phase 2 結果サマリー")
    print("="*80)
    
    completed_investigations = sum(1 for inv in phase2_results['investigations'].values() if inv['status'] == 'completed')
    total_investigations = len(investigations)
    
    print(f"完了調査項目: {completed_investigations}/{total_investigations}")
    
    # 重要発見の特定
    critical_findings = []
    
    # MultiStrategyManager問題の確認
    multistrategy_analysis = phase2_results['investigations'].get('MultiStrategyManager処理分析', {})
    if multistrategy_analysis.get('status') == 'completed':
        matches = multistrategy_analysis.get('result', [])
        if len(matches) > 0:
            critical_findings.append(f"MultiStrategyManager関連処理発見: {len(matches)}箇所")
        else:
            critical_findings.append("⚠️ MultiStrategyManager処理が見つからない - 実装されていない可能性")
    
    # シグナル統合処理問題の確認
    signal_analysis = phase2_results['investigations'].get('シグナル統合処理分析', {})
    if signal_analysis.get('status') == 'completed':
        matches = signal_analysis.get('result', [])
        exit_signal_matches = [m for m in matches if 'Exit_Signal' in m.get('match', '')]
        if len(exit_signal_matches) > 0:
            critical_findings.append(f"Exit_Signal処理箇所発見: {len(exit_signal_matches)}箇所")
            critical_findings.append("🚨 TODO-003で修正済みの箇所の再確認が必要")
    
    # データフロー整合性の確認
    dataflow_analysis = phase2_results['investigations'].get('データフロー整合性分析', {})
    if dataflow_analysis.get('status') == 'completed':
        phase1_data = dataflow_analysis.get('result')
        if phase1_data:
            total_exit_minus_one = phase1_data.get('total_exit_minus_one_generated', 0)
            if total_exit_minus_one > 0:
                critical_findings.append(f"Phase 1確認: 戦略レベルで{total_exit_minus_one}件のExit_Signal=-1生成")
                critical_findings.append("🔥 591件 → 0件の消失メカニズム特定が最優先")
    
    # 重要発見の表示
    if critical_findings:
        print("\n🔥 重要発見:")
        for finding in critical_findings:
            print(f"  {finding}")
    else:
        print("\n⚠️ 重要発見なし - より詳細な調査が必要")
    
    # Phase 3への移行判定
    if completed_investigations >= len(investigations) * 0.7:  # 70%以上完了
        print("\n✅ Phase 2調査完了 - Phase 3への移行準備完了")
        phase2_results['next_phase_ready'] = True
    else:
        print("\n⚠️ Phase 2調査不完全 - 追加調査が必要")
        phase2_results['next_phase_ready'] = False
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"todo_007_phase2_fallback_integration_analysis_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(phase2_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nPhase 2調査結果保存: {results_file}")
    except Exception as e:
        print(f"❌ 結果保存エラー: {e}")
    
    print("TODO-007 Phase 2 完了")
    
    return phase2_results

if __name__ == "__main__":
    main()