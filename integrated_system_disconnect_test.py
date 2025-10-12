#!/usr/bin/env python3
"""
統合システム切り離しテスト
目的: strategy_execution_adapter.py と multi_strategy_manager_fixed.py を切り離してmain.py実行
影響測定: Exit_Signal生成への影響確認
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
import json

def backup_integrated_system_files():
    """統合システムファイルをバックアップ"""
    print("=== 統合システムファイルバックアップ開始 ===")
    
    files_to_backup = [
        'config/strategy_execution_adapter.py',
        'config/multi_strategy_manager_fixed.py'
    ]
    
    backup_info = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{timestamp}"
            shutil.copy2(file_path, backup_path)
            backup_info[file_path] = backup_path
            print(f"✅ バックアップ完了: {file_path} -> {backup_path}")
        else:
            print(f"❌ ファイルが見つかりません: {file_path}")
    
    return backup_info

def temporarily_disable_integrated_system():
    """統合システムを一時的に無効化"""
    print("\n=== 統合システム一時無効化開始 ===")
    
    # strategy_execution_adapter.py を一時的にリネーム
    if os.path.exists('config/strategy_execution_adapter.py'):
        os.rename('config/strategy_execution_adapter.py', 'config/strategy_execution_adapter.py.disabled')
        print("✅ strategy_execution_adapter.py を無効化")
    
    # multi_strategy_manager_fixed.py を一時的にリネーム
    if os.path.exists('config/multi_strategy_manager_fixed.py'):
        os.rename('config/multi_strategy_manager_fixed.py', 'config/multi_strategy_manager_fixed.py.disabled')
        print("✅ multi_strategy_manager_fixed.py を無効化")
    
    print("統合システム無効化完了")

def run_main_py_without_integrated_system():
    """統合システムなしでmain.py実行"""
    print("\n=== 統合システムなしでmain.py実行開始 ===")
    
    try:
        # main.py実行
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True,
            text=True,
            timeout=120,  # 2分でタイムアウト
            cwd='.'
        )
        
        print(f"実行終了コード: {result.returncode}")
        
        # 標準出力から重要な情報を抽出
        stdout_lines = result.stdout.split('\n') if result.stdout else []
        stderr_lines = result.stderr.split('\n') if result.stderr else []
        
        # Exit_Signal関連の出力を検索
        exit_signal_lines = []
        entry_signal_lines = []
        error_lines = []
        
        for line in stdout_lines + stderr_lines:
            if any(keyword in line.lower() for keyword in ['exit_signal', 'exit signal', 'exits']):
                exit_signal_lines.append(line)
            if any(keyword in line.lower() for keyword in ['entry_signal', 'entry signal', 'entries']):
                entry_signal_lines.append(line)
            if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception']):
                error_lines.append(line)
        
        analysis_result = {
            'execution_success': result.returncode == 0,
            'exit_signal_mentions': len(exit_signal_lines),
            'entry_signal_mentions': len(entry_signal_lines),
            'error_count': len(error_lines),
            'exit_signal_lines': exit_signal_lines[:10],  # 最初の10行
            'entry_signal_lines': entry_signal_lines[:10],
            'error_lines': error_lines[:5],  # 最初の5行
            'stdout_length': len(result.stdout) if result.stdout else 0,
            'stderr_length': len(result.stderr) if result.stderr else 0
        }
        
        print(f"実行結果:")
        print(f"  成功: {analysis_result['execution_success']}")
        print(f"  Exit_Signal言及: {analysis_result['exit_signal_mentions']}回")
        print(f"  Entry_Signal言及: {analysis_result['entry_signal_mentions']}回")
        print(f"  エラー数: {analysis_result['error_count']}件")
        
        if exit_signal_lines:
            print(f"\nExit_Signal関連出力（最初の5行）:")
            for line in exit_signal_lines[:5]:
                print(f"  {line}")
        
        if error_lines:
            print(f"\nエラー関連出力（最初の3行）:")
            for line in error_lines[:3]:
                print(f"  {line}")
        
        return analysis_result
        
    except subprocess.TimeoutExpired:
        print("❌ main.py実行がタイムアウトしました（2分）")
        return {'execution_success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"❌ main.py実行エラー: {e}")
        return {'execution_success': False, 'error': str(e)}

def check_output_files():
    """出力ファイルの確認"""
    print("\n=== 出力ファイル確認開始 ===")
    
    output_patterns = [
        'backtest_results_*.csv',
        'backtest_results_*.json',
        'backtest_results_*.txt',
        'output/*.csv',
        'output/*.json'
    ]
    
    found_files = []
    
    import glob
    for pattern in output_patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    output_analysis = {
        'files_generated': len(found_files),
        'file_list': found_files[:10],  # 最初の10ファイル
        'csv_files': [f for f in found_files if f.endswith('.csv')],
        'json_files': [f for f in found_files if f.endswith('.json')]
    }
    
    print(f"生成されたファイル数: {output_analysis['files_generated']}")
    
    # 最新のCSVファイルがあれば簡単にチェック
    csv_files = output_analysis['csv_files']
    if csv_files:
        latest_csv = max(csv_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
        print(f"最新CSVファイル: {latest_csv}")
        
        try:
            import pandas as pd
            df = pd.read_csv(latest_csv)
            
            exit_signal_analysis = {}
            if 'Exit_Signal' in df.columns:
                exit_signal_counts = df['Exit_Signal'].value_counts()
                exit_signal_analysis = {
                    'exit_signal_column_exists': True,
                    'exit_signal_distribution': exit_signal_counts.to_dict(),
                    'exit_signal_minus_one_count': (df['Exit_Signal'] == -1).sum(),
                    'exit_signal_one_count': (df['Exit_Signal'] == 1).sum(),
                    'exit_signal_zero_count': (df['Exit_Signal'] == 0).sum()
                }
            else:
                exit_signal_analysis['exit_signal_column_exists'] = False
            
            print(f"Exit_Signal分析:")
            print(f"  Exit_Signal列存在: {exit_signal_analysis.get('exit_signal_column_exists', False)}")
            if exit_signal_analysis.get('exit_signal_column_exists'):
                print(f"  Exit_Signal=-1: {exit_signal_analysis.get('exit_signal_minus_one_count', 0)}件")
                print(f"  Exit_Signal=1: {exit_signal_analysis.get('exit_signal_one_count', 0)}件")
                print(f"  Exit_Signal=0: {exit_signal_analysis.get('exit_signal_zero_count', 0)}件")
            
            output_analysis.update(exit_signal_analysis)
            
        except Exception as e:
            print(f"CSVファイル分析エラー: {e}")
            output_analysis['csv_analysis_error'] = str(e)
    
    return output_analysis

def restore_integrated_system():
    """統合システムを復元"""
    print("\n=== 統合システム復元開始 ===")
    
    # 無効化したファイルを復元
    if os.path.exists('config/strategy_execution_adapter.py.disabled'):
        os.rename('config/strategy_execution_adapter.py.disabled', 'config/strategy_execution_adapter.py')
        print("✅ strategy_execution_adapter.py を復元")
    
    if os.path.exists('config/multi_strategy_manager_fixed.py.disabled'):
        os.rename('config/multi_strategy_manager_fixed.py.disabled', 'config/multi_strategy_manager_fixed.py')
        print("✅ multi_strategy_manager_fixed.py を復元")
    
    print("統合システム復元完了")

def generate_impact_assessment_report(execution_result, output_analysis, backup_info):
    """影響評価レポート生成"""
    print("\n=== 影響評価レポート生成開始 ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = f"""# 統合システム切り離し影響評価レポート

## 📋 テスト概要
**日付**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**目的**: strategy_execution_adapter.py と multi_strategy_manager_fixed.py を切り離してmain.py実行  
**影響測定**: Exit_Signal生成への影響確認  

## 🔧 実行結果

### **main.py実行状況**
- **実行成功**: {execution_result.get('execution_success', False)}
- **Exit_Signal言及**: {execution_result.get('exit_signal_mentions', 0)}回
- **Entry_Signal言及**: {execution_result.get('entry_signal_mentions', 0)}回
- **エラー数**: {execution_result.get('error_count', 0)}件

"""
    
    if execution_result.get('exit_signal_lines'):
        report += "### **Exit_Signal関連出力**\n```\n"
        for line in execution_result['exit_signal_lines'][:5]:
            report += f"{line}\n"
        report += "```\n\n"
    
    if execution_result.get('error_lines'):
        report += "### **エラー関連出力**\n```\n"
        for line in execution_result['error_lines'][:3]:
            report += f"{line}\n"
        report += "```\n\n"
    
    report += f"""## 📊 出力ファイル分析

### **生成されたファイル**
- **ファイル数**: {output_analysis.get('files_generated', 0)}件
- **CSVファイル**: {len(output_analysis.get('csv_files', []))}件
- **JSONファイル**: {len(output_analysis.get('json_files', []))}件

"""
    
    if output_analysis.get('exit_signal_column_exists'):
        report += f"""### **Exit_Signal分析結果**
- **Exit_Signal列存在**: ✅ あり
- **Exit_Signal=-1**: {output_analysis.get('exit_signal_minus_one_count', 0)}件
- **Exit_Signal=1**: {output_analysis.get('exit_signal_one_count', 0)}件
- **Exit_Signal=0**: {output_analysis.get('exit_signal_zero_count', 0)}件

"""
    else:
        report += "### **Exit_Signal分析結果**\n- **Exit_Signal列存在**: ❌ なし\n\n"
    
    # 統合システムありなしの比較（既知の情報から）
    report += """## 🔍 **統合システム影響評価**

### **統合システムあり（従来）**
- main.py実行時に統合システムインポートエラー → フォールバック処理発生
- MultiStrategyManager失敗 → 従来システム使用
- 結果: Entry_Signal=62件, Exit_Signal=62件（同一行で同時発生）

### **統合システムなし（今回テスト）**"""
    
    if execution_result.get('execution_success'):
        report += f"""
- main.py実行成功
- Exit_Signal処理: 従来システムのみで実行
- Exit_Signal言及数: {execution_result.get('exit_signal_mentions', 0)}回
"""
        
        if output_analysis.get('exit_signal_minus_one_count', 0) > 0:
            report += f"- 🎯 **重要**: Exit_Signal=-1が{output_analysis.get('exit_signal_minus_one_count', 0)}件生成された！\n"
        else:
            report += "- Exit_Signal=-1生成: 確認できず\n"
    else:
        report += """
- main.py実行失敗
- 統合システム切り離しによる副作用の可能性
"""
    
    report += """
## 📝 **結論と推奨アクション**

"""
    
    # 結論の生成
    if execution_result.get('execution_success') and output_analysis.get('exit_signal_minus_one_count', 0) > 0:
        report += """### **🎯 重要な発見**
- **統合システム切り離しによりExit_Signal=-1が正常生成される可能性**
- 統合システムがExit_Signal消失の原因である可能性が高い

### **推奨アクション**
1. **統合システムのExit_Signal処理修正**: multi_strategy_manager_fixed.py内の処理改善
2. **strategy_execution_adapter.pyのシグナル処理確認**: ハードコーディング値の修正
3. **統合システム使用時のExit_Signal保持機能実装**
4. **フォールバック処理の改善**: 統合システム失敗時の適切な処理"""
    else:
        report += """### **結論**
- 統合システム切り離しでも問題が解決しない場合は、より深い根本原因が存在
- TODO-007 Phase 3で発見されたmain.pyの9箇所修正が依然として必要

### **推奨アクション**
1. **TODO-003修正の完全実装**: main.pyの9箇所のExit_Signal == 1 → != 0修正
2. **戦略レベルでのExit_Signal生成確認**: 個別戦略でのExit_Signal=-1生成検証
3. **unified_exporter入力データ検証**: 最終出力前のデータ整合性確認"""
    
    report += f"""

## 📋 **テスト環境情報**
- **バックアップファイル**: {len(backup_info)}件作成
- **テスト時刻**: {timestamp}
- **復元状況**: 統合システムファイル復元済み

---
**重要**: このテストにより統合システムのExit_Signal処理への影響が測定されました。
"""
    
    return report

def main():
    """メイン実行"""
    print("🔧 統合システム切り離しテスト開始")
    print("目的: strategy_execution_adapter.py と multi_strategy_manager_fixed.py を切り離してmain.py実行")
    
    try:
        # Step 1: バックアップ
        backup_info = backup_integrated_system_files()
        
        # Step 2: 統合システム無効化
        temporarily_disable_integrated_system()
        
        # Step 3: main.py実行
        execution_result = run_main_py_without_integrated_system()
        
        # Step 4: 出力ファイル確認
        output_analysis = check_output_files()
        
        # Step 5: 統合システム復元
        restore_integrated_system()
        
        # Step 6: レポート生成
        report = generate_impact_assessment_report(execution_result, output_analysis, backup_info)
        
        # レポート保存
        report_path = f"docs/dssms/main normal operation problem/integrated_system_disconnect_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📝 影響評価レポート生成完了: {report_path}")
        print("🎯 統合システム切り離しテスト完了")
        
        # 重要な発見があれば強調表示
        if output_analysis.get('exit_signal_minus_one_count', 0) > 0:
            print(f"\n🔥 重要な発見: Exit_Signal=-1が{output_analysis['exit_signal_minus_one_count']}件生成されました！")
            print("統合システムがExit_Signal消失の原因である可能性があります。")
        
        return {
            'execution_result': execution_result,
            'output_analysis': output_analysis,
            'report_path': report_path
        }
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        # エラーが発生した場合でも統合システムを復元
        try:
            restore_integrated_system()
            print("統合システム復元完了（エラー後）")
        except:
            print("❌ 統合システム復元に失敗しました")
        
        return {'error': str(e)}

if __name__ == "__main__":
    main()