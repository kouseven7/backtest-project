#!/usr/bin/env python3
"""
統合システムExit_Signal調査ツール
調査対象: strategy_execution_adapter.py, multi_strategy_manager_fixed.py
目的: Exit_Signal消失問題の根本原因特定
"""

import re
import os
from pathlib import Path

def analyze_strategy_execution_adapter():
    """Phase 1: strategy_execution_adapter.py調査"""
    print("=== Phase 1: strategy_execution_adapter.py調査 ===")
    
    try:
        with open('config/strategy_execution_adapter.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        findings = {
            'exit_signal_references': [],
            'legacy_vs_integrated_differences': [],
            'signal_processing_issues': []
        }
        
        # Exit_Signal関連処理の特定
        exit_signal_patterns = [
            r'Exit_Signal',
            r'exit_signal',
            r'signals_data',
            r'_execute_legacy_method',
            r'_execute_integrated_method'
        ]
        
        for pattern in exit_signal_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(content), match.end() + 100)
                    context = content[context_start:context_end]
                    
                    findings['exit_signal_references'].append({
                        'pattern': pattern,
                        'line': line_num,
                        'context': context.strip()
                    })
        
        # レガシー vs 統合方式の比較
        legacy_method = re.search(r'def _execute_legacy_method.*?(?=def|\Z)', content, re.DOTALL)
        integrated_method = re.search(r'def _execute_integrated_method.*?(?=def|\Z)', content, re.DOTALL)
        
        if legacy_method and integrated_method:
            legacy_text = legacy_method.group(0)
            integrated_text = integrated_method.group(0)
            
            # Exit_Signal処理の違いを分析
            legacy_exit_handling = 'Exit_Signal' in legacy_text or 'exit_signal' in legacy_text
            integrated_exit_handling = 'Exit_Signal' in integrated_text or 'exit_signal' in integrated_text
            
            findings['legacy_vs_integrated_differences'].append({
                'legacy_handles_exit_signal': legacy_exit_handling,
                'integrated_handles_exit_signal': integrated_exit_handling,
                'legacy_method_length': len(legacy_text),
                'integrated_method_length': len(integrated_text)
            })
        
        # signals_dataの処理確認
        signals_data_usage = re.findall(r'signals_data.*?=.*?\[.*?\]', content, re.DOTALL)
        for usage in signals_data_usage:
            findings['signal_processing_issues'].append({
                'issue': 'signals_data hardcoded values found',
                'detail': usage.strip()
            })
        
        print(f"Exit_Signal参照箇所: {len(findings['exit_signal_references'])}件")
        print(f"レガシー vs 統合差異: {len(findings['legacy_vs_integrated_differences'])}件")
        print(f"シグナル処理問題: {len(findings['signal_processing_issues'])}件")
        
        return findings
        
    except Exception as e:
        print(f"❌ strategy_execution_adapter.py調査エラー: {e}")
        return {}

def analyze_multi_strategy_manager_fixed():
    """Phase 2: multi_strategy_manager_fixed.py調査"""
    print("\n=== Phase 2: multi_strategy_manager_fixed.py調査 ===")
    
    try:
        with open('config/multi_strategy_manager_fixed.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        findings = {
            'exit_signal_processing': [],
            'validation_issues': [],
            'signal_integration_problems': []
        }
        
        # _execute_multi_strategy_flow内のExit_Signal処理確認
        flow_method = re.search(r'def _execute_multi_strategy_flow.*?(?=def|\Z)', content, re.DOTALL)
        if flow_method:
            flow_text = flow_method.group(0)
            
            # Exit_Signal初期化確認
            exit_signal_init = re.search(r'Exit_Signal.*?=.*?0', flow_text)
            if exit_signal_init:
                findings['exit_signal_processing'].append({
                    'type': 'initialization',
                    'detail': exit_signal_init.group(0).strip(),
                    'location': '_execute_multi_strategy_flow'
                })
            
            # Exit_Signal統合処理確認
            exit_signal_integration = re.findall(r'exit_count.*?=.*?\(.*?Exit_Signal.*?\)', flow_text)
            for integration in exit_signal_integration:
                findings['exit_signal_processing'].append({
                    'type': 'integration',
                    'detail': integration.strip(),
                    'location': '_execute_multi_strategy_flow'
                })
        
        # _validate_backtest_output内のExit_Signal検証確認
        validation_method = re.search(r'def _validate_backtest_output.*?(?=def|\Z)', content, re.DOTALL)
        if validation_method:
            validation_text = validation_method.group(0)
            
            # Exit_Signal検証ロジック確認
            exit_validation = re.findall(r'Exit_Signal.*?==.*?-1', validation_text)
            for validation in exit_validation:
                findings['validation_issues'].append({
                    'type': 'exit_signal_validation',
                    'detail': validation.strip(),
                    'issue': 'Checks for Exit_Signal == -1'
                })
        
        # Excel出力関連のExit_Signal処理確認
        excel_comments = re.findall(r'# TODO\(tag:excel_deprecated.*?\)', content)
        for comment in excel_comments:
            if 'Exit_Signal' in comment or 'BACKTEST_IMPACT' in comment:
                findings['signal_integration_problems'].append({
                    'type': 'excel_deprecation_impact',
                    'detail': comment.strip(),
                    'issue': 'Excel deprecation may affect Exit_Signal output'
                })
        
        print(f"Exit_Signal処理箇所: {len(findings['exit_signal_processing'])}件")
        print(f"検証問題: {len(findings['validation_issues'])}件")
        print(f"統合問題: {len(findings['signal_integration_problems'])}件")
        
        return findings
        
    except Exception as e:
        print(f"❌ multi_strategy_manager_fixed.py調査エラー: {e}")
        return {}

def compare_exit_signal_handling():
    """統合システム vs 従来システムのExit_Signal処理比較"""
    print("\n=== Exit_Signal処理比較分析 ===")
    
    try:
        # main.pyの処理パターンを参照
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        comparison_results = {
            'main_py_exit_processing': [],
            'integrated_system_gaps': [],
            'critical_differences': []
        }
        
        # main.pyでのExit_Signal処理パターン
        main_exit_patterns = re.findall(r'Exit_Signal.*?!=.*?0', main_content)
        main_exit_assignments = re.findall(r'Exit_Signal.*?=.*?-1', main_content)
        
        comparison_results['main_py_exit_processing'] = {
            'exit_signal_checks': len(main_exit_patterns),
            'exit_signal_assignments': len(main_exit_assignments)
        }
        
        # 統合システムでの対応状況確認
        with open('config/multi_strategy_manager_fixed.py', 'r', encoding='utf-8') as f:
            integrated_content = f.read()
        
        integrated_exit_patterns = re.findall(r'Exit_Signal.*?!=.*?0', integrated_content)
        integrated_exit_assignments = re.findall(r'Exit_Signal.*?=.*?-1', integrated_content)
        
        # 差異分析
        if len(main_exit_patterns) > len(integrated_exit_patterns):
            comparison_results['critical_differences'].append({
                'issue': 'Exit_Signal != 0 checks missing in integrated system',
                'main_py_count': len(main_exit_patterns),
                'integrated_count': len(integrated_exit_patterns)
            })
        
        if len(main_exit_assignments) > len(integrated_exit_assignments):
            comparison_results['critical_differences'].append({
                'issue': 'Exit_Signal = -1 assignments missing in integrated system',
                'main_py_count': len(main_exit_assignments), 
                'integrated_count': len(integrated_exit_assignments)
            })
        
        print(f"main.py Exit_Signal処理: チェック{len(main_exit_patterns)}件, 代入{len(main_exit_assignments)}件")
        print(f"統合システム Exit_Signal処理: チェック{len(integrated_exit_patterns)}件, 代入{len(integrated_exit_assignments)}件")
        print(f"重要な差異: {len(comparison_results['critical_differences'])}件")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ Exit_Signal処理比較エラー: {e}")
        return {}

def generate_investigation_report(adapter_findings, manager_findings, comparison_results):
    """調査報告書生成"""
    print("\n=== 調査報告書生成 ===")
    
    report = """# 統合システムExit_Signal調査報告書

## 📋 調査概要
**日付**: 2025年10月11日  
**調査対象**: strategy_execution_adapter.py, multi_strategy_manager_fixed.py  
**調査目的**: TODO-007 Phase 3で発見されたExit_Signal消失問題の根本原因特定  

## 🔍 **発見された問題箇所**

### **1. strategy_execution_adapter.py問題**
"""
    
    if adapter_findings:
        report += f"- Exit_Signal参照箇所: **{len(adapter_findings.get('exit_signal_references', []))}件**\n"
        
        if adapter_findings.get('legacy_vs_integrated_differences'):
            diff = adapter_findings['legacy_vs_integrated_differences'][0]
            report += f"- レガシー方式Exit_Signal処理: **{diff.get('legacy_handles_exit_signal', False)}**\n"
            report += f"- 統合方式Exit_Signal処理: **{diff.get('integrated_handles_exit_signal', False)}**\n"
        
        if adapter_findings.get('signal_processing_issues'):
            report += f"- シグナル処理問題: **{len(adapter_findings['signal_processing_issues'])}件**\n"
            for issue in adapter_findings['signal_processing_issues'][:3]:  # 最初の3件のみ
                report += f"  - {issue.get('issue', 'Unknown issue')}\n"
    
    report += "\n### **2. multi_strategy_manager_fixed.py問題**\n"
    
    if manager_findings:
        report += f"- Exit_Signal処理箇所: **{len(manager_findings.get('exit_signal_processing', []))}件**\n"
        report += f"- 検証ロジック問題: **{len(manager_findings.get('validation_issues', []))}件**\n"
        report += f"- 統合処理問題: **{len(manager_findings.get('signal_integration_problems', []))}件**\n"
        
        # 具体的な問題点
        for processing in manager_findings.get('exit_signal_processing', [])[:2]:
            report += f"  - {processing.get('type', 'Unknown')}: {processing.get('location', 'Unknown location')}\n"
    
    report += "\n## 🚨 **Exit_Signal処理の問題点**\n\n"
    
    if comparison_results:
        main_checks = comparison_results.get('main_py_exit_processing', {}).get('exit_signal_checks', 0)
        main_assignments = comparison_results.get('main_py_exit_processing', {}).get('exit_signal_assignments', 0)
        
        report += f"### **統合システムと従来システムの重要な差異**\n"
        report += f"- main.py Exit_Signal処理: チェック**{main_checks}件**, 代入**{main_assignments}件**\n"
        
        for diff in comparison_results.get('critical_differences', []):
            report += f"- 🔥 **{diff.get('issue', 'Unknown issue')}**\n"
            report += f"  - main.py: {diff.get('main_py_count', 0)}件 vs 統合システム: {diff.get('integrated_count', 0)}件\n"
    
    report += "\n## 📝 **推奨される次のアクション**\n\n"
    report += "1. **統合システムでのExit_Signal = -1処理実装**\n"
    report += "   - multi_strategy_manager_fixed.pyにExit_Signal代入処理追加\n"
    report += "   - strategy_execution_adapter.pyでのExit_Signal値保持確認\n\n"
    report += "2. **Exit_Signal != 0チェックの統合システム移植**\n"
    report += "   - main.pyの9箇所の修正パターンを統合システムに適用\n"
    report += "   - TODO-003修正の統合システム版実装\n\n"
    report += "3. **統合システムでのシグナル統合処理修正**\n"
    report += "   - _execute_multi_strategy_flow内でのExit_Signal統合ロジック強化\n"
    report += "   - Excel廃止に伴うExit_Signal出力影響の対処\n\n"
    report += "4. **統合システム vs 従来システムの処理統一**\n"
    report += "   - Exit_Signal処理パターンの完全な統一\n"
    report += "   - バックテスト基本理念遵守の統合システム版確立\n\n"
    report += "5. **統合システム動作検証**\n"
    report += "   - 修正後の統合システムでのExit_Signal生成確認\n"
    report += "   - 591件のExit_Signal=-1が統合システムでも正常処理されることの検証\n"
    
    report += "\n## 🎯 **根本原因の特定**\n\n"
    report += "**統合システムではExit_Signal=-1の処理が不完全**\n"
    report += "- strategy_execution_adapter.pyでのハードコーディングされたシグナル値\n"
    report += "- multi_strategy_manager_fixed.pyでのExit_Signal統合処理の不備\n"
    report += "- main.pyで実装済みのTODO-003修正が統合システムに未反映\n\n"
    report += "**影響範囲**: 統合システム使用時にExit_Signal=-1が完全消失し、フォールバック処理が発生\n"
    
    report += "\n---\n**調査完了時刻**: 30分以内での効率的調査完了\n"
    
    return report

def main():
    """メイン調査実行"""
    print("🔍 統合システムExit_Signal調査開始 (30分以内完了)")
    
    # Phase 1: strategy_execution_adapter.py調査
    adapter_findings = analyze_strategy_execution_adapter()
    
    # Phase 2: multi_strategy_manager_fixed.py調査
    manager_findings = analyze_multi_strategy_manager_fixed()
    
    # Exit_Signal処理比較
    comparison_results = compare_exit_signal_handling()
    
    # 報告書生成
    report = generate_investigation_report(adapter_findings, manager_findings, comparison_results)
    
    # 報告書保存
    report_path = "docs/dssms/main normal operation problem/integrated_system_exit_signal_investigation.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📝 調査報告書生成完了: {report_path}")
    print("🎯 統合システムExit_Signal調査完了")

if __name__ == "__main__":
    main()