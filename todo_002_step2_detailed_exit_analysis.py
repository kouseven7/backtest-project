#!/usr/bin/env python3
"""
TODO-002 Step2: 戦略ファイル詳細確認
Step1で「エグジットメソッド: 0個」という結果が出たが、
contrarian_strategyとgc_strategy_signalで「generate_exit_signal」が検出されている。
より詳細な解析を実行する。
"""

import os
import ast
import re

print("=" * 80)
print("🔍 TODO-002 Step2: 戦略ファイル詳細確認")
print("=" * 80)

def detailed_exit_analysis(file_path):
    """戦略ファイルの詳細なエグジットロジック解析"""
    print(f"\n📁 詳細分析: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ ファイル存在せず: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return None
    
    # 1. ASTによる正確なメソッド検出
    try:
        tree = ast.parse(content)
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'exit' in node.name.lower():
                    methods.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
        
        print(f"🔧 **正確なエグジットメソッド検出**: {len(methods)}個")
        for method in methods:
            print(f"  - {method['name']}() (line {method['line']}) args: {method['args']}")
    
    except Exception as e:
        print(f"⚠️ AST解析エラー: {e}")
    
    # 2. backtest()メソッド内のExit_Signal処理確認
    backtest_pattern = r'def backtest\(.*?\):(.*?)(?=def|\Z)'
    backtest_matches = re.findall(backtest_pattern, content, re.DOTALL)
    
    if backtest_matches:
        backtest_content = backtest_matches[0]
        print(f"\n📊 **backtest()メソッド内のExit_Signal処理**:")
        
        # Exit_Signal代入の検索
        exit_assignments = re.findall(r'(.*Exit_Signal.*=.*)', backtest_content)
        print(f"Exit_Signal代入箇所: {len(exit_assignments)}件")
        for i, assignment in enumerate(exit_assignments[:5], 1):
            print(f"  {i}. {assignment.strip()}")
        
        # 条件分岐でのExit_Signal処理
        if_patterns = re.findall(r'(if.*?Exit_Signal.*?:.*)', backtest_content, re.DOTALL)
        print(f"Exit_Signal条件分岐: {len(if_patterns)}件")
        for i, pattern in enumerate(if_patterns[:3], 1):
            clean_pattern = ' '.join(pattern.split())[:80]
            print(f"  {i}. {clean_pattern}...")
    
    # 3. generate_exit_signal()メソッドの詳細解析
    generate_exit_pattern = r'def generate_exit_signal\(.*?\):(.*?)(?=def|\Z)'
    generate_matches = re.findall(generate_exit_pattern, content, re.DOTALL)
    
    if generate_matches:
        generate_content = generate_matches[0]
        print(f"\n🎯 **generate_exit_signal()メソッド解析**:")
        
        # return文の検索
        return_statements = re.findall(r'return\s+(.+)', generate_content)
        print(f"return文: {len(return_statements)}件")
        for i, ret in enumerate(return_statements, 1):
            print(f"  {i}. return {ret.strip()}")
        
        # 条件判定の検索
        conditions = re.findall(r'if\s+(.+?):', generate_content)
        print(f"条件判定: {len(conditions)}件")
        for i, condition in enumerate(conditions[:3], 1):
            print(f"  {i}. if {condition.strip()}")
    
    # 4. TODO-001類似問題の検証（-1 vs 1の使用状況）
    print(f"\n🚨 **TODO-001類似問題検証**:")
    
    # -1の使用箇所
    negative_one_usage = re.findall(r'(.*-1.*)', content)
    negative_one_exit = [usage for usage in negative_one_usage if 'exit' in usage.lower() or 'Exit_Signal' in usage]
    
    print(f"'-1'使用箇所（エグジット関連）: {len(negative_one_exit)}件")
    for i, usage in enumerate(negative_one_exit[:3], 1):
        print(f"  {i}. {usage.strip()}")
    
    # 1の使用箇所
    positive_one_usage = re.findall(r'(.*Exit_Signal.*=.*1.*)', content)
    print(f"'Exit_Signal = 1'使用箇所: {len(positive_one_usage)}件")
    for i, usage in enumerate(positive_one_usage[:3], 1):
        print(f"  {i}. {usage.strip()}")
    
    return {
        'methods_count': len(methods) if 'methods' in locals() else 0,
        'backtest_exit_assignments': len(exit_assignments) if 'exit_assignments' in locals() else 0,
        'generate_exit_returns': len(return_statements) if 'return_statements' in locals() else 0,
        'negative_one_usage': len(negative_one_exit),
        'positive_one_usage': len(positive_one_usage)
    }

# 重点分析対象（Step1で疑わしい結果が出た戦略）
focus_strategies = [
    "strategies/contrarian_strategy.py",  # generate_exit_signalが検出済み
    "strategies/gc_strategy_signal.py",   # generate_exit_signalが検出済み
    "strategies/VWAP_Breakout.py",        # Exit_Signal更新6箇所
    "strategies/Momentum_Investing.py"    # Exit_Signal更新6箇所、シグナル値多用
]

print("🔍 **重点戦略の詳細解析開始**")
detailed_results = {}

for strategy_file in focus_strategies:
    result = detailed_exit_analysis(strategy_file)
    if result:
        detailed_results[strategy_file] = result

print("\n" + "=" * 80)
print("📊 **詳細解析結果サマリー**")
print("=" * 80)

for file_path, result in detailed_results.items():
    strategy_name = file_path.split('/')[-1].replace('.py', '')
    print(f"\n📋 **{strategy_name}**:")
    print(f"  - エグジットメソッド: {result['methods_count']}個")
    print(f"  - backtest()内Exit_Signal代入: {result['backtest_exit_assignments']}件")
    print(f"  - generate_exit_signal()return文: {result['generate_exit_returns']}件")
    print(f"  - '-1'エグジット関連使用: {result['negative_one_usage']}件")
    print(f"  - 'Exit_Signal=1'使用: {result['positive_one_usage']}件")

print("\n🎯 **重要な発見**:")

# TODO-001類似問題の確認
total_negative_usage = sum(r['negative_one_usage'] for r in detailed_results.values())
total_positive_usage = sum(r['positive_one_usage'] for r in detailed_results.values())

if total_negative_usage > 0:
    print(f"⚠️  **-1使用戦略**: {total_negative_usage}件のエグジット関連-1使用を確認")
    print("   → TODO-001と同様のExit_Signal: -1 → 1 変換問題の可能性")

if total_positive_usage > 0:
    print(f"📊 **Exit_Signal=1直接設定**: {total_positive_usage}件確認")
    print("   → 戦略レベルで既に1を設定している可能性")

# Step1の「エグジットメソッド: 0個」結果の検証  
total_methods = sum(r['methods_count'] for r in detailed_results.values())
if total_methods > 0:
    print(f"🔧 **メソッド検出修正**: 実際は{total_methods}個のエグジットメソッドが存在")
    print("   → Step1の正規表現パターンが不完全だった")

print("\n✅ TODO-002 Step2 詳細確認完了")
print("📋 次: TODO-002 Step3で個別戦略のシグナル変換問題を検証")