#!/usr/bin/env python3
"""
TODO-002 Step1: 他6戦略のエグジット実装状況確認
- 各戦略ファイルでエグジットロジックの存在確認
- Exit_Signal列更新処理の実装状況調査
- TODO-001で発見されたExit_Signal: -1 → 1 変換問題の検証
"""

import os
import sys
import ast
import re

print("=" * 80)
print("🔍 TODO-002 Step1: 他6戦略エグジット実装状況確認")
print("=" * 80)

# 対象の6戦略
strategy_files = [
    "strategies/VWAP_Breakout.py",      # VWAPBreakoutStrategy
    "strategies/Momentum_Investing.py", # MomentumInvestingStrategy  
    "strategies/VWAP_Bounce.py",        # VWAPBounceStrategy
    "strategies/Opening_Gap.py",        # OpeningGapStrategy
    "strategies/contrarian_strategy.py", # ContrarianStrategy
    "strategies/gc_strategy_signal.py"  # GCStrategy
]

def analyze_exit_logic(file_path):
    """戦略ファイルのエグジットロジックを解析"""
    print(f"\n📁 分析対象: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ ファイルが存在しません: {file_path}")
        return {
            'file': file_path,
            'exists': False,
            'exit_methods': [],
            'exit_signal_updates': [],
            'signal_values': []
        }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return {
            'file': file_path,
            'exists': True,
            'exit_methods': [],
            'exit_signal_updates': [],
            'signal_values': [],
            'error': str(e)
        }
    
    # 解析結果格納
    result = {
        'file': file_path,
        'exists': True,
        'exit_methods': [],
        'exit_signal_updates': [],
        'signal_values': [],
        'has_backtest_method': False,
        'exit_logic_patterns': []
    }
    
    # 1. エグジット関連メソッドの検索
    exit_method_patterns = [
        r'def\s+.*exit.*\(.*\):',
        r'def\s+generate_exit_signal\(.*\):',
        r'def\s+check_exit.*\(.*\):',
        r'def\s+.*_exit\(.*\):'
    ]
    
    for pattern in exit_method_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        result['exit_methods'].extend(matches)
    
    # 2. Exit_Signal列更新の検索
    exit_signal_patterns = [
        r'Exit_Signal.*=.*(-?\d+)',
        r'data\[.*Exit_Signal.*\].*=.*(-?\d+)',
        r'df\[.*Exit_Signal.*\].*=.*(-?\d+)',
        r'\.loc\[.*Exit_Signal.*\].*=.*(-?\d+)'
    ]
    
    for pattern in exit_signal_patterns:
        matches = re.findall(pattern, content)
        result['exit_signal_updates'].extend(matches)
    
    # 3. シグナル値の検索（-1, 0, 1の使用状況）
    signal_value_patterns = [
        r'return\s+(-?\d+)',
        r'Exit_Signal.*=.*(-?\d+)',
        r'exit.*=.*(-?\d+)'
    ]
    
    for pattern in signal_value_patterns:
        matches = re.findall(pattern, content)
        result['signal_values'].extend(matches)
    
    # 4. backtest()メソッドの存在確認
    result['has_backtest_method'] = 'def backtest(' in content
    
    # 5. エグジットロジック全般の検索
    exit_logic_keywords = [
        'exit', 'profit', 'loss', 'stop', 'take_profit', 
        'stop_loss', 'close', 'sell', 'liquidate'
    ]
    
    for keyword in exit_logic_keywords:
        pattern = f'.*{keyword}.*'
        matches = re.findall(pattern, content, re.IGNORECASE)
        result['exit_logic_patterns'].extend([m for m in matches if len(m) < 100])
    
    # 結果表示
    print(f"✅ ファイル存在: {result['exists']}")
    print(f"📋 backtest()メソッド: {'✅ 存在' if result['has_backtest_method'] else '❌ 未実装'}")
    print(f"🔧 エグジットメソッド数: {len(result['exit_methods'])}")
    if result['exit_methods']:
        for method in result['exit_methods']:
            print(f"  - {method}")
    
    print(f"📊 Exit_Signal更新箇所: {len(result['exit_signal_updates'])}")
    if result['exit_signal_updates']:
        for update in result['exit_signal_updates']:
            print(f"  - Exit_Signal = {update}")
    
    print(f"🔢 シグナル値使用状況: {set(result['signal_values'])}")
    
    if len(result['exit_logic_patterns']) > 0:
        print(f"🎯 エグジット関連処理: {min(len(result['exit_logic_patterns']), 3)}件（抜粋）")
        for pattern in result['exit_logic_patterns'][:3]:
            if len(pattern.strip()) > 10:
                print(f"  - {pattern.strip()[:80]}...")
    
    return result

# 全戦略を解析
print("\n🔍 **6戦略のエグジット実装状況調査開始**")
analysis_results = []

for strategy_file in strategy_files:
    result = analyze_exit_logic(strategy_file)
    analysis_results.append(result)

# 統合分析結果
print("\n" + "=" * 80)
print("📊 **統合分析結果**")
print("=" * 80)

# 実装状況サマリー
implemented_count = sum(1 for r in analysis_results if r['exists'] and r['has_backtest_method'])
exit_method_count = sum(len(r['exit_methods']) for r in analysis_results if r['exists'])
exit_signal_count = sum(len(r['exit_signal_updates']) for r in analysis_results if r['exists'])

print(f"📋 **実装状況サマリー**")
print(f"  - ファイル存在: {sum(1 for r in analysis_results if r['exists'])}/6戦略")
print(f"  - backtest()実装: {implemented_count}/6戦略")
print(f"  - エグジットメソッド総数: {exit_method_count}")
print(f"  - Exit_Signal更新箇所: {exit_signal_count}")

# TODO-001問題（-1 → 1変換）の検証
print(f"\n🚨 **TODO-001問題検証: Exit_Signal値の使用状況**")
all_signal_values = []
for r in analysis_results:
    if r['exists']:
        all_signal_values.extend(r['signal_values'])

unique_values = set(all_signal_values)
print(f"使用されているシグナル値: {unique_values}")

if '-1' in unique_values:
    print("⚠️  **重要**: -1を使用している戦略が存在 → TODO-001と同様の変換問題の可能性")
else:
    print("ℹ️  -1使用なし → TODO-001とは異なるパターン")

# 各戦略の詳細分析
print(f"\n📝 **戦略別詳細結果**")
for i, result in enumerate(analysis_results):
    strategy_name = strategy_files[i].split('/')[-1].replace('.py', '')
    print(f"\n{i+1}. **{strategy_name}**")
    
    if not result['exists']:
        print("   ❌ ファイル未存在")
        continue
    
    if 'error' in result:
        print(f"   ❌ エラー: {result['error']}")
        continue
    
    print(f"   - backtest(): {'✅' if result['has_backtest_method'] else '❌'}")
    print(f"   - エグジットメソッド: {len(result['exit_methods'])}個")
    print(f"   - Exit_Signal更新: {len(result['exit_signal_updates'])}箇所")
    print(f"   - 使用シグナル値: {set(result['signal_values'])}")

print("\n" + "=" * 80)
print("🎯 **TODO-002 Step1 完了**")
print("=" * 80)

# 次のステップの提案
print(f"\n📋 **次のアクション**")

if exit_method_count == 0:
    print("🚨 **重大発見**: 全戦略でエグジットメソッドが未実装の可能性")
    print("   → TODO-002 Step2: 戦略ファイル詳細確認が必要")

if exit_signal_count == 0:
    print("🚨 **重大発見**: 全戦略でExit_Signal更新処理が未実装")  
    print("   → これがエグジットシグナル未生成の根本原因の可能性")

if '-1' in unique_values:
    print("⚠️  **TODO-001類似問題**: -1使用戦略でシグナル変換問題の可能性")
    print("   → TODO-002 Step3: 個別戦略での-1→1変換確認が必要")

print("\n✅ TODO-002 Step1調査完了")