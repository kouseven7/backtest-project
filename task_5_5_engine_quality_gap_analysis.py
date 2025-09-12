#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 5.5: エンジン品質格差根本原因分析
Task 4.2で発見された深刻な品質格差の根本原因を特定
"""

import os
import sys
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

def analyze_engine_quality_gap_root_cause():
    """エンジン品質格差の根本原因と修正優先度分析"""
    
    print("=" * 80)
    print("Task 5.5: エンジン品質格差根本原因分析")
    print("=" * 80)
    
    # エンジンファイルのパス設定
    engine_paths = {
        'v1': 'dssms_unified_output_engine.py',
        'v2': 'dssms_unified_output_engine_fixed.py', 
        'v3': 'dssms_unified_output_engine_fixed_v3.py',
        'v4': 'dssms_unified_output_engine_fixed_v4.py'
    }
    
    # Task 4.2の品質スコア結果
    quality_scores = {
        'v1': 85.0,  # 最優秀
        'v2': 31.7,  # 低品質
        'v3': 0.0,   # 完全未実装
        'v4': 55.0   # 中品質
    }
    
    analysis_results = {}
    
    print(f"📊 品質格差の概要:")
    print(f"最高点: v1 = {quality_scores['v1']}点")
    print(f"最低点: v3 = {quality_scores['v3']}点")
    print(f"格差: {quality_scores['v1'] - quality_scores['v3']}点")
    print()
    
    # 1. v1エンジン成功要因の詳細分析（85.0点の理由）
    print("🏆 1. v1エンジン成功要因分析（85.0点の理由）")
    v1_analysis = analyze_v1_success_factors(engine_paths['v1'])
    analysis_results['v1_success_factors'] = v1_analysis
    
    # 2. v2,v3,v4エンジンの実装不備の具体的原因特定
    print("\n❌ 2. 低品質エンジンの実装不備原因特定")
    failure_analysis = {}
    for version in ['v2', 'v3', 'v4']:
        if quality_scores[version] < 70.0:  # 70点未満を低品質とする
            print(f"\n--- {version}エンジン分析（{quality_scores[version]}点）---")
            failure_analysis[version] = analyze_implementation_failures(
                engine_paths[version], version, quality_scores[version]
            )
    analysis_results['failure_analysis'] = failure_analysis
    
    # 3. 計算式実装エラーのパターン分析
    print("\n🔍 3. 計算式実装エラーパターン分析")
    error_patterns = analyze_calculation_error_patterns(engine_paths)
    analysis_results['error_patterns'] = error_patterns
    
    # 4. エンジン品質統一のための実装ガイドライン策定
    print("\n📋 4. エンジン品質統一ガイドライン策定")
    guidelines = create_quality_guidelines(v1_analysis, failure_analysis, error_patterns)
    analysis_results['quality_guidelines'] = guidelines
    
    # 5. 品質向上優先順位とコスト効率分析
    print("\n📈 5. 品質向上優先順位とコスト効率分析")
    priority_analysis = analyze_improvement_priority(quality_scores, failure_analysis)
    analysis_results['priority_analysis'] = priority_analysis
    
    # 結果総括
    print("\n" + "=" * 80)
    print("📝 Task 5.5 分析結果総括")
    print("=" * 80)
    
    print(f"🏆 最優秀エンジン: v1 ({quality_scores['v1']}点)")
    print(f"   成功要因: {len(v1_analysis.get('success_factors', []))}個特定")
    
    print(f"❌ 最低品質エンジン: v3 ({quality_scores['v3']}点)")
    if 'v3' in failure_analysis:
        print(f"   主要問題: {len(failure_analysis['v3'].get('critical_issues', []))}個特定")
    
    print(f"📊 実装エラーパターン: {len(error_patterns.get('patterns', []))}種類特定")
    print(f"📋 品質ガイドライン: {len(guidelines.get('rules', []))}項目策定")
    print(f"🎯 改善優先順位: {len(priority_analysis.get('priorities', []))}段階設定")
    
    return analysis_results

def analyze_v1_success_factors(engine_path: str) -> Dict[str, Any]:
    """v1エンジンの成功要因を詳細分析"""
    
    success_factors = {
        'file_structure': {},
        'implementation_quality': {},
        'code_patterns': {},
        'success_factors': []
    }
    
    if not os.path.exists(engine_path):
        print(f"❌ {engine_path} が見つかりません")
        return success_factors
    
    try:
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ {engine_path} 読み込みエラー: {e}")
        return success_factors
    
    # ファイル構造分析
    file_size = len(content)
    line_count = len(content.splitlines())
    
    success_factors['file_structure'] = {
        'file_size': file_size,
        'line_count': line_count,
        'is_substantial': file_size > 1000  # 1KB以上を実質的実装とする
    }
    
    print(f"   ファイルサイズ: {file_size:,} bytes")
    print(f"   行数: {line_count:,} lines")
    
    # 実装品質分析
    class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
    method_count = len(re.findall(r'^\s+def\s+\w+', content, re.MULTILINE))
    comment_count = len(re.findall(r'#.*$', content, re.MULTILINE))
    docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL))
    
    success_factors['implementation_quality'] = {
        'class_count': class_count,
        'method_count': method_count,
        'comment_count': comment_count,
        'docstring_count': docstring_count,
        'documentation_ratio': (comment_count + docstring_count) / line_count if line_count > 0 else 0
    }
    
    print(f"   クラス数: {class_count}")
    print(f"   メソッド数: {method_count}")
    print(f"   コメント・ドキュメント: {comment_count + docstring_count}個")
    print(f"   文書化率: {success_factors['implementation_quality']['documentation_ratio']:.1%}")
    
    # 重要コードパターン検索
    patterns = {
        'error_handling': len(re.findall(r'try:|except|raise', content)),
        'type_hints': len(re.findall(r':\s*\w+\s*[=\)]', content)),
        'logging': len(re.findall(r'log\.|print\(', content)),
        'data_validation': len(re.findall(r'isinstance\(|assert|if.*not', content)),
        'pandas_usage': len(re.findall(r'pd\.|DataFrame|Series', content)),
        'calculation_formulas': len(re.findall(r'total_profit|win_rate|profit_factor', content))
    }
    
    success_factors['code_patterns'] = patterns
    
    print(f"   エラーハンドリング: {patterns['error_handling']}箇所")
    print(f"   型ヒント: {patterns['type_hints']}箇所")
    print(f"   ログ出力: {patterns['logging']}箇所")
    print(f"   データ検証: {patterns['data_validation']}箇所")
    print(f"   統計計算式: {patterns['calculation_formulas']}箇所")
    
    # 成功要因の特定
    factors = []
    
    if file_size > 5000:
        factors.append("実質的なコード量（5KB以上）")
    
    if success_factors['implementation_quality']['documentation_ratio'] > 0.1:
        factors.append("高い文書化率（10%以上）")
    
    if patterns['error_handling'] > 5:
        factors.append("充実したエラーハンドリング")
    
    if patterns['calculation_formulas'] > 3:
        factors.append("統計計算式の実装")
    
    if method_count > 10:
        factors.append("豊富なメソッド実装")
    
    success_factors['success_factors'] = factors
    
    print(f"   🏆 特定された成功要因: {len(factors)}個")
    for i, factor in enumerate(factors, 1):
        print(f"     {i}. {factor}")
    
    return success_factors

def analyze_implementation_failures(engine_path: str, version: str, score: float) -> Dict[str, Any]:
    """低品質エンジンの実装不備原因を特定"""
    
    failure_analysis = {
        'critical_issues': [],
        'implementation_gaps': {},
        'quality_problems': []
    }
    
    if not os.path.exists(engine_path):
        failure_analysis['critical_issues'].append(f"ファイル未存在: {engine_path}")
        print(f"   ❌ ファイルが存在しません: {engine_path}")
        return failure_analysis
    
    try:
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        failure_analysis['critical_issues'].append(f"ファイル読み込みエラー: {e}")
        print(f"   ❌ ファイル読み込みエラー: {e}")
        return failure_analysis
    
    file_size = len(content)
    line_count = len(content.splitlines())
    
    print(f"   ファイルサイズ: {file_size:,} bytes")
    print(f"   行数: {line_count:,} lines")
    
    # 重大問題の特定
    if file_size == 0:
        failure_analysis['critical_issues'].append("完全空ファイル（0 bytes）")
    elif file_size < 100:
        failure_analysis['critical_issues'].append(f"極小ファイル（{file_size} bytes）- 実装不十分")
    
    if line_count < 10:
        failure_analysis['critical_issues'].append(f"極少行数（{line_count} lines）- 実装不足")
    
    # 実装ギャップ分析
    gaps = {}
    
    # 重要メソッドの存在確認
    important_methods = [
        '_convert_backtester_results',
        '_fix_date_inconsistencies', 
        'calculate_win_rate',
        'calculate_profit_factor',
        'calculate_average_profit_loss'
    ]
    
    missing_methods = []
    for method in important_methods:
        if method not in content:
            missing_methods.append(method)
    
    gaps['missing_methods'] = missing_methods
    
    # 統計計算の実装確認
    calculation_patterns = {
        'win_rate_calculation': 'profitable_trades.*total_trades',
        'profit_factor_calculation': 'total_profit.*total_loss',
        'average_profit_calculation': 'profit.*mean\(\)',
        'trade_count_calculation': 'len\(.*trades'
    }
    
    missing_calculations = []
    for calc_name, pattern in calculation_patterns.items():
        if not re.search(pattern, content):
            missing_calculations.append(calc_name)
    
    gaps['missing_calculations'] = missing_calculations
    
    failure_analysis['implementation_gaps'] = gaps
    
    # 品質問題の特定
    quality_issues = []
    
    # エラーハンドリング不足
    error_handling_count = len(re.findall(r'try:|except|raise', content))
    if error_handling_count < 2:
        quality_issues.append(f"エラーハンドリング不足（{error_handling_count}箇所のみ）")
    
    # コメント・ドキュメント不足
    comment_count = len(re.findall(r'#.*$', content, re.MULTILINE))
    docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL))
    total_docs = comment_count + docstring_count
    
    if line_count > 0:
        doc_ratio = total_docs / line_count
        if doc_ratio < 0.05:  # 5%未満
            quality_issues.append(f"文書化不足（{doc_ratio:.1%}）")
    
    # 型ヒント不足
    type_hint_count = len(re.findall(r':\s*\w+\s*[=\)]', content))
    if type_hint_count < 3:
        quality_issues.append(f"型ヒント不足（{type_hint_count}箇所のみ）")
    
    failure_analysis['quality_problems'] = quality_issues
    
    # 結果出力
    print(f"   ❌ 重大問題: {len(failure_analysis['critical_issues'])}個")
    for issue in failure_analysis['critical_issues']:
        print(f"     - {issue}")
    
    print(f"   📋 実装ギャップ:")
    print(f"     - 欠如メソッド: {len(missing_methods)}個")
    print(f"     - 欠如計算式: {len(missing_calculations)}個")
    
    print(f"   ⚠️ 品質問題: {len(quality_issues)}個")
    for issue in quality_issues:
        print(f"     - {issue}")
    
    return failure_analysis

def analyze_calculation_error_patterns(engine_paths: Dict[str, str]) -> Dict[str, Any]:
    """計算式実装エラーのパターンを分析"""
    
    error_patterns = {
        'patterns': [],
        'common_errors': {},
        'formula_accuracy': {}
    }
    
    # 正しい計算式定義
    correct_formulas = {
        'win_rate': 'profitable_trades / total_trades',
        'profit_factor': 'total_profit / abs(total_loss)',
        'average_profit': 'sum(profitable_trades) / count(profitable_trades)',
        'average_loss': 'sum(loss_trades) / count(loss_trades)',
        'total_trades': 'count(all_trades)',
        'max_drawdown': 'max(cumulative_losses)'
    }
    
    print("📊 正しい計算式定義:")
    for name, formula in correct_formulas.items():
        print(f"   {name}: {formula}")
    
    # 各エンジンの計算式実装を検証
    formula_implementations = {}
    
    for version, engine_path in engine_paths.items():
        if not os.path.exists(engine_path):
            continue
            
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        implementations = {}
        
        # 勝率計算の検索
        win_rate_patterns = re.findall(r'win.*rate.*=.*([^\n]+)', content, re.IGNORECASE)
        if win_rate_patterns:
            implementations['win_rate'] = win_rate_patterns[0].strip()
        
        # プロフィットファクター計算の検索
        pf_patterns = re.findall(r'profit.*factor.*=.*([^\n]+)', content, re.IGNORECASE)
        if pf_patterns:
            implementations['profit_factor'] = pf_patterns[0].strip()
        
        # 平均利益計算の検索
        avg_profit_patterns = re.findall(r'average.*profit.*=.*([^\n]+)', content, re.IGNORECASE)
        if avg_profit_patterns:
            implementations['average_profit'] = avg_profit_patterns[0].strip()
        
        formula_implementations[version] = implementations
    
    # エラーパターンの特定
    common_errors = {
        'missing_denominator': [],  # 分母欠如
        'wrong_formula': [],        # 完全に間違った公式
        'variable_misuse': [],      # 変数誤用
        'incomplete_calculation': [] # 不完全計算
    }
    
    for version, implementations in formula_implementations.items():
        print(f"\n--- {version}エンジンの計算式実装 ---")
        
        for formula_name, implementation in implementations.items():
            print(f"   {formula_name}: {implementation}")
            
            # エラーパターンの分類
            if '/' not in implementation and formula_name in ['win_rate', 'profit_factor']:
                common_errors['missing_denominator'].append((version, formula_name, implementation))
            
            if 'len(' in implementation and formula_name == 'profit_factor':
                common_errors['wrong_formula'].append((version, formula_name, implementation))
            
            if 'get(' in implementation and '0)' in implementation:
                common_errors['incomplete_calculation'].append((version, formula_name, implementation))
    
    error_patterns['common_errors'] = common_errors
    
    # パターン総括
    patterns_found = []
    
    if common_errors['missing_denominator']:
        patterns_found.append(f"分母欠如パターン: {len(common_errors['missing_denominator'])}件")
    
    if common_errors['wrong_formula']:
        patterns_found.append(f"公式間違いパターン: {len(common_errors['wrong_formula'])}件")
    
    if common_errors['incomplete_calculation']:
        patterns_found.append(f"不完全計算パターン: {len(common_errors['incomplete_calculation'])}件")
    
    error_patterns['patterns'] = patterns_found
    
    print(f"\n📊 特定されたエラーパターン: {len(patterns_found)}種類")
    for pattern in patterns_found:
        print(f"   - {pattern}")
    
    return error_patterns

def create_quality_guidelines(v1_analysis: Dict, failure_analysis: Dict, error_patterns: Dict) -> Dict[str, Any]:
    """エンジン品質統一のための実装ガイドライン策定"""
    
    guidelines = {
        'rules': [],
        'best_practices': [],
        'quality_metrics': {},
        'implementation_standards': {}
    }
    
    # v1の成功要因から品質ルールを策定
    rules = []
    
    if v1_analysis.get('file_structure', {}).get('file_size', 0) > 5000:
        rules.append("最小ファイルサイズ: 5KB以上（実質的な実装の証拠）")
    
    if v1_analysis.get('implementation_quality', {}).get('method_count', 0) > 10:
        rules.append("最小メソッド数: 10個以上（機能の完全性）")
    
    if v1_analysis.get('implementation_quality', {}).get('documentation_ratio', 0) > 0.1:
        rules.append("文書化率: 10%以上（保守性の確保）")
    
    if v1_analysis.get('code_patterns', {}).get('error_handling', 0) > 5:
        rules.append("エラーハンドリング: 5箇所以上（堅牢性の確保）")
    
    guidelines['rules'] = rules
    
    # ベストプラクティスの策定
    best_practices = [
        "統計計算式は数学的に正確な公式を使用する",
        "勝率 = profitable_trades / total_trades",
        "プロフィットファクター = total_profit / abs(total_loss)",
        "分母がゼロの場合の例外処理を必須とする",
        "型ヒントを活用してコードの可読性を向上させる",
        "重要メソッドには必ずdocstringを記述する",
        "データ検証ロジックを各計算の前に実装する"
    ]
    
    guidelines['best_practices'] = best_practices
    
    # 品質メトリクス基準
    quality_metrics = {
        'minimum_score': 80.0,  # 最低品質スコア
        'file_size_minimum': 5000,  # 最小ファイルサイズ（bytes）
        'method_count_minimum': 10,  # 最小メソッド数
        'documentation_ratio_minimum': 0.10,  # 最小文書化率
        'error_handling_minimum': 5,  # 最小エラーハンドリング箇所
        'calculation_accuracy': 1.0  # 計算式正確性（100%）
    }
    
    guidelines['quality_metrics'] = quality_metrics
    
    # 実装標準
    implementation_standards = {
        'required_methods': [
            '_convert_backtester_results',
            '_fix_date_inconsistencies',
            'calculate_win_rate',
            'calculate_profit_factor',
            'calculate_average_profit_loss'
        ],
        'required_calculations': [
            'win_rate_calculation',
            'profit_factor_calculation', 
            'average_profit_calculation',
            'total_trades_calculation'
        ],
        'code_style': [
            'type_hints_required',
            'docstrings_required',
            'error_handling_required',
            'data_validation_required'
        ]
    }
    
    guidelines['implementation_standards'] = implementation_standards
    
    print(f"📋 策定されたガイドライン:")
    print(f"   品質ルール: {len(rules)}項目")
    print(f"   ベストプラクティス: {len(best_practices)}項目")
    print(f"   品質メトリクス: {len(quality_metrics)}指標")
    print(f"   実装標準: {len(implementation_standards)}カテゴリ")
    
    return guidelines

def analyze_improvement_priority(quality_scores: Dict[str, float], failure_analysis: Dict) -> Dict[str, Any]:
    """品質向上優先順位とコスト効率分析"""
    
    priority_analysis = {
        'priorities': [],
        'cost_efficiency': {},
        'improvement_roadmap': {}
    }
    
    # 品質スコアと改善コストの分析
    improvement_needs = []
    
    for version, score in quality_scores.items():
        if score < 80.0:  # 80点未満を改善対象とする
            gap = 80.0 - score
            
            # 改善コスト見積もり（ファイルサイズベース）
            if version in failure_analysis:
                issues = failure_analysis[version]
                critical_count = len(issues.get('critical_issues', []))
                gap_count = len(issues.get('implementation_gaps', {}).get('missing_methods', []))
                quality_count = len(issues.get('quality_problems', []))
                
                # コスト計算（問題数ベース）
                estimated_cost = critical_count * 10 + gap_count * 5 + quality_count * 2
            else:
                estimated_cost = gap * 2  # デフォルトコスト
            
            improvement_needs.append({
                'version': version,
                'current_score': score,
                'gap': gap,
                'estimated_cost': estimated_cost,
                'efficiency': gap / estimated_cost if estimated_cost > 0 else 0
            })
    
    # 効率順でソート（gap/cost比率）
    improvement_needs.sort(key=lambda x: x['efficiency'], reverse=True)
    
    # 優先順位の決定
    priorities = []
    for i, need in enumerate(improvement_needs, 1):
        priority = {
            'rank': i,
            'version': need['version'],
            'priority_level': 'High' if need['gap'] > 50 else 'Medium' if need['gap'] > 20 else 'Low',
            'gap': need['gap'],
            'estimated_cost': need['estimated_cost'],
            'efficiency': need['efficiency']
        }
        priorities.append(priority)
    
    priority_analysis['priorities'] = priorities
    
    # コスト効率分析
    total_gap = sum(need['gap'] for need in improvement_needs)
    total_cost = sum(need['estimated_cost'] for need in improvement_needs)
    overall_efficiency = total_gap / total_cost if total_cost > 0 else 0
    
    priority_analysis['cost_efficiency'] = {
        'total_gap': total_gap,
        'total_cost': total_cost,
        'overall_efficiency': overall_efficiency,
        'recommended_order': [p['version'] for p in priorities]
    }
    
    # 改善ロードマップ
    roadmap = {}
    
    # Phase 1: 最高効率（v3空ファイル問題など）
    high_priority = [p for p in priorities if p['priority_level'] == 'High']
    if high_priority:
        roadmap['phase_1_emergency'] = {
            'targets': [p['version'] for p in high_priority],
            'timeline': '1-2日',
            'focus': '完全未実装エンジンの基本実装'
        }
    
    # Phase 2: 中効率（部分実装エンジンの改善）
    medium_priority = [p for p in priorities if p['priority_level'] == 'Medium']
    if medium_priority:
        roadmap['phase_2_improvement'] = {
            'targets': [p['version'] for p in medium_priority],
            'timeline': '3-5日',
            'focus': '計算式正確性とエラーハンドリング改善'
        }
    
    # Phase 3: 低効率（微細調整）
    low_priority = [p for p in priorities if p['priority_level'] == 'Low']
    if low_priority:
        roadmap['phase_3_optimization'] = {
            'targets': [p['version'] for p in low_priority],
            'timeline': '1週間',
            'focus': 'コード品質と文書化の向上'
        }
    
    priority_analysis['improvement_roadmap'] = roadmap
    
    print(f"🎯 改善優先順位（効率順）:")
    for priority in priorities:
        print(f"   {priority['rank']}. {priority['version']} - {priority['priority_level']}")
        print(f"      スコア差: {priority['gap']:.1f}点, コスト: {priority['estimated_cost']}, 効率: {priority['efficiency']:.3f}")
    
    print(f"\n📈 全体効率: {overall_efficiency:.3f} (ギャップ/コスト比)")
    print(f"推奨改善順序: {' → '.join(priority_analysis['cost_efficiency']['recommended_order'])}")
    
    return priority_analysis

if __name__ == "__main__":
    try:
        results = analyze_engine_quality_gap_root_cause()
        
        # 結果をJSONファイルに保存
        output_file = "task_5_5_engine_quality_gap_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 分析結果を {output_file} に保存しました")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()