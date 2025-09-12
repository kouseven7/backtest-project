#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 5.6: 計算式実装エラーパターン調査（Critical）
Task 4.2で発見された計算式の深刻な問題の詳細分析
"""

import os
import sys
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

def analyze_calculation_formula_error_patterns():
    """計算式実装エラーのパターンと修正方針分析"""
    
    print("=" * 80)
    print("Task 5.6: 計算式実装エラーパターン調査（Critical）")
    print("=" * 80)
    
    # エンジンファイルのパス設定
    engine_paths = {
        'v1': 'dssms_unified_output_engine.py',
        'v2': 'dssms_unified_output_engine_fixed.py', 
        'v3': 'dssms_unified_output_engine_fixed_v3.py',
        'v4': 'dssms_unified_output_engine_fixed_v4.py'
    }
    
    analysis_results = {}
    
    # 1. 全エンジンの統計計算式総点検
    print("🔍 1. 全エンジンの統計計算式総点検")
    formula_analysis = analyze_all_engines_formulas(engine_paths)
    analysis_results['formula_analysis'] = formula_analysis
    
    # 2. 数学的に正しい計算式の定義・検証
    print("\n📐 2. 数学的に正しい計算式の定義・検証")
    correct_formulas = define_correct_formulas()
    analysis_results['correct_formulas'] = correct_formulas
    
    # 3. エラーパターンの分類（分母欠如、変数誤用、公式間違い等）
    print("\n❌ 3. エラーパターンの分類")
    error_patterns = classify_error_patterns(formula_analysis, correct_formulas)
    analysis_results['error_patterns'] = error_patterns
    
    # 4. 計算式正確性検証テストケース作成
    print("\n🧪 4. 計算式正確性検証テストケース作成")
    test_cases = create_formula_test_cases(correct_formulas)
    analysis_results['test_cases'] = test_cases
    
    # 5. 段階的修正計画とテスト方針策定
    print("\n📋 5. 段階的修正計画とテスト方針策定")
    correction_plan = create_correction_plan(error_patterns, test_cases)
    analysis_results['correction_plan'] = correction_plan
    
    # 結果総括
    print("\n" + "=" * 80)
    print("📝 Task 5.6 分析結果総括")
    print("=" * 80)
    
    total_formulas = sum(len(engine['formulas']) for engine in formula_analysis.values())
    total_errors = len(error_patterns.get('classified_errors', []))
    error_rate = (total_errors / total_formulas * 100) if total_formulas > 0 else 0
    
    print(f"📊 総計算式数: {total_formulas}個")
    print(f"❌ 総エラー数: {total_errors}個")
    print(f"📈 エラー率: {error_rate:.1f}%")
    print(f"🧪 作成テストケース: {len(test_cases.get('test_scenarios', []))}シナリオ")
    print(f"📋 修正計画: {len(correction_plan.get('phases', []))}段階")
    
    return analysis_results

def analyze_all_engines_formulas(engine_paths: Dict[str, str]) -> Dict[str, Any]:
    """全エンジンの計算式を詳細分析"""
    
    analysis = {}
    
    # 検索対象の計算式パターン
    formula_patterns = {
        'win_rate': [
            r'win.*rate.*=.*([^\n]+)',
            r'勝率.*=.*([^\n]+)',
            r'profitable.*total.*([^\n]+)',
            r'success.*rate.*([^\n]+)'
        ],
        'profit_factor': [
            r'profit.*factor.*=.*([^\n]+)',
            r'プロフィット.*ファクター.*=.*([^\n]+)', 
            r'total.*profit.*total.*loss.*([^\n]+)',
            r'利益.*損失.*比.*([^\n]+)'
        ],
        'average_profit': [
            r'average.*profit.*=.*([^\n]+)',
            r'平均.*利益.*=.*([^\n]+)',
            r'mean.*profit.*([^\n]+)',
            r'avg.*gain.*([^\n]+)'
        ],
        'average_loss': [
            r'average.*loss.*=.*([^\n]+)',
            r'平均.*損失.*=.*([^\n]+)', 
            r'mean.*loss.*([^\n]+)',
            r'avg.*loss.*([^\n]+)'
        ],
        'max_drawdown': [
            r'max.*drawdown.*=.*([^\n]+)',
            r'最大.*ドローダウン.*=.*([^\n]+)',
            r'maximum.*loss.*([^\n]+)'
        ],
        'total_trades': [
            r'total.*trades.*=.*([^\n]+)',
            r'総.*取引.*数.*=.*([^\n]+)',
            r'trade.*count.*([^\n]+)',
            r'len\(.*trades.*\).*([^\n]+)'
        ]
    }
    
    for version, engine_path in engine_paths.items():
        print(f"--- {version}エンジン分析 ---")
        
        engine_analysis = {
            'file_exists': False,
            'file_size': 0,
            'formulas': {},
            'implementation_quality': 'unknown',
            'critical_issues': []
        }
        
        if not os.path.exists(engine_path):
            engine_analysis['critical_issues'].append(f"ファイル未存在: {engine_path}")
            print(f"   ❌ ファイルが存在しません: {engine_path}")
            analysis[version] = engine_analysis
            continue
        
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            engine_analysis['critical_issues'].append(f"読み込みエラー: {e}")
            print(f"   ❌ 読み込みエラー: {e}")
            analysis[version] = engine_analysis
            continue
        
        engine_analysis['file_exists'] = True
        engine_analysis['file_size'] = len(content)
        
        print(f"   ファイルサイズ: {len(content):,} bytes")
        
        # 各計算式パターンの検索
        found_formulas = {}
        
        for formula_type, patterns in formula_patterns.items():
            implementations = []
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    implementations.extend([match.strip() for match in matches])
            
            if implementations:
                found_formulas[formula_type] = implementations
                print(f"   ✅ {formula_type}: {len(implementations)}個の実装発見")
                for impl in implementations[:2]:  # 最初の2個を表示
                    print(f"      - {impl[:50]}...")
            else:
                print(f"   ❌ {formula_type}: 実装なし")
        
        engine_analysis['formulas'] = found_formulas
        
        # 実装品質の評価
        formula_count = len(found_formulas)
        if formula_count == 0:
            engine_analysis['implementation_quality'] = 'none'
        elif formula_count <= 2:
            engine_analysis['implementation_quality'] = 'minimal'
        elif formula_count <= 4:
            engine_analysis['implementation_quality'] = 'partial'
        else:
            engine_analysis['implementation_quality'] = 'comprehensive'
        
        print(f"   📊 実装品質: {engine_analysis['implementation_quality']} ({formula_count}個の計算式)")
        
        analysis[version] = engine_analysis
    
    return analysis

def define_correct_formulas() -> Dict[str, Any]:
    """数学的に正しい計算式の定義"""
    
    correct_formulas = {
        'win_rate': {
            'formula': 'profitable_trades / total_trades',
            'description': '勝率 = 利益取引数 ÷ 総取引数',
            'expected_range': (0.0, 1.0),
            'data_type': 'percentage',
            'validation_rules': [
                'profitable_trades <= total_trades',
                'total_trades > 0',
                '0 <= result <= 1'
            ],
            'common_errors': [
                '分母（total_trades）の欠如',
                'count()とlen()の混同',
                '結果の範囲チェック不備'
            ]
        },
        'profit_factor': {
            'formula': 'total_profit / abs(total_loss)',
            'description': 'プロフィットファクター = 総利益 ÷ 総損失の絶対値',
            'expected_range': (0.0, float('inf')),
            'data_type': 'ratio',
            'validation_rules': [
                'total_loss != 0',
                'total_profit >= 0',
                'result >= 0'
            ],
            'common_errors': [
                '分母（total_loss）の欠如',
                'abs()による絶対値変換の欠如',
                'ゼロ除算の未対応'
            ]
        },
        'average_profit': {
            'formula': 'sum(profitable_trades) / count(profitable_trades)',
            'description': '平均利益 = 利益取引の合計 ÷ 利益取引数',
            'expected_range': (0.0, float('inf')),
            'data_type': 'currency',
            'validation_rules': [
                'count(profitable_trades) > 0',
                'all(profitable_trades) > 0',
                'result > 0'
            ],
            'common_errors': [
                '全取引での平均計算（利益取引のみでない）',
                'sum()とcount()の不一致',
                'ゼロ除算の未対応'
            ]
        },
        'average_loss': {
            'formula': 'sum(loss_trades) / count(loss_trades)',
            'description': '平均損失 = 損失取引の合計 ÷ 損失取引数',
            'expected_range': (float('-inf'), 0.0),
            'data_type': 'currency',
            'validation_rules': [
                'count(loss_trades) > 0',
                'all(loss_trades) < 0',
                'result < 0'
            ],
            'common_errors': [
                '損失の符号処理（負値→正値変換）',
                '全取引での平均計算',
                'abs()の誤用'
            ]
        },
        'max_drawdown': {
            'formula': 'min(cumulative_returns) - peak_value',
            'description': '最大ドローダウン = 累積リターンの最小値 - ピーク値',
            'expected_range': (float('-inf'), 0.0),
            'data_type': 'percentage',
            'validation_rules': [
                'result <= 0',
                'cumulative_returns is not empty',
                'peak_value >= max(cumulative_returns)'
            ],
            'common_errors': [
                '単純な最小値計算（ピークからの落差でない）',
                '結果の符号間違い（正値で表示）',
                '累積計算の実装不備'
            ]
        },
        'total_trades': {
            'formula': 'count(all_trades)',
            'description': '総取引数 = 全取引のカウント',
            'expected_range': (0, float('inf')),
            'data_type': 'integer',
            'validation_rules': [
                'result >= 0',
                'result == count(profitable_trades) + count(loss_trades)',
                'result is integer'
            ],
            'common_errors': [
                'float型での計算',
                'null/NaN値の未除外',
                '重複取引の重複カウント'
            ]
        }
    }
    
    print("📐 数学的に正しい計算式の定義:")
    for name, formula_def in correct_formulas.items():
        print(f"   {name}:")
        print(f"     公式: {formula_def['formula']}")
        print(f"     説明: {formula_def['description']}")
        print(f"     範囲: {formula_def['expected_range']}")
        print(f"     一般的エラー: {len(formula_def['common_errors'])}種類")
    
    return correct_formulas

def classify_error_patterns(formula_analysis: Dict, correct_formulas: Dict) -> Dict[str, Any]:
    """エラーパターンの分類"""
    
    error_patterns = {
        'classified_errors': [],
        'error_categories': {
            'missing_denominator': [],      # 分母欠如
            'wrong_formula': [],            # 完全に間違った公式
            'variable_misuse': [],          # 変数誤用
            'incomplete_calculation': [],   # 不完全計算
            'type_mismatch': [],           # データ型不一致
            'range_violation': [],         # 範囲外結果
            'mathematical_error': []       # 数学的エラー
        },
        'severity_levels': {
            'critical': [],     # 致命的（結果が完全に無意味）
            'major': [],        # 重大（結果が大きく歪む）
            'minor': [],        # 軽微（精度が若干落ちる）
            'warning': []       # 警告（将来問題になる可能性）
        }
    }
    
    for version, analysis in formula_analysis.items():
        if not analysis['file_exists']:
            continue
            
        print(f"--- {version}エンジンのエラー分析 ---")
        
        for formula_type, implementations in analysis['formulas'].items():
            if formula_type not in correct_formulas:
                continue
                
            correct_def = correct_formulas[formula_type]
            correct_formula = correct_def['formula']
            
            for impl in implementations:
                errors = analyze_single_implementation(
                    version, formula_type, impl, correct_def
                )
                
                for error in errors:
                    error_patterns['classified_errors'].append(error)
                    
                    # カテゴリ別分類
                    category = error['category']
                    if category in error_patterns['error_categories']:
                        error_patterns['error_categories'][category].append(error)
                    
                    # 重要度別分類
                    severity = error['severity']
                    if severity in error_patterns['severity_levels']:
                        error_patterns['severity_levels'][severity].append(error)
    
    # 結果サマリ
    total_errors = len(error_patterns['classified_errors'])
    print(f"\n📊 エラー分類結果:")
    print(f"   総エラー数: {total_errors}個")
    
    for category, errors in error_patterns['error_categories'].items():
        if errors:
            print(f"   {category}: {len(errors)}個")
    
    print(f"\n⚠️ 重要度別:")
    for severity, errors in error_patterns['severity_levels'].items():
        if errors:
            print(f"   {severity}: {len(errors)}個")
    
    return error_patterns

def analyze_single_implementation(version: str, formula_type: str, implementation: str, correct_def: Dict) -> List[Dict]:
    """単一実装の詳細エラー分析"""
    
    errors = []
    correct_formula = correct_def['formula']
    
    print(f"   分析中: {formula_type} = {implementation[:50]}...")
    
    # 1. 分母欠如チェック
    if '/' in correct_formula and '/' not in implementation:
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'missing_denominator',
            'severity': 'critical',
            'description': f'分母が欠如: 期待公式 {correct_formula} に対して除算なし',
            'expected': correct_formula,
            'actual': implementation
        })
    
    # 2. 完全に間違った公式チェック
    if formula_type == 'profit_factor' and 'len(' in implementation:
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'wrong_formula',
            'severity': 'critical',
            'description': 'プロフィットファクターでlen()使用は完全に間違い',
            'expected': correct_formula,
            'actual': implementation
        })
    
    # 3. 変数誤用チェック
    if formula_type == 'win_rate' and 'len(' in implementation and 'total' not in implementation.lower():
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'variable_misuse',
            'severity': 'major',
            'description': '勝率計算で分母（total_trades）が未使用',
            'expected': correct_formula,
            'actual': implementation
        })
    
    # 4. 不完全計算チェック
    if '.get(' in implementation and ', 0)' in implementation:
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'incomplete_calculation',
            'severity': 'major',
            'description': 'get()メソッドでデフォルト値0使用は不完全計算の兆候',
            'expected': correct_formula,
            'actual': implementation
        })
    
    # 5. 型不一致チェック
    if correct_def['data_type'] == 'integer' and 'float' in implementation:
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'type_mismatch',
            'severity': 'minor',
            'description': f'期待型 {correct_def["data_type"]} に対して不適切な型使用',
            'expected': correct_formula,
            'actual': implementation
        })
    
    # 6. 数学的エラーチェック
    if formula_type == 'profit_factor' and 'abs(' not in implementation and 'total_loss' in implementation:
        errors.append({
            'version': version,
            'formula_type': formula_type,
            'implementation': implementation,
            'category': 'mathematical_error',
            'severity': 'major',
            'description': 'プロフィットファクターで損失の絶対値処理なし',
            'expected': correct_formula,
            'actual': implementation
        })
    
    if errors:
        print(f"     ❌ {len(errors)}個のエラー発見")
        for error in errors:
            print(f"       - {error['category']}: {error['description']}")
    else:
        print(f"     ✅ エラーなし")
    
    return errors

def create_formula_test_cases(correct_formulas: Dict) -> Dict[str, Any]:
    """計算式正確性検証テストケース作成"""
    
    test_cases = {
        'test_scenarios': [],
        'validation_functions': {},
        'expected_results': {}
    }
    
    # 基本テストデータセット
    test_data = {
        'profitable_trades': [100, 150, 200, 75, 120],  # 5回の利益取引
        'loss_trades': [-50, -80, -30, -100],           # 4回の損失取引
        'total_profit': 645,    # 利益取引の合計
        'total_loss': -260,     # 損失取引の合計
        'total_trades': 9,      # 総取引数
        'cumulative_returns': [0, 100, 50, 250, 170, 370, 290, 390, 310, 430],
        'peak_value': 430
    }
    
    print("🧪 テストケース作成:")
    
    for formula_name, formula_def in correct_formulas.items():
        scenario = create_test_scenario(formula_name, formula_def, test_data)
        test_cases['test_scenarios'].append(scenario)
        
        print(f"   {formula_name}:")
        print(f"     期待値: {scenario['expected_result']}")
        print(f"     バリデーション: {len(scenario['validation_checks'])}項目")
    
    # 共通バリデーション関数
    test_cases['validation_functions'] = {
        'check_range': lambda value, min_val, max_val: min_val <= value <= max_val,
        'check_type': lambda value, expected_type: isinstance(value, expected_type),
        'check_non_zero': lambda value: value != 0,
        'check_positive': lambda value: value > 0,
        'check_negative': lambda value: value < 0,
        'check_percentage': lambda value: 0 <= value <= 1
    }
    
    return test_cases

def create_test_scenario(formula_name: str, formula_def: Dict, test_data: Dict) -> Dict:
    """単一計算式のテストシナリオ作成"""
    
    scenario = {
        'formula_name': formula_name,
        'test_data': test_data.copy(),
        'expected_result': None,
        'validation_checks': [],
        'error_conditions': []
    }
    
    # 期待値計算
    if formula_name == 'win_rate':
        profitable_count = len(test_data['profitable_trades'])
        scenario['expected_result'] = profitable_count / test_data['total_trades']
        
    elif formula_name == 'profit_factor':
        scenario['expected_result'] = test_data['total_profit'] / abs(test_data['total_loss'])
        
    elif formula_name == 'average_profit':
        scenario['expected_result'] = test_data['total_profit'] / len(test_data['profitable_trades'])
        
    elif formula_name == 'average_loss':
        scenario['expected_result'] = test_data['total_loss'] / len(test_data['loss_trades'])
        
    elif formula_name == 'max_drawdown':
        cumulative = test_data['cumulative_returns']
        peak = test_data['peak_value']
        scenario['expected_result'] = min(cumulative) - peak
        
    elif formula_name == 'total_trades':
        scenario['expected_result'] = test_data['total_trades']
    
    # バリデーション条件
    scenario['validation_checks'] = formula_def['validation_rules'].copy()
    
    # エラー条件（ゼロ除算等）
    if 'total_trades' in formula_def['formula']:
        scenario['error_conditions'].append('total_trades = 0')
    if 'total_loss' in formula_def['formula']:
        scenario['error_conditions'].append('total_loss = 0')
    
    return scenario

def create_correction_plan(error_patterns: Dict, test_cases: Dict) -> Dict[str, Any]:
    """段階的修正計画とテスト方針策定"""
    
    correction_plan = {
        'phases': [],
        'priority_matrix': {},
        'testing_strategy': {},
        'success_criteria': {}
    }
    
    # 重要度別エラーグループ化
    critical_errors = error_patterns['severity_levels']['critical']
    major_errors = error_patterns['severity_levels']['major']
    minor_errors = error_patterns['severity_levels']['minor']
    
    print("📋 修正計画策定:")
    
    # Phase 1: 緊急修正（Critical）
    if critical_errors:
        phase1 = {
            'phase': 1,
            'name': '緊急修正（Critical）',
            'timeline': '1-2日',
            'targets': list(set(error['formula_type'] for error in critical_errors)),
            'errors_to_fix': len(critical_errors),
            'priority': 'highest',
            'description': '数学的に完全に間違った計算式の修正'
        }
        correction_plan['phases'].append(phase1)
        print(f"   Phase 1: {phase1['name']} - {len(critical_errors)}個のエラー")
    
    # Phase 2: 重要修正（Major）
    if major_errors:
        phase2 = {
            'phase': 2,
            'name': '重要修正（Major）',
            'timeline': '2-3日',
            'targets': list(set(error['formula_type'] for error in major_errors)),
            'errors_to_fix': len(major_errors),
            'priority': 'high',
            'description': '結果に大きく影響する計算式の修正'
        }
        correction_plan['phases'].append(phase2)
        print(f"   Phase 2: {phase2['name']} - {len(major_errors)}個のエラー")
    
    # Phase 3: 品質改善（Minor）
    if minor_errors:
        phase3 = {
            'phase': 3,
            'name': '品質改善（Minor）',
            'timeline': '3-5日',
            'targets': list(set(error['formula_type'] for error in minor_errors)),
            'errors_to_fix': len(minor_errors),
            'priority': 'medium',
            'description': '精度向上と型安全性の確保'
        }
        correction_plan['phases'].append(phase3)
        print(f"   Phase 3: {phase3['name']} - {len(minor_errors)}個のエラー")
    
    # 優先度マトリクス
    correction_plan['priority_matrix'] = {
        'critical_formulas': ['profit_factor', 'win_rate'],  # 最重要
        'important_formulas': ['average_profit', 'average_loss'],  # 重要
        'secondary_formulas': ['max_drawdown', 'total_trades']  # 二次的
    }
    
    # テスト戦略
    correction_plan['testing_strategy'] = {
        'unit_tests': f'{len(test_cases["test_scenarios"])}個のシナリオテスト',
        'integration_tests': '全エンジン統合テスト',
        'regression_tests': '既存機能回帰テスト',
        'performance_tests': '計算パフォーマンステスト',
        'validation_frequency': '各修正後に即座実行'
    }
    
    # 成功基準
    total_errors = len(error_patterns['classified_errors'])
    correction_plan['success_criteria'] = {
        'error_reduction_target': '90%以上（エラー削減率）',
        'accuracy_target': '100%（数学的正確性）',
        'performance_target': '修正前比120%以内（実行時間）',
        'regression_target': '0件（既存機能劣化）',
        'current_error_count': total_errors,
        'target_error_count': max(1, total_errors // 10)  # 90%削減目標
    }
    
    print(f"\n🎯 成功基準:")
    print(f"   エラー削減: {total_errors} → {correction_plan['success_criteria']['target_error_count']}個")
    print(f"   正確性: 100%（数学的正確性確保）")
    print(f"   テストケース: {len(test_cases['test_scenarios'])}シナリオで検証")
    
    return correction_plan

if __name__ == "__main__":
    try:
        results = analyze_calculation_formula_error_patterns()
        
        # 結果をJSONファイルに保存
        output_file = "task_5_6_calculation_formula_error_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ 分析結果を {output_file} に保存しました")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()