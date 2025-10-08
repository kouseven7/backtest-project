#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
品質スコア矛盾の原因特定調査
Task 4.2: dssms_unified_output_engine.py = 85.0点
Task 6.1: dssms_unified_output_engine.py = 0点
この重大な矛盾の原因を特定する
"""

import os
import json
from datetime import datetime
from pathlib import Path

class QualityScoreContradictionAnalyzer:
    def __init__(self):
        self.target_file = 'dssms_unified_output_engine.py'
        self.task42_results = 'task_4_2_results_20250912_115837.json'
        self.task61_results = 'task_6_1_current_engine_detailed_analysis_20250914_093615.json'
        self.analysis_results = {}
        
    def analyze_score_contradiction(self):
        """品質スコア矛盾の詳細分析"""
        print("[ALERT] 品質スコア矛盾の原因特定調査")
        print("=" * 80)
        
        # 1. Task 4.2とTask 6.1の評価方法比較
        self._compare_evaluation_methods()
        
        # 2. ファイルの変更履歴確認
        self._check_file_modification_history()
        
        # 3. 評価基準の差異分析
        self._analyze_evaluation_criteria_differences()
        
        # 4. 矛盾の根本原因特定
        self._identify_contradiction_root_cause()
        
        return self.analysis_results
    
    def _compare_evaluation_methods(self):
        """1. Task 4.2とTask 6.1の評価方法比較"""
        print("\n[SEARCH] 1. 評価方法の比較分析")
        print("-" * 60)
        
        comparison = {
            'task42_method': {},
            'task61_method': {},
            'methodological_differences': []
        }
        
        try:
            # Task 4.2の評価方法確認
            if os.path.exists(self.task42_results):
                with open(self.task42_results, 'r', encoding='utf-8') as f:
                    task42_data = json.load(f)
                
                # dssms_unified_output_engine.pyの評価詳細
                engine_details = task42_data.get('detailed_implementations', {}).get('dssms_unified_output_engine.py', {})
                quality_summary = task42_data.get('quality_summary', {}).get('dssms_unified_output_engine.py', {})
                
                comparison['task42_method'] = {
                    'evaluation_date': task42_data.get('analysis_timestamp'),
                    'score': quality_summary.get('total_score', 0),
                    'implementation_percentage': quality_summary.get('implementation_percentage', 0),
                    'quality_percentage': quality_summary.get('quality_percentage', 0),
                    'formula_match_percentage': quality_summary.get('formula_match_percentage', 0),
                    'total_calculations': quality_summary.get('total_calculations', 0),
                    'implemented_count': quality_summary.get('implemented_count', 0),
                    'evaluation_criteria': 'calculated_based_on_implementation_quality_and_formula_match'
                }
                
                print(f"[CHART] Task 4.2評価結果:")
                print(f"   スコア: {comparison['task42_method']['score']}点")
                print(f"   実装率: {comparison['task42_method']['implementation_percentage']}%")
                print(f"   品質率: {comparison['task42_method']['quality_percentage']}%")
                print(f"   公式一致率: {comparison['task42_method']['formula_match_percentage']}%")
            
            # Task 6.1の評価方法確認
            if os.path.exists(self.task61_results):
                with open(self.task61_results, 'r', encoding='utf-8') as f:
                    task61_data = json.load(f)
                
                # Task 6.1の評価方法（task_4_2との比較に基づく）
                task42_comparison = task61_data.get('task42_comparison', {})
                quality_scores = task42_comparison.get('quality_scores', {})
                
                comparison['task61_method'] = {
                    'evaluation_approach': 'comparison_with_task42_results',
                    'reported_score': quality_scores.get('current_engine', 0),
                    'task42_reference_score': quality_scores.get('task42_v1', 0),
                    'score_gap': quality_scores.get('score_gap', 0),
                    'evaluation_criteria': 'cross_reference_with_task42_results'
                }
                
                print(f"[CHART] Task 6.1評価結果:")
                print(f"   報告スコア: {comparison['task61_method']['reported_score']}点")
                print(f"   Task 4.2参照スコア: {comparison['task61_method']['task42_reference_score']}点")
                print(f"   評価アプローチ: {comparison['task61_method']['evaluation_approach']}")
            
            # 方法論的差異の特定
            if comparison['task42_method'] and comparison['task61_method']:
                if comparison['task42_method']['score'] != comparison['task61_method']['reported_score']:
                    comparison['methodological_differences'].append(
                        f"スコア不一致: Task 4.2({comparison['task42_method']['score']}点) vs Task 6.1({comparison['task61_method']['reported_score']}点)"
                    )
                
                if comparison['task42_method']['evaluation_criteria'] != comparison['task61_method']['evaluation_criteria']:
                    comparison['methodological_differences'].append(
                        "評価基準が異なる: Task 4.2は直接分析、Task 6.1は間接参照"
                    )
            
            self.analysis_results['evaluation_comparison'] = comparison
            
        except Exception as e:
            print(f"[ERROR] 評価方法比較エラー: {e}")
            self.analysis_results['evaluation_comparison'] = {'error': str(e)}
    
    def _check_file_modification_history(self):
        """2. ファイルの変更履歴確認"""
        print("\n[SEARCH] 2. ファイル変更履歴の確認")
        print("-" * 60)
        
        modification_history = {}
        
        try:
            file_path = Path(self.target_file)
            
            if file_path.exists():
                stat_info = file_path.stat()
                
                modification_history = {
                    'file_exists': True,
                    'file_size': stat_info.st_size,
                    'last_modified': datetime.fromtimestamp(stat_info.st_mtime),
                    'creation_time': datetime.fromtimestamp(stat_info.st_ctime),
                    'task42_date': datetime.strptime('20250912_115837', '%Y%m%d_%H%M%S'),
                    'task61_date': datetime.strptime('20250914_093615', '%Y%m%d_%H%M%S')
                }
                
                # 評価間でのファイル変更確認
                if modification_history['last_modified'] > modification_history['task42_date']:
                    if modification_history['last_modified'] < modification_history['task61_date']:
                        modification_history['changed_between_evaluations'] = True
                        modification_history['change_timing'] = 'between_task42_and_task61'
                    else:
                        modification_history['changed_between_evaluations'] = True
                        modification_history['change_timing'] = 'after_task61'
                else:
                    modification_history['changed_between_evaluations'] = False
                    modification_history['change_timing'] = 'no_change_detected'
                
                print(f"📁 ファイル情報:")
                print(f"   サイズ: {modification_history['file_size']:,} bytes")
                print(f"   最終更新: {modification_history['last_modified']}")
                print(f"   Task 4.2実行: {modification_history['task42_date']}")
                print(f"   Task 6.1実行: {modification_history['task61_date']}")
                print(f"   評価間変更: {'あり' if modification_history['changed_between_evaluations'] else 'なし'}")
                
            else:
                modification_history = {
                    'file_exists': False,
                    'error': 'ファイルが存在しません'
                }
                print("[ERROR] ファイルが存在しません")
            
            self.analysis_results['modification_history'] = modification_history
            
        except Exception as e:
            print(f"[ERROR] ファイル履歴確認エラー: {e}")
            self.analysis_results['modification_history'] = {'error': str(e)}
    
    def _analyze_evaluation_criteria_differences(self):
        """3. 評価基準の差異分析"""
        print("\n[SEARCH] 3. 評価基準の差異分析")
        print("-" * 60)
        
        criteria_analysis = {
            'task42_criteria': {},
            'task61_criteria': {},
            'discrepancies': []
        }
        
        try:
            # Task 4.2の評価基準詳細確認
            if os.path.exists(self.task42_results):
                with open(self.task42_results, 'r', encoding='utf-8') as f:
                    task42_data = json.load(f)
                
                engine_data = task42_data.get('detailed_implementations', {}).get('dssms_unified_output_engine.py', {})
                
                criteria_analysis['task42_criteria'] = {
                    'calculations_evaluated': list(engine_data.get('calculations', {}).keys()),
                    'scoring_method': 'implementation_quality_formula_match_weighted',
                    'file_analysis': {
                        'file_size': engine_data.get('file_size', 0),
                        'calculations_found': len(engine_data.get('calculations', {}))
                    }
                }
                
                print(f"[LIST] Task 4.2評価基準:")
                print(f"   評価対象計算: {len(criteria_analysis['task42_criteria']['calculations_evaluated'])}個")
                print(f"   ファイルサイズ: {criteria_analysis['task42_criteria']['file_analysis']['file_size']:,} bytes")
                print(f"   発見計算数: {criteria_analysis['task42_criteria']['file_analysis']['calculations_found']}")
            
            # Task 6.1の評価基準確認
            if os.path.exists(self.task61_results):
                with open(self.task61_results, 'r', encoding='utf-8') as f:
                    task61_data = json.load(f)
                
                calc_analysis = task61_data.get('calculation_analysis', {})
                
                criteria_analysis['task61_criteria'] = {
                    'evaluation_method': 'independent_analysis_then_task42_comparison',
                    'implemented_calculations': list(calc_analysis.get('implemented_calculations', {}).keys()),
                    'missing_calculations': calc_analysis.get('missing_calculations', []),
                    'implementation_completeness': calc_analysis.get('implementation_completeness', 0),
                    'file_analysis': calc_analysis.get('file_info', {})
                }
                
                print(f"[LIST] Task 6.1評価基準:")
                print(f"   実装済み計算: {len(criteria_analysis['task61_criteria']['implemented_calculations'])}個")
                print(f"   実装完全性: {criteria_analysis['task61_criteria']['implementation_completeness']:.1f}%")
            
            # 差異の特定
            if criteria_analysis['task42_criteria'] and criteria_analysis['task61_criteria']:
                # ファイルサイズの一致確認
                task42_size = criteria_analysis['task42_criteria']['file_analysis']['file_size']
                task61_size = criteria_analysis['task61_criteria']['file_analysis'].get('size_bytes', 0)
                
                if task42_size != task61_size:
                    criteria_analysis['discrepancies'].append(
                        f"ファイルサイズ不一致: Task 4.2({task42_size:,}) vs Task 6.1({task61_size:,})"
                    )
                
                # 評価対象計算の比較
                task42_calcs = set(criteria_analysis['task42_criteria']['calculations_evaluated'])
                task61_calcs = set(criteria_analysis['task61_criteria']['implemented_calculations'])
                
                if task42_calcs != task61_calcs:
                    criteria_analysis['discrepancies'].append(
                        f"評価対象計算の差異: Task 4.2({len(task42_calcs)}個) vs Task 6.1({len(task61_calcs)}個)"
                    )
                
                print(f"[WARNING] 差異: {len(criteria_analysis['discrepancies'])}件")
                for discrepancy in criteria_analysis['discrepancies']:
                    print(f"   - {discrepancy}")
            
            self.analysis_results['criteria_analysis'] = criteria_analysis
            
        except Exception as e:
            print(f"[ERROR] 評価基準分析エラー: {e}")
            self.analysis_results['criteria_analysis'] = {'error': str(e)}
    
    def _identify_contradiction_root_cause(self):
        """4. 矛盾の根本原因特定"""
        print("\n[SEARCH] 4. 矛盾の根本原因特定")
        print("-" * 60)
        
        root_cause = {
            'identified_causes': [],
            'most_likely_cause': 'unknown',
            'evidence': {},
            'confidence_level': 'low'
        }
        
        try:
            evaluation_comparison = self.analysis_results.get('evaluation_comparison', {})
            modification_history = self.analysis_results.get('modification_history', {})
            criteria_analysis = self.analysis_results.get('criteria_analysis', {})
            
            # 原因1: Task 6.1の評価ロジック誤り
            task61_method = evaluation_comparison.get('task61_method', {})
            if task61_method.get('reported_score') == 0 and task61_method.get('task42_reference_score') == 0:
                root_cause['identified_causes'].append(
                    "Task 6.1で間違ったTask 4.2参照値(0点)を使用"
                )
                root_cause['evidence']['task61_reference_error'] = True
            
            # 原因2: ファイルの実際の変更
            if modification_history.get('changed_between_evaluations'):
                root_cause['identified_causes'].append(
                    f"評価間でのファイル変更: {modification_history.get('change_timing')}"
                )
                root_cause['evidence']['file_modification'] = modification_history
            
            # 原因3: 評価方法の根本的違い
            if criteria_analysis.get('discrepancies'):
                root_cause['identified_causes'].append(
                    "評価基準・方法の根本的差異"
                )
                root_cause['evidence']['evaluation_discrepancies'] = criteria_analysis['discrepancies']
            
            # 原因4: Task 6.1のロジック実装誤り
            # Task 6.1がTask 4.2の結果を正しく読み取れていない可能性
            task42_score = evaluation_comparison.get('task42_method', {}).get('score')
            task61_reported = evaluation_comparison.get('task61_method', {}).get('reported_score')
            
            if task42_score == 85.0 and task61_reported == 0:
                root_cause['identified_causes'].append(
                    "Task 6.1のTask 4.2結果読み取りロジック誤り"
                )
                root_cause['evidence']['logic_error'] = {
                    'task42_actual': task42_score,
                    'task61_reported': task61_reported
                }
            
            # 最も可能性の高い原因の特定
            if len(root_cause['identified_causes']) >= 1:
                # Task 6.1のロジック誤りが最も可能性が高い
                if 'Task 6.1のTask 4.2結果読み取りロジック誤り' in root_cause['identified_causes']:
                    root_cause['most_likely_cause'] = 'Task 6.1のロジック実装誤り'
                    root_cause['confidence_level'] = 'high'
                else:
                    root_cause['most_likely_cause'] = root_cause['identified_causes'][0]
                    root_cause['confidence_level'] = 'medium'
            
            print(f"[TARGET] 特定された原因: {len(root_cause['identified_causes'])}件")
            for i, cause in enumerate(root_cause['identified_causes'], 1):
                print(f"   {i}. {cause}")
            
            print(f"[SEARCH] 最有力原因: {root_cause['most_likely_cause']}")
            print(f"[CHART] 信頼度: {root_cause['confidence_level']}")
            
            self.analysis_results['root_cause'] = root_cause
            
        except Exception as e:
            print(f"[ERROR] 根本原因特定エラー: {e}")
            self.analysis_results['root_cause'] = {'error': str(e)}
    
    def generate_problem16_corrected_definition(self):
        """修正されたProblem 16定義の生成"""
        root_cause = self.analysis_results.get('root_cause', {})
        
        problem16_corrected = {
            'problem_id': 'Problem 16',
            'title': '品質評価システムの信頼性問題（Critical）',
            'severity': 'critical',
            'description': 'Task 4.2とTask 6.1で同一エンジンの品質スコアが85.0点→0点と矛盾、評価システムの信頼性に重大な問題',
            'contradiction_details': {
                'task42_score': 85.0,
                'task61_reported_score': 0.0,
                'score_gap': 85.0,
                'evaluation_inconsistency': True
            },
            'root_cause': root_cause.get('most_likely_cause', 'unknown'),
            'confidence': root_cause.get('confidence_level', 'unknown'),
            'impact_assessment': {
                'evaluation_system_reliability': 'compromised',
                'decision_making_accuracy': 'questionable',
                'quality_management_effectiveness': 'impaired'
            },
            'recommended_resolution': [
                'Task 6.1のロジック修正と再実行',
                '品質評価システムの検証・統一',
                '評価基準の明確化と標準化',
                'dssms_unified_output_engine.pyの実際の品質再確認'
            ]
        }
        
        return problem16_corrected
    
    def save_results(self):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"quality_score_contradiction_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n[OK] 矛盾分析結果保存: {output_file}")
        return output_file

def main():
    """メイン実行"""
    print("[ALERT] 品質スコア矛盾の原因特定調査")
    print("=" * 80)
    
    analyzer = QualityScoreContradictionAnalyzer()
    
    # 矛盾分析実行
    analyzer.analyze_score_contradiction()
    
    # 修正されたProblem 16定義生成
    problem16_corrected = analyzer.generate_problem16_corrected_definition()
    
    # 結果保存
    output_file = analyzer.save_results()
    
    print("\n" + "=" * 80)
    print("[LIST] 矛盾分析完了サマリー")
    print("=" * 80)
    print(f"[TARGET] Problem 16修正版: {problem16_corrected['title']}")
    print(f"[WARNING] 重要度: {problem16_corrected['severity']}")
    print(f"[SEARCH] 根本原因: {problem16_corrected['root_cause']}")
    print(f"[CHART] 信頼度: {problem16_corrected['confidence']}")
    print(f"[UP] スコア矛盾: {problem16_corrected['contradiction_details']['score_gap']}点差")
    
    print(f"\n[IDEA] 推奨解決策:")
    for i, resolution in enumerate(problem16_corrected['recommended_resolution'], 1):
        print(f"   {i}. {resolution}")
    
    return output_file, problem16_corrected

if __name__ == "__main__":
    main()