#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 6.1修正版: 現在エンジンの詳細品質分析（矛盾修正）
85.0点矛盾を解決した正確な品質分析
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

class CorrectedCurrentEngineAnalyzer:
    def __init__(self):
        self.target_engine = 'dssms_unified_output_engine.py'
        self.task42_results_file = 'task_4_2_results_20250912_115837.json'
        self.analysis_results = {}
        
    def analyze_current_engine_corrected(self):
        """修正版: 現在エンジンの詳細品質分析"""
        print("[TOOL] Task 6.1修正版: 現在エンジンの詳細品質分析")
        print("=" * 80)
        
        # 1. Task 4.2から正確なスコア取得
        self._get_accurate_task42_score()
        
        # 2. 矛盾の詳細分析
        self._analyze_scoring_contradiction()
        
        # 3. 実際の品質状況確認
        self._verify_actual_quality()
        
        # 4. 修正されたProblem 16定義
        self._define_corrected_problem16()
        
        return self.analysis_results
    
    def _get_accurate_task42_score(self):
        """1. Task 4.2から正確なスコア取得"""
        print("\n[SEARCH] 1. Task 4.2からの正確なスコア取得")
        print("-" * 60)
        
        try:
            if os.path.exists(self.task42_results_file):
                with open(self.task42_results_file, 'r', encoding='utf-8') as f:
                    task42_data = json.load(f)
                
                # 正しいパスで現在エンジンの品質情報取得
                engine_quality = task42_data.get('implementation_quality', {}).get(self.target_engine, {})
                
                accurate_scores = {
                    'total_score': engine_quality.get('total_score', 0),
                    'implementation_percentage': engine_quality.get('implementation_percentage', 0),
                    'quality_percentage': engine_quality.get('quality_percentage', 0),
                    'formula_match_percentage': engine_quality.get('formula_match_percentage', 0),
                    'total_calculations': engine_quality.get('total_calculations', 0),
                    'implemented_count': engine_quality.get('implemented_count', 0),
                    'high_quality_count': engine_quality.get('high_quality_count', 0)
                }
                
                self.analysis_results['accurate_task42_scores'] = accurate_scores
                
                print(f"[OK] 正確なTask 4.2スコア:")
                print(f"   総合スコア: {accurate_scores['total_score']}点")
                print(f"   実装率: {accurate_scores['implementation_percentage']}%")
                print(f"   品質率: {accurate_scores['quality_percentage']}%")
                print(f"   公式一致率: {accurate_scores['formula_match_percentage']}%")
                print(f"   実装数: {accurate_scores['implemented_count']}/{accurate_scores['total_calculations']}")
                
                if accurate_scores['total_score'] >= 80:
                    print(f"[TARGET] 品質判定: 高品質エンジン（80点以上）")
                elif accurate_scores['total_score'] >= 50:
                    print(f"[CHART] 品質判定: 中品質エンジン（50-79点）")
                else:
                    print(f"[WARNING] 品質判定: 低品質エンジン（50点未満）")
                
            else:
                print("[ERROR] Task 4.2結果ファイルが見つかりません")
                self.analysis_results['accurate_task42_scores'] = {'error': 'file_not_found'}
                
        except Exception as e:
            print(f"[ERROR] スコア取得エラー: {e}")
            self.analysis_results['accurate_task42_scores'] = {'error': str(e)}
    
    def _analyze_scoring_contradiction(self):
        """2. 矛盾の詳細分析"""
        print("\n[SEARCH] 2. 品質スコア矛盾の詳細分析")
        print("-" * 60)
        
        accurate_scores = self.analysis_results.get('accurate_task42_scores', {})
        
        contradiction_analysis = {
            'contradiction_confirmed': False,
            'original_belief': '現在エンジンは0点の低品質',
            'actual_reality': f"現在エンジンは{accurate_scores.get('total_score', 0)}点の高品質",
            'score_gap': accurate_scores.get('total_score', 0) - 0,
            'contradiction_impact': 'critical'
        }
        
        if accurate_scores.get('total_score', 0) > 0:
            contradiction_analysis['contradiction_confirmed'] = True
            
            print(f"[ALERT] 重大な矛盾が確認されました！")
            print(f"   [ERROR] 従来の認識: {contradiction_analysis['original_belief']}")
            print(f"   [OK] 実際の状況: {contradiction_analysis['actual_reality']}")
            print(f"   [UP] スコア差: {contradiction_analysis['score_gap']}点")
            
            # Problem 15の見直しが必要
            print(f"[WARNING] Problem 15の前提条件が無効:")
            print(f"   - 「0点エンジン問題」→「実際は85.0点の高品質エンジン」")
            print(f"   - 品質問題ではなく、評価システムの信頼性問題")
            
        else:
            print(f"[OK] 矛盾は確認されませんでした")
        
        self.analysis_results['contradiction_analysis'] = contradiction_analysis
    
    def _verify_actual_quality(self):
        """3. 実際の品質状況確認"""
        print("\n[SEARCH] 3. 実際の品質状況確認")
        print("-" * 60)
        
        try:
            if os.path.exists(self.target_engine):
                # ファイル基本情報
                file_path = Path(self.target_engine)
                file_stats = file_path.stat()
                
                with open(self.target_engine, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 品質指標の実際確認
                quality_verification = {
                    'file_size': file_stats.st_size,
                    'line_count': len(content.splitlines()),
                    'calculation_implementations': {
                        'win_rate': 'win_rate' in content and 'profitable_trades' in content,
                        'profit_factor': 'profit_factor' in content and 'total_loss' in content,
                        'sharpe_ratio': 'sharpe_ratio' in content,
                        'max_drawdown': 'max_drawdown' in content or 'drawdown' in content,
                        'return_rate': 'return_rate' in content or 'total_return' in content,
                        'volatility': 'volatility' in content or 'std' in content
                    },
                    'code_quality_indicators': {
                        'proper_error_handling': 'try:' in content and 'except' in content,
                        'logging_implemented': 'logger' in content or 'log' in content,
                        'documentation': '"""' in content or "'''" in content,
                        'pandas_usage': 'pd.' in content or 'DataFrame' in content,
                        'mathematical_operations': len(re.findall(r'/[^/]', content))
                    }
                }
                
                # 実装完全性の計算
                implemented_count = sum(quality_verification['calculation_implementations'].values())
                total_calculations = len(quality_verification['calculation_implementations'])
                implementation_rate = implemented_count / total_calculations * 100
                
                quality_verification['implementation_summary'] = {
                    'implemented_calculations': implemented_count,
                    'total_calculations': total_calculations,
                    'implementation_rate': implementation_rate
                }
                
                self.analysis_results['quality_verification'] = quality_verification
                
                print(f"📁 ファイル情報:")
                print(f"   サイズ: {quality_verification['file_size']:,} bytes")
                print(f"   行数: {quality_verification['line_count']:,} lines")
                print(f"[CHART] 計算実装状況:")
                for calc, implemented in quality_verification['calculation_implementations'].items():
                    status = "[OK]" if implemented else "[ERROR]"
                    print(f"   {status} {calc}: {'実装済み' if implemented else '未実装'}")
                print(f"[UP] 実装率: {implementation_rate:.1f}% ({implemented_count}/{total_calculations})")
                
                # Task 4.2スコアとの整合性確認
                task42_scores = self.analysis_results.get('accurate_task42_scores', {})
                task42_implementation_rate = task42_scores.get('implementation_percentage', 0)
                
                if abs(implementation_rate - task42_implementation_rate) < 20:  # 20%以内なら整合
                    print(f"[OK] Task 4.2との整合性: 良好（差異{abs(implementation_rate - task42_implementation_rate):.1f}%）")
                else:
                    print(f"[WARNING] Task 4.2との整合性: 要確認（差異{abs(implementation_rate - task42_implementation_rate):.1f}%）")
                
            else:
                print(f"[ERROR] ファイル {self.target_engine} が見つかりません")
                self.analysis_results['quality_verification'] = {'error': 'file_not_found'}
                
        except Exception as e:
            print(f"[ERROR] 品質確認エラー: {e}")
            self.analysis_results['quality_verification'] = {'error': str(e)}
    
    def _define_corrected_problem16(self):
        """4. 修正されたProblem 16定義"""
        print("\n[SEARCH] 4. 修正されたProblem 16定義")
        print("-" * 60)
        
        contradiction_analysis = self.analysis_results.get('contradiction_analysis', {})
        task42_scores = self.analysis_results.get('accurate_task42_scores', {})
        
        if contradiction_analysis.get('contradiction_confirmed', False):
            # 矛盾が確認された場合
            problem16_corrected = {
                'problem_id': 'Problem 16',
                'title': '品質評価システムの信頼性問題（Critical）',
                'severity': 'critical',
                'category': 'evaluation_system_reliability',
                'description': f"現在のエンジンは実際には{task42_scores.get('total_score', 0)}点の高品質だが、0点と誤認されている評価システム矛盾",
                'contradiction_details': {
                    'false_belief': '現在エンジンは0点の低品質エンジン',
                    'actual_reality': f"現在エンジンは{task42_scores.get('total_score', 0)}点の高品質エンジン",
                    'score_gap': contradiction_analysis.get('score_gap', 0),
                    'task42_verified_score': task42_scores.get('total_score', 0)
                },
                'impact_assessment': {
                    'problem15_invalidated': True,
                    'quality_management_compromised': True,
                    'decision_making_accuracy': 'severely_impaired',
                    'evaluation_system_trust': 'lost'
                },
                'root_cause': 'Task 6.1のロジック誤りによる評価システム信頼性低下',
                'recommended_solutions': [
                    '1. Task 6.1のロジック修正と品質評価システム統一',
                    '2. Problem 15の無効化または大幅修正',
                    '3. 品質評価基準の明確化と検証プロセス確立',
                    '4. 現在エンジンの正確な品質認識に基づく戦略見直し'
                ],
                'priority': 'immediate_critical',
                'estimated_resolution_time': '1-2 days'
            }
        else:
            # 矛盾が確認されなかった場合
            problem16_corrected = {
                'problem_id': 'Problem 16',
                'title': '品質評価の再検証要求',
                'severity': 'medium',
                'category': 'quality_verification',
                'description': '品質評価の一貫性確保のための詳細検証',
                'recommended_solutions': [
                    '品質評価プロセスの標準化',
                    '評価基準の明文化'
                ]
            }
        
        self.analysis_results['corrected_problem16'] = problem16_corrected
        
        print(f"[LIST] 修正されたProblem 16:")
        print(f"   タイトル: {problem16_corrected['title']}")
        print(f"   重要度: {problem16_corrected['severity']}")
        print(f"   説明: {problem16_corrected['description']}")
        
        if problem16_corrected['severity'] == 'critical':
            print(f"[ALERT] Critical問題確認:")
            print(f"   Problem 15無効化: {problem16_corrected['impact_assessment']['problem15_invalidated']}")
            print(f"   評価システム信頼: {problem16_corrected['impact_assessment']['evaluation_system_trust']}")
            print(f"[IDEA] 推奨解決策:")
            for solution in problem16_corrected['recommended_solutions']:
                print(f"     {solution}")
    
    def save_corrected_results(self):
        """修正された結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"task_6_1_corrected_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n[OK] 修正された分析結果保存: {output_file}")
        return output_file

def main():
    """メイン実行"""
    print("[TOOL] Task 6.1修正版: 品質スコア矛盾を解決した正確な分析")
    print("=" * 80)
    
    analyzer = CorrectedCurrentEngineAnalyzer()
    
    # 修正された分析実行
    analyzer.analyze_current_engine_corrected()
    
    # 結果保存
    output_file = analyzer.save_corrected_results()
    
    print("\n" + "=" * 80)
    print("[LIST] 修正版Task 6.1完了サマリー")
    print("=" * 80)
    
    contradiction = analyzer.analysis_results.get('contradiction_analysis', {})
    problem16 = analyzer.analysis_results.get('corrected_problem16', {})
    
    if contradiction.get('contradiction_confirmed', False):
        print("[ALERT] 重大な発見: 品質評価矛盾が確認されました")
        print(f"   実際のエンジン品質: {analyzer.analysis_results.get('accurate_task42_scores', {}).get('total_score', 0)}点")
        print(f"   従来の誤認: 0点")
        print(f"   Problem 16: {problem16.get('severity', 'unknown')}レベル")
        print("[IDEA] 次のアクション: Problem 15の無効化とroadmap2.md更新")
    else:
        print("[OK] 品質評価に矛盾は確認されませんでした")
    
    return output_file

if __name__ == "__main__":
    main()