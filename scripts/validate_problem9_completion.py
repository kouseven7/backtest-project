#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem 9: 最終KPI評価・完了確認
全完了条件の達成度評価とProblem 9完了判定
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import traceback

# プロジェクトルート追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロガー設定
from config.logger_config import setup_logger
logger = setup_logger(__name__)

class Problem9CompletionValidator:
    """Problem 9完了条件検証"""
    
    def __init__(self):
        self.project_root = project_root
        
        # Problem 9完了条件（ロードマップより）
        self.completion_criteria = {
            'quantitative_criteria': [
                {
                    'id': 'QC1',
                    'description': '採用エンジンの品質評価全てが85.0点基準以上達成',
                    'target_value': 85.0,
                    'unit': 'points',
                    'required': True
                },
                {
                    'id': 'QC2', 
                    'description': '重複機能整理率>90%達成（機能統合による重複解消）',
                    'target_value': 90.0,
                    'unit': 'percentage',
                    'required': True
                },
                {
                    'id': 'QC3',
                    'description': '品質統一後のDSSMS出力一貫性100%確認（複数実行での結果一致）',
                    'target_value': 100.0,
                    'unit': 'percentage',
                    'required': True
                }
            ],
            'qualitative_criteria': [
                {
                    'id': 'QL1',
                    'description': '85.0点エンジン出力品質維持',
                    'required': True
                },
                {
                    'id': 'QL2',
                    'description': 'DSSMS Core機能の非破壊保証',
                    'required': True
                },
                {
                    'id': 'QL3',
                    'description': '品質標準文書化完了',
                    'required': True
                }
            ]
        }
        
    def validate_completion(self):
        """Problem 9完了条件検証実行"""
        logger.info("=== Problem 9: 最終KPI評価・完了確認開始 ===")
        
        completion_assessment = {
            'assessment_timestamp': datetime.now().isoformat(),
            'quantitative_results': {},
            'qualitative_results': {},
            'overall_completion_status': 'UNKNOWN',
            'completion_percentage': 0.0,
            'recommendations': [],
            'deliverables': {}
        }
        
        try:
            # 1. 定量的条件評価
            logger.info("📊 定量的条件評価実行")
            quantitative_results = self._evaluate_quantitative_criteria()
            completion_assessment['quantitative_results'] = quantitative_results
            
            # 2. 定性的条件評価
            logger.info("📋 定性的条件評価実行")
            qualitative_results = self._evaluate_qualitative_criteria()
            completion_assessment['qualitative_results'] = qualitative_results
            
            # 3. 全体完了判定
            logger.info("🎯 全体完了判定実行")
            overall_status, completion_percentage = self._calculate_overall_completion(
                quantitative_results, qualitative_results
            )
            completion_assessment['overall_completion_status'] = overall_status
            completion_assessment['completion_percentage'] = completion_percentage
            
            # 4. 成果物確認
            logger.info("📦 成果物確認実行")
            deliverables = self._verify_deliverables()
            completion_assessment['deliverables'] = deliverables
            
            # 5. 推奨事項生成
            completion_assessment['recommendations'] = self._generate_final_recommendations(
                quantitative_results, qualitative_results, deliverables
            )
            
            # 結果保存
            self._save_completion_assessment(completion_assessment)
            
            # 結果報告
            logger.info("=== Problem 9: 最終KPI評価・完了確認完了 ===")
            if overall_status == 'COMPLETED':
                logger.info("🎉 Problem 9: エンジン品質統一 完了")
            elif overall_status == 'PARTIAL_COMPLETION':
                logger.warning("⚠️ Problem 9: 一部条件未達成")
            else:
                logger.error("❌ Problem 9: 完了条件未達成")
            
            return completion_assessment
            
        except Exception as e:
            logger.error(f"❌ 完了確認エラー: {str(e)}")
            completion_assessment['overall_completion_status'] = 'ERROR'
            completion_assessment['error'] = str(e)
            completion_assessment['traceback'] = traceback.format_exc()
            return completion_assessment
    
    def _evaluate_quantitative_criteria(self) -> dict:
        """定量的条件評価"""
        quantitative_results = {}
        
        for criterion in self.completion_criteria['quantitative_criteria']:
            criterion_id = criterion['id']
            
            try:
                if criterion_id == 'QC1':
                    # 採用エンジン品質評価85.0点基準達成確認
                    result = self._check_engine_quality_standard()
                elif criterion_id == 'QC2':
                    # 重複機能整理率>90%達成確認
                    result = self._check_duplicate_function_organization()
                elif criterion_id == 'QC3':
                    # DSSMS出力一貫性100%確認
                    result = self._check_output_consistency()
                else:
                    result = {'status': 'UNKNOWN', 'value': 0.0, 'achieved': False}
                
                quantitative_results[criterion_id] = {
                    'criterion': criterion,
                    'result': result,
                    'achievement_status': 'ACHIEVED' if result.get('achieved', False) else 'NOT_ACHIEVED'
                }
                
            except Exception as e:
                quantitative_results[criterion_id] = {
                    'criterion': criterion,
                    'result': {'status': 'ERROR', 'error': str(e)},
                    'achievement_status': 'ERROR'
                }
        
        return quantitative_results
    
    def _evaluate_qualitative_criteria(self) -> dict:
        """定性的条件評価"""
        qualitative_results = {}
        
        for criterion in self.completion_criteria['qualitative_criteria']:
            criterion_id = criterion['id']
            
            try:
                if criterion_id == 'QL1':
                    # 85.0点エンジン出力品質維持確認
                    result = self._check_reference_engine_quality()
                elif criterion_id == 'QL2':
                    # DSSMS Core機能非破壊保証確認
                    result = self._check_dssms_core_integrity()
                elif criterion_id == 'QL3':
                    # 品質標準文書化完了確認
                    result = self._check_quality_documentation()
                else:
                    result = {'status': 'UNKNOWN', 'achieved': False}
                
                qualitative_results[criterion_id] = {
                    'criterion': criterion,
                    'result': result,
                    'achievement_status': 'ACHIEVED' if result.get('achieved', False) else 'NOT_ACHIEVED'
                }
                
            except Exception as e:
                qualitative_results[criterion_id] = {
                    'criterion': criterion,
                    'result': {'status': 'ERROR', 'error': str(e)},
                    'achievement_status': 'ERROR'
                }
        
        return qualitative_results
    
    def _check_engine_quality_standard(self) -> dict:
        """エンジン品質基準達成確認"""
        try:
            # 最新の品質改善結果確認
            improvement_files = list(self.project_root.glob('engine_quality_improvement_results_*.json'))
            
            if not improvement_files:
                return {'status': 'NO_DATA', 'achieved': False, 'value': 0.0}
            
            latest_file = max(improvement_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                improvement_data = json.load(f)
            
            total_engines = improvement_data.get('total_engines', 0)
            improved_engines = improvement_data.get('improved', 0)
            
            if total_engines == 0:
                achievement_rate = 0.0
            else:
                achievement_rate = (improved_engines / total_engines) * 100
            
            return {
                'status': 'EVALUATED',
                'value': achievement_rate,
                'achieved': achievement_rate >= 85.0,
                'total_engines': total_engines,
                'improved_engines': improved_engines,
                'source_file': str(latest_file)
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _check_duplicate_function_organization(self) -> dict:
        """重複機能整理率確認"""
        try:
            # 最新の重複機能分析結果確認
            analysis_files = list(self.project_root.glob('duplicate_function_analysis_*.json'))
            
            if not analysis_files:
                return {'status': 'NO_DATA', 'achieved': False, 'value': 0.0}
            
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            organization_rate = analysis_data.get('organization_rate', 0.0)
            kpi_achievement = analysis_data.get('kpi_achievement', False)
            
            return {
                'status': 'EVALUATED',
                'value': organization_rate,
                'achieved': kpi_achievement and organization_rate >= 90.0,
                'source_file': str(latest_file)
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _check_output_consistency(self) -> dict:
        """出力一貫性確認"""
        try:
            # DSSMS統合品質検証結果確認
            validation_files = list(self.project_root.glob('dssms_integrated_quality_validation_*.json'))
            
            if not validation_files:
                return {'status': 'NO_DATA', 'achieved': False, 'value': 0.0}
            
            latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            overall_score = validation_data.get('quality_scores', {}).get('overall', 0.0)
            overall_status = validation_data.get('overall_status', 'UNKNOWN')
            
            return {
                'status': 'EVALUATED',
                'value': overall_score,
                'achieved': overall_status == 'PASS' and overall_score >= 85.0,
                'overall_status': overall_status,
                'source_file': str(latest_file)
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _check_reference_engine_quality(self) -> dict:
        """基準エンジン品質維持確認"""
        try:
            reference_engine_path = self.project_root / 'output' / 'dssms_unified_output_engine.py'
            
            if not reference_engine_path.exists():
                return {'status': 'MISSING', 'achieved': False}
            
            # 基準エンジンファイルの整合性確認
            file_size = reference_engine_path.stat().st_size
            last_modified = datetime.fromtimestamp(reference_engine_path.stat().st_mtime)
            
            # 内容確認
            with open(reference_engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 品質維持確認
            has_quality_markers = all([
                'dssms_unified_output' in content.lower(),
                'ENGINE_QUALITY_STANDARD' in content or 'quality' in content.lower(),
                len(content) > 10000  # 最小サイズ確認
            ])
            
            return {
                'status': 'VERIFIED',
                'achieved': has_quality_markers,
                'file_size': file_size,
                'last_modified': last_modified.isoformat(),
                'quality_markers_present': has_quality_markers
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _check_dssms_core_integrity(self) -> dict:
        """DSSMS Core機能整合性確認"""
        try:
            # 主要DSSMS Coreファイル確認
            core_files = [
                'src/dssms/dssms_backtester.py',
                'config/dssms/dssms_backtester_config.json',
                'output/dssms_unified_output_engine.py'
            ]
            
            integrity_status = {
                'files_checked': len(core_files),
                'files_present': 0,
                'files_valid': 0
            }
            
            for file_path in core_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    integrity_status['files_present'] += 1
                    
                    # ファイル有効性確認
                    try:
                        if file_path.endswith('.py'):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            compile(content, str(full_path), 'exec')  # 構文チェック
                        elif file_path.endswith('.json'):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                json.load(f)  # JSON有効性チェック
                        
                        integrity_status['files_valid'] += 1
                    except:
                        pass  # ファイル無効
            
            integrity_achieved = (
                integrity_status['files_present'] == integrity_status['files_checked'] and
                integrity_status['files_valid'] == integrity_status['files_checked']
            )
            
            return {
                'status': 'VERIFIED',
                'achieved': integrity_achieved,
                'integrity_status': integrity_status
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _check_quality_documentation(self) -> dict:
        """品質標準文書化確認"""
        try:
            # 品質関連文書確認
            quality_documents = [
                'engine_quality_standardization_report_*.md',
                'engine_quality_improvement_results_*.json',
                'duplicate_function_analysis_*.json'
            ]
            
            documentation_status = {
                'required_documents': len(quality_documents),
                'documents_present': 0,
                'latest_documents': []
            }
            
            for doc_pattern in quality_documents:
                matching_files = list(self.project_root.glob(doc_pattern))
                if matching_files:
                    documentation_status['documents_present'] += 1
                    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                    documentation_status['latest_documents'].append(str(latest_file))
            
            documentation_achieved = (
                documentation_status['documents_present'] == documentation_status['required_documents']
            )
            
            return {
                'status': 'VERIFIED',
                'achieved': documentation_achieved,
                'documentation_status': documentation_status
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e), 'achieved': False}
    
    def _calculate_overall_completion(self, quantitative_results: dict, qualitative_results: dict) -> tuple:
        """全体完了判定計算"""
        total_criteria = len(quantitative_results) + len(qualitative_results)
        achieved_criteria = 0
        
        # 定量的条件達成数
        for result in quantitative_results.values():
            if result.get('achievement_status') == 'ACHIEVED':
                achieved_criteria += 1
        
        # 定性的条件達成数
        for result in qualitative_results.values():
            if result.get('achievement_status') == 'ACHIEVED':
                achieved_criteria += 1
        
        completion_percentage = (achieved_criteria / total_criteria) * 100 if total_criteria > 0 else 0.0
        
        # 完了ステータス判定
        if completion_percentage == 100.0:
            overall_status = 'COMPLETED'
        elif completion_percentage >= 80.0:
            overall_status = 'PARTIAL_COMPLETION'
        else:
            overall_status = 'INCOMPLETE'
        
        return overall_status, completion_percentage
    
    def _verify_deliverables(self) -> dict:
        """成果物確認"""
        deliverables = {
            'changed_files': [],
            'new_files': [],
            'test_results': [],
            'documentation': []
        }
        
        # 変更ファイル一覧
        backup_files = list(self.project_root.glob('**/*.bak'))
        for backup_file in backup_files:
            original_file = backup_file.with_suffix('')
            if original_file.exists():
                deliverables['changed_files'].append(str(original_file))
        
        # 新規作成ファイル
        script_files = list((self.project_root / 'scripts').glob('*.py'))
        for script_file in script_files:
            if any(keyword in script_file.name for keyword in ['quality', 'improvement', 'test', 'analyze']):
                deliverables['new_files'].append(str(script_file))
        
        # テスト結果ファイル
        test_result_patterns = [
            '*test_results*.json',
            '*quality_validation*.json',
            '*improvement_results*.json'
        ]
        for pattern in test_result_patterns:
            matching_files = list(self.project_root.glob(pattern))
            deliverables['test_results'].extend([str(f) for f in matching_files])
        
        # 文書ファイル
        doc_patterns = [
            '*standardization_report*.md',
            '*analysis*.json'
        ]
        for pattern in doc_patterns:
            matching_files = list(self.project_root.glob(pattern))
            deliverables['documentation'].extend([str(f) for f in matching_files])
        
        return deliverables
    
    def _generate_final_recommendations(self, quantitative_results: dict, qualitative_results: dict, deliverables: dict) -> list:
        """最終推奨事項生成"""
        recommendations = []
        
        # 未達成条件への推奨事項
        for criterion_id, result in quantitative_results.items():
            if result.get('achievement_status') != 'ACHIEVED':
                recommendations.append(f"定量的条件{criterion_id}未達成: {result['criterion']['description']}")
        
        for criterion_id, result in qualitative_results.items():
            if result.get('achievement_status') != 'ACHIEVED':
                recommendations.append(f"定性的条件{criterion_id}未達成: {result['criterion']['description']}")
        
        # 成果物関連推奨事項
        if len(deliverables['changed_files']) == 0:
            recommendations.append("変更ファイルが検出されませんでした。品質改善の実行を確認してください。")
        
        if len(deliverables['documentation']) == 0:
            recommendations.append("品質文書が不十分です。文書化の完了を確認してください。")
        
        # 全て達成の場合
        if not recommendations:
            recommendations.append("全ての完了条件が達成されています。Problem 9: エンジン品質統一が正常に完了しました。")
        
        return recommendations
    
    def _save_completion_assessment(self, assessment: dict):
        """完了評価結果保存"""
        results_file = self.project_root / f"problem9_completion_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Problem 9完了評価結果保存: {results_file}")

def main():
    """メイン実行"""
    try:
        logger.info("=== Problem 9: 最終KPI評価・完了確認開始 ===")
        
        validator = Problem9CompletionValidator()
        results = validator.validate_completion()
        
        # 結果報告
        overall_status = results.get('overall_completion_status', 'UNKNOWN')
        completion_percentage = results.get('completion_percentage', 0.0)
        
        print(f"\n🎯 Problem 9: エンジン品質統一 最終評価")
        print(f"📊 完了率: {completion_percentage:.1f}%")
        
        if overall_status == 'COMPLETED':
            print(f"✅ ステータス: 完了")
            print(f"🎉 Problem 9: エンジン品質統一 正常完了！")
        elif overall_status == 'PARTIAL_COMPLETION':
            print(f"⚠️ ステータス: 一部完了")
            print(f"📋 追加対応が必要です")
        else:
            print(f"❌ ステータス: 未完了")
            print(f"🔧 完了条件の確認と対応が必要です")
        
        # 推奨事項表示
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n📋 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 成果物サマリー
        deliverables = results.get('deliverables', {})
        total_deliverables = sum(len(v) if isinstance(v, list) else 0 for v in deliverables.values())
        print(f"\n📦 成果物: {total_deliverables}個のファイル")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 完了確認エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    result = main()
    if result and result.get('overall_completion_status') == 'COMPLETED':
        print("\n🎊 Problem 9: エンジン品質統一 完了確認")
        sys.exit(0)
    else:
        print("\n❌ Problem 9: 完了条件未達成")
        sys.exit(1)