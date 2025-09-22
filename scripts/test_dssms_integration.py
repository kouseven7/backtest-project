#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem 9: DSSMS全体出力品質検証テスト
改善後のDSSMS統合システム品質確認
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

class DSSMSIntegratedQualityValidator:
    """DSSMS統合品質検証"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        
        # 85.0点品質基準
        self.quality_standards = {
            'output_consistency': 95.0,    # 出力一貫性
            'integration_accuracy': 85.0,  # 統合精度
            'performance_standard': 75.0,  # 処理性能
            'error_handling': 90.0         # エラーハンドリング
        }
        
    def validate_dssms_integration(self):
        """DSSMS統合品質検証実行"""
        logger.info("=== DSSMS全体出力品質検証開始 ===")
        
        validation_summary = {
            'test_timestamp': datetime.now().isoformat(),
            'integration_tests': {},
            'quality_scores': {},
            'overall_status': 'UNKNOWN',
            'recommendations': []
        }
        
        try:
            # 1. 基本動作確認
            logger.info("🔍 基本動作確認テスト")
            basic_test = self._test_basic_functionality()
            validation_summary['integration_tests']['basic_functionality'] = basic_test
            
            # 2. エンジン統合確認  
            logger.info("🔍 エンジン統合確認テスト")
            integration_test = self._test_engine_integration()
            validation_summary['integration_tests']['engine_integration'] = integration_test
            
            # 3. 出力品質確認
            logger.info("🔍 出力品質確認テスト")
            output_test = self._test_output_quality()
            validation_summary['integration_tests']['output_quality'] = output_test
            
            # 4. 85.0点基準適合確認
            logger.info("🔍 85.0点基準適合確認テスト")
            compliance_test = self._test_quality_compliance()
            validation_summary['integration_tests']['quality_compliance'] = compliance_test
            
            # 5. 総合評価
            overall_score = self._calculate_overall_score(validation_summary['integration_tests'])
            validation_summary['quality_scores']['overall'] = overall_score
            validation_summary['overall_status'] = 'PASS' if overall_score >= 85.0 else 'FAIL'
            
            # 推奨事項生成
            validation_summary['recommendations'] = self._generate_recommendations(validation_summary)
            
        except Exception as e:
            logger.error(f"❌ DSSMS統合品質検証エラー: {str(e)}")
            validation_summary['overall_status'] = 'ERROR'
            validation_summary['error'] = str(e)
            validation_summary['traceback'] = traceback.format_exc()
        
        # 結果保存
        self._save_validation_results(validation_summary)
        
        # 結果報告
        logger.info("=== DSSMS全体出力品質検証完了 ===")
        if validation_summary['overall_status'] == 'PASS':
            logger.info("✅ DSSMS統合品質検証 合格")
        elif validation_summary['overall_status'] == 'FAIL':
            logger.warning("⚠️ DSSMS統合品質検証 要改善")
        else:
            logger.error("❌ DSSMS統合品質検証 エラー")
        
        return validation_summary
    
    def _test_basic_functionality(self) -> dict:
        """基本動作確認テスト"""
        try:
            # DSSMSBacktester読み込み確認
            try:
                from src.dssms.dssms_backtester import DSSMSBacktester
                backtester_status = 'AVAILABLE'
            except ImportError as e:
                backtester_status = f'IMPORT_ERROR: {str(e)}'
            
            # 統一出力エンジン確認
            unified_engine_path = self.project_root / 'output' / 'dssms_unified_output_engine.py'
            unified_engine_status = 'AVAILABLE' if unified_engine_path.exists() else 'MISSING'
            
            # 設定ファイル確認
            config_path = self.project_root / 'config' / 'dssms' / 'dssms_backtester_config.json'
            config_status = 'AVAILABLE' if config_path.exists() else 'MISSING'
            
            # 基本機能スコア計算
            available_count = sum([
                1 if backtester_status == 'AVAILABLE' else 0,
                1 if unified_engine_status == 'AVAILABLE' else 0,
                1 if config_status == 'AVAILABLE' else 0
            ])
            
            score = (available_count / 3) * 100
            
            return {
                'status': 'PASS' if score >= 85.0 else 'FAIL',
                'score': score,
                'components': {
                    'dssms_backtester': backtester_status,
                    'unified_output_engine': unified_engine_status,
                    'config_file': config_status
                },
                'details': f'{available_count}/3 主要コンポーネント利用可能'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_engine_integration(self) -> dict:
        """エンジン統合確認テスト"""
        try:
            # 改善済みエンジンファイルの存在確認
            improved_engines = [
                'data_cleaning_engine.py',
                'engine_audit_manager.py', 
                'hybrid_ranking_engine.py',
                'simulation_handler.py',
                'comprehensive_scoring_engine.py',
                'unified_output_engine.py',
                'dssms_excel_exporter_v2.py',
                'simple_excel_exporter.py',
                'dssms_switch_engine_v2.py',
                'quality_assurance_engine.py'
            ]
            
            available_engines = 0
            engine_details = {}
            
            for engine_name in improved_engines:
                engine_path = self._find_engine_path(engine_name)
                if engine_path and engine_path.exists():
                    # 品質改善メタデータ確認
                    with open(engine_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    has_quality_metadata = 'ENGINE_QUALITY_STANDARD' in content
                    has_dssms_compatibility = 'DSSMS_UNIFIED_COMPATIBLE' in content
                    
                    if has_quality_metadata and has_dssms_compatibility:
                        available_engines += 1
                        engine_details[engine_name] = 'QUALITY_IMPROVED'
                    else:
                        engine_details[engine_name] = 'PARTIAL_IMPROVEMENT'
                else:
                    engine_details[engine_name] = 'MISSING'
            
            integration_score = (available_engines / len(improved_engines)) * 100
            
            return {
                'status': 'PASS' if integration_score >= 85.0 else 'FAIL',
                'score': integration_score,
                'improved_engines_count': available_engines,
                'total_engines_count': len(improved_engines),
                'engine_details': engine_details,
                'details': f'{available_engines}/{len(improved_engines)} エンジンが品質改善済み'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_output_quality(self) -> dict:
        """出力品質確認テスト"""
        try:
            # 最新の品質統一レポート確認
            quality_reports = list(self.project_root.glob('engine_quality_standardization_report_*.md'))
            
            if not quality_reports:
                return {
                    'status': 'FAIL',
                    'score': 0.0,
                    'error': '品質統一レポートが見つかりません'
                }
            
            latest_report = max(quality_reports, key=lambda x: x.stat().st_mtime)
            
            with open(latest_report, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # レポート品質評価
            quality_indicators = {
                'has_analysis_results': '品質分析結果' in report_content,
                'has_improvement_actions': '改善アクション' in report_content,
                'has_problem13_integration': 'Problem 13統合効果' in report_content,
                'has_kpi_measurement': 'KPI' in report_content or '基準' in report_content
            }
            
            quality_score = (sum(quality_indicators.values()) / len(quality_indicators)) * 100
            
            return {
                'status': 'PASS' if quality_score >= 85.0 else 'FAIL',
                'score': quality_score,
                'latest_report': str(latest_report),
                'quality_indicators': quality_indicators,
                'details': f'{sum(quality_indicators.values())}/{len(quality_indicators)} 品質指標達成'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_quality_compliance(self) -> dict:
        """85.0点基準適合確認テスト"""
        try:
            # 品質改善結果ファイル確認
            improvement_results = list(self.project_root.glob('engine_quality_improvement_results_*.json'))
            
            if not improvement_results:
                return {
                    'status': 'FAIL',
                    'score': 0.0,
                    'error': '品質改善結果が見つかりません'
                }
            
            latest_result = max(improvement_results, key=lambda x: x.stat().st_mtime)
            
            with open(latest_result, 'r', encoding='utf-8') as f:
                improvement_data = json.load(f)
            
            # 改善実績評価
            total_engines = improvement_data.get('total_engines', 0)
            improved_engines = improvement_data.get('improved', 0)
            
            if total_engines == 0:
                compliance_score = 0.0
            else:
                compliance_score = (improved_engines / total_engines) * 100
            
            return {
                'status': 'PASS' if compliance_score >= 90.0 else 'FAIL',  # 90%以上で合格
                'score': compliance_score,
                'total_engines': total_engines,
                'improved_engines': improved_engines,
                'improvement_file': str(latest_result),
                'details': f'{improved_engines}/{total_engines} エンジンが品質改善済み'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'score': 0.0,
                'error': str(e)
            }
    
    def _find_engine_path(self, engine_name: str) -> Path:
        """エンジンファイルパス検索"""
        potential_paths = [
            self.project_root / 'output' / engine_name,
            self.project_root / 'src' / 'dssms' / engine_name,
            self.project_root / engine_name
        ]
        
        for path in potential_paths:
            if path.exists():
                return path
        return None
    
    def _calculate_overall_score(self, test_results: dict) -> float:
        """総合スコア計算"""
        scores = []
        
        for test_name, test_result in test_results.items():
            if isinstance(test_result, dict) and 'score' in test_result:
                scores.append(test_result['score'])
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def _generate_recommendations(self, validation_summary: dict) -> list:
        """推奨事項生成"""
        recommendations = []
        
        overall_score = validation_summary['quality_scores'].get('overall', 0.0)
        
        if overall_score < 85.0:
            recommendations.append("総合品質スコアが85.0点未満です。品質改善の継続が必要です。")
        
        for test_name, test_result in validation_summary['integration_tests'].items():
            if isinstance(test_result, dict):
                if test_result.get('status') == 'FAIL':
                    recommendations.append(f"{test_name}テストで問題検出。詳細確認と改善が必要です。")
                elif test_result.get('status') == 'ERROR':
                    recommendations.append(f"{test_name}テストでエラー発生。技術的調査が必要です。")
        
        if not recommendations:
            recommendations.append("全テストが合格しています。品質統一が正常に完了しています。")
        
        return recommendations
    
    def _save_validation_results(self, results: dict):
        """検証結果保存"""
        results_file = self.project_root / f"dssms_integrated_quality_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"DSSMS統合品質検証結果保存: {results_file}")

def main():
    """メイン実行"""
    try:
        logger.info("=== Problem 9: DSSMS全体出力品質検証開始 ===")
        
        validator = DSSMSIntegratedQualityValidator()
        results = validator.validate_dssms_integration()
        
        # 結果報告
        overall_status = results.get('overall_status', 'UNKNOWN')
        overall_score = results.get('quality_scores', {}).get('overall', 0.0)
        
        if overall_status == 'PASS':
            print(f"\n🎉 DSSMS統合品質検証 合格")
            print(f"📊 総合スコア: {overall_score:.1f}点")
        elif overall_status == 'FAIL':
            print(f"\n⚠️ DSSMS統合品質検証 要改善")
            print(f"📊 総合スコア: {overall_score:.1f}点")
        else:
            print(f"\n❌ DSSMS統合品質検証 エラー")
        
        # 推奨事項表示
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n📋 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ DSSMS統合品質検証エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    result = main()
    if result and result.get('overall_status') == 'PASS':
        print("\n✅ DSSMS全体出力品質検証完了")
        sys.exit(0)
    else:
        print("\n❌ DSSMS全体出力品質検証失敗")
        sys.exit(1)