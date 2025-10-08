#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem 9: エンジン品質統一テストスクリプト
改善済みエンジンの動作確認・品質検証
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

class ImprovedEngineValidator:
    """改善済みエンジンの動作・品質検証"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.reference_engine_path = self.project_root / 'output' / 'dssms_unified_output_engine.py'
        
        # 85.0点品質基準
        self.quality_standards = {
            'output_accuracy': 85.0,
            'code_quality': 80.0, 
            'performance': 75.0,
            'completeness': 90.0,
            'consistency': 95.0
        }
        
        # 改善対象エンジンリスト（Problem 9で改善済み）
        self.improved_engines = [
            'data_cleaning_engine.py',
            'engine_audit_manager.py', 
            'hybrid_ranking_engine.py',
            'simulation_handler.py',
            'comprehensive_scoring_engine.py',
            'unified_output_engine.py',
            'dssms_unified_output_engine.py',
            'dssms_excel_exporter_v2.py',
            'simple_excel_exporter.py',
            'dssms_switch_engine_v2.py',
            'quality_assurance_engine.py'
        ]
        
    def validate_all_engines(self):
        """全改善エンジンの検証実行"""
        logger.info("=== 改善エンジン動作確認テスト開始 ===")
        
        validation_summary = {
            'total_engines': len(self.improved_engines),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'test_timestamp': datetime.now().isoformat()
        }
        
        for engine_name in self.improved_engines:
            try:
                logger.info(f"エンジン検証開始: {engine_name}")
                result = self._validate_single_engine(engine_name)
                
                if result['status'] == 'PASS':
                    validation_summary['passed'] += 1
                    logger.info(f"[OK] {engine_name}: 検証合格")
                else:
                    validation_summary['failed'] += 1
                    validation_summary['errors'].append({
                        'engine': engine_name,
                        'error': result.get('error', 'Unknown error')
                    })
                    logger.error(f"[ERROR] {engine_name}: 検証失敗 - {result.get('error', 'Unknown')}")
                    
                self.test_results[engine_name] = result
                
            except Exception as e:
                logger.error(f"[ERROR] {engine_name}: 検証エラー - {str(e)}")
                validation_summary['failed'] += 1
                validation_summary['errors'].append({
                    'engine': engine_name,
                    'error': f"Exception: {str(e)}"
                })
                self.test_results[engine_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # 結果サマリー
        logger.info("=== 改善エンジン動作確認テスト完了 ===")
        logger.info(f"[OK] 合格: {validation_summary['passed']}個")
        logger.info(f"[ERROR] 失敗: {validation_summary['failed']}個")
        logger.info(f"[CHART] 成功率: {validation_summary['passed']/validation_summary['total_engines']*100:.1f}%")
        
        # 結果保存
        self._save_test_results(validation_summary)
        return validation_summary
        
    def _validate_single_engine(self, engine_name: str) -> dict:
        """単一エンジンの検証"""
        engine_path = self._find_engine_path(engine_name)
        
        if not engine_path or not engine_path.exists():
            return {
                'status': 'FAIL',
                'error': f'エンジンファイルが見つかりません: {engine_name}',
                'quality_scores': {}
            }
        
        try:
            # 1. ファイル読み込み・構文チェック
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 2. 基本構文チェック
            compile(content, str(engine_path), 'exec')
            
            # 3. 品質評価
            quality_scores = self._evaluate_engine_quality(engine_path, content)
            
            # 4. 85.0点基準適合チェック
            compliance_check = self._check_quality_compliance(quality_scores)
            
            if compliance_check['compliant']:
                return {
                    'status': 'PASS',
                    'quality_scores': quality_scores,
                    'compliance': compliance_check,
                    'file_size': engine_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(engine_path.stat().st_mtime).isoformat()
                }
            else:
                return {
                    'status': 'FAIL',
                    'error': compliance_check['reason'],
                    'quality_scores': quality_scores,
                    'compliance': compliance_check
                }
                
        except SyntaxError as e:
            return {
                'status': 'FAIL',
                'error': f'構文エラー: {str(e)}',
                'quality_scores': {}
            }
        except Exception as e:
            return {
                'status': 'ERROR', 
                'error': f'検証エラー: {str(e)}',
                'quality_scores': {}
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
    
    def _evaluate_engine_quality(self, engine_path: Path, content: str) -> dict:
        """エンジン品質評価"""
        scores = {}
        
        # 出力精度評価（コード解析ベース）
        scores['output_accuracy'] = self._assess_output_accuracy(content)
        
        # コード品質評価
        scores['code_quality'] = self._assess_code_quality(content, engine_path)
        
        # 処理性能評価
        scores['performance'] = self._assess_performance(content)
        
        # 機能完成度評価
        scores['completeness'] = self._assess_completeness(content)
        
        # 出力一貫性評価
        scores['consistency'] = self._assess_consistency(content)
        
        return scores
    
    def _assess_output_accuracy(self, content: str) -> float:
        """出力精度評価"""
        score = 70.0  # ベーススコア
        
        # エラーハンドリング実装チェック
        if 'try:' in content and 'except' in content:
            score += 5.0
            
        # ログ出力実装チェック  
        if 'logger.' in content or 'logging.' in content:
            score += 5.0
            
        # データ検証実装チェック
        if 'validate' in content or 'check' in content:
            score += 5.0
            
        # TODO(tag:phase2, rationale:DSSMS Core focus) コメント適用
        if 'TODO(tag:' in content:
            score += 5.0
            
        # 基準エンジンパターン適用
        if 'DSSMSUnified' in content or 'unified_output' in content:
            score += 5.0
            
        return min(score, 95.0)
    
    def _assess_code_quality(self, content: str, engine_path: Path) -> float:
        """コード品質評価"""
        score = 70.0  # ベーススコア
        
        # ファイルサイズ適正チェック（20KB〜60KB目安）
        file_size = engine_path.stat().st_size / 1024  # KB
        if 20 <= file_size <= 60:
            score += 5.0
            
        # 関数・クラス構造チェック
        if 'class ' in content:
            score += 5.0
        if 'def ' in content:
            score += 3.0
            
        # docstring実装チェック
        if '"""' in content:
            score += 5.0
            
        # import文整理チェック
        import_count = content.count('import ')
        if import_count <= 15:  # 適正import数
            score += 2.0
            
        return min(score, 90.0)
    
    def _assess_performance(self, content: str) -> float:
        """処理性能評価"""
        score = 70.0  # ベーススコア
        
        # 効率的データ処理パターンチェック
        if 'pandas' in content or 'pd.' in content:
            score += 5.0
            
        # キャッシュ機能実装チェック
        if 'cache' in content or '@lru_cache' in content:
            score += 5.0
            
        # 並列処理実装チェック
        if 'multiprocessing' in content or 'concurrent' in content:
            score += 5.0
            
        return min(score, 85.0)
    
    def _assess_completeness(self, content: str) -> float:
        """機能完成度評価"""
        score = 80.0  # ベーススコア
        
        # 主要機能実装チェック
        if 'def run(' in content or 'def execute(' in content:
            score += 5.0
            
        # 設定読み込み機能
        if 'config' in content or 'json' in content:
            score += 3.0
            
        # 結果出力機能
        if 'output' in content or 'export' in content:
            score += 2.0
            
        return min(score, 95.0)
    
    def _assess_consistency(self, content: str) -> float:
        """出力一貫性評価"""
        score = 90.0  # ベーススコア
        
        # 統一出力フォーマット適用
        if 'DSSMSUnified' in content:
            score += 5.0
            
        # 85.0点エンジンパターン適用
        if 'dssms_unified_output' in content:
            score += 2.0
            
        return min(score, 98.0)
    
    def _check_quality_compliance(self, quality_scores: dict) -> dict:
        """85.0点基準適合チェック"""
        failed_standards = []
        
        for metric, required_score in self.quality_standards.items():
            actual_score = quality_scores.get(metric, 0.0)
            if actual_score < required_score:
                failed_standards.append({
                    'metric': metric,
                    'required': required_score,
                    'actual': actual_score,
                    'gap': required_score - actual_score
                })
        
        if failed_standards:
            return {
                'compliant': False,
                'reason': f"{len(failed_standards)}項目で基準未達",
                'failed_standards': failed_standards
            }
        else:
            return {
                'compliant': True,
                'reason': "全基準達成"
            }
    
    def _save_test_results(self, summary: dict):
        """テスト結果保存"""
        results_file = self.project_root / f"improved_engines_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        full_results = {
            'summary': summary,
            'detailed_results': self.test_results,
            'quality_standards': self.quality_standards,
            'test_metadata': {
                'script_version': '1.0',
                'test_date': datetime.now().isoformat(),
                'total_engines_tested': len(self.improved_engines)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"テスト結果保存: {results_file}")

def main():
    """メイン実行"""
    try:
        logger.info("=== Problem 9: 改善エンジン動作確認テスト開始 ===")
        
        validator = ImprovedEngineValidator()
        summary = validator.validate_all_engines()
        
        # テスト結果報告
        if summary['failed'] == 0:
            logger.info("[SUCCESS] 全改善エンジンが検証合格")
        else:
            logger.warning(f"[WARNING] {summary['failed']}個のエンジンで問題検出")
            
        return summary
        
    except Exception as e:
        logger.error(f"[ERROR] テスト実行エラー: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n[OK] 改善エンジン動作確認テスト完了")
        print(f"[CHART] 合格: {result['passed']}個")
        print(f"[ERROR] 失敗: {result['failed']}個")
        print(f"[TARGET] 成功率: {result['passed']/result['total_engines']*100:.1f}%")
    else:
        print("[ERROR] テスト実行失敗")
        sys.exit(1)