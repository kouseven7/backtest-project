"""
DSSMS Integration Manager
Phase 2.3 Task 2.3.1: バックテストデータ収集最適化

Purpose:
  - DSSMS品質保証システムの統合管理
  - main.pyとの完全統合インターフェース
  - 出力システムとの連携管理
  - 総合品質レポーティング

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - main.pyからの呼び出し対応
  - output/dssms_excel_exporter_v2.pyとの連携
  - src/dssms/enhanced_backtester.pyとの統合
  - 既存DSSMSシステムとの完全互換性
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.enhanced_backtester import DSSMSEnhancedBacktester
from src.dssms.data_quality_enhancer import DSSMSQualityConfig
from src.dssms.output_validator import DSSMSOutputConfig


class DSSMSIntegrationManager:
    """DSSMS品質保証システム統合管理"""
    
    def __init__(self, initial_capital: float = 1000000.0, 
                 auto_export: bool = True,
                 output_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            initial_capital: 初期資本金
            auto_export: 自動Excel出力フラグ
            output_dir: 出力ディレクトリ
        """
        self.initial_capital = initial_capital
        self.auto_export = auto_export
        self.output_dir = output_dir or "backtest_results/dssms_enhanced"
        self.logger = setup_logger(__name__)
        
        # Enhanced Backtester初期化
        self.enhanced_backtester = DSSMSEnhancedBacktester(initial_capital=initial_capital)
        
        # 実行履歴
        self.execution_history = []
        
        # 品質基準設定
        self.quality_standards = {
            'minimum_quality_score': 0.75,
            'minimum_validation_score': 0.70,
            'minimum_reliability_score': 0.80,
            'maximum_allowed_errors': 5
        }
        
        self.logger.info("DSSMS Integration Manager 初期化完了")
    
    def execute_dssms_with_qa(self, stock_data: pd.DataFrame, 
                            strategy_name: Optional[str] = None,
                            custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        品質保証付きDSSMS実行（main.py統合用）
        
        Args:
            stock_data: 株価データ
            strategy_name: 戦略名
            custom_config: カスタム設定
        
        Returns:
            Dict[str, Any]: 品質保証済み実行結果
        """
        try:
            execution_id = f"dssms_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"DSSMS品質保証実行開始: {execution_id}")
            
            # カスタム設定の適用
            if custom_config:
                self._apply_custom_config(custom_config)
            
            # Enhanced Backtester実行
            result = self.enhanced_backtester.run_enhanced_backtest(
                stock_data, strategy_name, use_existing_backtester=True
            )
            
            # 品質基準チェック
            quality_check = self._check_quality_standards(result)
            
            # 実行結果の強化
            enhanced_result = self._enhance_execution_result(result, execution_id, quality_check)
            
            # 自動出力（設定されている場合）
            if self.auto_export:
                export_path = self._auto_export_results(enhanced_result, execution_id)
                enhanced_result['export_path'] = export_path
            
            # 実行履歴への追加
            self._add_to_execution_history(enhanced_result, execution_id)
            
            self.logger.info(f"DSSMS品質保証実行完了: {execution_id}")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"DSSMS品質保証実行中にエラー: {e}")
            return self._create_error_result(str(e))
    
    def execute_multi_strategy_dssms(self, strategy_data_dict: Dict[str, pd.DataFrame],
                                   custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        複数戦略DSSMS品質保証実行
        
        Args:
            strategy_data_dict: 戦略別データ辞書
            custom_config: カスタム設定
        
        Returns:
            Dict[str, Any]: 複数戦略実行結果
        """
        try:
            execution_id = f"multi_dssms_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"複数戦略DSSMS品質保証実行開始: {execution_id}, {len(strategy_data_dict)}戦略")
            
            # カスタム設定の適用
            if custom_config:
                self._apply_custom_config(custom_config)
            
            # 複数戦略Enhanced Backtester実行
            result = self.enhanced_backtester.run_multi_strategy_enhanced_backtest(strategy_data_dict)
            
            # 全体品質チェック
            overall_quality_check = self._check_multi_strategy_quality(result)
            
            # 結果の強化
            enhanced_result = self._enhance_multi_strategy_result(result, execution_id, overall_quality_check)
            
            # 自動出力
            if self.auto_export:
                export_path = self._auto_export_multi_strategy_results(enhanced_result, execution_id)
                enhanced_result['export_path'] = export_path
            
            # 実行履歴への追加
            self._add_to_execution_history(enhanced_result, execution_id)
            
            self.logger.info(f"複数戦略DSSMS品質保証実行完了: {execution_id}")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"複数戦略DSSMS実行中にエラー: {e}")
            return self._create_error_result(str(e))
    
    def _apply_custom_config(self, config: Dict[str, Any]) -> None:
        """カスタム設定の適用"""
        try:
            # 品質設定の更新
            if 'quality_config' in config:
                quality_config = DSSMSQualityConfig(**config['quality_config'])
                self.enhanced_backtester.quality_enhancer.config = quality_config
            
            # 検証設定の更新
            if 'validation_config' in config:
                validation_config = DSSMSOutputConfig(**config['validation_config'])
                self.enhanced_backtester.output_validator.config = validation_config
            
            # 品質基準の更新
            if 'quality_standards' in config:
                self.quality_standards.update(config['quality_standards'])
            
            # 出力設定の更新
            if 'auto_export' in config:
                self.auto_export = config['auto_export']
            
            if 'output_dir' in config:
                self.output_dir = config['output_dir']
            
            self.logger.info("カスタム設定を適用しました")
            
        except Exception as e:
            self.logger.error(f"カスタム設定適用中にエラー: {e}")
    
    def _check_quality_standards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """品質基準チェック"""
        try:
            qa_info = result.get('quality_assurance', {})
            
            quality_check = {
                'meets_quality_standard': True,
                'meets_validation_standard': True,
                'meets_reliability_standard': True,
                'within_error_limit': True,
                'overall_pass': True,
                'issues': []
            }
            
            # 品質スコアチェック
            quality_score = qa_info.get('data_quality_score', 0.0)
            if quality_score < self.quality_standards['minimum_quality_score']:
                quality_check['meets_quality_standard'] = False
                quality_check['issues'].append(f"品質スコア不足: {quality_score:.3f} < {self.quality_standards['minimum_quality_score']}")
            
            # 検証スコアチェック
            validation_score = qa_info.get('validation_score', 0.0)
            if validation_score < self.quality_standards['minimum_validation_score']:
                quality_check['meets_validation_standard'] = False
                quality_check['issues'].append(f"検証スコア不足: {validation_score:.3f} < {self.quality_standards['minimum_validation_score']}")
            
            # 信頼性スコアチェック
            reliability_score = result.get('reliability_score', 0.0)
            if reliability_score < self.quality_standards['minimum_reliability_score']:
                quality_check['meets_reliability_standard'] = False
                quality_check['issues'].append(f"信頼性スコア不足: {reliability_score:.3f} < {self.quality_standards['minimum_reliability_score']}")
            
            # エラー数チェック
            error_count = len(qa_info.get('validation_errors', []))
            if error_count > self.quality_standards['maximum_allowed_errors']:
                quality_check['within_error_limit'] = False
                quality_check['issues'].append(f"エラー数超過: {error_count} > {self.quality_standards['maximum_allowed_errors']}")
            
            # 総合判定
            quality_check['overall_pass'] = all([
                quality_check['meets_quality_standard'],
                quality_check['meets_validation_standard'],
                quality_check['meets_reliability_standard'],
                quality_check['within_error_limit']
            ])
            
            return quality_check
            
        except Exception as e:
            self.logger.error(f"品質基準チェック中にエラー: {e}")
            return {'overall_pass': False, 'issues': [f"品質チェックエラー: {e}"]}
    
    def _check_multi_strategy_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """複数戦略品質チェック"""
        try:
            overall_metrics = result.get('overall_qa_metrics', {})
            
            quality_check = {
                'success_rate_acceptable': True,
                'average_quality_acceptable': True,
                'average_validation_acceptable': True,
                'overall_pass': True,
                'issues': []
            }
            
            # 成功率チェック
            total_strategies = overall_metrics.get('total_strategies', 0)
            successful_strategies = overall_metrics.get('successful_strategies', 0)
            success_rate = successful_strategies / total_strategies if total_strategies > 0 else 0.0
            
            if success_rate < 0.8:  # 80%以上の成功率を要求
                quality_check['success_rate_acceptable'] = False
                quality_check['issues'].append(f"戦略成功率低下: {success_rate:.1%} < 80%")
            
            # 平均品質スコアチェック
            avg_quality = overall_metrics.get('average_quality_score', 0.0)
            if avg_quality < self.quality_standards['minimum_quality_score']:
                quality_check['average_quality_acceptable'] = False
                quality_check['issues'].append(f"平均品質スコア不足: {avg_quality:.3f}")
            
            # 平均検証スコアチェック
            avg_validation = overall_metrics.get('average_validation_score', 0.0)
            if avg_validation < self.quality_standards['minimum_validation_score']:
                quality_check['average_validation_acceptable'] = False
                quality_check['issues'].append(f"平均検証スコア不足: {avg_validation:.3f}")
            
            # 総合判定
            quality_check['overall_pass'] = all([
                quality_check['success_rate_acceptable'],
                quality_check['average_quality_acceptable'],
                quality_check['average_validation_acceptable']
            ])
            
            return quality_check
            
        except Exception as e:
            self.logger.error(f"複数戦略品質チェック中にエラー: {e}")
            return {'overall_pass': False, 'issues': [f"複数戦略品質チェックエラー: {e}"]}
    
    def _enhance_execution_result(self, result: Dict[str, Any], execution_id: str, 
                                quality_check: Dict[str, Any]) -> Dict[str, Any]:
        """実行結果の強化"""
        try:
            enhanced = result.copy()
            
            # 実行メタデータの追加
            enhanced['execution_metadata'] = {
                'execution_id': execution_id,
                'execution_timestamp': datetime.now().isoformat(),
                'integration_manager_version': '1.0',
                'quality_check_passed': quality_check['overall_pass'],
                'quality_issues': quality_check.get('issues', [])
            }
            
            # 推奨アクションの追加
            enhanced['recommended_actions'] = self._generate_recommended_actions(result, quality_check)
            
            # 要約統計の追加
            enhanced['summary_statistics'] = self._generate_summary_statistics(result)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"実行結果強化中にエラー: {e}")
            return result
    
    def _enhance_multi_strategy_result(self, result: Dict[str, Any], execution_id: str,
                                     quality_check: Dict[str, Any]) -> Dict[str, Any]:
        """複数戦略実行結果の強化"""
        try:
            enhanced = result.copy()
            
            # 実行メタデータの追加
            enhanced['execution_metadata'] = {
                'execution_id': execution_id,
                'execution_timestamp': datetime.now().isoformat(),
                'integration_manager_version': '1.0',
                'multi_strategy_execution': True,
                'overall_quality_check_passed': quality_check['overall_pass'],
                'quality_issues': quality_check.get('issues', [])
            }
            
            # 戦略別要約の生成
            enhanced['strategy_summary'] = self._generate_strategy_summary(result)
            
            # 推奨アクションの追加
            enhanced['recommended_actions'] = self._generate_multi_strategy_recommendations(result, quality_check)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"複数戦略結果強化中にエラー: {e}")
            return result
    
    def _generate_recommended_actions(self, result: Dict[str, Any], quality_check: Dict[str, Any]) -> list:
        """推奨アクション生成"""
        actions = []
        
        try:
            if not quality_check['overall_pass']:
                actions.append("品質基準未達のため、パラメータ調整を検討してください")
            
            qa_summary = result.get('qa_summary', {})
            if qa_summary.get('overall_quality') == 'POOR':
                actions.append("データ品質が低いため、入力データの見直しを推奨します")
            
            reliability_score = result.get('reliability_score', 0.0)
            if reliability_score < 0.6:
                actions.append("信頼性スコアが低いため、戦略の再評価を推奨します")
            
            if not actions:
                actions.append("結果は品質基準を満たしています。本番運用を検討できます")
            
        except Exception as e:
            self.logger.error(f"推奨アクション生成中にエラー: {e}")
            actions.append("推奨アクションの生成中にエラーが発生しました")
        
        return actions
    
    def _generate_multi_strategy_recommendations(self, result: Dict[str, Any], quality_check: Dict[str, Any]) -> list:
        """複数戦略推奨アクション生成"""
        actions = []
        
        try:
            overall_metrics = result.get('overall_qa_metrics', {})
            
            if overall_metrics.get('failed_strategies', 0) > 0:
                actions.append("失敗した戦略の原因分析と修正を推奨します")
            
            avg_quality = overall_metrics.get('average_quality_score', 0.0)
            if avg_quality < 0.7:
                actions.append("平均品質スコアが低いため、全戦略の見直しを推奨します")
            
            if not quality_check['overall_pass']:
                actions.append("品質基準未達の戦略があります。個別分析を実施してください")
            
            if not actions:
                actions.append("全戦略が品質基準を満たしています。統合運用を検討できます")
            
        except Exception as e:
            self.logger.error(f"複数戦略推奨アクション生成中にエラー: {e}")
            actions.append("複数戦略推奨アクションの生成中にエラーが発生しました")
        
        return actions
    
    def _generate_summary_statistics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """要約統計生成"""
        try:
            return {
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0.0),
                'total_profit_loss': result.get('total_profit_loss', 0.0),
                'reliability_score': result.get('reliability_score', 0.0),
                'data_quality_score': result.get('quality_assurance', {}).get('data_quality_score', 0.0),
                'validation_score': result.get('quality_assurance', {}).get('validation_score', 0.0)
            }
        except Exception as e:
            self.logger.error(f"要約統計生成中にエラー: {e}")
            return {}
    
    def _generate_strategy_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """戦略別要約生成"""
        try:
            strategy_results = result.get('strategy_results', {})
            summary = {}
            
            for strategy_name, strategy_result in strategy_results.items():
                summary[strategy_name] = {
                    'total_trades': strategy_result.get('total_trades', 0),
                    'win_rate': strategy_result.get('win_rate', 0.0),
                    'profit_loss': strategy_result.get('total_profit_loss', 0.0),
                    'reliability_score': strategy_result.get('reliability_score', 0.0),
                    'quality_passed': strategy_result.get('quality_assurance', {}).get('output_validity', False)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"戦略別要約生成中にエラー: {e}")
            return {}
    
    def _auto_export_results(self, result: Dict[str, Any], execution_id: str) -> str:
        """自動結果出力"""
        try:
            # 出力ディレクトリの作成
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # JSON形式で結果保存
            json_path = output_path / f"{execution_id}_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"結果をJSON形式で出力: {json_path}")
            
            # Excel出力の試行
            try:
                from src.dssms.dssms_excel_exporter import DSSMSExcelExporter
                
                exporter = DSSMSExcelExporter(initial_capital=self.initial_capital)
                excel_path = exporter.export_dssms_results(result, None)
                self.logger.info(f"結果を統合Excel形式で出力: {excel_path}")
                return str(excel_path)
                
            except ImportError:
                self.logger.warning("Excel出力システムが利用できません")
                return str(json_path)
            
        except Exception as e:
            self.logger.error(f"自動出力中にエラー: {e}")
            return ""
    
    def _auto_export_multi_strategy_results(self, result: Dict[str, Any], execution_id: str) -> str:
        """複数戦略結果の自動出力"""
        try:
            # 基本的にはJSONでの出力
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            json_path = output_path / f"{execution_id}_multi_strategy_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"複数戦略結果をJSON形式で出力: {json_path}")
            return str(json_path)
            
        except Exception as e:
            self.logger.error(f"複数戦略自動出力中にエラー: {e}")
            return ""
    
    def _add_to_execution_history(self, result: Dict[str, Any], execution_id: str) -> None:
        """実行履歴への追加"""
        try:
            history_entry = {
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'total_trades': result.get('total_trades', 0),
                'reliability_score': result.get('reliability_score', 0.0),
                'quality_passed': result.get('execution_metadata', {}).get('quality_check_passed', False)
            }
            
            self.execution_history.append(history_entry)
            
            # 履歴が長くなりすぎないよう制限
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
        except Exception as e:
            self.logger.error(f"実行履歴追加中にエラー: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果の作成"""
        return {
            'error': error_message,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_profit_loss': 0.0,
            'reliability_score': 0.0,
            'quality_assurance': {
                'data_quality_score': 0.0,
                'validation_score': 0.0,
                'output_validity': False
            },
            'execution_metadata': {
                'execution_timestamp': datetime.now().isoformat(),
                'error_occurred': True
            }
        }
    
    def get_execution_history(self) -> list:
        """実行履歴の取得"""
        return self.execution_history.copy()
    
    def test_integration_manager(self) -> bool:
        """Integration Managerのテスト"""
        try:
            self.logger.info("DSSMS Integration Manager テスト開始")
            
            # テストデータ生成
            dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
            test_data = pd.DataFrame({
                'Close': [100, 102, 98, 105, 103, 107, 109, 106],
                'Entry_Signal': [1, 0, 0, 1, 0, 0, 0, 0],
                'Exit_Signal': [0, 1, 0, 0, 1, 0, 1, 0],
                'Strategy': ['TestStrategy'] * 8
            }, index=dates[:8])
            
            # Integration Manager実行
            result = self.execute_dssms_with_qa(test_data, "TestStrategy", {'auto_export': False})
            
            # 結果検証
            success = (
                'execution_metadata' in result and
                'recommended_actions' in result and
                'summary_statistics' in result and
                isinstance(result.get('reliability_score', 0), (int, float))
            )
            
            self.logger.info(f"テスト{'成功' if success else '失敗'}: 信頼性スコア {result.get('reliability_score', 0):.3f}")
            return success
            
        except Exception as e:
            self.logger.error(f"テスト中にエラー: {e}")
            return False


if __name__ == "__main__":
    # テスト実行
    manager = DSSMSIntegrationManager(auto_export=False)
    manager.test_integration_manager()
