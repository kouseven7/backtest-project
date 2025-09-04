"""
DSSMS Enhanced Backtester
Phase 2.3 Task 2.3.1: バックテストデータ収集最適化

Purpose:
  - DSSMSバックテストの品質向上統合システム
  - データ品質向上 + 出力検証の統合実行
  - 既存DSSMSバックテスターとの完全統合
  - エラーフリーなバックテスト実行

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - src/dssms/data_quality_enhancer.py との連携
  - src/dssms/output_validator.py との連携
  - 既存src/dssms/dssms_backtester.py との統合
  - main.pyからの呼び出し対応
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.data_quality_enhancer import DSSMSDataQualityEnhancer, DSSMSQualityConfig
from src.dssms.output_validator import DSSMSOutputValidator, DSSMSOutputConfig


class DSSMSEnhancedBacktester:
    """DSSMS品質保証付きバックテスター"""
    
    def __init__(self, initial_capital: float = 1000000.0, 
                 quality_config: Optional[DSSMSQualityConfig] = None,
                 validation_config: Optional[DSSMSOutputConfig] = None):
        """
        初期化
        
        Args:
            initial_capital: 初期資本金
            quality_config: データ品質設定
            validation_config: 出力検証設定
        """
        self.initial_capital = initial_capital
        self.logger = setup_logger(__name__)
        
        # 品質向上システム
        self.quality_enhancer = DSSMSDataQualityEnhancer(
            config=quality_config or DSSMSQualityConfig(),
            initial_capital=initial_capital
        )
        
        # 出力検証システム
        self.output_validator = DSSMSOutputValidator(
            config=validation_config or DSSMSOutputConfig(),
            initial_capital=initial_capital
        )
        
        # DSSMS戦略リスト
        self.dssms_strategies = [
            "VWAPBreakoutStrategy",
            "BreakoutStrategy", 
            "OpeningGapStrategy",
            "MomentumInvestingStrategy",
            "VWAPBounceStrategy",
            "ContrarianStrategy",
            "GCStrategy"
        ]
        
        self.logger.info("DSSMS Enhanced Backtester 初期化完了")
    
    def run_enhanced_backtest(self, input_data: pd.DataFrame, 
                            strategy_name: Optional[str] = None,
                            use_existing_backtester: bool = True) -> Dict[str, Any]:
        """
        品質保証付きDSSMSバックテスト実行
        
        Args:
            input_data: 入力データ
            strategy_name: 対象戦略名
            use_existing_backtester: 既存DSSMSバックテスター使用フラグ
        
        Returns:
            Dict[str, Any]: 品質保証済みバックテスト結果
        """
        try:
            self.logger.info(f"品質保証付きDSSMSバックテスト開始: {len(input_data)}行")
            
            # ステップ1: データ品質向上
            enhanced_data, quality_metrics = self.quality_enhancer.enhance_dssms_data(
                input_data, strategy_name
            )
            
            # ステップ2: 既存DSSMSバックテスター実行（オプション）
            if use_existing_backtester:
                backtest_result = self._run_with_existing_backtester(enhanced_data, strategy_name)
            else:
                backtest_result = self._run_basic_backtest(enhanced_data, strategy_name)
            
            # ステップ3: 出力検証
            validated_result, validation_metrics = self.output_validator.validate_dssms_output(
                backtest_result
            )
            
            # ステップ4: 統合結果の構築
            final_result = self._build_integrated_result(
                validated_result, quality_metrics, validation_metrics, enhanced_data
            )
            
            self.logger.info("品質保証付きバックテスト完了")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Enhanced Backtester実行中にエラー: {e}")
            return self._create_error_result(str(e))
    
    def _run_with_existing_backtester(self, data: pd.DataFrame, strategy_name: Optional[str]) -> Dict[str, Any]:
        """既存DSSMSバックテスターとの統合実行"""
        try:
            # 既存DSSMSバックテスターのインポートを試行
            try:
                from src.dssms.dssms_backtester import DSSMSBacktester
                
                # 既存バックテスター実行
                dssms_backtester = DSSMSBacktester()
                result = dssms_backtester.run_backtest(data)
                
                self.logger.info("既存DSSMSバックテスターで実行完了")
                return result
                
            except ImportError as ie:
                self.logger.warning(f"既存DSSMSバックテスター読み込み失敗: {ie}")
                return self._run_basic_backtest(data, strategy_name)
                
        except Exception as e:
            self.logger.error(f"既存バックテスター実行中にエラー: {e}")
            return self._run_basic_backtest(data, strategy_name)
    
    def _run_basic_backtest(self, data: pd.DataFrame, strategy_name: Optional[str]) -> Dict[str, Any]:
        """基本バックテスト実行（フォールバック）"""
        try:
            # MainDataExtractorを使用した基本バックテスト
            trades = self.quality_enhancer.main_extractor.extract_accurate_trades(data)
            
            # 基本統計の計算
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            total_profit_loss = sum(t.get('profit_loss', 0) for t in trades) * self.initial_capital
            
            # 基本結果構築
            result = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit_loss': total_profit_loss,
                'trades': trades,
                'data': data,
                'strategy': strategy_name or 'DSSMS_Combined',
                'backtest_type': 'basic_enhanced'
            }
            
            self.logger.info(f"基本バックテスト完了: {total_trades}取引, 勝率{win_rate:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"基本バックテスト実行中にエラー: {e}")
            return self._create_error_result(str(e))
    
    def _build_integrated_result(self, backtest_result: Dict[str, Any],
                               quality_metrics: Any,
                               validation_metrics: Any,
                               enhanced_data: pd.DataFrame) -> Dict[str, Any]:
        """統合結果の構築"""
        try:
            # 基本結果の拡張
            integrated_result = backtest_result.copy()
            
            # 品質保証メタデータの追加
            integrated_result['quality_assurance'] = {
                'data_quality_score': quality_metrics.quality_score,
                'data_enhancement_applied': quality_metrics.enhancement_applied,
                'validation_score': validation_metrics.validation_score,
                'output_validity': validation_metrics.is_valid_output,
                'total_corrections': quality_metrics.corrected_records,
                'validation_errors': validation_metrics.validation_errors
            }
            
            # 拡張データの保存
            integrated_result['enhanced_data'] = enhanced_data
            
            # 品質保証サマリー
            qa_summary = self._generate_qa_summary(quality_metrics, validation_metrics)
            integrated_result['qa_summary'] = qa_summary
            
            # 信頼性スコアの計算
            reliability_score = (quality_metrics.quality_score + validation_metrics.validation_score) / 2.0
            integrated_result['reliability_score'] = reliability_score
            
            self.logger.info(f"統合結果構築完了: 信頼性スコア {reliability_score:.3f}")
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"統合結果構築中にエラー: {e}")
            return backtest_result
    
    def _generate_qa_summary(self, quality_metrics: Any, validation_metrics: Any) -> Dict[str, Any]:
        """品質保証サマリー生成"""
        try:
            return {
                'overall_quality': 'EXCELLENT' if quality_metrics.quality_score >= 0.9 else
                                 'GOOD' if quality_metrics.quality_score >= 0.75 else
                                 'ACCEPTABLE' if quality_metrics.quality_score >= 0.6 else 'POOR',
                
                'output_validity': 'VALID' if validation_metrics.is_valid_output else 'INVALID',
                
                'data_corrections': quality_metrics.corrected_records,
                'data_issues': quality_metrics.invalid_records,
                
                'validation_passed': len(validation_metrics.validation_errors) == 0,
                'critical_errors': [err for err in validation_metrics.validation_errors 
                                  if 'エラー' in err or 'ERROR' in err.upper()],
                
                'recommendation': self._get_qa_recommendation(quality_metrics, validation_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"QAサマリー生成中にエラー: {e}")
            return {'overall_quality': 'UNKNOWN', 'output_validity': 'UNKNOWN', 
                   'recommendation': 'システムエラーが発生しました'}
    
    def _get_qa_recommendation(self, quality_metrics: Any, validation_metrics: Any) -> str:
        """品質保証に基づく推奨事項"""
        try:
            if (quality_metrics.quality_score >= 0.85 and 
                validation_metrics.is_valid_output and 
                len(validation_metrics.validation_errors) == 0):
                return "結果は高品質で信頼性が高く、本番利用に適しています"
            
            elif quality_metrics.quality_score >= 0.7 and validation_metrics.validation_score >= 0.7:
                return "結果は品質基準を満たしており、注意深く利用することを推奨します"
            
            elif quality_metrics.corrected_records > quality_metrics.total_records * 0.3:
                return "データに多くの修正が適用されました。元データの見直しを推奨します"
            
            elif len(validation_metrics.validation_errors) > 0:
                return "出力検証でエラーが検出されました。結果の詳細確認を推奨します"
            
            else:
                return "品質スコアが低いため、パラメータ調整や戦略見直しを推奨します"
                
        except Exception as e:
            self.logger.error(f"推奨事項生成中にエラー: {e}")
            return "推奨事項の生成中にエラーが発生しました"
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果の作成"""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_profit_loss': 0.0,
            'trades': [],
            'error': error_message,
            'quality_assurance': {
                'data_quality_score': 0.0,
                'validation_score': 0.0,
                'output_validity': False,
                'validation_errors': [error_message]
            },
            'reliability_score': 0.0
        }
    
    def run_multi_strategy_enhanced_backtest(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """複数戦略対応の品質保証付きバックテスト"""
        try:
            self.logger.info(f"複数戦略品質保証バックテスト開始: {len(data_dict)}戦略")
            
            strategy_results = {}
            overall_qa_metrics = {
                'total_strategies': len(data_dict),
                'successful_strategies': 0,
                'failed_strategies': 0,
                'average_quality_score': 0.0,
                'average_validation_score': 0.0
            }
            
            quality_scores = []
            validation_scores = []
            
            for strategy_name, strategy_data in data_dict.items():
                try:
                    result = self.run_enhanced_backtest(strategy_data, strategy_name)
                    strategy_results[strategy_name] = result
                    
                    if 'quality_assurance' in result:
                        quality_scores.append(result['quality_assurance']['data_quality_score'])
                        validation_scores.append(result['quality_assurance']['validation_score'])
                        overall_qa_metrics['successful_strategies'] += 1
                    else:
                        overall_qa_metrics['failed_strategies'] += 1
                    
                except Exception as e:
                    self.logger.error(f"戦略{strategy_name}のバックテスト中にエラー: {e}")
                    strategy_results[strategy_name] = self._create_error_result(str(e))
                    overall_qa_metrics['failed_strategies'] += 1
            
            # 全体メトリクス計算
            if quality_scores:
                overall_qa_metrics['average_quality_score'] = np.mean(quality_scores)
                overall_qa_metrics['average_validation_score'] = np.mean(validation_scores)
            
            # 統合結果
            integrated_result = {
                'strategy_results': strategy_results,
                'overall_qa_metrics': overall_qa_metrics,
                'execution_timestamp': datetime.now().isoformat(),
                'enhancement_version': '1.0'
            }
            
            self.logger.info(f"複数戦略バックテスト完了: {overall_qa_metrics['successful_strategies']}/{overall_qa_metrics['total_strategies']}成功")
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"複数戦略バックテスト実行中にエラー: {e}")
            return {'error': str(e), 'strategy_results': {}}
    
    def test_enhanced_backtester(self) -> bool:
        """Enhanced Backtesterのテスト"""
        try:
            self.logger.info("DSSMS Enhanced Backtester テスト開始")
            
            # テストデータ生成
            dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
            test_data = pd.DataFrame({
                'Close': [100, 102, 98, 105, 103, 107, 109, 106],
                'Entry_Signal': [1, 0, 0, 1, 0, 0, 0, 0],
                'Exit_Signal': [0, 1, 0, 0, 1, 0, 1, 0],
                'Strategy': ['TestStrategy'] * 8
            }, index=dates[:8])
            
            # Enhanced Backtester実行
            result = self.run_enhanced_backtest(test_data, "TestStrategy", use_existing_backtester=False)
            
            # 結果検証
            success = (
                'quality_assurance' in result and
                'reliability_score' in result and
                result.get('total_trades', 0) > 0 and
                isinstance(result.get('reliability_score', 0), (int, float))
            )
            
            self.logger.info(f"テスト{'成功' if success else '失敗'}: 信頼性スコア {result.get('reliability_score', 0):.3f}")
            return success
            
        except Exception as e:
            self.logger.error(f"テスト中にエラー: {e}")
            return False


if __name__ == "__main__":
    # テスト実行
    backtester = DSSMSEnhancedBacktester()
    backtester.test_enhanced_backtester()
