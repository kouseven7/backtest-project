"""
DSSMS Data Quality Enhancer
Phase 2.3 Task 2.3.1: バックテストデータ収集最適化

Purpose:
  - DSSMSバックテスト結果データの品質向上
  - output/data_extraction_enhancer.pyの実績を活用
  - データ整合性の自動検証・修正
  - 無効データの自動補正

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - 既存DSSMSシステムとの完全互換性保持
  - output/data_extraction_enhancer.pyの成功パターン拡張
  - src/dssms/dssms_backtester.pyとの統合対応
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, field
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from output.data_extraction_enhancer import MainDataExtractor
from config.logger_config import setup_logger


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    corrected_records: int = 0
    missing_signals: int = 0
    inconsistent_prices: int = 0
    quality_score: float = 0.0
    enhancement_applied: bool = False


@dataclass
class DSSMSQualityConfig:
    """DSSMS品質設定"""
    min_price_threshold: float = 0.01
    max_price_change_ratio: float = 0.30  # 30%以上の価格変動を異常とする
    signal_consistency_check: bool = True
    auto_correction_enabled: bool = True
    quality_threshold: float = 0.85  # 85%以上の品質スコアを目標
    fallback_to_previous: bool = True


class DSSMSDataQualityEnhancer:
    """DSSMS専用データ品質向上システム"""
    
    def __init__(self, config: Optional[DSSMSQualityConfig] = None, initial_capital: float = 1000000.0):
        """
        初期化
        
        Args:
            config: 品質設定（Noneの場合はデフォルト使用）
            initial_capital: 初期資本金
        """
        self.config = config or DSSMSQualityConfig()
        self.initial_capital = initial_capital
        self.logger = setup_logger(__name__)
        
        # MainDataExtractorを活用（実績あるシステム）
        self.main_extractor = MainDataExtractor(initial_capital)
        
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
        
        self.logger.info("DSSMS Data Quality Enhancer 初期化完了")
    
    def enhance_dssms_data(self, dssms_data: pd.DataFrame, 
                          strategy_name: Optional[str] = None) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """
        DSSMSデータの品質向上処理
        
        Args:
            dssms_data: DSSMS生データ
            strategy_name: 対象戦略名（Noneの場合は全戦略対象）
        
        Returns:
            Tuple[pd.DataFrame, DataQualityMetrics]: 改善済みデータと品質メトリクス
        """
        try:
            self.logger.info(f"DSSMS データ品質向上開始: {len(dssms_data)}行")
            
            # メトリクス初期化
            metrics = DataQualityMetrics(total_records=len(dssms_data))
            
            # データが空の場合の処理
            if dssms_data.empty:
                self.logger.warning("空のDataFrameが渡されました")
                return dssms_data.copy(), metrics
            
            # データのコピーを作成
            enhanced_data = dssms_data.copy()
            
            # ステップ1: 基本品質チェック
            enhanced_data, basic_metrics = self._perform_basic_quality_check(enhanced_data)
            
            # ステップ2: 価格データの整合性チェック
            enhanced_data, price_metrics = self._validate_price_consistency(enhanced_data)
            
            # ステップ3: シグナル整合性の検証
            enhanced_data, signal_metrics = self._validate_signal_consistency(enhanced_data)
            
            # ステップ4: MainDataExtractorを活用した検証
            enhanced_data, extraction_metrics = self._validate_with_main_extractor(enhanced_data)
            
            # ステップ5: 最終品質スコア計算
            metrics = self._calculate_quality_metrics(enhanced_data, basic_metrics, 
                                                    price_metrics, signal_metrics, extraction_metrics)
            
            # ステップ6: 品質レポート
            self._generate_quality_report(metrics, strategy_name)
            
            self.logger.info(f"データ品質向上完了: 品質スコア {metrics.quality_score:.3f}")
            return enhanced_data, metrics
            
        except Exception as e:
            self.logger.error(f"データ品質向上中にエラー: {e}")
            return dssms_data.copy(), DataQualityMetrics(total_records=len(dssms_data))
    
    def _perform_basic_quality_check(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """基本品質チェック"""
        try:
            metrics = {"corrected": 0, "invalid": 0}
            
            # 必要な列の存在確認
            required_columns = ['Close', 'Entry_Signal', 'Exit_Signal']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.warning(f"必要な列が不足: {missing_columns}")
                for col in missing_columns:
                    data[col] = 0
                    metrics["corrected"] += len(data)
            
            # 価格データの基本検証
            if 'Close' in data.columns:
                # 負の価格や極端に小さい価格を修正
                invalid_prices = data['Close'] <= self.config.min_price_threshold
                if invalid_prices.any():
                    invalid_count = invalid_prices.sum()
                    metrics["invalid"] += invalid_count
                    
                    if self.config.auto_correction_enabled and self.config.fallback_to_previous:
                        data.loc[invalid_prices, 'Close'] = data['Close'].shift(1)
                        metrics["corrected"] += invalid_count
                        self.logger.info(f"無効価格を前日価格で補正: {invalid_count}件")
            
            # NaN値の処理
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                data = data.fillna(method='ffill').fillna(method='bfill')
                metrics["corrected"] += nan_count
                self.logger.info(f"NaN値を補正: {nan_count}件")
            
            return data, metrics
            
        except Exception as e:
            self.logger.error(f"基本品質チェック中にエラー: {e}")
            return data, {"corrected": 0, "invalid": 0}
    
    def _validate_price_consistency(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """価格データ整合性検証"""
        try:
            metrics = {"price_anomalies": 0, "corrected": 0}
            
            if 'Close' not in data.columns or len(data) < 2:
                return data, metrics
            
            # 価格変動率の計算
            price_changes = data['Close'].pct_change().abs()
            
            # 異常な価格変動の検出
            anomalies = price_changes > self.config.max_price_change_ratio
            anomaly_count = anomalies.sum()
            
            if anomaly_count > 0:
                metrics["price_anomalies"] = anomaly_count
                
                if self.config.auto_correction_enabled:
                    # 異常な価格変動を平滑化
                    data.loc[anomalies, 'Close'] = data['Close'].rolling(window=3, center=True).mean()
                    data['Close'] = data['Close'].fillna(method='ffill').fillna(method='bfill')
                    metrics["corrected"] = anomaly_count
                    self.logger.info(f"価格異常を平滑化で修正: {anomaly_count}件")
            
            return data, metrics
            
        except Exception as e:
            self.logger.error(f"価格整合性検証中にエラー: {e}")
            return data, {"price_anomalies": 0, "corrected": 0}
    
    def _validate_signal_consistency(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """シグナル整合性検証"""
        try:
            metrics = {"signal_conflicts": 0, "corrected": 0}
            
            if not self.config.signal_consistency_check:
                return data, metrics
            
            # Entry_SignalとExit_Signalの同時発生チェック
            if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
                conflicts = (data['Entry_Signal'] == 1) & (data['Exit_Signal'] == 1)
                conflict_count = conflicts.sum()
                
                if conflict_count > 0:
                    metrics["signal_conflicts"] = conflict_count
                    
                    if self.config.auto_correction_enabled:
                        # Exit_Signalを優先（安全性重視）
                        data.loc[conflicts, 'Entry_Signal'] = 0
                        metrics["corrected"] = conflict_count
                        self.logger.info(f"シグナル競合を修正（Exit優先）: {conflict_count}件")
            
            return data, metrics
            
        except Exception as e:
            self.logger.error(f"シグナル整合性検証中にエラー: {e}")
            return data, {"signal_conflicts": 0, "corrected": 0}
    
    def _validate_with_main_extractor(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """MainDataExtractorを活用した検証"""
        try:
            metrics = {"extraction_valid": False, "trade_count": 0, "validation_score": 0.0}
            
            # MainDataExtractorで取引抽出テスト
            trades = self.main_extractor.extract_accurate_trades(data)
            metrics["trade_count"] = len(trades)
            
            # 取引データの妥当性スコア計算
            if trades:
                # 取引の基本妥当性チェック
                valid_trades = [
                    trade for trade in trades 
                    if trade.get('entry_price', 0) > 0 and trade.get('exit_price', 0) > 0
                ]
                metrics["validation_score"] = len(valid_trades) / len(trades) if trades else 0.0
                metrics["extraction_valid"] = metrics["validation_score"] > 0.8
                
                self.logger.info(f"MainDataExtractor検証: {len(trades)}取引中{len(valid_trades)}件有効")
            
            return data, metrics
            
        except Exception as e:
            self.logger.error(f"MainDataExtractor検証中にエラー: {e}")
            return data, {"extraction_valid": False, "trade_count": 0, "validation_score": 0.0}
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, 
                                 basic_metrics: Dict[str, int],
                                 price_metrics: Dict[str, int],
                                 signal_metrics: Dict[str, int],
                                 extraction_metrics: Dict[str, Any]) -> DataQualityMetrics:
        """総合品質メトリクス計算"""
        try:
            total_records = len(data)
            total_corrections = (basic_metrics.get("corrected", 0) + 
                               price_metrics.get("corrected", 0) + 
                               signal_metrics.get("corrected", 0))
            
            total_issues = (basic_metrics.get("invalid", 0) + 
                          price_metrics.get("price_anomalies", 0) + 
                          signal_metrics.get("signal_conflicts", 0))
            
            # 品質スコア計算
            if total_records > 0:
                correction_ratio = total_corrections / total_records
                issue_ratio = total_issues / total_records
                extraction_score = extraction_metrics.get("validation_score", 0.0)
                
                # 総合品質スコア（重み付き平均）
                quality_score = (
                    (1.0 - issue_ratio) * 0.4 +  # 問題発生率（低いほど良い）
                    (1.0 - min(correction_ratio, 0.3)) * 0.3 +  # 修正率（適度が良い）
                    extraction_score * 0.3  # 抽出妥当性（高いほど良い）
                )
            else:
                quality_score = 0.0
            
            return DataQualityMetrics(
                total_records=total_records,
                valid_records=total_records - total_issues,
                invalid_records=total_issues,
                corrected_records=total_corrections,
                missing_signals=basic_metrics.get("invalid", 0),
                inconsistent_prices=price_metrics.get("price_anomalies", 0),
                quality_score=max(0.0, min(1.0, quality_score)),
                enhancement_applied=total_corrections > 0
            )
            
        except Exception as e:
            self.logger.error(f"品質メトリクス計算中にエラー: {e}")
            return DataQualityMetrics(total_records=len(data))
    
    def _generate_quality_report(self, metrics: DataQualityMetrics, strategy_name: Optional[str] = None) -> None:
        """品質レポート生成"""
        try:
            strategy_info = f"戦略: {strategy_name}" if strategy_name else "全戦略"
            
            self.logger.info("=" * 50)
            self.logger.info(f"DSSMS データ品質レポート ({strategy_info})")
            self.logger.info("=" * 50)
            self.logger.info(f"総レコード数: {metrics.total_records:,}")
            self.logger.info(f"有効レコード数: {metrics.valid_records:,}")
            self.logger.info(f"無効レコード数: {metrics.invalid_records:,}")
            self.logger.info(f"修正レコード数: {metrics.corrected_records:,}")
            self.logger.info(f"品質スコア: {metrics.quality_score:.3f}")
            self.logger.info(f"品質向上適用: {'はい' if metrics.enhancement_applied else 'いいえ'}")
            
            if metrics.quality_score >= self.config.quality_threshold:
                self.logger.info("✅ 品質基準を満たしています")
            else:
                self.logger.warning(f"⚠️  品質基準未達（目標: {self.config.quality_threshold:.3f}）")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"品質レポート生成中にエラー: {e}")
    
    def test_enhancement(self, sample_data: Optional[pd.DataFrame] = None) -> bool:
        """品質向上機能のテスト"""
        try:
            self.logger.info("DSSMS Data Quality Enhancer テスト開始")
            
            # テストデータ生成
            if sample_data is None:
                dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
                sample_data = pd.DataFrame({
                    'Close': [100, 102, -1, 105, 150, 98, 99, 101],  # 異常価格含む
                    'Entry_Signal': [1, 0, 0, 1, 1, 0, 0, 1],
                    'Exit_Signal': [0, 1, 0, 1, 0, 1, 0, 0],  # シグナル競合含む
                }, index=dates[:8])
            
            # 品質向上実行
            enhanced_data, metrics = self.enhance_dssms_data(sample_data)
            
            # 結果検証
            success = (
                not enhanced_data.empty and
                metrics.quality_score > 0.5 and
                metrics.enhancement_applied
            )
            
            self.logger.info(f"テスト{'成功' if success else '失敗'}: 品質スコア {metrics.quality_score:.3f}")
            return success
            
        except Exception as e:
            self.logger.error(f"テスト中にエラー: {e}")
            return False


if __name__ == "__main__":
    # テスト実行
    enhancer = DSSMSDataQualityEnhancer()
    enhancer.test_enhancement()
