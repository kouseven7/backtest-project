"""
DSSMS Output Validator
Phase 2.3 Task 2.3.1: バックテストデータ収集最適化

Purpose:
  - DSSMSバックテスト出力結果の検証
  - 出力前の最終品質チェック
  - 一貫性検証と自動修正
  - 統計的妥当性の確認

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - DSSMSDataQualityEnhancerとの連携
  - output/dssms_excel_exporter_v2.pyとの統合対応
  - 既存DSSMS出力システムとの互換性保持
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

from config.logger_config import setup_logger


@dataclass
class ValidationMetrics:
    """出力検証メトリクス"""
    total_trades: int = 0
    valid_trades: int = 0
    invalid_trades: int = 0
    profit_loss_sum: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    validation_score: float = 0.0
    is_valid_output: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class DSSMSOutputConfig:
    """DSSMS出力検証設定"""
    min_trade_count: int = 1
    max_drawdown_threshold: float = 0.50  # 50%以上のドローダウンで警告
    min_win_rate: float = 0.20  # 20%未満の勝率で警告
    min_sharpe_ratio: float = -2.0  # -2.0未満のシャープレシオで警告
    max_single_loss: float = 0.10  # 単一取引で10%以上の損失で警告
    consistency_check: bool = True
    statistical_validation: bool = True
    auto_correction: bool = True


class DSSMSOutputValidator:
    """DSSMS出力結果検証システム"""
    
    def __init__(self, config: Optional[DSSMSOutputConfig] = None, initial_capital: float = 1000000.0):
        """
        初期化
        
        Args:
            config: 検証設定（Noneの場合はデフォルト使用）
            initial_capital: 初期資本金
        """
        self.config = config or DSSMSOutputConfig()
        self.initial_capital = initial_capital
        self.logger = setup_logger(__name__)
        
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
        
        self.logger.info("DSSMS Output Validator 初期化完了")
    
    def validate_dssms_output(self, backtest_result: Dict[str, Any], 
                            trades_data: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], ValidationMetrics]:
        """
        DSSMSバックテスト出力の検証
        
        Args:
            backtest_result: DSSMSバックテスト結果
            trades_data: 取引データ（Noneの場合は結果から抽出）
        
        Returns:
            Tuple[Dict[str, Any], ValidationMetrics]: 検証済み結果と検証メトリクス
        """
        try:
            self.logger.info("DSSMS 出力検証開始")
            
            # メトリクス初期化
            metrics = ValidationMetrics()
            
            # 結果データの基本構造チェック
            validated_result, structure_valid = self._validate_result_structure(backtest_result)
            if not structure_valid:
                metrics.validation_errors.append("結果データの構造が無効")
                return validated_result, metrics
            
            # 取引データの抽出・検証
            if trades_data is None:
                trades_data = self._extract_trades_from_result(validated_result)
            
            # ステップ1: 取引データの妥当性検証
            trades_metrics = self._validate_trades_data(trades_data)
            
            # ステップ2: 統計的指標の検証
            statistical_metrics = self._validate_statistical_indicators(trades_data)
            
            # ステップ3: 一貫性検証
            consistency_metrics = self._validate_consistency(validated_result, trades_data)
            
            # ステップ4: 総合検証メトリクス計算
            metrics = self._calculate_validation_metrics(trades_metrics, statistical_metrics, consistency_metrics)
            
            # ステップ5: 自動修正（必要に応じて）
            if not metrics.is_valid_output and self.config.auto_correction:
                validated_result = self._apply_auto_corrections(validated_result, metrics)
            
            # ステップ6: 最終検証レポート
            self._generate_validation_report(metrics)
            
            self.logger.info(f"出力検証完了: 検証スコア {metrics.validation_score:.3f}")
            return validated_result, metrics
            
        except Exception as e:
            self.logger.error(f"出力検証中にエラー: {e}")
            metrics.validation_errors.append(f"検証エラー: {e}")
            return backtest_result, metrics
    
    def _validate_result_structure(self, result: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """結果データ構造の検証"""
        try:
            validated_result = result.copy()
            
            # 必須キーの確認
            required_keys = ['total_profit_loss', 'win_rate', 'total_trades']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                self.logger.warning(f"必須キーが不足: {missing_keys}")
                # デフォルト値で補完
                for key in missing_keys:
                    if key == 'total_profit_loss':
                        validated_result[key] = 0.0
                    elif key == 'win_rate':
                        validated_result[key] = 0.0
                    elif key == 'total_trades':
                        validated_result[key] = 0
            
            # データ型の検証
            if not isinstance(validated_result.get('total_profit_loss', 0), (int, float)):
                validated_result['total_profit_loss'] = 0.0
            
            if not isinstance(validated_result.get('win_rate', 0), (int, float)):
                validated_result['win_rate'] = 0.0
            
            if not isinstance(validated_result.get('total_trades', 0), int):
                validated_result['total_trades'] = 0
            
            return validated_result, True
            
        except Exception as e:
            self.logger.error(f"構造検証中にエラー: {e}")
            return result, False
    
    def _extract_trades_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """結果から取引データを抽出"""
        try:
            trades = []
            
            # 複数の可能なキーを確認
            for key in ['trades', 'trade_history', 'transactions', 'trade_details']:
                if key in result and isinstance(result[key], list):
                    trades = result[key]
                    break
            
            # DataFrameの場合の処理
            if not trades:
                for key in ['data', 'backtest_data', 'results_data']:
                    if key in result and isinstance(result[key], pd.DataFrame):
                        df = result[key]
                        trades = self._extract_trades_from_dataframe(df)
                        break
            
            self.logger.info(f"取引データ抽出: {len(trades)}件")
            return trades
            
        except Exception as e:
            self.logger.error(f"取引データ抽出中にエラー: {e}")
            return []
    
    def _extract_trades_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """DataFrameから取引データを抽出"""
        try:
            trades = []
            
            if df.empty:
                return trades
            
            # Entry_Signal/Exit_Signalから取引を構築
            position_open = False
            entry_price = 0.0
            entry_date = None
            
            for idx, row in df.iterrows():
                if row.get('Entry_Signal', 0) == 1 and not position_open:
                    entry_price = row.get('Close', 0)
                    entry_date = idx
                    position_open = True
                
                elif row.get('Exit_Signal', 0) == 1 and position_open:
                    exit_price = row.get('Close', 0)
                    exit_date = idx
                    
                    if entry_price > 0 and exit_price > 0:
                        profit_loss = (exit_price - entry_price) / entry_price
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit_loss': profit_loss,
                            'strategy': row.get('Strategy', 'Unknown')
                        })
                    
                    position_open = False
            
            return trades
            
        except Exception as e:
            self.logger.error(f"DataFrame取引抽出中にエラー: {e}")
            return []
    
    def _validate_trades_data(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """取引データの妥当性検証"""
        try:
            metrics = {
                "total_trades": len(trades),
                "valid_trades": 0,
                "invalid_trades": 0,
                "extreme_losses": 0,
                "trade_validation_score": 0.0
            }
            
            if not trades:
                return metrics
            
            valid_count = 0
            extreme_loss_count = 0
            
            for trade in trades:
                # 基本データの妥当性チェック
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                profit_loss = trade.get('profit_loss', 0)
                
                is_valid = (
                    entry_price > 0 and
                    exit_price > 0 and
                    isinstance(profit_loss, (int, float)) and
                    abs(profit_loss) < 1.0  # 100%以上の損益は異常
                )
                
                if is_valid:
                    valid_count += 1
                
                # 極端な損失のチェック
                if profit_loss < -self.config.max_single_loss:
                    extreme_loss_count += 1
            
            metrics["valid_trades"] = valid_count
            metrics["invalid_trades"] = len(trades) - valid_count
            metrics["extreme_losses"] = extreme_loss_count
            metrics["trade_validation_score"] = valid_count / len(trades) if trades else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"取引データ検証中にエラー: {e}")
            return {"total_trades": 0, "valid_trades": 0, "invalid_trades": 0, 
                   "extreme_losses": 0, "trade_validation_score": 0.0}
    
    def _validate_statistical_indicators(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """統計指標の検証"""
        try:
            metrics = {
                "calculated_win_rate": 0.0,
                "calculated_profit_loss": 0.0,
                "calculated_sharpe": 0.0,
                "calculated_drawdown": 0.0,
                "statistical_validity": False
            }
            
            if not trades:
                return metrics
            
            # 勝率計算
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            metrics["calculated_win_rate"] = len(winning_trades) / len(trades)
            
            # 総損益計算
            total_pl = sum(t.get('profit_loss', 0) for t in trades)
            metrics["calculated_profit_loss"] = total_pl
            
            # シャープレシオ計算（簡易版）
            if trades:
                returns = [t.get('profit_loss', 0) for t in trades]
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                metrics["calculated_sharpe"] = mean_return / std_return if std_return > 0 else 0.0
            
            # ドローダウン計算（累積損益から）
            cumulative_returns = np.cumsum([t.get('profit_loss', 0) for t in trades])
            if len(cumulative_returns) > 0:
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak)
                metrics["calculated_drawdown"] = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
            # 統計的妥当性の判定
            metrics["statistical_validity"] = (
                len(trades) >= self.config.min_trade_count and
                metrics["calculated_win_rate"] >= self.config.min_win_rate and
                metrics["calculated_sharpe"] >= self.config.min_sharpe_ratio and
                metrics["calculated_drawdown"] <= self.config.max_drawdown_threshold
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"統計指標検証中にエラー: {e}")
            return {"calculated_win_rate": 0.0, "calculated_profit_loss": 0.0, 
                   "calculated_sharpe": 0.0, "calculated_drawdown": 0.0, "statistical_validity": False}
    
    def _validate_consistency(self, result: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """一貫性検証"""
        try:
            metrics = {
                "consistency_score": 0.0,
                "profit_loss_match": False,
                "trade_count_match": False,
                "win_rate_match": False
            }
            
            if not self.config.consistency_check or not trades:
                return metrics
            
            # 損益の一貫性チェック
            reported_pl = result.get('total_profit_loss', 0)
            calculated_pl = sum(t.get('profit_loss', 0) for t in trades) * self.initial_capital
            pl_diff = abs(reported_pl - calculated_pl)
            metrics["profit_loss_match"] = pl_diff < (self.initial_capital * 0.01)  # 1%以内の誤差
            
            # 取引数の一貫性チェック
            reported_trades = result.get('total_trades', 0)
            calculated_trades = len(trades)
            metrics["trade_count_match"] = reported_trades == calculated_trades
            
            # 勝率の一貫性チェック
            reported_win_rate = result.get('win_rate', 0)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            calculated_win_rate = len(winning_trades) / len(trades) if trades else 0
            win_rate_diff = abs(reported_win_rate - calculated_win_rate)
            metrics["win_rate_match"] = win_rate_diff < 0.05  # 5%以内の誤差
            
            # 総合一貫性スコア
            consistency_count = sum([
                metrics["profit_loss_match"],
                metrics["trade_count_match"],
                metrics["win_rate_match"]
            ])
            metrics["consistency_score"] = consistency_count / 3.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"一貫性検証中にエラー: {e}")
            return {"consistency_score": 0.0, "profit_loss_match": False, 
                   "trade_count_match": False, "win_rate_match": False}
    
    def _calculate_validation_metrics(self, trades_metrics: Dict[str, Any],
                                    statistical_metrics: Dict[str, Any],
                                    consistency_metrics: Dict[str, Any]) -> ValidationMetrics:
        """総合検証メトリクス計算"""
        try:
            metrics = ValidationMetrics()
            
            # 基本メトリクス設定
            metrics.total_trades = trades_metrics.get("total_trades", 0)
            metrics.valid_trades = trades_metrics.get("valid_trades", 0)
            metrics.invalid_trades = trades_metrics.get("invalid_trades", 0)
            
            # 統計メトリクス設定
            metrics.profit_loss_sum = statistical_metrics.get("calculated_profit_loss", 0.0)
            metrics.win_rate = statistical_metrics.get("calculated_win_rate", 0.0)
            metrics.max_drawdown = statistical_metrics.get("calculated_drawdown", 0.0)
            metrics.sharpe_ratio = statistical_metrics.get("calculated_sharpe", 0.0)
            
            # 検証スコア計算（重み付き平均）
            trade_score = trades_metrics.get("trade_validation_score", 0.0)
            statistical_validity = 1.0 if statistical_metrics.get("statistical_validity", False) else 0.0
            consistency_score = consistency_metrics.get("consistency_score", 0.0)
            
            metrics.validation_score = (
                trade_score * 0.4 +
                statistical_validity * 0.3 +
                consistency_score * 0.3
            )
            
            # 出力の妥当性判定
            metrics.is_valid_output = (
                metrics.validation_score >= 0.7 and
                metrics.total_trades >= self.config.min_trade_count and
                trade_score >= 0.8
            )
            
            # エラーメッセージの設定
            if not metrics.is_valid_output:
                if metrics.total_trades < self.config.min_trade_count:
                    metrics.validation_errors.append(f"取引数不足: {metrics.total_trades}")
                if trade_score < 0.8:
                    metrics.validation_errors.append(f"取引データ品質不良: {trade_score:.3f}")
                if not statistical_validity:
                    metrics.validation_errors.append("統計指標異常")
                if consistency_score < 0.7:
                    metrics.validation_errors.append(f"一貫性不良: {consistency_score:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"検証メトリクス計算中にエラー: {e}")
            return ValidationMetrics()
    
    def _apply_auto_corrections(self, result: Dict[str, Any], metrics: ValidationMetrics) -> Dict[str, Any]:
        """自動修正の適用"""
        try:
            corrected_result = result.copy()
            
            # 統計指標の再計算による修正
            if metrics.total_trades > 0:
                corrected_result['total_trades'] = metrics.total_trades
                corrected_result['win_rate'] = metrics.win_rate
                corrected_result['total_profit_loss'] = metrics.profit_loss_sum * self.initial_capital
                
                if 'sharpe_ratio' in corrected_result:
                    corrected_result['sharpe_ratio'] = metrics.sharpe_ratio
                
                if 'max_drawdown' in corrected_result:
                    corrected_result['max_drawdown'] = metrics.max_drawdown
                
                self.logger.info("統計指標を再計算値で修正")
            
            return corrected_result
            
        except Exception as e:
            self.logger.error(f"自動修正中にエラー: {e}")
            return result
    
    def _generate_validation_report(self, metrics: ValidationMetrics) -> None:
        """検証レポート生成"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("DSSMS 出力検証レポート")
            self.logger.info("=" * 50)
            self.logger.info(f"総取引数: {metrics.total_trades}")
            self.logger.info(f"有効取引数: {metrics.valid_trades}")
            self.logger.info(f"無効取引数: {metrics.invalid_trades}")
            self.logger.info(f"総損益: {metrics.profit_loss_sum:.4f}")
            self.logger.info(f"勝率: {metrics.win_rate:.3f}")
            self.logger.info(f"最大ドローダウン: {metrics.max_drawdown:.3f}")
            self.logger.info(f"シャープレシオ: {metrics.sharpe_ratio:.3f}")
            self.logger.info(f"検証スコア: {metrics.validation_score:.3f}")
            self.logger.info(f"出力妥当性: {'[OK] 有効' if metrics.is_valid_output else '[ERROR] 無効'}")
            
            if metrics.validation_errors:
                self.logger.warning("検証エラー:")
                for error in metrics.validation_errors:
                    self.logger.warning(f"  - {error}")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"検証レポート生成中にエラー: {e}")
    
    def test_validation(self, sample_result: Optional[Dict[str, Any]] = None) -> bool:
        """出力検証機能のテスト"""
        try:
            self.logger.info("DSSMS Output Validator テスト開始")
            
            # テストデータ生成
            if sample_result is None:
                sample_trades = [
                    {'entry_price': 100, 'exit_price': 105, 'profit_loss': 0.05, 'strategy': 'Test'},
                    {'entry_price': 105, 'exit_price': 103, 'profit_loss': -0.019, 'strategy': 'Test'},
                    {'entry_price': 103, 'exit_price': 108, 'profit_loss': 0.049, 'strategy': 'Test'}
                ]
                
                sample_result = {
                    'total_profit_loss': 80000.0,  # 8%の利益
                    'win_rate': 0.667,  # 2勝1敗
                    'total_trades': 3,
                    'trades': sample_trades
                }
            
            # 検証実行
            validated_result, metrics = self.validate_dssms_output(sample_result)
            
            # 結果検証
            success = (
                metrics.validation_score > 0.5 and
                metrics.total_trades > 0 and
                not metrics.validation_errors
            )
            
            self.logger.info(f"テスト{'成功' if success else '失敗'}: 検証スコア {metrics.validation_score:.3f}")
            return success
            
        except Exception as e:
            self.logger.error(f"テスト中にエラー: {e}")
            return False


if __name__ == "__main__":
    # テスト実行
    validator = DSSMSOutputValidator()
    validator.test_validation()
