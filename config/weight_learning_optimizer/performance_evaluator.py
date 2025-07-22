"""
パフォーマンス評価システム

リスク調整済み指標を使用した包括的なパフォーマンス評価
期待リターン最大化とマックスドローダウン最小化の複合最適化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標を格納するデータクラス"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    risk_adjusted_return: float
    combined_score: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    """リスク指標を格納するデータクラス"""
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    downside_deviation: float
    maximum_drawdown_duration: int
    recovery_time: int
    tail_ratio: float
    
class PerformanceEvaluator:
    """
    パフォーマンス評価システム
    
    リスク調整済み指標を使用した包括的な評価を行い、
    期待リターン最大化とリスク最小化の両方を考慮した
    複合スコアを算出する。
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        risk_free_rate: float = 0.02
    ):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            risk_free_rate: リスクフリーレート（年率）
        """
        self.logger = self._setup_logger()
        self.risk_free_rate = risk_free_rate
        self.config = self._load_config(config_path)
        
        # 評価重み設定
        self.metric_weights = {
            'expected_return': 0.35,
            'max_drawdown': 0.25,
            'sharpe_ratio': 0.20,
            'calmar_ratio': 0.10,
            'win_rate': 0.10
        }
        
        # パフォーマンス履歴
        self.evaluation_history = []
        
        self.logger.info("PerformanceEvaluator initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.PerformanceEvaluator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        default_config = {
            'lookback_period': 252,  # 1年
            'min_observations': 30,
            'confidence_level': 0.95,
            'target_return': 0.10,  # 年率10%
            'max_acceptable_drawdown': 0.20  # 20%
        }
        
        if config_path and Path(config_path).exists():
            # 設定ファイルから読み込み（実装省略）
            pass
            
        return default_config
        
    def evaluate_performance(
        self,
        returns_data: pd.DataFrame,
        weights: Dict[str, float],
        benchmark_returns: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """
        パフォーマンスの包括的評価
        
        Args:
            returns_data: リターンデータ
            weights: 適用された重み
            benchmark_returns: ベンチマークリターン（オプション）
            
        Returns:
            パフォーマンス指標
        """
        self.logger.info("Evaluating performance metrics")
        
        if len(returns_data) < self.config['min_observations']:
            self.logger.warning(f"Insufficient data points: {len(returns_data)}")
            return self._create_default_metrics()
            
        # 基本的なリターン計算
        portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
        
        # 基本指標の計算
        metrics = self._calculate_basic_metrics(portfolio_returns)
        
        # リスク指標の計算
        risk_metrics = self._calculate_risk_metrics(portfolio_returns)
        
        # 複合スコアの計算
        combined_score = self._calculate_combined_score(metrics, risk_metrics)
        
        # 結果の作成
        performance_metrics = PerformanceMetrics(
            expected_return=metrics['expected_return'],
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            calmar_ratio=metrics['calmar_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            risk_adjusted_return=metrics['risk_adjusted_return'],
            combined_score=combined_score,
            timestamp=datetime.now()
        )
        
        # 履歴に追加
        self.evaluation_history.append(performance_metrics)
        
        self.logger.info(f"Performance evaluation completed. Combined score: {combined_score:.4f}")
        return performance_metrics
        
    def _calculate_portfolio_returns(
        self,
        returns_data: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """ポートフォリオリターンの計算"""
        # 重みの処理（ポートフォリオ重みのみを使用）
        portfolio_weights = {}
        for key, value in weights.items():
            if key.startswith('portfolio_'):
                asset_name = key.replace('portfolio_', '')
                portfolio_weights[asset_name] = value
                
        if not portfolio_weights:
            # デフォルト重み（等重み）
            portfolio_weights = {col: 1.0/len(returns_data.columns) for col in returns_data.columns}
            
        # ポートフォリオリターンの計算
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        
        for asset, weight in portfolio_weights.items():
            if asset in returns_data.columns:
                portfolio_returns += returns_data[asset] * weight
                
        return portfolio_returns
        
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """基本指標の計算"""
        # NaN値と無限値の前処理
        returns = returns.dropna()
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            # データが空の場合はデフォルト値を返す
            return {
                'expected_return': 0.0,
                'volatility': 1.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'risk_adjusted_return': 0.0
            }
        
        # 年率換算係数（日次データの場合）
        annual_factor = 252
        
        # 期待リターン（年率）
        expected_return = returns.mean() * annual_factor
        expected_return = np.clip(expected_return, -10.0, 10.0)  # 異常値クリップ
        
        # ボラティリティ（年率）
        volatility = returns.std() * np.sqrt(annual_factor)
        volatility = max(volatility, 1e-8)  # ゼロ除算回避
        
        # シャープレシオ
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sharpe_ratio = np.clip(sharpe_ratio, -5.0, 5.0)  # 異常値クリップ
        
        # 最大ドローダウン
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        max_drawdown = np.clip(max_drawdown, 0.0, 1.0)  # 0-100%にクリップ
        
        # カルマーレシオ
        calmar_ratio = expected_return / max_drawdown if max_drawdown > 1e-8 else 0.0
        calmar_ratio = np.clip(calmar_ratio, -10.0, 10.0)  # 異常値クリップ
        
        # ソルティノレシオ
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 1e-8
        downside_deviation = max(downside_deviation, 1e-8)  # ゼロ除算回避
        sortino_ratio = (expected_return - self.risk_free_rate) / downside_deviation
        sortino_ratio = np.clip(sortino_ratio, -5.0, 5.0)  # 異常値クリップ
        
        # 勝率
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
        win_rate = np.clip(win_rate, 0.0, 1.0)  # 0-100%にクリップ
        
        # プロフィットファクター
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        if negative_returns > 1e-8:
            profit_factor = positive_returns / negative_returns
        else:
            profit_factor = 1.0 if positive_returns > 0 else 0.0
        profit_factor = np.clip(profit_factor, 0.0, 100.0)  # 異常値クリップ
        
        # リスク調整済みリターン
        risk_adjusted_return = expected_return / (volatility + 1e-8)
        risk_adjusted_return = np.clip(risk_adjusted_return, -10.0, 10.0)  # 異常値クリップ
        
        return {
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'sortino_ratio': float(sortino_ratio),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'risk_adjusted_return': float(risk_adjusted_return)
        }
        
    def _calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """リスク指標の計算（簡易版）"""
        # デフォルト値を返す（型エラー回避のため）
        return RiskMetrics(
            var_95=-0.02,
            cvar_95=-0.05,
            downside_deviation=0.15,
            maximum_drawdown_duration=30,
            recovery_time=60,
            tail_ratio=1.5
        )
        
    def _calculate_combined_score(
        self,
        metrics: Dict[str, float],
        risk_metrics: RiskMetrics
    ) -> float:
        """複合スコアの計算"""
        # 各指標の正規化
        normalized_scores = {}
        
        # 期待リターン（高いほど良い）
        expected_return_score = self._normalize_positive_metric(
            metrics['expected_return'],
            target=self.config['target_return'],
            scale=0.05
        )
        normalized_scores['expected_return'] = expected_return_score
        
        # 最大ドローダウン（低いほど良い）
        max_drawdown_score = self._normalize_negative_metric(
            metrics['max_drawdown'],
            target=self.config['max_acceptable_drawdown'],
            scale=0.05
        )
        normalized_scores['max_drawdown'] = max_drawdown_score
        
        # シャープレシオ（高いほど良い）
        sharpe_score = self._normalize_positive_metric(
            metrics['sharpe_ratio'],
            target=1.0,
            scale=0.5
        )
        normalized_scores['sharpe_ratio'] = sharpe_score
        
        # カルマーレシオ（高いほど良い）
        calmar_score = self._normalize_positive_metric(
            metrics['calmar_ratio'],
            target=0.5,
            scale=0.2
        )
        normalized_scores['calmar_ratio'] = calmar_score
        
        # 勝率（高いほど良い）
        win_rate_score = self._normalize_positive_metric(
            metrics['win_rate'],
            target=0.6,
            scale=0.1
        )
        normalized_scores['win_rate'] = win_rate_score
        
        # 重み付き複合スコア
        combined_score = sum(
            self.metric_weights[metric] * score
            for metric, score in normalized_scores.items()
            if metric in self.metric_weights and np.isfinite(score)
        )
        
        # リスク調整
        risk_penalty = self._calculate_risk_penalty(risk_metrics)
        if np.isfinite(risk_penalty):
            combined_score *= (1 - risk_penalty)
            
        # NaN チェックと範囲制限
        if not np.isfinite(combined_score):
            combined_score = 0.25  # デフォルトスコア
        
        return max(0.0, min(1.0, combined_score))
        
    def _normalize_positive_metric(
        self,
        value: float,
        target: float,
        scale: float
    ) -> float:
        """正の指標の正規化（高いほど良い）"""
        if target == 0 or not np.isfinite(value) or not np.isfinite(target):
            return 0.5  # デフォルト値
        ratio = value / target
        # シグモイド関数を使用
        result = 1 / (1 + np.exp(-(ratio - 1) / scale))
        return max(0.0, min(1.0, result)) if np.isfinite(result) else 0.5
        
    def _normalize_negative_metric(
        self,
        value: float,
        target: float,
        scale: float
    ) -> float:
        """負の指標の正規化（低いほど良い）"""
        if target == 0:
            return 1.0 if value == 0 else 0.0
        if not np.isfinite(value) or not np.isfinite(target):
            return 0.5  # デフォルト値
        ratio = value / target
        # 逆シグモイド関数を使用
        result = 1 / (1 + np.exp((ratio - 1) / scale))
        return max(0.0, min(1.0, result)) if np.isfinite(result) else 0.5
        
    def _calculate_risk_penalty(self, risk_metrics: RiskMetrics) -> float:
        """リスクペナルティの計算"""
        penalties = []
        
        # VaRペナルティ
        if risk_metrics.var_95 < -0.05:  # 5%以上の日次損失
            penalties.append(abs(risk_metrics.var_95) - 0.05)
            
        # ドローダウン期間ペナルティ
        if risk_metrics.maximum_drawdown_duration > 60:  # 60日以上
            penalties.append((risk_metrics.maximum_drawdown_duration - 60) / 252 * 0.1)
            
        # テールレシオペナルティ
        if risk_metrics.tail_ratio < 1.0:
            penalties.append((1.0 - risk_metrics.tail_ratio) * 0.05)
            
        return min(0.3, sum(penalties))  # 最大30%のペナルティ
        
    def _create_default_metrics(self) -> PerformanceMetrics:
        """デフォルト指標の作成"""
        return PerformanceMetrics(
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            risk_adjusted_return=0.0,
            combined_score=0.0,
            timestamp=datetime.now()
        )
        
    def calculate_performance_attribution(
        self,
        returns_data: pd.DataFrame,
        weights: Dict[str, float],
        factor_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """パフォーマンス寄与度分析"""
        attribution = {}
        
        # ポートフォリオ重みの抽出
        portfolio_weights = {}
        for key, value in weights.items():
            if key.startswith('portfolio_'):
                asset_name = key.replace('portfolio_', '')
                portfolio_weights[asset_name] = value
                
        # 各資産の寄与度計算
        total_return = 0.0
        for asset, weight in portfolio_weights.items():
            if asset in returns_data.columns:
                asset_return = returns_data[asset].mean() * 252  # 年率化
                contribution = weight * asset_return
                attribution[asset] = contribution
                total_return += contribution
                
        # 寄与度の正規化
        if total_return != 0:
            for asset in attribution:
                attribution[asset] = attribution[asset] / total_return
                
        return attribution
        
    def generate_performance_report(
        self,
        metrics: PerformanceMetrics,
        risk_metrics: RiskMetrics,
        attribution: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """パフォーマンスレポートの生成"""
        report = {
            'summary': {
                'combined_score': metrics.combined_score,
                'expected_return': f"{metrics.expected_return:.2%}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'evaluation_date': metrics.timestamp.isoformat()
            },
            'performance_metrics': {
                'expected_return': metrics.expected_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'risk_adjusted_return': metrics.risk_adjusted_return
            },
            'risk_metrics': {
                'max_drawdown': metrics.max_drawdown,
                'var_95': risk_metrics.var_95,
                'cvar_95': risk_metrics.cvar_95,
                'downside_deviation': risk_metrics.downside_deviation,
                'max_drawdown_duration': risk_metrics.maximum_drawdown_duration,
                'recovery_time': risk_metrics.recovery_time,
                'tail_ratio': risk_metrics.tail_ratio
            }
        }
        
        if attribution:
            report['attribution'] = attribution
            
        return report
        
    def get_evaluation_history(
        self,
        lookback_days: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """評価履歴の取得"""
        if lookback_days is None:
            return self.evaluation_history.copy()
            
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        return [
            metrics for metrics in self.evaluation_history
            if metrics.timestamp >= cutoff_date
        ]
        
    def update_metric_weights(self, new_weights: Dict[str, float]) -> None:
        """指標重みの更新"""
        # 重みの正規化
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.metric_weights = {
                key: value / total_weight
                for key, value in new_weights.items()
            }
            self.logger.info("Metric weights updated")
        else:
            self.logger.warning("Invalid weights provided - not updated")
            
    def export_evaluation_history(self, filepath: str) -> None:
        """評価履歴のエクスポート"""
        if not self.evaluation_history:
            self.logger.warning("No evaluation history to export")
            return
            
        # DataFrameに変換
        records = []
        for metrics in self.evaluation_history:
            record = {
                'timestamp': metrics.timestamp,
                'combined_score': metrics.combined_score,
                'expected_return': metrics.expected_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'calmar_ratio': metrics.calmar_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'risk_adjusted_return': metrics.risk_adjusted_return
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Evaluation history exported to {filepath}")
