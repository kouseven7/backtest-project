"""
Module: Portfolio Risk Manager
File: portfolio_risk_manager.py
Description: 
  3-3-3「ポートフォリオレベルのリスク調整機能」
  ポートフォリオ全体のリスクを評価し、動的に調整する統合システム
  VaR、ドローダウン、相関リスク、集中度リスクを包括的に管理

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Dependencies:
  - config.portfolio_weight_calculator
  - config.position_size_adjuster
  - config.signal_integrator
  - config.risk_management
  - config.strategy_scoring_model
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# 数値計算・統計ライブラリ
try:
    from scipy import stats
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    ADVANCED_STATS = True
except ImportError:
    ADVANCED_STATS = False
    warnings.warn("Advanced statistical libraries not available. Some features will be limited.")

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, AllocationResult, WeightAllocationConfig,
        PortfolioConstraints, AllocationMethod
    )
    from config.position_size_adjuster import (
        PositionSizeAdjuster, PositionSizeResult, PortfolioPositionSizing
    )
    from config.signal_integrator import SignalIntegrator, StrategySignal, SignalType
    from config.risk_management import RiskManagement
    from config.strategy_scoring_model import StrategyScore, StrategyScoreManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error in portfolio_risk_manager: {e}. Some functionality may be limited.")
    
    # フォールバック用の空クラス定義
    class SignalIntegrator:
        pass
    class StrategySignal:
        pass
    class PositionSizeAdjuster:
        def __init__(self, config): pass
    class PortfolioWeightCalculator:
        def __init__(self, config): pass

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskMetricType(Enum):
    """リスク指標タイプ"""
    VAR_95 = "var_95"                    # 95% Value at Risk
    VAR_99 = "var_99"                    # 99% Value at Risk
    CVAR_95 = "cvar_95"                  # 95% Conditional VaR
    CVAR_99 = "cvar_99"                  # 99% Conditional VaR
    MAX_DRAWDOWN = "max_drawdown"        # 最大ドローダウン
    VOLATILITY = "volatility"            # ボラティリティ
    SHARPE_RATIO = "sharpe_ratio"        # シャープレシオ
    SORTINO_RATIO = "sortino_ratio"      # ソルティノレシオ
    CORRELATION_RISK = "correlation_risk" # 相関リスク
    CONCENTRATION_RISK = "concentration_risk" # 集中度リスク

class RiskLimitType(Enum):
    """リスク制限タイプ"""
    HARD_LIMIT = "hard_limit"       # 厳格な制限（超過時は強制調整）
    SOFT_LIMIT = "soft_limit"       # 柔軟な制限（超過時は警告）
    DYNAMIC_LIMIT = "dynamic_limit" # 動的制限（市場環境に応じて調整）

class RiskAdjustmentAction(Enum):
    """リスク調整アクション"""
    NO_ACTION = "no_action"             # アクションなし
    REDUCE_POSITIONS = "reduce_positions" # ポジション減少
    INCREASE_HEDGING = "increase_hedging" # ヘッジ増加
    REBALANCE_WEIGHTS = "rebalance_weights" # ウェイト再配分
    STOP_NEW_POSITIONS = "stop_new_positions" # 新規ポジション停止
    EMERGENCY_EXIT = "emergency_exit"   # 緊急終了

@dataclass
class RiskMetric:
    """リスク指標データクラス"""
    metric_type: RiskMetricType
    current_value: float
    limit_value: float
    limit_type: RiskLimitType
    breach_severity: float = 0.0  # 制限超過の深刻度 (0-1)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_breached(self) -> bool:
        """制限違反の判定"""
        return self.current_value > self.limit_value
    
    @property
    def breach_ratio(self) -> float:
        """制限超過比率"""
        if self.limit_value == 0:
            return 0.0
        return max(0.0, (self.current_value - self.limit_value) / self.limit_value)

@dataclass
class RiskAdjustmentResult:
    """リスク調整結果"""
    timestamp: datetime
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    adjustment_actions: List[RiskAdjustmentAction]
    risk_metrics_before: Dict[str, RiskMetric]
    risk_metrics_after: Dict[str, RiskMetric]
    adjustment_reason: str
    effectiveness_score: float = 0.0
    
    def get_weight_changes(self) -> Dict[str, float]:
        """ウェイト変更量を取得"""
        changes = {}
        all_strategies = set(self.original_weights.keys()) | set(self.adjusted_weights.keys())
        
        for strategy in all_strategies:
            original = self.original_weights.get(strategy, 0.0)
            adjusted = self.adjusted_weights.get(strategy, 0.0)
            changes[strategy] = adjusted - original
            
        return changes

@dataclass 
class RiskConfiguration:
    """リスク管理設定"""
    # VaRリミット
    var_95_limit: float = 0.05  # 5%
    var_99_limit: float = 0.08  # 8%
    
    # ドローダウンリミット
    max_drawdown_limit: float = 0.15  # 15%
    
    # ボラティリティリミット
    volatility_limit: float = 0.25  # 25%
    
    # 相関リスクリミット
    max_correlation: float = 0.8  # 80%
    
    # 集中度リミット
    max_single_position: float = 0.2  # 20%
    max_sector_concentration: float = 0.4  # 40%
    
    # 動的調整パラメータ
    adjustment_sensitivity: float = 0.5  # 調整感度
    rebalance_threshold: float = 0.1  # リバランス閾値
    
    # 制限タイプ設定
    limit_types: Dict[RiskMetricType, RiskLimitType] = field(default_factory=lambda: {
        RiskMetricType.VAR_95: RiskLimitType.SOFT_LIMIT,
        RiskMetricType.VAR_99: RiskLimitType.HARD_LIMIT,
        RiskMetricType.MAX_DRAWDOWN: RiskLimitType.HARD_LIMIT,
        RiskMetricType.VOLATILITY: RiskLimitType.SOFT_LIMIT,
        RiskMetricType.CONCENTRATION_RISK: RiskLimitType.DYNAMIC_LIMIT,
        RiskMetricType.CORRELATION_RISK: RiskLimitType.SOFT_LIMIT
    })

class RiskCalculationEngine(ABC):
    """リスク計算エンジンの抽象基底クラス"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, weights: Dict[str, float]) -> float:
        """リスク指標を計算"""
        pass

class VaRCalculator(RiskCalculationEngine):
    """Value at Risk 計算器"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """VaRを計算"""
        try:
            if returns.empty or not weights:
                return 0.0
            
            # ポートフォリオリターンを計算
            portfolio_returns = self._calculate_portfolio_returns(returns, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0
            
            # VaRを計算
            var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
            return abs(var)
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0
    
    def _calculate_portfolio_returns(self, returns: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """ポートフォリオリターンを計算"""
        try:
            # ウェイトを正規化
            total_weight = sum(weights.values())
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # 共通する戦略のみを対象
            common_strategies = set(returns.columns) & set(normalized_weights.keys())
            if not common_strategies:
                return np.array([])
            
            # ポートフォリオリターンを計算
            portfolio_returns = np.zeros(len(returns))
            for strategy in common_strategies:
                if strategy in returns.columns:
                    portfolio_returns += returns[strategy].fillna(0).values * normalized_weights[strategy]
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])

class ConditionalVaRCalculator(VaRCalculator):
    """Conditional Value at Risk 計算器"""
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """CVaRを計算"""
        try:
            if returns.empty or not weights:
                return 0.0
            
            # ポートフォリオリターンを計算
            portfolio_returns = self._calculate_portfolio_returns(returns, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0
            
            # VaRを計算
            var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
            
            # CVaRを計算（VaR以下の値の平均）
            tail_losses = portfolio_returns[portfolio_returns <= var]
            if len(tail_losses) == 0:
                return abs(var)
            
            cvar = np.mean(tail_losses)
            return abs(cvar)
            
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return 0.0

class DrawdownCalculator(RiskCalculationEngine):
    """ドローダウン計算器"""
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """最大ドローダウンを計算"""
        try:
            if returns.empty or not weights:
                return 0.0
            
            # ポートフォリオリターンを計算
            portfolio_returns = self._calculate_portfolio_returns(returns, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0
            
            # 累積リターンを計算
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # 過去最高値からのドローダウンを計算
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Drawdown calculation error: {e}")
            return 0.0
    
    def _calculate_portfolio_returns(self, returns: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """ポートフォリオリターンを計算"""
        try:
            # ウェイトを正規化
            total_weight = sum(weights.values())
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # 共通する戦略のみを対象
            common_strategies = set(returns.columns) & set(normalized_weights.keys())
            if not common_strategies:
                return np.array([])
            
            # ポートフォリオリターンを計算
            portfolio_returns = np.zeros(len(returns))
            for strategy in common_strategies:
                if strategy in returns.columns:
                    portfolio_returns += returns[strategy].fillna(0).values * normalized_weights[strategy]
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])

class VolatilityCalculator(RiskCalculationEngine):
    """ボラティリティ計算器"""
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """ポートフォリオボラティリティを計算"""
        try:
            if returns.empty or not weights:
                return 0.0
            
            # ポートフォリオリターンを計算
            portfolio_returns = self._calculate_portfolio_returns(returns, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0
            
            # ボラティリティを計算（年率換算）
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # 252営業日
            return volatility
            
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.0
    
    def _calculate_portfolio_returns(self, returns: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """ポートフォリオリターンを計算"""
        try:
            # ウェイトを正規化
            total_weight = sum(weights.values())
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # 共通する戦略のみを対象
            common_strategies = set(returns.columns) & set(normalized_weights.keys())
            if not common_strategies:
                return np.array([])
            
            # ポートフォリオリターンを計算
            portfolio_returns = np.zeros(len(returns))
            for strategy in common_strategies:
                if strategy in returns.columns:
                    portfolio_returns += returns[strategy].fillna(0).values * normalized_weights[strategy]
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])

class CorrelationRiskCalculator(RiskCalculationEngine):
    """相関リスク計算器"""
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """戦略間の相関リスクを計算"""
        try:
            if returns.empty or not weights or len(weights) < 2:
                return 0.0
            
            # 共通する戦略のみを対象
            common_strategies = list(set(returns.columns) & set(weights.keys()))
            if len(common_strategies) < 2:
                return 0.0
            
            # 相関行列を計算
            strategy_returns = returns[common_strategies].dropna()
            if len(strategy_returns) < 10:  # 最小データ数チェック
                return 0.0
            
            correlation_matrix = strategy_returns.corr()
            
            # 重み付き平均相関を計算
            total_correlation = 0.0
            total_weight_pairs = 0.0
            
            for i, strategy1 in enumerate(common_strategies):
                for j, strategy2 in enumerate(common_strategies):
                    if i < j:  # 上三角行列のみ
                        correlation = correlation_matrix.loc[strategy1, strategy2]
                        weight1 = weights.get(strategy1, 0.0)
                        weight2 = weights.get(strategy2, 0.0)
                        weight_product = weight1 * weight2
                        
                        total_correlation += abs(correlation) * weight_product
                        total_weight_pairs += weight_product
            
            if total_weight_pairs == 0:
                return 0.0
            
            average_correlation = total_correlation / total_weight_pairs
            return average_correlation
            
        except Exception as e:
            logger.error(f"Correlation risk calculation error: {e}")
            return 0.0

class ConcentrationRiskCalculator(RiskCalculationEngine):
    """集中度リスク計算器"""
    
    def calculate(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """ポートフォリオの集中度リスクを計算"""
        try:
            if not weights:
                return 0.0
            
            # ウェイトを正規化
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            normalized_weights = [w / total_weight for w in weights.values()]
            
            # HHI (Herfindahl-Hirschman Index) を計算
            hhi = sum(w ** 2 for w in normalized_weights)
            
            # 集中度リスクを0-1の範囲で正規化
            n = len(normalized_weights)
            if n <= 1:
                return 1.0
            
            min_hhi = 1.0 / n  # 完全分散時のHHI
            max_hhi = 1.0      # 完全集中時のHHI
            
            concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)
            return max(0.0, min(1.0, concentration_risk))
            
        except Exception as e:
            logger.error(f"Concentration risk calculation error: {e}")
            return 0.0

class PortfolioRiskManager:
    """ポートフォリオリスク管理メインクラス"""
    
    def __init__(self, 
                 config: RiskConfiguration,
                 portfolio_weight_calculator: PortfolioWeightCalculator,
                 position_size_adjuster: PositionSizeAdjuster,
                 signal_integrator: SignalIntegrator):
        """
        初期化
        
        Parameters:
            config: リスク管理設定
            portfolio_weight_calculator: ポートフォリオウェイト計算器
            position_size_adjuster: ポジションサイズ調整器  
            signal_integrator: シグナル統合器
        """
        self.config = config
        self.portfolio_weight_calculator = portfolio_weight_calculator
        self.position_size_adjuster = position_size_adjuster
        self.signal_integrator = signal_integrator
        
        # リスク計算器の初期化
        self.risk_calculators = {
            RiskMetricType.VAR_95: VaRCalculator(0.95),
            RiskMetricType.VAR_99: VaRCalculator(0.99),
            RiskMetricType.CVAR_95: ConditionalVaRCalculator(0.95),
            RiskMetricType.CVAR_99: ConditionalVaRCalculator(0.99),
            RiskMetricType.MAX_DRAWDOWN: DrawdownCalculator(),
            RiskMetricType.VOLATILITY: VolatilityCalculator(),
            RiskMetricType.CORRELATION_RISK: CorrelationRiskCalculator(),
            RiskMetricType.CONCENTRATION_RISK: ConcentrationRiskCalculator()
        }
        
        # 履歴保存
        self.risk_history: List[Dict[str, RiskMetric]] = []
        self.adjustment_history: List[RiskAdjustmentResult] = []
        
        # スレッドセーフティ
        self._lock = threading.Lock()
        
        logger.info("PortfolioRiskManager initialized successfully")
    
    def calculate_all_risk_metrics(self, 
                                   returns_data: pd.DataFrame, 
                                   current_weights: Dict[str, float]) -> Dict[str, RiskMetric]:
        """全リスク指標を計算"""
        risk_metrics = {}
        
        try:
            # 各リスク指標を並行計算
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_metric = {
                    executor.submit(self._calculate_single_metric, metric_type, returns_data, current_weights): metric_type
                    for metric_type in self.risk_calculators.keys()
                }
                
                for future in as_completed(future_to_metric):
                    metric_type = future_to_metric[future]
                    try:
                        risk_metric = future.result()
                        if risk_metric:
                            risk_metrics[metric_type.value] = risk_metric
                    except Exception as e:
                        logger.error(f"Error calculating {metric_type.value}: {e}")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_all_risk_metrics: {e}")
            return {}
    
    def _calculate_single_metric(self, 
                                metric_type: RiskMetricType, 
                                returns_data: pd.DataFrame, 
                                current_weights: Dict[str, float]) -> Optional[RiskMetric]:
        """単一リスク指標を計算"""
        try:
            calculator = self.risk_calculators.get(metric_type)
            if not calculator:
                return None
            
            current_value = calculator.calculate(returns_data, current_weights)
            limit_value = self._get_risk_limit(metric_type)
            limit_type = self.config.limit_types.get(metric_type, RiskLimitType.SOFT_LIMIT)
            
            risk_metric = RiskMetric(
                metric_type=metric_type,
                current_value=current_value,
                limit_value=limit_value,
                limit_type=limit_type
            )
            
            # 制限超過の深刻度を計算
            if risk_metric.is_breached:
                risk_metric.breach_severity = min(1.0, risk_metric.breach_ratio)
            
            return risk_metric
            
        except Exception as e:
            logger.error(f"Error calculating single metric {metric_type.value}: {e}")
            return None
    
    def _get_risk_limit(self, metric_type: RiskMetricType) -> float:
        """リスク制限値を取得"""
        limit_map = {
            RiskMetricType.VAR_95: self.config.var_95_limit,
            RiskMetricType.VAR_99: self.config.var_99_limit,
            RiskMetricType.MAX_DRAWDOWN: self.config.max_drawdown_limit,
            RiskMetricType.VOLATILITY: self.config.volatility_limit,
            RiskMetricType.CORRELATION_RISK: self.config.max_correlation,
            RiskMetricType.CONCENTRATION_RISK: self.config.max_single_position
        }
        return limit_map.get(metric_type, 1.0)
    
    def assess_portfolio_risk(self, 
                              returns_data: pd.DataFrame,
                              current_weights: Dict[str, float]) -> Tuple[Dict[str, RiskMetric], bool]:
        """ポートフォリオリスクを評価"""
        try:
            # 全リスク指標を計算
            risk_metrics = self.calculate_all_risk_metrics(returns_data, current_weights)
            
            # リスク制限違反をチェック
            needs_adjustment = False
            critical_breaches = 0
            
            for metric in risk_metrics.values():
                if metric.is_breached:
                    if metric.limit_type == RiskLimitType.HARD_LIMIT:
                        needs_adjustment = True
                        critical_breaches += 1
                    elif metric.limit_type == RiskLimitType.SOFT_LIMIT and metric.breach_severity > 0.5:
                        needs_adjustment = True
            
            # 履歴に保存
            with self._lock:
                self.risk_history.append(risk_metrics)
                # 履歴サイズ制限（最新1000件）
                if len(self.risk_history) > 1000:
                    self.risk_history = self.risk_history[-1000:]
            
            logger.info(f"Portfolio risk assessment completed. "
                       f"Needs adjustment: {needs_adjustment}, "
                       f"Critical breaches: {critical_breaches}")
            
            return risk_metrics, needs_adjustment
            
        except Exception as e:
            logger.error(f"Error in assess_portfolio_risk: {e}")
            return {}, False
    
    def adjust_portfolio_weights(self,
                                 returns_data: pd.DataFrame,
                                 current_weights: Dict[str, float],
                                 risk_metrics: Dict[str, RiskMetric]) -> RiskAdjustmentResult:
        """ポートフォリオウェイトを調整"""
        try:
            # 調整前の状態を記録
            original_weights = current_weights.copy()
            
            # 調整戦略を決定
            adjustment_actions = self._determine_adjustment_actions(risk_metrics)
            
            # ウェイト調整を実行
            adjusted_weights = self._execute_weight_adjustments(
                current_weights, risk_metrics, adjustment_actions
            )
            
            # 調整後のリスク指標を計算
            risk_metrics_after = self.calculate_all_risk_metrics(returns_data, adjusted_weights)
            
            # 調整結果を作成
            adjustment_result = RiskAdjustmentResult(
                timestamp=datetime.now(),
                original_weights=original_weights,
                adjusted_weights=adjusted_weights,
                adjustment_actions=adjustment_actions,
                risk_metrics_before=risk_metrics,
                risk_metrics_after=risk_metrics_after,
                adjustment_reason=self._generate_adjustment_reason(risk_metrics)
            )
            
            # 効果性スコアを計算
            adjustment_result.effectiveness_score = self._calculate_effectiveness_score(
                risk_metrics, risk_metrics_after
            )
            
            # 履歴に保存
            with self._lock:
                self.adjustment_history.append(adjustment_result)
                if len(self.adjustment_history) > 500:
                    self.adjustment_history = self.adjustment_history[-500:]
            
            logger.info(f"Portfolio weights adjusted. "
                       f"Effectiveness score: {adjustment_result.effectiveness_score:.3f}")
            
            return adjustment_result
            
        except Exception as e:
            logger.error(f"Error in adjust_portfolio_weights: {e}")
            return RiskAdjustmentResult(
                timestamp=datetime.now(),
                original_weights=current_weights,
                adjusted_weights=current_weights,
                adjustment_actions=[RiskAdjustmentAction.NO_ACTION],
                risk_metrics_before=risk_metrics,
                risk_metrics_after=risk_metrics,
                adjustment_reason=f"Adjustment failed: {str(e)}"
            )
    
    def _determine_adjustment_actions(self, risk_metrics: Dict[str, RiskMetric]) -> List[RiskAdjustmentAction]:
        """調整アクションを決定"""
        actions = []
        
        try:
            # 緊急事態チェック
            emergency_count = 0
            for metric in risk_metrics.values():
                if (metric.limit_type == RiskLimitType.HARD_LIMIT and 
                    metric.breach_severity > 0.8):
                    emergency_count += 1
            
            if emergency_count > 2:
                actions.append(RiskAdjustmentAction.EMERGENCY_EXIT)
                return actions
            
            # 制限違反タイプ別の対応
            for metric_name, metric in risk_metrics.items():
                if not metric.is_breached:
                    continue
                
                if metric.metric_type in [RiskMetricType.VAR_95, RiskMetricType.VAR_99]:
                    if metric.breach_severity > 0.5:
                        actions.append(RiskAdjustmentAction.REDUCE_POSITIONS)
                    else:
                        actions.append(RiskAdjustmentAction.INCREASE_HEDGING)
                
                elif metric.metric_type == RiskMetricType.CONCENTRATION_RISK:
                    actions.append(RiskAdjustmentAction.REBALANCE_WEIGHTS)
                
                elif metric.metric_type == RiskMetricType.CORRELATION_RISK:
                    actions.append(RiskAdjustmentAction.REBALANCE_WEIGHTS)
                
                elif metric.metric_type == RiskMetricType.MAX_DRAWDOWN:
                    if metric.breach_severity > 0.7:
                        actions.append(RiskAdjustmentAction.STOP_NEW_POSITIONS)
                    actions.append(RiskAdjustmentAction.REDUCE_POSITIONS)
            
            # デフォルトアクション
            if not actions:
                actions.append(RiskAdjustmentAction.NO_ACTION)
            
            # 重複除去
            actions = list(set(actions))
            
            return actions
            
        except Exception as e:
            logger.error(f"Error determining adjustment actions: {e}")
            return [RiskAdjustmentAction.NO_ACTION]
    
    def _execute_weight_adjustments(self,
                                    current_weights: Dict[str, float],
                                    risk_metrics: Dict[str, RiskMetric],
                                    actions: List[RiskAdjustmentAction]) -> Dict[str, float]:
        """ウェイト調整を実行"""
        try:
            adjusted_weights = current_weights.copy()
            
            for action in actions:
                if action == RiskAdjustmentAction.NO_ACTION:
                    continue
                    
                elif action == RiskAdjustmentAction.REDUCE_POSITIONS:
                    adjusted_weights = self._reduce_positions(adjusted_weights, risk_metrics)
                    
                elif action == RiskAdjustmentAction.REBALANCE_WEIGHTS:
                    adjusted_weights = self._rebalance_weights(adjusted_weights, risk_metrics)
                    
                elif action == RiskAdjustmentAction.EMERGENCY_EXIT:
                    # 全ポジション50%削減
                    adjusted_weights = {k: v * 0.5 for k, v in adjusted_weights.items()}
                    
                elif action == RiskAdjustmentAction.STOP_NEW_POSITIONS:
                    # 新規ポジションは外部制御で実装
                    pass
            
            # ウェイト正規化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error executing weight adjustments: {e}")
            return current_weights
    
    def _reduce_positions(self, 
                          weights: Dict[str, float], 
                          risk_metrics: Dict[str, RiskMetric]) -> Dict[str, float]:
        """ポジション削減"""
        try:
            adjusted_weights = weights.copy()
            
            # 最も高リスクの戦略を特定
            risk_scores = {}
            for strategy in weights.keys():
                risk_score = 0.0
                # 各リスク指標の寄与度を計算（簡易版）
                for metric in risk_metrics.values():
                    if metric.is_breached:
                        risk_score += metric.breach_severity
                risk_scores[strategy] = risk_score
            
            # リスクの高い戦略から削減
            sorted_strategies = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
            
            reduction_factor = 0.8  # 20%削減
            for i, (strategy, _) in enumerate(sorted_strategies):
                if i < len(sorted_strategies) // 2:  # 上位半分を削減
                    adjusted_weights[strategy] *= reduction_factor
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error in reduce_positions: {e}")
            return weights
    
    def _rebalance_weights(self, 
                           weights: Dict[str, float], 
                           risk_metrics: Dict[str, RiskMetric]) -> Dict[str, float]:
        """ウェイト再配分"""
        try:
            adjusted_weights = weights.copy()
            
            # 集中度リスクが高い場合の均等化
            concentration_metric = risk_metrics.get('concentration_risk')
            if concentration_metric and concentration_metric.is_breached:
                # より均等な配分に調整
                n_strategies = len(weights)
                if n_strategies > 0:
                    target_weight = 1.0 / n_strategies
                    
                    # 現在のウェイトと目標ウェイトの中間値に調整
                    for strategy in adjusted_weights:
                        current = adjusted_weights[strategy]
                        adjusted_weights[strategy] = (current + target_weight) / 2
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error in rebalance_weights: {e}")
            return weights
    
    def _generate_adjustment_reason(self, risk_metrics: Dict[str, RiskMetric]) -> str:
        """調整理由を生成"""
        try:
            breached_metrics = [
                metric.metric_type.value for metric in risk_metrics.values() 
                if metric.is_breached
            ]
            
            if not breached_metrics:
                return "No risk limit breaches detected"
            
            return f"Risk limit breaches detected: {', '.join(breached_metrics)}"
            
        except Exception as e:
            logger.error(f"Error generating adjustment reason: {e}")
            return "Unknown adjustment reason"
    
    def _calculate_effectiveness_score(self, 
                                       before_metrics: Dict[str, RiskMetric],
                                       after_metrics: Dict[str, RiskMetric]) -> float:
        """調整効果性スコアを計算"""
        try:
            if not before_metrics or not after_metrics:
                return 0.0
            
            improvements = 0
            total_comparisons = 0
            
            for metric_name in before_metrics.keys():
                if metric_name in after_metrics:
                    before_value = before_metrics[metric_name].current_value
                    after_value = after_metrics[metric_name].current_value
                    limit_value = before_metrics[metric_name].limit_value
                    
                    # 制限値に対する改善度を計算
                    if limit_value > 0:
                        before_distance = abs(before_value - limit_value) / limit_value
                        after_distance = abs(after_value - limit_value) / limit_value
                        
                        if after_distance < before_distance:
                            improvements += (before_distance - after_distance) / before_distance
                        
                        total_comparisons += 1
            
            if total_comparisons == 0:
                return 0.0
            
            effectiveness_score = max(0.0, min(1.0, improvements / total_comparisons))
            return effectiveness_score
            
        except Exception as e:
            logger.error(f"Error calculating effectiveness score: {e}")
            return 0.0
    
    def run_portfolio_risk_management(self, 
                                      returns_data: pd.DataFrame,
                                      current_weights: Dict[str, float]) -> RiskAdjustmentResult:
        """ポートフォリオリスク管理のメインフロー"""
        try:
            logger.info("Starting portfolio risk management cycle")
            
            # 1. リスク評価
            risk_metrics, needs_adjustment = self.assess_portfolio_risk(returns_data, current_weights)
            
            # 2. 調整が必要な場合のみ実行
            if needs_adjustment:
                logger.info("Risk adjustment required")
                adjustment_result = self.adjust_portfolio_weights(returns_data, current_weights, risk_metrics)
            else:
                logger.info("No risk adjustment required")
                adjustment_result = RiskAdjustmentResult(
                    timestamp=datetime.now(),
                    original_weights=current_weights,
                    adjusted_weights=current_weights,
                    adjustment_actions=[RiskAdjustmentAction.NO_ACTION],
                    risk_metrics_before=risk_metrics,
                    risk_metrics_after=risk_metrics,
                    adjustment_reason="All risk metrics within acceptable limits"
                )
            
            return adjustment_result
            
        except Exception as e:
            logger.error(f"Error in run_portfolio_risk_management: {e}")
            # エラー時のフォールバック
            return RiskAdjustmentResult(
                timestamp=datetime.now(),
                original_weights=current_weights,
                adjusted_weights=current_weights,
                adjustment_actions=[RiskAdjustmentAction.NO_ACTION],
                risk_metrics_before={},
                risk_metrics_after={},
                adjustment_reason=f"Error in risk management: {str(e)}"
            )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """リスク管理サマリーを取得"""
        try:
            if not self.risk_history:
                return {"status": "no_data"}
            
            latest_metrics = self.risk_history[-1]
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_strategies": len(self.portfolio_weight_calculator.strategy_weights) if hasattr(self.portfolio_weight_calculator, 'strategy_weights') else 0,
                "risk_metrics": {},
                "breaches": [],
                "adjustment_history_count": len(self.adjustment_history),
                "last_adjustment": None
            }
            
            # 最新のリスク指標
            for metric_name, metric in latest_metrics.items():
                summary["risk_metrics"][metric_name] = {
                    "current_value": metric.current_value,
                    "limit_value": metric.limit_value,
                    "is_breached": metric.is_breached,
                    "breach_severity": metric.breach_severity
                }
                
                if metric.is_breached:
                    summary["breaches"].append({
                        "metric": metric_name,
                        "severity": metric.breach_severity,
                        "limit_type": metric.limit_type.value
                    })
            
            # 最新の調整情報
            if self.adjustment_history:
                latest_adjustment = self.adjustment_history[-1]
                summary["last_adjustment"] = {
                    "timestamp": latest_adjustment.timestamp.isoformat(),
                    "actions": [action.value for action in latest_adjustment.adjustment_actions],
                    "effectiveness_score": latest_adjustment.effectiveness_score,
                    "reason": latest_adjustment.adjustment_reason
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_risk_report(self, filepath: str) -> bool:
        """リスクレポートをファイルに保存"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "config": {
                    "var_95_limit": self.config.var_95_limit,
                    "var_99_limit": self.config.var_99_limit,
                    "max_drawdown_limit": self.config.max_drawdown_limit,
                    "volatility_limit": self.config.volatility_limit,
                    "max_correlation": self.config.max_correlation,
                    "max_single_position": self.config.max_single_position
                },
                "summary": self.get_risk_summary(),
                "recent_adjustments": [
                    {
                        "timestamp": adj.timestamp.isoformat(),
                        "actions": [action.value for action in adj.adjustment_actions],
                        "effectiveness_score": adj.effectiveness_score,
                        "weight_changes": adj.get_weight_changes()
                    }
                    for adj in self.adjustment_history[-10:]  # 最新10件
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Risk report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving risk report: {e}")
            return False

# 統合システムクラス
class IntegratedRiskManagementSystem:
    """統合リスク管理システム"""
    
    def __init__(self, 
                 risk_config: RiskConfiguration,
                 weight_config: WeightAllocationConfig,
                 adjustment_config: str):  # 設定ファイルパスを受け取る
        """
        統合システムの初期化
        
        Parameters:
            risk_config: リスク管理設定
            weight_config: ウェイト配分設定
            adjustment_config: ポジション調整設定ファイルパス
        """
        try:
            # 既存システムの初期化
            self.portfolio_weight_calculator = PortfolioWeightCalculator(None)
            self.position_size_adjuster = PositionSizeAdjuster(adjustment_config)
            self.signal_integrator = SignalIntegrator()
            
            # リスク管理システムの初期化
            self.portfolio_risk_manager = PortfolioRiskManager(
                config=risk_config,
                portfolio_weight_calculator=self.portfolio_weight_calculator,
                position_size_adjuster=self.position_size_adjuster,
                signal_integrator=self.signal_integrator
            )
            
            logger.info("IntegratedRiskManagementSystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing IntegratedRiskManagementSystem: {e}")
            raise
    
    def run_complete_portfolio_management(self, 
                                          returns_data: pd.DataFrame,
                                          strategy_signals: Dict[str, Any],  # StrategySignalの代替
                                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """完全なポートフォリオ管理フローを実行"""
        try:
            logger.info("Starting complete portfolio management flow")
            
            # 1. シグナル統合（簡易版）
            integrated_signals = strategy_signals  # シンプルなパススルー
            
            # 2. ポートフォリオウェイト計算（ダミー実装）
            portfolio_weights = {
                strategy: 0.25 for strategy in strategy_signals.keys()
            }
            
            # 3. ポジションサイズ調整（ダミー実装）
            adjusted_positions = portfolio_weights.copy()
            
            # 4. ポートフォリオリスク管理
            risk_adjustment = self.portfolio_risk_manager.run_portfolio_risk_management(
                returns_data, adjusted_positions
            )
            
            # 5. 結果統合
            final_result = {
                "timestamp": datetime.now().isoformat(),
                "integrated_signals": integrated_signals,
                "portfolio_weights": portfolio_weights,
                "position_adjustments": adjusted_positions,
                "risk_adjustment": risk_adjustment,
                "final_weights": risk_adjustment.adjusted_weights,
                "total_effectiveness": 0.8  # ダミー値
            }
            
            logger.info("Complete portfolio management flow completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in complete portfolio management flow: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(e),
                "final_weights": {}
            }
    
    def _calculate_total_effectiveness(self, 
                                       portfolio_result: Any,
                                       adjustment_result: Any,
                                       risk_result: RiskAdjustmentResult) -> float:
        """総合効果性スコアを計算"""
        try:
            # 簡易版: リスク調整の効果性スコアのみを返す
            return risk_result.effectiveness_score
            
        except Exception as e:
            logger.error(f"Error calculating total effectiveness: {e}")
            return 0.0

if __name__ == "__main__":
    # 使用例とテスト
    print("Portfolio Risk Manager - Test Mode")
    
    try:
        # 設定の初期化
        risk_config = RiskConfiguration()
        
        # ダミー設定でテスト
        weight_calculator = PortfolioWeightCalculator(None)
        position_adjuster = PositionSizeAdjuster("dummy_config.json")
        signal_integrator = SignalIntegrator()
        
        # リスク管理システムの初期化
        risk_manager = PortfolioRiskManager(
            config=risk_config,
            portfolio_weight_calculator=weight_calculator,
            position_size_adjuster=position_adjuster,
            signal_integrator=signal_integrator
        )
        
        # テスト用データ
        test_returns = pd.DataFrame({
            'strategy_1': np.random.normal(0.001, 0.02, 100),
            'strategy_2': np.random.normal(0.0005, 0.015, 100),
            'strategy_3': np.random.normal(0.0008, 0.018, 100)
        })
        
        test_weights = {
            'strategy_1': 0.4,
            'strategy_2': 0.35,
            'strategy_3': 0.25
        }
        
        # リスク評価テスト
        risk_metrics, needs_adjustment = risk_manager.assess_portfolio_risk(
            test_returns, test_weights
        )
        
        print(f"Risk assessment completed:")
        print(f"  Needs adjustment: {needs_adjustment}")
        print(f"  Risk metrics count: {len(risk_metrics)}")
        
        # リスク調整テスト
        if needs_adjustment:
            adjustment_result = risk_manager.adjust_portfolio_weights(
                test_returns, test_weights, risk_metrics
            )
            print(f"Risk adjustment completed:")
            print(f"  Effectiveness score: {adjustment_result.effectiveness_score:.3f}")
            print(f"  Actions: {[action.value for action in adjustment_result.adjustment_actions]}")
        
        # サマリー出力
        summary = risk_manager.get_risk_summary()
        print(f"Risk summary: {json.dumps(summary, indent=2)}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
