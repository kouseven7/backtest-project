"""
Module: Portfolio Weight Calculator
File: portfolio_weight_calculator.py
Description: 
  3-2-1「スコアベースの資金配分計算式設計」
  戦略スコアを基にしたポートフォリオ重み計算システム
  既存のStrategyScoreとの完全統合

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_scoring_model
  - config.strategy_selector
  - config.metric_weight_optimizer
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

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.strategy_scoring_model import (
        StrategyScoreCalculator, StrategyScoreManager, StrategyScore, ScoreWeights
    )
    from config.strategy_selector import StrategySelector, StrategySelection, SelectionCriteria
    from config.metric_weight_optimizer import MetricWeightOptimizer
    from config.risk_management import RiskManagement
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class AllocationMethod(Enum):
    """資金配分手法"""
    SCORE_PROPORTIONAL = "score_proportional"      # スコア比例配分
    RISK_ADJUSTED = "risk_adjusted"                # リスク調整配分
    EQUAL_WEIGHT = "equal_weight"                  # 等重み配分
    HIERARCHICAL = "hierarchical"                  # 階層的配分
    KELLY_CRITERION = "kelly_criterion"            # ケリー基準

class ConstraintType(Enum):
    """制約タイプ"""
    MAX_WEIGHT = "max_weight"          # 最大重み制約
    MIN_WEIGHT = "min_weight"          # 最小重み制約
    STRATEGY_LIMIT = "strategy_limit"  # 戦略数制約
    CORRELATION = "correlation"        # 相関制約
    SECTOR_LIMIT = "sector_limit"      # セクター制約

class RebalanceFrequency(Enum):
    """リバランス頻度"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    DYNAMIC = "dynamic"  # 動的（条件ベース）

@dataclass
class PortfolioConstraints:
    """ポートフォリオ制約設定"""
    max_individual_weight: float = 0.4           # 個別戦略最大重み
    min_individual_weight: float = 0.05          # 個別戦略最小重み
    max_strategies: int = 5                      # 最大戦略数
    min_strategies: int = 2                      # 最小戦略数
    max_correlation_threshold: float = 0.8       # 最大相関閾値
    min_score_threshold: float = 0.3             # 最小スコア閾値
    max_turnover: float = 0.2                    # 最大ターンオーバー率
    risk_budget: float = 0.15                    # リスクバジェット
    leverage_limit: float = 1.0                  # レバレッジ制限
    concentration_limit: float = 0.6             # 集中度制限（上位3戦略の合計重み）

@dataclass
class WeightAllocationConfig:
    """重み配分設定"""
    method: AllocationMethod = AllocationMethod.RISK_ADJUSTED
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    risk_aversion: float = 2.0                   # リスク回避度
    confidence_weight: float = 0.3               # 信頼度ウェイト
    trend_weight: float = 0.2                    # トレンド適応ウェイト
    volatility_lookback: int = 252               # ボラティリティ計算期間
    enable_dynamic_adjustment: bool = True       # 動的調整有効化
    enable_momentum_bias: bool = False           # モメンタムバイアス有効化

@dataclass
class AllocationResult:
    """配分結果"""
    strategy_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    constraint_violations: List[str]
    allocation_reason: str
    confidence_level: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAllocationStrategy(ABC):
    """配分戦略基底クラス"""
    
    @abstractmethod
    def calculate_weights(self, 
                         strategy_scores: Dict[str, StrategyScore],
                         config: WeightAllocationConfig,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """重み計算の抽象メソッド"""
        pass

class ScoreProportionalAllocation(BaseAllocationStrategy):
    """スコア比例配分"""
    
    def calculate_weights(self, 
                         strategy_scores: Dict[str, StrategyScore],
                         config: WeightAllocationConfig,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """スコア比例による重み計算"""
        if not strategy_scores:
            return {}
        
        # 最小スコア閾値でフィルタリング
        filtered_scores = {
            name: score for name, score in strategy_scores.items()
            if score.total_score >= config.constraints.min_score_threshold
        }
        
        if not filtered_scores:
            logger.warning("No strategies meet minimum score threshold")
            return {}
        
        # スコア合計を計算
        total_score = sum(score.total_score for score in filtered_scores.values())
        
        if total_score == 0:
            # 等重み配分にフォールバック
            equal_weight = 1.0 / len(filtered_scores)
            return {name: equal_weight for name in filtered_scores}
        
        # スコア比例重み計算
        weights = {}
        for name, score in filtered_scores.items():
            raw_weight = score.total_score / total_score
            weights[name] = raw_weight
        
        # 制約適用
        return self._apply_constraints(weights, config.constraints)
    
    def _apply_constraints(self, weights: Dict[str, float], 
                          constraints: PortfolioConstraints) -> Dict[str, float]:
        """制約の適用"""
        # 最大・最小重み制約
        constrained_weights = {}
        for name, weight in weights.items():
            constrained_weights[name] = max(
                constraints.min_individual_weight,
                min(constraints.max_individual_weight, weight)
            )
        
        # 重みの正規化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                name: weight / total_weight 
                for name, weight in constrained_weights.items()
            }
        
        return constrained_weights

class RiskAdjustedAllocation(BaseAllocationStrategy):
    """リスク調整配分"""
    
    def calculate_weights(self, 
                         strategy_scores: Dict[str, StrategyScore],
                         config: WeightAllocationConfig,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """リスク調整による重み計算"""
        if not strategy_scores:
            return {}
        
        # リスク指標の取得
        risk_adjusted_scores = {}
        for name, score in strategy_scores.items():
            if score.total_score >= config.constraints.min_score_threshold:
                # リスク調整スコア = 基本スコア × (1 + リスク調整係数)
                risk_component = score.component_scores.get('risk_adjusted', 0.5)
                trend_component = score.component_scores.get('trend_adaptation', 0.5) 
                confidence = score.confidence
                
                risk_adjustment = (
                    risk_component * 0.4 + 
                    trend_component * config.trend_weight + 
                    confidence * config.confidence_weight
                )
                
                adjusted_score = score.total_score * (1 + risk_adjustment)
                risk_adjusted_scores[name] = adjusted_score
        
        if not risk_adjusted_scores:
            return {}
        
        # 重み計算
        total_adjusted_score = sum(risk_adjusted_scores.values())
        weights = {
            name: score / total_adjusted_score 
            for name, score in risk_adjusted_scores.items()
        }
        
        # 制約適用
        return self._apply_risk_constraints(weights, config)
    
    def _apply_risk_constraints(self, weights: Dict[str, float], 
                               config: WeightAllocationConfig) -> Dict[str, float]:
        """リスク制約の適用"""
        constraints = config.constraints
        
        # 集中度制約
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_3_weight = sum(weight for _, weight in sorted_weights[:3])
        
        if top_3_weight > constraints.concentration_limit:
            # 重みの再配分
            adjustment_factor = constraints.concentration_limit / top_3_weight
            for i in range(min(3, len(sorted_weights))):
                name, weight = sorted_weights[i]
                weights[name] = weight * adjustment_factor
        
        # 重みの正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights

class HierarchicalAllocation(BaseAllocationStrategy):
    """階層的配分"""
    
    def calculate_weights(self, 
                         strategy_scores: Dict[str, StrategyScore],
                         config: WeightAllocationConfig,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """階層的配分による重み計算"""
        if not strategy_scores:
            return {}
        
        # 戦略をティアに分類
        tiers = self._classify_strategies_into_tiers(strategy_scores, config)
        
        # ティア別重み配分
        tier_weights = {
            'tier_1': 0.6,  # 高スコア戦略
            'tier_2': 0.3,  # 中スコア戦略
            'tier_3': 0.1   # 低スコア戦略
        }
        
        weights = {}
        for tier_name, strategies in tiers.items():
            if not strategies:
                continue
            
            tier_total_weight = tier_weights.get(tier_name, 0)
            individual_weight = tier_total_weight / len(strategies)
            
            for strategy_name in strategies:
                weights[strategy_name] = individual_weight
        
        return weights
    
    def _classify_strategies_into_tiers(self, 
                                      strategy_scores: Dict[str, StrategyScore],
                                      config: WeightAllocationConfig) -> Dict[str, List[str]]:
        """戦略をティアに分類"""
        scores = [(name, score.total_score) for name, score in strategy_scores.items()
                 if score.total_score >= config.constraints.min_score_threshold]
        
        if not scores:
            return {'tier_1': [], 'tier_2': [], 'tier_3': []}
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        n = len(scores)
        tier_1_count = max(1, n // 3)
        tier_2_count = max(1, n // 2)
        
        tiers = {
            'tier_1': [name for name, _ in scores[:tier_1_count]],
            'tier_2': [name for name, _ in scores[tier_1_count:tier_1_count + tier_2_count]],
            'tier_3': [name for name, _ in scores[tier_1_count + tier_2_count:]]
        }
        
        return tiers

class KellyCriterionAllocation(BaseAllocationStrategy):
    """ケリー基準配分"""
    
    def calculate_weights(self, 
                         strategy_scores: Dict[str, StrategyScore],
                         config: WeightAllocationConfig,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """ケリー基準による重み計算"""
        if not strategy_scores:
            return {}
        
        kelly_weights = {}
        for name, score in strategy_scores.items():
            if score.total_score < config.constraints.min_score_threshold:
                continue
            
            # 期待リターンと勝率を推定
            win_rate = score.component_scores.get('performance', 0.5)
            expected_return = score.total_score * 0.1  # スコアから期待リターンを推定
            
            # ケリー基準の計算 f = (bp - q) / b
            # ここでは簡略化した計算を使用
            if win_rate > 0.5 and expected_return > 0:
                odds = win_rate / (1 - win_rate)
                kelly_fraction = (odds * win_rate - (1 - win_rate)) / odds
                kelly_fraction = max(0, min(kelly_fraction, config.constraints.max_individual_weight))
                kelly_weights[name] = kelly_fraction
        
        if not kelly_weights:
            return {}
        
        # 重みの正規化
        total_weight = sum(kelly_weights.values())
        if total_weight > 0:
            kelly_weights = {name: weight / total_weight for name, weight in kelly_weights.items()}
        
        return kelly_weights

class PortfolioWeightCalculator:
    """
    ポートフォリオ重み計算エンジン
    
    機能:
    1. 複数の配分手法による重み計算
    2. 制約管理とリスク制御
    3. 既存スコアリングシステムとの統合
    4. 動的リバランシング
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 base_dir: Optional[str] = None):
        """重み計算エンジンの初期化"""
        self.base_dir = Path(base_dir) if base_dir else Path("config/portfolio_weights")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定の読み込み
        self.config = self._load_config(config_file)
        
        # 既存システムとの連携
        self.score_manager = StrategyScoreManager()
        self.strategy_selector = StrategySelector()
        self.weight_optimizer = MetricWeightOptimizer()
        self.risk_manager = RiskManagement(total_assets=1000000)  # デフォルト資産額
        
        # 配分戦略の登録
        self.allocation_strategies = {
            AllocationMethod.SCORE_PROPORTIONAL: ScoreProportionalAllocation(),
            AllocationMethod.RISK_ADJUSTED: RiskAdjustedAllocation(),
            AllocationMethod.EQUAL_WEIGHT: self._get_equal_weight_strategy(),
            AllocationMethod.HIERARCHICAL: HierarchicalAllocation(),
            AllocationMethod.KELLY_CRITERION: KellyCriterionAllocation()
        }
        
        # キャッシュとパフォーマンス
        self._weight_cache = {}
        self._last_calculation_time = None
        self._allocation_history = []
        
        logger.info("PortfolioWeightCalculator initialized")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # デフォルト設定
        return {
            "default_allocation_config": {
                "method": "risk_adjusted",
                "risk_aversion": 2.0,
                "confidence_weight": 0.3,
                "trend_weight": 0.2
            },
            "default_constraints": {
                "max_individual_weight": 0.4,
                "min_individual_weight": 0.05,
                "max_strategies": 5,
                "min_strategies": 2,
                "max_correlation_threshold": 0.8,
                "min_score_threshold": 0.3
            },
            "cache_ttl_seconds": 600,
            "enable_logging": True
        }

    def _get_equal_weight_strategy(self) -> BaseAllocationStrategy:
        """等重み配分戦略を取得"""
        class EqualWeightAllocation(BaseAllocationStrategy):
            def calculate_weights(self, strategy_scores, config, market_context=None):
                if not strategy_scores:
                    return {}
                
                filtered_scores = {
                    name: score for name, score in strategy_scores.items()
                    if score.total_score >= config.constraints.min_score_threshold
                }
                
                if not filtered_scores:
                    return {}
                
                equal_weight = 1.0 / len(filtered_scores)
                return {name: equal_weight for name in filtered_scores}
        
        return EqualWeightAllocation()

    def calculate_portfolio_weights(self,
                                  ticker: str,
                                  market_data: pd.DataFrame,
                                  config: Optional[WeightAllocationConfig] = None,
                                  strategy_filter: Optional[List[str]] = None) -> AllocationResult:
        """
        ポートフォリオ重み計算のメインメソッド
        
        Parameters:
            ticker: 対象銘柄
            market_data: 市場データ
            config: 配分設定
            strategy_filter: 戦略フィルター
            
        Returns:
            AllocationResult: 配分結果
        """
        start_time = datetime.now()
        
        try:
            # デフォルト設定の適用
            if config is None:
                config = self._create_default_config()
            
            # 戦略スコアの取得
            strategy_scores = self._get_strategy_scores(ticker, strategy_filter)
            
            if not strategy_scores:
                return self._create_empty_result("No valid strategy scores available")
            
            # 市場コンテキストの構築
            market_context = self._build_market_context(market_data, ticker)
            
            # 重み計算
            allocation_strategy = self.allocation_strategies.get(config.method)
            if not allocation_strategy:
                raise ValueError(f"Unsupported allocation method: {config.method}")
            
            strategy_weights = allocation_strategy.calculate_weights(
                strategy_scores, config, market_context
            )
            
            # 制約チェック
            constraint_violations = self._check_constraints(strategy_weights, config.constraints)
            
            # パフォーマンス指標の計算
            expected_return, expected_risk, sharpe_ratio = self._calculate_portfolio_metrics(
                strategy_weights, strategy_scores, market_context
            )
            
            # 分散化比率の計算
            diversification_ratio = self._calculate_diversification_ratio(strategy_weights)
            
            # 信頼度の計算
            confidence_level = self._calculate_allocation_confidence(
                strategy_weights, strategy_scores, constraint_violations
            )
            
            # 結果の作成
            result = AllocationResult(
                strategy_weights=strategy_weights,
                expected_return=expected_return,
                expected_risk=expected_risk,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio,
                constraint_violations=constraint_violations,
                allocation_reason=self._generate_allocation_reason(config, strategy_weights),
                confidence_level=confidence_level,
                metadata={
                    "ticker": ticker,
                    "config": config.__dict__,
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "num_strategies": len(strategy_weights),
                    "allocation_method": config.method.value
                }
            )
            
            # 履歴に追加
            self._allocation_history.append(result)
            self._last_calculation_time = datetime.now()
            
            logger.info(f"Portfolio weights calculated for {ticker}: {len(strategy_weights)} strategies")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating portfolio weights for {ticker}: {e}")
            return self._create_empty_result(f"Calculation error: {str(e)}")

    def _create_default_config(self) -> WeightAllocationConfig:
        """デフォルト設定の作成"""
        return WeightAllocationConfig(
            method=AllocationMethod.RISK_ADJUSTED,
            constraints=PortfolioConstraints(),
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            risk_aversion=2.0,
            confidence_weight=0.3,
            trend_weight=0.2
        )

    def _get_strategy_scores(self, ticker: str, 
                           strategy_filter: Optional[List[str]] = None) -> Dict[str, StrategyScore]:
        """戦略スコアの取得"""
        try:
            # 既存のスコアマネージャーからスコアを取得
            all_scores = self.score_manager.calculate_comprehensive_scores([ticker])
            
            if ticker not in all_scores:
                logger.warning(f"No scores available for ticker: {ticker}")
                return {}
            
            ticker_scores = all_scores[ticker]
            
            # フィルター適用
            if strategy_filter:
                filtered_scores = {
                    name: score for name, score in ticker_scores.items()
                    if name in strategy_filter
                }
                return filtered_scores
            
            return ticker_scores
            
        except Exception as e:
            logger.error(f"Error getting strategy scores for {ticker}: {e}")
            return {}

    def _build_market_context(self, market_data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """市場コンテキストの構築"""
        try:
            if market_data.empty:
                return {"volatility": 0.2, "trend": "unknown"}
            
            # ボラティリティ計算
            returns = market_data['Adj Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
            
            # 簡単なトレンド判定
            if len(market_data) >= 20:
                short_ma = market_data['Adj Close'].rolling(10).mean().iloc[-1]
                long_ma = market_data['Adj Close'].rolling(20).mean().iloc[-1]
                trend = "uptrend" if short_ma > long_ma else "downtrend"
            else:
                trend = "unknown"
            
            return {
                "volatility": volatility,
                "trend": trend,
                "ticker": ticker,
                "data_points": len(market_data)
            }
            
        except Exception as e:
            logger.warning(f"Error building market context: {e}")
            return {"volatility": 0.2, "trend": "unknown"}

    def _check_constraints(self, weights: Dict[str, float], 
                          constraints: PortfolioConstraints) -> List[str]:
        """制約チェック"""
        violations = []
        
        if not weights:
            return ["No weights calculated"]
        
        # 戦略数制約
        num_strategies = len(weights)
        if num_strategies > constraints.max_strategies:
            violations.append(f"Too many strategies: {num_strategies} > {constraints.max_strategies}")
        if num_strategies < constraints.min_strategies:
            violations.append(f"Too few strategies: {num_strategies} < {constraints.min_strategies}")
        
        # 重み制約
        for name, weight in weights.items():
            if weight > constraints.max_individual_weight:
                violations.append(f"Weight too high for {name}: {weight:.3f} > {constraints.max_individual_weight}")
            if weight < constraints.min_individual_weight:
                violations.append(f"Weight too low for {name}: {weight:.3f} < {constraints.min_individual_weight}")
        
        # 集中度制約
        sorted_weights = sorted(weights.values(), reverse=True)
        top_3_concentration = sum(sorted_weights[:3])
        if top_3_concentration > constraints.concentration_limit:
            violations.append(f"Concentration too high: {top_3_concentration:.3f} > {constraints.concentration_limit}")
        
        return violations

    def _calculate_portfolio_metrics(self, 
                                   weights: Dict[str, float],
                                   strategy_scores: Dict[str, StrategyScore],
                                   market_context: Dict[str, Any]) -> Tuple[float, float, float]:
        """ポートフォリオ指標の計算"""
        try:
            if not weights:
                return 0.0, 0.0, 0.0
            
            # 期待リターンの計算
            expected_return = sum(
                weight * strategy_scores[name].total_score * 0.1 
                for name, weight in weights.items()
                if name in strategy_scores
            )
            
            # 期待リスクの計算（簡略化）
            portfolio_volatility = market_context.get("volatility", 0.2)
            diversification_benefit = 1.0 - (1.0 / len(weights)) if len(weights) > 1 else 1.0
            expected_risk = portfolio_volatility * diversification_benefit
            
            # シャープレシオの計算
            sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
            
            return expected_return, expected_risk, sharpe_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating portfolio metrics: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """分散化比率の計算"""
        if not weights:
            return 0.0
        
        # 簡単な分散化指標（1 - ハーフィンダール指数）
        hhi = sum(weight ** 2 for weight in weights.values())
        return 1.0 - hhi

    def _calculate_allocation_confidence(self, 
                                       weights: Dict[str, float],
                                       strategy_scores: Dict[str, StrategyScore],
                                       constraint_violations: List[str]) -> float:
        """配分信頼度の計算"""
        if not weights:
            return 0.0
        
        # スコア信頼度の重み付き平均
        score_confidence = sum(
            weight * strategy_scores[name].confidence
            for name, weight in weights.items()
            if name in strategy_scores
        )
        
        # 制約違反によるペナルティ
        violation_penalty = len(constraint_violations) * 0.1
        
        # 分散化ボーナス
        diversification_bonus = min(0.2, len(weights) * 0.05)
        
        confidence = max(0.0, min(1.0, score_confidence - violation_penalty + diversification_bonus))
        return confidence

    def _generate_allocation_reason(self, config: WeightAllocationConfig, 
                                  weights: Dict[str, float]) -> str:
        """配分理由の生成"""
        method_name = config.method.value.replace('_', ' ').title()
        num_strategies = len(weights)
        
        if num_strategies == 0:
            return "No valid strategies found"
        
        top_strategy = max(weights.items(), key=lambda x: x[1])[0] if weights else "None"
        
        return (f"{method_name} allocation with {num_strategies} strategies. "
                f"Top weighted strategy: {top_strategy}")

    def _create_empty_result(self, reason: str) -> AllocationResult:
        """空の結果の作成"""
        return AllocationResult(
            strategy_weights={},
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            diversification_ratio=0.0,
            constraint_violations=[reason],
            allocation_reason=reason,
            confidence_level=0.0
        )

    def get_allocation_history(self, limit: Optional[int] = None) -> List[AllocationResult]:
        """配分履歴の取得"""
        if limit:
            return self._allocation_history[-limit:]
        return self._allocation_history.copy()

    def clear_cache(self):
        """キャッシュのクリア"""
        self._weight_cache.clear()
        logger.info("Portfolio weight cache cleared")

    def save_weights_to_file(self, result: AllocationResult, 
                           filepath: Optional[str] = None):
        """重み結果をファイルに保存"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.base_dir / f"portfolio_weights_{timestamp}.json"
        
        try:
            output_data = {
                "timestamp": result.timestamp.isoformat(),
                "strategy_weights": result.strategy_weights,
                "expected_return": result.expected_return,
                "expected_risk": result.expected_risk,
                "sharpe_ratio": result.sharpe_ratio,
                "diversification_ratio": result.diversification_ratio,
                "constraint_violations": result.constraint_violations,
                "allocation_reason": result.allocation_reason,
                "confidence_level": result.confidence_level,
                "metadata": result.metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Portfolio weights saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving weights to file: {e}")

if __name__ == "__main__":
    # 簡単なテスト
    calculator = PortfolioWeightCalculator()
    print("PortfolioWeightCalculator initialized successfully")
