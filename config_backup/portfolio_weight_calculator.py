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
    """ポートフォリオ制約設定（3-2-2機能拡張）"""
    max_individual_weight: float = 0.4           # 個別戦略最大重み
    min_individual_weight: float = 0.05          # 個別戦略最小重み（従来デフォルト）
    max_strategies: int = 5                      # 最大戦略数
    min_strategies: int = 2                      # 最小戦略数
    max_correlation_threshold: float = 0.8       # 最大相関閾値
    min_score_threshold: float = 0.3             # 最小スコア閾値
    max_turnover: float = 0.2                    # 最大ターンオーバー率
    risk_budget: float = 0.15                    # リスクバジェット
    leverage_limit: float = 1.0                  # レバレッジ制限
    concentration_limit: float = 0.6             # 集中度制限（上位3戦略の合計重み）
    
    # 3-2-2: 階層的最小重み設定
    enable_hierarchical_minimum_weights: bool = True    # 階層的最小重み機能有効化
    default_category_min_weight: float = 0.03           # カテゴリーデフォルト最小重み
    portfolio_min_weight_sum: float = 0.8               # ポートフォリオ最小重み合計上限
    weight_adjustment_method: str = "proportional"      # 重み調整手法（proportional/equal/score_weighted）
    enable_conditional_exclusion: bool = True           # 条件付き除外機能
    exclusion_score_threshold: float = 0.2              # 除外スコア閾値

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

# 3-2-2 機能: 階層的最小重み設定クラス
class WeightAdjustmentMethod(Enum):
    """重み調整手法"""
    PROPORTIONAL = "proportional"      # 比例再配分
    EQUAL = "equal"                    # 等量再配分
    SCORE_WEIGHTED = "score_weighted"  # スコア重み付け再配分

class MinimumWeightLevel(Enum):
    """最小重み設定階層"""
    STRATEGY_SPECIFIC = "strategy_specific"  # 戦略固有
    CATEGORY = "category"                    # カテゴリー別
    PORTFOLIO_DEFAULT = "portfolio_default"  # ポートフォリオデフォルト

@dataclass
class MinimumWeightRule:
    """
    3-2-2: 最小重み設定ルール
    階層的な最小資金割合設定
    """
    strategy_name: str
    min_weight: float
    level: MinimumWeightLevel
    category: Optional[str] = None
    is_conditional: bool = False
    conditions: Dict[str, float] = field(default_factory=dict)
    exclusion_threshold: Optional[float] = None  # この値以下の場合は除外
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """バリデーション"""
        if not (0.0 <= self.min_weight <= 1.0):
            raise ValueError(f"min_weight must be between 0.0 and 1.0, got {self.min_weight}")
        
        if self.exclusion_threshold is not None:
            if not (0.0 <= self.exclusion_threshold <= self.min_weight):
                raise ValueError(f"exclusion_threshold must be between 0.0 and min_weight")

@dataclass 
class WeightAdjustmentResult:
    """重み調整結果"""
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    excluded_strategies: List[str]
    applied_rules: List[MinimumWeightRule]
    adjustment_method: WeightAdjustmentMethod
    total_adjustment: float
    constraint_violations: List[str]
    success: bool
    reason: str
    metadata: Dict[str, str] = field(default_factory=dict)

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
        
        # 3-2-2: 最小重み管理システム
        self.minimum_weight_manager = MinimumWeightManager(str(self.base_dir / "minimum_weights"))
        self.weight_adjustment_engine = WeightAdjustmentEngine(self.minimum_weight_manager)
        
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
        
        logger.info("PortfolioWeightCalculator initialized with 3-2-2 functionality")

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
            
            # 3-2-2: 階層的最小重み調整の適用
            constraint_violations = []
            adjustment_metadata = {}
            
            if config.constraints.enable_hierarchical_minimum_weights:
                strategy_categories = self._get_strategy_categories(strategy_scores)
                weight_adjustment_result = self.weight_adjustment_engine.adjust_weights(
                    strategy_weights, strategy_scores, config.constraints, strategy_categories
                )
                
                # 調整結果の適用
                strategy_weights = weight_adjustment_result.adjusted_weights
                constraint_violations.extend(weight_adjustment_result.constraint_violations)
                
                # 調整情報をメタデータに追加
                adjustment_metadata = {
                    "hierarchical_adjustment_applied": True,
                    "total_adjustment": weight_adjustment_result.total_adjustment,
                    "excluded_strategies": weight_adjustment_result.excluded_strategies,
                    "applied_minimum_weight_rules": len(weight_adjustment_result.applied_rules),
                    "adjustment_method": weight_adjustment_result.adjustment_method.value,
                    "adjustment_success": weight_adjustment_result.success,
                    "adjustment_reason": weight_adjustment_result.reason
                }
            else:
                adjustment_metadata = {"hierarchical_adjustment_applied": False}
            
            # 従来の制約チェック
            additional_violations = self._check_constraints(strategy_weights, config.constraints)
            constraint_violations.extend(additional_violations)
            
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

    def _get_strategy_categories(self, strategy_scores: Dict[str, StrategyScore]) -> Dict[str, str]:
        """
        戦略カテゴリーの取得
        戦略名から推定、または設定ファイルから読み込み
        """
        categories = {}
        
        # 戦略名からカテゴリーを推定
        category_mapping = {
            'trend_following': ['trend', 'momentum', 'moving_average'],
            'mean_reversion': ['mean_revert', 'bollinger', 'rsi'],
            'momentum': ['momentum', 'macd', 'stochastic'],
            'volatility': ['volatility', 'vix', 'atr'],
            'breakout': ['breakout', 'channel', 'support_resistance']
        }
        
        for strategy_name in strategy_scores.keys():
            strategy_lower = strategy_name.lower()
            assigned_category = "general"  # デフォルトカテゴリー
            
            for category, keywords in category_mapping.items():
                if any(keyword in strategy_lower for keyword in keywords):
                    assigned_category = category
                    break
            
            categories[strategy_name] = assigned_category
        
        return categories
    
    def _build_market_context(self, market_data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """市場コンテキストの構築"""
        context = {}
        
        if market_data is not None and not market_data.empty:
            try:
                # 基本統計の計算
                if 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    context['volatility'] = returns.std() * np.sqrt(252)  # 年率ボラティリティ
                    context['mean_return'] = returns.mean() * 252  # 年率リターン
                    context['current_price'] = market_data['close'].iloc[-1]
                    
                    # トレンド指標
                    if len(market_data) >= 20:
                        sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                        context['trend'] = 'up' if context['current_price'] > sma_20 else 'down'
                
            except Exception as e:
                logger.warning(f"Error building market context: {e}")
        
        # デフォルト値
        context.setdefault('volatility', 0.2)
        context.setdefault('mean_return', 0.1)
        context.setdefault('trend', 'neutral')
        context['ticker'] = ticker
        
        return context
    
    def _check_constraints(self, weights: Dict[str, float], constraints: PortfolioConstraints) -> List[str]:
        """制約チェック"""
        violations = []
        
        if not weights:
            return violations
        
        # 個別戦略最大重みチェック
        for name, weight in weights.items():
            if weight > constraints.max_individual_weight:
                violations.append(f"Strategy {name} weight {weight:.3f} exceeds max {constraints.max_individual_weight:.3f}")
        
        # 戦略数チェック
        num_strategies = len(weights)
        if num_strategies > constraints.max_strategies:
            violations.append(f"Number of strategies {num_strategies} exceeds max {constraints.max_strategies}")
        elif num_strategies < constraints.min_strategies:
            violations.append(f"Number of strategies {num_strategies} below min {constraints.min_strategies}")
        
        # 集中度チェック
        if len(weights) >= 3:
            sorted_weights = sorted(weights.values(), reverse=True)
            top_3_concentration = sum(sorted_weights[:3])
            if top_3_concentration > constraints.concentration_limit:
                violations.append(f"Top 3 concentration {top_3_concentration:.3f} exceeds limit {constraints.concentration_limit:.3f}")
        
        return violations

    def _calculate_portfolio_metrics(self, 
                                   weights: Dict[str, float],
                                   strategy_scores: Dict[str, StrategyScore],
                                   market_context: Dict[str, Any]) -> Tuple[float, float, float]:
        """ポートフォリオ指標の計算"""
        try:
            if not weights:
                return 0.0, 0.0, 0.0
            
            # 期待リターンの計算（スコアベース）
            expected_return = sum(
                weight * strategy_scores[name].total_score * 0.1  # スコアから期待リターンに変換
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

    # 3-2-2: PortfolioWeightCalculatorクラス用の管理メソッド
    def add_strategy_minimum_weight(self, strategy_name: str, min_weight: float, exclusion_threshold: Optional[float] = None) -> bool:
        """戦略固有の最小重み設定を追加"""
        return self.weight_adjustment_engine.add_strategy_minimum_weight(strategy_name, min_weight, exclusion_threshold)
    
    def add_category_minimum_weight(self, category: str, min_weight: float) -> bool:
        """カテゴリー別最小重み設定を追加"""
        return self.weight_adjustment_engine.add_category_minimum_weight(category, min_weight)
    
    def set_default_minimum_weight(self, min_weight: float) -> bool:
        """デフォルト最小重み設定"""
        return self.weight_adjustment_engine.set_default_minimum_weight(min_weight)
    
    def get_minimum_weight_summary(self) -> Dict[str, Any]:
        """最小重み設定の要約を取得"""
        return self.weight_adjustment_engine.get_minimum_weight_rules()
    
    def enable_hierarchical_minimum_weights(self, enable: bool = True):
        """階層的最小重み機能の有効/無効を切り替え"""
        self.config.setdefault("default_constraints", {})["enable_hierarchical_minimum_weights"] = enable
        logger.info(f"Hierarchical minimum weights {'enabled' if enable else 'disabled'}")
    
    def set_weight_adjustment_method(self, method: str):
        """重み調整手法の設定"""
        valid_methods = [m.value for m in WeightAdjustmentMethod]
        if method not in valid_methods:
            raise ValueError(f"Invalid adjustment method. Must be one of: {valid_methods}")
        
        self.config.setdefault("default_constraints", {})["weight_adjustment_method"] = method
        logger.info(f"Weight adjustment method set to: {method}")
    
    def create_minimum_weight_config_template(self) -> Dict[str, Any]:
        """最小重み設定のテンプレート設定を作成"""
        return {
            "strategy_rules": [
                {
                    "strategy_name": "example_trend_following",
                    "min_weight": 0.08,
                    "level": "strategy_specific",
                    "category": "trend_following", 
                    "exclusion_threshold": 0.3,
                    "description": "トレンドフォロー戦略の固有設定"
                }
            ],
            "category_rules": [
                {
                    "category": "trend_following",
                    "min_weight": 0.06,
                    "level": "category",
                    "description": "トレンドフォロー系戦略のカテゴリー設定"
                },
                {
                    "category": "mean_reversion", 
                    "min_weight": 0.04,
                    "level": "category",
                    "description": "平均回帰系戦略のカテゴリー設定"
                }
            ],
            "default_rule": {
                "min_weight": 0.03,
                "level": "portfolio_default",
                "description": "ポートフォリオデフォルト設定"
            },
            "constraints": {
                "enable_hierarchical_minimum_weights": True,
                "enable_conditional_exclusion": True,
                "weight_adjustment_method": "proportional",
                "portfolio_min_weight_sum": 0.8,
                "exclusion_score_threshold": 0.2
            }
        }

class MinimumWeightManager:
    """
    3-2-2: 階層的最小重み管理システム
    戦略固有 > カテゴリー > ポートフォリオデフォルトの順で最小重み設定を管理
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("config/minimum_weights")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 階層的ルール管理
        self.strategy_rules: Dict[str, MinimumWeightRule] = {}
        self.category_rules: Dict[str, MinimumWeightRule] = {}
        self.default_rule: Optional[MinimumWeightRule] = None
        
        # 設定ファイル
        self.rules_file = self.base_dir / "minimum_weight_rules.json"
        
        # ルールの読み込み
        self._load_rules()
        
        logger.info("MinimumWeightManager initialized")
    
    def add_strategy_rule(self, rule: MinimumWeightRule) -> bool:
        """戦略固有ルールの追加"""
        try:
            if rule.level != MinimumWeightLevel.STRATEGY_SPECIFIC:
                raise ValueError("Rule level must be STRATEGY_SPECIFIC")
            
            self.strategy_rules[rule.strategy_name] = rule
            self._save_rules()
            logger.info(f"Added strategy rule for {rule.strategy_name}: min_weight={rule.min_weight}")
            return True
        except Exception as e:
            logger.error(f"Failed to add strategy rule: {e}")
            return False
    
    def add_category_rule(self, category: str, min_weight: float) -> bool:
        """カテゴリールールの追加"""
        try:
            rule = MinimumWeightRule(
                strategy_name=f"category_{category}",
                min_weight=min_weight,
                level=MinimumWeightLevel.CATEGORY,
                category=category
            )
            self.category_rules[category] = rule
            self._save_rules()
            logger.info(f"Added category rule for {category}: min_weight={min_weight}")
            return True
        except Exception as e:
            logger.error(f"Failed to add category rule: {e}")
            return False
    
    def set_default_rule(self, min_weight: float) -> bool:
        """デフォルトルールの設定"""
        try:
            self.default_rule = MinimumWeightRule(
                strategy_name="default",
                min_weight=min_weight,
                level=MinimumWeightLevel.PORTFOLIO_DEFAULT
            )
            self._save_rules()
            logger.info(f"Set default rule: min_weight={min_weight}")
            return True
        except Exception as e:
            logger.error(f"Failed to set default rule: {e}")
            return False
    
    def get_minimum_weight(self, strategy_name: str, category: Optional[str] = None) -> float:
        """戦略の最小重みを階層的に取得"""
        # 1. 戦略固有ルール
        if strategy_name in self.strategy_rules:
            return self.strategy_rules[strategy_name].min_weight
        
        # 2. カテゴリールール
        if category and category in self.category_rules:
            return self.category_rules[category].min_weight
        
        # 3. デフォルトルール
        if self.default_rule:
            return self.default_rule.min_weight
        
        # 4. システムデフォルト
        return 0.05
    
    def get_exclusion_threshold(self, strategy_name: str, category: Optional[str] = None) -> Optional[float]:
        """除外閾値の取得"""
        # 戦略固有ルール
        if strategy_name in self.strategy_rules:
            rule = self.strategy_rules[strategy_name]
            return rule.exclusion_threshold
        
        # カテゴリールール
        if category and category in self.category_rules:
            rule = self.category_rules[category]
            return rule.exclusion_threshold
        
        return None
    
    def _load_rules(self):
        """ルールファイルの読み込み"""
        if not self.rules_file.exists():
            return
        
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 戦略固有ルール
            for rule_data in data.get('strategy_rules', []):
                rule = self._dict_to_rule(rule_data)
                self.strategy_rules[rule.strategy_name] = rule
            
            # カテゴリールール
            for rule_data in data.get('category_rules', []):
                rule = self._dict_to_rule(rule_data)
                if rule.category:
                    self.category_rules[rule.category] = rule
            
            # デフォルトルール
            default_data = data.get('default_rule')
            if default_data:
                self.default_rule = self._dict_to_rule(default_data)
                
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
    
    def _save_rules(self):
        """ルールファイルの保存"""
        try:
            data = {
                'strategy_rules': [self._rule_to_dict(rule) for rule in self.strategy_rules.values()],
                'category_rules': [self._rule_to_dict(rule) for rule in self.category_rules.values()],
                'default_rule': self._rule_to_dict(self.default_rule) if self.default_rule else None
            }
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")
    
    def _dict_to_rule(self, data: Dict[str, Any]) -> MinimumWeightRule:
        """辞書からルールオブジェクトに変換"""
        return MinimumWeightRule(
            strategy_name=str(data['strategy_name']),
            min_weight=float(data['min_weight']),
            level=MinimumWeightLevel(data['level']),
            category=data.get('category'),
            is_conditional=bool(data.get('is_conditional', False)),
            conditions=dict(data.get('conditions', {})),
            exclusion_threshold=data.get('exclusion_threshold')
        )
    
    def _rule_to_dict(self, rule: MinimumWeightRule) -> Dict[str, Any]:
        """ルールオブジェクトから辞書に変換"""
        return {
            'strategy_name': rule.strategy_name,
            'min_weight': rule.min_weight,
            'level': rule.level.value,
            'category': rule.category,
            'is_conditional': rule.is_conditional,
            'conditions': rule.conditions,
            'exclusion_threshold': rule.exclusion_threshold,
            'created_at': rule.created_at.isoformat()
        }


class WeightAdjustmentEngine:
    """
    3-2-2: 重み調整エンジン
    最小重み制約を満たすように戦略重みを調整
    """
    
    def __init__(self, minimum_weight_manager: MinimumWeightManager):
        self.min_weight_manager = minimum_weight_manager
        
    def adjust_weights(self, 
                      original_weights: Dict[str, float],
                      strategy_scores: Dict[str, StrategyScore],
                      constraints: PortfolioConstraints,
                      strategy_categories: Optional[Dict[str, str]] = None) -> WeightAdjustmentResult:
        """
        重み調整のメインメソッド
        
        Parameters:
            original_weights: 元の重み
            strategy_scores: 戦略スコア
            constraints: ポートフォリオ制約
            strategy_categories: 戦略カテゴリー分類
        
        Returns:
            WeightAdjustmentResult: 調整結果
        """
        try:
            if not original_weights:
                return self._create_empty_result("No original weights provided")
            
            # 3-2-2機能が無効化されている場合は元の重みをそのまま返す
            if not constraints.enable_hierarchical_minimum_weights:
                return WeightAdjustmentResult(
                    original_weights=original_weights,
                    adjusted_weights=original_weights,
                    excluded_strategies=[],
                    applied_rules=[],
                    adjustment_method=WeightAdjustmentMethod.PROPORTIONAL,
                    total_adjustment=0.0,
                    constraint_violations=[],
                    success=True,
                    reason="Hierarchical minimum weights disabled"
                )
            
            strategy_categories = strategy_categories or {}
            adjusted_weights = original_weights.copy()
            excluded_strategies = []
            applied_rules = []
            constraint_violations = []
            
            # 1. 除外処理
            if constraints.enable_conditional_exclusion:
                adjusted_weights, excluded_strategies = self._apply_exclusion_logic(
                    adjusted_weights, strategy_scores, strategy_categories, constraints
                )
            
            # 2. 最小重み制約の適用
            adjusted_weights, applied_rules, violations = self._apply_minimum_weight_constraints(
                adjusted_weights, strategy_scores, strategy_categories, constraints
            )
            constraint_violations.extend(violations)
            
            # 3. 重みの正規化
            adjusted_weights = self._normalize_weights(adjusted_weights)
            
            # 4. 最終検証
            final_violations = self._validate_final_weights(adjusted_weights, constraints)
            constraint_violations.extend(final_violations)
            
            # 調整量の計算
            total_adjustment = sum(
                abs(adjusted_weights.get(name, 0) - original_weights.get(name, 0))
                for name in set(list(adjusted_weights.keys()) + list(original_weights.keys()))
            )
            
            # 調整手法の取得
            adjustment_method_str = constraints.weight_adjustment_method
            adjustment_method = WeightAdjustmentMethod(adjustment_method_str)
            
            return WeightAdjustmentResult(
                original_weights=original_weights,
                adjusted_weights=adjusted_weights,
                excluded_strategies=excluded_strategies,
                applied_rules=applied_rules,
                adjustment_method=adjustment_method,
                total_adjustment=total_adjustment,
                constraint_violations=constraint_violations,
                success=len(constraint_violations) == 0,
                reason=self._generate_adjustment_reason(adjusted_weights, applied_rules, excluded_strategies)
            )
            
        except Exception as e:
            logger.error(f"Weight adjustment failed: {e}")
            return self._create_empty_result(f"Adjustment error: {str(e)}")
    
    def _apply_exclusion_logic(self, 
                              weights: Dict[str, float],
                              strategy_scores: Dict[str, StrategyScore],
                              strategy_categories: Dict[str, str],
                              constraints: PortfolioConstraints) -> Tuple[Dict[str, float], List[str]]:
        """条件付き除外の適用"""
        excluded_strategies = []
        remaining_weights = {}
        
        for strategy_name, weight in weights.items():
            # スコア情報の取得
            score = strategy_scores.get(strategy_name)
            if not score:
                excluded_strategies.append(strategy_name)
                continue
            
            # 除外閾値の取得
            category = strategy_categories.get(strategy_name)
            exclusion_threshold = self.min_weight_manager.get_exclusion_threshold(strategy_name, category)
            
            # 除外判定
            if exclusion_threshold is not None and score.total_score <= exclusion_threshold:
                excluded_strategies.append(strategy_name)
                logger.info(f"Strategy {strategy_name} excluded: score {score.total_score:.3f} <= threshold {exclusion_threshold:.3f}")
            else:
                remaining_weights[strategy_name] = weight
        
        return remaining_weights, excluded_strategies
    
    def _apply_minimum_weight_constraints(self, 
                                        weights: Dict[str, float],
                                        strategy_scores: Dict[str, StrategyScore],
                                        strategy_categories: Dict[str, str],
                                        constraints: PortfolioConstraints) -> Tuple[Dict[str, float], List[MinimumWeightRule], List[str]]:
        """最小重み制約の適用"""
        adjusted_weights = weights.copy()
        applied_rules = []
        violations = []
        
        # 各戦略の最小重み要件を取得
        min_weight_requirements = {}
        for strategy_name in weights.keys():
            category = strategy_categories.get(strategy_name)
            min_weight = self.min_weight_manager.get_minimum_weight(strategy_name, category)
            min_weight_requirements[strategy_name] = min_weight
        
        # 制約違反のチェックと調整
        total_min_weight_needed = 0
        for strategy_name, current_weight in weights.items():
            required_min_weight = min_weight_requirements[strategy_name]
            
            if current_weight < required_min_weight:
                adjustment_needed = required_min_weight - current_weight
                adjusted_weights[strategy_name] = required_min_weight
                total_min_weight_needed += adjustment_needed
                
                # 適用ルールの記録
                rule = MinimumWeightRule(
                    strategy_name=strategy_name,
                    min_weight=required_min_weight,
                    level=MinimumWeightLevel.STRATEGY_SPECIFIC
                )
                applied_rules.append(rule)
        
        # 全体制約のチェック
        total_required_min_weight = sum(min_weight_requirements.values())
        if total_required_min_weight > constraints.portfolio_min_weight_sum:
            violations.append(f"Total minimum weight requirements ({total_required_min_weight:.3f}) exceed portfolio limit ({constraints.portfolio_min_weight_sum:.3f})")
        
        # 重み再配分が必要な場合
        if total_min_weight_needed > 0:
            adjusted_weights = self._redistribute_weights(
                adjusted_weights, min_weight_requirements, constraints
            )
        
        return adjusted_weights, applied_rules, violations
    
    def _redistribute_weights(self, 
                            weights: Dict[str, float],
                            min_weight_requirements: Dict[str, float],
                            constraints: PortfolioConstraints) -> Dict[str, float]:
        """重みの再配分"""
        adjustment_method_str = constraints.weight_adjustment_method
        
        # 最小重み要件を満たしていない戦略の調整後重み
        adjusted_weights = {}
        excess_weight = 0
        
        for strategy_name, current_weight in weights.items():
            min_weight = min_weight_requirements[strategy_name]
            if current_weight < min_weight:
                adjusted_weights[strategy_name] = min_weight
            else:
                adjusted_weights[strategy_name] = current_weight
                excess_weight += max(0, current_weight - min_weight)
        
        # 重み合計が1を超える場合の調整
        total_weight = sum(adjusted_weights.values())
        if total_weight > 1.0:
            excess_amount = total_weight - 1.0
            
            # 調整可能な戦略（最小重み以上の重みを持つ戦略）
            adjustable_strategies = {
                name: weight for name, weight in adjusted_weights.items()
                if weight > min_weight_requirements[name]
            }
            
            if adjustable_strategies:
                if adjustment_method_str == "proportional":
                    # 比例調整
                    total_adjustable = sum(adjustable_strategies.values())
                    for strategy_name in adjustable_strategies:
                        reduction_ratio = excess_amount / total_adjustable
                        current_weight = adjusted_weights[strategy_name]
                        min_weight = min_weight_requirements[strategy_name]
                        max_reduction = current_weight - min_weight
                        actual_reduction = min(reduction_ratio * current_weight, max_reduction)
                        adjusted_weights[strategy_name] = current_weight - actual_reduction
                
                elif adjustment_method_str == "equal":
                    # 等量調整
                    per_strategy_reduction = excess_amount / len(adjustable_strategies)
                    for strategy_name in adjustable_strategies:
                        current_weight = adjusted_weights[strategy_name]
                        min_weight = min_weight_requirements[strategy_name]
                        max_reduction = current_weight - min_weight
                        actual_reduction = min(per_strategy_reduction, max_reduction)
                        adjusted_weights[strategy_name] = current_weight - actual_reduction
        
        return adjusted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重みの正規化"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return weights
        
        return {name: weight / total_weight for name, weight in weights.items()}
    
    def _validate_final_weights(self, weights: Dict[str, float], constraints: PortfolioConstraints) -> List[str]:
        """最終重みの検証"""
        violations = []
        
        # 重み合計チェック
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            violations.append(f"Total weight {total_weight:.3f} != 1.0")
        
        # 個別制約チェック
        for name, weight in weights.items():
            if weight > constraints.max_individual_weight:
                violations.append(f"Strategy {name} weight {weight:.3f} exceeds max {constraints.max_individual_weight:.3f}")
        
        return violations
    
    def _generate_adjustment_reason(self, weights: Dict[str, float], applied_rules: List[MinimumWeightRule], excluded_strategies: List[str]) -> str:
        """調整理由の生成"""
        reasons = []
        
        if applied_rules:
            reasons.append(f"Applied {len(applied_rules)} minimum weight rules")
        
        if excluded_strategies:
            reasons.append(f"Excluded {len(excluded_strategies)} strategies below threshold")
        
        if not reasons:
            reasons.append("No adjustments needed")
        
        return "; ".join(reasons)
    
    def _create_empty_result(self, reason: str) -> WeightAdjustmentResult:
        """空の結果を作成"""
        return WeightAdjustmentResult(
            original_weights={},
            adjusted_weights={},
            excluded_strategies=[],
            applied_rules=[],
            adjustment_method=WeightAdjustmentMethod.PROPORTIONAL,
            total_adjustment=0.0,
            constraint_violations=[],
            success=False,
            reason=reason
        )

    # 3-2-2: 階層的最小重み設定のための管理メソッド
    def add_strategy_minimum_weight(self, strategy_name: str, min_weight: float, exclusion_threshold: Optional[float] = None) -> bool:
        """戦略固有の最小重み設定を追加"""
        try:
            rule = MinimumWeightRule(
                strategy_name=strategy_name,
                min_weight=min_weight,
                level=MinimumWeightLevel.STRATEGY_SPECIFIC,
                exclusion_threshold=exclusion_threshold
            )
            return self.min_weight_manager.add_strategy_rule(rule)
        except Exception as e:
            logger.error(f"Failed to add strategy minimum weight: {e}")
            return False
    
    def add_category_minimum_weight(self, category: str, min_weight: float) -> bool:
        """カテゴリー別最小重み設定を追加"""
        return self.min_weight_manager.add_category_rule(category, min_weight)
    
    def set_default_minimum_weight(self, min_weight: float) -> bool:
        """デフォルト最小重み設定"""
        return self.min_weight_manager.set_default_rule(min_weight)
    
    def get_minimum_weight_rules(self) -> Dict[str, Any]:
        """設定されている最小重みルールの取得"""
        return {
            "strategy_rules": {name: rule.__dict__ for name, rule in self.min_weight_manager.strategy_rules.items()},
            "category_rules": {name: rule.__dict__ for name, rule in self.min_weight_manager.category_rules.items()},
            "default_rule": self.min_weight_manager.default_rule.__dict__ if self.min_weight_manager.default_rule else None
        }
    
    def enable_hierarchical_minimum_weights(self, enable: bool = True):
        """階層的最小重み機能の有効/無効切り替え"""
        # デフォルト設定の更新（実際の設定は計算時のConfigで指定）
        logger.info(f"Hierarchical minimum weights feature {'enabled' if enable else 'disabled'}")
    
    def create_enhanced_allocation_config(self, 
                                        base_config: Optional[WeightAllocationConfig] = None,
                                        enable_hierarchical_weights: bool = True,
                                        weight_adjustment_method: str = "proportional",
                                        enable_conditional_exclusion: bool = True) -> WeightAllocationConfig:
        """3-2-2機能を含む拡張配分設定の作成"""
        if base_config:
            config = base_config
        else:
            config = self._create_default_config()
        
        # 3-2-2機能の設定
        config.constraints.enable_hierarchical_minimum_weights = enable_hierarchical_weights
        config.constraints.weight_adjustment_method = weight_adjustment_method
        config.constraints.enable_conditional_exclusion = enable_conditional_exclusion
        
        return config
