"""
Module: Position Size Adjuster
File: position_size_adjuster.py
Description: 
  3-3-2「各戦略のポジションサイズ調整機能」
  既存のportfolio_weight_calculatorと統合した動的ポジションサイズ調整システム
  戦略スコア + リスク調整 + 市場環境調整のハイブリッド方式

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Dependencies:
  - config.portfolio_weight_calculator
  - config.signal_integrator
  - config.strategy_scoring_model
  - config.risk_management
  - indicators.unified_trend_detector
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
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, AllocationResult, WeightAllocationConfig,
        PortfolioConstraints, AllocationMethod
    )
    from config.strategy_scoring_model import StrategyScore, StrategyScoreManager
    from config.signal_integrator import SignalIntegrator, StrategySignal, SignalType
    from config.risk_management import RiskManagement
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error in position_size_adjuster: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class PositionSizeMethod(Enum):
    """ポジションサイズ計算方法"""
    FIXED_PERCENTAGE = "fixed_percentage"           # 固定割合
    SCORE_BASED = "score_based"                    # スコアベース
    RISK_PARITY = "risk_parity"                    # リスクパリティ
    KELLY_CRITERION = "kelly_criterion"            # ケリー基準
    HYBRID_ADAPTIVE = "hybrid_adaptive"            # ハイブリッド適応型（推奨）

class RiskAdjustmentType(Enum):
    """リスク調整タイプ"""
    VOLATILITY_BASED = "volatility_based"          # ボラティリティベース
    ATR_BASED = "atr_based"                       # ATRベース  
    DRAWDOWN_BASED = "drawdown_based"             # ドローダウンベース
    VAR_BASED = "var_based"                       # VaRベース
    COMPOSITE = "composite"                        # 複合型

class MarketRegime(Enum):
    """市場環境"""
    TRENDING_UP = "trending_up"                    # 上昇トレンド
    TRENDING_DOWN = "trending_down"                # 下降トレンド
    RANGE_BOUND = "range_bound"                    # レンジ相場
    HIGH_VOLATILITY = "high_volatility"            # 高ボラティリティ
    LOW_VOLATILITY = "low_volatility"              # 低ボラティリティ
    CRISIS = "crisis"                              # 危機モード

@dataclass
class PositionSizingConfig:
    """ポジションサイジング設定"""
    # 基本設定
    base_position_size: float = 0.02               # ベースポジションサイズ（2%）
    max_position_size: float = 0.10                # 最大ポジションサイズ（10%）
    min_position_size: float = 0.005               # 最小ポジションサイズ（0.5%）
    
    # 計算方式
    sizing_method: PositionSizeMethod = PositionSizeMethod.HYBRID_ADAPTIVE
    risk_adjustment_type: RiskAdjustmentType = RiskAdjustmentType.COMPOSITE
    
    # スコア重み付け設定
    score_weight: float = 0.4                      # 戦略スコアの重み
    risk_weight: float = 0.3                       # リスク調整の重み
    market_weight: float = 0.2                     # 市場環境の重み
    trend_confidence_weight: float = 0.1           # トレンド信頼度の重み
    
    # リスク管理設定
    max_portfolio_risk: float = 0.20               # 最大ポートフォリオリスク（20%）
    risk_free_rate: float = 0.02                   # リスクフリーレート（2%）
    volatility_lookback: int = 252                 # ボラティリティ計算期間（1年）
    
    # 動的調整設定
    enable_dynamic_adjustment: bool = True          # 動的調整有効化
    adjustment_frequency: str = "daily"            # 調整頻度
    regime_sensitivity: float = 0.7                # 市場環境感応度
    
    # 制約設定
    enable_correlation_adjustment: bool = True     # 相関調整有効化
    max_sector_concentration: float = 0.40         # セクター集中度上限
    enable_liquidity_adjustment: bool = True       # 流動性調整有効化

@dataclass
class PositionSizeResult:
    """ポジションサイズ計算結果"""
    strategy_name: str
    base_size: float                               # ベースサイズ（%）
    adjusted_size: float                          # 調整後サイズ（%）
    absolute_amount: Optional[float] = None        # 絶対金額
    share_count: Optional[int] = None             # 株数
    
    # 調整要因
    score_multiplier: float = 1.0                 # スコア乗数
    risk_multiplier: float = 1.0                  # リスク乗数
    market_multiplier: float = 1.0                # 市場環境乗数
    trend_confidence_multiplier: float = 1.0      # トレンド信頼度乗数
    
    # メタデータ
    market_regime: MarketRegime = MarketRegime.RANGE_BOUND
    volatility_percentile: float = 0.5            # ボラティリティパーセンタイル
    confidence_level: float = 0.5                 # 計算信頼度
    constraints_applied: List[str] = field(default_factory=list)
    calculation_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioPositionSizing:
    """ポートフォリオレベルのポジションサイジング結果"""
    total_portfolio_value: float
    position_results: Dict[str, PositionSizeResult]
    total_allocated_percentage: float
    remaining_cash_percentage: float
    portfolio_risk_estimate: float
    diversification_score: float
    regime_analysis: Dict[str, Any]
    constraint_violations: List[str] = field(default_factory=list)
    rebalancing_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePositionSizer(ABC):
    """ポジションサイザー基底クラス"""
    
    @abstractmethod
    def calculate_position_size(self, 
                              strategy_name: str,
                              strategy_score: StrategyScore,
                              market_data: pd.DataFrame,
                              portfolio_context: Dict[str, Any],
                              config: PositionSizingConfig) -> PositionSizeResult:
        """ポジションサイズ計算の抽象メソッド"""
        pass

class HybridAdaptivePositionSizer(BasePositionSizer):
    """
    ハイブリッド適応型ポジションサイザー（推奨実装）
    戦略スコア + リスク調整 + 市場環境調整を統合
    """
    
    def __init__(self):
        self.volatility_cache = {}
        
    def calculate_position_size(self, 
                              strategy_name: str,
                              strategy_score: StrategyScore,
                              market_data: pd.DataFrame,
                              portfolio_context: Dict[str, Any],
                              config: PositionSizingConfig) -> PositionSizeResult:
        """ハイブリッド方式でのポジションサイズ計算"""
        
        # 1. ベースサイズの設定
        base_size = config.base_position_size
        
        # 2. 戦略スコア乗数の計算
        score_multiplier = self._calculate_score_multiplier(strategy_score, config)
        
        # 3. リスク調整乗数の計算
        risk_multiplier = self._calculate_risk_multiplier(
            strategy_name, market_data, portfolio_context, config
        )
        
        # 4. 市場環境乗数の計算
        market_multiplier, market_regime = self._calculate_market_multiplier(
            market_data, portfolio_context, config
        )
        
        # 5. トレンド信頼度乗数の計算
        trend_confidence_multiplier = self._calculate_trend_confidence_multiplier(
            market_data, config
        )
        
        # 6. 複合調整の適用
        composite_multiplier = (
            score_multiplier * config.score_weight +
            risk_multiplier * config.risk_weight +
            market_multiplier * config.market_weight +
            trend_confidence_multiplier * config.trend_confidence_weight
        )
        
        # 7. 調整後サイズの計算
        adjusted_size = base_size * composite_multiplier
        
        # 8. 制約の適用
        adjusted_size, constraints_applied = self._apply_constraints(
            adjusted_size, strategy_name, portfolio_context, config
        )
        
        # 9. ボラティリティパーセンタイルの計算
        volatility_percentile = self._calculate_volatility_percentile(market_data)
        
        # 10. 信頼度の計算
        confidence_level = self._calculate_confidence_level(
            strategy_score, market_data, composite_multiplier
        )
        
        # 11. 結果の構築
        result = PositionSizeResult(
            strategy_name=strategy_name,
            base_size=base_size,
            adjusted_size=adjusted_size,
            score_multiplier=score_multiplier,
            risk_multiplier=risk_multiplier,
            market_multiplier=market_multiplier,
            trend_confidence_multiplier=trend_confidence_multiplier,
            market_regime=market_regime,
            volatility_percentile=volatility_percentile,
            confidence_level=confidence_level,
            constraints_applied=constraints_applied,
            calculation_reason=self._generate_calculation_reason(
                composite_multiplier, constraints_applied
            )
        )
        
        return result
    
    def _calculate_score_multiplier(self, strategy_score: StrategyScore, config: PositionSizingConfig) -> float:
        """戦略スコア乗数の計算"""
        # スコアを0.5-2.0の範囲にマッピング
        # 高スコア戦略により多くの資金を配分
        normalized_score = max(0.0, min(1.0, strategy_score.total_score))
        score_multiplier = 0.5 + (normalized_score * 1.5)  # 0.5 ~ 2.0
        
        # 信頼度による調整
        confidence_adjustment = 0.8 + (strategy_score.confidence * 0.4)  # 0.8 ~ 1.2
        
        return score_multiplier * confidence_adjustment
    
    def _calculate_risk_multiplier(self, 
                                 strategy_name: str,
                                 market_data: pd.DataFrame,
                                 portfolio_context: Dict[str, Any],
                                 config: PositionSizingConfig) -> float:
        """リスク調整乗数の計算"""
        
        # ボラティリティベースの調整
        volatility = self._calculate_volatility(market_data, config)
        volatility_adjustment = max(0.5, min(1.5, 1.0 / (1.0 + volatility)))
        
        # ポートフォリオレベルのリスク調整
        portfolio_risk = portfolio_context.get('current_portfolio_risk', 0.15)
        max_risk = config.max_portfolio_risk
        risk_capacity = max(0.2, (max_risk - portfolio_risk) / max_risk)
        
        # 複合リスク乗数
        risk_multiplier = (volatility_adjustment * 0.6 + risk_capacity * 0.4)
        
        return max(0.3, min(1.8, risk_multiplier))
    
    def _calculate_market_multiplier(self, 
                                   market_data: pd.DataFrame,
                                   portfolio_context: Dict[str, Any],
                                   config: PositionSizingConfig) -> Tuple[float, MarketRegime]:
        """市場環境乗数の計算"""
        
        # 市場環境の判定
        regime = self._detect_market_regime(market_data)
        
        # 環境別乗数マッピング
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.7,
            MarketRegime.RANGE_BOUND: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.CRISIS: 0.4
        }
        
        market_multiplier = regime_multipliers.get(regime, 1.0)
        
        # 感応度による調整
        sensitivity = config.regime_sensitivity
        adjusted_multiplier = 1.0 + (market_multiplier - 1.0) * sensitivity
        
        return adjusted_multiplier, regime
    
    def _calculate_trend_confidence_multiplier(self, 
                                             market_data: pd.DataFrame,
                                             config: PositionSizingConfig) -> float:
        """トレンド信頼度乗数の計算"""
        try:
            # unified_trend_detectorを使用した信頼度取得
            if market_data is not None and not market_data.empty:
                # 価格カラムの確認
                price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
                
                # UnifiedTrendDetectorの初期化
                trend_detector = UnifiedTrendDetector(
                    data=market_data, 
                    price_column=price_column,
                    strategy_name="position_sizing"
                )
                
                # トレンド検出
                trend_result = trend_detector.detect_trend()
                confidence = trend_detector.get_confidence()
                
                # 信頼度を乗数に変換（0.8 ~ 1.2）
                confidence_multiplier = 0.8 + (confidence * 0.4)
                
                return confidence_multiplier
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Error calculating trend confidence multiplier: {e}")
            return 1.0
    
    def _calculate_volatility(self, market_data: pd.DataFrame, config: PositionSizingConfig) -> float:
        """ボラティリティの計算"""
        try:
            if market_data is not None and not market_data.empty:
                # 価格カラムの確認
                price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
                
                if price_column in market_data.columns and len(market_data) >= 20:
                    returns = market_data[price_column].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ
                    return volatility
                else:
                    return 0.2  # デフォルトボラティリティ
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.2
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """市場環境の検出"""
        try:
            if market_data is None or market_data.empty:
                return MarketRegime.RANGE_BOUND
            
            # ボラティリティ分析
            volatility = self._calculate_volatility(market_data, None)
            
            # トレンド分析
            price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
            
            if price_column in market_data.columns and len(market_data) >= 50:
                close_prices = market_data[price_column]
                sma_20 = close_prices.rolling(20).mean().iloc[-1]
                sma_50 = close_prices.rolling(50).mean().iloc[-1]
                current_price = close_prices.iloc[-1]
                
                # 高ボラティリティ判定
                if volatility > 0.35:
                    return MarketRegime.HIGH_VOLATILITY
                elif volatility < 0.12:
                    return MarketRegime.LOW_VOLATILITY
                
                # トレンド判定
                if current_price > sma_20 > sma_50:
                    return MarketRegime.TRENDING_UP
                elif current_price < sma_20 < sma_50:
                    return MarketRegime.TRENDING_DOWN
                else:
                    return MarketRegime.RANGE_BOUND
            
            return MarketRegime.RANGE_BOUND
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return MarketRegime.RANGE_BOUND
    
    def _apply_constraints(self, 
                         size: float,
                         strategy_name: str,
                         portfolio_context: Dict[str, Any],
                         config: PositionSizingConfig) -> Tuple[float, List[str]]:
        """制約の適用"""
        constraints_applied = []
        original_size = size
        
        # 最小・最大制約
        if size > config.max_position_size:
            size = config.max_position_size
            constraints_applied.append(f"Max position limit applied: {config.max_position_size:.3f}")
        
        if size < config.min_position_size:
            size = config.min_position_size
            constraints_applied.append(f"Min position limit applied: {config.min_position_size:.3f}")
        
        # ポートフォリオレベル制約
        current_allocation = portfolio_context.get('total_allocated_percentage', 0.0)
        remaining_capacity = 1.0 - current_allocation
        
        if size > remaining_capacity:
            size = max(config.min_position_size, remaining_capacity)
            constraints_applied.append(f"Portfolio capacity limit applied: {remaining_capacity:.3f}")
        
        # 相関調整（簡略版）
        if config.enable_correlation_adjustment:
            similar_strategies_allocation = portfolio_context.get('similar_strategies_allocation', 0.0)
            if similar_strategies_allocation > 0.3:  # 類似戦略が30%以上
                correlation_penalty = 0.8
                size *= correlation_penalty
                constraints_applied.append("Correlation penalty applied")
        
        return size, constraints_applied
    
    def _calculate_volatility_percentile(self, market_data: pd.DataFrame) -> float:
        """ボラティリティパーセンタイルの計算"""
        try:
            if market_data is None or len(market_data) < 50:
                return 0.5
            
            # 価格カラムの確認
            price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
            
            # 過去のボラティリティ分布と比較
            returns = market_data[price_column].pct_change().dropna()
            rolling_vol = returns.rolling(20).std()
            current_vol = rolling_vol.iloc[-1]
            
            percentile = (rolling_vol <= current_vol).mean()
            return percentile
            
        except Exception as e:
            logger.warning(f"Error calculating volatility percentile: {e}")
            return 0.5
    
    def _calculate_confidence_level(self, 
                                  strategy_score: StrategyScore,
                                  market_data: pd.DataFrame,
                                  composite_multiplier: float) -> float:
        """計算信頼度の算出"""
        # 戦略スコア信頼度
        score_confidence = strategy_score.confidence
        
        # データ品質による調整
        data_quality = min(1.0, len(market_data) / 252) if market_data is not None else 0.5
        
        # 乗数の安定性による調整
        multiplier_stability = max(0.5, 1.0 - abs(composite_multiplier - 1.0))
        
        # 複合信頼度
        composite_confidence = (
            score_confidence * 0.5 +
            data_quality * 0.3 +
            multiplier_stability * 0.2
        )
        
        return max(0.0, min(1.0, composite_confidence))
    
    def _generate_calculation_reason(self, composite_multiplier: float, constraints_applied: List[str]) -> str:
        """計算理由の生成"""
        reasons = []
        
        # 乗数による理由
        if composite_multiplier > 1.1:
            reasons.append("Position increased due to favorable conditions")
        elif composite_multiplier < 0.9:
            reasons.append("Position reduced due to risk factors")
        else:
            reasons.append("Standard position sizing applied")
        
        # 制約による理由
        if constraints_applied:
            reasons.append(f"Constraints applied: {len(constraints_applied)}")
        
        return "; ".join(reasons)

class PositionSizeAdjuster:
    """
    ポジションサイズ調整器（メインクラス）
    既存のPortfolioWeightCalculatorと統合してポジションサイズを調整
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 portfolio_value: float = 1000000.0,
                 base_dir: Optional[str] = None):
        """ポジションサイズ調整器の初期化"""
        
        self.portfolio_value = portfolio_value
        self.base_dir = Path(base_dir) if base_dir else Path("config/position_sizing")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定の読み込み
        self.config = self._load_config(config_file)
        
        # 既存システムとの統合
        try:
            self.portfolio_weight_calculator = PortfolioWeightCalculator()
            self.score_manager = StrategyScoreManager()
            self.risk_manager = RiskManagement(total_assets=portfolio_value)
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")
            self.portfolio_weight_calculator = None
            self.score_manager = None
            self.risk_manager = None
        
        # ポジションサイザーの登録
        self.position_sizers = {
            PositionSizeMethod.HYBRID_ADAPTIVE: HybridAdaptivePositionSizer(),
            PositionSizeMethod.SCORE_BASED: self._create_score_based_sizer(),
            PositionSizeMethod.RISK_PARITY: self._create_risk_parity_sizer(),
            PositionSizeMethod.FIXED_PERCENTAGE: self._create_fixed_percentage_sizer()
        }
        
        # キャッシュとパフォーマンス
        self._position_cache = {}
        self._last_calculation_time = None
        self._calculation_history = []
        
        logger.info(f"PositionSizeAdjuster initialized with portfolio value: ${portfolio_value:,.2f}")

    def _load_config(self, config_file: Optional[str]) -> PositionSizingConfig:
        """設定ファイルの読み込み"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # JSON設定をPositionSizingConfigに変換
                return self._dict_to_config(config_data)
                
            except Exception as e:
                logger.warning(f"Failed to load position sizing config: {e}")
        
        # デフォルト設定
        return PositionSizingConfig()

    def _dict_to_config(self, config_data: Dict[str, Any]) -> PositionSizingConfig:
        """辞書からPositionSizingConfigに変換"""
        return PositionSizingConfig(
            base_position_size=config_data.get('base_position_size', 0.02),
            max_position_size=config_data.get('max_position_size', 0.10),
            min_position_size=config_data.get('min_position_size', 0.005),
            sizing_method=PositionSizeMethod(config_data.get('sizing_method', 'hybrid_adaptive')),
            risk_adjustment_type=RiskAdjustmentType(config_data.get('risk_adjustment_type', 'composite')),
            score_weight=config_data.get('score_weight', 0.4),
            risk_weight=config_data.get('risk_weight', 0.3),
            market_weight=config_data.get('market_weight', 0.2),
            trend_confidence_weight=config_data.get('trend_confidence_weight', 0.1),
            max_portfolio_risk=config_data.get('max_portfolio_risk', 0.20),
            enable_dynamic_adjustment=config_data.get('enable_dynamic_adjustment', True),
            regime_sensitivity=config_data.get('regime_sensitivity', 0.7)
        )

    def calculate_portfolio_position_sizes(self,
                                         ticker: str,
                                         market_data: pd.DataFrame,
                                         config: Optional[PositionSizingConfig] = None,
                                         current_positions: Optional[Dict[str, float]] = None) -> PortfolioPositionSizing:
        """
        ポートフォリオレベルのポジションサイズ計算
        
        Parameters:
            ticker: 対象銘柄
            market_data: 市場データ
            config: ポジションサイジング設定
            current_positions: 現在のポジション（リバランス用）
            
        Returns:
            PortfolioPositionSizing: ポートフォリオポジションサイジング結果
        """
        start_time = datetime.now()
        
        try:
            # デフォルト設定の適用
            if config is None:
                config = self.config
            
            # 1. ポートフォリオ重みの計算（既存システム活用）
            allocation_result = None
            if self.portfolio_weight_calculator:
                try:
                    allocation_result = self.portfolio_weight_calculator.calculate_portfolio_weights(
                        ticker=ticker,
                        market_data=market_data
                    )
                except Exception as e:
                    logger.warning(f"Portfolio weight calculation failed: {e}")
            
            # 2. 戦略スコアの取得
            strategy_scores = self._get_strategy_scores(ticker)
            
            # 3. ポートフォリオコンテキストの構築
            portfolio_context = self._build_portfolio_context(
                allocation_result, current_positions, market_data
            )
            
            # 4. 各戦略のポジションサイズ計算
            position_results = {}
            total_allocated_percentage = 0.0
            
            # ポジションサイザーの取得
            sizer = self.position_sizers.get(config.sizing_method)
            if not sizer:
                raise ValueError(f"Unsupported position sizing method: {config.sizing_method}")
            
            # 戦略リストの決定
            strategies_to_process = []
            if allocation_result and allocation_result.strategy_weights:
                strategies_to_process = list(allocation_result.strategy_weights.keys())
            elif strategy_scores:
                strategies_to_process = list(strategy_scores.keys())
            else:
                # デフォルト戦略リスト
                strategies_to_process = ['momentum_strategy', 'mean_reversion', 'trend_following']
            
            for strategy_name in strategies_to_process:
                # 戦略スコアの取得
                strategy_score = strategy_scores.get(strategy_name)
                if not strategy_score:
                    # デフォルトスコアを作成
                    strategy_score = StrategyScore(
                        strategy_name=strategy_name,
                        total_score=0.5,
                        component_scores={'default': 0.5},
                        confidence=0.5
                    )
                
                # ポートフォリオ重みをベースサイズとして使用
                if allocation_result and strategy_name in allocation_result.strategy_weights:
                    portfolio_weight = allocation_result.strategy_weights[strategy_name]
                    adjusted_config = self._adjust_config_for_strategy(config, portfolio_weight)
                else:
                    adjusted_config = config
                
                # ポジションサイズ計算
                position_result = sizer.calculate_position_size(
                    strategy_name=strategy_name,
                    strategy_score=strategy_score,
                    market_data=market_data,
                    portfolio_context=portfolio_context,
                    config=adjusted_config
                )
                
                # 絶対金額と株数の計算
                if market_data is not None and not market_data.empty:
                    price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
                    current_price = market_data[price_column].iloc[-1] if price_column in market_data.columns else 100.0
                    position_result.absolute_amount = self.portfolio_value * position_result.adjusted_size
                    position_result.share_count = int(position_result.absolute_amount / current_price)
                
                position_results[strategy_name] = position_result
                total_allocated_percentage += position_result.adjusted_size
                
                # ポートフォリオコンテキストの更新
                portfolio_context['total_allocated_percentage'] = total_allocated_percentage
            
            # 5. ポートフォリオレベル指標の計算
            remaining_cash = 1.0 - total_allocated_percentage
            portfolio_risk = self._estimate_portfolio_risk(position_results, market_data)
            diversification_score = self._calculate_diversification_score(position_results)
            regime_analysis = self._analyze_market_regime(market_data)
            
            # 6. 制約チェック
            constraint_violations = self._check_portfolio_constraints(
                position_results, total_allocated_percentage, config
            )
            
            # 7. リバランス必要性の判定
            rebalancing_needed = self._assess_rebalancing_need(
                position_results, current_positions, config
            )
            
            # 8. 結果の構築
            portfolio_result = PortfolioPositionSizing(
                total_portfolio_value=self.portfolio_value,
                position_results=position_results,
                total_allocated_percentage=total_allocated_percentage,
                remaining_cash_percentage=remaining_cash,
                portfolio_risk_estimate=portfolio_risk,
                diversification_score=diversification_score,
                regime_analysis=regime_analysis,
                constraint_violations=constraint_violations,
                rebalancing_needed=rebalancing_needed,
                metadata={
                    "ticker": ticker,
                    "calculation_time": datetime.now(),
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "config": config.__dict__,
                    "allocation_confidence": allocation_result.confidence_level if allocation_result else 0.5
                }
            )
            
            # 履歴への追加
            self._calculation_history.append(portfolio_result)
            self._last_calculation_time = datetime.now()
            
            logger.info(f"Position sizes calculated for {ticker}: {len(position_results)} strategies")
            return portfolio_result
            
        except Exception as e:
            logger.error(f"Error calculating position sizes for {ticker}: {e}")
            return self._create_empty_portfolio_result(f"Calculation error: {str(e)}")

    def _create_score_based_sizer(self) -> BasePositionSizer:
        """スコアベースのポジションサイザーを作成"""
        class ScoreBasedPositionSizer(BasePositionSizer):
            def calculate_position_size(self, strategy_name, strategy_score, market_data, portfolio_context, config):
                base_size = config.base_position_size
                score_multiplier = max(0.3, min(2.0, strategy_score.total_score * 2))
                adjusted_size = base_size * score_multiplier
                
                # 制約適用
                adjusted_size = max(config.min_position_size, min(config.max_position_size, adjusted_size))
                
                return PositionSizeResult(
                    strategy_name=strategy_name,
                    base_size=base_size,
                    adjusted_size=adjusted_size,
                    score_multiplier=score_multiplier,
                    calculation_reason="Score-based sizing"
                )
        
        return ScoreBasedPositionSizer()

    def _create_risk_parity_sizer(self) -> BasePositionSizer:
        """リスクパリティのポジションサイザーを作成"""
        class RiskParityPositionSizer(BasePositionSizer):
            def calculate_position_size(self, strategy_name, strategy_score, market_data, portfolio_context, config):
                # 簡略化されたリスクパリティ実装
                base_risk = config.max_portfolio_risk / 5  # 5戦略想定
                strategy_risk = strategy_score.component_scores.get('risk_adjusted', 0.2)
                
                if strategy_risk > 0:
                    risk_based_size = base_risk / strategy_risk
                else:
                    risk_based_size = config.base_position_size
                
                adjusted_size = max(config.min_position_size, min(config.max_position_size, risk_based_size))
                
                return PositionSizeResult(
                    strategy_name=strategy_name,
                    base_size=config.base_position_size,
                    adjusted_size=adjusted_size,
                    risk_multiplier=risk_based_size / config.base_position_size,
                    calculation_reason="Risk parity sizing"
                )
        
        return RiskParityPositionSizer()

    def _create_fixed_percentage_sizer(self) -> BasePositionSizer:
        """固定割合のポジションサイザーを作成"""
        class FixedPercentagePositionSizer(BasePositionSizer):
            def calculate_position_size(self, strategy_name, strategy_score, market_data, portfolio_context, config):
                return PositionSizeResult(
                    strategy_name=strategy_name,
                    base_size=config.base_position_size,
                    adjusted_size=config.base_position_size,
                    calculation_reason="Fixed percentage sizing"
                )
        
        return FixedPercentagePositionSizer()

    def _get_strategy_scores(self, ticker: str) -> Dict[str, StrategyScore]:
        """戦略スコアの取得"""
        try:
            if self.score_manager:
                all_scores = self.score_manager.calculate_comprehensive_scores([ticker])
                return all_scores.get(ticker, {})
            else:
                # デフォルトスコアを作成
                default_strategies = ['momentum_strategy', 'mean_reversion', 'trend_following']
                return {
                    strategy: StrategyScore(
                        strategy_name=strategy,
                        total_score=0.5 + (i * 0.1),
                        component_scores={'default': 0.5 + (i * 0.1)},
                        confidence=0.7
                    )
                    for i, strategy in enumerate(default_strategies)
                }
        except Exception as e:
            logger.error(f"Error getting strategy scores for {ticker}: {e}")
            return {}

    def _build_portfolio_context(self, 
                                allocation_result: Optional[AllocationResult],
                                current_positions: Optional[Dict[str, float]],
                                market_data: pd.DataFrame) -> Dict[str, Any]:
        """ポートフォリオコンテキストの構築"""
        context = {
            'total_allocated_percentage': 0.0,
            'current_portfolio_risk': allocation_result.expected_risk if allocation_result else 0.15,
            'portfolio_return': allocation_result.expected_return if allocation_result else 0.08,
            'similar_strategies_allocation': 0.0,
            'current_positions': current_positions or {},
            'market_volatility': 0.2  # デフォルト
        }
        
        # 市場ボラティリティの計算
        if market_data is not None and not market_data.empty:
            try:
                price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
                returns = market_data[price_column].pct_change().dropna()
                context['market_volatility'] = returns.std() * np.sqrt(252)
            except Exception as e:
                logger.warning(f"Error calculating market volatility: {e}")
        
        return context

    def _adjust_config_for_strategy(self, base_config: PositionSizingConfig, portfolio_weight: float) -> PositionSizingConfig:
        """戦略別に設定を調整"""
        adjusted_config = PositionSizingConfig(**base_config.__dict__)
        # ポートフォリオ重みをベースサイズとして使用
        adjusted_config.base_position_size = portfolio_weight
        return adjusted_config

    def _estimate_portfolio_risk(self, position_results: Dict[str, PositionSizeResult], market_data: pd.DataFrame) -> float:
        """ポートフォリオリスクの推定"""
        try:
            if not position_results:
                return 0.0
            
            # 単純化されたリスク推定（実際は共分散行列を使用）
            total_position_variance = sum(
                (result.adjusted_size ** 2) * (result.volatility_percentile + 0.1)
                for result in position_results.values()
            )
            
            # 分散化効果を考慮
            diversification_benefit = max(0.3, 1.0 - (1.0 / len(position_results)))
            portfolio_risk = np.sqrt(total_position_variance) * diversification_benefit
            
            return portfolio_risk
            
        except Exception as e:
            logger.warning(f"Error estimating portfolio risk: {e}")
            return 0.15  # デフォルトリスク

    def _calculate_diversification_score(self, position_results: Dict[str, PositionSizeResult]) -> float:
        """分散化スコアの計算"""
        if not position_results:
            return 0.0
        
        # ハーフィンダール指数を使用
        weights = [result.adjusted_size for result in position_results.values()]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
        
        normalized_weights = [w / total_weight for w in weights]
        hhi = sum(w ** 2 for w in normalized_weights)
        diversification_score = 1.0 - hhi
        
        return diversification_score

    def _analyze_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """市場環境の分析"""
        analysis = {
            'regime': 'range_bound',
            'volatility_regime': 'normal',
            'trend_strength': 0.5,
            'momentum': 0.0
        }
        
        try:
            if market_data is not None and not market_data.empty:
                price_column = 'close' if 'close' in market_data.columns else market_data.columns[1]
                
                if price_column in market_data.columns:
                    # 基本的な市場分析
                    close_prices = market_data[price_column]
                    returns = close_prices.pct_change().dropna()
                    
                    # ボラティリティ分析
                    volatility = returns.std() * np.sqrt(252)
                    if volatility > 0.3:
                        analysis['volatility_regime'] = 'high'
                    elif volatility < 0.15:
                        analysis['volatility_regime'] = 'low'
                    
                    # トレンド分析
                    if len(close_prices) >= 20:
                        sma_20 = close_prices.rolling(20).mean()
                        trend_slope = (sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]
                        analysis['trend_strength'] = abs(trend_slope)
                        analysis['momentum'] = trend_slope
                    
                    # レジーム判定
                    if analysis['trend_strength'] > 0.1:
                        if analysis['momentum'] > 0:
                            analysis['regime'] = 'trending_up'
                        else:
                            analysis['regime'] = 'trending_down'
                
        except Exception as e:
            logger.warning(f"Error in market regime analysis: {e}")
        
        return analysis

    def _check_portfolio_constraints(self, 
                                   position_results: Dict[str, PositionSizeResult],
                                   total_allocation: float,
                                   config: PositionSizingConfig) -> List[str]:
        """ポートフォリオ制約のチェック"""
        violations = []
        
        # 総配分制約
        if total_allocation > 0.95:
            violations.append(f"Total allocation {total_allocation:.3f} too high")
        
        # 個別ポジション制約
        for name, result in position_results.items():
            if result.adjusted_size > config.max_position_size:
                violations.append(f"Strategy {name} exceeds max position size")
        
        # 集中度制約
        if len(position_results) >= 3:
            sorted_positions = sorted(position_results.values(), key=lambda x: x.adjusted_size, reverse=True)
            top_3_allocation = sum(pos.adjusted_size for pos in sorted_positions[:3])
            if top_3_allocation > config.max_sector_concentration:
                violations.append(f"Top 3 strategies concentration {top_3_allocation:.3f} too high")
        
        return violations

    def _assess_rebalancing_need(self, 
                               position_results: Dict[str, PositionSizeResult],
                               current_positions: Optional[Dict[str, float]],
                               config: PositionSizingConfig) -> bool:
        """リバランス必要性の評価"""
        if not current_positions or not config.enable_dynamic_adjustment:
            return False
        
        # ポジション差分の計算
        total_drift = 0.0
        for strategy_name, result in position_results.items():
            current_pos = current_positions.get(strategy_name, 0.0)
            drift = abs(result.adjusted_size - current_pos)
            total_drift += drift
        
        # リバランス閾値（5%の差分で実行）
        rebalance_threshold = 0.05
        return total_drift > rebalance_threshold

    def _create_empty_portfolio_result(self, reason: str) -> PortfolioPositionSizing:
        """空のポートフォリオ結果を作成"""
        return PortfolioPositionSizing(
            total_portfolio_value=self.portfolio_value,
            position_results={},
            total_allocated_percentage=0.0,
            remaining_cash_percentage=1.0,
            portfolio_risk_estimate=0.0,
            diversification_score=0.0,
            regime_analysis={'regime': 'unknown'},
            constraint_violations=[reason]
        )

    # ユーティリティメソッド

    def save_config(self, filepath: Optional[str] = None):
        """設定の保存"""
        if not filepath:
            filepath = self.base_dir / "position_sizing_config.json"
        
        try:
            config_dict = {
                'base_position_size': self.config.base_position_size,
                'max_position_size': self.config.max_position_size,
                'min_position_size': self.config.min_position_size,
                'sizing_method': self.config.sizing_method.value,
                'risk_adjustment_type': self.config.risk_adjustment_type.value,
                'score_weight': self.config.score_weight,
                'risk_weight': self.config.risk_weight,
                'market_weight': self.config.market_weight,
                'trend_confidence_weight': self.config.trend_confidence_weight,
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'enable_dynamic_adjustment': self.config.enable_dynamic_adjustment,
                'regime_sensitivity': self.config.regime_sensitivity
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Position sizing config saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get_calculation_history(self, limit: Optional[int] = None) -> List[PortfolioPositionSizing]:
        """計算履歴の取得"""
        if limit:
            return self._calculation_history[-limit:]
        return self._calculation_history.copy()

    def clear_cache(self):
        """キャッシュのクリア"""
        self._position_cache.clear()
        logger.info("Position sizing cache cleared")

    def update_portfolio_value(self, new_value: float):
        """ポートフォリオ価値の更新"""
        self.portfolio_value = new_value
        if self.risk_manager:
            self.risk_manager.total_assets = new_value
        logger.info(f"Portfolio value updated to ${new_value:,.2f}")

    def create_position_sizing_report(self, result: PortfolioPositionSizing) -> Dict[str, Any]:
        """ポジションサイジングレポートの作成"""
        report = {
            "summary": {
                "total_strategies": len(result.position_results),
                "total_allocated": f"{result.total_allocated_percentage:.2%}",
                "remaining_cash": f"{result.remaining_cash_percentage:.2%}",
                "portfolio_risk": f"{result.portfolio_risk_estimate:.2%}",
                "diversification_score": f"{result.diversification_score:.3f}"
            },
            "positions": {},
            "regime_analysis": result.regime_analysis,
            "constraints": result.constraint_violations,
            "rebalancing_needed": result.rebalancing_needed
        }
        
        # 個別ポジション詳細
        for strategy_name, pos_result in result.position_results.items():
            report["positions"][strategy_name] = {
                "adjusted_size": f"{pos_result.adjusted_size:.2%}",
                "absolute_amount": f"${pos_result.absolute_amount:,.2f}" if pos_result.absolute_amount else "N/A",
                "share_count": pos_result.share_count or "N/A",
                "confidence": f"{pos_result.confidence_level:.2%}",
                "regime": pos_result.market_regime.value,
                "reason": pos_result.calculation_reason
            }
        
        return report

# モジュールレベルのユーティリティ関数

def create_default_position_sizing_config() -> Dict[str, Any]:
    """デフォルトポジションサイジング設定の作成"""
    return {
        "base_position_size": 0.02,
        "max_position_size": 0.10,
        "min_position_size": 0.005,
        "sizing_method": "hybrid_adaptive",
        "risk_adjustment_type": "composite",
        "score_weight": 0.4,
        "risk_weight": 0.3,
        "market_weight": 0.2,
        "trend_confidence_weight": 0.1,
        "max_portfolio_risk": 0.20,
        "risk_free_rate": 0.02,
        "volatility_lookback": 252,
        "enable_dynamic_adjustment": True,
        "adjustment_frequency": "daily",
        "regime_sensitivity": 0.7,
        "enable_correlation_adjustment": True,
        "max_sector_concentration": 0.40,
        "enable_liquidity_adjustment": True
    }

def quick_position_size_calculation(ticker: str, 
                                  portfolio_value: float = 1000000.0,
                                  base_position_size: float = 0.02) -> Dict[str, Any]:
    """クイックポジションサイズ計算（簡易版）"""
    try:
        # 簡易設定での計算
        config = PositionSizingConfig(base_position_size=base_position_size)
        adjuster = PositionSizeAdjuster(portfolio_value=portfolio_value)
        
        # ダミー市場データ（実際の実装では実データを使用）
        dummy_data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105],
            'volume': [1000, 1200, 900, 1100, 1300]
        })
        
        result = adjuster.calculate_portfolio_position_sizes(
            ticker=ticker,
            market_data=dummy_data,
            config=config
        )
        
        return adjuster.create_position_sizing_report(result)
        
    except Exception as e:
        logger.error(f"Quick position size calculation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 簡単なテスト実行
    logging.basicConfig(level=logging.INFO)
    
    # デフォルト設定で簡単なテスト
    result = quick_position_size_calculation("AAPL")
    print("Quick Position Size Test Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
