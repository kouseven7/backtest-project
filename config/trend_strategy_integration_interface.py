"""
Module: Trend Strategy Integration Interface
File: trend_strategy_integration_interface.py
Description: 
  3-1-2「トレンド判定と戦略スコアの統合インターフェース」
  トレンド判定器、戦略スコアリング、戦略選択器の統合インターフェース
  リアルタイム判定とバッチ処理に対応した厚い統合レイヤー

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.strategy_selector
  - config.strategy_scoring_model
  - indicators.unified_trend_detector
  - config.score_history_manager
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールのインポート
try:
    from config.strategy_selector import (
        StrategySelector, SelectionCriteria, StrategySelection, SelectionMethod
    )
    from config.strategy_scoring_model import (
        StrategyScoreCalculator, StrategyScoreManager, StrategyScore
    )
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    logging.getLogger(__name__).warning(f"Import error: {e}")

# オプショナルインポート
ScoreHistoryManager = None
TimeDecayFactorCalculator = None

try:
    from config.score_history_manager import ScoreHistoryManager
except ImportError:
    pass

try:
    from config.time_decay_factor import TimeDecayFactorCalculator
except ImportError:
    pass

# ロガーの設定
logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """統合処理ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class ProcessingMode(Enum):
    """処理モード"""
    REALTIME = "realtime"          # リアルタイム処理
    BATCH = "batch"                # バッチ処理
    SCHEDULED = "scheduled"        # スケジュール処理
    ON_DEMAND = "on_demand"        # オンデマンド処理

@dataclass
class TrendAnalysisResult:
    """トレンド分析結果"""
    trend_type: str
    confidence: float
    strength: float
    reliability: float
    trend_change_probability: float
    supporting_indicators: Dict[str, float]
    analysis_timestamp: datetime
    data_quality_score: float
    trend_duration_estimate: Optional[int] = None
    next_review_time: Optional[datetime] = None

@dataclass
class StrategyScoreBundle:
    """戦略スコアバンドル"""
    strategy_name: str
    base_score: float
    trend_adjusted_score: float
    confidence_adjusted_score: float
    time_decay_adjusted_score: float
    final_score: float
    score_components: Dict[str, float]
    calculation_metadata: Dict[str, Any]
    last_updated: datetime

@dataclass
class IntegratedDecisionResult:
    """統合判定結果"""
    # トレンド情報
    trend_analysis: TrendAnalysisResult
    
    # 戦略スコア情報
    strategy_scores: Dict[str, StrategyScoreBundle]
    
    # 戦略選択結果
    strategy_selection: StrategySelection
    
    # 統合メタデータ
    processing_mode: ProcessingMode
    integration_status: IntegrationStatus
    processing_time_ms: float
    cache_hit_rate: float
    data_quality_assessment: Dict[str, float]
    
    # リスク評価
    risk_assessment: Dict[str, float]
    
    # 推奨アクション
    recommended_actions: List[Dict[str, Any]]
    
    # 処理詳細
    ticker: str
    data_period: Tuple[datetime, datetime]
    result_timestamp: datetime = field(default_factory=datetime.now)
    result_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8])

@dataclass
class BatchProcessingResult:
    """バッチ処理結果"""
    total_processed: int
    successful_results: List[IntegratedDecisionResult]
    failed_tickers: List[Tuple[str, str]]  # (ticker, error_message)
    processing_summary: Dict[str, Any]
    batch_start_time: datetime
    batch_end_time: datetime
    batch_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12])

class TrendStrategyIntegrationInterface:
    """
    トレンド戦略統合インターフェース
    
    機能:
    1. トレンド判定→戦略スコア計算→戦略選択の統合ワークフロー
    2. リアルタイム処理とバッチ処理の両対応
    3. 高度なキャッシュシステム
    4. エラーハンドリングとフォールバック
    5. パフォーマンス監視と最適化
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 enable_async: bool = True):
        """統合インターフェースの初期化"""
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/integration")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_async = enable_async
        self.config = self._load_config(config_file)
        
        # コンポーネントの初期化
        self._initialize_components()
        
        # キャッシュシステム
        self._initialize_cache_system()
        
        # パフォーマンス監視
        self._initialize_performance_monitoring()
        
        # 非同期処理用
        if enable_async:
            self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        
        logger.info("TrendStrategyIntegrationInterface initialized")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """設定の読み込み"""
        default_config = {
            "cache": {
                "trend_ttl_seconds": 300,       # 5分
                "score_ttl_seconds": 600,       # 10分
                "result_ttl_seconds": 180,      # 3分
                "max_cache_size": 1000
            },
            "processing": {
                "max_workers": 4,
                "batch_size": 50,
                "timeout_seconds": 30,
                "retry_attempts": 3
            },
            "fallback": {
                "default_trend": "sideways",
                "default_confidence": 0.5,
                "fallback_strategies": ["VWAPBounceStrategy", "MomentumInvestingStrategy"]
            },
            "quality": {
                "min_data_points": 20,
                "min_confidence_threshold": 0.3,
                "data_quality_threshold": 0.6
            },
            "trend_strategy_mapping": {
                "uptrend": ["MomentumInvestingStrategy", "BreakoutStrategy", "TrendFollowingStrategy"],
                "downtrend": ["MeanReversionStrategy", "RSIStrategy"],
                "sideways": ["VWAPBounceStrategy", "RSIStrategy", "BollingerBandsStrategy"],
                "unknown": ["VWAPBounceStrategy", "MomentumInvestingStrategy"]
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # ディープマージ
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """辞書の深いマージ"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _initialize_components(self):
        """コンポーネントの初期化"""
        try:
            self.strategy_selector = StrategySelector()
            self.score_calculator = StrategyScoreCalculator()
            self.score_manager = StrategyScoreManager()
            
            # オプショナルコンポーネント
            if ScoreHistoryManager is not None:
                try:
                    self.score_history_manager = ScoreHistoryManager()
                except:
                    self.score_history_manager = None
                    logger.warning("ScoreHistoryManager initialization failed")
            else:
                self.score_history_manager = None
                logger.warning("ScoreHistoryManager not available")
                
            if TimeDecayFactorCalculator is not None:
                try:
                    self.time_decay_calculator = TimeDecayFactorCalculator()
                except:
                    self.time_decay_calculator = None
                    logger.warning("TimeDecayFactorCalculator initialization failed")
            else:
                self.time_decay_calculator = None
                logger.warning("TimeDecayFactorCalculator not available")
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    def _initialize_cache_system(self):
        """キャッシュシステムの初期化"""
        self.trend_cache: Dict[str, Tuple[datetime, TrendAnalysisResult]] = {}
        self.score_cache: Dict[str, Tuple[datetime, StrategyScoreBundle]] = {}
        self.result_cache: Dict[str, Tuple[datetime, IntegratedDecisionResult]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def _initialize_performance_monitoring(self):
        """パフォーマンス監視の初期化"""
        self.performance_metrics: Dict[str, Union[int, float, Dict[str, Any]]] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "component_performance": {}
        }

    def integrate_decision(self,
                          market_data: pd.DataFrame,
                          ticker: str,
                          selection_criteria: Optional[SelectionCriteria] = None,
                          force_refresh: bool = False) -> IntegratedDecisionResult:
        """統合判定（メインAPI）"""
        start_time = datetime.now()
        
        try:
            # データ品質チェック
            quality_assessment = self._assess_data_quality(market_data)
            if quality_assessment["overall_quality"] < self.config["quality"]["data_quality_threshold"]:
                logger.warning(f"Low data quality for {ticker}: {quality_assessment}")
            
            # キャッシュチェック
            if not force_refresh:
                cached_result = self._get_cached_result(ticker, market_data)
                if cached_result:
                    self.cache_stats["hits"] += 1
                    cached_result.integration_status = IntegrationStatus.CACHED
                    return cached_result
            
            self.cache_stats["misses"] += 1
            
            # 1. トレンド分析
            trend_result = self._analyze_trend(market_data, ticker)
            
            # 2. 戦略スコア計算
            score_bundles = self._calculate_strategy_scores(
                market_data, ticker, trend_result
            )
            
            # 3. 戦略選択
            strategy_selection = self._select_strategies(
                market_data, ticker, trend_result, score_bundles, selection_criteria
            )
            
            # 4. リスク評価
            risk_assessment = self._assess_risk(trend_result, strategy_selection, market_data)
            
            # 5. 推奨アクション生成
            recommended_actions = self._generate_recommendations(
                trend_result, strategy_selection, risk_assessment
            )
            
            # 6. 結果統合
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            integrated_result = IntegratedDecisionResult(
                trend_analysis=trend_result,
                strategy_scores=score_bundles,
                strategy_selection=strategy_selection,
                processing_mode=ProcessingMode.REALTIME,
                integration_status=IntegrationStatus.COMPLETED,
                processing_time_ms=processing_time,
                cache_hit_rate=self._calculate_cache_hit_rate(),
                data_quality_assessment=quality_assessment,
                risk_assessment=risk_assessment,
                recommended_actions=recommended_actions,
                ticker=ticker,
                data_period=(market_data.index[0].to_pydatetime(), market_data.index[-1].to_pydatetime())
            )
            
            # キャッシュに保存
            self._cache_result(integrated_result)
            
            # パフォーマンス更新
            self._update_performance_metrics(True, processing_time)
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Integration failed for {ticker}: {e}")
            self._update_performance_metrics(False, 0)
            return self._create_fallback_result(ticker, market_data, e)

    def _analyze_trend(self, market_data: pd.DataFrame, ticker: str) -> TrendAnalysisResult:
        """トレンド分析"""
        try:
            # UnifiedTrendDetectorを使用
            trend_detector = UnifiedTrendDetector(
                data=market_data,
                strategy_name="IntegrationInterface",
                method="advanced"
            )
            
            # 基本トレンド情報
            trend = trend_detector.detect_trend()
            confidence = trend_detector.get_confidence()
            strength = trend_detector.get_trend_strength()
            reliability = trend_detector.get_reliability_score()
            
            # 追加分析
            supporting_indicators = self._calculate_supporting_indicators(market_data)
            trend_change_prob = self._estimate_trend_change_probability(market_data, trend)
            data_quality = self._assess_data_quality_for_trend(market_data)
            
            return TrendAnalysisResult(
                trend_type=trend,
                confidence=confidence,
                strength=strength,
                reliability=reliability,
                trend_change_probability=trend_change_prob,
                supporting_indicators=supporting_indicators,
                analysis_timestamp=datetime.now(),
                data_quality_score=data_quality["overall_quality"],
                trend_duration_estimate=self._estimate_trend_duration(market_data, trend),
                next_review_time=datetime.now() + timedelta(minutes=30)
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {ticker}: {e}")
            return self._create_fallback_trend_analysis()

    def _calculate_strategy_scores(self,
                                 market_data: pd.DataFrame,
                                 ticker: str,
                                 trend_result: TrendAnalysisResult) -> Dict[str, StrategyScoreBundle]:
        """戦略スコア計算"""
        score_bundles = {}
        
        for strategy_name in self.strategy_selector.available_strategies:
            try:
                # キャッシュチェック
                cache_key = f"score_{strategy_name}_{ticker}_{trend_result.trend_type}"
                cached_score = self._get_cached_score(cache_key)
                if cached_score:
                    score_bundles[strategy_name] = cached_score
                    continue
                
                # スコア計算
                base_score = self.score_calculator.calculate_strategy_score(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    market_data=market_data
                )
                
                if base_score is None:
                    continue
                
                # 調整スコアの計算
                trend_adjusted = self._apply_trend_adjustment(base_score.total_score, trend_result)
                confidence_adjusted = trend_adjusted * trend_result.confidence
                time_decay_adjusted = self._apply_time_decay(confidence_adjusted, strategy_name)
                final_score = min(1.0, max(0.0, time_decay_adjusted))
                
                score_bundle = StrategyScoreBundle(
                    strategy_name=strategy_name,
                    base_score=base_score.total_score,
                    trend_adjusted_score=trend_adjusted,
                    confidence_adjusted_score=confidence_adjusted,
                    time_decay_adjusted_score=time_decay_adjusted,
                    final_score=final_score,
                    score_components=base_score.component_scores.copy(),
                    calculation_metadata={
                        "trend_type": trend_result.trend_type,
                        "trend_confidence": trend_result.confidence,
                        "data_quality": trend_result.data_quality_score,
                        "calculation_method": "integrated"
                    },
                    last_updated=datetime.now()
                )
                
                score_bundles[strategy_name] = score_bundle
                self._cache_score(cache_key, score_bundle)
                
            except Exception as e:
                logger.warning(f"Score calculation failed for {strategy_name}: {e}")
                continue
        
        return score_bundles

    def _select_strategies(self,
                         market_data: pd.DataFrame,
                         ticker: str,
                         trend_result: TrendAnalysisResult,
                         score_bundles: Dict[str, StrategyScoreBundle],
                         criteria: Optional[SelectionCriteria]) -> StrategySelection:
        """戦略選択"""
        try:
            # 戦略選択器を使用
            selection = self.strategy_selector.select_strategies(
                market_data=market_data,
                ticker=ticker,
                criteria=criteria
            )
            
            # トレンド情報で更新
            selection.trend_analysis.update({
                "integrated_trend": trend_result.trend_type,
                "integrated_confidence": trend_result.confidence,
                "trend_strength": trend_result.strength,
                "trend_reliability": trend_result.reliability
            })
            
            return selection
            
        except Exception as e:
            logger.error(f"Strategy selection failed for {ticker}: {e}")
            return self._create_fallback_strategy_selection(ticker)

    def _apply_trend_adjustment(self, base_score: float, trend_result: TrendAnalysisResult) -> float:
        """トレンド調整の適用"""
        try:
            # トレンド強度による調整
            trend_factor = 1.0 + (trend_result.strength - 0.5) * 0.2
            
            # 信頼度による調整
            confidence_factor = 0.8 + trend_result.confidence * 0.4
            
            adjusted_score = base_score * trend_factor * confidence_factor
            return min(1.0, max(0.0, adjusted_score))
            
        except Exception as e:
            logger.warning(f"Trend adjustment failed: {e}")
            return base_score

    def _apply_time_decay(self, score: float, strategy_name: str) -> float:
        """時間減衰の適用"""
        try:
            if self.time_decay_calculator:
                decay_factor = self.time_decay_calculator.calculate_decay_factor(
                    strategy_name=strategy_name,
                    current_time=datetime.now()
                )
                return score * decay_factor
            else:
                return score * 0.95  # デフォルト軽微な減衰
                
        except Exception as e:
            logger.warning(f"Time decay calculation failed for {strategy_name}: {e}")
            return score * 0.9

    # ============================================================================
    # キャッシュ管理・データ品質評価・リスク分析メソッド
    # ============================================================================

    def _get_cached_result(self, ticker: str, market_data: pd.DataFrame) -> Optional[IntegratedDecisionResult]:
        """キャッシュされた統合結果の取得"""
        cache_key = self._generate_result_cache_key(ticker, market_data)
        
        if cache_key in self.result_cache:
            cached_time, cached_result = self.result_cache[cache_key]
            ttl = self.config["cache"]["result_ttl_seconds"]
            
            if (datetime.now() - cached_time).total_seconds() < ttl:
                logger.debug(f"Cache hit for result: {ticker}")
                return cached_result
            else:
                # 期限切れのキャッシュを削除
                del self.result_cache[cache_key]
                self.cache_stats["evictions"] += 1
        
        return None

    def _cache_result(self, result: IntegratedDecisionResult):
        """統合結果のキャッシュ"""
        cache_key = self._generate_result_cache_key(result.ticker, None)
        self.result_cache[cache_key] = (datetime.now(), result)
        
        # キャッシュサイズ制限
        max_size = self.config["cache"]["max_cache_size"]
        if len(self.result_cache) > max_size:
            self._cleanup_cache("result")

    def _get_cached_score(self, cache_key: str) -> Optional[StrategyScoreBundle]:
        """キャッシュされたスコアの取得"""
        if cache_key in self.score_cache:
            cached_time, cached_score = self.score_cache[cache_key]
            ttl = self.config["cache"]["score_ttl_seconds"]
            
            if (datetime.now() - cached_time).total_seconds() < ttl:
                return cached_score
            else:
                del self.score_cache[cache_key]
                self.cache_stats["evictions"] += 1
        
        return None

    def _cache_score(self, cache_key: str, score_bundle: StrategyScoreBundle):
        """スコアのキャッシュ"""
        self.score_cache[cache_key] = (datetime.now(), score_bundle)

    def _generate_result_cache_key(self, ticker: str, market_data: Optional[pd.DataFrame]) -> str:
        """結果キャッシュキーの生成"""
        if market_data is not None:
            # データのハッシュを含める
            data_hash = hashlib.md5(str(market_data.tail(20).values.tobytes())).hexdigest()[:8]
            return f"result_{ticker}_{data_hash}"
        return f"result_{ticker}"

    def _cleanup_cache(self, cache_type: str):
        """キャッシュのクリーンアップ"""
        if cache_type == "result":
            # 古いエントリから削除
            sorted_entries = sorted(
                self.result_cache.items(),
                key=lambda x: x[1][0]  # タイムスタンプでソート
            )
            # 半分削除
            for key, _ in sorted_entries[:len(sorted_entries)//2]:
                del self.result_cache[key]
                self.cache_stats["evictions"] += 1

    def _calculate_cache_hit_rate(self) -> float:
        """キャッシュヒット率の計算"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / total if total > 0 else 0.0

    def _assess_data_quality(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """データ品質の評価"""
        quality_metrics = {}
        
        # 1. データ完全性
        total_points = len(market_data)
        missing_points = market_data.isnull().sum().sum()
        quality_metrics["completeness"] = 1.0 - (missing_points / (total_points * len(market_data.columns))) if total_points > 0 else 0
        
        # 2. データ量の充足性
        min_points = self.config["quality"]["min_data_points"]
        quality_metrics["sufficiency"] = min(1.0, total_points / min_points) if min_points > 0 else 1.0
        
        # 3. 価格データの妥当性
        try:
            prices = market_data['Adj Close']
            price_changes = prices.pct_change().dropna()
            
            # 異常な価格変動のチェック（日次5%以上の変動）
            extreme_changes = abs(price_changes) > 0.05
            quality_metrics["price_stability"] = 1.0 - (extreme_changes.sum() / len(price_changes)) if len(price_changes) > 0 else 1.0
            
        except:
            quality_metrics["price_stability"] = 0.5
        
        # 4. 総合品質スコア
        weights = {
            "completeness": 0.4,
            "sufficiency": 0.3,
            "price_stability": 0.3
        }
        
        overall_quality = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items() 
            if metric in quality_metrics
        )
        
        quality_metrics["overall_quality"] = overall_quality
        
        return quality_metrics

    def _assess_data_quality_for_trend(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """トレンド分析用データ品質評価"""
        basic_quality = self._assess_data_quality(market_data)
        
        # トレンド分析特有の品質チェック
        try:
            prices = market_data['Adj Close']
            
            # トレンドの明確性
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            trend_clarity = 1.0 / (1.0 + volatility * 10) if volatility > 0 else 1.0
            
            basic_quality["trend_clarity"] = trend_clarity
            
        except Exception as e:
            logger.warning(f"Trend-specific data quality assessment failed: {e}")
            basic_quality["trend_clarity"] = 0.5
        
        return basic_quality

    def _calculate_supporting_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """支持指標の計算"""
        indicators = {}
        
        try:
            prices = market_data['Adj Close']
            
            # 移動平均
            ma_20 = prices.rolling(20).mean()
            ma_50 = prices.rolling(50).mean()
            
            if len(ma_20) > 0 and len(ma_50) > 0:
                current_price = prices.iloc[-1]
                indicators["ma20_position"] = (current_price - ma_20.iloc[-1]) / ma_20.iloc[-1] if ma_20.iloc[-1] != 0 else 0
                indicators["ma50_position"] = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1] if ma_50.iloc[-1] != 0 else 0
                indicators["ma_cross"] = 1.0 if ma_20.iloc[-1] > ma_50.iloc[-1] else 0.0
            
            # RSI近似
            returns = prices.pct_change().dropna()
            if len(returns) >= 14:
                gains = returns.where(returns > 0, 0)
                losses = -returns.where(returns < 0, 0)
                avg_gain = gains.rolling(14).mean().iloc[-1]
                avg_loss = losses.rolling(14).mean().iloc[-1]
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"] = rsi / 100.0  # 0-1スケール
            
        except Exception as e:
            logger.warning(f"Supporting indicators calculation failed: {e}")
        
        return indicators

    def _estimate_trend_change_probability(self, market_data: pd.DataFrame, current_trend: str) -> float:
        """トレンド変化確率の推定"""
        try:
            prices = market_data['Adj Close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < 20:
                return 0.5  # データ不足の場合はデフォルト
            
            # ボラティリティベースの変化確率
            recent_vol = returns.tail(10).std()
            historical_vol = returns.std()
            vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1.0
            
            # 高ボラティリティ期間はトレンド変化確率が高い
            vol_based_prob = min(0.8, vol_ratio * 0.3)
            
            # 複合確率
            change_probability = vol_based_prob * 0.7 + 0.2  # 基準確率20%
            
            return min(0.9, max(0.1, change_probability))
            
        except Exception as e:
            logger.warning(f"Trend change probability estimation failed: {e}")
            return 0.5

    def _estimate_trend_duration(self, market_data: pd.DataFrame, trend: str) -> Optional[int]:
        """トレンド継続期間の推定（日数）"""
        try:
            # 簡略化された実装
            volatility = market_data['Adj Close'].pct_change().std()
            
            if volatility < 0.01:  # 低ボラティリティ
                return 30  # 1ヶ月程度
            elif volatility < 0.02:  # 中ボラティリティ
                return 15  # 2週間程度
            else:  # 高ボラティリティ
                return 7   # 1週間程度
                
        except:
            return None

    def _assess_risk(self, trend_result: TrendAnalysisResult, 
                    strategy_selection: StrategySelection, 
                    market_data: pd.DataFrame) -> Dict[str, float]:
        """包括的リスク評価"""
        risk_metrics = {}
        
        # 1. トレンドリスク
        risk_metrics["trend_uncertainty_risk"] = 1.0 - trend_result.confidence
        
        # 2. 戦略集中度リスク
        weights = list(strategy_selection.strategy_weights.values())
        if weights:
            # ハーフィンダール指数（集中度指標）
            hhi = sum(w**2 for w in weights)
            n_strategies = len(weights)
            min_hhi = 1.0 / n_strategies if n_strategies > 0 else 1.0
            
            if hhi <= min_hhi:
                risk_metrics["concentration_risk"] = 0.0
            else:
                risk_metrics["concentration_risk"] = (hhi - min_hhi) / (1.0 - min_hhi)
        else:
            risk_metrics["concentration_risk"] = 1.0
        
        # 3. 市場ボラティリティリスク
        try:
            returns = market_data['Adj Close'].pct_change().dropna()
            if len(returns) >= 10:
                current_vol = returns.std()
                annual_vol = current_vol * np.sqrt(252)
                
                # ボラティリティを0-1スケールにマッピング
                if annual_vol < 0.2:
                    risk_metrics["volatility_risk"] = annual_vol / 0.2 * 0.3  # 0-0.3
                elif annual_vol < 0.4:
                    risk_metrics["volatility_risk"] = 0.3 + (annual_vol - 0.2) / 0.2 * 0.4  # 0.3-0.7
                else:
                    risk_metrics["volatility_risk"] = min(1.0, 0.7 + (annual_vol - 0.4) / 0.6 * 0.3)  # 0.7-1.0
            else:
                risk_metrics["volatility_risk"] = 0.5
        except:
            risk_metrics["volatility_risk"] = 0.5
        
        # 4. 総合リスクスコア
        risk_metrics["overall_risk"] = (
            risk_metrics["trend_uncertainty_risk"] * 0.4 +
            risk_metrics["concentration_risk"] * 0.3 +
            risk_metrics["volatility_risk"] * 0.3
        )
        
        return risk_metrics

    def _generate_recommendations(self, trend_result: TrendAnalysisResult,
                                strategy_selection: StrategySelection,
                                risk_assessment: Dict[str, float]) -> List[Dict[str, Any]]:
        """推奨アクションの生成"""
        recommendations = []
        
        # 1. 高リスク警告
        if risk_assessment["overall_risk"] > 0.7:
            recommendations.append({
                "type": "risk_warning",
                "priority": "high",
                "title": "高リスク警告",
                "description": f"総合リスクスコア {risk_assessment['overall_risk']:.2f} - 慎重な取引を推奨",
                "action_items": [
                    "ポジションサイズの縮小を検討",
                    "追加的なリスク管理手法の適用",
                    "頻繁な監視とレビュー"
                ]
            })
        
        # 2. トレンド信頼度に基づく推奨
        if trend_result.confidence < 0.6:
            recommendations.append({
                "type": "trend_confidence",
                "priority": "medium",
                "title": "トレンド信頼度低下",
                "description": f"トレンド信頼度 {trend_result.confidence:.2f} - 追加分析が必要",
                "action_items": [
                    "より長期データでの再分析",
                    "複数の時間軸での確認",
                    "ボラティリティ指標の確認"
                ]
            })
        
        # 3. 戦略集中度警告
        if risk_assessment.get("concentration_risk", 0) > 0.6:
            recommendations.append({
                "type": "diversification",
                "priority": "medium",
                "title": "戦略分散化推奨",
                "description": "戦略が集中しています - 分散化を検討",
                "action_items": [
                    "追加戦略の選択を検討",
                    "戦略重みの再調整",
                    "異なる特性の戦略を追加"
                ]
            })
        
        # 4. 定期レビュー推奨
        recommendations.append({
            "type": "review_schedule",
            "priority": "info",
            "title": "定期レビュー",
            "description": "戦略選択の定期的な見直しを推奨",
            "action_items": [
                f"次回レビュー: {trend_result.next_review_time.strftime('%Y-%m-%d %H:%M') if trend_result.next_review_time else '24時間後'}",
                "市場環境変化の監視",
                "パフォーマンス追跡"
            ]
        })
        
        return recommendations

    # ============================================================================
    # フォールバック・エラーハンドリング
    # ============================================================================

    def _create_fallback_result(self, ticker: str, market_data: pd.DataFrame, error: Exception) -> IntegratedDecisionResult:
        """フォールバック統合結果の作成"""
        logger.warning(f"Creating fallback result for {ticker}: {error}")
        
        # フォールバック用トレンド分析
        fallback_trend = self._create_fallback_trend_analysis()
        
        # フォールバック用戦略選択
        fallback_strategies = self.config["fallback"]["fallback_strategies"]
        fallback_selection = self._create_fallback_strategy_selection(ticker, fallback_strategies)
        
        # フォールバック用スコアバンドル
        fallback_score_bundles = {}
        for strategy in fallback_strategies:
            fallback_score_bundles[strategy] = StrategyScoreBundle(
                strategy_name=strategy,
                base_score=0.5,
                trend_adjusted_score=0.5,
                confidence_adjusted_score=0.4,
                time_decay_adjusted_score=0.4,
                final_score=0.4,
                score_components={"fallback": 0.4},
                calculation_metadata={"fallback": True, "error": str(error)},
                last_updated=datetime.now()
            )
        
        return IntegratedDecisionResult(
            trend_analysis=fallback_trend,
            strategy_scores=fallback_score_bundles,
            strategy_selection=fallback_selection,
            processing_mode=ProcessingMode.ON_DEMAND,
            integration_status=IntegrationStatus.FAILED,
            processing_time_ms=0.0,
            cache_hit_rate=0.0,
            data_quality_assessment={"overall_quality": 0.3},
            risk_assessment={"overall_risk": 0.8, "fallback_risk": 1.0},
            recommended_actions=[{
                "type": "system_error",
                "priority": "high",
                "title": "システムエラー - フォールバック処理",
                "description": f"統合処理でエラーが発生しました: {str(error)[:100]}",
                "action_items": [
                    "システム状態の確認",
                    "データ品質の確認", 
                    "手動での戦略選択を検討"
                ]
            }],
            ticker=ticker,
            data_period=(datetime.now() - timedelta(days=30), datetime.now()),
            result_timestamp=datetime.now()
        )

    def _create_fallback_trend_analysis(self) -> TrendAnalysisResult:
        """フォールバックトレンド分析の作成"""
        return TrendAnalysisResult(
            trend_type=self.config["fallback"]["default_trend"],
            confidence=self.config["fallback"]["default_confidence"],
            strength=0.5,
            reliability=0.5,
            trend_change_probability=0.5,
            supporting_indicators={},
            analysis_timestamp=datetime.now(),
            data_quality_score=0.5
        )

    def _create_fallback_strategy_selection(self, ticker: str, strategies: Optional[List[str]] = None) -> StrategySelection:
        """フォールバック戦略選択の作成"""
        if strategies is None:
            strategies = self.config["fallback"]["fallback_strategies"]
        
        if not strategies:
            strategies = ["conservative"]  # デフォルト戦略
        
        equal_weight = 1.0 / len(strategies) if strategies else 0.0
        
        return StrategySelection(
            selected_strategies=strategies,
            strategy_scores={s: 0.5 for s in strategies},
            strategy_weights={s: equal_weight for s in strategies},
            selection_reason="Fallback selection due to system error",
            trend_analysis={"trend": "unknown", "confidence": 0.5},
            confidence_level=0.5,
            total_score=0.5,
            metadata={"fallback": True, "ticker": ticker}
        )

    def _update_performance_metrics(self, success: bool, processing_time: float):
        """パフォーマンスメトリクスの更新"""
        self.performance_metrics["total_requests"] += 1  # type: ignore
        
        if success:
            self.performance_metrics["successful_requests"] += 1  # type: ignore
        else:
            self.performance_metrics["failed_requests"] += 1  # type: ignore
        
        # 平均処理時間の更新
        current_avg = self.performance_metrics["average_processing_time"]  # type: ignore
        total_requests = self.performance_metrics["total_requests"]  # type: ignore
        new_avg = (current_avg * (total_requests - 1) + processing_time) / total_requests  # type: ignore
        self.performance_metrics["average_processing_time"] = new_avg
        
        # キャッシュヒット率の更新
        self.performance_metrics["cache_hit_rate"] = self._calculate_cache_hit_rate()

    # ============================================================================
    # 公開API・ユーティリティ
    # ============================================================================

    def get_performance_statistics(self) -> Dict[str, Any]:
        """パフォーマンス統計の取得"""
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "cache_statistics": self.cache_stats.copy(),
            "cache_sizes": {
                "result_cache": len(self.result_cache),
                "score_cache": len(self.score_cache),
                "trend_cache": len(self.trend_cache)
            }
        }

    def validate_market_data(self, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """市場データの検証"""
        issues: List[str] = []
        
        # 必須カラムのチェック
        required_columns = ['Adj Close']
        for col in required_columns:
            if col not in market_data.columns:
                issues.append(f"Missing required column: {col}")
        
        # データ量のチェック
        min_points = self.config["quality"]["min_data_points"]
        if len(market_data) < min_points:
            issues.append(f"Insufficient data points: {len(market_data)} < {min_points}")
        
        return len(issues) == 0, issues

    def cleanup_system(self):
        """システムクリーンアップ"""
        # キャッシュのクリーンアップ
        self._cleanup_cache("result")
        logger.info("System cleanup completed")


# ============================================================================
# ファクトリー関数・便利関数
# ============================================================================

def create_integration_interface(config_file: Optional[str] = None,
                                cache_dir: Optional[str] = None,
                                enable_async: bool = True) -> TrendStrategyIntegrationInterface:
    """統合インターフェースの作成"""
    return TrendStrategyIntegrationInterface(
        config_file=config_file,
        cache_dir=cache_dir,
        enable_async=enable_async
    )

def quick_strategy_decision(market_data: pd.DataFrame,
                           ticker: str,
                           max_strategies: int = 3) -> IntegratedDecisionResult:
    """クイック戦略判定（簡易版）"""
    interface = create_integration_interface(enable_async=False)
    
    criteria = SelectionCriteria(
        method=SelectionMethod.HYBRID,
        max_strategies=max_strategies,
        min_score_threshold=0.5
    )
    
    return interface.integrate_decision(market_data, ticker, criteria)


# 一時的な型定義（実際の実装が利用できない場合のフォールバック）
if StrategySelection is None:
    @dataclass
    class StrategySelection:  # type: ignore
        selected_strategies: List[str] = field(default_factory=list)
        strategy_scores: Dict[str, float] = field(default_factory=dict)
        strategy_weights: Dict[str, float] = field(default_factory=dict)
        selection_reason: str = ""
        trend_analysis: Dict[str, Any] = field(default_factory=dict)
        confidence_level: float = 0.0
        total_score: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

if SelectionCriteria is None:
    @dataclass
    class SelectionCriteria:  # type: ignore
        method: str = "hybrid"
        max_strategies: int = 3
        min_score_threshold: float = 0.5

if SelectionMethod is None:
    class SelectionMethod:  # type: ignore
        HYBRID = "hybrid"
        TOP_SCORE = "top_score"
        WEIGHT_BASED = "weight_based"


# エクスポート
__all__ = [
    "TrendStrategyIntegrationInterface",
    "IntegratedDecisionResult",
    "TrendAnalysisResult",
    "StrategyScoreBundle",
    "BatchProcessingResult",
    "IntegrationStatus",
    "ProcessingMode",
    "create_integration_interface",
    "quick_strategy_decision"
]


if __name__ == "__main__":
    # 簡単なテスト
    import pandas as pd
    import numpy as np
    
    print("[TOOL] TrendStrategyIntegrationInterface テスト開始")
    
    # サンプルデータ作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Adj Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 統合インターフェースのテスト
    print("[CHART] 統合インターフェース初期化テスト...")
    try:
        interface = create_integration_interface(enable_async=False)
        print("  ✓ 初期化成功")
        
        # データ検証テスト
        is_valid, issues = interface.validate_market_data(sample_data)
        print(f"  データ検証: {'✓ 有効' if is_valid else '[ERROR] 無効'}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")
        
        # シングル判定テスト
        print("[CHART] シングル判定テスト...")
        result = interface.integrate_decision(sample_data, "TEST")
        print(f"  選択戦略: {result.strategy_selection.selected_strategies}")
        print(f"  トレンド: {result.trend_analysis.trend_type} (信頼度: {result.trend_analysis.confidence:.2f})")
        print(f"  総合リスク: {result.risk_assessment.get('overall_risk', 'N/A')}")
        print(f"  処理時間: {result.processing_time_ms:.1f}ms")
        
        # パフォーマンス統計
        stats = interface.get_performance_statistics()
        print(f"  成功率: {stats['performance_metrics']['successful_requests']}/{stats['performance_metrics']['total_requests']}")
        
        print("[OK] 3-1-2「トレンド戦略統合インターフェース」実装完了！")
        
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        import traceback
        traceback.print_exc()
