"""
統合市場状態管理システム - A→B市場分類システム基盤
複数の分析コンポーネントを統合し、リアルタイム市場状態の管理を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength
from .market_condition_detector import MarketConditionDetector, DetectionResult
from .technical_indicator_analyzer import TechnicalIndicatorAnalyzer, TechnicalAnalysisResult
from .market_regime_classifier import MarketRegimeClassifier, RegimeClassificationResult
from .volatility_analyzer import VolatilityAnalyzer, VolatilityAnalysisResult
from .trend_strength_evaluator import TrendStrengthEvaluator, TrendStrengthResult
from .market_correlation_analyzer import MarketCorrelationAnalyzer, MarketCorrelationAnalysis

class MarketStateUpdateMode(Enum):
    """市場状態更新モード"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    HYBRID = "hybrid"
    EVENT_DRIVEN = "event_driven"

class StateIntegrationMethod(Enum):
    """状態統合手法"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS_VOTING = "consensus_voting"
    HIERARCHICAL = "hierarchical"
    MACHINE_LEARNING = "machine_learning"

class AlertLevel(Enum):
    """アラートレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComponentWeight:
    """コンポーネント重み設定"""
    condition_detector: float = 0.25
    technical_analyzer: float = 0.20
    regime_classifier: float = 0.20
    volatility_analyzer: float = 0.15
    trend_evaluator: float = 0.15
    correlation_analyzer: float = 0.05
    
    def __post_init__(self):
        total = (self.condition_detector + self.technical_analyzer + 
                self.regime_classifier + self.volatility_analyzer + 
                self.trend_evaluator + self.correlation_analyzer)
        if abs(total - 1.0) > 0.01:
            self.normalize()
    
    def normalize(self):
        """重みの正規化"""
        total = (self.condition_detector + self.technical_analyzer + 
                self.regime_classifier + self.volatility_analyzer + 
                self.trend_evaluator + self.correlation_analyzer)
        if total > 0:
            self.condition_detector /= total
            self.technical_analyzer /= total
            self.regime_classifier /= total
            self.volatility_analyzer /= total
            self.trend_evaluator /= total
            self.correlation_analyzer /= total

@dataclass
class IntegratedMarketState:
    """統合市場状態"""
    market_condition: MarketCondition
    market_strength: MarketStrength
    confidence_level: float
    component_results: Dict[str, Any]
    integration_method: StateIntegrationMethod
    state_timestamp: datetime
    next_update_time: Optional[datetime] = None
    state_stability: float = 0.0
    alert_level: AlertLevel = AlertLevel.LOW
    state_duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    def __post_init__(self):
        if self.next_update_time is None:
            self.next_update_time = self.state_timestamp + timedelta(minutes=15)

@dataclass
class StateChangeEvent:
    """状態変化イベント"""
    previous_state: IntegratedMarketState
    new_state: IntegratedMarketState
    change_magnitude: float
    change_components: List[str]
    event_time: datetime
    event_description: str

class IntegratedMarketStateManager:
    """
    統合市場状態管理システムのメインクラス
    複数の分析コンポーネントを統合し、リアルタイム市場状態管理を提供
    """
    
    def __init__(self, 
                 update_interval: int = 300,  # 5分間隔
                 integration_method: StateIntegrationMethod = StateIntegrationMethod.WEIGHTED_AVERAGE,
                 component_weights: Optional[ComponentWeight] = None,
                 auto_update: bool = False):
        """
        統合市場状態管理器の初期化
        
        Args:
            update_interval: 自動更新間隔（秒）
            integration_method: 状態統合手法
            component_weights: コンポーネント重み
            auto_update: 自動更新有効化
        """
        self.update_interval = update_interval
        self.integration_method = integration_method
        self.component_weights = component_weights or ComponentWeight()
        self.auto_update = auto_update
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # 分析コンポーネント初期化
        self._initialize_components()
        
        # 状態管理
        self.current_state: Optional[IntegratedMarketState] = None
        self.state_history: List[IntegratedMarketState] = []
        self.change_events: List[StateChangeEvent] = []
        
        # スレッド制御
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        
        # イベントハンドラー
        self._state_change_handlers: List[Callable[[StateChangeEvent], None]] = []
        
        # 最新データキャッシュ
        self._latest_data: Optional[Dict[str, pd.DataFrame]] = None
        self._data_timestamp: Optional[datetime] = None
        
        self.logger.info("IntegratedMarketStateManager初期化完了")

    def _initialize_components(self):
        """分析コンポーネント初期化"""
        try:
            self.condition_detector = MarketConditionDetector()
            self.technical_analyzer = TechnicalIndicatorAnalyzer()
            self.regime_classifier = MarketRegimeClassifier()
            self.volatility_analyzer = VolatilityAnalyzer()
            self.trend_evaluator = TrendStrengthEvaluator()
            self.correlation_analyzer = MarketCorrelationAnalyzer()
            
            self.logger.info("分析コンポーネント初期化完了")
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise

    def start_auto_update(self, data_provider: Callable[[], Dict[str, pd.DataFrame]]):
        """
        自動更新開始
        
        Args:
            data_provider: データ提供関数
        """
        if self._update_thread and self._update_thread.is_alive():
            self.logger.warning("自動更新は既に動作中です")
            return
        
        self.data_provider = data_provider
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
        self._update_thread.start()
        
        self.logger.info(f"自動更新開始 (間隔: {self.update_interval}秒)")

    def stop_auto_update(self):
        """自動更新停止"""
        if self._update_thread and self._update_thread.is_alive():
            self._stop_event.set()
            self._update_thread.join(timeout=5)
            self.logger.info("自動更新停止")

    def _auto_update_loop(self):
        """自動更新ループ"""
        while not self._stop_event.is_set():
            try:
                # データ取得
                if hasattr(self, 'data_provider'):
                    new_data = self.data_provider()
                    self.update_market_state(new_data)
                
                # 指定間隔で待機
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"自動更新エラー: {e}")
                self._stop_event.wait(min(self.update_interval, 60))  # エラー時は最大1分待機

    def update_market_state(self, 
                          data: Dict[str, pd.DataFrame],
                          force_update: bool = False) -> IntegratedMarketState:
        """
        市場状態の更新
        
        Args:
            data: 市場データ辞書
            force_update: 強制更新フラグ
            
        Returns:
            IntegratedMarketState: 更新された市場状態
        """
        try:
            with self._state_lock:
                # データ検証
                if not self._validate_data(data):
                    if self.current_state:
                        return self.current_state
                    else:
                        return self._create_default_state()
                
                # 更新タイミングチェック
                if not force_update and not self._should_update():
                    if self.current_state:
                        return self.current_state
                
                # データキャッシュ更新
                self._latest_data = data
                self._data_timestamp = datetime.now()
                
                # 各コンポーネントで分析実行
                component_results = self._run_component_analysis(data)
                
                # 状態統合
                integrated_state = self._integrate_market_state(component_results)
                
                # 状態変化検出とイベント生成
                if self.current_state:
                    change_event = self._detect_state_change(self.current_state, integrated_state)
                    if change_event:
                        self._handle_state_change(change_event)
                
                # 現在状態更新
                self.current_state = integrated_state
                self.state_history.append(integrated_state)
                
                # 履歴サイズ制限
                if len(self.state_history) > 1000:
                    self.state_history = self.state_history[-500:]
                
                self.logger.info(
                    f"市場状態更新: {integrated_state.market_condition.value} "
                    f"(強度: {integrated_state.market_strength.value}, 信頼度: {integrated_state.confidence_level:.3f})"
                )
                
                return integrated_state
                
        except Exception as e:
            self.logger.error(f"市場状態更新エラー: {e}")
            return self.current_state or self._create_default_state()

    def _run_component_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """各コンポーネントでの分析実行"""
        component_results = {}
        
        # メインデータ（最初の資産）を取得
        primary_data = list(data.values())[0] if data else None
        
        if primary_data is None:
            return component_results
        
        # 並列実行で各コンポーネント分析
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            # 各コンポーネント分析を並列実行
            try:
                futures['condition'] = executor.submit(
                    self.condition_detector.detect_market_conditions, primary_data
                )
                futures['technical'] = executor.submit(
                    self.technical_analyzer.analyze_technical_indicators, primary_data
                )
                futures['regime'] = executor.submit(
                    self.regime_classifier.classify_market_regime, primary_data
                )
                futures['volatility'] = executor.submit(
                    self.volatility_analyzer.analyze_volatility, primary_data
                )
                futures['trend'] = executor.submit(
                    self.trend_evaluator.evaluate_trend_strength, primary_data
                )
                
                # 相関分析は複数資産が必要
                if len(data) > 1:
                    futures['correlation'] = executor.submit(
                        self.correlation_analyzer.analyze_market_correlations, data
                    )
                
                # 結果取得
                for component, future in futures.items():
                    try:
                        result = future.result(timeout=30)  # 30秒タイムアウト
                        component_results[component] = result
                    except Exception as e:
                        self.logger.warning(f"{component}分析エラー: {e}")
                        component_results[component] = None
                        
            except Exception as e:
                self.logger.error(f"並列分析実行エラー: {e}")
        
        return component_results

    def _integrate_market_state(self, component_results: Dict[str, Any]) -> IntegratedMarketState:
        """コンポーネント結果の統合"""
        try:
            if self.integration_method == StateIntegrationMethod.WEIGHTED_AVERAGE:
                return self._weighted_average_integration(component_results)
            elif self.integration_method == StateIntegrationMethod.CONSENSUS_VOTING:
                return self._consensus_voting_integration(component_results)
            elif self.integration_method == StateIntegrationMethod.HIERARCHICAL:
                return self._hierarchical_integration(component_results)
            else:
                # デフォルトは重み付き平均
                return self._weighted_average_integration(component_results)
                
        except Exception as e:
            self.logger.error(f"状態統合エラー: {e}")
            return self._create_default_state()

    def _weighted_average_integration(self, component_results: Dict[str, Any]) -> IntegratedMarketState:
        """重み付き平均による統合"""
        try:
            # 市場状況スコア計算
            condition_scores = {}
            strength_scores = {}
            total_confidence = 0.0
            total_weight = 0.0
            
            # 各コンポーネント結果を数値化
            component_contributions = {}
            
            # 市場状況検出結果
            if component_results.get('condition'):
                detection_result = component_results['condition']
                if isinstance(detection_result, DetectionResult):
                    score = self._market_condition_to_score(detection_result.market_condition)
                    weight = self.component_weights.condition_detector * detection_result.confidence
                    condition_scores['condition'] = score * weight
                    strength_scores['condition'] = self._market_strength_to_score(detection_result.market_strength) * weight
                    total_confidence += detection_result.confidence * self.component_weights.condition_detector
                    total_weight += weight
                    component_contributions['condition_detector'] = weight
            
            # 技術指標分析結果
            if component_results.get('technical'):
                tech_result = component_results['technical']
                if isinstance(tech_result, TechnicalAnalysisResult):
                    # 技術指標の総合判定から市場状況を推定
                    score = self._technical_signals_to_score(tech_result)
                    weight = self.component_weights.technical_analyzer * tech_result.confidence
                    condition_scores['technical'] = score * weight
                    strength_scores['technical'] = abs(score) * weight
                    total_confidence += tech_result.confidence * self.component_weights.technical_analyzer
                    total_weight += weight
                    component_contributions['technical_analyzer'] = weight
            
            # レジーム分類結果
            if component_results.get('regime'):
                regime_result = component_results['regime']
                if isinstance(regime_result, RegimeClassificationResult):
                    score = self._regime_to_score(regime_result.regime_type)
                    weight = self.component_weights.regime_classifier * regime_result.confidence
                    condition_scores['regime'] = score * weight
                    strength_scores['regime'] = abs(score) * weight
                    total_confidence += regime_result.confidence * self.component_weights.regime_classifier
                    total_weight += weight
                    component_contributions['regime_classifier'] = weight
            
            # ボラティリティ分析結果
            if component_results.get('volatility'):
                vol_result = component_results['volatility']
                if isinstance(vol_result, VolatilityAnalysisResult):
                    # ボラティリティから市場強度を推定
                    weight = self.component_weights.volatility_analyzer * vol_result.confidence
                    vol_strength = min(vol_result.current_volatility / 0.3, 1.0)  # 正規化
                    strength_scores['volatility'] = vol_strength * weight
                    total_confidence += vol_result.confidence * self.component_weights.volatility_analyzer
                    total_weight += weight
                    component_contributions['volatility_analyzer'] = weight
            
            # トレンド強度結果
            if component_results.get('trend'):
                trend_result = component_results['trend']
                if isinstance(trend_result, TrendStrengthResult):
                    score = self._trend_direction_to_score(trend_result)
                    weight = self.component_weights.trend_evaluator * trend_result.confidence
                    condition_scores['trend'] = score * weight
                    strength_scores['trend'] = trend_result.composite_strength * weight
                    total_confidence += trend_result.confidence * self.component_weights.trend_evaluator
                    total_weight += weight
                    component_contributions['trend_evaluator'] = weight
            
            # 相関分析結果
            if component_results.get('correlation'):
                corr_result = component_results['correlation']
                if isinstance(corr_result, MarketCorrelationAnalysis):
                    # 相関は市場リスクに関連する補助情報として利用
                    weight = self.component_weights.correlation_analyzer * 0.8
                    # 相関が高いほど市場リスクが高い
                    avg_corr = abs(corr_result.diversification_metrics.get('average_correlation', 0))
                    risk_score = min(avg_corr * 2, 1.0)  # リスクスコア
                    # リスクスコアは強度に影響
                    strength_scores['correlation'] = risk_score * weight
                    total_weight += weight
                    component_contributions['correlation_analyzer'] = weight
            
            # 統合スコア計算
            if total_weight > 0:
                final_condition_score = sum(condition_scores.values()) / total_weight
                final_strength_score = sum(strength_scores.values()) / total_weight
                final_confidence = total_confidence / sum([
                    self.component_weights.condition_detector,
                    self.component_weights.technical_analyzer,
                    self.component_weights.regime_classifier,
                    self.component_weights.volatility_analyzer,
                    self.component_weights.trend_evaluator,
                    self.component_weights.correlation_analyzer
                ])
            else:
                final_condition_score = 0.0
                final_strength_score = 0.5
                final_confidence = 0.1
            
            # スコアから市場状況・強度を決定
            market_condition = self._score_to_market_condition(final_condition_score)
            market_strength = self._score_to_market_strength(final_strength_score)
            
            # 安定性評価
            state_stability = self._calculate_state_stability(component_contributions)
            
            # アラートレベル決定
            alert_level = self._determine_alert_level(final_confidence, state_stability, component_results)
            
            return IntegratedMarketState(
                market_condition=market_condition,
                market_strength=market_strength,
                confidence_level=final_confidence,
                component_results=component_results,
                integration_method=self.integration_method,
                state_timestamp=datetime.now(),
                state_stability=state_stability,
                alert_level=alert_level
            )
            
        except Exception as e:
            self.logger.error(f"重み付き平均統合エラー: {e}")
            return self._create_default_state()

    def _consensus_voting_integration(self, component_results: Dict[str, Any]) -> IntegratedMarketState:
        """コンセンサス投票による統合"""
        try:
            # 各コンポーネントの「投票」を収集
            condition_votes = []
            strength_votes = []
            confidence_values = []
            
            # 各コンポーネントからの投票
            for component, result in component_results.items():
                if result is None:
                    continue
                
                if component == 'condition' and isinstance(result, DetectionResult):
                    condition_votes.append(result.market_condition)
                    strength_votes.append(result.market_strength)
                    confidence_values.append(result.confidence)
                
                elif component == 'technical' and isinstance(result, TechnicalAnalysisResult):
                    # 技術指標から市場状況を推定
                    tech_condition = self._technical_to_condition(result)
                    tech_strength = self._technical_to_strength(result)
                    condition_votes.append(tech_condition)
                    strength_votes.append(tech_strength)
                    confidence_values.append(result.confidence)
                
                elif component == 'regime' and isinstance(result, RegimeClassificationResult):
                    regime_condition = self._regime_to_condition(result.regime_type)
                    regime_strength = self._regime_to_strength(result.regime_type)
                    condition_votes.append(regime_condition)
                    strength_votes.append(regime_strength)
                    confidence_values.append(result.confidence)
            
            # 投票結果の集計
            if condition_votes:
                # 最頻値または重み付き投票
                condition_counts = {}
                strength_counts = {}
                
                for i, (condition, strength) in enumerate(zip(condition_votes, strength_votes)):
                    weight = confidence_values[i] if i < len(confidence_values) else 1.0
                    
                    condition_counts[condition] = condition_counts.get(condition, 0) + weight
                    strength_counts[strength] = strength_counts.get(strength, 0) + weight
                
                # 最高得票の状況・強度を選択
                final_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
                final_strength = max(strength_counts.items(), key=lambda x: x[1])[0]
                
                # 信頼度は合意度を反映
                total_votes = sum(condition_counts.values())
                max_votes = max(condition_counts.values())
                consensus_ratio = max_votes / total_votes if total_votes > 0 else 0
                
                final_confidence = np.mean(confidence_values) * consensus_ratio
                
            else:
                final_condition = MarketCondition.NEUTRAL_SIDEWAYS
                final_strength = MarketStrength.MODERATE
                final_confidence = 0.1
            
            return IntegratedMarketState(
                market_condition=final_condition,
                market_strength=final_strength,
                confidence_level=final_confidence,
                component_results=component_results,
                integration_method=self.integration_method,
                state_timestamp=datetime.now(),
                state_stability=consensus_ratio if 'consensus_ratio' in locals() else 0.5,
                alert_level=AlertLevel.LOW
            )
            
        except Exception as e:
            self.logger.error(f"コンセンサス投票統合エラー: {e}")
            return self._create_default_state()

    def _hierarchical_integration(self, component_results: Dict[str, Any]) -> IntegratedMarketState:
        """階層的統合"""
        try:
            # 階層1: 主要判定（市場状況検出器）
            primary_result = component_results.get('condition')
            if primary_result and isinstance(primary_result, DetectionResult):
                base_condition = primary_result.market_condition
                base_strength = primary_result.market_strength
                base_confidence = primary_result.confidence
            else:
                base_condition = MarketCondition.NEUTRAL_SIDEWAYS
                base_strength = MarketStrength.MODERATE
                base_confidence = 0.3
            
            # 階層2: レジーム分析による調整
            regime_result = component_results.get('regime')
            if regime_result and isinstance(regime_result, RegimeClassificationResult):
                regime_condition = self._regime_to_condition(regime_result.regime_type)
                # レジーム結果が主要判定と大きく異なる場合は調整
                if self._conditions_diverge(base_condition, regime_condition):
                    # 重み付き調整
                    base_confidence *= 0.8  # 信頼度を下げる
                    # 条件を中間的なものに調整する場合の処理
                    if regime_result.confidence > base_confidence:
                        base_condition = regime_condition
            
            # 階層3: 技術指標による微調整
            tech_result = component_results.get('technical')
            if tech_result and isinstance(tech_result, TechnicalAnalysisResult):
                tech_condition = self._technical_to_condition(tech_result)
                tech_strength = self._technical_to_strength(tech_result)
                
                # 技術指標の信頼度が高い場合は強度を調整
                if tech_result.confidence > 0.7:
                    if tech_strength != base_strength:
                        # 強度の平均
                        base_strength = self._average_strengths(base_strength, tech_strength)
            
            # 階層4: ボラティリティによるリスク調整
            vol_result = component_results.get('volatility')
            if vol_result and isinstance(vol_result, VolatilityAnalysisResult):
                if vol_result.current_volatility > vol_result.historical_volatility * 1.5:
                    # 高ボラティリティ時はアラートレベルを上げる
                    alert_level = AlertLevel.HIGH
                    base_confidence *= 0.9  # 不確実性増加
                else:
                    alert_level = AlertLevel.LOW
            else:
                alert_level = AlertLevel.MEDIUM
            
            return IntegratedMarketState(
                market_condition=base_condition,
                market_strength=base_strength,
                confidence_level=base_confidence,
                component_results=component_results,
                integration_method=self.integration_method,
                state_timestamp=datetime.now(),
                state_stability=0.7,  # 階層的統合は比較的安定
                alert_level=alert_level
            )
            
        except Exception as e:
            self.logger.error(f"階層的統合エラー: {e}")
            return self._create_default_state()

    # ヘルパーメソッド群
    def _market_condition_to_score(self, condition: MarketCondition) -> float:
        """市場状況を数値スコアに変換 (-1.0 to 1.0)"""
        mapping = {
            MarketCondition.STRONG_BEAR: -1.0,
            MarketCondition.MODERATE_BEAR: -0.6,
            MarketCondition.SIDEWAYS_BEAR: -0.3,
            MarketCondition.NEUTRAL_SIDEWAYS: 0.0,
            MarketCondition.SIDEWAYS_BULL: 0.3,
            MarketCondition.MODERATE_BULL: 0.6,
            MarketCondition.STRONG_BULL: 1.0
        }
        return mapping.get(condition, 0.0)

    def _market_strength_to_score(self, strength: MarketStrength) -> float:
        """市場強度を数値スコアに変換 (0.0 to 1.0)"""
        mapping = {
            MarketStrength.VERY_WEAK: 0.1,
            MarketStrength.WEAK: 0.3,
            MarketStrength.MODERATE: 0.5,
            MarketStrength.STRONG: 0.7,
            MarketStrength.VERY_STRONG: 0.9
        }
        return mapping.get(strength, 0.5)

    def _score_to_market_condition(self, score: float) -> MarketCondition:
        """数値スコアから市場状況に変換"""
        if score >= 0.8:
            return MarketCondition.STRONG_BULL
        elif score >= 0.4:
            return MarketCondition.MODERATE_BULL
        elif score >= 0.1:
            return MarketCondition.SIDEWAYS_BULL
        elif score >= -0.1:
            return MarketCondition.NEUTRAL_SIDEWAYS
        elif score >= -0.4:
            return MarketCondition.SIDEWAYS_BEAR
        elif score >= -0.8:
            return MarketCondition.MODERATE_BEAR
        else:
            return MarketCondition.STRONG_BEAR

    def _score_to_market_strength(self, score: float) -> MarketStrength:
        """数値スコアから市場強度に変換"""
        if score >= 0.8:
            return MarketStrength.VERY_STRONG
        elif score >= 0.6:
            return MarketStrength.STRONG
        elif score >= 0.4:
            return MarketStrength.MODERATE
        elif score >= 0.2:
            return MarketStrength.WEAK
        else:
            return MarketStrength.VERY_WEAK

    def _technical_signals_to_score(self, tech_result: TechnicalAnalysisResult) -> float:
        """技術指標結果をスコアに変換"""
        try:
            # 各指標の信号を統合
            signals = tech_result.indicator_signals
            total_score = 0.0
            count = 0
            
            for indicator, signal in signals.items():
                if signal == 'BUY':
                    total_score += 1.0
                elif signal == 'SELL':
                    total_score -= 1.0
                elif signal == 'HOLD':
                    total_score += 0.0
                count += 1
            
            return total_score / count if count > 0 else 0.0
            
        except:
            return 0.0

    def _technical_to_condition(self, tech_result: TechnicalAnalysisResult) -> MarketCondition:
        """技術指標結果から市場状況を推定"""
        score = self._technical_signals_to_score(tech_result)
        return self._score_to_market_condition(score)

    def _technical_to_strength(self, tech_result: TechnicalAnalysisResult) -> MarketStrength:
        """技術指標結果から市場強度を推定"""
        # 信号の一致度から強度を推定
        signals = list(tech_result.indicator_signals.values())
        if not signals:
            return MarketStrength.MODERATE
        
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        total_count = len(signals)
        
        consensus = max(buy_count, sell_count) / total_count
        
        if consensus >= 0.8:
            return MarketStrength.VERY_STRONG
        elif consensus >= 0.6:
            return MarketStrength.STRONG
        else:
            return MarketStrength.MODERATE

    def _regime_to_score(self, regime_type: str) -> float:
        """レジームタイプをスコアに変換"""
        mapping = {
            'bull': 0.7,
            'bear': -0.7,
            'sideways': 0.0,
            'transitional': 0.0,
            'crisis': -0.9,
            'recovery': 0.5
        }
        return mapping.get(regime_type.lower(), 0.0)

    def _regime_to_condition(self, regime_type: str) -> MarketCondition:
        """レジームタイプから市場状況を変換"""
        mapping = {
            'bull': MarketCondition.MODERATE_BULL,
            'bear': MarketCondition.MODERATE_BEAR,
            'sideways': MarketCondition.NEUTRAL_SIDEWAYS,
            'transitional': MarketCondition.NEUTRAL_SIDEWAYS,
            'crisis': MarketCondition.STRONG_BEAR,
            'recovery': MarketCondition.SIDEWAYS_BULL
        }
        return mapping.get(regime_type.lower(), MarketCondition.NEUTRAL_SIDEWAYS)

    def _regime_to_strength(self, regime_type: str) -> MarketStrength:
        """レジームタイプから市場強度を変換"""
        mapping = {
            'bull': MarketStrength.STRONG,
            'bear': MarketStrength.STRONG,
            'sideways': MarketStrength.WEAK,
            'transitional': MarketStrength.MODERATE,
            'crisis': MarketStrength.VERY_STRONG,
            'recovery': MarketStrength.MODERATE
        }
        return mapping.get(regime_type.lower(), MarketStrength.MODERATE)

    def _trend_direction_to_score(self, trend_result: TrendStrengthResult) -> float:
        """トレンド方向をスコアに変換"""
        if trend_result.trend_direction == 'upward':
            return trend_result.composite_strength
        elif trend_result.trend_direction == 'downward':
            return -trend_result.composite_strength
        else:
            return 0.0

    def _conditions_diverge(self, condition1: MarketCondition, condition2: MarketCondition) -> bool:
        """2つの市場状況が大きく異なるかチェック"""
        score1 = self._market_condition_to_score(condition1)
        score2 = self._market_condition_to_score(condition2)
        return abs(score1 - score2) > 0.5

    def _average_strengths(self, strength1: MarketStrength, strength2: MarketStrength) -> MarketStrength:
        """2つの強度の平均"""
        score1 = self._market_strength_to_score(strength1)
        score2 = self._market_strength_to_score(strength2)
        avg_score = (score1 + score2) / 2
        return self._score_to_market_strength(avg_score)

    def _calculate_state_stability(self, component_contributions: Dict[str, float]) -> float:
        """状態安定性計算"""
        if not component_contributions:
            return 0.5
        
        # 貢献度の分散が小さいほど安定
        contributions = list(component_contributions.values())
        if len(contributions) <= 1:
            return 0.5
        
        variance = np.var(contributions)
        # 分散を0-1にマッピング（分散が小さいほど安定性が高い）
        stability = max(0, 1 - variance * 10)
        return min(stability, 1.0)

    def _determine_alert_level(self, confidence: float, stability: float, component_results: Dict[str, Any]) -> AlertLevel:
        """アラートレベル決定"""
        try:
            # 基本レベル（信頼度と安定性から）
            base_score = (confidence + stability) / 2
            
            # ボラティリティチェック
            vol_result = component_results.get('volatility')
            if vol_result and isinstance(vol_result, VolatilityAnalysisResult):
                if vol_result.current_volatility > vol_result.historical_volatility * 2:
                    return AlertLevel.CRITICAL
                elif vol_result.current_volatility > vol_result.historical_volatility * 1.5:
                    return AlertLevel.HIGH
            
            # 基本スコアからレベル決定
            if base_score >= 0.8:
                return AlertLevel.LOW
            elif base_score >= 0.6:
                return AlertLevel.MEDIUM
            elif base_score >= 0.4:
                return AlertLevel.HIGH
            else:
                return AlertLevel.CRITICAL
                
        except:
            return AlertLevel.MEDIUM

    def _detect_state_change(self, 
                           previous_state: IntegratedMarketState, 
                           new_state: IntegratedMarketState) -> Optional[StateChangeEvent]:
        """状態変化検出"""
        try:
            # 市場状況の変化
            condition_changed = previous_state.market_condition != new_state.market_condition
            strength_changed = previous_state.market_strength != new_state.market_strength
            
            if not (condition_changed or strength_changed):
                return None
            
            # 変化の大きさ計算
            prev_condition_score = self._market_condition_to_score(previous_state.market_condition)
            new_condition_score = self._market_condition_to_score(new_state.market_condition)
            condition_change = abs(new_condition_score - prev_condition_score)
            
            prev_strength_score = self._market_strength_to_score(previous_state.market_strength)
            new_strength_score = self._market_strength_to_score(new_state.market_strength)
            strength_change = abs(new_strength_score - prev_strength_score)
            
            change_magnitude = max(condition_change, strength_change)
            
            # 変化したコンポーネント特定
            change_components = []
            if condition_changed:
                change_components.append('market_condition')
            if strength_changed:
                change_components.append('market_strength')
            
            # イベント説明生成
            description = self._generate_change_description(previous_state, new_state)
            
            return StateChangeEvent(
                previous_state=previous_state,
                new_state=new_state,
                change_magnitude=change_magnitude,
                change_components=change_components,
                event_time=datetime.now(),
                event_description=description
            )
            
        except Exception as e:
            self.logger.error(f"状態変化検出エラー: {e}")
            return None

    def _generate_change_description(self, 
                                   previous_state: IntegratedMarketState, 
                                   new_state: IntegratedMarketState) -> str:
        """状態変化説明生成"""
        try:
            prev_condition = previous_state.market_condition.value
            new_condition = new_state.market_condition.value
            prev_strength = previous_state.market_strength.value
            new_strength = new_state.market_strength.value
            
            description_parts = []
            
            if prev_condition != new_condition:
                description_parts.append(f"市場状況: {prev_condition} → {new_condition}")
            
            if prev_strength != new_strength:
                description_parts.append(f"市場強度: {prev_strength} → {new_strength}")
            
            confidence_change = new_state.confidence_level - previous_state.confidence_level
            if abs(confidence_change) > 0.1:
                direction = "上昇" if confidence_change > 0 else "低下"
                description_parts.append(f"信頼度{direction}: {confidence_change:+.2f}")
            
            return "; ".join(description_parts)
            
        except:
            return "市場状態が変化しました"

    def _handle_state_change(self, change_event: StateChangeEvent):
        """状態変化ハンドリング"""
        try:
            # イベント履歴に追加
            self.change_events.append(change_event)
            
            # 履歴サイズ制限
            if len(self.change_events) > 100:
                self.change_events = self.change_events[-50:]
            
            # 登録されたハンドラーを実行
            for handler in self._state_change_handlers:
                try:
                    handler(change_event)
                except Exception as e:
                    self.logger.error(f"状態変化ハンドラーエラー: {e}")
            
            # ログ出力
            self.logger.info(f"状態変化検出: {change_event.event_description}")
            
        except Exception as e:
            self.logger.error(f"状態変化ハンドリングエラー: {e}")

    def add_state_change_handler(self, handler: Callable[[StateChangeEvent], None]):
        """状態変化ハンドラー追加"""
        self._state_change_handlers.append(handler)

    def remove_state_change_handler(self, handler: Callable[[StateChangeEvent], None]):
        """状態変化ハンドラー削除"""
        if handler in self._state_change_handlers:
            self._state_change_handlers.remove(handler)

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """データ検証"""
        if not data:
            return False
        
        for asset_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 10:
                return False
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                return False
        
        return True

    def _should_update(self) -> bool:
        """更新タイミング判定"""
        if not self.current_state:
            return True
        
        return datetime.now() >= self.current_state.next_update_time

    def _create_default_state(self) -> IntegratedMarketState:
        """デフォルト状態生成"""
        return IntegratedMarketState(
            market_condition=MarketCondition.NEUTRAL_SIDEWAYS,
            market_strength=MarketStrength.MODERATE,
            confidence_level=0.1,
            component_results={},
            integration_method=self.integration_method,
            state_timestamp=datetime.now(),
            state_stability=0.5,
            alert_level=AlertLevel.MEDIUM
        )

    # パブリックインターフェース
    def get_current_state(self) -> Optional[IntegratedMarketState]:
        """現在の市場状態取得"""
        with self._state_lock:
            return self.current_state

    def get_state_history(self, limit: int = 10) -> List[IntegratedMarketState]:
        """状態履歴取得"""
        with self._state_lock:
            return self.state_history[-limit:] if limit > 0 else self.state_history.copy()

    def get_change_events(self, limit: int = 10) -> List[StateChangeEvent]:
        """変化イベント履歴取得"""
        return self.change_events[-limit:] if limit > 0 else self.change_events.copy()

    def update_component_weights(self, weights: ComponentWeight):
        """コンポーネント重み更新"""
        self.component_weights = weights
        self.logger.info("コンポーネント重みを更新しました")

    def change_integration_method(self, method: StateIntegrationMethod):
        """統合手法変更"""
        self.integration_method = method
        self.logger.info(f"統合手法を {method.value} に変更しました")

    def get_component_status(self) -> Dict[str, bool]:
        """コンポーネント状態確認"""
        return {
            'condition_detector': hasattr(self, 'condition_detector'),
            'technical_analyzer': hasattr(self, 'technical_analyzer'),
            'regime_classifier': hasattr(self, 'regime_classifier'),
            'volatility_analyzer': hasattr(self, 'volatility_analyzer'),
            'trend_evaluator': hasattr(self, 'trend_evaluator'),
            'correlation_analyzer': hasattr(self, 'correlation_analyzer')
        }

    def export_state_to_dict(self, state: Optional[IntegratedMarketState] = None) -> Dict[str, Any]:
        """状態を辞書形式でエクスポート"""
        if state is None:
            state = self.current_state
        
        if state is None:
            return {}
        
        return {
            'market_condition': state.market_condition.value,
            'market_strength': state.market_strength.value,
            'confidence_level': state.confidence_level,
            'integration_method': state.integration_method.value,
            'state_timestamp': state.state_timestamp.isoformat(),
            'state_stability': state.state_stability,
            'alert_level': state.alert_level.value,
            'state_duration': str(state.state_duration) if state.state_duration else None
        }

    def __del__(self):
        """デストラクター"""
        self.stop_auto_update()

# 利便性関数
def create_basic_market_state_manager(auto_update: bool = False) -> IntegratedMarketStateManager:
    """基本的な市場状態管理器作成"""
    return IntegratedMarketStateManager(
        update_interval=300,  # 5分
        integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE,
        auto_update=auto_update
    )

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== 統合市場状態管理システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # 市場データシミュレーション
    base_price = 100
    volatility = 0.02
    returns = np.random.normal(0, volatility, 100)
    prices = base_price * (1 + returns).cumprod()
    
    market_data = {
        'ASSET1': pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
            'Close': prices,
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
    }
    
    # 統合管理器テスト
    manager = IntegratedMarketStateManager(
        integration_method=StateIntegrationMethod.WEIGHTED_AVERAGE
    )
    
    print("\n1. 初期状態更新")
    state = manager.update_market_state(market_data)
    print(f"市場状況: {state.market_condition.value}")
    print(f"市場強度: {state.market_strength.value}")
    print(f"信頼度: {state.confidence_level:.3f}")
    print(f"安定性: {state.state_stability:.3f}")
    print(f"アラートレベル: {state.alert_level.value}")
    
    print("\n2. 状態辞書エクスポート")
    state_dict = manager.export_state_to_dict()
    for key, value in state_dict.items():
        print(f"  {key}: {value}")
    
    print("\n3. 統合手法変更テスト")
    manager.change_integration_method(StateIntegrationMethod.CONSENSUS_VOTING)
    state2 = manager.update_market_state(market_data, force_update=True)
    print(f"コンセンサス投票結果: {state2.market_condition.value} (信頼度: {state2.confidence_level:.3f})")
    
    print("\n4. コンポーネント状態確認")
    component_status = manager.get_component_status()
    for component, status in component_status.items():
        status_str = "正常" if status else "エラー"
        print(f"  {component}: {status_str}")
    
    print("\n5. 状態変化ハンドラーテスト")
    def state_change_handler(event: StateChangeEvent):
        print(f"  状態変化イベント: {event.event_description}")
    
    manager.add_state_change_handler(state_change_handler)
    
    # 価格変動でデータ更新
    market_data_updated = market_data.copy()
    market_data_updated['ASSET1']['Close'] *= 1.05  # 5%上昇
    
    state3 = manager.update_market_state(market_data_updated, force_update=True)
    
    print("\n=== テスト完了 ===")
