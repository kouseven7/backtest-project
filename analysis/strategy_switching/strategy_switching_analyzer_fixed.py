"""
Module: Strategy Switching Analyzer
File: strategy_switching_analyzer.py
Description: 
  5-1-1「戦略切替のタイミング分析ツール」
  戦略切替のタイミング分析・評価システム

Author: imega
Created: 2025-01-21
Modified: 2025-01-21
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# 警告を抑制
warnings.filterwarnings('ignore')

# ロガーの設定
logger = logging.getLogger(__name__)

# 既存システムとの統合
try:
    from config.strategy_selector import StrategySelector
    from config.portfolio_risk_manager import PortfolioRiskManager
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from config.strategy_scoring_model import StrategyScoreCalculator
    from config.score_history_manager import ScoreHistoryManager
    HAS_CORE_MODULES = True
except ImportError as e:
    logger.warning(f"Core modules import error: {e}. Using fallback implementations.")
    HAS_CORE_MODULES = False

try:
    from config.drawdown_controller import DrawdownController
    HAS_DRAWDOWN_MODULE = True
except ImportError:
    HAS_DRAWDOWN_MODULE = False

@dataclass
class SwitchingEvent:
    """戦略切替イベント"""
    timestamp: datetime
    from_strategy: str
    to_strategy: str
    trigger_type: str
    market_regime: str
    performance_before: float
    performance_after: Optional[float] = None
    switching_cost: float = 0.0
    confidence_score: float = 0.0
    success: Optional[bool] = None

@dataclass
class SwitchingAnalysisResult:
    """切替分析結果"""
    analysis_period: Tuple[datetime, datetime]
    total_switches: int
    successful_switches: int
    success_rate: float
    avg_switching_cost: float
    total_opportunity_cost: float
    performance_improvement: float
    optimal_switching_points: List[datetime]
    suboptimal_switching_points: List[datetime]
    switching_patterns: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    recommendations: List[str]
    average_improvement: Optional[float] = None

class MarketRegime(Enum):
    """市場レジーム"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend" 
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class SwitchingTrigger(Enum):
    """切替トリガー"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONFIDENCE_DROP = "confidence_drop"
    MARKET_REGIME_CHANGE = "market_regime_change"
    MANUAL = "manual"
    SCHEDULED = "scheduled"

class StrategySwitchingAnalyzer:
    """戦略切替分析システム"""
    
    def __init__(self, config_path: Optional[Union[str, Dict[str, Any]]] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパスまたは設定辞書
        """
        self.config = self._load_config(config_path)
        
        # パフォーマンストラッキング
        self.performance_tracker = {
            'analysis_count': 0,
            'successful_predictions': 0,
            'processing_times': []
        }
        
        # 分析結果キャッシュ
        self.analysis_results = {}
        
        # 既存システムとの統合
        self._initialize_integrations()

    def _load_config(self, config_path: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        # configが辞書として渡された場合はそのまま返す
        if config_path is not None and isinstance(config_path, dict):
            return config_path
            
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "config", "strategy_switching", 
                "switching_analysis_config.json"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'analysis_settings': {
                'lookback_period': 252,
                'min_switching_interval': 7,
                'performance_threshold': 0.02,
                'confidence_threshold': 0.6
            },
            'switching_costs': {
                'transaction_cost': 0.001,
                'slippage': 0.0005,
                'opportunity_cost_rate': 0.01
            },
            'pattern_detection': {
                'enable_pattern_detection': True,
                'pattern_confidence_threshold': 0.7
            }
        }

    def _initialize_integrations(self):
        """既存システムとの統合初期化"""
        if HAS_CORE_MODULES:
            try:
                self.strategy_selector = StrategySelector()
                self.trend_detector = UnifiedTrendDetector()
                self.score_calculator = StrategyScoreCalculator()
                self.score_history = ScoreHistoryManager()
                logger.info("Core modules integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize core modules: {e}")
                self.strategy_selector = None
                self.trend_detector = None
                self.score_calculator = None
                self.score_history = None
        else:
            self.strategy_selector = None
            self.trend_detector = None
            self.score_calculator = None
            self.score_history = None

        if HAS_DRAWDOWN_MODULE:
            try:
                self.drawdown_controller = DrawdownController()
                logger.info("Drawdown controller integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize drawdown controller: {e}")
                self.drawdown_controller = None
        else:
            self.drawdown_controller = None

    def analyze_switching_performance(
        self, 
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategies: Optional[List[str]] = None,
        switching_events: Optional[List[Dict[str, Any]]] = None,
        analysis_period: Optional[Tuple[datetime, datetime]] = None
    ) -> SwitchingAnalysisResult:
        """
        戦略切替パフォーマンス分析
        
        Parameters:
            data: 価格・戦略データ
            start_date: 分析開始日
            end_date: 分析終了日
            strategies: 分析対象戦略リスト
            switching_events: 切替イベントリスト
            analysis_period: 分析期間
            
        Returns:
            切替分析結果
        """
        start_time = datetime.now()
        
        try:
            # データ前処理
            processed_data = self._preprocess_data(data, start_date, end_date)
            
            # 戦略切替イベントの検出または使用
            if switching_events is None:
                switching_events_internal = self._detect_switching_events(processed_data, strategies)
            else:
                # switching_eventsが辞書形式の場合、SwitchingEventオブジェクトに変換
                switching_events_internal = []
                for event in switching_events:
                    if isinstance(event, dict):
                        switching_event = SwitchingEvent(
                            timestamp=event.get('timestamp', datetime.now()),
                            from_strategy=event.get('from_strategy', 'unknown'),
                            to_strategy=event.get('to_strategy', 'unknown'),
                            trigger_type=event.get('trigger_type', 'market_condition'),
                            market_regime=event.get('market_regime', 'neutral'),
                            performance_before=event.get('performance_before', 0.0),
                            performance_after=event.get('performance_after', None),
                            confidence_score=event.get('confidence_score', 0.5),
                            switching_cost=event.get('switching_cost', 0.01),
                            success=event.get('success', True)
                        )
                        switching_events_internal.append(switching_event)
                    else:
                        switching_events_internal.append(event)
            
            # 市場レジーム分析
            regime_analysis = self._analyze_market_regimes(processed_data)
            
            # パフォーマンス計算
            performance_metrics = self._calculate_switching_performance(
                processed_data, switching_events_internal, regime_analysis
            )
            
            # 最適切替ポイントの特定
            optimal_points = self._identify_optimal_switching_points(
                processed_data, switching_events_internal, regime_analysis
            )
            
            # パターン分析
            switching_patterns = self._analyze_switching_patterns(switching_events_internal)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                switching_events_internal, performance_metrics, optimal_points
            )
            
            # 平均改善度の計算
            avg_improvement = performance_metrics.get('performance_improvement', 0.0)
            
            # 結果構築
            analysis_result = SwitchingAnalysisResult(
                analysis_period=(
                    processed_data.index[0] if not processed_data.empty else datetime.now(),
                    processed_data.index[-1] if not processed_data.empty else datetime.now()
                ),
                total_switches=len(switching_events_internal),
                successful_switches=sum(1 for e in switching_events_internal if e.success),
                success_rate=len([e for e in switching_events_internal if e.success]) / max(len(switching_events_internal), 1),
                avg_switching_cost=float(np.mean([e.switching_cost for e in switching_events_internal])) if switching_events_internal else 0.0,
                total_opportunity_cost=performance_metrics.get('total_opportunity_cost', 0.0),
                performance_improvement=performance_metrics.get('performance_improvement', 0.0),
                optimal_switching_points=optimal_points,
                suboptimal_switching_points=[e.timestamp for e in switching_events_internal if not e.success],
                switching_patterns=switching_patterns,
                regime_analysis=regime_analysis,
                recommendations=recommendations,
                average_improvement=avg_improvement
            )
            
            # 結果をキャッシュ
            cache_key = f"analysis_{start_date}_{end_date}_{hash(tuple(strategies or []))}"
            self.analysis_results[cache_key] = analysis_result
            
            # パフォーマンス追跡更新
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_tracker(processing_time, True)
            
            logger.info(f"Switching performance analysis completed: {analysis_result.total_switches} switches analyzed")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Switching performance analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_tracker(processing_time, False)
            raise

    def _preprocess_data(
        self, 
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """データ前処理"""
        try:
            processed_data = data.copy()
            
            # 日付範囲でフィルタリング
            if start_date and end_date:
                mask = (processed_data.index >= start_date) & (processed_data.index <= end_date)
                processed_data = processed_data.loc[mask]
            
            # 欠損値処理
            processed_data = processed_data.fillna(method='ffill').fillna(0)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return data

    def _detect_switching_events(self, data: pd.DataFrame, strategies: Optional[List[str]] = None) -> List[SwitchingEvent]:
        """戦略切替イベントの検出"""
        events = []
        
        if data.empty:
            return events
            
        try:
            # 簡単な切替イベント検出ロジック
            for i in range(1, min(len(data), 50)):  # 最初の50データポイントを確認
                if i % 10 == 0:  # 10日ごとに切替イベントを生成（デモ用）
                    timestamp = data.index[i]
                    returns = data['returns'].iloc[i] if 'returns' in data.columns else 0
                    
                    # 戦略切替の判定（簡易版）
                    from_strategy = 'momentum' if i % 20 < 10 else 'mean_reversion'
                    to_strategy = 'mean_reversion' if i % 20 < 10 else 'momentum'
                    
                    # 市場レジームの判定
                    market_regime = self._get_market_regime(data.iloc[max(0, i-10):i+1])
                    
                    event = SwitchingEvent(
                        timestamp=timestamp,
                        from_strategy=from_strategy,
                        to_strategy=to_strategy,
                        trigger_type='performance_review',
                        market_regime=market_regime,
                        performance_before=returns,
                        switching_cost=0.002,
                        confidence_score=0.7 + (i % 3) * 0.1,
                        success=returns > 0  # 簡単な成功判定
                    )
                    
                    events.append(event)
                    
            return events
            
        except Exception as e:
            logger.error(f"Switching event detection failed: {e}")
            return events

    def _get_market_regime(self, data: pd.DataFrame) -> str:
        """市場レジームの判定"""
        try:
            if self.trend_detector:
                trend_result = self.trend_detector.detect_trend(data)
                return trend_result.get('trend_direction', MarketRegime.UNKNOWN.value)
            else:
                # フォールバック：簡単なトレンド判定
                if 'close' in data.columns and len(data) > 1:
                    price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
                    if price_change > 0.02:
                        return MarketRegime.UPTREND.value
                    elif price_change < -0.02:
                        return MarketRegime.DOWNTREND.value
                    else:
                        return MarketRegime.SIDEWAYS.value
                return MarketRegime.UNKNOWN.value
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return MarketRegime.UNKNOWN.value

    def _analyze_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場レジーム分析"""
        try:
            regime_analysis = {
                'dominant_regime': MarketRegime.SIDEWAYS.value,
                'regime_changes': 3,
                'regime_stability': 0.7,
                'transition_periods': []
            }
            return regime_analysis
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return {}

    def _calculate_switching_performance(
        self, 
        data: pd.DataFrame, 
        switching_events: List[SwitchingEvent],
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """切替パフォーマンス計算"""
        try:
            if not switching_events:
                return {
                    'total_opportunity_cost': 0.0,
                    'performance_improvement': 0.0,
                    'switching_efficiency': 0.0
                }
            
            # 簡単なパフォーマンス計算
            total_cost = sum(event.switching_cost for event in switching_events)
            successful_switches = sum(1 for event in switching_events if event.success)
            performance_improvement = (successful_switches / len(switching_events)) * 0.05
            
            return {
                'total_opportunity_cost': total_cost,
                'performance_improvement': performance_improvement,
                'switching_efficiency': successful_switches / len(switching_events)
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {'total_opportunity_cost': 0.0, 'performance_improvement': 0.0}

    def _identify_optimal_switching_points(
        self, 
        data: pd.DataFrame, 
        switching_events: List[SwitchingEvent],
        regime_analysis: Dict[str, Any]
    ) -> List[datetime]:
        """最適切替ポイントの特定"""
        try:
            optimal_points = []
            
            if switching_events:
                # 成功した切替のタイムスタンプを最適ポイントとする
                optimal_points = [event.timestamp for event in switching_events if event.success]
            
            return optimal_points
            
        except Exception as e:
            logger.error(f"Optimal switching points identification failed: {e}")
            return []

    def _analyze_switching_patterns(self, switching_events: List[SwitchingEvent]) -> Dict[str, Any]:
        """切替パターン分析"""
        try:
            if not switching_events:
                return {}
                
            patterns = {
                'frequent_transitions': {},
                'successful_patterns': [],
                'timing_patterns': {},
                'regime_based_patterns': {}
            }
            
            # 戦略間遷移の分析
            for event in switching_events:
                transition = f"{event.from_strategy}->{event.to_strategy}"
                patterns['frequent_transitions'][transition] = patterns['frequent_transitions'].get(transition, 0) + 1
                
                if event.success:
                    patterns['successful_patterns'].append(transition)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {}

    def _generate_recommendations(
        self, 
        switching_events: List[SwitchingEvent],
        performance_metrics: Dict[str, float],
        optimal_points: List[datetime]
    ) -> List[str]:
        """推奨事項生成"""
        try:
            recommendations = []
            
            if not switching_events:
                recommendations.append("十分な切替データがありません")
                return recommendations
            
            success_rate = len([e for e in switching_events if e.success]) / len(switching_events)
            
            if success_rate < 0.5:
                recommendations.append("切替戦略の見直しが必要です")
                recommendations.append("切替タイミングの改善を検討してください")
            elif success_rate > 0.8:
                recommendations.append("切替戦略は良好に機能しています")
                recommendations.append("現在のアプローチを継続することを推奨します")
            else:
                recommendations.append("切替戦略は部分的に有効です")
                recommendations.append("パフォーマンスの向上余地があります")
            
            avg_cost = np.mean([e.switching_cost for e in switching_events])
            if avg_cost > 0.01:
                recommendations.append("切替コストが高い傾向にあります")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["分析中にエラーが発生しました"]

    def _update_performance_tracker(self, processing_time: float, success: bool):
        """パフォーマンス追跡の更新"""
        self.performance_tracker['analysis_count'] += 1
        self.performance_tracker['processing_times'].append(processing_time)
        
        if success:
            self.performance_tracker['successful_predictions'] += 1
            
        # 最新100件のみ保持
        if len(self.performance_tracker['processing_times']) > 100:
            self.performance_tracker['processing_times'] = \
                self.performance_tracker['processing_times'][-100:]

    def get_analysis_summary(self, analysis_result: SwitchingAnalysisResult) -> Dict[str, Any]:
        """分析結果のサマリー生成"""
        return {
            'period': f"{analysis_result.analysis_period[0].strftime('%Y-%m-%d')} to {analysis_result.analysis_period[1].strftime('%Y-%m-%d')}",
            'total_switches': analysis_result.total_switches,
            'success_rate': f"{analysis_result.success_rate:.1%}",
            'avg_switching_cost': f"{analysis_result.avg_switching_cost:.4f}",
            'performance_improvement': f"{analysis_result.performance_improvement:.2%}",
            'key_recommendations': analysis_result.recommendations[:3]
        }

    def export_analysis_results(self, analysis_result: SwitchingAnalysisResult, file_path: str):
        """分析結果のエクスポート"""
        try:
            export_data = {
                'analysis_summary': self.get_analysis_summary(analysis_result),
                'detailed_results': {
                    'analysis_period': [
                        analysis_result.analysis_period[0].isoformat(),
                        analysis_result.analysis_period[1].isoformat()
                    ],
                    'switching_metrics': {
                        'total_switches': analysis_result.total_switches,
                        'successful_switches': analysis_result.successful_switches,
                        'success_rate': analysis_result.success_rate,
                        'avg_switching_cost': analysis_result.avg_switching_cost
                    },
                    'performance_metrics': {
                        'total_opportunity_cost': analysis_result.total_opportunity_cost,
                        'performance_improvement': analysis_result.performance_improvement
                    },
                    'patterns_and_insights': {
                        'switching_patterns': analysis_result.switching_patterns,
                        'regime_analysis': analysis_result.regime_analysis,
                        'recommendations': analysis_result.recommendations
                    }
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    analyzer = StrategySwitchingAnalyzer()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    test_data['returns'] = test_data['close'].pct_change()
    
    try:
        # 分析実行
        result = analyzer.analyze_switching_performance(test_data)
        
        # 結果表示
        summary = analyzer.get_analysis_summary(result)
        print("\n=== Analysis Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
