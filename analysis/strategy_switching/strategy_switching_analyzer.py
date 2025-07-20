"""
Module: Strategy Switching Analyzer
File: strategy_switching_analyzer.py
Description: 
  5-1-1「戦略切替のタイミング分析ツール」
  戦略切替のタイミング分析とパフォーマンス最適化のメインエンジン

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
    VOLATILITY_SPIKE = "volatility_spike"
    MANUAL = "manual"
    SCHEDULED = "scheduled"

class StrategySwitchingAnalyzer:
    """戦略切替分析システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.switching_history: List[SwitchingEvent] = []
        self.analysis_results: Dict[str, SwitchingAnalysisResult] = {}
        
        # 既存システムとの統合
        self._initialize_integrations()
        
        # 分析データキャッシュ
        self.cache = {}
        self.cache_timestamp = None
        
        # パフォーマンス追跡
        self.performance_tracker = {
            'analysis_count': 0,
            'successful_predictions': 0,
            'cache_hits': 0,
            'processing_times': []
        }
        
        logger.info("StrategySwitchingAnalyzer initialized successfully")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
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
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'analysis_settings': {
                'analysis_mode': 'hybrid',
                'evaluation_method': 'comprehensive', 
                'market_regime_analysis': True,
                'time_granularity': 'hourly',
                'minimum_switching_interval': 60,
                'performance_lookback_periods': [30, 60, 90, 180]
            },
            'evaluation_criteria': {
                'performance_weight': 0.30,
                'risk_weight': 0.25,
                'timing_accuracy_weight': 0.25,
                'transaction_cost_weight': 0.20
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
                logger.info("Core module integrations initialized")
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
        strategies: Optional[List[str]] = None
    ) -> SwitchingAnalysisResult:
        """
        戦略切替パフォーマンス分析
        
        Parameters:
            data: 価格・戦略データ
            start_date: 分析開始日
            end_date: 分析終了日
            strategies: 分析対象戦略リスト
            
        Returns:
            切替分析結果
        """
        start_time = datetime.now()
        
        try:
            # データ前処理
            processed_data = self._preprocess_data(data, start_date, end_date)
            
            # 戦略切替イベントの検出
            switching_events = self._detect_switching_events(processed_data, strategies)
            
            # 市場レジーム分析
            regime_analysis = self._analyze_market_regimes(processed_data)
            
            # パフォーマンス計算
            performance_metrics = self._calculate_switching_performance(
                processed_data, switching_events, regime_analysis
            )
            
            # 最適切替ポイントの特定
            optimal_points = self._identify_optimal_switching_points(
                processed_data, switching_events, regime_analysis
            )
            
            # パターン分析
            switching_patterns = self._analyze_switching_patterns(switching_events)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                switching_events, performance_metrics, optimal_points
            )
            
            # 結果構築
            analysis_result = SwitchingAnalysisResult(
                analysis_period=(
                    processed_data.index[0] if not processed_data.empty else datetime.now(),
                    processed_data.index[-1] if not processed_data.empty else datetime.now()
                ),
                total_switches=len(switching_events),
                successful_switches=sum(1 for e in switching_events if e.success),
                success_rate=len([e for e in switching_events if e.success]) / max(len(switching_events), 1),
                avg_switching_cost=np.mean([e.switching_cost for e in switching_events]) if switching_events else 0.0,
                total_opportunity_cost=performance_metrics.get('total_opportunity_cost', 0.0),
                performance_improvement=performance_metrics.get('performance_improvement', 0.0),
                optimal_switching_points=optimal_points,
                suboptimal_switching_points=[e.timestamp for e in switching_events if not e.success],
                switching_patterns=switching_patterns,
                regime_analysis=regime_analysis,
                recommendations=recommendations
            )
            
            # 結果をキャッシュ
            cache_key = f"analysis_{start_date}_{end_date}_{hash(tuple(strategies or []))}"
            self.analysis_results[cache_key] = analysis_result
            
            # パフォーマンス追跡更新
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_tracker(processing_time, True)
            
            logger.info(f"Switching analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
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
        if data.empty:
            logger.warning("Empty data provided for analysis")
            return data
        
        # 日付フィルタリング
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # 必要列の確認・追加
        required_columns = ['close', 'volume', 'returns']
        
        if 'close' in data.columns:
            if 'returns' not in data.columns:
                data['returns'] = data['close'].pct_change()
        
        if 'volume' not in data.columns:
            data['volume'] = 1000000  # デフォルトボリューム
            
        # 欠損値処理
        data = data.fillna(method='ffill').fillna(0)
        
        return data

    def _detect_switching_events(
        self, 
        data: pd.DataFrame, 
        strategies: Optional[List[str]] = None
    ) -> List[SwitchingEvent]:
        """戦略切替イベントの検出"""
        events = []
        
        if data.empty:
            return events
            
        if strategies is None:
            strategies = ['momentum', 'mean_reversion', 'vwap', 'breakout']
            
        # 既存の戦略選択履歴から切替イベントを検出
        if self.strategy_selector:
            try:
                # 戦略選択履歴の取得
                selection_history = self._get_strategy_selection_history(data, strategies)
                events = self._extract_switching_events_from_history(selection_history)
            except Exception as e:
                logger.warning(f"Failed to get strategy selection history: {e}")
                events = self._simulate_switching_events(data, strategies)
        else:
            # フォールバック: シミュレートされた切替イベント
            events = self._simulate_switching_events(data, strategies)
            
        return events

    def _simulate_switching_events(self, data: pd.DataFrame, strategies: List[str]) -> List[SwitchingEvent]:
        """切替イベントのシミュレーション（フォールバック用）"""
        events = []
        
        if data.empty or len(strategies) < 2:
            return events
            
        # 簡単な切替ロジックでイベント生成
        current_strategy = strategies[0]
        
        for i in range(1, len(data)):
            timestamp = data.index[i]
            
            # 切替条件のシミュレーション
            returns = data['returns'].iloc[i] if 'returns' in data.columns else 0
            volatility = data['returns'].rolling(20).std().iloc[i] if 'returns' in data.columns else 0.01
            
            should_switch = False
            trigger_type = SwitchingTrigger.PERFORMANCE_DEGRADATION.value
            
            # パフォーマンス悪化
            if returns < -0.02:
                should_switch = True
                trigger_type = SwitchingTrigger.PERFORMANCE_DEGRADATION.value
            # ボラティリティ急上昇
            elif volatility > 0.03:
                should_switch = True
                trigger_type = SwitchingTrigger.VOLATILITY_SPIKE.value
                
            if should_switch:
                new_strategy = np.random.choice([s for s in strategies if s != current_strategy])
                
                event = SwitchingEvent(
                    timestamp=timestamp,
                    from_strategy=current_strategy,
                    to_strategy=new_strategy,
                    trigger_type=trigger_type,
                    market_regime=self._determine_market_regime(data, i),
                    performance_before=returns,
                    switching_cost=self._estimate_switching_cost(current_strategy, new_strategy),
                    confidence_score=np.random.uniform(0.5, 0.9),
                    success=returns > 0  # 簡単な成功判定
                )
                
                events.append(event)
                current_strategy = new_strategy
                
        return events

    def _determine_market_regime(self, data: pd.DataFrame, index: int) -> str:
        """市場レジームの判定"""
        if self.trend_detector:
            try:
                # 既存のトレンド検出器を使用
                trend_data = data.iloc[max(0, index-20):index+1]
                trend_result = self.trend_detector.detect_trend(trend_data)
                return trend_result.get('trend_direction', MarketRegime.UNKNOWN.value)
            except Exception as e:
                logger.warning(f"Trend detection failed: {e}")
                
        # フォールバック: 簡単なトレンド判定
        if 'returns' in data.columns and index >= 10:
            recent_returns = data['returns'].iloc[index-10:index].mean()
            if recent_returns > 0.001:
                return MarketRegime.UPTREND.value
            elif recent_returns < -0.001:
                return MarketRegime.DOWNTREND.value
            else:
                return MarketRegime.SIDEWAYS.value
                
        return MarketRegime.UNKNOWN.value

    def _estimate_switching_cost(self, from_strategy: str, to_strategy: str) -> float:
        """切替コストの推定"""
        base_cost = self.config.get('performance_calculation', {}).get('transaction_cost_bps', 5.0) / 10000
        
        # 戦略間の距離に基づくコスト調整
        strategy_costs = {
            ('momentum', 'mean_reversion'): 2.0,
            ('momentum', 'vwap'): 1.5,
            ('momentum', 'breakout'): 1.2,
            ('mean_reversion', 'vwap'): 1.3,
            ('mean_reversion', 'breakout'): 2.2,
            ('vwap', 'breakout'): 1.4
        }
        
        cost_multiplier = strategy_costs.get((from_strategy, to_strategy), 1.0)
        cost_multiplier = strategy_costs.get((to_strategy, from_strategy), cost_multiplier)
        
        return base_cost * cost_multiplier

    def _analyze_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場レジーム分析"""
        regime_analysis = {
            'regime_periods': [],
            'regime_transitions': [],
            'regime_performance': {},
            'regime_statistics': {}
        }
        
        if data.empty:
            return regime_analysis
            
        # 各時点でのレジーム判定
        regimes = []
        for i in range(len(data)):
            regime = self._determine_market_regime(data, i)
            regimes.append(regime)
            
        data['regime'] = regimes
        
        # レジーム統計
        regime_counts = pd.Series(regimes).value_counts()
        regime_analysis['regime_statistics'] = regime_counts.to_dict()
        
        # レジーム別パフォーマンス
        if 'returns' in data.columns:
            regime_performance = data.groupby('regime')['returns'].agg(['mean', 'std', 'count'])
            regime_analysis['regime_performance'] = regime_performance.to_dict()
            
        # レジーム遷移の検出
        transitions = []
        current_regime = regimes[0] if regimes else MarketRegime.UNKNOWN.value
        
        for i, regime in enumerate(regimes[1:], 1):
            if regime != current_regime:
                transitions.append({
                    'timestamp': data.index[i],
                    'from_regime': current_regime,
                    'to_regime': regime,
                    'index': i
                })
                current_regime = regime
                
        regime_analysis['regime_transitions'] = transitions
        
        return regime_analysis

    def _calculate_switching_performance(
        self,
        data: pd.DataFrame,
        switching_events: List[SwitchingEvent], 
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """切替パフォーマンスの計算"""
        performance = {
            'total_opportunity_cost': 0.0,
            'performance_improvement': 0.0,
            'successful_switches_performance': 0.0,
            'failed_switches_performance': 0.0,
            'regime_adjusted_performance': {}
        }
        
        if not switching_events or data.empty:
            return performance
            
        # 各切替イベントの前後パフォーマンス計算
        total_cost = 0.0
        total_improvement = 0.0
        
        for event in switching_events:
            # 切替前後のパフォーマンス比較
            event_index = data.index.get_loc(event.timestamp) if event.timestamp in data.index else -1
            
            if event_index > 0 and event_index < len(data) - 10:
                # 切替前10期間のパフォーマンス
                before_returns = data['returns'].iloc[event_index-10:event_index].mean() if 'returns' in data.columns else 0
                # 切替後10期間のパフォーマンス
                after_returns = data['returns'].iloc[event_index:event_index+10].mean() if 'returns' in data.columns else 0
                
                event.performance_after = after_returns
                improvement = after_returns - before_returns
                total_improvement += improvement
                
                # 機会損失の計算
                opportunity_cost = max(0, -improvement) * 0.1  # 簡易計算
                total_cost += opportunity_cost + event.switching_cost
                
        performance['total_opportunity_cost'] = total_cost
        performance['performance_improvement'] = total_improvement / max(len(switching_events), 1)
        
        # 成功・失敗別パフォーマンス
        successful_events = [e for e in switching_events if e.success]
        failed_events = [e for e in switching_events if not e.success]
        
        if successful_events:
            performance['successful_switches_performance'] = np.mean([
                e.performance_after - e.performance_before 
                for e in successful_events 
                if e.performance_after is not None
            ])
            
        if failed_events:
            performance['failed_switches_performance'] = np.mean([
                e.performance_after - e.performance_before 
                for e in failed_events 
                if e.performance_after is not None
            ])
            
        return performance

    def _identify_optimal_switching_points(
        self,
        data: pd.DataFrame,
        switching_events: List[SwitchingEvent],
        regime_analysis: Dict[str, Any]
    ) -> List[datetime]:
        """最適切替ポイントの特定"""
        optimal_points = []
        
        if data.empty or 'returns' in data.columns:
            return optimal_points
            
        # レジーム変化点を最適切替候補とする
        transitions = regime_analysis.get('regime_transitions', [])
        
        for transition in transitions:
            timestamp = transition['timestamp']
            
            # 前後のパフォーマンス確認
            try:
                index = data.index.get_loc(timestamp)
                if index > 5 and index < len(data) - 5:
                    before_perf = data['returns'].iloc[index-5:index].mean()
                    after_perf = data['returns'].iloc[index:index+5].mean()
                    
                    # パフォーマンス改善が期待できる点を最適とする
                    if after_perf > before_perf + 0.001:  # 閾値
                        optimal_points.append(timestamp)
                        
            except (KeyError, IndexError):
                continue
                
        return optimal_points

    def _analyze_switching_patterns(self, switching_events: List[SwitchingEvent]) -> Dict[str, Any]:
        """切替パターンの分析"""
        patterns = {
            'frequency_by_trigger': {},
            'success_rate_by_trigger': {},
            'strategy_transition_matrix': {},
            'time_patterns': {},
            'seasonal_patterns': {}
        }
        
        if not switching_events:
            return patterns
            
        # トリガー別分析
        trigger_counts = {}
        trigger_success = {}
        
        for event in switching_events:
            trigger = event.trigger_type
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            
            if event.success:
                trigger_success[trigger] = trigger_success.get(trigger, 0) + 1
                
        patterns['frequency_by_trigger'] = trigger_counts
        
        for trigger, count in trigger_counts.items():
            success_count = trigger_success.get(trigger, 0)
            patterns['success_rate_by_trigger'][trigger] = success_count / count if count > 0 else 0
            
        # 戦略遷移マトリックス
        transition_matrix = {}
        for event in switching_events:
            from_strategy = event.from_strategy
            to_strategy = event.to_strategy
            
            if from_strategy not in transition_matrix:
                transition_matrix[from_strategy] = {}
            transition_matrix[from_strategy][to_strategy] = \
                transition_matrix[from_strategy].get(to_strategy, 0) + 1
                
        patterns['strategy_transition_matrix'] = transition_matrix
        
        return patterns

    def _generate_recommendations(
        self,
        switching_events: List[SwitchingEvent],
        performance_metrics: Dict[str, Any],
        optimal_points: List[datetime]
    ) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        if not switching_events:
            recommendations.append("十分な切替データがありません。より長い期間でのデータ収集を推奨します。")
            return recommendations
            
        success_rate = len([e for e in switching_events if e.success]) / len(switching_events)
        avg_cost = performance_metrics.get('total_opportunity_cost', 0) / max(len(switching_events), 1)
        
        # 成功率に基づく推奨
        if success_rate < 0.5:
            recommendations.append("切替成功率が低いです。切替条件の見直しを推奨します。")
        elif success_rate > 0.8:
            recommendations.append("切替成功率が高く、現在の戦略は良好です。")
            
        # コストに基づく推奨
        if avg_cost > 0.01:
            recommendations.append("切替コストが高いです。頻度の調整または閾値の見直しを推奨します。")
            
        # 最適切替ポイント活用
        if len(optimal_points) > len(switching_events) * 1.5:
            recommendations.append("見逃している最適切替機会があります。トリガー感度の向上を検討してください。")
            
        # パフォーマンス改善
        improvement = performance_metrics.get('performance_improvement', 0)
        if improvement < 0:
            recommendations.append("切替によるパフォーマンス改善が見られません。戦略選択ロジックの見直しが必要です。")
            
        return recommendations

    def _get_strategy_selection_history(self, data: pd.DataFrame, strategies: List[str]) -> Dict[str, Any]:
        """戦略選択履歴の取得（既存システムから）"""
        # 実装予定: 既存のストラテジーセレクターから履歴を取得
        return {}

    def _extract_switching_events_from_history(self, selection_history: Dict[str, Any]) -> List[SwitchingEvent]:
        """選択履歴から切替イベントを抽出"""
        # 実装予定: 履歴データから切替イベントオブジェクトを生成
        return []

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
            # JSON形式でエクスポート
            export_data = {
                'analysis_metadata': {
                    'analysis_period_start': analysis_result.analysis_period[0].isoformat(),
                    'analysis_period_end': analysis_result.analysis_period[1].isoformat(),
                    'export_timestamp': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0'
                },
                'summary_metrics': {
                    'total_switches': analysis_result.total_switches,
                    'successful_switches': analysis_result.successful_switches,
                    'success_rate': analysis_result.success_rate,
                    'avg_switching_cost': analysis_result.avg_switching_cost,
                    'total_opportunity_cost': analysis_result.total_opportunity_cost,
                    'performance_improvement': analysis_result.performance_improvement
                },
                'optimal_switching_points': [
                    point.isoformat() for point in analysis_result.optimal_switching_points
                ],
                'suboptimal_switching_points': [
                    point.isoformat() for point in analysis_result.suboptimal_switching_points  
                ],
                'switching_patterns': analysis_result.switching_patterns,
                'regime_analysis': analysis_result.regime_analysis,
                'recommendations': analysis_result.recommendations
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Analysis results exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis results: {e}")
            raise

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
        result = analyzer.analyze_switching_performance(
            test_data, 
            strategies=['momentum', 'mean_reversion', 'vwap']
        )
        
        # 結果表示
        summary = analyzer.get_analysis_summary(result)
        print("\n=== 戦略切替分析結果サマリー ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
            
        print(f"\n分析成功: {result.total_switches}回の切替を分析しました")
        
    except Exception as e:
        print(f"分析エラー: {e}")
        raise
