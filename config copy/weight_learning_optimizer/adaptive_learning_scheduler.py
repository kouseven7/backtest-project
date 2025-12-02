"""
適応的学習スケジューラー

日次マイクロ調整、週次標準最適化、月次主要リバランシングを管理する
ハイブリッド学習アプローチの実装
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

class LearningMode(Enum):
    """学習モード"""
    MICRO_ADJUSTMENT = "micro_adjustment"  # 日次マイクロ調整（±2%）
    STANDARD_OPTIMIZATION = "standard_optimization"  # 週次標準最適化（±5%）
    MAJOR_REBALANCING = "major_rebalancing"  # 月次主要リバランシング（±20%）
    
@dataclass
class LearningSchedule:
    """学習スケジュール設定"""
    mode: LearningMode
    max_adjustment: float  # 最大調整幅
    frequency_days: int  # 実行頻度（日数）
    min_performance_threshold: float  # 最小パフォーマンス閾値
    trigger_conditions: List[str]  # トリガー条件
    
@dataclass
class AdjustmentResult:
    """調整結果"""
    mode: LearningMode
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    adjustment_magnitude: float
    expected_improvement: float
    confidence_score: float
    timestamp: datetime
    trigger_reason: str
    
class AdaptiveLearningScheduler:
    """
    適応的学習スケジューラー
    
    市場条件とパフォーマンスに基づいて動的に学習頻度と調整幅を調整し、
    日次マイクロ調整から月次主要リバランシングまでの
    マルチタイムフレーム学習を管理する。
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        performance_evaluator: Optional[Any] = None
    ):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            performance_evaluator: パフォーマンス評価器
        """
        self.logger = self._setup_logger()
        self.performance_evaluator = performance_evaluator
        self.config = self._load_config(config_path)
        
        # 学習スケジュール設定
        self.learning_schedules = self._initialize_learning_schedules()
        
        # 調整履歴
        self.adjustment_history = []
        
        # パフォーマンス追跡
        self.performance_tracking = {
            'daily_performance': [],
            'weekly_performance': [],
            'monthly_performance': [],
            'last_adjustment_date': None,
            'consecutive_poor_performance': 0
        }
        
        # 適応的パラメータ
        self.adaptive_parameters = {
            'volatility_multiplier': 1.0,
            'performance_sensitivity': 1.0,
            'market_regime_factor': 1.0,
            'learning_rate_decay': 0.95
        }
        
        self.logger.info("AdaptiveLearningScheduler initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.AdaptiveLearningScheduler")
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
            'micro_adjustment_threshold': 0.02,  # 2%
            'standard_optimization_threshold': 0.05,  # 5%
            'major_rebalancing_threshold': 0.20,  # 20%
            'performance_lookback_days': 30,
            'volatility_lookback_days': 20,
            'poor_performance_threshold': -0.005,  # -0.5%
            'consecutive_trigger_limit': 3,
            'min_days_between_major': 21,  # 3週間
            'market_stress_threshold': 0.03  # 3% 日次ボラティリティ
        }
        
        if config_path:
            # 設定ファイルから読み込み（実装省略）
            pass
            
        return default_config
        
    def _initialize_learning_schedules(self) -> Dict[LearningMode, LearningSchedule]:
        """学習スケジュールの初期化"""
        return {
            LearningMode.MICRO_ADJUSTMENT: LearningSchedule(
                mode=LearningMode.MICRO_ADJUSTMENT,
                max_adjustment=self.config['micro_adjustment_threshold'],
                frequency_days=1,
                min_performance_threshold=0.0,
                trigger_conditions=['daily_performance', 'minor_volatility']
            ),
            LearningMode.STANDARD_OPTIMIZATION: LearningSchedule(
                mode=LearningMode.STANDARD_OPTIMIZATION,
                max_adjustment=self.config['standard_optimization_threshold'],
                frequency_days=7,
                min_performance_threshold=-0.01,
                trigger_conditions=['weekly_performance', 'trend_change']
            ),
            LearningMode.MAJOR_REBALANCING: LearningSchedule(
                mode=LearningMode.MAJOR_REBALANCING,
                max_adjustment=self.config['major_rebalancing_threshold'],
                frequency_days=30,
                min_performance_threshold=-0.05,
                trigger_conditions=['monthly_performance', 'market_stress', 'regime_change']
            )
        }
        
    def determine_learning_mode(
        self,
        current_performance: float,
        market_data: pd.DataFrame,
        current_date: datetime = None
    ) -> Tuple[LearningMode, str]:
        """
        学習モードの決定
        
        Args:
            current_performance: 現在のパフォーマンス
            market_data: 市場データ
            current_date: 現在の日付
            
        Returns:
            学習モードと理由のタプル
        """
        if current_date is None:
            current_date = datetime.now()
            
        # 市場状況の分析
        market_analysis = self._analyze_market_conditions(market_data)
        
        # パフォーマンス分析
        performance_analysis = self._analyze_performance_trends(current_performance)
        
        # 時間ベースの条件チェック
        time_conditions = self._check_time_conditions(current_date)
        
        # モード決定ロジック
        mode, reason = self._select_optimal_mode(
            market_analysis, performance_analysis, time_conditions
        )
        
        self.logger.info(f"Learning mode determined: {mode.value}, reason: {reason}")
        return mode, reason
        
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """市場条件の分析"""
        analysis = {}
        
        if len(market_data) < 10:
            return {'volatility': 'unknown', 'trend': 'unknown', 'stress_level': 0.0}
            
        # ボラティリティ分析
        returns = market_data.pct_change().dropna()
        recent_volatility = returns.tail(self.config['volatility_lookback_days']).std().mean()
        
        if recent_volatility > self.config['market_stress_threshold']:
            analysis['volatility'] = 'high'
            analysis['stress_level'] = min(1.0, recent_volatility / self.config['market_stress_threshold'])
        elif recent_volatility < self.config['market_stress_threshold'] * 0.5:
            analysis['volatility'] = 'low'
            analysis['stress_level'] = 0.0
        else:
            analysis['volatility'] = 'normal'
            analysis['stress_level'] = recent_volatility / self.config['market_stress_threshold']
            
        # トレンド分析
        short_ma = market_data.rolling(5).mean()
        long_ma = market_data.rolling(20).mean()
        
        if len(short_ma) > 0 and len(long_ma) > 0:
            trend_strength = (short_ma.iloc[-1] / long_ma.iloc[-1] - 1).mean()
            
            if trend_strength > 0.02:
                analysis['trend'] = 'strong_upward'
            elif trend_strength < -0.02:
                analysis['trend'] = 'strong_downward'
            elif abs(trend_strength) > 0.005:
                analysis['trend'] = 'moderate'
            else:
                analysis['trend'] = 'sideways'
        else:
            analysis['trend'] = 'unknown'
            
        return analysis
        
    def _analyze_performance_trends(self, current_performance: float) -> Dict[str, Any]:
        """パフォーマンストレンドの分析"""
        analysis = {
            'current_performance': current_performance,
            'trend': 'unknown',
            'consecutive_poor': 0,
            'improvement_needed': False
        }
        
        # 履歴データがある場合のトレンド分析
        if len(self.performance_tracking['daily_performance']) > 0:
            recent_performance = self.performance_tracking['daily_performance'][-10:]
            
            # トレンド計算
            if len(recent_performance) >= 3:
                trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                
                if trend > 0.001:
                    analysis['trend'] = 'improving'
                elif trend < -0.001:
                    analysis['trend'] = 'deteriorating'
                else:
                    analysis['trend'] = 'stable'
                    
            # 連続的な低パフォーマンスの検出
            poor_performance_count = 0
            for perf in reversed(recent_performance):
                if perf < self.config['poor_performance_threshold']:
                    poor_performance_count += 1
                else:
                    break
                    
            analysis['consecutive_poor'] = poor_performance_count
            analysis['improvement_needed'] = poor_performance_count >= self.config['consecutive_trigger_limit']
            
        # 現在のパフォーマンスが特に悪い場合
        if current_performance < self.config['poor_performance_threshold'] * 2:
            analysis['improvement_needed'] = True
            
        return analysis
        
    def _check_time_conditions(self, current_date: datetime) -> Dict[str, bool]:
        """時間ベース条件のチェック"""
        conditions = {
            'daily_due': True,  # 日次は常に実行可能
            'weekly_due': False,
            'monthly_due': False,
            'sufficient_gap_from_major': True
        }
        
        if self.performance_tracking['last_adjustment_date']:
            last_date = self.performance_tracking['last_adjustment_date']
            days_since_last = (current_date - last_date).days
            
            # 週次条件
            conditions['weekly_due'] = days_since_last >= 7
            
            # 月次条件
            conditions['monthly_due'] = days_since_last >= 30
            
            # 主要リバランシングからの十分な間隔
            conditions['sufficient_gap_from_major'] = days_since_last >= self.config['min_days_between_major']
            
        return conditions
        
    def _select_optimal_mode(
        self,
        market_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        time_conditions: Dict[str, bool]
    ) -> Tuple[LearningMode, str]:
        """最適な学習モードの選択"""
        
        # 緊急調整が必要な場合
        if (performance_analysis['improvement_needed'] and 
            market_analysis['stress_level'] > 0.7 and
            time_conditions['sufficient_gap_from_major']):
            return LearningMode.MAJOR_REBALANCING, "emergency_rebalancing_due_to_poor_performance_and_market_stress"
            
        # 月次リバランシング条件
        if (time_conditions['monthly_due'] and
            (performance_analysis['trend'] == 'deteriorating' or
             market_analysis['volatility'] == 'high')):
            return LearningMode.MAJOR_REBALANCING, "monthly_rebalancing_due_to_performance_or_volatility"
            
        # 週次最適化条件
        if (time_conditions['weekly_due'] and
            (performance_analysis['consecutive_poor'] >= 2 or
             market_analysis['trend'] in ['strong_upward', 'strong_downward'])):
            return LearningMode.STANDARD_OPTIMIZATION, "weekly_optimization_due_to_trend_or_performance"
            
        # レジーム変化検出
        if (market_analysis['trend'] != 'unknown' and
            market_analysis['stress_level'] > 0.5):
            if time_conditions['sufficient_gap_from_major']:
                return LearningMode.MAJOR_REBALANCING, "regime_change_detected"
            else:
                return LearningMode.STANDARD_OPTIMIZATION, "regime_change_detected_recent_major"
                
        # デフォルトは日次マイクロ調整
        return LearningMode.MICRO_ADJUSTMENT, "daily_micro_adjustment"
        
    def calculate_adjustment_weights(
        self,
        current_weights: Dict[str, float],
        learning_mode: LearningMode,
        performance_gradient: Optional[Dict[str, float]] = None,
        market_signals: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        調整重みの計算
        
        Args:
            current_weights: 現在の重み
            learning_mode: 学習モード
            performance_gradient: パフォーマンス勾配
            market_signals: 市場シグナル
            
        Returns:
            調整された重み
        """
        schedule = self.learning_schedules[learning_mode]
        max_adjustment = schedule.max_adjustment
        
        # 適応的調整幅の計算
        adaptive_adjustment = self._calculate_adaptive_adjustment(max_adjustment, learning_mode)
        
        adjusted_weights = current_weights.copy()
        
        if performance_gradient:
            # パフォーマンス勾配に基づく調整
            adjusted_weights = self._apply_performance_gradient(
                adjusted_weights, performance_gradient, adaptive_adjustment
            )
            
        if market_signals:
            # 市場シグナルに基づく調整
            adjusted_weights = self._apply_market_signals(
                adjusted_weights, market_signals, adaptive_adjustment, learning_mode
            )
            
        # 制約の適用
        adjusted_weights = self._apply_weight_constraints(adjusted_weights, learning_mode)
        
        return adjusted_weights
        
    def _calculate_adaptive_adjustment(
        self,
        base_adjustment: float,
        learning_mode: LearningMode
    ) -> float:
        """適応的調整幅の計算"""
        adjustment = base_adjustment
        
        # ボラティリティ乗数
        adjustment *= self.adaptive_parameters['volatility_multiplier']
        
        # パフォーマンス感度
        adjustment *= self.adaptive_parameters['performance_sensitivity']
        
        # 市場レジーム要因
        adjustment *= self.adaptive_parameters['market_regime_factor']
        
        # 学習率減衰
        if learning_mode != LearningMode.MAJOR_REBALANCING:
            adjustment *= self.adaptive_parameters['learning_rate_decay']
            
        return min(adjustment, 0.5)  # 最大50%の調整
        
    def _apply_performance_gradient(
        self,
        weights: Dict[str, float],
        gradient: Dict[str, float],
        max_adjustment: float
    ) -> Dict[str, float]:
        """パフォーマンス勾配の適用"""
        adjusted_weights = weights.copy()
        
        for key in weights:
            if key in gradient:
                # 勾配方向への調整
                gradient_magnitude = abs(gradient[key])
                gradient_direction = np.sign(gradient[key])
                
                # 調整量の計算
                adjustment = min(
                    max_adjustment * gradient_magnitude,
                    max_adjustment
                ) * gradient_direction
                
                # 重みの更新
                new_weight = weights[key] + adjustment
                adjusted_weights[key] = max(0.0, min(1.0, new_weight))
                
        return adjusted_weights
        
    def _apply_market_signals(
        self,
        weights: Dict[str, float],
        signals: Dict[str, float],
        max_adjustment: float,
        learning_mode: LearningMode
    ) -> Dict[str, float]:
        """市場シグナルの適用"""
        adjusted_weights = weights.copy()
        
        # モードに応じた感度調整
        sensitivity = {
            LearningMode.MICRO_ADJUSTMENT: 0.3,
            LearningMode.STANDARD_OPTIMIZATION: 0.7,
            LearningMode.MAJOR_REBALANCING: 1.0
        }[learning_mode]
        
        for key in weights:
            # 対応するシグナルの検索
            signal_key = key.replace('portfolio_', '').replace('strategy_', '')
            
            if signal_key in signals:
                signal_strength = signals[signal_key]
                
                # シグナルに基づく調整
                adjustment = (
                    max_adjustment * 
                    signal_strength * 
                    sensitivity
                )
                
                new_weight = weights[key] + adjustment
                adjusted_weights[key] = max(0.0, min(1.0, new_weight))
                
        return adjusted_weights
        
    def _apply_weight_constraints(
        self,
        weights: Dict[str, float],
        learning_mode: LearningMode
    ) -> Dict[str, float]:
        """重み制約の適用"""
        constrained_weights = weights.copy()
        
        # ストラテジー重みの正規化
        strategy_keys = [k for k in weights if k.startswith('strategy_')]
        if strategy_keys:
            strategy_sum = sum(constrained_weights[k] for k in strategy_keys)
            if strategy_sum > 0:
                for k in strategy_keys:
                    constrained_weights[k] /= strategy_sum
                    
        # ポートフォリオ重みの正規化
        portfolio_keys = [k for k in weights if k.startswith('portfolio_')]
        if portfolio_keys:
            portfolio_sum = sum(constrained_weights[k] for k in portfolio_keys)
            if portfolio_sum > 0:
                for k in portfolio_keys:
                    constrained_weights[k] /= portfolio_sum
                    
        # 最小/最大制約
        for key in constrained_weights:
            if key.startswith('portfolio_'):
                constrained_weights[key] = max(0.001, min(0.999, constrained_weights[key]))
            elif key.startswith('meta_'):
                constrained_weights[key] = max(0.1, min(3.0, constrained_weights[key]))
                
        return constrained_weights
        
    def execute_adaptive_learning(
        self,
        current_weights: Dict[str, float],
        performance_data: pd.DataFrame,
        market_data: pd.DataFrame,
        optimization_callback: Optional[callable] = None
    ) -> AdjustmentResult:
        """
        適応的学習の実行
        
        Args:
            current_weights: 現在の重み
            performance_data: パフォーマンスデータ
            market_data: 市場データ
            optimization_callback: 最適化コールバック
            
        Returns:
            調整結果
        """
        current_date = datetime.now()
        
        # 現在のパフォーマンスの計算
        if self.performance_evaluator:
            current_performance = self.performance_evaluator.evaluate_performance(
                performance_data, current_weights
            ).combined_score
        else:
            current_performance = 0.0
            
        # 学習モードの決定
        learning_mode, reason = self.determine_learning_mode(
            current_performance, market_data, current_date
        )
        
        # パフォーマンス追跡の更新
        self._update_performance_tracking(current_performance, current_date)
        
        # 調整重みの計算
        if optimization_callback:
            adjusted_weights = optimization_callback(current_weights, learning_mode)
        else:
            adjusted_weights = self.calculate_adjustment_weights(
                current_weights, learning_mode
            )
            
        # 調整結果の作成
        adjustment_magnitude = self._calculate_adjustment_magnitude(
            current_weights, adjusted_weights
        )
        
        # 期待改善度の推定
        expected_improvement = self._estimate_expected_improvement(
            current_weights, adjusted_weights, performance_data
        )
        
        # 信頼度スコアの計算
        confidence_score = self._calculate_confidence_score(
            learning_mode, adjustment_magnitude, expected_improvement
        )
        
        result = AdjustmentResult(
            mode=learning_mode,
            original_weights=current_weights.copy(),
            adjusted_weights=adjusted_weights,
            adjustment_magnitude=adjustment_magnitude,
            expected_improvement=expected_improvement,
            confidence_score=confidence_score,
            timestamp=current_date,
            trigger_reason=reason
        )
        
        # 履歴に追加
        self.adjustment_history.append(result)
        
        # 適応パラメータの更新
        self._update_adaptive_parameters(result)
        
        self.logger.info(
            f"Adaptive learning executed: {learning_mode.value}, "
            f"adjustment: {adjustment_magnitude:.3f}, "
            f"expected improvement: {expected_improvement:.3f}"
        )
        
        return result
        
    def _update_performance_tracking(
        self,
        performance: float,
        current_date: datetime
    ) -> None:
        """パフォーマンス追跡の更新"""
        self.performance_tracking['daily_performance'].append(performance)
        self.performance_tracking['last_adjustment_date'] = current_date
        
        # データの保持期間制限
        max_daily_records = self.config['performance_lookback_days']
        if len(self.performance_tracking['daily_performance']) > max_daily_records:
            self.performance_tracking['daily_performance'] = (
                self.performance_tracking['daily_performance'][-max_daily_records:]
            )
            
    def _calculate_adjustment_magnitude(
        self,
        original: Dict[str, float],
        adjusted: Dict[str, float]
    ) -> float:
        """調整幅の計算"""
        total_adjustment = 0.0
        
        for key in original:
            if key in adjusted:
                total_adjustment += abs(adjusted[key] - original[key])
                
        return total_adjustment / len(original) if original else 0.0
        
    def _estimate_expected_improvement(
        self,
        original_weights: Dict[str, float],
        adjusted_weights: Dict[str, float],
        performance_data: pd.DataFrame
    ) -> float:
        """期待改善度の推定"""
        # 簡易推定（実際の実装ではより複雑な計算が必要）
        adjustment_magnitude = self._calculate_adjustment_magnitude(
            original_weights, adjusted_weights
        )
        
        # 過去の調整実績に基づく推定
        if len(self.adjustment_history) > 0:
            recent_improvements = [
                result.expected_improvement for result in self.adjustment_history[-5:]
                if result.adjustment_magnitude > 0
            ]
            
            if recent_improvements:
                avg_improvement_rate = np.mean(recent_improvements) / np.mean([
                    result.adjustment_magnitude for result in self.adjustment_history[-5:]
                ])
                return adjustment_magnitude * avg_improvement_rate
                
        # デフォルト推定
        return adjustment_magnitude * 0.1  # 10%の改善を期待
        
    def _calculate_confidence_score(
        self,
        learning_mode: LearningMode,
        adjustment_magnitude: float,
        expected_improvement: float
    ) -> float:
        """信頼度スコアの計算"""
        base_confidence = {
            LearningMode.MICRO_ADJUSTMENT: 0.8,
            LearningMode.STANDARD_OPTIMIZATION: 0.6,
            LearningMode.MAJOR_REBALANCING: 0.4
        }[learning_mode]
        
        # 調整幅による調整
        magnitude_factor = 1 - min(adjustment_magnitude * 2, 0.3)
        
        # 期待改善度による調整
        improvement_factor = min(expected_improvement * 5, 0.2) + 0.8
        
        confidence = base_confidence * magnitude_factor * improvement_factor
        return max(0.0, min(1.0, confidence))
        
    def _update_adaptive_parameters(self, result: AdjustmentResult) -> None:
        """適応パラメータの更新"""
        # 学習率減衰の更新
        if result.confidence_score > 0.7:
            self.adaptive_parameters['learning_rate_decay'] *= 1.02  # 成功時は少し増加
        else:
            self.adaptive_parameters['learning_rate_decay'] *= 0.98  # 不成功時は減少
            
        # 境界値の維持
        self.adaptive_parameters['learning_rate_decay'] = max(
            0.5, min(1.0, self.adaptive_parameters['learning_rate_decay'])
        )
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計の取得"""
        if not self.adjustment_history:
            return {}
            
        recent_history = self.adjustment_history[-30:]  # 最近30回
        
        mode_counts = {}
        for result in recent_history:
            mode = result.mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
        return {
            'total_adjustments': len(self.adjustment_history),
            'recent_mode_distribution': mode_counts,
            'average_adjustment_magnitude': np.mean([
                result.adjustment_magnitude for result in recent_history
            ]),
            'average_confidence_score': np.mean([
                result.confidence_score for result in recent_history
            ]),
            'success_rate': np.mean([
                1 if result.expected_improvement > 0 else 0
                for result in recent_history
            ]),
            'adaptive_parameters': self.adaptive_parameters.copy()
        }
