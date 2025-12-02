"""
メタパラメータコントローラー

学習率、ボラティリティスケーリング、リスク回避などの
メタパラメータの動的調整
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

class ParameterType(Enum):
    """パラメータタイプ"""
    LEARNING_RATE = "learning_rate"
    VOLATILITY_SCALING = "volatility_scaling"
    RISK_AVERSION = "risk_aversion"
    REBALANCING_THRESHOLD = "rebalancing_threshold"
    MOMENTUM_FACTOR = "momentum_factor"
    DECAY_FACTOR = "decay_factor"

@dataclass
class MetaParameter:
    """メタパラメータ定義"""
    name: str
    parameter_type: ParameterType
    current_value: float
    default_value: float
    min_value: float
    max_value: float
    adaptation_rate: float  # 適応率
    sensitivity: float  # 市場変化への感度
    last_update: datetime
    update_count: int = 0

@dataclass
class ParameterUpdate:
    """パラメータ更新記録"""
    parameter_name: str
    old_value: float
    new_value: float
    update_reason: str
    market_condition: str
    performance_impact: float
    timestamp: datetime

class MetaParameterController:
    """
    メタパラメータコントローラー
    
    市場条件とパフォーマンスフィードバックに基づいて
    メタパラメータを動的に調整し、
    学習システムの適応性を向上させる。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = self._setup_logger()
        
        # メタパラメータの初期化
        self.parameters = self._initialize_parameters()
        
        # 更新履歴
        self.update_history = []
        
        # 市場状態追跡
        self.market_state = {
            'volatility_regime': 'normal',
            'trend_strength': 0.0,
            'stress_level': 0.0,
            'liquidity_condition': 'normal'
        }
        
        # パフォーマンス追跡
        self.performance_tracking = {
            'recent_performance': [],
            'parameter_impact_scores': {},
            'adaptation_success_rate': 0.0
        }
        
        # 設定の読み込み
        self.config = self._load_config(config_path)
        
        self.logger.info("MetaParameterController initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.MetaParameterController")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _initialize_parameters(self) -> Dict[str, MetaParameter]:
        """メタパラメータの初期化"""
        parameters = {}
        
        # 学習率
        parameters['learning_rate'] = MetaParameter(
            name='learning_rate',
            parameter_type=ParameterType.LEARNING_RATE,
            current_value=1.0,
            default_value=1.0,
            min_value=0.1,
            max_value=3.0,
            adaptation_rate=0.1,
            sensitivity=0.8,
            last_update=datetime.now()
        )
        
        # ボラティリティスケーリング
        parameters['volatility_scaling'] = MetaParameter(
            name='volatility_scaling',
            parameter_type=ParameterType.VOLATILITY_SCALING,
            current_value=1.0,
            default_value=1.0,
            min_value=0.5,
            max_value=2.5,
            adaptation_rate=0.15,
            sensitivity=1.0,
            last_update=datetime.now()
        )
        
        # リスク回避
        parameters['risk_aversion'] = MetaParameter(
            name='risk_aversion',
            parameter_type=ParameterType.RISK_AVERSION,
            current_value=1.0,
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            adaptation_rate=0.05,
            sensitivity=0.6,
            last_update=datetime.now()
        )
        
        # リバランシング閾値
        parameters['rebalancing_threshold'] = MetaParameter(
            name='rebalancing_threshold',
            parameter_type=ParameterType.REBALANCING_THRESHOLD,
            current_value=0.05,
            default_value=0.05,
            min_value=0.01,
            max_value=0.1,
            adaptation_rate=0.2,
            sensitivity=0.5,
            last_update=datetime.now()
        )
        
        # モメンタムファクター
        parameters['momentum_factor'] = MetaParameter(
            name='momentum_factor',
            parameter_type=ParameterType.MOMENTUM_FACTOR,
            current_value=0.9,
            default_value=0.9,
            min_value=0.5,
            max_value=0.99,
            adaptation_rate=0.05,
            sensitivity=0.4,
            last_update=datetime.now()
        )
        
        # 減衰ファクター
        parameters['decay_factor'] = MetaParameter(
            name='decay_factor',
            parameter_type=ParameterType.DECAY_FACTOR,
            current_value=0.95,
            default_value=0.95,
            min_value=0.8,
            max_value=0.99,
            adaptation_rate=0.03,
            sensitivity=0.3,
            last_update=datetime.now()
        )
        
        return parameters
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定の読み込み"""
        default_config = {
            'update_frequency_hours': 6,  # 6時間ごとの更新
            'min_performance_samples': 10,
            'market_analysis_window': 20,  # 20日間の市場分析窓
            'adaptation_threshold': 0.02,  # 2%以上の変化で適応
            'stress_detection_threshold': 0.03,  # 3%のボラティリティでストレス検出
            'performance_decay_factor': 0.9
        }
        
        if config_path:
            # 設定ファイルから読み込み（実装省略）
            pass
            
        return default_config
        
    def update_market_state(
        self,
        market_data: pd.DataFrame,
        performance_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        市場状態の更新
        
        Args:
            market_data: 市場データ
            performance_data: パフォーマンスデータ
        """
        if len(market_data) < 5:
            return
            
        # ボラティリティレジームの分析
        returns = market_data.pct_change().dropna()
        recent_vol = returns.tail(self.config['market_analysis_window']).std().mean()
        long_vol = returns.std().mean()
        
        if recent_vol > long_vol * 1.5:
            self.market_state['volatility_regime'] = 'high'
        elif recent_vol < long_vol * 0.7:
            self.market_state['volatility_regime'] = 'low'
        else:
            self.market_state['volatility_regime'] = 'normal'
            
        # トレンド強度の分析
        short_ma = market_data.rolling(5).mean()
        long_ma = market_data.rolling(20).mean()
        
        if len(short_ma) > 0 and len(long_ma) > 0:
            trend_strength = (short_ma.iloc[-1] / long_ma.iloc[-1] - 1).mean()
            self.market_state['trend_strength'] = trend_strength
            
        # ストレスレベルの計算
        if recent_vol > self.config['stress_detection_threshold']:
            self.market_state['stress_level'] = min(1.0, recent_vol / self.config['stress_detection_threshold'])
        else:
            self.market_state['stress_level'] = 0.0
            
        # 流動性状況の推定
        if 'volume' in market_data.columns:
            recent_volume = market_data['volume'].tail(10).mean()
            avg_volume = market_data['volume'].mean()
            
            if recent_volume < avg_volume * 0.7:
                self.market_state['liquidity_condition'] = 'low'
            elif recent_volume > avg_volume * 1.3:
                self.market_state['liquidity_condition'] = 'high'
            else:
                self.market_state['liquidity_condition'] = 'normal'
        else:
            self.market_state['liquidity_condition'] = 'normal'
            
        self.logger.debug(f"Market state updated: {self.market_state}")
        
    def adapt_parameters(
        self,
        performance_feedback: Dict[str, float],
        force_update: bool = False
    ) -> List[ParameterUpdate]:
        """
        パラメータの適応的調整
        
        Args:
            performance_feedback: パフォーマンスフィードバック
            force_update: 強制更新フラグ
            
        Returns:
            更新記録のリスト
        """
        updates = []
        
        # パフォーマンス追跡の更新
        self._update_performance_tracking(performance_feedback)
        
        for param_name, parameter in self.parameters.items():
            # 更新タイミングの確認
            time_since_update = datetime.now() - parameter.last_update
            should_update = (
                force_update or
                time_since_update.total_seconds() >= self.config['update_frequency_hours'] * 3600 or
                self._parameter_needs_adaptation(param_name, performance_feedback)
            )
            
            if should_update:
                # 新しい値の計算
                new_value = self._calculate_new_parameter_value(
                    parameter, performance_feedback
                )
                
                # 有意な変化がある場合のみ更新
                change_ratio = abs(new_value - parameter.current_value) / parameter.current_value
                
                if change_ratio >= self.config['adaptation_threshold'] or force_update:
                    # 更新記録の作成
                    update = ParameterUpdate(
                        parameter_name=param_name,
                        old_value=parameter.current_value,
                        new_value=new_value,
                        update_reason=self._determine_update_reason(param_name, performance_feedback),
                        market_condition=self._summarize_market_condition(),
                        performance_impact=self._estimate_performance_impact(param_name, new_value),
                        timestamp=datetime.now()
                    )
                    
                    # パラメータの更新
                    parameter.current_value = new_value
                    parameter.last_update = datetime.now()
                    parameter.update_count += 1
                    
                    updates.append(update)
                    
                    self.logger.info(
                        f"Parameter {param_name} updated: "
                        f"{update.old_value:.3f} -> {update.new_value:.3f} "
                        f"({update.update_reason})"
                    )
                    
        # 更新履歴に記録
        self.update_history.extend(updates)
        
        return updates
        
    def _update_performance_tracking(self, performance_feedback: Dict[str, float]) -> None:
        """パフォーマンス追跡の更新"""
        if 'combined_score' in performance_feedback:
            self.performance_tracking['recent_performance'].append({
                'score': performance_feedback['combined_score'],
                'timestamp': datetime.now()
            })
            
            # 古いデータの削除
            cutoff_time = datetime.now() - timedelta(days=30)
            self.performance_tracking['recent_performance'] = [
                perf for perf in self.performance_tracking['recent_performance']
                if perf['timestamp'] >= cutoff_time
            ]
            
    def _parameter_needs_adaptation(
        self,
        param_name: str,
        performance_feedback: Dict[str, float]
    ) -> bool:
        """パラメータ適応の必要性判定"""
        parameter = self.parameters[param_name]
        
        # 市場状態に基づく判定
        if parameter.parameter_type == ParameterType.VOLATILITY_SCALING:
            return self.market_state['volatility_regime'] != 'normal'
        elif parameter.parameter_type == ParameterType.RISK_AVERSION:
            return self.market_state['stress_level'] > 0.3
        elif parameter.parameter_type == ParameterType.LEARNING_RATE:
            return self._recent_performance_declining()
        elif parameter.parameter_type == ParameterType.REBALANCING_THRESHOLD:
            return self.market_state['volatility_regime'] == 'high'
            
        return False
        
    def _recent_performance_declining(self) -> bool:
        """最近のパフォーマンス低下の判定"""
        recent_scores = self.performance_tracking['recent_performance']
        
        if len(recent_scores) < 5:
            return False
            
        # 最近5回のスコア
        last_5_scores = [perf['score'] for perf in recent_scores[-5:]]
        
        # トレンド分析
        trend = np.polyfit(range(len(last_5_scores)), last_5_scores, 1)[0]
        
        return trend < -0.01  # 1%以上の低下トレンド
        
    def _calculate_new_parameter_value(
        self,
        parameter: MetaParameter,
        performance_feedback: Dict[str, float]
    ) -> float:
        """新しいパラメータ値の計算"""
        current_value = parameter.current_value
        adaptation_rate = parameter.adaptation_rate
        sensitivity = parameter.sensitivity
        
        # 基本調整量の計算
        adjustment = 0.0
        
        if parameter.parameter_type == ParameterType.LEARNING_RATE:
            adjustment = self._calculate_learning_rate_adjustment(
                performance_feedback, adaptation_rate, sensitivity
            )
        elif parameter.parameter_type == ParameterType.VOLATILITY_SCALING:
            adjustment = self._calculate_volatility_scaling_adjustment(
                adaptation_rate, sensitivity
            )
        elif parameter.parameter_type == ParameterType.RISK_AVERSION:
            adjustment = self._calculate_risk_aversion_adjustment(
                performance_feedback, adaptation_rate, sensitivity
            )
        elif parameter.parameter_type == ParameterType.REBALANCING_THRESHOLD:
            adjustment = self._calculate_rebalancing_threshold_adjustment(
                adaptation_rate, sensitivity
            )
        elif parameter.parameter_type == ParameterType.MOMENTUM_FACTOR:
            adjustment = self._calculate_momentum_factor_adjustment(
                performance_feedback, adaptation_rate, sensitivity
            )
        elif parameter.parameter_type == ParameterType.DECAY_FACTOR:
            adjustment = self._calculate_decay_factor_adjustment(
                performance_feedback, adaptation_rate, sensitivity
            )
            
        # 新しい値の計算
        new_value = current_value + adjustment
        
        # 境界値の適用
        new_value = max(parameter.min_value, min(parameter.max_value, new_value))
        
        return new_value
        
    def _calculate_learning_rate_adjustment(
        self,
        performance_feedback: Dict[str, float],
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """学習率調整の計算"""
        adjustment = 0.0
        
        # パフォーマンスに基づく調整
        if 'combined_score' in performance_feedback:
            recent_performance = self.performance_tracking['recent_performance']
            
            if len(recent_performance) >= 3:
                current_score = performance_feedback['combined_score']
                avg_recent = np.mean([p['score'] for p in recent_performance[-3:]])
                
                if current_score < avg_recent * 0.95:  # 5%以上の低下
                    adjustment = adaptation_rate * sensitivity  # 学習率を上げる
                elif current_score > avg_recent * 1.05:  # 5%以上の改善
                    adjustment = -adaptation_rate * sensitivity * 0.5  # 学習率をゆっくり下げる
                    
        # 市場状態に基づく調整
        if self.market_state['stress_level'] > 0.5:
            adjustment += adaptation_rate * 0.3  # ストレス時は学習率を上げる
            
        return adjustment
        
    def _calculate_volatility_scaling_adjustment(
        self,
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """ボラティリティスケーリング調整の計算"""
        adjustment = 0.0
        
        # ボラティリティレジームに基づく調整
        if self.market_state['volatility_regime'] == 'high':
            adjustment = adaptation_rate * sensitivity  # 高ボラ時はスケーリング増加
        elif self.market_state['volatility_regime'] == 'low':
            adjustment = -adaptation_rate * sensitivity * 0.5  # 低ボラ時はスケーリング減少
            
        # ストレスレベルに基づく調整
        stress_adjustment = self.market_state['stress_level'] * adaptation_rate * 0.5
        adjustment += stress_adjustment
        
        return adjustment
        
    def _calculate_risk_aversion_adjustment(
        self,
        performance_feedback: Dict[str, float],
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """リスク回避調整の計算"""
        adjustment = 0.0
        
        # ストレスレベルに基づく調整
        if self.market_state['stress_level'] > 0.4:
            adjustment = adaptation_rate * sensitivity * self.market_state['stress_level']
            
        # ドローダウンに基づく調整
        if 'max_drawdown' in performance_feedback:
            if performance_feedback['max_drawdown'] > 0.15:  # 15%以上のドローダウン
                adjustment += adaptation_rate * 0.5
                
        return adjustment
        
    def _calculate_rebalancing_threshold_adjustment(
        self,
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """リバランシング閾値調整の計算"""
        adjustment = 0.0
        
        # ボラティリティに基づる調整
        if self.market_state['volatility_regime'] == 'high':
            adjustment = -adaptation_rate * sensitivity  # 高ボラ時は閾値を下げる
        elif self.market_state['volatility_regime'] == 'low':
            adjustment = adaptation_rate * sensitivity * 0.3  # 低ボラ時は閾値を上げる
            
        return adjustment
        
    def _calculate_momentum_factor_adjustment(
        self,
        performance_feedback: Dict[str, float],
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """モメンタムファクター調整の計算"""
        adjustment = 0.0
        
        # トレンド強度に基づく調整
        trend_strength = abs(self.market_state['trend_strength'])
        
        if trend_strength > 0.02:  # 強いトレンド
            adjustment = adaptation_rate * sensitivity * 0.5  # モメンタム強化
        elif trend_strength < 0.005:  # 弱いトレンド
            adjustment = -adaptation_rate * sensitivity * 0.3  # モメンタム減少
            
        return adjustment
        
    def _calculate_decay_factor_adjustment(
        self,
        performance_feedback: Dict[str, float],
        adaptation_rate: float,
        sensitivity: float
    ) -> float:
        """減衰ファクター調整の計算"""
        adjustment = 0.0
        
        # 市場の変動性に基づく調整
        if self.market_state['volatility_regime'] == 'high':
            adjustment = -adaptation_rate * sensitivity * 0.5  # 高ボラ時は減衰を弱める
        elif self.market_state['volatility_regime'] == 'low':
            adjustment = adaptation_rate * sensitivity * 0.3  # 低ボラ時は減衰を強める
            
        return adjustment
        
    def _determine_update_reason(
        self,
        param_name: str,
        performance_feedback: Dict[str, float]
    ) -> str:
        """更新理由の決定"""
        reasons = []
        
        # 市場状態に基づく理由
        if self.market_state['volatility_regime'] != 'normal':
            reasons.append(f"volatility_regime_{self.market_state['volatility_regime']}")
            
        if self.market_state['stress_level'] > 0.3:
            reasons.append("market_stress")
            
        # パフォーマンスに基づく理由
        if self._recent_performance_declining():
            reasons.append("performance_decline")
            
        # トレンドに基づく理由
        if abs(self.market_state['trend_strength']) > 0.02:
            reasons.append("trend_change")
            
        return "_".join(reasons) if reasons else "scheduled_update"
        
    def _summarize_market_condition(self) -> str:
        """市場状況のサマリー"""
        conditions = []
        
        conditions.append(f"vol_{self.market_state['volatility_regime']}")
        
        if self.market_state['stress_level'] > 0.3:
            conditions.append("stressed")
            
        if abs(self.market_state['trend_strength']) > 0.02:
            trend_dir = "up" if self.market_state['trend_strength'] > 0 else "down"
            conditions.append(f"trend_{trend_dir}")
            
        conditions.append(f"liquidity_{self.market_state['liquidity_condition']}")
        
        return "_".join(conditions)
        
    def _estimate_performance_impact(self, param_name: str, new_value: float) -> float:
        """パフォーマンス影響度の推定"""
        parameter = self.parameters[param_name]
        change_ratio = abs(new_value - parameter.current_value) / parameter.current_value
        
        # パラメータタイプ別の影響度係数
        impact_coefficients = {
            ParameterType.LEARNING_RATE: 0.02,
            ParameterType.VOLATILITY_SCALING: 0.015,
            ParameterType.RISK_AVERSION: 0.01,
            ParameterType.REBALANCING_THRESHOLD: 0.005,
            ParameterType.MOMENTUM_FACTOR: 0.008,
            ParameterType.DECAY_FACTOR: 0.003
        }
        
        coefficient = impact_coefficients.get(parameter.parameter_type, 0.01)
        return change_ratio * coefficient
        
    def get_current_parameters(self) -> Dict[str, float]:
        """現在のパラメータ値の取得"""
        return {
            name: param.current_value
            for name, param in self.parameters.items()
        }
        
    def get_parameter_summary(self) -> Dict[str, Any]:
        """パラメータサマリーの取得"""
        summary = {}
        
        for name, param in self.parameters.items():
            summary[name] = {
                'current_value': param.current_value,
                'default_value': param.default_value,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'update_count': param.update_count,
                'last_update': param.last_update.isoformat(),
                'deviation_from_default': abs(param.current_value - param.default_value) / param.default_value
            }
            
        # 全体統計
        total_updates = sum(param.update_count for param in self.parameters.values())
        avg_deviation = np.mean([
            abs(param.current_value - param.default_value) / param.default_value
            for param in self.parameters.values()
        ])
        
        summary['overall_statistics'] = {
            'total_parameter_updates': total_updates,
            'average_deviation_from_default': avg_deviation,
            'market_state': self.market_state.copy(),
            'last_adaptation': max(param.last_update for param in self.parameters.values()).isoformat()
        }
        
        return summary
        
    def reset_parameter(self, param_name: str) -> bool:
        """パラメータのリセット"""
        if param_name not in self.parameters:
            return False
            
        parameter = self.parameters[param_name]
        old_value = parameter.current_value
        parameter.current_value = parameter.default_value
        parameter.last_update = datetime.now()
        parameter.update_count += 1
        
        # 更新履歴に記録
        update = ParameterUpdate(
            parameter_name=param_name,
            old_value=old_value,
            new_value=parameter.current_value,
            update_reason="manual_reset",
            market_condition=self._summarize_market_condition(),
            performance_impact=0.0,
            timestamp=datetime.now()
        )
        
        self.update_history.append(update)
        
        self.logger.info(f"Parameter {param_name} reset to default value {parameter.default_value}")
        return True
        
    def get_update_history(self, days: Optional[int] = None) -> List[ParameterUpdate]:
        """更新履歴の取得"""
        if days is None:
            return self.update_history.copy()
            
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            update for update in self.update_history
            if update.timestamp >= cutoff_date
        ]
