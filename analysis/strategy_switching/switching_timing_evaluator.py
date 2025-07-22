"""
Module: Switching Timing Evaluator
File: switching_timing_evaluator.py
Description: 
  5-1-1「戦略切替のタイミング分析ツール」
  切替タイミングの評価と最適化ロジック

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class TimingEvaluationResult:
    """タイミング評価結果"""
    timestamp: datetime
    strategy_from: str
    strategy_to: str
    timing_score: float  # 0-100のスコア
    optimal_timing_offset: int  # 最適タイミングまでの期間（正負で方向）
    confidence_level: float
    evaluation_factors: Dict[str, float]
    market_conditions: Dict[str, Any]
    risk_assessment: Dict[str, float]

@dataclass 
class OptimalTimingPoint:
    """最適タイミングポイント"""
    timestamp: datetime
    recommended_action: str  # 'switch', 'hold', 'prepare'
    from_strategy: Optional[str]
    to_strategy: Optional[str]
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    confidence_score: float
    expected_benefit: float
    risk_level: float
    supporting_factors: List[str]

class TimingConfidence(Enum):
    """タイミング信頼度レベル"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60% 
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%

class TimingUrgency(Enum):
    """タイミング緊急度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SwitchingTimingEvaluator:
    """戦略切替タイミング評価器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Parameters:
            config: 設定辞書
        """
        self.config = config or self._get_default_config()
        self.evaluation_history: List[TimingEvaluationResult] = []
        
        # 評価基準の重み
        self.evaluation_weights = self.config.get('evaluation_criteria', {})
        
        # タイミング評価のための各種閾値
        self.thresholds = self.config.get('timing_thresholds', {})
        
        logger.info("SwitchingTimingEvaluator initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'evaluation_criteria': {
                'performance_weight': 0.30,
                'risk_weight': 0.25,
                'market_conditions_weight': 0.20,
                'momentum_weight': 0.15,
                'volatility_weight': 0.10
            },
            'timing_thresholds': {
                'performance_deterioration': -0.02,
                'risk_spike': 0.05,
                'volatility_threshold': 0.03,
                'momentum_reversal': 0.01,
                'confidence_minimum': 0.6
            },
            'lookback_periods': {
                'short_term': 5,
                'medium_term': 20,
                'long_term': 60
            }
        }

    def evaluate_switching_timing(
        self,
        data: pd.DataFrame,
        current_strategy: str,
        candidate_strategies: List[str],
        timestamp: Optional[datetime] = None
    ) -> TimingEvaluationResult:
        """
        特定時点での切替タイミング評価
        
        Parameters:
            data: 市場データ
            current_strategy: 現在の戦略
            candidate_strategies: 候補戦略リスト
            timestamp: 評価時点（Noneの場合は最新）
            
        Returns:
            タイミング評価結果
        """
        if timestamp is None:
            timestamp = data.index[-1] if not data.empty else datetime.now()
            
        try:
            # データの前処理
            processed_data = self._prepare_evaluation_data(data, timestamp)
            
            # 各評価要素の計算
            evaluation_factors = self._calculate_evaluation_factors(
                processed_data, current_strategy, candidate_strategies, timestamp
            )
            
            # 市場状況の評価
            market_conditions = self._assess_market_conditions(processed_data, timestamp)
            
            # リスク評価
            risk_assessment = self._assess_switching_risk(
                processed_data, current_strategy, candidate_strategies, timestamp
            )
            
            # 総合タイミングスコアの計算
            timing_score = self._calculate_timing_score(
                evaluation_factors, market_conditions, risk_assessment
            )
            
            # 最適タイミングオフセットの計算
            optimal_offset = self._calculate_optimal_timing_offset(
                processed_data, evaluation_factors, timestamp
            )
            
            # 信頼度レベルの決定
            confidence_level = self._determine_confidence_level(
                evaluation_factors, market_conditions, risk_assessment
            )
            
            # 最適候補戦略の選定
            best_candidate = self._select_best_candidate_strategy(
                candidate_strategies, evaluation_factors, risk_assessment
            )
            
            # 評価結果の構築
            result = TimingEvaluationResult(
                timestamp=timestamp,
                strategy_from=current_strategy,
                strategy_to=best_candidate,
                timing_score=timing_score,
                optimal_timing_offset=optimal_offset,
                confidence_level=confidence_level,
                evaluation_factors=evaluation_factors,
                market_conditions=market_conditions,
                risk_assessment=risk_assessment
            )
            
            # 履歴に追加
            self.evaluation_history.append(result)
            
            logger.debug(f"Timing evaluation completed: score={timing_score:.2f}, confidence={confidence_level:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Timing evaluation failed: {e}")
            raise

    def _prepare_evaluation_data(self, data: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """評価用データの準備"""
        if data.empty:
            return data
            
        # タイムスタンプまでのデータを取得
        eval_data = data[data.index <= timestamp].copy()
        
        # 必要な技術指標の計算
        if 'close' in eval_data.columns:
            # リターンの計算
            if 'returns' not in eval_data.columns:
                eval_data['returns'] = eval_data['close'].pct_change()
                
            # ボラティリティ（複数期間）
            for period in [5, 20, 60]:
                eval_data[f'volatility_{period}'] = eval_data['returns'].rolling(period).std()
                
            # モメンタム指標
            eval_data['momentum_5'] = eval_data['close'].pct_change(5)
            eval_data['momentum_20'] = eval_data['close'].pct_change(20)
            
            # 移動平均
            for period in [5, 20, 60]:
                eval_data[f'sma_{period}'] = eval_data['close'].rolling(period).mean()
                
            # RSI
            eval_data['rsi'] = self._calculate_rsi(eval_data['close'], 14)
            
            # ボリンジャーバンド
            bb_data = self._calculate_bollinger_bands(eval_data['close'], 20)
            eval_data = pd.concat([eval_data, bb_data], axis=1)
            
        # 欠損値処理
        eval_data = eval_data.fillna(method='ffill').fillna(0)
        
        return eval_data

    def _calculate_evaluation_factors(
        self,
        data: pd.DataFrame,
        current_strategy: str,
        candidate_strategies: List[str],
        timestamp: datetime
    ) -> Dict[str, float]:
        """評価要素の計算"""
        factors = {}
        
        if data.empty:
            return factors
            
        try:
            current_index = data.index.get_loc(timestamp) if timestamp in data.index else -1
            if current_index < 20:  # 十分なデータがない場合
                return factors
                
            # パフォーマンス要素
            factors['recent_performance'] = self._evaluate_recent_performance(data, current_index)
            factors['trend_strength'] = self._evaluate_trend_strength(data, current_index)
            factors['momentum_quality'] = self._evaluate_momentum_quality(data, current_index)
            
            # ボラティリティ要素
            factors['volatility_level'] = self._evaluate_volatility_level(data, current_index)
            factors['volatility_change'] = self._evaluate_volatility_change(data, current_index)
            
            # 市場効率性要素  
            factors['market_efficiency'] = self._evaluate_market_efficiency(data, current_index)
            factors['mean_reversion_signal'] = self._evaluate_mean_reversion_signal(data, current_index)
            
            # 戦略固有要素
            factors['strategy_suitability'] = self._evaluate_strategy_suitability(
                data, current_strategy, candidate_strategies, current_index
            )
            
            # 正規化（0-1の範囲に調整）
            factors = {k: max(0, min(1, v)) for k, v in factors.items()}
            
        except Exception as e:
            logger.warning(f"Failed to calculate evaluation factors: {e}")
            
        return factors

    def _evaluate_recent_performance(self, data: pd.DataFrame, index: int) -> float:
        """直近パフォーマンスの評価"""
        if 'returns' not in data.columns or index < 10:
            return 0.5  # 中立
            
        recent_returns = data['returns'].iloc[index-10:index].mean()
        return 0.5 + np.tanh(recent_returns * 100) * 0.5  # -1〜1を0〜1にマッピング

    def _evaluate_trend_strength(self, data: pd.DataFrame, index: int) -> float:
        """トレンド強度の評価"""
        if 'close' not in data.columns or index < 20:
            return 0.5
            
        # 短期・長期移動平均の関係
        short_ma = data['close'].iloc[index-5:index].mean()
        long_ma = data['close'].iloc[index-20:index].mean()
        current_price = data['close'].iloc[index]
        
        if long_ma == 0:
            return 0.5
            
        # トレンド強度の計算
        trend_strength = abs((current_price - long_ma) / long_ma)
        return min(1.0, trend_strength * 5)  # 正規化

    def _evaluate_momentum_quality(self, data: pd.DataFrame, index: int) -> float:
        """モメンタム品質の評価"""
        if 'returns' not in data.columns or index < 10:
            return 0.5
            
        returns = data['returns'].iloc[index-10:index]
        
        # モメンタムの一貫性を評価
        positive_days = (returns > 0).sum()
        momentum_consistency = positive_days / len(returns) if len(returns) > 0 else 0.5
        
        # モメンタムの強さ
        momentum_strength = abs(returns.mean()) * 100
        
        return (momentum_consistency + min(1.0, momentum_strength)) / 2

    def _evaluate_volatility_level(self, data: pd.DataFrame, index: int) -> float:
        """ボラティリティレベルの評価"""
        if 'returns' not in data.columns or index < 20:
            return 0.5
            
        volatility = data['returns'].iloc[index-20:index].std()
        
        # ボラティリティが高すぎず、低すぎない範囲を評価
        optimal_vol = 0.02  # 2%程度を最適とする
        vol_score = 1.0 - abs(volatility - optimal_vol) / optimal_vol
        return max(0.0, min(1.0, vol_score))

    def _evaluate_volatility_change(self, data: pd.DataFrame, index: int) -> float:
        """ボラティリティ変化の評価"""
        if 'returns' not in data.columns or index < 40:
            return 0.5
            
        recent_vol = data['returns'].iloc[index-20:index].std()
        past_vol = data['returns'].iloc[index-40:index-20].std()
        
        if past_vol == 0:
            return 0.5
            
        vol_change = (recent_vol - past_vol) / past_vol
        
        # 急激な変化は切替のシグナル
        return min(1.0, abs(vol_change) * 2)

    def _evaluate_market_efficiency(self, data: pd.DataFrame, index: int) -> float:
        """市場効率性の評価"""
        if 'returns' not in data.columns or index < 30:
            return 0.5
            
        returns = data['returns'].iloc[index-30:index]
        
        # 自己相関の計算（効率市場では0に近い）
        autocorr = returns.autocorr(lag=1)
        if pd.isna(autocorr):
            return 0.5
            
        # 自己相関が高い場合、市場は非効率（平均回帰の機会）
        return abs(autocorr)

    def _evaluate_mean_reversion_signal(self, data: pd.DataFrame, index: int) -> float:
        """平均回帰シグナルの評価"""
        if 'close' not in data.columns or index < 20:
            return 0.5
            
        current_price = data['close'].iloc[index]
        mean_price = data['close'].iloc[index-20:index].mean()
        std_price = data['close'].iloc[index-20:index].std()
        
        if std_price == 0:
            return 0.5
            
        # Zスコアの計算
        z_score = abs((current_price - mean_price) / std_price)
        
        # 高いZスコアは平均回帰の機会を示す
        return min(1.0, z_score / 2)

    def _evaluate_strategy_suitability(
        self,
        data: pd.DataFrame,
        current_strategy: str,
        candidate_strategies: List[str],
        index: int
    ) -> float:
        """戦略適合性の評価"""
        # 簡易的な戦略適合性評価
        market_conditions = self._get_simple_market_conditions(data, index)
        
        # 戦略と市場条件の適合性マッピング
        strategy_fitness = {
            'momentum': {
                'trending': 0.9,
                'volatile': 0.6,
                'sideways': 0.3
            },
            'mean_reversion': {
                'trending': 0.3,
                'volatile': 0.7,
                'sideways': 0.9
            },
            'vwap': {
                'trending': 0.7,
                'volatile': 0.8,
                'sideways': 0.6
            },
            'breakout': {
                'trending': 0.8,
                'volatile': 0.9,
                'sideways': 0.4
            }
        }
        
        current_fitness = strategy_fitness.get(current_strategy, {}).get(market_conditions, 0.5)
        
        # 候補戦略の最高適合性
        best_candidate_fitness = max([
            strategy_fitness.get(strategy, {}).get(market_conditions, 0.5)
            for strategy in candidate_strategies
        ], default=0.5)
        
        # 改善余地があるかどうか
        return best_candidate_fitness - current_fitness

    def _get_simple_market_conditions(self, data: pd.DataFrame, index: int) -> str:
        """簡易市場状況判定"""
        if 'returns' not in data.columns or index < 20:
            return 'sideways'
            
        recent_returns = data['returns'].iloc[index-20:index]
        volatility = recent_returns.std()
        trend = recent_returns.mean()
        
        if volatility > 0.03:  # 高ボラティリティ
            return 'volatile'
        elif abs(trend) > 0.001:  # 明確なトレンド
            return 'trending'
        else:
            return 'sideways'

    def _assess_market_conditions(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """市場状況の評価"""
        conditions = {
            'volatility_regime': 'normal',
            'trend_regime': 'sideways',
            'momentum_regime': 'neutral',
            'risk_regime': 'normal'
        }
        
        if data.empty:
            return conditions
            
        try:
            current_index = data.index.get_loc(timestamp) if timestamp in data.index else -1
            if current_index < 20:
                return conditions
                
            # ボラティリティレジーム
            if 'volatility_20' in data.columns:
                current_vol = data['volatility_20'].iloc[current_index]
                avg_vol = data['volatility_20'].iloc[current_index-60:current_index].mean()
                
                if current_vol > avg_vol * 1.5:
                    conditions['volatility_regime'] = 'high'
                elif current_vol < avg_vol * 0.5:
                    conditions['volatility_regime'] = 'low'
                    
            # トレンドレジーム
            if 'momentum_20' in data.columns:
                momentum = data['momentum_20'].iloc[current_index]
                if momentum > 0.02:
                    conditions['trend_regime'] = 'uptrend'
                elif momentum < -0.02:
                    conditions['trend_regime'] = 'downtrend'
                    
            # モメンタムレジーム
            if 'returns' in data.columns:
                recent_returns = data['returns'].iloc[current_index-10:current_index]
                momentum_strength = abs(recent_returns.mean()) / recent_returns.std() if recent_returns.std() > 0 else 0
                
                if momentum_strength > 0.5:
                    conditions['momentum_regime'] = 'strong'
                elif momentum_strength < 0.1:
                    conditions['momentum_regime'] = 'weak'
                    
        except Exception as e:
            logger.warning(f"Market conditions assessment failed: {e}")
            
        return conditions

    def _assess_switching_risk(
        self,
        data: pd.DataFrame,
        current_strategy: str,
        candidate_strategies: List[str],
        timestamp: datetime
    ) -> Dict[str, float]:
        """切替リスクの評価"""
        risk_assessment = {
            'execution_risk': 0.1,  # 執行リスク
            'timing_risk': 0.1,     # タイミングリスク
            'strategy_risk': 0.1,   # 戦略リスク
            'market_risk': 0.1,     # 市場リスク
            'overall_risk': 0.1     # 総合リスク
        }
        
        if data.empty:
            return risk_assessment
            
        try:
            current_index = data.index.get_loc(timestamp) if timestamp in data.index else -1
            if current_index < 10:
                return risk_assessment
                
            # 執行リスク（ボラティリティベース）
            if 'volatility_5' in data.columns:
                recent_vol = data['volatility_5'].iloc[current_index]
                risk_assessment['execution_risk'] = min(0.5, recent_vol * 10)
                
            # タイミングリスク（トレンドの不安定性）
            if 'returns' in data.columns:
                returns = data['returns'].iloc[current_index-10:current_index]
                direction_changes = ((returns > 0) != (returns.shift(1) > 0)).sum()
                risk_assessment['timing_risk'] = direction_changes / len(returns)
                
            # 戦略リスク（戦略間の相関）
            # 簡易的に戦略の類似度をリスクとする
            strategy_similarity = self._calculate_strategy_similarity(current_strategy, candidate_strategies)
            risk_assessment['strategy_risk'] = 1.0 - strategy_similarity  # 類似度が高いほどリスクは低い
            
            # 市場リスク（市場の不安定性）
            if 'volatility_20' in data.columns:
                long_term_vol = data['volatility_20'].iloc[current_index]
                risk_assessment['market_risk'] = min(0.5, long_term_vol * 5)
                
            # 総合リスク
            weights = [0.3, 0.2, 0.2, 0.3]  # 各リスクの重み
            risks = [risk_assessment['execution_risk'], risk_assessment['timing_risk'],
                    risk_assessment['strategy_risk'], risk_assessment['market_risk']]
            risk_assessment['overall_risk'] = sum(w * r for w, r in zip(weights, risks))
            
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            
        return risk_assessment

    def _calculate_strategy_similarity(self, current_strategy: str, candidates: List[str]) -> float:
        """戦略間の類似度計算"""
        # 簡易的な戦略類似度マッピング
        similarity_matrix = {
            ('momentum', 'breakout'): 0.8,
            ('momentum', 'trend_following'): 0.9,
            ('mean_reversion', 'vwap'): 0.7,
            ('mean_reversion', 'range_trading'): 0.8,
            ('vwap', 'volume_weighted'): 0.9
        }
        
        similarities = []
        for candidate in candidates:
            key1 = (current_strategy, candidate)
            key2 = (candidate, current_strategy)
            similarity = similarity_matrix.get(key1, similarity_matrix.get(key2, 0.5))
            similarities.append(similarity)
            
        return np.mean(similarities) if similarities else 0.5

    def _calculate_timing_score(
        self,
        evaluation_factors: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_assessment: Dict[str, float]
    ) -> float:
        """総合タイミングスコアの計算"""
        if not evaluation_factors:
            return 50.0  # 中立スコア
            
        # 重み付けした要素スコア
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, value in evaluation_factors.items():
            weight = self.evaluation_weights.get(f"{factor}_weight", 0.1)
            weighted_score += value * weight
            total_weight += weight
            
        base_score = (weighted_score / total_weight) * 100 if total_weight > 0 else 50.0
        
        # リスク調整
        risk_penalty = risk_assessment.get('overall_risk', 0.1) * 20  # リスクに応じて最大20点減点
        
        # 市場状況調整
        market_adjustment = self._calculate_market_adjustment(market_conditions)
        
        # 最終スコア
        final_score = base_score - risk_penalty + market_adjustment
        return max(0.0, min(100.0, final_score))

    def _calculate_market_adjustment(self, market_conditions: Dict[str, Any]) -> float:
        """市場状況に基づくスコア調整"""
        adjustment = 0.0
        
        # ボラティリティレジーム調整
        vol_regime = market_conditions.get('volatility_regime', 'normal')
        if vol_regime == 'high':
            adjustment += 5  # 高ボラティリティ時は切替機会
        elif vol_regime == 'low':
            adjustment -= 3  # 低ボラティリティ時は切替不要
            
        # トレンドレジーム調整
        trend_regime = market_conditions.get('trend_regime', 'sideways')
        if trend_regime in ['uptrend', 'downtrend']:
            adjustment += 3  # 明確なトレンド時は切替機会
            
        return adjustment

    def _calculate_optimal_timing_offset(
        self,
        data: pd.DataFrame,
        evaluation_factors: Dict[str, float],
        timestamp: datetime
    ) -> int:
        """最適タイミングオフセットの計算"""
        # 簡易的な実装: 評価要素の変化率から最適タイミングを推定
        if data.empty or not evaluation_factors:
            return 0
            
        try:
            current_index = data.index.get_loc(timestamp) if timestamp in data.index else -1
            if current_index < 10:
                return 0
                
            # トレンドの勢いから判定
            if 'returns' in data.columns:
                recent_momentum = data['returns'].iloc[current_index-5:current_index].mean()
                
                if recent_momentum > 0.005:  # 上昇トレンド加速
                    return -2  # 2期間早く切替
                elif recent_momentum < -0.005:  # 下降トレンド加速
                    return -1  # 1期間早く切替
                else:
                    return 1   # 1期間待つ
                    
        except Exception as e:
            logger.warning(f"Optimal timing offset calculation failed: {e}")
            
        return 0

    def _determine_confidence_level(
        self,
        evaluation_factors: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_assessment: Dict[str, float]
    ) -> float:
        """信頼度レベルの決定"""
        if not evaluation_factors:
            return 0.5
            
        # 評価要素の一貫性チェック
        factor_values = list(evaluation_factors.values())
        consistency = 1.0 - np.std(factor_values) if factor_values else 0.5
        
        # リスクレベルに基づく調整
        risk_factor = 1.0 - risk_assessment.get('overall_risk', 0.1)
        
        # 市場条件の安定性
        stability_bonus = 0.1 if market_conditions.get('volatility_regime') == 'normal' else 0.0
        
        confidence = (consistency * 0.5 + risk_factor * 0.4 + stability_bonus)
        return max(0.1, min(1.0, confidence))

    def _select_best_candidate_strategy(
        self,
        candidate_strategies: List[str],
        evaluation_factors: Dict[str, float],
        risk_assessment: Dict[str, float]
    ) -> str:
        """最適候補戦略の選定"""
        if not candidate_strategies:
            return "hold"  # 切替しない
            
        # 簡易的な戦略評価
        strategy_scores = {}
        
        for strategy in candidate_strategies:
            # 戦略固有の適合性評価（簡易版）
            base_score = evaluation_factors.get('strategy_suitability', 0.5)
            
            # 戦略リスク調整
            strategy_risk = self._get_strategy_risk(strategy)
            risk_adjusted_score = base_score * (1 - strategy_risk)
            
            strategy_scores[strategy] = risk_adjusted_score
            
        # 最高スコアの戦略を選択
        best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        return best_strategy

    def _get_strategy_risk(self, strategy: str) -> float:
        """戦略固有のリスク評価"""
        strategy_risks = {
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'vwap': 0.15,
            'breakout': 0.35,
            'trend_following': 0.25
        }
        return strategy_risks.get(strategy, 0.2)

    def identify_optimal_timing_points(
        self,
        data: pd.DataFrame,
        strategy_pairs: List[Tuple[str, str]],
        lookforward_periods: int = 10
    ) -> List[OptimalTimingPoint]:
        """最適タイミングポイントの特定"""
        optimal_points = []
        
        if data.empty:
            return optimal_points
            
        try:
            # 各時点で評価を実行
            for i in range(len(data) - lookforward_periods):
                timestamp = data.index[i]
                
                for from_strategy, to_strategy in strategy_pairs:
                    evaluation = self.evaluate_switching_timing(
                        data.iloc[:i+1], from_strategy, [to_strategy], timestamp
                    )
                    
                    # 高スコアかつ高信頼度の点を最適とする
                    if (evaluation.timing_score > 70 and 
                        evaluation.confidence_level > 0.7):
                        
                        # 緊急度の判定
                        urgency = self._determine_urgency_level(evaluation)
                        
                        # 期待利益の推定
                        expected_benefit = self._estimate_expected_benefit(
                            data, i, from_strategy, to_strategy, lookforward_periods
                        )
                        
                        # 支持要因の特定
                        supporting_factors = self._identify_supporting_factors(evaluation)
                        
                        optimal_point = OptimalTimingPoint(
                            timestamp=timestamp,
                            recommended_action='switch',
                            from_strategy=from_strategy,
                            to_strategy=to_strategy,
                            urgency_level=urgency,
                            confidence_score=evaluation.confidence_level,
                            expected_benefit=expected_benefit,
                            risk_level=evaluation.risk_assessment.get('overall_risk', 0.1),
                            supporting_factors=supporting_factors
                        )
                        
                        optimal_points.append(optimal_point)
                        
        except Exception as e:
            logger.error(f"Optimal timing points identification failed: {e}")
            
        return optimal_points

    def _determine_urgency_level(self, evaluation: TimingEvaluationResult) -> str:
        """緊急度レベルの判定"""
        score = evaluation.timing_score
        risk = evaluation.risk_assessment.get('overall_risk', 0.1)
        
        if score > 90 and risk > 0.3:
            return TimingUrgency.CRITICAL.value
        elif score > 80:
            return TimingUrgency.HIGH.value
        elif score > 65:
            return TimingUrgency.MEDIUM.value
        else:
            return TimingUrgency.LOW.value

    def _estimate_expected_benefit(
        self, 
        data: pd.DataFrame, 
        index: int, 
        from_strategy: str, 
        to_strategy: str, 
        periods: int
    ) -> float:
        """期待利益の推定"""
        # 簡易的な利益推定
        if 'returns' not in data.columns or index + periods >= len(data):
            return 0.0
            
        future_returns = data['returns'].iloc[index:index+periods]
        
        # 戦略に応じた利益推定（簡易版）
        strategy_multipliers = {
            'momentum': 1.2,
            'mean_reversion': 0.8,
            'vwap': 1.0,
            'breakout': 1.5
        }
        
        base_return = future_returns.mean()
        from_multiplier = strategy_multipliers.get(from_strategy, 1.0)
        to_multiplier = strategy_multipliers.get(to_strategy, 1.0)
        
        expected_improvement = base_return * (to_multiplier - from_multiplier)
        return expected_improvement

    def _identify_supporting_factors(self, evaluation: TimingEvaluationResult) -> List[str]:
        """支持要因の特定"""
        factors = []
        
        # 評価要素から支持要因を特定
        for factor, value in evaluation.evaluation_factors.items():
            if value > 0.7:  # 強い支持
                factors.append(f"Strong {factor.replace('_', ' ')}")
            elif value > 0.6:  # 中程度の支持
                factors.append(f"Moderate {factor.replace('_', ' ')}")
                
        # 市場条件から支持要因を特定
        market_conditions = evaluation.market_conditions
        if market_conditions.get('volatility_regime') == 'high':
            factors.append("High volatility environment")
        if market_conditions.get('trend_regime') in ['uptrend', 'downtrend']:
            factors.append("Clear trend direction")
            
        return factors[:5]  # 最大5つの要因を返す

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> pd.DataFrame:
        """ボリンジャーバンド計算"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        return pd.DataFrame({
            'bb_upper': sma + (std * 2),
            'bb_middle': sma,
            'bb_lower': sma - (std * 2)
        })

    def get_timing_confidence_label(self, confidence: float) -> str:
        """信頼度ラベルの取得"""
        if confidence >= 0.8:
            return TimingConfidence.VERY_HIGH.value
        elif confidence >= 0.6:
            return TimingConfidence.HIGH.value
        elif confidence >= 0.4:
            return TimingConfidence.MEDIUM.value
        elif confidence >= 0.2:
            return TimingConfidence.LOW.value
        else:
            return TimingConfidence.VERY_LOW.value

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    evaluator = SwitchingTimingEvaluator()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    test_data['returns'] = test_data['close'].pct_change()
    
    try:
        # タイミング評価の実行
        result = evaluator.evaluate_switching_timing(
            test_data,
            current_strategy='momentum',
            candidate_strategies=['mean_reversion', 'vwap'],
            timestamp=test_data.index[-50]  # 50日前の時点で評価
        )
        
        print("\n=== 戦略切替タイミング評価結果 ===")
        print(f"評価時刻: {result.timestamp}")
        print(f"現在戦略: {result.strategy_from}")
        print(f"推奨戦略: {result.strategy_to}")
        print(f"タイミングスコア: {result.timing_score:.1f}/100")
        print(f"信頼度: {result.confidence_level:.1%}")
        print(f"信頼度レベル: {evaluator.get_timing_confidence_label(result.confidence_level)}")
        print(f"最適タイミングオフセット: {result.optimal_timing_offset}期間")
        
        print("\n評価要素:")
        for factor, value in result.evaluation_factors.items():
            print(f"  {factor}: {value:.3f}")
            
        print("評価成功")
        
    except Exception as e:
        print(f"評価エラー: {e}")
        raise
