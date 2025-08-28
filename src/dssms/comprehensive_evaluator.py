"""
DSSMS 総合評価システム
Task 3.4: 多次元パフォーマンス評価とスコアリング
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from .performance_target_manager import TargetResult, AchievementLevel, TargetPhase

@dataclass
class DimensionScore:
    """評価次元のスコア情報"""
    dimension_name: str
    score: float
    weight: float
    weighted_score: float
    status: str
    metrics_count: int
    details: Dict[str, Any]

@dataclass
class ComprehensiveEvaluationResult:
    """総合評価結果"""
    overall_score: float
    dimension_scores: List[DimensionScore]
    risk_adjusted_score: float
    confidence_level: float
    evaluation_timestamp: datetime
    recommendations: List[str]
    alerts: List[str]

class ComprehensiveEvaluator:
    """多次元パフォーマンス評価システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dimension_weights = self._initialize_dimension_weights()
        self.scoring_thresholds = self._initialize_scoring_thresholds()
        
    def _initialize_dimension_weights(self) -> Dict[str, float]:
        """評価次元の重みを初期化"""
        return {
            "profitability": 0.35,      # 収益性
            "risk_management": 0.25,    # リスク管理
            "stability": 0.20,          # 安定性
            "efficiency": 0.15,         # 効率性
            "adaptability": 0.05        # 適応性
        }
    
    def _initialize_scoring_thresholds(self) -> Dict[str, Dict[str, float]]:
        """スコアリング閾値を初期化"""
        return {
            "excellent": {"min": 85.0, "color": "green"},
            "good": {"min": 70.0, "color": "blue"},
            "acceptable": {"min": 55.0, "color": "yellow"},
            "needs_improvement": {"min": 40.0, "color": "orange"},
            "critical": {"min": 0.0, "color": "red"}
        }
    
    def evaluate_comprehensive_performance(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any],
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> ComprehensiveEvaluationResult:
        """総合パフォーマンス評価の実行"""
        
        try:
            # 各次元のスコア計算
            dimension_scores = self._calculate_dimension_scores(
                target_results, performance_data, risk_metrics
            )
            
            # 総合スコアの計算
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # リスク調整後スコア
            risk_adjusted_score = self._calculate_risk_adjusted_score(
                overall_score, risk_metrics
            )
            
            # 信頼度レベル
            confidence_level = self._calculate_confidence_level(
                target_results, performance_data
            )
            
            # 推奨事項とアラート
            recommendations = self._generate_recommendations(dimension_scores)
            alerts = self._generate_alerts(dimension_scores, risk_metrics)
            
            return ComprehensiveEvaluationResult(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                risk_adjusted_score=risk_adjusted_score,
                confidence_level=confidence_level,
                evaluation_timestamp=datetime.now(),
                recommendations=recommendations,
                alerts=alerts
            )
            
        except Exception as e:
            self.logger.error(f"総合評価エラー: {e}")
            return self._create_error_result()
    
    def _calculate_dimension_scores(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any],
        risk_metrics: Optional[Dict[str, float]]
    ) -> List[DimensionScore]:
        """各評価次元のスコア計算"""
        
        dimension_scores = []
        
        # 収益性次元
        profitability_score = self._calculate_profitability_score(
            target_results, performance_data
        )
        dimension_scores.append(profitability_score)
        
        # リスク管理次元
        risk_score = self._calculate_risk_management_score(
            target_results, risk_metrics
        )
        dimension_scores.append(risk_score)
        
        # 安定性次元
        stability_score = self._calculate_stability_score(
            target_results, performance_data
        )
        dimension_scores.append(stability_score)
        
        # 効率性次元
        efficiency_score = self._calculate_efficiency_score(
            target_results, performance_data
        )
        dimension_scores.append(efficiency_score)
        
        # 適応性次元
        adaptability_score = self._calculate_adaptability_score(
            target_results, performance_data
        )
        dimension_scores.append(adaptability_score)
        
        return dimension_scores
    
    def _calculate_profitability_score(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any]
    ) -> DimensionScore:
        """収益性スコアの計算"""
        
        profitability_metrics = [
            "total_return", "annual_return", "portfolio_value", 
            "profit_factor", "average_win"
        ]
        
        relevant_results = [
            r for r in target_results 
            if r.metric_name in profitability_metrics
        ]
        
        if not relevant_results:
            return DimensionScore(
                dimension_name="profitability",
                score=0.0,
                weight=self.dimension_weights["profitability"],
                weighted_score=0.0,
                status="no_data",
                metrics_count=0,
                details={}
            )
        
        # 達成レベルベースのスコア計算
        achievement_scores = {
            AchievementLevel.STRETCH: 100.0,
            AchievementLevel.TARGET: 80.0,
            AchievementLevel.MINIMUM: 60.0,
            AchievementLevel.FAILED: 20.0
        }
        
        total_score = sum(
            achievement_scores[result.achievement_level] 
            for result in relevant_results
        )
        avg_score = total_score / len(relevant_results)
        
        # 重要指標への重み付け
        if any(r.metric_name == "total_return" for r in relevant_results):
            total_return_result = next(
                r for r in relevant_results if r.metric_name == "total_return"
            )
            return_weight = 0.4
            avg_score = (avg_score * (1 - return_weight) + 
                        achievement_scores[total_return_result.achievement_level] * return_weight)
        
        weight = self.dimension_weights["profitability"]
        status = self._determine_dimension_status(avg_score)
        
        return DimensionScore(
            dimension_name="profitability",
            score=avg_score,
            weight=weight,
            weighted_score=avg_score * weight,
            status=status,
            metrics_count=len(relevant_results),
            details={
                "metrics": [r.metric_name for r in relevant_results],
                "avg_achievement": avg_score
            }
        )
    
    def _calculate_risk_management_score(
        self, 
        target_results: List[TargetResult],
        risk_metrics: Optional[Dict[str, float]]
    ) -> DimensionScore:
        """リスク管理スコアの計算"""
        
        risk_management_metrics = [
            "max_drawdown", "value_at_risk", "sharpe_ratio", 
            "sortino_ratio", "risk_return_ratio"
        ]
        
        relevant_results = [
            r for r in target_results 
            if r.metric_name in risk_management_metrics
        ]
        
        # 基本スコア（目標達成ベース）
        base_score = 50.0
        if relevant_results:
            achievement_scores = {
                AchievementLevel.STRETCH: 100.0,
                AchievementLevel.TARGET: 80.0,
                AchievementLevel.MINIMUM: 60.0,
                AchievementLevel.FAILED: 20.0
            }
            base_score = sum(
                achievement_scores[r.achievement_level] for r in relevant_results
            ) / len(relevant_results)
        
        # リスク指標による調整
        if risk_metrics:
            risk_adjustment = self._calculate_risk_adjustment(risk_metrics)
            adjusted_score = base_score * risk_adjustment
        else:
            adjusted_score = base_score
        
        weight = self.dimension_weights["risk_management"]
        status = self._determine_dimension_status(adjusted_score)
        
        return DimensionScore(
            dimension_name="risk_management",
            score=adjusted_score,
            weight=weight,
            weighted_score=adjusted_score * weight,
            status=status,
            metrics_count=len(relevant_results),
            details={
                "base_score": base_score,
                "risk_adjustment": risk_metrics.get("adjustment_factor", 1.0) if risk_metrics else 1.0,
                "metrics": [r.metric_name for r in relevant_results]
            }
        )
    
    def _calculate_stability_score(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any]
    ) -> DimensionScore:
        """安定性スコアの計算"""
        
        stability_metrics = [
            "volatility", "consistency_ratio", "win_rate", 
            "switching_success_rate", "trade_frequency"
        ]
        
        relevant_results = [
            r for r in target_results 
            if r.metric_name in stability_metrics
        ]
        
        base_score = 50.0
        if relevant_results:
            achievement_scores = {
                AchievementLevel.STRETCH: 100.0,
                AchievementLevel.TARGET: 80.0,
                AchievementLevel.MINIMUM: 60.0,
                AchievementLevel.FAILED: 20.0
            }
            base_score = sum(
                achievement_scores[r.achievement_level] for r in relevant_results
            ) / len(relevant_results)
        
        # 戦略切り替え成功率による調整
        switching_bonus = 0.0
        if "switching_success_rate" in performance_data:
            switch_rate = performance_data["switching_success_rate"]
            if switch_rate > 0.7:
                switching_bonus = 10.0
            elif switch_rate > 0.5:
                switching_bonus = 5.0
        
        final_score = min(100.0, base_score + switching_bonus)
        weight = self.dimension_weights["stability"]
        status = self._determine_dimension_status(final_score)
        
        return DimensionScore(
            dimension_name="stability",
            score=final_score,
            weight=weight,
            weighted_score=final_score * weight,
            status=status,
            metrics_count=len(relevant_results),
            details={
                "base_score": base_score,
                "switching_bonus": switching_bonus,
                "metrics": [r.metric_name for r in relevant_results]
            }
        )
    
    def _calculate_efficiency_score(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any]
    ) -> DimensionScore:
        """効率性スコアの計算"""
        
        efficiency_metrics = [
            "trades_per_day", "execution_speed", "cost_efficiency",
            "capital_utilization", "information_ratio"
        ]
        
        relevant_results = [
            r for r in target_results 
            if r.metric_name in efficiency_metrics
        ]
        
        base_score = 60.0  # 効率性のデフォルトスコア
        if relevant_results:
            achievement_scores = {
                AchievementLevel.STRETCH: 100.0,
                AchievementLevel.TARGET: 80.0,
                AchievementLevel.MINIMUM: 60.0,
                AchievementLevel.FAILED: 30.0
            }
            base_score = sum(
                achievement_scores[r.achievement_level] for r in relevant_results
            ) / len(relevant_results)
        
        weight = self.dimension_weights["efficiency"]
        status = self._determine_dimension_status(base_score)
        
        return DimensionScore(
            dimension_name="efficiency",
            score=base_score,
            weight=weight,
            weighted_score=base_score * weight,
            status=status,
            metrics_count=len(relevant_results),
            details={
                "metrics": [r.metric_name for r in relevant_results]
            }
        )
    
    def _calculate_adaptability_score(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any]
    ) -> DimensionScore:
        """適応性スコアの計算"""
        
        # 市場環境変化への適応度を評価
        adaptability_score = 70.0  # デフォルト値
        
        # 複数戦略の統合効果
        if "strategy_correlation" in performance_data:
            correlation = performance_data["strategy_correlation"]
            if correlation < 0.3:  # 低相関は良い多様化
                adaptability_score += 15.0
            elif correlation < 0.6:
                adaptability_score += 5.0
        
        # 動的パラメータ調整の効果
        if "parameter_adaptation_rate" in performance_data:
            adaptation_rate = performance_data["parameter_adaptation_rate"]
            if adaptation_rate > 0.8:
                adaptability_score += 10.0
            elif adaptation_rate > 0.6:
                adaptability_score += 5.0
        
        adaptability_score = min(100.0, adaptability_score)
        weight = self.dimension_weights["adaptability"]
        status = self._determine_dimension_status(adaptability_score)
        
        return DimensionScore(
            dimension_name="adaptability",
            score=adaptability_score,
            weight=weight,
            weighted_score=adaptability_score * weight,
            status=status,
            metrics_count=len([r for r in target_results if "adapt" in r.metric_name.lower()]),
            details={
                "strategy_diversification": performance_data.get("strategy_correlation", "N/A"),
                "parameter_adaptation": performance_data.get("parameter_adaptation_rate", "N/A")
            }
        )
    
    def _calculate_overall_score(self, dimension_scores: List[DimensionScore]) -> float:
        """総合スコアの計算"""
        return sum(score.weighted_score for score in dimension_scores)
    
    def _calculate_risk_adjusted_score(
        self, 
        overall_score: float, 
        risk_metrics: Optional[Dict[str, float]]
    ) -> float:
        """リスク調整後スコアの計算"""
        if not risk_metrics:
            return overall_score
        
        risk_adjustment = self._calculate_risk_adjustment(risk_metrics)
        return overall_score * risk_adjustment
    
    def _calculate_risk_adjustment(self, risk_metrics: Dict[str, float]) -> float:
        """リスク調整係数の計算"""
        adjustment = 1.0
        
        # ドローダウンによる調整
        if "max_drawdown" in risk_metrics:
            dd = risk_metrics["max_drawdown"]
            if dd > 50.0:
                adjustment *= 0.7
            elif dd > 30.0:
                adjustment *= 0.85
            elif dd > 20.0:
                adjustment *= 0.95
        
        # VaRによる調整
        if "value_at_risk" in risk_metrics:
            var = risk_metrics["value_at_risk"]
            if var > 10.0:
                adjustment *= 0.8
            elif var > 5.0:
                adjustment *= 0.9
        
        return max(0.5, adjustment)  # 最小50%まで調整
    
    def _calculate_confidence_level(
        self, 
        target_results: List[TargetResult],
        performance_data: Dict[str, Any]
    ) -> float:
        """評価の信頼度レベルの計算"""
        
        # データ完全性
        data_completeness = len(target_results) / 15.0  # 想定15指標
        data_completeness = min(1.0, data_completeness)
        
        # 評価期間の適切性
        period_factor = 1.0
        if "evaluation_period_days" in performance_data:
            days = performance_data["evaluation_period_days"]
            if days < 30:
                period_factor = 0.6
            elif days < 60:
                period_factor = 0.8
        
        # 統計的有意性
        statistical_factor = 1.0
        if "trade_count" in performance_data:
            trades = performance_data["trade_count"]
            if trades < 10:
                statistical_factor = 0.5
            elif trades < 30:
                statistical_factor = 0.7
        
        confidence = data_completeness * period_factor * statistical_factor
        return max(0.1, min(1.0, confidence))
    
    def _determine_dimension_status(self, score: float) -> str:
        """次元スコアのステータス判定"""
        for status, threshold in self.scoring_thresholds.items():
            if score >= threshold["min"]:
                return status
        return "critical"
    
    def _generate_recommendations(self, dimension_scores: List[DimensionScore]) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []
        
        for score in dimension_scores:
            if score.score < 60.0:
                if score.dimension_name == "profitability":
                    recommendations.append("収益性改善のため戦略パラメータの最適化を検討してください")
                elif score.dimension_name == "risk_management":
                    recommendations.append("リスク管理強化のため位置サイズやストップロス設定を見直してください")
                elif score.dimension_name == "stability":
                    recommendations.append("安定性向上のため戦略の多様化や切り替えロジックの改善を検討してください")
                elif score.dimension_name == "efficiency":
                    recommendations.append("効率性向上のため取引頻度や執行コストの最適化を検討してください")
                elif score.dimension_name == "adaptability":
                    recommendations.append("適応性向上のため動的パラメータ調整機能の導入を検討してください")
        
        return recommendations
    
    def _generate_alerts(
        self, 
        dimension_scores: List[DimensionScore],
        risk_metrics: Optional[Dict[str, float]]
    ) -> List[str]:
        """アラートの生成"""
        alerts = []
        
        # クリティカルな次元スコア
        for score in dimension_scores:
            if score.score < 40.0:
                alerts.append(f"【緊急】{score.dimension_name}が危険レベルです（スコア: {score.score:.1f}）")
        
        # リスク指標アラート
        if risk_metrics:
            if risk_metrics.get("max_drawdown", 0) > 50.0:
                alerts.append("【警告】最大ドローダウンが50%を超えています")
            if risk_metrics.get("value_at_risk", 0) > 15.0:
                alerts.append("【警告】VaRが高リスクレベルに達しています")
        
        return alerts
    
    def _create_error_result(self) -> ComprehensiveEvaluationResult:
        """エラー時のデフォルト結果"""
        return ComprehensiveEvaluationResult(
            overall_score=0.0,
            dimension_scores=[],
            risk_adjusted_score=0.0,
            confidence_level=0.0,
            evaluation_timestamp=datetime.now(),
            recommendations=["システムエラーのため評価を完了できませんでした"],
            alerts=["評価システムに問題が発生しています"]
        )
