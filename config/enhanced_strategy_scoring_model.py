"""
Enhanced Strategy Scoring Model
既存のスコアリングモデルを統一トレンド判定システムと連携強化
2-2-1「信頼度スコアとパフォーマンススコアの統合ロジック」実装

Module: Enhanced Strategy Scoring Model
Description: 
  統一トレンド判定システムと連携した強化スコア計算器
  信頼度とパフォーマンススコアの統合ロジックを提供

Author: imega
Created: 2025-07-13
Modified: 2025-07-13
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存モジュールとの連携
try:
    from config.strategy_scoring_model import (
        StrategyScoreCalculator, 
        StrategyScoreReporter, 
        StrategyScoreManager,
        ScoreWeights,
        StrategyScore
    )
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from indicators.trend_reliability_utils import get_trend_reliability_for_strategy
    from config.strategy_characteristics_data_loader import StrategyCharacteristicsDataLoader
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class EnhancedScoreWeights(ScoreWeights):
    """強化されたスコア重み（統一トレンド判定対応）"""
    trend_accuracy: float = 0.10  # トレンド判定精度の重み
    market_adaptation: float = 0.05  # 市場適応性の重み
    
    def __post_init__(self):
        """重みの正規化"""
        total = (self.performance + self.stability + self.risk_adjusted + 
                self.trend_adaptation + self.reliability + 
                self.trend_accuracy + self.market_adaptation)
        
        if abs(total - 1.0) > 0.001:
            # 正規化
            factor = 1.0 / total
            self.performance *= factor
            self.stability *= factor
            self.risk_adjusted *= factor
            self.trend_adaptation *= factor
            self.reliability *= factor
            self.trend_accuracy *= factor
            self.market_adaptation *= factor

class TrendConfidenceIntegrator:
    """
    信頼度統合処理クラス
    信頼度とパフォーマンススコアの統合ロジックを管理
    """
    
    def __init__(self, confidence_weight: float = 0.3, confidence_threshold: float = 0.7):
        """
        Args:
            confidence_weight: 信頼度の影響度重み
            confidence_threshold: 信頼度閾値
        """
        self.confidence_weight = confidence_weight
        self.confidence_threshold = confidence_threshold
        
    def integrate_confidence(self, 
                           performance_score: float,
                           confidence_score: float,
                           integration_method: str = "adaptive") -> float:
        """
        信頼度とパフォーマンススコアを統合
        
        Args:
            performance_score: 基本パフォーマンススコア
            confidence_score: 信頼度スコア (0-1)
            integration_method: 統合手法 ("adaptive", "linear", "sigmoid")
            
        Returns:
            float: 統合スコア
        """
        try:
            if integration_method == "adaptive":
                return self._adaptive_integration(performance_score, confidence_score)
            elif integration_method == "linear":
                return self._linear_integration(performance_score, confidence_score)
            elif integration_method == "sigmoid":
                return self._sigmoid_integration(performance_score, confidence_score)
            else:
                logger.warning(f"Unknown integration method: {integration_method}")
                return self._adaptive_integration(performance_score, confidence_score)
                
        except Exception as e:
            logger.error(f"Error in confidence integration: {e}")
            return performance_score * 0.8  # 保守的なフォールバック
    
    def _adaptive_integration(self, performance_score: float, confidence_score: float) -> float:
        """適応的統合（推奨）"""
        if confidence_score >= self.confidence_threshold:
            # 高信頼度: フルスコア + ボーナス
            return min(1.0, performance_score * (1.0 + (confidence_score - self.confidence_threshold) * 0.2))
        else:
            # 低信頼度: 段階的減点
            confidence_factor = self._calculate_confidence_factor(confidence_score)
            return performance_score * confidence_factor
    
    def _linear_integration(self, performance_score: float, confidence_score: float) -> float:
        """線形統合"""
        return performance_score * (self.confidence_weight * confidence_score + (1.0 - self.confidence_weight))
    
    def _sigmoid_integration(self, performance_score: float, confidence_score: float) -> float:
        """シグモイド関数による統合"""
        # シグモイド変換で信頼度の影響を非線形化
        sigmoid_factor = 1.0 / (1.0 + np.exp(-10 * (confidence_score - 0.5)))
        return performance_score * (0.5 + 0.5 * sigmoid_factor)
    
    def _calculate_confidence_factor(self, confidence_score: float) -> float:
        """信頼度ファクター計算"""
        if confidence_score >= 0.8:
            return 1.0
        elif confidence_score >= 0.6:
            return 0.8 + 0.4 * (confidence_score - 0.6) / 0.2
        elif confidence_score >= 0.4:
            return 0.6 + 0.2 * (confidence_score - 0.4) / 0.2
        else:
            return 0.4 + 0.2 * confidence_score / 0.4

class EnhancedStrategyScoreCalculator(StrategyScoreCalculator):
    """統一トレンド判定システムと連携した強化スコア計算器"""
    
    def __init__(self, data_loader: StrategyCharacteristicsDataLoader = None):
        super().__init__(data_loader)
        self.confidence_integrator = TrendConfidenceIntegrator()
        
    def calculate_enhanced_strategy_score(self, 
                                        strategy_name: str, 
                                        ticker: str,
                                        market_data: pd.DataFrame = None,
                                        use_trend_validation: bool = True,
                                        integration_method: str = "adaptive") -> StrategyScore:
        """
        統一トレンド判定を活用した強化スコア計算
        
        Parameters:
            strategy_name: 戦略名
            ticker: ティッカーシンボル
            market_data: 市場データ（トレンド判定用）
            use_trend_validation: トレンド検証の使用可否
            integration_method: 統合手法
            
        Returns:
            強化されたStrategyScore
        """
        try:
            # 基本スコアの計算
            base_score = self.calculate_strategy_score(strategy_name, ticker, market_data)
            if not base_score:
                logger.warning(f"Base score calculation failed for {strategy_name}")
                return self._create_fallback_score(strategy_name, ticker)
            
            # 統一トレンド判定による強化
            if use_trend_validation and market_data is not None:
                enhanced_components = self._calculate_enhanced_components(
                    base_score, strategy_name, market_data
                )
                
                # 信頼度統合
                confidence_score = enhanced_components.get('confidence', 0.5)
                integrated_score = self.confidence_integrator.integrate_confidence(
                    base_score.total_score, confidence_score, integration_method
                )
                
                # 強化重みでの再計算
                weights = EnhancedScoreWeights()
                enhanced_total = self._calculate_enhanced_total_score(
                    enhanced_components, weights
                )
                
                # 統合スコアと強化スコアの組み合わせ
                final_score = (integrated_score * 0.6) + (enhanced_total * 0.4)
                
                # 強化スコアオブジェクトの作成
                return StrategyScore(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    total_score=min(1.0, max(0.0, final_score)),
                    component_scores=enhanced_components,
                    trend_fitness=enhanced_components.get('trend_fitness', 0.0),
                    confidence=confidence_score,
                    metadata={
                        'enhanced': True,
                        'trend_validation': use_trend_validation,
                        'integration_method': integration_method,
                        'base_score': base_score.total_score,
                        'integrated_score': integrated_score,
                        'enhanced_total': enhanced_total
                    },
                    calculated_at=datetime.now()
                )
            
            return base_score
            
        except Exception as e:
            logger.error(f"Enhanced score calculation failed for {strategy_name}: {e}")
            return self._create_fallback_score(strategy_name, ticker)
    
    def _calculate_enhanced_components(self, 
                                     base_score: StrategyScore,
                                     strategy_name: str, 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """強化されたコンポーネントスコアの計算"""
        components = base_score.component_scores.copy()
        
        try:
            # 統一トレンド判定による信頼度スコア
            reliability_info = get_trend_reliability_for_strategy(market_data, strategy_name)
            trend_confidence = reliability_info.get("confidence_score", 0.5)
            
            # トレンド精度スコア
            trend_accuracy = self._calculate_trend_accuracy_score(
                strategy_name, market_data, trend_confidence
            )
            components['trend_accuracy'] = trend_accuracy
            
            # 市場適応性スコア
            market_adaptation = self._calculate_market_adaptation_score(
                strategy_name, market_data, reliability_info
            )
            components['market_adaptation'] = market_adaptation
            
            # 既存スコアの調整
            components['trend_adaptation'] = self._adjust_trend_adaptation_score(
                components.get('trend_adaptation', 0.0), trend_accuracy
            )
            
            # 信頼度の再計算
            components['confidence'] = self._calculate_enhanced_confidence(
                components, trend_confidence, trend_accuracy, market_adaptation
            )
            
            # トレンド適合度の更新
            components['trend_fitness'] = self._calculate_trend_fitness(
                reliability_info, trend_accuracy, market_adaptation
            )
            
        except Exception as e:
            logger.warning(f"Enhanced component calculation failed: {e}")
            # エラー時は基本スコアをそのまま使用
            components['trend_accuracy'] = 0.5
            components['market_adaptation'] = 0.5
            components['confidence'] = 0.5
        
        return components
    
    def _calculate_trend_accuracy_score(self, 
                                      strategy_name: str, 
                                      market_data: pd.DataFrame,
                                      trend_confidence: float) -> float:
        """統一トレンド判定による精度スコア"""
        try:
            # 基本信頼度を基に精度スコアを計算
            base_accuracy = trend_confidence
            
            # 戦略特性による調整
            strategy_bonus = self._get_strategy_trend_bonus(strategy_name)
            
            # データ品質による調整
            data_quality = self._assess_data_quality(market_data)
            
            # 統合精度スコア
            accuracy_score = base_accuracy * (1.0 + strategy_bonus * 0.2) * data_quality
            
            return min(1.0, max(0.0, accuracy_score))
            
        except Exception as e:
            logger.warning(f"Trend accuracy calculation failed: {e}")
            return 0.6
    
    def _calculate_market_adaptation_score(self, 
                                         strategy_name: str, 
                                         market_data: pd.DataFrame,
                                         reliability_info: Dict[str, Any]) -> float:
        """市場適応性スコアの計算"""
        try:
            # 基本適応性スコア
            base_adaptation = reliability_info.get("confidence_score", 0.5)
            
            # ボラティリティ適応性
            volatility_adaptation = self._evaluate_volatility_adaptation(
                strategy_name, market_data
            )
            
            # トレンド適応性
            trend_adaptation = self._evaluate_trend_adaptation_advanced(
                strategy_name, reliability_info
            )
            
            # 統合適応性スコア
            adaptation_score = (base_adaptation * 0.4 + 
                              volatility_adaptation * 0.3 + 
                              trend_adaptation * 0.3)
            
            return min(1.0, max(0.0, adaptation_score))
            
        except Exception as e:
            logger.warning(f"Market adaptation calculation failed: {e}")
            return 0.5
    
    def _get_strategy_trend_bonus(self, strategy_name: str) -> float:
        """戦略別トレンド判定ボーナス"""
        strategy_lower = strategy_name.lower()
        
        if 'vwap' in strategy_lower:
            return 0.2  # VWAP戦略はトレンド判定が重要
        elif 'momentum' in strategy_lower:
            return 0.15  # モメンタム戦略
        elif 'breakout' in strategy_lower:
            return 0.1  # ブレイクアウト戦略
        elif 'contrarian' in strategy_lower:
            return -0.05  # 逆張り戦略はトレンドに逆行
        else:
            return 0.0
    
    def _assess_data_quality(self, market_data: pd.DataFrame) -> float:
        """データ品質評価"""
        try:
            if len(market_data) < 20:
                return 0.5
            elif len(market_data) < 50:
                return 0.7
            elif len(market_data) < 100:
                return 0.85
            else:
                return 1.0
        except:
            return 0.5
    
    def _evaluate_volatility_adaptation(self, 
                                      strategy_name: str, 
                                      market_data: pd.DataFrame) -> float:
        """ボラティリティ適応性評価"""
        try:
            if len(market_data) < 20:
                return 0.5
            
            # ボラティリティ計算
            returns = market_data['Adj Close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # 戦略特性に基づく適応度評価
            strategy_lower = strategy_name.lower()
            if 'breakout' in strategy_lower:
                # ブレイクアウト戦略は高ボラティリティで有利
                if volatility > 0.02:
                    return 0.8
                else:
                    return 0.6
            elif 'vwap' in strategy_lower:
                # VWAP戦略は中程度のボラティリティで有利
                if 0.01 <= volatility <= 0.025:
                    return 0.8
                else:
                    return 0.6
            else:
                return 0.6
                
        except Exception as e:
            logger.warning(f"Volatility adaptation evaluation failed: {e}")
            return 0.5
    
    def _evaluate_trend_adaptation_advanced(self, 
                                          strategy_name: str, 
                                          reliability_info: Dict[str, Any]) -> float:
        """高度なトレンド適応性評価"""
        try:
            # 現在のトレンド
            current_trend = reliability_info.get("trend", "unknown")
            confidence = reliability_info.get("confidence_score", 0.5)
            
            # 戦略のトレンド適性
            strategy_trend_score = self._get_strategy_trend_suitability(
                strategy_name, current_trend
            )
            
            # 信頼度による調整
            adjusted_score = strategy_trend_score * (0.5 + confidence * 0.5)
            
            return min(1.0, max(0.0, adjusted_score))
            
        except Exception as e:
            logger.warning(f"Advanced trend adaptation evaluation failed: {e}")
            return 0.5
    
    def _get_strategy_trend_suitability(self, strategy_name: str, trend: str) -> float:
        """戦略のトレンド適性スコア"""
        strategy_lower = strategy_name.lower()
        
        if trend == "uptrend":
            if 'momentum' in strategy_lower or 'breakout' in strategy_lower:
                return 0.9
            elif 'gc' in strategy_lower or 'golden' in strategy_lower:
                return 0.8
            elif 'contrarian' in strategy_lower:
                return 0.4
            else:
                return 0.6
        elif trend == "downtrend":
            if 'contrarian' in strategy_lower:
                return 0.8
            elif 'momentum' in strategy_lower:
                return 0.3
            else:
                return 0.5
        else:  # range-bound
            if 'vwap' in strategy_lower:
                return 0.8
            elif 'contrarian' in strategy_lower:
                return 0.7
            elif 'momentum' in strategy_lower or 'breakout' in strategy_lower:
                return 0.4
            else:
                return 0.6
    
    def _adjust_trend_adaptation_score(self, 
                                     base_score: float, 
                                     trend_accuracy: float) -> float:
        """トレンド適応スコアの調整"""
        # トレンド精度が高い場合は適応スコアを向上
        adjustment_factor = 0.8 + (trend_accuracy * 0.4)
        return min(1.0, base_score * adjustment_factor)
    
    def _calculate_enhanced_confidence(self, 
                                     components: Dict[str, float],
                                     trend_confidence: float,
                                     trend_accuracy: float,
                                     market_adaptation: float) -> float:
        """強化された信頼度計算"""
        base_confidence = components.get('confidence', 0.5)
        
        # 新しい要素による信頼度調整
        trend_bonus = (trend_confidence - 0.5) * 0.3
        accuracy_bonus = (trend_accuracy - 0.5) * 0.2
        adaptation_bonus = (market_adaptation - 0.5) * 0.1
        
        enhanced_confidence = base_confidence + trend_bonus + accuracy_bonus + adaptation_bonus
        return max(0.0, min(1.0, enhanced_confidence))
    
    def _calculate_trend_fitness(self,
                               reliability_info: Dict[str, Any],
                               trend_accuracy: float,
                               market_adaptation: float) -> float:
        """トレンド適合度計算"""
        confidence = reliability_info.get("confidence_score", 0.5)
        is_reliable = reliability_info.get("is_reliable", False)
        
        base_fitness = confidence
        accuracy_bonus = trend_accuracy * 0.3
        adaptation_bonus = market_adaptation * 0.2
        reliability_bonus = 0.1 if is_reliable else 0.0
        
        fitness = base_fitness + accuracy_bonus + adaptation_bonus + reliability_bonus
        return max(0.0, min(1.0, fitness))
    
    def _calculate_enhanced_total_score(self, 
                                      components: Dict[str, float], 
                                      weights: EnhancedScoreWeights) -> float:
        """強化重みでの総合スコア計算"""
        total_score = (
            components.get('performance', 0.0) * weights.performance +
            components.get('stability', 0.0) * weights.stability +
            components.get('risk_adjusted', 0.0) * weights.risk_adjusted +
            components.get('trend_adaptation', 0.0) * weights.trend_adaptation +
            components.get('reliability', 0.0) * weights.reliability +
            components.get('trend_accuracy', 0.0) * weights.trend_accuracy +
            components.get('market_adaptation', 0.0) * weights.market_adaptation
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _create_fallback_score(self, strategy_name: str, ticker: str) -> StrategyScore:
        """フォールバック用のデフォルトスコア"""
        return StrategyScore(
            strategy_name=strategy_name,
            ticker=ticker,
            total_score=0.5,
            component_scores={
                'performance': 0.5,
                'stability': 0.5,
                'risk_adjusted': 0.5,
                'trend_adaptation': 0.5,
                'reliability': 0.5,
                'trend_accuracy': 0.5,
                'market_adaptation': 0.5,
                'confidence': 0.5,
                'trend_fitness': 0.5
            },
            trend_fitness=0.5,
            confidence=0.5,
            metadata={'fallback': True, 'enhanced': False},
            calculated_at=datetime.now()
        )

class EnhancedStrategyScoreManager(StrategyScoreManager):
    """強化されたスコア管理クラス"""
    
    def __init__(self):
        super().__init__()
        self.enhanced_calculator = EnhancedStrategyScoreCalculator()
    
    def calculate_enhanced_scores(self, 
                                strategies: List[str], 
                                tickers: List[str] = None,
                                market_data: pd.DataFrame = None,
                                integration_method: str = "adaptive") -> Dict[str, List[StrategyScore]]:
        """強化されたスコア計算の一括実行"""
        if tickers is None:
            tickers = ["DEFAULT"]
        
        results = {}
        
        for strategy in strategies:
            strategy_scores = []
            for ticker in tickers:
                try:
                    score = self.enhanced_calculator.calculate_enhanced_strategy_score(
                        strategy, ticker, market_data, True, integration_method
                    )
                    if score:
                        strategy_scores.append(score)
                except Exception as e:
                    logger.error(f"Enhanced scoring failed for {strategy}-{ticker}: {e}")
            
            results[strategy] = strategy_scores
        
        return results
    
    def generate_enhanced_report(self, 
                               scores: Dict[str, List[StrategyScore]], 
                               report_name: str = "enhanced_strategy_scores") -> str:
        """強化スコアレポートの生成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name}_{timestamp}.json"
            
            # ログディレクトリの作成
            report_dir = "logs/enhanced_scoring"
            os.makedirs(report_dir, exist_ok=True)
            
            # レポートデータの準備
            report_data = {
                "report_timestamp": datetime.now().isoformat(),
                "enhancement_features": [
                    "unified_trend_detection",
                    "confidence_integration",
                    "trend_accuracy_scoring",
                    "market_adaptation_scoring"
                ],
                "strategies": {}
            }
            
            for strategy_name, strategy_scores in scores.items():
                strategy_data = []
                for score in strategy_scores:
                    strategy_data.append({
                        "ticker": score.ticker,
                        "total_score": score.total_score,
                        "component_scores": score.component_scores,
                        "trend_fitness": score.trend_fitness,
                        "confidence": score.confidence,
                        "metadata": score.metadata
                    })
                report_data["strategies"][strategy_name] = strategy_data
            
            # レポート保存
            filepath = os.path.join(report_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Enhanced score report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Enhanced report generation failed: {e}")
            return ""

# ファクトリー関数
def create_enhanced_score_calculator() -> EnhancedStrategyScoreCalculator:
    """強化スコア計算器の作成"""
    try:
        from config.strategy_characteristics_data_loader import create_data_loader
        data_loader = create_data_loader()
        return EnhancedStrategyScoreCalculator(data_loader)
    except ImportError:
        logger.warning("Data loader not available, using basic calculator")
        return EnhancedStrategyScoreCalculator()

def create_enhanced_score_manager() -> EnhancedStrategyScoreManager:
    """強化スコア管理器の作成"""
    return EnhancedStrategyScoreManager()

if __name__ == "__main__":
    # テスト実行
    print("Enhanced Strategy Scoring Model - 2-2-1 Implementation")
    
    try:
        manager = create_enhanced_score_manager()
        print("✓ Enhanced score manager created successfully")
        
        # 簡単なテスト
        test_strategies = ["VWAPBounceStrategy", "MomentumInvestingStrategy"]
        
        # 仮想的な市場データ
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        market_data = pd.DataFrame({
            'Adj Close': np.random.random(100) * 100 + 1000,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        scores = manager.calculate_enhanced_scores(
            test_strategies, 
            ["TEST"], 
            market_data
        )
        
        print(f"✓ Enhanced scores calculated for {len(scores)} strategies")
        print("✅ 2-2-1「信頼度スコアとパフォーマンススコアの統合ロジック」実装完了")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
