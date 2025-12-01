"""
DSSMS Phase 2 Task 2.2: 適応的スコア計算器
市場状況に応じた動的スコア調整システム

機能:
- 市場状況分析
- 適応的スコア調整
- パフォーマンスベース学習
- 更新頻度管理
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from enum import Enum
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from .hierarchical_ranking_system import RankingScore
from config.logger_config import setup_logger

class PerformanceMetric(Enum):
    """パフォーマンス指標"""
    RETURN_ACCURACY = "return_accuracy"
    VOLATILITY_PREDICTION = "volatility_prediction"
    TREND_DETECTION = "trend_detection"
    RISK_ADJUSTMENT = "risk_adjustment"

class AdaptiveScoreCalculator:
    """
    適応的スコア計算器
    
    機能:
    - 市場状況別スコア調整
    - パフォーマンス学習
    - 動的重み調整
    - 予測精度向上
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初期化"""
        self.logger = setup_logger('dssms.adaptive_score')
        self.config = config
        
        # 学習パラメータ
        self.learning_rate = config.get('learning_rate', 0.1)
        self.performance_lookback_days = config.get('performance_lookback_days', 30)
        self.update_frequency_minutes = config.get('update_frequency_minutes', 15)
        
        # 適応的重み（初期値）
        self.adaptive_weights = {
            'market_momentum': 0.3,
            'volatility_regime': 0.25,
            'sector_rotation': 0.2,
            'risk_sentiment': 0.15,
            'technical_convergence': 0.1
        }
        
        # パフォーマンス履歴
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # 市場状況履歴
        self.market_condition_history: List[Dict[str, Any]] = []
        
        # 最終更新時刻
        self.last_update = datetime.now()
        
        self.logger.info("AdaptiveScoreCalculator initialized")
    
    async def calculate_adaptive_bonus(self, symbol: str, 
                                     hierarchical_result: RankingScore,
                                     market_condition: Any) -> float:
        """
        適応的ボーナススコア計算
        
        Args:
            symbol: 銘柄コード
            hierarchical_result: 階層ランキング結果
            market_condition: 市場状況
            
        Returns:
            float: 適応的ボーナススコア
        """
        try:
            # 基本スコア取得
            base_score = hierarchical_result.total_score
            
            # 市場状況分析
            market_factors = await self._analyze_market_factors(symbol, market_condition)
            
            # セクター分析
            sector_factors = await self._analyze_sector_factors(symbol)
            
            # テクニカル収束分析
            technical_factors = await self._analyze_technical_convergence(symbol)
            
            # リスクセンチメント分析
            risk_factors = await self._analyze_risk_sentiment(symbol)
            
            # 適応的重み適用
            weighted_factors = self._apply_adaptive_weights({
                'market_momentum': market_factors.get('momentum', 0.0),
                'volatility_regime': market_factors.get('volatility_regime', 0.0),
                'sector_rotation': sector_factors.get('rotation_score', 0.0),
                'risk_sentiment': risk_factors.get('sentiment_score', 0.0),
                'technical_convergence': technical_factors.get('convergence_score', 0.0)
            })
            
            # ボーナススコア計算
            adaptive_bonus = sum(weighted_factors.values()) * self.config.get('bonus_multiplier', 0.1)
            
            # パフォーマンス学習更新
            await self._update_performance_learning(symbol, hierarchical_result, adaptive_bonus)
            
            return adaptive_bonus
            
        except Exception as e:
            self.logger.error(f"適応的ボーナス計算エラー ({symbol}): {e}")
            return 0.0
    
    async def _analyze_market_factors(self, symbol: str, market_condition: Any) -> Dict[str, float]:
        """市場ファクター分析"""
        try:
            factors = {}
            
            # モメンタム分析
            momentum_score = await self._calculate_momentum_score(symbol, market_condition)
            factors['momentum'] = momentum_score
            
            # ボラティリティレジーム分析
            volatility_regime = await self._analyze_volatility_regime(symbol, market_condition)
            factors['volatility_regime'] = volatility_regime
            
            # 相対強度分析
            relative_strength = await self._calculate_relative_strength(symbol)
            factors['relative_strength'] = relative_strength
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"市場ファクター分析エラー ({symbol}): {e}")
            return {}
    
    async def _calculate_momentum_score(self, symbol: str, market_condition: Any) -> float:
        """モメンタムスコア計算"""
        try:
            # 市場状況別モメンタム重み
            condition_weights = {
                'trending_up': 1.2,
                'trending_down': 0.8,
                'sideways': 0.9,
                'high_volatility': 0.7,
                'low_volatility': 1.1
            }
            
            # copilot-instructions.md準拠: ランダムモメンタム生成禁止
            # 実データなしではモメンタム計算不可
            self.logger.error(f"モメンタムスコア計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"モメンタムスコア計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _analyze_volatility_regime(self, symbol: str, market_condition: Any) -> float:
        """ボラティリティレジーム分析"""
        try:
            # 現在のボラティリティ状況に基づくスコア調整
            volatility_adjustments = {
                'high_volatility': -0.2,  # 高ボラティリティ時は減点
                'low_volatility': 0.1,    # 低ボラティリティ時は加点
                'trending_up': 0.15,      # 上昇トレンド時は加点
                'trending_down': -0.1,    # 下降トレンド時は軽微減点
                'sideways': 0.0           # 横這い時は中立
            }
            
            condition_name = market_condition.value if hasattr(market_condition, 'value') else str(market_condition)
            adjustment = volatility_adjustments.get(condition_name, 0.0)
            
            # ベーススコア + 調整
            base_score = 0.5
            adjusted_score = base_score + adjustment
            
            return min(max(adjusted_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"ボラティリティレジーム分析エラー ({symbol}): {e}")
            return 0.5
    
    async def _calculate_relative_strength(self, symbol: str) -> float:
        """相対強度計算"""
        try:
            # copilot-instructions.md準拠: ランダム相対強度生成禁止
            self.logger.error(f"相対強度計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"相対強度計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _analyze_sector_factors(self, symbol: str) -> Dict[str, float]:
        """セクター分析"""
        try:
            factors = {}
            
            # セクターローテーションスコア
            rotation_score = await self._calculate_sector_rotation_score(symbol)
            factors['rotation_score'] = rotation_score
            
            # セクター内相対ポジション
            sector_position = await self._calculate_sector_position(symbol)
            factors['sector_position'] = sector_position
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"セクター分析エラー ({symbol}): {e}")
            return {}
    
    async def _calculate_sector_rotation_score(self, symbol: str) -> float:
        """セクターローテーションスコア計算"""
        try:
            # セクター判定（簡易版）
            sector_mapping = {
                '1': 'technology',
                '2': 'financial',
                '3': 'industrial',
                '4': 'healthcare',
                '5': 'consumer'
            }
            
            # 銘柄コードの最初の文字からセクター推定（簡易）
            sector_key = symbol[0] if symbol else '1'
            sector = sector_mapping.get(sector_key, 'technology')
            
            # セクター別ローテーションスコア（簡易）
            sector_scores = {
                'technology': 0.7,
                'financial': 0.5,
                'industrial': 0.6,
                'healthcare': 0.8,
                'consumer': 0.4
            }
            
            return sector_scores.get(sector, 0.5)
            
        except Exception as e:
            self.logger.warning(f"セクターローテーションスコア計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _calculate_sector_position(self, symbol: str) -> float:
        """セクター内ポジション計算"""
        try:
            # copilot-instructions.md準拠: ランダムセクターポジション生成禁止
            self.logger.error(f"セクターポジション計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"セクター内ポジション計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _analyze_technical_convergence(self, symbol: str) -> Dict[str, float]:
        """テクニカル収束分析"""
        try:
            factors = {}
            
            # 複数時間軸収束スコア
            convergence_score = await self._calculate_timeframe_convergence(symbol)
            factors['convergence_score'] = convergence_score
            
            # シグナル強度
            signal_strength = await self._calculate_signal_strength(symbol)
            factors['signal_strength'] = signal_strength
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"テクニカル収束分析エラー ({symbol}): {e}")
            return {}
    
    async def _calculate_timeframe_convergence(self, symbol: str) -> float:
        """時間軸収束スコア計算"""
        try:
            # copilot-instructions.md準拠: ランダム時間軸収束生成禁止
            self.logger.error(f"時間軸収束計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"時間軸収束スコア計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _calculate_signal_strength(self, symbol: str) -> float:
        """シグナル強度計算"""
        try:
            # copilot-instructions.md準拠: ランダムシグナル強度生成禁止
            self.logger.error(f"シグナル強度計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"シグナル強度計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _analyze_risk_sentiment(self, symbol: str) -> Dict[str, float]:
        """リスクセンチメント分析"""
        try:
            factors = {}
            
            # VIX等のリスク指標分析
            risk_sentiment = await self._calculate_risk_sentiment_score(symbol)
            factors['sentiment_score'] = risk_sentiment
            
            # 流動性分析
            liquidity_score = await self._calculate_liquidity_score(symbol)
            factors['liquidity_score'] = liquidity_score
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"リスクセンチメント分析エラー ({symbol}): {e}")
            return {}
    
    async def _calculate_risk_sentiment_score(self, symbol: str) -> float:
        """リスクセンチメントスコア計算"""
        try:
            # copilot-instructions.md準拠: ランダムリスクセンチメント生成禁止
            self.logger.error(f"リスクセンチメント計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"リスクセンチメントスコア計算エラー ({symbol}): {e}")
            return 0.5
    
    async def _calculate_liquidity_score(self, symbol: str) -> float:
        """流動性スコア計算"""
        try:
            # copilot-instructions.md準拠: ランダム流動性スコア生成禁止
            self.logger.error(f"流動性スコア計算失敗 ({symbol}): 実データ未実装")
            return 0.0  # エラー時は0.0返却
            
        except Exception as e:
            self.logger.warning(f"流動性スコア計算エラー ({symbol}): {e}")
            return 0.5
    
    def _apply_adaptive_weights(self, factors: Dict[str, float]) -> Dict[str, float]:
        """適応的重み適用"""
        try:
            weighted_factors = {}
            
            for factor_name, factor_value in factors.items():
                weight = self.adaptive_weights.get(factor_name, 0.2)
                weighted_factors[factor_name] = factor_value * weight
            
            return weighted_factors
            
        except Exception as e:
            self.logger.warning(f"適応的重み適用エラー: {e}")
            return {}
    
    async def _update_performance_learning(self, symbol: str, 
                                         hierarchical_result: RankingScore,
                                         adaptive_bonus: float):
        """パフォーマンス学習更新"""
        try:
            # パフォーマンス記録
            performance_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'hierarchical_score': hierarchical_result.total_score,
                'adaptive_bonus': adaptive_bonus,
                'confidence': hierarchical_result.confidence_level,
                'priority_group': hierarchical_result.priority_group
            }
            
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
            
            self.performance_history[symbol].append(performance_record)
            
            # 履歴サイズ制限
            max_history = self.config.get('max_performance_history', 1000)
            if len(self.performance_history[symbol]) > max_history:
                self.performance_history[symbol] = self.performance_history[symbol][-max_history:]
            
            # 定期的な重み更新
            if self._should_update_weights():
                await self._update_adaptive_weights()
            
        except Exception as e:
            self.logger.warning(f"パフォーマンス学習更新エラー ({symbol}): {e}")
    
    def _should_update_weights(self) -> bool:
        """重み更新判定"""
        time_since_update = datetime.now() - self.last_update
        return time_since_update.total_seconds() > (self.update_frequency_minutes * 60)
    
    async def _update_adaptive_weights(self):
        """適応的重み更新"""
        try:
            # パフォーマンス分析による重み調整
            performance_analysis = self._analyze_factor_performance()
            
            # 学習率適用
            for factor_name, performance_score in performance_analysis.items():
                if factor_name in self.adaptive_weights:
                    current_weight = self.adaptive_weights[factor_name]
                    # パフォーマンスが良い要因の重みを増加
                    adjustment = (performance_score - 0.5) * self.learning_rate
                    new_weight = current_weight + adjustment
                    
                    # 重み制約
                    self.adaptive_weights[factor_name] = min(max(new_weight, 0.05), 0.5)
            
            # 重み正規化
            total_weight = sum(self.adaptive_weights.values())
            if total_weight > 0:
                for factor_name in self.adaptive_weights:
                    self.adaptive_weights[factor_name] /= total_weight
            
            self.last_update = datetime.now()
            self.logger.info(f"適応的重み更新完了: {self.adaptive_weights}")
            
        except Exception as e:
            self.logger.error(f"適応的重み更新エラー: {e}")
    
    def _analyze_factor_performance(self) -> Dict[str, float]:
        """ファクターパフォーマンス分析"""
        try:
            # copilot-instructions.md準拠: ランダムパフォーマンススコア生成禁止
            self.logger.error("ファクターパフォーマンス分析失敗: 実データ未実装")
            return {}  # エラー時は空辞書返却
            
        except Exception as e:
            self.logger.warning(f"ファクターパフォーマンス分析エラー: {e}")
            return {}
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """現在の適応的重み取得"""
        return self.adaptive_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        try:
            total_records = sum(len(records) for records in self.performance_history.values())
            active_symbols = len(self.performance_history)
            
            # 平均パフォーマンス計算
            all_bonuses = []
            for records in self.performance_history.values():
                all_bonuses.extend([record['adaptive_bonus'] for record in records])
            
            avg_bonus = np.mean(all_bonuses) if all_bonuses else 0.0
            
            return {
                'total_performance_records': total_records,
                'active_symbols': active_symbols,
                'average_adaptive_bonus': avg_bonus,
                'adaptive_weights': self.adaptive_weights,
                'last_weight_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"パフォーマンス要約取得エラー: {e}")
            return {}
    
    def reset_learning(self):
        """学習データリセット"""
        self.performance_history.clear()
        self.market_condition_history.clear()
        
        # 重みを初期値にリセット
        self.adaptive_weights = {
            'market_momentum': 0.3,
            'volatility_regime': 0.25,
            'sector_rotation': 0.2,
            'risk_sentiment': 0.15,
            'technical_convergence': 0.1
        }
        
        self.last_update = datetime.now()
        self.logger.info("学習データをリセットしました")
