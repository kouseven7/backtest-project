"""
市場レジーム分類器 - A→B市場分類システム基盤
長期的な市場状況の識別と分類機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength
from .market_condition_detector import MarketConditionDetector
from .technical_indicator_analyzer import TechnicalIndicatorAnalyzer

class MarketRegime(Enum):
    """市場レジームの種類"""
    BULL_MARKET = "bull_market"                    # 強気相場
    BEAR_MARKET = "bear_market"                    # 弱気相場
    SIDEWAYS_MARKET = "sideways_market"            # 横ばい相場
    TRANSITIONAL_MARKET = "transitional_market"    # 移行期
    CRISIS_MARKET = "crisis_market"                # 危機相場
    RECOVERY_MARKET = "recovery_market"            # 回復相場

class RegimeStrength(Enum):
    """レジーム強度"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

@dataclass
class RegimeClassificationResult:
    """市場レジーム分類結果"""
    regime: MarketRegime
    strength: RegimeStrength
    confidence: float
    duration_estimate: int  # 推定継続期間（日数）
    supporting_factors: Dict[str, float]
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    characteristics: Dict[str, Any]
    classification_time: datetime
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}

class MarketRegimeClassifier:
    """
    市場レジーム分類システムのメインクラス
    長期的な市場環境の識別と分類を行う
    """
    
    def __init__(self, 
                 regime_lookback_period: int = 60,
                 confidence_threshold: float = 0.7,
                 volatility_threshold: float = 0.25,
                 trend_threshold: float = 0.15):
        """
        市場レジーム分類器の初期化
        
        Args:
            regime_lookback_period: レジーム分析期間
            confidence_threshold: 信頼度閾値
            volatility_threshold: ボラティリティ閾値
            trend_threshold: トレンド閾値
        """
        self.regime_lookback_period = regime_lookback_period
        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
        # サブシステム初期化
        self.condition_detector = MarketConditionDetector()
        self.technical_analyzer = TechnicalIndicatorAnalyzer()
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # 分類結果履歴
        self._regime_history = []
        self._regime_cache = {}
        
        self.logger.info("MarketRegimeClassifier初期化完了")

    def classify_market_regime(self, 
                             data: pd.DataFrame,
                             use_cache: bool = True,
                             custom_params: Optional[Dict] = None) -> RegimeClassificationResult:
        """
        市場レジームの分類
        
        Args:
            data: 市場データ (OHLCV形式)
            use_cache: キャッシュ使用フラグ
            custom_params: カスタムパラメータ
            
        Returns:
            RegimeClassificationResult: 分類結果
        """
        try:
            # データ検証
            if not self._validate_data(data):
                raise ValueError("無効なデータフォーマット")
            
            # キャッシュチェック
            cache_key = self._generate_cache_key(data)
            if use_cache and cache_key in self._regime_cache:
                self.logger.debug(f"キャッシュから結果を返却: {cache_key}")
                return self._regime_cache[cache_key]
            
            # パラメータ設定
            params = self._merge_params(custom_params)
            
            # 分析期間データの取得
            analysis_data = data.tail(self.regime_lookback_period) if len(data) > self.regime_lookback_period else data
            
            # 基本統計分析
            basic_stats = self._calculate_basic_statistics(analysis_data)
            
            # トレンド分析
            trend_analysis = self._analyze_long_term_trend(analysis_data)
            
            # ボラティリティ分析
            volatility_analysis = self._analyze_volatility_regime(analysis_data)
            
            # テクニカル統合分析
            technical_result = self.technical_analyzer.analyze_technical_indicators(analysis_data)
            
            # レジーム判定
            regime_result = self._determine_market_regime(
                basic_stats, trend_analysis, volatility_analysis, technical_result
            )
            
            # 結果をキャッシュ
            if use_cache:
                self._regime_cache[cache_key] = regime_result
            
            # 履歴に追加
            self._regime_history.append(regime_result)
            if len(self._regime_history) > 100:  # 履歴サイズ制限
                self._regime_history.pop(0)
            
            self.logger.info(f"市場レジーム分類完了: {regime_result.regime.value} (信頼度: {regime_result.confidence:.3f})")
            return regime_result
            
        except Exception as e:
            self.logger.error(f"市場レジーム分類エラー: {e}")
            return self._create_fallback_result()

    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """基本統計量の計算"""
        try:
            close = data['Close']
            
            # リターン計算
            returns = close.pct_change().dropna()
            
            # 基本統計
            stats = {
                'total_return': (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 0 else 0,
                'annualized_return': returns.mean() * 252 if len(returns) > 0 else 0,
                'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(close),
                'skewness': returns.skew() if len(returns) > 0 else 0,
                'kurtosis': returns.kurtosis() if len(returns) > 0 else 0,
                'win_rate': (returns > 0).mean() if len(returns) > 0 else 0.5
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"基本統計計算エラー: {e}")
            return {
                'total_return': 0, 'annualized_return': 0, 'volatility': 0.2,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'skewness': 0,
                'kurtosis': 0, 'win_rate': 0.5
            }

    def _analyze_long_term_trend(self, data: pd.DataFrame) -> Dict[str, float]:
        """長期トレンド分析"""
        try:
            close = data['Close']
            
            # 複数期間移動平均
            sma_short = close.rolling(10).mean()
            sma_medium = close.rolling(30).mean()
            sma_long = close.rolling(50).mean()
            
            # トレンド方向
            current_price = close.iloc[-1]
            
            # 移動平均の配列
            ma_alignment = 0
            if not pd.isna(sma_short.iloc[-1]) and not pd.isna(sma_medium.iloc[-1]) and not pd.isna(sma_long.iloc[-1]):
                if current_price > sma_short.iloc[-1] > sma_medium.iloc[-1] > sma_long.iloc[-1]:
                    ma_alignment = 1  # 完全な上昇トレンド
                elif current_price < sma_short.iloc[-1] < sma_medium.iloc[-1] < sma_long.iloc[-1]:
                    ma_alignment = -1  # 完全な下降トレンド
                else:
                    # 部分的なアライメント
                    alignment_score = 0
                    if current_price > sma_short.iloc[-1]:
                        alignment_score += 0.25
                    if sma_short.iloc[-1] > sma_medium.iloc[-1]:
                        alignment_score += 0.25
                    if sma_medium.iloc[-1] > sma_long.iloc[-1]:
                        alignment_score += 0.25
                    if current_price > sma_long.iloc[-1]:
                        alignment_score += 0.25
                    
                    ma_alignment = alignment_score if alignment_score > 0.5 else -(1 - alignment_score)
            
            # トレンド強度
            if len(close) >= 20:
                linear_trend = np.polyfit(range(len(close)), close, 1)[0]
                trend_strength = abs(linear_trend) / close.mean() * 100 if close.mean() > 0 else 0
            else:
                trend_strength = 0
            
            # 最近のモメンタム
            recent_momentum = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
            
            return {
                'ma_alignment': ma_alignment,
                'trend_strength': trend_strength,
                'recent_momentum': recent_momentum,
                'sma_slope': (sma_long.iloc[-1] / sma_long.iloc[-10] - 1) if len(sma_long) >= 10 and not pd.isna(sma_long.iloc[-1]) else 0
            }
            
        except Exception as e:
            self.logger.error(f"長期トレンド分析エラー: {e}")
            return {
                'ma_alignment': 0, 'trend_strength': 0,
                'recent_momentum': 0, 'sma_slope': 0
            }

    def _analyze_volatility_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティレジーム分析"""
        try:
            close = data['Close']
            returns = close.pct_change().dropna()
            
            if len(returns) == 0:
                return {'current_vol': 0.2, 'vol_regime': 0, 'vol_persistence': 0.5}
            
            # 現在のボラティリティ
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else returns.std() * np.sqrt(252)
            
            # 長期平均ボラティリティ
            long_term_vol = returns.std() * np.sqrt(252)
            
            # ボラティリティレジーム
            vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                vol_regime = 1  # 高ボラティリティ
            elif vol_ratio < 0.7:
                vol_regime = -1  # 低ボラティリティ
            else:
                vol_regime = 0  # 通常ボラティリティ
            
            # ボラティリティの持続性
            if len(returns) >= 20:
                vol_series = returns.rolling(10).std()
                vol_persistence = vol_series.autocorr(lag=1) if len(vol_series.dropna()) > 1 else 0.5
            else:
                vol_persistence = 0.5
            
            # VIX的指標（将来ボラティリティの予測）
            if len(returns) >= 30:
                garch_vol = self._estimate_garch_volatility(returns)
            else:
                garch_vol = current_vol
            
            return {
                'current_vol': current_vol,
                'long_term_vol': long_term_vol,
                'vol_ratio': vol_ratio,
                'vol_regime': vol_regime,
                'vol_persistence': vol_persistence,
                'predicted_vol': garch_vol
            }
            
        except Exception as e:
            self.logger.error(f"ボラティリティレジーム分析エラー: {e}")
            return {
                'current_vol': 0.2, 'long_term_vol': 0.2, 'vol_ratio': 1.0,
                'vol_regime': 0, 'vol_persistence': 0.5, 'predicted_vol': 0.2
            }

    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """GARCH風ボラティリティ推定"""
        try:
            # 簡易GARCH(1,1)風の推定
            returns_squared = returns ** 2
            
            # 長期平均
            long_term_var = returns_squared.mean()
            
            # 直近の重み付き分散
            weights = np.exp(-np.arange(len(returns_squared)) / 10)[::-1]
            weights = weights / weights.sum()
            
            weighted_var = np.sum(returns_squared * weights)
            
            # GARCH推定（簡易版）
            alpha = 0.1  # 直近リターンの重み
            beta = 0.8   # 過去分散の重み
            omega = long_term_var * (1 - alpha - beta)  # 長期成分
            
            predicted_var = omega + alpha * returns_squared.iloc[-1] + beta * weighted_var
            
            return np.sqrt(predicted_var * 252)  # 年率化
            
        except:
            return returns.std() * np.sqrt(252)

    def _determine_market_regime(self, 
                               basic_stats: Dict[str, float],
                               trend_analysis: Dict[str, float],
                               volatility_analysis: Dict[str, float],
                               technical_result) -> RegimeClassificationResult:
        """市場レジームの決定"""
        try:
            # 各要素のスコア計算
            scores = {}
            
            # 1. トレンドベーススコア
            ma_alignment = trend_analysis['ma_alignment']
            trend_strength = trend_analysis['trend_strength']
            
            if ma_alignment > 0.7 and basic_stats['total_return'] > 0.15:
                scores['bull'] = 0.8 + min(trend_strength / 10, 0.2)
            elif ma_alignment < -0.7 and basic_stats['total_return'] < -0.15:
                scores['bear'] = 0.8 + min(trend_strength / 10, 0.2)
            else:
                scores['sideways'] = 0.6
            
            # 2. ボラティリティベーススコア
            vol_regime = volatility_analysis['vol_regime']
            current_vol = volatility_analysis['current_vol']
            
            if current_vol > 0.4:  # 極高ボラティリティ
                scores['crisis'] = scores.get('crisis', 0) + 0.7
            elif vol_regime == 1:  # 高ボラティリティ
                # 既存のスコアを増幅
                for key in ['bull', 'bear']:
                    if key in scores:
                        scores[key] *= 0.8  # 信頼度低下
            
            # 3. リターンパターン分析
            total_return = basic_stats['total_return']
            max_drawdown = abs(basic_stats['max_drawdown'])
            
            if total_return > 0.2 and max_drawdown < 0.1:
                scores['bull'] = scores.get('bull', 0) + 0.3
            elif total_return < -0.2:
                scores['bear'] = scores.get('bear', 0) + 0.3
            elif max_drawdown > 0.3:
                scores['crisis'] = scores.get('crisis', 0) + 0.5
            
            # 4. 回復パターン検出
            if total_return > 0.1 and basic_stats.get('recent_momentum', 0) > 0.05 and max_drawdown > 0.15:
                scores['recovery'] = 0.6
            
            # 5. 移行期パターン検出
            if abs(ma_alignment) < 0.3 and current_vol > 0.25:
                scores['transitional'] = 0.5
            
            # 最高スコアのレジームを選択
            if not scores:
                scores['sideways'] = 0.3
            
            best_regime = max(scores, key=scores.get)
            confidence = scores[best_regime]
            
            # レジーム変換
            regime_mapping = {
                'bull': MarketRegime.BULL_MARKET,
                'bear': MarketRegime.BEAR_MARKET,
                'sideways': MarketRegime.SIDEWAYS_MARKET,
                'crisis': MarketRegime.CRISIS_MARKET,
                'recovery': MarketRegime.RECOVERY_MARKET,
                'transitional': MarketRegime.TRANSITIONAL_MARKET
            }
            
            final_regime = regime_mapping[best_regime]
            
            # 強度決定
            if confidence > 0.8:
                strength = RegimeStrength.VERY_STRONG
            elif confidence > 0.7:
                strength = RegimeStrength.STRONG
            elif confidence > 0.6:
                strength = RegimeStrength.MODERATE
            elif confidence > 0.4:
                strength = RegimeStrength.WEAK
            else:
                strength = RegimeStrength.VERY_WEAK
            
            # リスクレベル決定
            risk_level = self._determine_risk_level(final_regime, current_vol, max_drawdown)
            
            # 継続期間推定
            duration_estimate = self._estimate_regime_duration(final_regime, confidence, trend_strength)
            
            # 支持要因
            supporting_factors = {
                'trend_alignment': ma_alignment,
                'volatility_ratio': volatility_analysis.get('vol_ratio', 1.0),
                'return_performance': total_return,
                'drawdown_risk': max_drawdown,
                'momentum': trend_analysis.get('recent_momentum', 0)
            }
            
            # 特性
            characteristics = {
                'primary_driver': best_regime,
                'confidence_score': confidence,
                'volatility_environment': 'high' if current_vol > 0.3 else 'normal' if current_vol > 0.15 else 'low',
                'trend_environment': 'strong' if abs(ma_alignment) > 0.7 else 'weak',
                'regime_scores': scores
            }
            
            return RegimeClassificationResult(
                regime=final_regime,
                strength=strength,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_factors=supporting_factors,
                risk_level=risk_level,
                characteristics=characteristics,
                classification_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"レジーム決定エラー: {e}")
            return self._create_fallback_result()

    def _determine_risk_level(self, regime: MarketRegime, volatility: float, max_drawdown: float) -> str:
        """リスクレベルの決定"""
        try:
            base_risk = {
                MarketRegime.BULL_MARKET: 'medium',
                MarketRegime.BEAR_MARKET: 'high',
                MarketRegime.SIDEWAYS_MARKET: 'low',
                MarketRegime.CRISIS_MARKET: 'extreme',
                MarketRegime.RECOVERY_MARKET: 'medium',
                MarketRegime.TRANSITIONAL_MARKET: 'high'
            }
            
            risk = base_risk.get(regime, 'medium')
            
            # ボラティリティ調整
            if volatility > 0.4:
                if risk == 'low':
                    risk = 'medium'
                elif risk == 'medium':
                    risk = 'high'
                elif risk == 'high':
                    risk = 'extreme'
            
            # ドローダウン調整
            if max_drawdown > 0.3:
                if risk in ['low', 'medium']:
                    risk = 'high'
                elif risk == 'high':
                    risk = 'extreme'
            
            return risk
            
        except:
            return 'medium'

    def _estimate_regime_duration(self, regime: MarketRegime, confidence: float, trend_strength: float) -> int:
        """レジーム継続期間の推定"""
        try:
            base_duration = {
                MarketRegime.BULL_MARKET: 200,
                MarketRegime.BEAR_MARKET: 150,
                MarketRegime.SIDEWAYS_MARKET: 100,
                MarketRegime.CRISIS_MARKET: 30,
                MarketRegime.RECOVERY_MARKET: 80,
                MarketRegime.TRANSITIONAL_MARKET: 40
            }
            
            duration = base_duration.get(regime, 60)
            
            # 信頼度による調整
            duration = int(duration * confidence)
            
            # トレンド強度による調整
            if trend_strength > 5:
                duration = int(duration * 1.2)
            elif trend_strength < 2:
                duration = int(duration * 0.8)
            
            return max(duration, 10)  # 最小10日
            
        except:
            return 60

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            if len(prices) == 0:
                return 0
            
            # 累積リターン
            cumulative = (1 + prices.pct_change()).cumprod()
            
            # ランニングマックス
            running_max = cumulative.expanding().max()
            
            # ドローダウン
            drawdown = (cumulative - running_max) / running_max
            
            return drawdown.min()
            
        except:
            return 0

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        return all(col in data.columns for col in required_columns) and len(data) >= 30

    def _merge_params(self, custom_params: Optional[Dict]) -> Dict:
        """パラメータ統合"""
        default_params = {
            'regime_period': self.regime_lookback_period,
            'vol_threshold': self.volatility_threshold,
            'trend_threshold': self.trend_threshold
        }
        
        if custom_params:
            default_params.update(custom_params)
        
        return default_params

    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """キャッシュキー生成"""
        try:
            last_timestamp = str(data.index[-1]) if hasattr(data.index, '__getitem__') else str(len(data))
            return f"regime_{last_timestamp}_{len(data)}"
        except:
            return f"regime_{datetime.now().isoformat()}"

    def _create_fallback_result(self) -> RegimeClassificationResult:
        """フォールバック結果生成"""
        return RegimeClassificationResult(
            regime=MarketRegime.SIDEWAYS_MARKET,
            strength=RegimeStrength.WEAK,
            confidence=0.1,
            duration_estimate=30,
            supporting_factors={'error': 'fallback'},
            risk_level='medium',
            characteristics={'is_fallback': True},
            classification_time=datetime.now()
        )

    def get_regime_history(self) -> List[RegimeClassificationResult]:
        """レジーム履歴取得"""
        return self._regime_history.copy()

    def get_regime_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """レジームサマリー取得"""
        try:
            result = self.classify_market_regime(data)
            return {
                'regime': result.regime.value,
                'strength': result.strength.value,
                'confidence': result.confidence,
                'risk_level': result.risk_level,
                'duration_estimate': result.duration_estimate,
                'classification_time': result.classification_time.isoformat(),
                'supporting_factors': result.supporting_factors
            }
        except Exception as e:
            self.logger.error(f"レジームサマリー取得エラー: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """キャッシュクリア"""
        self._regime_cache.clear()
        self.logger.info("レジーム分類キャッシュをクリアしました")

# 利便性関数
def classify_market_regime_simple(data: pd.DataFrame) -> Dict[str, Any]:
    """
    簡単な市場レジーム分類関数
    
    Args:
        data: 市場データ
        
    Returns:
        Dict: 分類結果の辞書形式
    """
    classifier = MarketRegimeClassifier()
    result = classifier.classify_market_regime(data)
    
    return {
        'regime': result.regime.value,
        'strength': result.strength.value,
        'confidence': result.confidence,
        'risk_level': result.risk_level,
        'duration_estimate': result.duration_estimate,
        'classification_time': result.classification_time.isoformat()
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== 市場レジーム分類システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=120)
    np.random.seed(42)
    
    # 強気相場データ
    bull_prices = []
    base_price = 100
    for i in range(120):
        bull_prices.append(base_price * (1 + 0.001 * i + np.random.normal(0, 0.01)))
    
    bull_data = pd.DataFrame({
        'Open': [p + np.random.normal(0, 0.5) for p in bull_prices],
        'High': [p + abs(np.random.normal(0, 1)) for p in bull_prices],
        'Low': [p - abs(np.random.normal(0, 1)) for p in bull_prices],
        'Close': bull_prices,
        'Volume': [np.random.uniform(1000000, 5000000) for _ in range(120)]
    }, index=dates)
    
    # 分類器テスト
    classifier = MarketRegimeClassifier()
    
    print("\n1. 強気相場データ分析")
    result = classifier.classify_market_regime(bull_data)
    print(f"レジーム: {result.regime.value}")
    print(f"強度: {result.strength.value}")
    print(f"信頼度: {result.confidence:.3f}")
    print(f"リスクレベル: {result.risk_level}")
    print(f"推定継続期間: {result.duration_estimate}日")
    
    print("\n2. 弱気相場データ作成・分析")
    # 弱気相場データ
    bear_prices = []
    base_price = 100
    for i in range(120):
        bear_prices.append(base_price * (1 - 0.002 * i + np.random.normal(0, 0.015)))
    
    bear_data = bull_data.copy()
    bear_data['Close'] = bear_prices
    
    bear_result = classifier.classify_market_regime(bear_data)
    print(f"レジーム: {bear_result.regime.value}")
    print(f"強度: {bear_result.strength.value}")
    print(f"信頼度: {bear_result.confidence:.3f}")
    
    print("\n3. 簡単分類関数テスト")
    simple_result = classify_market_regime_simple(bull_data)
    print(f"簡単分類結果: {simple_result['regime']} (信頼度: {simple_result['confidence']:.3f})")
    
    print("\n4. レジーム履歴確認")
    history = classifier.get_regime_history()
    print(f"履歴エントリ数: {len(history)}")
    
    print("\n=== テスト完了 ===")
