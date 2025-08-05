"""
ボラティリティ分析システム - A→B市場分類システム基盤
高度なボラティリティ分析と予測機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
import math

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength

class VolatilityModel(Enum):
    """ボラティリティモデルの種類"""
    HISTORICAL = "historical"
    EWMA = "ewma"  # 指数重み付き移動平均
    GARCH = "garch"  # GARCH(1,1)
    PARKINSON = "parkinson"  # Parkinson推定
    GARMAN_KLASS = "garman_klass"  # Garman-Klass推定
    YANG_ZHANG = "yang_zhang"  # Yang-Zhang推定

class VolatilityRegime(Enum):
    """ボラティリティレジーム"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

@dataclass
class VolatilityResult:
    """ボラティリティ分析結果"""
    model: VolatilityModel
    current_volatility: float
    annualized_volatility: float
    volatility_regime: VolatilityRegime
    confidence: float
    forecast_volatility: Optional[float]
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    additional_metrics: Dict[str, float]
    calculation_time: datetime
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}

@dataclass
class VolatilityAnalysisResult:
    """総合ボラティリティ分析結果"""
    primary_model: VolatilityModel
    volatility_estimates: List[VolatilityResult]
    ensemble_volatility: float
    volatility_regime: VolatilityRegime
    regime_confidence: float
    risk_metrics: Dict[str, float]
    forecast_horizon: int  # 予測期間（日数）
    analysis_summary: Dict[str, Any]
    analysis_time: datetime

class VolatilityAnalyzer:
    """
    ボラティリティ分析システムのメインクラス
    複数のボラティリティモデルを用いた高度な分析機能を提供
    """
    
    def __init__(self, 
                 default_window: int = 20,
                 ewma_lambda: float = 0.94,
                 garch_alpha: float = 0.1,
                 garch_beta: float = 0.8,
                 forecast_horizon: int = 5):
        """
        ボラティリティ分析器の初期化
        
        Args:
            default_window: デフォルト分析期間
            ewma_lambda: EWMA平滑化パラメータ
            garch_alpha: GARCHアルファパラメータ
            garch_beta: GARCHベータパラメータ
            forecast_horizon: 予測期間
        """
        self.default_window = default_window
        self.ewma_lambda = ewma_lambda
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta
        self.forecast_horizon = forecast_horizon
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # ボラティリティレジーム閾値
        self.regime_thresholds = {
            'very_low': 0.1,
            'low': 0.15,
            'normal': 0.25,
            'high': 0.35,
            'very_high': 0.5
        }
        
        # 分析結果キャッシュ
        self._volatility_cache = {}
        self._cache_timeout = timedelta(minutes=10)
        
        self.logger.info("VolatilityAnalyzer初期化完了")

    def analyze_volatility(self, 
                         data: pd.DataFrame,
                         models: Optional[List[VolatilityModel]] = None,
                         custom_params: Optional[Dict] = None) -> VolatilityAnalysisResult:
        """
        総合ボラティリティ分析
        
        Args:
            data: 市場データ (OHLCV形式)
            models: 使用するモデル (None=全モデル)
            custom_params: カスタムパラメータ
            
        Returns:
            VolatilityAnalysisResult: 分析結果
        """
        try:
            # データ検証
            if not self._validate_data(data):
                raise ValueError("無効なデータフォーマット")
            
            # キャッシュチェック
            cache_key = self._generate_cache_key(data, models)
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"キャッシュから結果を返却: {cache_key}")
                return self._volatility_cache[cache_key]['result']
            
            # モデル設定
            if models is None:
                models = [VolatilityModel.HISTORICAL, VolatilityModel.EWMA, 
                         VolatilityModel.GARCH, VolatilityModel.PARKINSON]
            
            # パラメータ統合
            params = self._merge_params(custom_params)
            
            # 各モデルでボラティリティ推定
            volatility_estimates = []
            
            for model in models:
                try:
                    result = self._calculate_volatility(data, model, params)
                    if result:
                        volatility_estimates.append(result)
                except Exception as e:
                    self.logger.warning(f"モデル {model.value} でエラー: {e}")
                    continue
            
            if not volatility_estimates:
                return self._create_fallback_result()
            
            # アンサンブル推定
            ensemble_volatility = self._calculate_ensemble_volatility(volatility_estimates)
            
            # レジーム判定
            volatility_regime = self._determine_volatility_regime(ensemble_volatility)
            
            # リスクメトリクス計算
            risk_metrics = self._calculate_risk_metrics(data, ensemble_volatility)
            
            # 分析サマリー
            analysis_summary = self._create_analysis_summary(volatility_estimates, ensemble_volatility, risk_metrics)
            
            # 主要モデル選択（最高信頼度）
            primary_model = max(volatility_estimates, key=lambda x: x.confidence).model
            
            # レジーム信頼度
            regime_confidence = self._calculate_regime_confidence(volatility_estimates, volatility_regime)
            
            result = VolatilityAnalysisResult(
                primary_model=primary_model,
                volatility_estimates=volatility_estimates,
                ensemble_volatility=ensemble_volatility,
                volatility_regime=volatility_regime,
                regime_confidence=regime_confidence,
                risk_metrics=risk_metrics,
                forecast_horizon=self.forecast_horizon,
                analysis_summary=analysis_summary,
                analysis_time=datetime.now()
            )
            
            # 結果をキャッシュ
            self._cache_result(cache_key, result)
            
            self.logger.info(f"ボラティリティ分析完了: {ensemble_volatility:.3f} ({volatility_regime.value})")
            return result
            
        except Exception as e:
            self.logger.error(f"ボラティリティ分析エラー: {e}")
            return self._create_fallback_analysis_result()

    def _calculate_volatility(self, 
                            data: pd.DataFrame, 
                            model: VolatilityModel, 
                            params: Dict) -> Optional[VolatilityResult]:
        """個別モデルでのボラティリティ計算"""
        try:
            if model == VolatilityModel.HISTORICAL:
                return self._historical_volatility(data, params)
            elif model == VolatilityModel.EWMA:
                return self._ewma_volatility(data, params)
            elif model == VolatilityModel.GARCH:
                return self._garch_volatility(data, params)
            elif model == VolatilityModel.PARKINSON:
                return self._parkinson_volatility(data, params)
            elif model == VolatilityModel.GARMAN_KLASS:
                return self._garman_klass_volatility(data, params)
            elif model == VolatilityModel.YANG_ZHANG:
                return self._yang_zhang_volatility(data, params)
            else:
                self.logger.warning(f"未対応のボラティリティモデル: {model}")
                return None
                
        except Exception as e:
            self.logger.error(f"{model.value} ボラティリティ計算エラー: {e}")
            return None

    def _historical_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """ヒストリカルボラティリティ"""
        try:
            window = params.get('window', self.default_window)
            close = data['Close']
            
            # リターン計算
            returns = close.pct_change().dropna()
            
            # ローリングボラティリティ
            rolling_vol = returns.rolling(window=window).std()
            current_vol = rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else returns.std()
            annualized_vol = current_vol * np.sqrt(252)
            
            # トレンド分析
            if len(rolling_vol) >= 10:
                recent_vol = rolling_vol.tail(5).mean()
                past_vol = rolling_vol.tail(10).head(5).mean()
                
                if recent_vol > past_vol * 1.1:
                    trend = 'increasing'
                elif recent_vol < past_vol * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度（データ量とボラティリティの安定性基準）
            confidence = min(len(returns) / (window * 2), 1.0) * 0.8
            
            # 追加メトリクス
            additional_metrics = {
                'sample_size': len(returns),
                'min_volatility': rolling_vol.min() * np.sqrt(252) if len(rolling_vol) > 0 else annualized_vol,
                'max_volatility': rolling_vol.max() * np.sqrt(252) if len(rolling_vol) > 0 else annualized_vol,
                'volatility_std': rolling_vol.std() * np.sqrt(252) if len(rolling_vol) > 0 else 0
            }
            
            return VolatilityResult(
                model=VolatilityModel.HISTORICAL,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=annualized_vol,  # 単純予測
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ヒストリカルボラティリティ計算エラー: {e}")
            raise

    def _ewma_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """EWMA ボラティリティ"""
        try:
            lambda_param = params.get('ewma_lambda', self.ewma_lambda)
            close = data['Close']
            
            # リターン計算
            returns = close.pct_change().dropna()
            
            if len(returns) == 0:
                raise ValueError("リターンデータが空です")
            
            # EWMA計算
            ewma_var = returns.iloc[0] ** 2  # 初期値
            ewma_vars = [ewma_var]
            
            for i in range(1, len(returns)):
                ewma_var = lambda_param * ewma_var + (1 - lambda_param) * (returns.iloc[i] ** 2)
                ewma_vars.append(ewma_var)
            
            # 現在のボラティリティ
            current_vol = np.sqrt(ewma_vars[-1])
            annualized_vol = current_vol * np.sqrt(252)
            
            # 予測ボラティリティ（1期先）
            forecast_vol = np.sqrt(lambda_param * ewma_vars[-1] + (1 - lambda_param) * returns.iloc[-1] ** 2) * np.sqrt(252)
            
            # トレンド分析
            if len(ewma_vars) >= 10:
                recent_avg = np.mean(ewma_vars[-5:])
                past_avg = np.mean(ewma_vars[-10:-5])
                
                if recent_avg > past_avg * 1.1:
                    trend = 'increasing'
                elif recent_avg < past_avg * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度
            confidence = 0.85  # EWMAは一般的に信頼性が高い
            
            # 追加メトリクス
            additional_metrics = {
                'lambda_parameter': lambda_param,
                'ewma_variance': ewma_vars[-1],
                'trend_strength': abs(np.mean(ewma_vars[-5:]) - np.mean(ewma_vars[-10:-5])) if len(ewma_vars) >= 10 else 0
            }
            
            return VolatilityResult(
                model=VolatilityModel.EWMA,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=forecast_vol,
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"EWMA ボラティリティ計算エラー: {e}")
            raise

    def _garch_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """GARCH(1,1) ボラティリティ"""
        try:
            alpha = params.get('garch_alpha', self.garch_alpha)
            beta = params.get('garch_beta', self.garch_beta)
            close = data['Close']
            
            # リターン計算
            returns = close.pct_change().dropna()
            
            if len(returns) < 50:  # GARCH には十分なデータが必要
                # フォールバックとしてEWMAを使用
                return self._ewma_volatility(data, params)
            
            # 長期分散（unconditional variance）
            long_term_var = returns.var()
            omega = long_term_var * (1 - alpha - beta)
            
            # GARCH計算
            garch_vars = []
            
            # 初期分散
            h_t = long_term_var
            garch_vars.append(h_t)
            
            for i in range(1, len(returns)):
                h_t = omega + alpha * (returns.iloc[i-1] ** 2) + beta * h_t
                garch_vars.append(h_t)
            
            # 現在のボラティリティ
            current_vol = np.sqrt(garch_vars[-1])
            annualized_vol = current_vol * np.sqrt(252)
            
            # 予測ボラティリティ（1期先）
            forecast_var = omega + alpha * (returns.iloc[-1] ** 2) + beta * garch_vars[-1]
            forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
            
            # トレンド分析
            if len(garch_vars) >= 10:
                recent_avg = np.mean(garch_vars[-5:])
                past_avg = np.mean(garch_vars[-10:-5])
                
                if recent_avg > past_avg * 1.05:
                    trend = 'increasing'
                elif recent_avg < past_avg * 0.95:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度（パラメータの妥当性チェック）
            if alpha + beta < 1 and alpha > 0 and beta > 0:
                confidence = 0.9
            else:
                confidence = 0.6  # パラメータが境界値に近い
            
            # 追加メトリクス
            additional_metrics = {
                'alpha': alpha,
                'beta': beta,
                'omega': omega,
                'long_term_volatility': np.sqrt(long_term_var) * np.sqrt(252),
                'persistence': alpha + beta
            }
            
            return VolatilityResult(
                model=VolatilityModel.GARCH,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=forecast_vol,
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"GARCH ボラティリティ計算エラー: {e}")
            raise

    def _parkinson_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """Parkinson ボラティリティ推定器"""
        try:
            window = params.get('window', self.default_window)
            
            # High-Low の対数比
            high_low_ratio = np.log(data['High'] / data['Low'])
            
            # Parkinson 推定
            parkinson_vars = (high_low_ratio ** 2 / (4 * np.log(2))).rolling(window=window).mean()
            
            current_var = parkinson_vars.iloc[-1] if not pd.isna(parkinson_vars.iloc[-1]) else (high_low_ratio ** 2 / (4 * np.log(2))).mean()
            current_vol = np.sqrt(current_var)
            annualized_vol = current_vol * np.sqrt(252)
            
            # トレンド分析
            if len(parkinson_vars) >= 10:
                recent_avg = parkinson_vars.tail(5).mean()
                past_avg = parkinson_vars.tail(10).head(5).mean()
                
                if recent_avg > past_avg * 1.1:
                    trend = 'increasing'
                elif recent_avg < past_avg * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度（高頻度データに対して有効）
            confidence = 0.75
            
            # 追加メトリクス
            additional_metrics = {
                'sample_size': len(high_low_ratio),
                'mean_hl_ratio': high_low_ratio.mean(),
                'efficiency_ratio': 1.0 / (4 * np.log(2))  # Parkinson効率性
            }
            
            return VolatilityResult(
                model=VolatilityModel.PARKINSON,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=annualized_vol,  # 単純予測
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Parkinson ボラティリティ計算エラー: {e}")
            raise

    def _garman_klass_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """Garman-Klass ボラティリティ推定器"""
        try:
            window = params.get('window', self.default_window)
            
            # Garman-Klass 推定
            high = data['High']
            low = data['Low']
            close = data['Close']
            open_price = data['Open']
            
            # GK項1: ln(H/L) * ln(H/L)
            hl_component = np.log(high / low) ** 2
            
            # GK項2: (2*ln(2) - 1) * ln(C/O) * ln(C/O)  
            co_component = (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2)
            
            gk_estimator = 0.5 * hl_component - co_component
            gk_vars = gk_estimator.rolling(window=window).mean()
            
            current_var = gk_vars.iloc[-1] if not pd.isna(gk_vars.iloc[-1]) else gk_estimator.mean()
            current_vol = np.sqrt(abs(current_var))  # 負の値を避ける
            annualized_vol = current_vol * np.sqrt(252)
            
            # トレンド分析
            if len(gk_vars) >= 10:
                recent_avg = gk_vars.tail(5).mean()
                past_avg = gk_vars.tail(10).head(5).mean()
                
                if recent_avg > past_avg * 1.1:
                    trend = 'increasing'
                elif recent_avg < past_avg * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度
            confidence = 0.8  # GKは理論的に効率的
            
            # 追加メトリクス
            additional_metrics = {
                'hl_component_mean': hl_component.mean(),
                'co_component_mean': co_component.mean(),
                'gk_efficiency': 1.0  # 理論的最適
            }
            
            return VolatilityResult(
                model=VolatilityModel.GARMAN_KLASS,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=annualized_vol,
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Garman-Klass ボラティリティ計算エラー: {e}")
            raise

    def _yang_zhang_volatility(self, data: pd.DataFrame, params: Dict) -> VolatilityResult:
        """Yang-Zhang ボラティリティ推定器"""
        try:
            window = params.get('window', self.default_window)
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            open_price = data['Open']
            
            # 前日終値
            prev_close = close.shift(1)
            
            # Yang-Zhang 各成分
            # Overnight return variance
            overnight_ret = np.log(open_price / prev_close)
            overnight_var = overnight_ret.rolling(window=window).var()
            
            # Open-to-close variance  
            oc_ret = np.log(close / open_price)
            oc_var = oc_ret.rolling(window=window).var()
            
            # Rogers-Satchell variance
            rs_component = np.log(high / close) * np.log(high / open_price) + np.log(low / close) * np.log(low / open_price)
            rs_var = rs_component.rolling(window=window).mean()
            
            # Yang-Zhang combination
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
            
            current_var = yz_var.iloc[-1] if not pd.isna(yz_var.iloc[-1]) else yz_var.mean()
            current_vol = np.sqrt(abs(current_var))
            annualized_vol = current_vol * np.sqrt(252)
            
            # トレンド分析
            if len(yz_var) >= 10:
                recent_avg = yz_var.tail(5).mean()
                past_avg = yz_var.tail(10).head(5).mean()
                
                if recent_avg > past_avg * 1.1:
                    trend = 'increasing'
                elif recent_avg < past_avg * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # 信頼度
            confidence = 0.85  # YZは最も理論的に進歩
            
            # 追加メトリクス
            additional_metrics = {
                'overnight_component': overnight_var.iloc[-1] if not pd.isna(overnight_var.iloc[-1]) else 0,
                'open_close_component': oc_var.iloc[-1] if not pd.isna(oc_var.iloc[-1]) else 0,
                'rogers_satchell_component': rs_var.iloc[-1] if not pd.isna(rs_var.iloc[-1]) else 0,
                'k_parameter': k
            }
            
            return VolatilityResult(
                model=VolatilityModel.YANG_ZHANG,
                current_volatility=current_vol,
                annualized_volatility=annualized_vol,
                volatility_regime=self._determine_volatility_regime(annualized_vol),
                confidence=confidence,
                forecast_volatility=annualized_vol,
                volatility_trend=trend,
                additional_metrics=additional_metrics,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Yang-Zhang ボラティリティ計算エラー: {e}")
            raise

    def _calculate_ensemble_volatility(self, estimates: List[VolatilityResult]) -> float:
        """アンサンブルボラティリティ計算"""
        try:
            if not estimates:
                return 0.2  # デフォルト値
            
            # 信頼度重み付き平均
            total_weight = sum(est.confidence for est in estimates)
            if total_weight == 0:
                return np.mean([est.annualized_volatility for est in estimates])
            
            weighted_vol = sum(est.annualized_volatility * est.confidence for est in estimates) / total_weight
            
            # 外れ値除去（中央値からの乖離が大きすぎるものを調整）
            median_vol = np.median([est.annualized_volatility for est in estimates])
            if abs(weighted_vol - median_vol) / median_vol > 0.5:
                # 重み付き平均と中央値の平均を取る
                weighted_vol = (weighted_vol + median_vol) / 2
            
            return weighted_vol
            
        except Exception as e:
            self.logger.error(f"アンサンブルボラティリティ計算エラー: {e}")
            return np.mean([est.annualized_volatility for est in estimates]) if estimates else 0.2

    def _determine_volatility_regime(self, annualized_volatility: float) -> VolatilityRegime:
        """ボラティリティレジーム判定"""
        try:
            thresholds = self.regime_thresholds
            
            if annualized_volatility >= thresholds['very_high']:
                return VolatilityRegime.EXTREME
            elif annualized_volatility >= thresholds['high']:
                return VolatilityRegime.VERY_HIGH
            elif annualized_volatility >= thresholds['normal']:
                return VolatilityRegime.HIGH
            elif annualized_volatility >= thresholds['low']:
                return VolatilityRegime.NORMAL
            elif annualized_volatility >= thresholds['very_low']:
                return VolatilityRegime.LOW
            else:
                return VolatilityRegime.VERY_LOW
                
        except:
            return VolatilityRegime.NORMAL

    def _calculate_regime_confidence(self, estimates: List[VolatilityResult], regime: VolatilityRegime) -> float:
        """レジーム信頼度計算"""
        try:
            regime_votes = 0
            total_weight = 0
            
            for est in estimates:
                if est.volatility_regime == regime:
                    regime_votes += est.confidence
                total_weight += est.confidence
            
            return regime_votes / total_weight if total_weight > 0 else 0.5
            
        except:
            return 0.5

    def _calculate_risk_metrics(self, data: pd.DataFrame, volatility: float) -> Dict[str, float]:
        """リスクメトリクス計算"""
        try:
            close = data['Close']
            returns = close.pct_change().dropna()
            
            if len(returns) == 0:
                return {'var_95': 0, 'var_99': 0, 'expected_shortfall': 0, 'max_drawdown': 0}
            
            # VaR計算
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (CVaR)
            es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 年率化
            return {
                'var_95': var_95 * np.sqrt(252),
                'var_99': var_99 * np.sqrt(252),
                'expected_shortfall': es_95 * np.sqrt(252),
                'max_drawdown': max_drawdown,
                'volatility_ratio': volatility / 0.2,  # 基準ボラティリティ20%
                'downside_volatility': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"リスクメトリクス計算エラー: {e}")
            return {'var_95': 0, 'var_99': 0, 'expected_shortfall': 0, 'max_drawdown': 0, 'volatility_ratio': 1, 'downside_volatility': 0}

    def _create_analysis_summary(self, estimates: List[VolatilityResult], ensemble_vol: float, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """分析サマリー作成"""
        try:
            return {
                'model_count': len(estimates),
                'volatility_range': {
                    'min': min(est.annualized_volatility for est in estimates),
                    'max': max(est.annualized_volatility for est in estimates),
                    'ensemble': ensemble_vol
                },
                'model_consensus': {
                    'agreement_level': 1 - (max(est.annualized_volatility for est in estimates) - min(est.annualized_volatility for est in estimates)) / ensemble_vol if ensemble_vol > 0 else 0,
                    'average_confidence': np.mean([est.confidence for est in estimates])
                },
                'trend_consensus': {
                    'increasing': sum(1 for est in estimates if est.volatility_trend == 'increasing'),
                    'decreasing': sum(1 for est in estimates if est.volatility_trend == 'decreasing'),
                    'stable': sum(1 for est in estimates if est.volatility_trend == 'stable')
                },
                'risk_assessment': {
                    'risk_level': 'high' if ensemble_vol > 0.3 else 'medium' if ensemble_vol > 0.15 else 'low',
                    'risk_metrics': risk_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"分析サマリー作成エラー: {e}")
            return {'error': 'summary_creation_failed'}

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        return all(col in data.columns for col in required_columns) and len(data) >= 10

    def _merge_params(self, custom_params: Optional[Dict]) -> Dict:
        """パラメータ統合"""
        default_params = {
            'window': self.default_window,
            'ewma_lambda': self.ewma_lambda,
            'garch_alpha': self.garch_alpha,
            'garch_beta': self.garch_beta
        }
        
        if custom_params:
            default_params.update(custom_params)
        
        return default_params

    def _generate_cache_key(self, data: pd.DataFrame, models: Optional[List[VolatilityModel]]) -> str:
        """キャッシュキー生成"""
        try:
            last_timestamp = str(data.index[-1]) if hasattr(data.index, '__getitem__') else str(len(data))
            models_str = '_'.join([m.value for m in models]) if models else 'all'
            return f"vol_{last_timestamp}_{len(data)}_{models_str}"
        except:
            return f"vol_{datetime.now().isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュ有効性チェック"""
        if cache_key not in self._volatility_cache:
            return False
        
        cache_time = self._volatility_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self._cache_timeout

    def _cache_result(self, cache_key: str, result: VolatilityAnalysisResult):
        """結果をキャッシュ"""
        self._volatility_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # キャッシュサイズ制限
        if len(self._volatility_cache) > 50:
            oldest_key = min(self._volatility_cache.keys(), key=lambda k: self._volatility_cache[k]['timestamp'])
            del self._volatility_cache[oldest_key]

    def _create_fallback_result(self) -> VolatilityResult:
        """フォールバック結果生成"""
        return VolatilityResult(
            model=VolatilityModel.HISTORICAL,
            current_volatility=0.02,
            annualized_volatility=0.2,
            volatility_regime=VolatilityRegime.NORMAL,
            confidence=0.1,
            forecast_volatility=0.2,
            volatility_trend='stable',
            additional_metrics={'is_fallback': True},
            calculation_time=datetime.now()
        )

    def _create_fallback_analysis_result(self) -> VolatilityAnalysisResult:
        """フォールバック分析結果生成"""
        fallback_estimate = self._create_fallback_result()
        
        return VolatilityAnalysisResult(
            primary_model=VolatilityModel.HISTORICAL,
            volatility_estimates=[fallback_estimate],
            ensemble_volatility=0.2,
            volatility_regime=VolatilityRegime.NORMAL,
            regime_confidence=0.1,
            risk_metrics={'error': 'fallback'},
            forecast_horizon=self.forecast_horizon,
            analysis_summary={'is_fallback': True},
            analysis_time=datetime.now()
        )

    def clear_cache(self):
        """キャッシュクリア"""
        self._volatility_cache.clear()
        self.logger.info("ボラティリティ分析キャッシュをクリアしました")

    def get_volatility_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ボラティリティサマリー取得"""
        try:
            result = self.analyze_volatility(data)
            return {
                'ensemble_volatility': result.ensemble_volatility,
                'volatility_regime': result.volatility_regime.value,
                'regime_confidence': result.regime_confidence,
                'primary_model': result.primary_model.value,
                'model_count': len(result.volatility_estimates),
                'risk_level': result.analysis_summary.get('risk_assessment', {}).get('risk_level', 'unknown'),
                'analysis_time': result.analysis_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"ボラティリティサマリー取得エラー: {e}")
            return {'error': str(e)}

# 利便性関数
def analyze_volatility_simple(data: pd.DataFrame, models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    簡単なボラティリティ分析関数
    
    Args:
        data: 市場データ
        models: 使用モデル名のリスト
        
    Returns:
        Dict: 分析結果の辞書形式
    """
    # モデル名を列挙型に変換
    if models:
        model_enums = []
        for model_name in models:
            try:
                model_enums.append(VolatilityModel(model_name))
            except ValueError:
                continue
    else:
        model_enums = None
    
    analyzer = VolatilityAnalyzer()
    result = analyzer.analyze_volatility(data, model_enums)
    
    return {
        'ensemble_volatility': result.ensemble_volatility,
        'volatility_regime': result.volatility_regime.value,
        'confidence': result.regime_confidence,
        'primary_model': result.primary_model.value,
        'risk_metrics': result.risk_metrics,
        'analysis_time': result.analysis_time.isoformat()
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== ボラティリティ分析システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # 変動ボラティリティデータ
    base_price = 100
    volatility_changes = []
    
    for i in range(100):
        if i < 30:
            vol = 0.01  # 低ボラティリティ期間
        elif i < 60:
            vol = 0.03  # 高ボラティリティ期間
        else:
            vol = 0.015  # 正常ボラティリティ期間
        
        price_change = np.random.normal(0, vol)
        base_price *= (1 + price_change)
        volatility_changes.append(base_price)
    
    test_data = pd.DataFrame({
        'Open': [p + np.random.normal(0, p*0.001) for p in volatility_changes],
        'High': [p + abs(np.random.normal(0, p*0.002)) for p in volatility_changes],
        'Low': [p - abs(np.random.normal(0, p*0.002)) for p in volatility_changes],
        'Close': volatility_changes,
        'Volume': [np.random.uniform(1000000, 5000000) for _ in range(100)]
    }, index=dates)
    
    # 分析器テスト
    analyzer = VolatilityAnalyzer()
    
    print("\n1. 全モデル分析")
    result = analyzer.analyze_volatility(test_data)
    print(f"アンサンブルボラティリティ: {result.ensemble_volatility:.3f}")
    print(f"ボラティリティレジーム: {result.volatility_regime.value}")
    print(f"信頼度: {result.regime_confidence:.3f}")
    print(f"使用モデル数: {len(result.volatility_estimates)}")
    
    print("\n2. 個別モデル結果")
    for est in result.volatility_estimates:
        print(f"  {est.model.value}: {est.annualized_volatility:.3f} ({est.volatility_trend})")
    
    print("\n3. リスクメトリクス")
    for metric, value in result.risk_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n4. 簡単分析関数テスト")
    simple_result = analyze_volatility_simple(test_data, ['historical', 'ewma'])
    print(f"簡単分析結果: {simple_result['volatility_regime']} (ボラティリティ: {simple_result['ensemble_volatility']:.3f})")
    
    print("\n=== テスト完了 ===")
