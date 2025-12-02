"""
高度VaRエンジン - 5-3-2「ポートフォリオVaR計算」のメインエンジン

複数手法によるVaR計算を統合し、動的な市場環境対応を実現
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 数値計算ライブラリ
try:
    from scipy import stats
    from sklearn.covariance import LedoitWolf
    ADVANCED_STATS = True
except ImportError:
    ADVANCED_STATS = False
    warnings.warn("Advanced statistical libraries not available. Some features will be limited.")

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VaRCalculationConfig:
    """VaR計算設定"""
    primary_method: str = "hybrid"          # hybrid, parametric, historical, monte_carlo
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 22])  # 営業日
    portfolio_components: List[str] = field(default_factory=list)
    
    # ヒストリカル手法設定
    historical_window: int = 252            # 1年
    min_historical_periods: int = 60        # 最小期間
    
    # モンテカルロ設定
    monte_carlo_simulations: int = 10000    # シミュレーション回数
    random_seed: int = 42                   # 再現性確保
    
    # パラメトリック設定
    distribution_type: str = "student_t"    # normal, student_t, skewed_t
    volatility_model: str = "garch"         # ewma, garch, realized
    
    # ハイブリッド設定
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "parametric": 0.3,
        "historical": 0.4,
        "monte_carlo": 0.3
    })
    
    # 動的調整設定
    adaptive_window: bool = True            # 動的ウィンドウ
    regime_detection: bool = True           # レジーム検出
    extreme_value_adjustment: bool = True   # 極値調整

@dataclass
class VaRResult:
    """VaR計算結果"""
    timestamp: datetime
    portfolio_composition: Dict[str, float]
    var_estimates: Dict[str, float]  # confidence_level -> VaR value
    cvar_estimates: Dict[str, float]  # confidence_level -> CVaR value
    calculation_method: str
    market_regime: Optional[str] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    component_vars: Optional[Dict[str, float]] = None
    marginal_vars: Optional[Dict[str, float]] = None
    diversification_benefit: float = 0.0
    
    def get_var_95(self) -> float:
        """95% VaRを取得"""
        return self.var_estimates.get('95', 0.0)
    
    def get_var_99(self) -> float:
        """99% VaRを取得"""
        return self.var_estimates.get('99', 0.0)

class VaRCalculationMethod(Enum):
    """VaR計算手法"""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    HYBRID = "hybrid"

class AdvancedVaREngine:
    """高度VaRエンジン"""
    
    def __init__(self, config: VaRCalculationConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # 計算履歴
        self.calculation_history: List[VaRResult] = []
        
        # レジーム検出用データ
        self.regime_history: List[Dict[str, Any]] = []
        
        self.logger.info("AdvancedVaREngine initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.AdvancedVaREngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def calculate_comprehensive_var(self,
                                   returns_data: pd.DataFrame,
                                   weights: Dict[str, float],
                                   market_conditions: Optional[Dict[str, Any]] = None) -> VaRResult:
        """包括的VaR計算"""
        try:
            self.logger.info("Starting comprehensive VaR calculation")
            
            # データ前処理
            processed_data = self._preprocess_data(returns_data, weights)
            
            if processed_data.empty:
                return self._create_default_result(weights)
            
            # 市場レジーム検出
            current_regime = self._detect_market_regime(processed_data, market_conditions)
            
            # 最適計算ウィンドウの決定
            optimal_window = self._determine_optimal_window(processed_data, current_regime)
            
            # 各手法でVaR計算
            var_estimates = {}
            cvar_estimates = {}
            
            for confidence_level in self.config.confidence_levels:
                if self.config.primary_method == "hybrid":
                    var_value, cvar_value = self._calculate_hybrid_var(
                        processed_data.tail(optimal_window), weights, confidence_level
                    )
                elif self.config.primary_method == "parametric":
                    var_value, cvar_value = self._calculate_parametric_var(
                        processed_data.tail(optimal_window), weights, confidence_level
                    )
                elif self.config.primary_method == "historical":
                    var_value, cvar_value = self._calculate_historical_var(
                        processed_data.tail(optimal_window), weights, confidence_level
                    )
                else:  # monte_carlo
                    var_value, cvar_value = self._calculate_monte_carlo_var(
                        processed_data.tail(optimal_window), weights, confidence_level
                    )
                
                var_estimates[str(int(confidence_level * 100))] = var_value
                cvar_estimates[str(int(confidence_level * 100))] = cvar_value
            
            # Component VaR計算
            component_vars = self._calculate_component_vars(processed_data, weights, 0.95)
            
            # Marginal VaR計算  
            marginal_vars = self._calculate_marginal_vars(processed_data, weights, 0.95)
            
            # 分散効果計算
            diversification_benefit = self._calculate_diversification_benefit(
                component_vars, var_estimates.get('95', 0.0)
            )
            
            # 結果作成
            result = VaRResult(
                timestamp=datetime.now(),
                portfolio_composition=weights.copy(),
                var_estimates=var_estimates,
                cvar_estimates=cvar_estimates,
                calculation_method=self.config.primary_method,
                market_regime=current_regime.get("regime_type"),
                component_vars=component_vars,
                marginal_vars=marginal_vars,
                diversification_benefit=diversification_benefit
            )
            
            # 履歴に追加
            self.calculation_history.append(result)
            
            self.logger.info(f"VaR calculation completed. VaR 95%: {result.get_var_95():.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"VaR calculation error: {e}")
            return self._create_default_result(weights)
    
    def _preprocess_data(self, returns_data: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """データ前処理"""
        try:
            if returns_data.empty:
                return pd.DataFrame()
            
            # NaN値処理
            returns_data = returns_data.fillna(0.0)
            
            # 無限値処理
            returns_data = returns_data.replace([np.inf, -np.inf], 0.0)
            
            # 重みに含まれる戦略のみを抽出
            available_strategies = list(set(returns_data.columns) & set(weights.keys()))
            
            if not available_strategies:
                return pd.DataFrame()
            
            return returns_data[available_strategies].copy()
            
        except Exception as e:
            self.logger.error(f"Data preprocessing error: {e}")
            return pd.DataFrame()
    
    def _detect_market_regime(self, returns_data: pd.DataFrame, 
                             market_conditions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """市場レジーム検出"""
        try:
            if returns_data.empty:
                return {"regime_type": "normal", "confidence": 0.5}
            
            # 簡易レジーム検出
            recent_returns = returns_data.tail(22)  # 過去1ヶ月
            
            # ボラティリティ分析
            recent_vol = recent_returns.std().mean()
            historical_vol = returns_data.std().mean()
            
            vol_ratio = recent_vol / (historical_vol + 1e-8)
            
            # レジーム判定
            if vol_ratio > 1.5:
                regime = "high_volatility"
                confidence = min(0.9, vol_ratio / 2.0)
            elif vol_ratio < 0.7:
                regime = "low_volatility"
                confidence = min(0.9, 2.0 - vol_ratio)
            else:
                regime = "normal"
                confidence = 0.6
            
            return {
                "regime_type": regime,
                "confidence": confidence,
                "volatility_ratio": vol_ratio,
                "detected_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Regime detection error: {e}")
            return {"regime_type": "normal", "confidence": 0.5}
    
    def _determine_optimal_window(self, returns_data: pd.DataFrame, 
                                 regime_info: Dict[str, Any]) -> int:
        """最適計算ウィンドウの決定"""
        try:
            base_window = self.config.historical_window
            min_window = self.config.min_historical_periods
            
            # レジーム基づく調整
            regime_type = regime_info.get("regime_type", "normal")
            
            if regime_type == "high_volatility":
                # 高ボラ時は短いウィンドウ
                adjusted_window = int(base_window * 0.7)
            elif regime_type == "low_volatility":
                # 低ボラ時は長いウィンドウ
                adjusted_window = int(base_window * 1.2)
            else:
                adjusted_window = base_window
            
            # データ量制約
            available_data = len(returns_data)
            optimal_window = min(adjusted_window, available_data)
            
            return max(optimal_window, min_window)
            
        except Exception as e:
            self.logger.error(f"Optimal window determination error: {e}")
            return self.config.historical_window
    
    def _calculate_hybrid_var(self, returns_data: pd.DataFrame, 
                             weights: Dict[str, float], 
                             confidence_level: float) -> Tuple[float, float]:
        """ハイブリッドVaR計算"""
        try:
            # 各手法で計算
            param_var, param_cvar = self._calculate_parametric_var(returns_data, weights, confidence_level)
            hist_var, hist_cvar = self._calculate_historical_var(returns_data, weights, confidence_level)
            mc_var, mc_cvar = self._calculate_monte_carlo_var(returns_data, weights, confidence_level)
            
            # 重み付き平均
            method_weights = self.config.method_weights
            
            hybrid_var = (
                param_var * method_weights.get("parametric", 0.3) +
                hist_var * method_weights.get("historical", 0.4) +
                mc_var * method_weights.get("monte_carlo", 0.3)
            )
            
            hybrid_cvar = (
                param_cvar * method_weights.get("parametric", 0.3) +
                hist_cvar * method_weights.get("historical", 0.4) +
                mc_var * method_weights.get("monte_carlo", 0.3)
            )
            
            return hybrid_var, hybrid_cvar
            
        except Exception as e:
            self.logger.error(f"Hybrid VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_parametric_var(self, returns_data: pd.DataFrame, 
                                 weights: Dict[str, float], 
                                 confidence_level: float) -> Tuple[float, float]:
        """パラメトリックVaR計算"""
        try:
            # ポートフォリオリターン計算
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0, 0.0
            
            # 統計的パラメータ
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            if std_return == 0:
                return 0.0, 0.0
            
            # 分布仮定に基づくVaR計算
            if self.config.distribution_type == "student_t":
                # t分布
                if ADVANCED_STATS and len(portfolio_returns) > 10:
                    # 自由度推定
                    df = max(3, len(portfolio_returns) // 10)
                    var_quantile = stats.t.ppf(1 - confidence_level, df)
                else:
                    var_quantile = stats.norm.ppf(1 - confidence_level)
            else:
                # 正規分布
                var_quantile = stats.norm.ppf(1 - confidence_level)
            
            # VaR計算
            var_value = abs(mean_return + std_return * var_quantile)
            
            # CVaR計算（期待ショートフォール）
            if self.config.distribution_type == "student_t" and ADVANCED_STATS:
                # t分布のCVaR（近似）
                cvar_value = var_value * 1.2
            else:
                # 正規分布のCVaR
                phi = stats.norm.pdf(var_quantile)
                cvar_value = abs(mean_return + std_return * phi / (1 - confidence_level))
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_historical_var(self, returns_data: pd.DataFrame, 
                                 weights: Dict[str, float], 
                                 confidence_level: float) -> Tuple[float, float]:
        """ヒストリカルVaR計算"""
        try:
            # ポートフォリオリターン計算
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0, 0.0
            
            # VaR計算
            var_value = abs(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
            
            # CVaR計算
            tail_losses = portfolio_returns[portfolio_returns <= -var_value]
            if len(tail_losses) > 0:
                cvar_value = abs(np.mean(tail_losses))
            else:
                cvar_value = var_value * 1.2
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Historical VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_monte_carlo_var(self, returns_data: pd.DataFrame, 
                                  weights: Dict[str, float], 
                                  confidence_level: float) -> Tuple[float, float]:
        """モンテカルロVaR計算"""
        try:
            # ポートフォリオリターン計算
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.0, 0.0
            
            # 統計的パラメータ
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            if std_return == 0:
                return 0.0, 0.0
            
            # モンテカルロシミュレーション
            np.random.seed(self.config.random_seed)
            simulated_returns = np.random.normal(
                mean_return, std_return, self.config.monte_carlo_simulations
            )
            
            # VaR計算
            var_value = abs(np.percentile(simulated_returns, (1 - confidence_level) * 100))
            
            # CVaR計算
            tail_losses = simulated_returns[simulated_returns <= -var_value]
            if len(tail_losses) > 0:
                cvar_value = abs(np.mean(tail_losses))
            else:
                cvar_value = var_value * 1.2
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Monte Carlo VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame, 
                                   weights: Dict[str, float]) -> np.ndarray:
        """ポートフォリオリターン計算"""
        try:
            if returns_data.empty or not weights:
                return np.array([])
            
            # 重みの正規化
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # ポートフォリオリターン計算
            portfolio_returns = np.zeros(len(returns_data))
            
            for strategy, weight in normalized_weights.items():
                if strategy in returns_data.columns:
                    strategy_returns = returns_data[strategy].fillna(0.0).values
                    portfolio_returns += strategy_returns * weight
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])
    
    def _calculate_component_vars(self, returns_data: pd.DataFrame, 
                                 weights: Dict[str, float], 
                                 confidence_level: float) -> Dict[str, float]:
        """Component VaR計算"""
        try:
            component_vars = {}
            
            # ベースラインVaR
            baseline_var, _ = self._calculate_historical_var(returns_data, weights, confidence_level)
            
            for strategy in weights.keys():
                if strategy in returns_data.columns and weights[strategy] != 0:
                    # 戦略を除いた場合のVaR
                    modified_weights = weights.copy()
                    removed_weight = modified_weights.pop(strategy, 0.0)
                    
                    if modified_weights:
                        # 残りの重みを再正規化
                        remaining_total = sum(modified_weights.values())
                        if remaining_total > 0:
                            modified_weights = {k: v / remaining_total * (1 - abs(removed_weight)) 
                                             for k, v in modified_weights.items()}
                        
                        modified_var, _ = self._calculate_historical_var(
                            returns_data, modified_weights, confidence_level
                        )
                        
                        # Component VaR = Baseline VaR - Modified VaR
                        component_var = baseline_var - modified_var
                    else:
                        component_var = baseline_var
                    
                    component_vars[strategy] = component_var
            
            return component_vars
            
        except Exception as e:
            self.logger.error(f"Component VaR calculation error: {e}")
            return {}
    
    def _calculate_marginal_vars(self, returns_data: pd.DataFrame, 
                                weights: Dict[str, float], 
                                confidence_level: float) -> Dict[str, float]:
        """Marginal VaR計算"""
        try:
            marginal_vars = {}
            epsilon = 0.01  # 微小変更量
            
            # ベースラインVaR
            baseline_var, _ = self._calculate_historical_var(returns_data, weights, confidence_level)
            
            for strategy in weights.keys():
                if strategy in returns_data.columns:
                    # 微小変更後の重み
                    modified_weights = weights.copy()
                    modified_weights[strategy] = modified_weights.get(strategy, 0.0) + epsilon
                    
                    # 修正後のVaR
                    modified_var, _ = self._calculate_historical_var(
                        returns_data, modified_weights, confidence_level
                    )
                    
                    # Marginal VaR = (Modified VaR - Baseline VaR) / epsilon
                    marginal_var = (modified_var - baseline_var) / epsilon
                    marginal_vars[strategy] = marginal_var
            
            return marginal_vars
            
        except Exception as e:
            self.logger.error(f"Marginal VaR calculation error: {e}")
            return {}
    
    def _calculate_diversification_benefit(self, component_vars: Dict[str, float], 
                                          portfolio_var: float) -> float:
        """分散効果計算"""
        try:
            if not component_vars or portfolio_var == 0:
                return 0.0
            
            # Individual VaRの合計
            sum_individual_vars = sum(abs(cv) for cv in component_vars.values())
            
            if sum_individual_vars == 0:
                return 0.0
            
            # 分散効果 = (Individual VaRの合計 - ポートフォリオVaR) / Individual VaRの合計
            diversification_benefit = (sum_individual_vars - portfolio_var) / sum_individual_vars
            
            return max(0.0, min(1.0, diversification_benefit))
            
        except Exception as e:
            self.logger.error(f"Diversification benefit calculation error: {e}")
            return 0.0
    
    def _create_default_result(self, weights: Dict[str, float]) -> VaRResult:
        """デフォルト結果の作成"""
        return VaRResult(
            timestamp=datetime.now(),
            portfolio_composition=weights.copy(),
            var_estimates={'95': 0.05, '99': 0.08, '99.5': 0.10},
            cvar_estimates={'95': 0.08, '99': 0.12, '99.5': 0.15},
            calculation_method="default",
            market_regime="normal",
            diversification_benefit=0.0
        )
    
    def get_calculation_history(self, days: int = 30) -> List[VaRResult]:
        """計算履歴の取得"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            return [
                result for result in self.calculation_history
                if result.timestamp >= cutoff_date
            ]
        except Exception as e:
            self.logger.error(f"Get calculation history error: {e}")
            return []
    
    def export_var_report(self, output_path: str) -> bool:
        """VaRレポートのエクスポート"""
        try:
            if not self.calculation_history:
                self.logger.warning("No calculation history to export")
                return False
            
            # レポートデータ作成
            report_data = []
            for result in self.calculation_history:
                report_data.append({
                    'timestamp': result.timestamp.isoformat(),
                    'var_95': result.get_var_95(),
                    'var_99': result.get_var_99(),
                    'method': result.calculation_method,
                    'regime': result.market_regime,
                    'diversification_benefit': result.diversification_benefit
                })
            
            # JSONファイルに保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"VaR report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export VaR report error: {e}")
            return False
