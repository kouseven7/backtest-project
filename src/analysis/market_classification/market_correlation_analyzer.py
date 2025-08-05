"""
市場相関分析システム - A→B市場分類システム基盤
複数資産間の相関関係と市場連動性分析機能を提供
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import math

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength

class CorrelationMethod(Enum):
    """相関分析手法"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    DYNAMIC = "dynamic"
    REGIME_DEPENDENT = "regime_dependent"

class CorrelationRegime(Enum):
    """相関レジーム"""
    LOW_CORRELATION = "low_correlation"
    MODERATE_CORRELATION = "moderate_correlation"
    HIGH_CORRELATION = "high_correlation"
    CRISIS_CORRELATION = "crisis_correlation"
    DECOUPLING = "decoupling"

class MarketSector(Enum):
    """市場セクター（拡張用）"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    ALTERNATIVE = "alternative"

@dataclass
class CorrelationResult:
    """相関分析結果"""
    method: CorrelationMethod
    correlation_matrix: pd.DataFrame
    correlation_regime: CorrelationRegime
    dominant_correlations: Dict[str, float]
    cluster_analysis: Dict[str, Any]
    regime_stability: float
    confidence: float
    analysis_period: int
    calculation_time: datetime
    
    def __post_init__(self):
        if self.dominant_correlations is None:
            self.dominant_correlations = {}
        if self.cluster_analysis is None:
            self.cluster_analysis = {}

@dataclass
class MarketCorrelationAnalysis:
    """市場相関分析の総合結果"""
    primary_method: CorrelationMethod
    correlation_results: List[CorrelationResult]
    ensemble_correlation: pd.DataFrame
    market_regime: CorrelationRegime
    sector_analysis: Dict[str, Any]
    risk_implications: Dict[str, float]
    diversification_metrics: Dict[str, float]
    analysis_summary: Dict[str, Any]
    analysis_time: datetime

class MarketCorrelationAnalyzer:
    """
    市場相関分析システムのメインクラス
    複数資産間の相関関係と市場構造分析を提供
    """
    
    def __init__(self, 
                 default_window: int = 60,
                 rolling_window: int = 30,
                 correlation_threshold: float = 0.3,
                 crisis_threshold: float = 0.7):
        """
        市場相関分析器の初期化
        
        Args:
            default_window: デフォルト分析期間
            rolling_window: ローリング分析期間
            correlation_threshold: 相関レジーム判定閾値
            crisis_threshold: 危機時相関閾値
        """
        self.default_window = default_window
        self.rolling_window = rolling_window
        self.correlation_threshold = correlation_threshold
        self.crisis_threshold = crisis_threshold
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # 相関レジーム閾値
        self.regime_thresholds = {
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.7,
            'crisis': 0.85
        }
        
        # 分析結果キャッシュ
        self._correlation_cache = {}
        self._cache_timeout = timedelta(minutes=15)
        
        self.logger.info("MarketCorrelationAnalyzer初期化完了")

    def analyze_market_correlations(self, 
                                  data: Dict[str, pd.DataFrame],
                                  methods: Optional[List[CorrelationMethod]] = None,
                                  sector_mapping: Optional[Dict[str, MarketSector]] = None,
                                  custom_params: Optional[Dict] = None) -> MarketCorrelationAnalysis:
        """
        市場相関の総合分析
        
        Args:
            data: 資産別市場データ辞書 {asset_name: DataFrame}
            methods: 使用する分析手法 (None=全手法)
            sector_mapping: 資産のセクター分類
            custom_params: カスタムパラメータ
            
        Returns:
            MarketCorrelationAnalysis: 分析結果
        """
        try:
            # データ検証
            if not self._validate_data_dict(data):
                raise ValueError("無効なデータフォーマット")
            
            # キャッシュチェック
            cache_key = self._generate_cache_key(data, methods)
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"キャッシュから結果を返却: {cache_key}")
                return self._correlation_cache[cache_key]['result']
            
            # 手法設定
            if methods is None:
                methods = [CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN, 
                          CorrelationMethod.ROLLING, CorrelationMethod.DYNAMIC]
            
            # パラメータ統合
            params = self._merge_params(custom_params)
            
            # 価格データの統合・前処理
            combined_data = self._combine_price_data(data)
            
            # 各手法で相関分析
            correlation_results = []
            
            for method in methods:
                try:
                    result = self._calculate_correlation(combined_data, method, params)
                    if result:
                        correlation_results.append(result)
                except Exception as e:
                    self.logger.warning(f"手法 {method.value} でエラー: {e}")
                    continue
            
            if not correlation_results:
                return self._create_fallback_analysis()
            
            # アンサンブル相関マトリックス
            ensemble_correlation = self._calculate_ensemble_correlation(correlation_results)
            
            # 市場レジーム判定
            market_regime = self._determine_correlation_regime(ensemble_correlation)
            
            # セクター分析
            sector_analysis = self._analyze_sectors(ensemble_correlation, sector_mapping)
            
            # リスク含意分析
            risk_implications = self._analyze_risk_implications(ensemble_correlation, combined_data)
            
            # 分散投資メトリクス
            diversification_metrics = self._calculate_diversification_metrics(ensemble_correlation)
            
            # 分析サマリー
            analysis_summary = self._create_correlation_summary(
                correlation_results, ensemble_correlation, market_regime
            )
            
            # 主要手法選択（最高信頼度）
            primary_method = max(correlation_results, key=lambda x: x.confidence).method
            
            result = MarketCorrelationAnalysis(
                primary_method=primary_method,
                correlation_results=correlation_results,
                ensemble_correlation=ensemble_correlation,
                market_regime=market_regime,
                sector_analysis=sector_analysis,
                risk_implications=risk_implications,
                diversification_metrics=diversification_metrics,
                analysis_summary=analysis_summary,
                analysis_time=datetime.now()
            )
            
            # 結果をキャッシュ
            self._cache_result(cache_key, result)
            
            self.logger.info(f"市場相関分析完了: {market_regime.value} (手法数: {len(correlation_results)})")
            return result
            
        except Exception as e:
            self.logger.error(f"市場相関分析エラー: {e}")
            return self._create_fallback_analysis()

    def _combine_price_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """価格データの統合"""
        try:
            price_data = {}
            
            for asset_name, df in data.items():
                if 'Close' in df.columns:
                    price_data[asset_name] = df['Close']
                else:
                    self.logger.warning(f"{asset_name}: Close列が見つかりません")
                    continue
            
            if not price_data:
                raise ValueError("有効な価格データがありません")
            
            # データフレーム結合（外部結合でNaNを許可）
            combined = pd.DataFrame(price_data)
            
            # 共通の期間のみ使用
            combined = combined.dropna()
            
            if len(combined) < self.default_window:
                self.logger.warning(f"データ期間が短すぎます: {len(combined)}日")
            
            return combined
            
        except Exception as e:
            self.logger.error(f"価格データ統合エラー: {e}")
            raise

    def _calculate_correlation(self, 
                             data: pd.DataFrame, 
                             method: CorrelationMethod, 
                             params: Dict) -> Optional[CorrelationResult]:
        """個別手法での相関計算"""
        try:
            if method == CorrelationMethod.PEARSON:
                return self._pearson_correlation(data, params)
            elif method == CorrelationMethod.SPEARMAN:
                return self._spearman_correlation(data, params)
            elif method == CorrelationMethod.KENDALL:
                return self._kendall_correlation(data, params)
            elif method == CorrelationMethod.ROLLING:
                return self._rolling_correlation(data, params)
            elif method == CorrelationMethod.DYNAMIC:
                return self._dynamic_correlation(data, params)
            elif method == CorrelationMethod.REGIME_DEPENDENT:
                return self._regime_dependent_correlation(data, params)
            else:
                self.logger.warning(f"未対応の相関分析手法: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"{method.value} 相関計算エラー: {e}")
            return None

    def _pearson_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """ピアソン相関分析"""
        try:
            # リターン計算
            returns = data.pct_change().dropna()
            
            # ピアソン相関マトリックス
            correlation_matrix = returns.corr(method='pearson')
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価
            if len(returns) > self.rolling_window:
                rolling_corrs = []
                for i in range(self.rolling_window, len(returns)):
                    window_data = returns.iloc[i-self.rolling_window:i]
                    rolling_corr = window_data.corr(method='pearson')
                    avg_corr = self._calculate_average_correlation(rolling_corr)
                    rolling_corrs.append(avg_corr)
                
                regime_stability = 1 - np.std(rolling_corrs) if rolling_corrs else 0.5
            else:
                regime_stability = 0.5
            
            # 信頼度（データ量とピアソン相関の仮定適合性）
            confidence = min(len(returns) / (self.default_window * 1.5), 1.0) * 0.9
            
            return CorrelationResult(
                method=CorrelationMethod.PEARSON,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ピアソン相関計算エラー: {e}")
            raise

    def _spearman_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """スピアマン順位相関分析"""
        try:
            # リターン計算
            returns = data.pct_change().dropna()
            
            # スピアマン相関マトリックス
            correlation_matrix = returns.corr(method='spearman')
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価
            regime_stability = 0.7  # スピアマンは一般的に安定
            
            # 信頼度（順位相関は外れ値に強い）
            confidence = min(len(returns) / self.default_window, 1.0) * 0.85
            
            return CorrelationResult(
                method=CorrelationMethod.SPEARMAN,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"スピアマン相関計算エラー: {e}")
            raise

    def _kendall_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """ケンドール順位相関分析"""
        try:
            # リターン計算
            returns = data.pct_change().dropna()
            
            # ケンドール相関マトリックス（計算コストが高いため小さなデータセットに制限）
            if len(returns.columns) > 10:
                self.logger.warning("ケンドール相関: 資産数が多すぎます、最初の10資産のみ使用")
                returns = returns.iloc[:, :10]
            
            correlation_matrix = returns.corr(method='kendall')
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価
            regime_stability = 0.8  # ケンドールは非常に安定
            
            # 信頼度
            confidence = min(len(returns) / self.default_window, 1.0) * 0.8
            
            return CorrelationResult(
                method=CorrelationMethod.KENDALL,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ケンドール相関計算エラー: {e}")
            raise

    def _rolling_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """ローリング相関分析"""
        try:
            window = params.get('rolling_window', self.rolling_window)
            returns = data.pct_change().dropna()
            
            if len(returns) < window * 2:
                # フォールバックとしてピアソン相関を使用
                return self._pearson_correlation(data, params)
            
            # ローリング相関の計算
            rolling_correlations = []
            assets = returns.columns
            
            for i in range(window, len(returns)):
                window_data = returns.iloc[i-window:i]
                corr_matrix = window_data.corr(method='pearson')
                rolling_correlations.append(corr_matrix)
            
            # 最新の相関マトリックス
            correlation_matrix = rolling_correlations[-1]
            
            # 時間変動相関の分析
            correlation_evolution = self._analyze_correlation_evolution(rolling_correlations, assets)
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            cluster_analysis['evolution'] = correlation_evolution
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価（相関の時間変動）
            avg_correlations_over_time = [
                self._calculate_average_correlation(corr) for corr in rolling_correlations
            ]
            regime_stability = 1 - (np.std(avg_correlations_over_time) / np.mean(avg_correlations_over_time))
            regime_stability = max(0, min(regime_stability, 1))
            
            # 信頼度
            confidence = min(len(rolling_correlations) / 20, 1.0) * 0.8
            
            return CorrelationResult(
                method=CorrelationMethod.ROLLING,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ローリング相関計算エラー: {e}")
            raise

    def _dynamic_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """動的相関分析（EWMA）"""
        try:
            lambda_param = params.get('ewma_lambda', 0.94)
            returns = data.pct_change().dropna()
            
            if len(returns) < 20:
                # フォールバックとしてピアソン相関を使用
                return self._pearson_correlation(data, params)
            
            # EWMA相関の計算
            assets = returns.columns
            n_assets = len(assets)
            
            # 初期共分散マトリックス
            cov_matrix = returns.cov().values
            
            # EWMA更新
            ewma_covs = []
            for i in range(len(returns)):
                if i == 0:
                    ewma_cov = cov_matrix
                else:
                    # 現在のリターンベクトル
                    r_t = returns.iloc[i].values.reshape(-1, 1)
                    # EWMA更新式
                    ewma_cov = lambda_param * ewma_cov + (1 - lambda_param) * np.dot(r_t, r_t.T)
                
                ewma_covs.append(ewma_cov.copy())
            
            # 最新の共分散マトリックスから相関マトリックスを計算
            latest_cov = ewma_covs[-1]
            std_devs = np.sqrt(np.diag(latest_cov))
            correlation_matrix = latest_cov / np.outer(std_devs, std_devs)
            
            # DataFrameに変換
            correlation_matrix = pd.DataFrame(correlation_matrix, index=assets, columns=assets)
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価（最新の相関変動）
            if len(ewma_covs) > 10:
                recent_corrs = []
                for cov in ewma_covs[-10:]:
                    std_devs = np.sqrt(np.diag(cov))
                    corr = cov / np.outer(std_devs, std_devs)
                    avg_corr = np.mean(corr[np.triu_indices_from(corr, k=1)])
                    recent_corrs.append(avg_corr)
                
                regime_stability = 1 - np.std(recent_corrs) if recent_corrs else 0.5
            else:
                regime_stability = 0.5
            
            # 信頼度
            confidence = min(len(returns) / self.default_window, 1.0) * 0.85
            
            return CorrelationResult(
                method=CorrelationMethod.DYNAMIC,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"動的相関計算エラー: {e}")
            raise

    def _regime_dependent_correlation(self, data: pd.DataFrame, params: Dict) -> CorrelationResult:
        """レジーム依存相関分析"""
        try:
            returns = data.pct_change().dropna()
            
            # ボラティリティによるレジーム分類
            volatility = returns.std(axis=1).rolling(20).mean()
            vol_threshold = volatility.quantile(0.7)  # 上位30%を高ボラティリティ
            
            high_vol_periods = volatility > vol_threshold
            low_vol_periods = ~high_vol_periods
            
            # 各レジームでの相関計算
            if high_vol_periods.sum() > 10 and low_vol_periods.sum() > 10:
                high_vol_corr = returns[high_vol_periods].corr(method='pearson')
                low_vol_corr = returns[low_vol_periods].corr(method='pearson')
                
                # 現在のレジーム判定（最近の期間のボラティリティ）
                recent_vol = volatility.tail(5).mean()
                if recent_vol > vol_threshold:
                    current_regime_corr = high_vol_corr
                    current_regime = "high_volatility"
                else:
                    current_regime_corr = low_vol_corr
                    current_regime = "low_volatility"
                
                correlation_matrix = current_regime_corr
                
                # レジーム間の相関差異分析
                regime_difference = high_vol_corr - low_vol_corr
                max_difference = np.abs(regime_difference.values).max()
                
            else:
                # 十分なデータがない場合はピアソン相関を使用
                correlation_matrix = returns.corr(method='pearson')
                current_regime = "insufficient_data"
                max_difference = 0
            
            # 支配的相関の特定
            dominant_correlations = self._extract_dominant_correlations(correlation_matrix)
            
            # クラスター分析
            cluster_analysis = self._perform_cluster_analysis(correlation_matrix)
            cluster_analysis['current_regime'] = current_regime
            cluster_analysis['regime_difference'] = max_difference if 'max_difference' in locals() else 0
            
            # レジーム判定
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            correlation_regime = self._classify_correlation_regime(avg_correlation)
            
            # 安定性評価（レジーム依存性が高いほど不安定）
            regime_stability = 1 - min(max_difference if 'max_difference' in locals() else 0, 0.5)
            
            # 信頼度
            confidence = min(len(returns) / self.default_window, 1.0) * 0.75
            
            return CorrelationResult(
                method=CorrelationMethod.REGIME_DEPENDENT,
                correlation_matrix=correlation_matrix,
                correlation_regime=correlation_regime,
                dominant_correlations=dominant_correlations,
                cluster_analysis=cluster_analysis,
                regime_stability=regime_stability,
                confidence=confidence,
                analysis_period=len(returns),
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"レジーム依存相関計算エラー: {e}")
            raise

    def _extract_dominant_correlations(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """支配的相関の抽出"""
        try:
            correlations = {}
            assets = correlation_matrix.columns
            
            # 上三角マトリックスから相関値を取得
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    pair = f"{assets[i]}-{assets[j]}"
                    correlations[pair] = correlation_matrix.iloc[i, j]
            
            # 絶対値で並び替え
            sorted_correlations = dict(sorted(correlations.items(), 
                                            key=lambda x: abs(x[1]), reverse=True))
            
            # 上位の相関のみ返却
            top_correlations = dict(list(sorted_correlations.items())[:5])
            
            return top_correlations
            
        except Exception as e:
            self.logger.error(f"支配的相関抽出エラー: {e}")
            return {}

    def _perform_cluster_analysis(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """クラスター分析"""
        try:
            # 距離マトリックス（1 - |correlation|）
            distance_matrix = 1 - np.abs(correlation_matrix.values)
            
            # 階層クラスタリング
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # クラスター数の決定（2-5クラスター）
            n_clusters = min(max(2, len(correlation_matrix.columns) // 3), 5)
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # クラスター結果
            cluster_assignments = {}
            for i, asset in enumerate(correlation_matrix.columns):
                cluster_assignments[asset] = int(clusters[i])
            
            # クラスター内平均相関
            cluster_correlations = {}
            for cluster_id in range(1, n_clusters + 1):
                cluster_assets = [asset for asset, cid in cluster_assignments.items() if cid == cluster_id]
                if len(cluster_assets) > 1:
                    cluster_corr_subset = correlation_matrix.loc[cluster_assets, cluster_assets]
                    cluster_correlations[f"cluster_{cluster_id}"] = self._calculate_average_correlation(cluster_corr_subset)
                else:
                    cluster_correlations[f"cluster_{cluster_id}"] = 0
            
            return {
                'cluster_assignments': cluster_assignments,
                'cluster_correlations': cluster_correlations,
                'n_clusters': n_clusters,
                'linkage_matrix': linkage_matrix.tolist()  # JSONシリアライズ可能に
            }
            
        except Exception as e:
            self.logger.error(f"クラスター分析エラー: {e}")
            return {
                'cluster_assignments': {},
                'cluster_correlations': {},
                'n_clusters': 0,
                'error': str(e)
            }

    def _analyze_correlation_evolution(self, rolling_correlations: List[pd.DataFrame], assets: List[str]) -> Dict[str, Any]:
        """相関の時間発展分析"""
        try:
            evolution_metrics = {}
            
            # 各資産ペアの相関変化
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    pair = f"{assets[i]}-{assets[j]}"
                    correlations_over_time = [corr.iloc[i, j] for corr in rolling_correlations]
                    
                    evolution_metrics[pair] = {
                        'mean': np.mean(correlations_over_time),
                        'std': np.std(correlations_over_time),
                        'trend': np.polyfit(range(len(correlations_over_time)), correlations_over_time, 1)[0],
                        'min': np.min(correlations_over_time),
                        'max': np.max(correlations_over_time)
                    }
            
            # 全体の相関変動
            avg_correlations = [self._calculate_average_correlation(corr) for corr in rolling_correlations]
            
            overall_evolution = {
                'mean_correlation_trend': np.polyfit(range(len(avg_correlations)), avg_correlations, 1)[0],
                'correlation_volatility': np.std(avg_correlations),
                'correlation_range': [np.min(avg_correlations), np.max(avg_correlations)]
            }
            
            return {
                'pair_evolution': evolution_metrics,
                'overall_evolution': overall_evolution
            }
            
        except Exception as e:
            self.logger.error(f"相関発展分析エラー: {e}")
            return {}

    def _calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """相関マトリックスの平均相関"""
        try:
            # 上三角マトリックス（対角線を除く）の平均
            upper_triangle = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
            return np.mean(upper_triangle)
        except:
            return 0.0

    def _classify_correlation_regime(self, avg_correlation: float) -> CorrelationRegime:
        """相関レジームの分類"""
        try:
            abs_corr = abs(avg_correlation)
            
            if abs_corr >= self.regime_thresholds['crisis']:
                return CorrelationRegime.CRISIS_CORRELATION
            elif abs_corr >= self.regime_thresholds['high']:
                return CorrelationRegime.HIGH_CORRELATION
            elif abs_corr >= self.regime_thresholds['moderate']:
                return CorrelationRegime.MODERATE_CORRELATION
            elif abs_corr >= self.regime_thresholds['low']:
                return CorrelationRegime.LOW_CORRELATION
            else:
                return CorrelationRegime.DECOUPLING
                
        except:
            return CorrelationRegime.MODERATE_CORRELATION

    def _calculate_ensemble_correlation(self, results: List[CorrelationResult]) -> pd.DataFrame:
        """アンサンブル相関マトリックス計算"""
        try:
            if not results:
                return pd.DataFrame()
            
            # 信頼度重み付き平均
            weighted_matrices = []
            total_weight = 0
            
            for result in results:
                weight = result.confidence
                weighted_matrices.append(result.correlation_matrix * weight)
                total_weight += weight
            
            if total_weight == 0:
                return results[0].correlation_matrix
            
            # 重み付き平均
            ensemble_matrix = sum(weighted_matrices) / total_weight
            
            return ensemble_matrix
            
        except Exception as e:
            self.logger.error(f"アンサンブル相関計算エラー: {e}")
            return results[0].correlation_matrix if results else pd.DataFrame()

    def _determine_correlation_regime(self, correlation_matrix: pd.DataFrame) -> CorrelationRegime:
        """市場レジーム判定"""
        try:
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            return self._classify_correlation_regime(avg_correlation)
        except:
            return CorrelationRegime.MODERATE_CORRELATION

    def _analyze_sectors(self, correlation_matrix: pd.DataFrame, sector_mapping: Optional[Dict[str, MarketSector]]) -> Dict[str, Any]:
        """セクター分析"""
        try:
            if not sector_mapping:
                return {'sector_mapping_not_provided': True}
            
            sector_analysis = {}
            
            # セクター別グループ化
            sectors = {}
            for asset, sector in sector_mapping.items():
                if asset in correlation_matrix.columns:
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(asset)
            
            # セクター内・セクター間相関
            for sector, assets in sectors.items():
                if len(assets) > 1:
                    sector_corr_matrix = correlation_matrix.loc[assets, assets]
                    sector_analysis[f"{sector.value}_intra_correlation"] = self._calculate_average_correlation(sector_corr_matrix)
            
            # セクター間相関
            sector_keys = list(sectors.keys())
            for i in range(len(sector_keys)):
                for j in range(i+1, len(sector_keys)):
                    sector1, sector2 = sector_keys[i], sector_keys[j]
                    assets1, assets2 = sectors[sector1], sectors[sector2]
                    
                    cross_correlations = []
                    for a1 in assets1:
                        for a2 in assets2:
                            if a1 in correlation_matrix.index and a2 in correlation_matrix.columns:
                                cross_correlations.append(correlation_matrix.loc[a1, a2])
                    
                    if cross_correlations:
                        sector_analysis[f"{sector1.value}_{sector2.value}_cross_correlation"] = np.mean(cross_correlations)
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"セクター分析エラー: {e}")
            return {'error': str(e)}

    def _analyze_risk_implications(self, correlation_matrix: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """リスク含意分析"""
        try:
            returns = price_data.pct_change().dropna()
            
            # ポートフォリオリスク計算（等重み）
            n_assets = len(correlation_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            
            # 分散共分散マトリックス
            cov_matrix = returns.cov()
            
            # ポートフォリオ分散
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年率化
            
            # 個別資産の平均ボラティリティ
            individual_volatilities = returns.std() * np.sqrt(252)
            avg_individual_volatility = individual_volatilities.mean()
            
            # 分散効果
            diversification_ratio = avg_individual_volatility / portfolio_volatility
            
            # 最大相関
            correlations = correlation_matrix.values
            max_correlation = np.max(correlations[np.triu_indices_from(correlations, k=1)])
            
            # 最小相関
            min_correlation = np.min(correlations[np.triu_indices_from(correlations, k=1)])
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'average_individual_volatility': avg_individual_volatility,
                'diversification_ratio': diversification_ratio,
                'max_correlation': max_correlation,
                'min_correlation': min_correlation,
                'correlation_range': max_correlation - min_correlation
            }
            
        except Exception as e:
            self.logger.error(f"リスク含意分析エラー: {e}")
            return {
                'portfolio_volatility': 0.2,
                'average_individual_volatility': 0.25,
                'diversification_ratio': 1.25,
                'max_correlation': 0.5,
                'min_correlation': -0.1,
                'correlation_range': 0.6
            }

    def _calculate_diversification_metrics(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """分散投資メトリクス計算"""
        try:
            n_assets = len(correlation_matrix.columns)
            
            # 平均相関
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            
            # 分散比率（Diversification Ratio）
            # 理想的な分散投資では相関が低いほど効果的
            diversification_effectiveness = 1 - abs(avg_correlation)
            
            # エフェクティブ資産数
            # 完全に独立なら n_assets、完全相関なら 1
            effective_assets = 1 + (n_assets - 1) * (1 - abs(avg_correlation))
            
            # 相関の分散（相関の均一性）
            correlations = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
            correlation_dispersion = np.std(correlations)
            
            # ハーフィンダール指数（相関の集中度）
            squared_correlations = correlations ** 2
            herfindahl_index = np.sum(squared_correlations) / len(squared_correlations)
            
            return {
                'average_correlation': avg_correlation,
                'diversification_effectiveness': diversification_effectiveness,
                'effective_asset_count': effective_assets,
                'correlation_dispersion': correlation_dispersion,
                'correlation_concentration': herfindahl_index
            }
            
        except Exception as e:
            self.logger.error(f"分散投資メトリクス計算エラー: {e}")
            return {
                'average_correlation': 0.3,
                'diversification_effectiveness': 0.7,
                'effective_asset_count': 3.0,
                'correlation_dispersion': 0.2,
                'correlation_concentration': 0.1
            }

    def _create_correlation_summary(self, 
                                  results: List[CorrelationResult], 
                                  ensemble_correlation: pd.DataFrame, 
                                  market_regime: CorrelationRegime) -> Dict[str, Any]:
        """相関分析サマリー作成"""
        try:
            return {
                'method_count': len(results),
                'regime_consensus': {
                    'primary_regime': market_regime.value,
                    'regime_agreement': sum(1 for r in results if r.correlation_regime == market_regime) / len(results)
                },
                'stability_metrics': {
                    'average_stability': np.mean([r.regime_stability for r in results]),
                    'confidence_range': [min(r.confidence for r in results), max(r.confidence for r in results)]
                },
                'correlation_statistics': {
                    'ensemble_average': self._calculate_average_correlation(ensemble_correlation),
                    'method_range': [
                        min(self._calculate_average_correlation(r.correlation_matrix) for r in results),
                        max(self._calculate_average_correlation(r.correlation_matrix) for r in results)
                    ]
                },
                'asset_count': len(ensemble_correlation.columns) if not ensemble_correlation.empty else 0
            }
            
        except Exception as e:
            self.logger.error(f"相関サマリー作成エラー: {e}")
            return {'error': 'summary_creation_failed'}

    def _validate_data_dict(self, data: Dict[str, pd.DataFrame]) -> bool:
        """データ辞書の検証"""
        if not data or len(data) < 2:
            return False
        
        for asset_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns or len(df) < 20:
                return False
        
        return True

    def _merge_params(self, custom_params: Optional[Dict]) -> Dict:
        """パラメータ統合"""
        default_params = {
            'analysis_window': self.default_window,
            'rolling_window': self.rolling_window,
            'ewma_lambda': 0.94
        }
        
        if custom_params:
            default_params.update(custom_params)
        
        return default_params

    def _generate_cache_key(self, data: Dict[str, pd.DataFrame], methods: Optional[List[CorrelationMethod]]) -> str:
        """キャッシュキー生成"""
        try:
            # データハッシュ（資産名と最終日付）
            data_signature = '_'.join([
                f"{name}:{len(df)}" for name, df in data.items()
            ])
            methods_str = '_'.join([m.value for m in methods]) if methods else 'all'
            return f"corr_{hash(data_signature)}_{methods_str}"
        except:
            return f"corr_{datetime.now().isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュ有効性チェック"""
        if cache_key not in self._correlation_cache:
            return False
        
        cache_time = self._correlation_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self._cache_timeout

    def _cache_result(self, cache_key: str, result: MarketCorrelationAnalysis):
        """結果をキャッシュ"""
        self._correlation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # キャッシュサイズ制限
        if len(self._correlation_cache) > 20:
            oldest_key = min(self._correlation_cache.keys(), 
                           key=lambda k: self._correlation_cache[k]['timestamp'])
            del self._correlation_cache[oldest_key]

    def _create_fallback_analysis(self) -> MarketCorrelationAnalysis:
        """フォールバック分析結果生成"""
        fallback_matrix = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], 
                                     index=['Asset1', 'Asset2'], 
                                     columns=['Asset1', 'Asset2'])
        
        fallback_result = CorrelationResult(
            method=CorrelationMethod.PEARSON,
            correlation_matrix=fallback_matrix,
            correlation_regime=CorrelationRegime.MODERATE_CORRELATION,
            dominant_correlations={'Asset1-Asset2': 0.3},
            cluster_analysis={},
            regime_stability=0.5,
            confidence=0.1,
            analysis_period=30,
            calculation_time=datetime.now()
        )
        
        return MarketCorrelationAnalysis(
            primary_method=CorrelationMethod.PEARSON,
            correlation_results=[fallback_result],
            ensemble_correlation=fallback_matrix,
            market_regime=CorrelationRegime.MODERATE_CORRELATION,
            sector_analysis={'fallback': True},
            risk_implications={'fallback': True},
            diversification_metrics={'fallback': True},
            analysis_summary={'is_fallback': True},
            analysis_time=datetime.now()
        )

    def clear_cache(self):
        """キャッシュクリア"""
        self._correlation_cache.clear()
        self.logger.info("市場相関分析キャッシュをクリアしました")

    def get_correlation_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """相関サマリー取得"""
        try:
            result = self.analyze_market_correlations(data)
            return {
                'market_regime': result.market_regime.value,
                'primary_method': result.primary_method.value,
                'method_count': len(result.correlation_results),
                'average_correlation': self._calculate_average_correlation(result.ensemble_correlation),
                'diversification_effectiveness': result.diversification_metrics.get('diversification_effectiveness', 0),
                'analysis_time': result.analysis_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"相関サマリー取得エラー: {e}")
            return {'error': str(e)}

# 利便性関数
def analyze_market_correlations_simple(data: Dict[str, pd.DataFrame], 
                                     methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    簡単な市場相関分析関数
    
    Args:
        data: 資産別市場データ辞書
        methods: 使用手法名のリスト
        
    Returns:
        Dict: 分析結果の辞書形式
    """
    # 手法名を列挙型に変換
    if methods:
        method_enums = []
        for method_name in methods:
            try:
                method_enums.append(CorrelationMethod(method_name))
            except ValueError:
                continue
    else:
        method_enums = None
    
    analyzer = MarketCorrelationAnalyzer()
    result = analyzer.analyze_market_correlations(data, method_enums)
    
    return {
        'market_regime': result.market_regime.value,
        'average_correlation': analyzer._calculate_average_correlation(result.ensemble_correlation),
        'diversification_effectiveness': result.diversification_metrics.get('diversification_effectiveness', 0),
        'method_count': len(result.correlation_results),
        'asset_count': len(result.ensemble_correlation.columns) if not result.ensemble_correlation.empty else 0,
        'analysis_time': result.analysis_time.isoformat()
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== 市場相関分析システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # 複数の相関パターンを持つデータ
    base_factor = np.random.normal(0, 0.02, 100)  # 共通ファクター
    
    # 資産1: 強い共通ファクター依存
    asset1_returns = base_factor * 0.8 + np.random.normal(0, 0.01, 100)
    asset1_prices = 100 * (1 + asset1_returns).cumprod()
    
    # 資産2: 中程度の共通ファクター依存
    asset2_returns = base_factor * 0.5 + np.random.normal(0, 0.015, 100)
    asset2_prices = 100 * (1 + asset2_returns).cumprod()
    
    # 資産3: 独立性が高い
    asset3_returns = base_factor * 0.2 + np.random.normal(0, 0.02, 100)
    asset3_prices = 100 * (1 + asset3_returns).cumprod()
    
    # データフレーム作成
    def create_ohlcv(prices):
        return pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(prices))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices)))),
            'Close': prices,
            'Volume': np.random.uniform(1000000, 5000000, len(prices))
        }, index=dates)
    
    test_data = {
        'Asset1': create_ohlcv(asset1_prices),
        'Asset2': create_ohlcv(asset2_prices),
        'Asset3': create_ohlcv(asset3_prices)
    }
    
    # セクターマッピング
    sector_mapping = {
        'Asset1': MarketSector.EQUITY,
        'Asset2': MarketSector.EQUITY,
        'Asset3': MarketSector.BOND
    }
    
    # 分析器テスト
    analyzer = MarketCorrelationAnalyzer()
    
    print("\n1. 全手法相関分析")
    result = analyzer.analyze_market_correlations(test_data, sector_mapping=sector_mapping)
    print(f"市場レジーム: {result.market_regime.value}")
    print(f"使用手法数: {len(result.correlation_results)}")
    print(f"平均相関: {analyzer._calculate_average_correlation(result.ensemble_correlation):.3f}")
    
    print("\n2. アンサンブル相関マトリックス")
    print(result.ensemble_correlation.round(3))
    
    print("\n3. 分散投資メトリクス")
    for metric, value in result.diversification_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n4. 簡単分析関数テスト")
    simple_result = analyze_market_correlations_simple(test_data, ['pearson', 'rolling'])
    print(f"簡単分析結果: {simple_result['market_regime']} (平均相関: {simple_result['average_correlation']:.3f})")
    
    print("\n=== テスト完了 ===")
