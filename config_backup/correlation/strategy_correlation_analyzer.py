"""
戦略相関分析システム - 戦略間の相関性と共分散を計算

このモジュールは、複数の戦略間の相関性と共分散を計算し、
ポートフォリオ最適化に必要な情報を提供する。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 既存システムとの統合
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.strategy_scoring_model import StrategyScore as ExistingStrategyScore, StrategyScoreManager as ExistingStrategyScoreManager
    from config.portfolio_weight_calculator import PortfolioWeightCalculator as ExistingPortfolioWeightCalculator
    from config.strategy_selector import StrategySelector as ExistingStrategySelector
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # フォールバック用のダミークラス
    ExistingStrategyScore = None
    ExistingStrategyScoreManager = None
    ExistingPortfolioWeightCalculator = None
    ExistingStrategySelector = None

logger = logging.getLogger(__name__)

@dataclass
class CorrelationConfig:
    """相関分析設定"""
    lookback_period: int = 252  # 1年間
    min_periods: int = 60  # 最小期間
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    confidence_level: float = 0.95
    rolling_window: int = 30  # ローリング相関用
    significance_threshold: float = 0.05  # 有意性検定の閾値
    
@dataclass
class CorrelationMatrix:
    """相関行列データクラス"""
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    p_values: pd.DataFrame
    confidence_intervals: Dict[str, pd.DataFrame]
    calculation_timestamp: datetime
    period_info: Dict[str, Union[str, int]]
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.correlation_matrix.empty:
            logger.warning("空の相関行列が作成されました")

@dataclass
class StrategyPerformanceData:
    """戦略パフォーマンスデータ"""
    strategy_name: str
    returns: pd.Series
    cumulative_returns: pd.Series
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
class StrategyCorrelationAnalyzer:
    """戦略相関分析メインクラス"""
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.strategy_data: Dict[str, StrategyPerformanceData] = {}
        self.correlation_history: List[CorrelationMatrix] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 既存システムとの統合
        try:
            if ExistingStrategyScoreManager:
                self.score_manager = ExistingStrategyScoreManager()
            if ExistingPortfolioWeightCalculator:
                self.portfolio_calculator = ExistingPortfolioWeightCalculator()
            if ExistingStrategySelector:
                self.strategy_selector = ExistingStrategySelector()
        except Exception as e:
            self.logger.warning(f"既存システムとの統合に問題があります: {e}")
    
    def add_strategy_data(self, strategy_name: str, price_data: pd.DataFrame, 
                         signals: pd.Series) -> None:  # type: ignore
        """戦略データを追加"""
        try:
            # パフォーマンス計算
            returns = self._calculate_strategy_returns(price_data, signals)
            cumulative_returns = (1 + returns).cumprod() - 1
            
            # 統計指標計算
            volatility = returns.std() * np.sqrt(252)  # type: ignore
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            win_rate = (returns > 0).sum() / len(returns)  # type: ignore
            
            # データ格納
            self.strategy_data[strategy_name] = StrategyPerformanceData(
                strategy_name=strategy_name,
                returns=returns,  # type: ignore
                cumulative_returns=cumulative_returns,  # type: ignore
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate
            )
            
            self.logger.info(f"戦略データ追加: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"戦略データ追加エラー ({strategy_name}): {e}")
            raise
    
    def calculate_correlation_matrix(self, strategies: Optional[List[str]] = None) -> CorrelationMatrix:
        """相関行列を計算"""
        try:
            # 対象戦略の決定
            if strategies is None:
                strategies = list(self.strategy_data.keys())
            
            if len(strategies) < 2:
                raise ValueError("相関計算には最低2つの戦略が必要です")
            
            # リターンデータの準備
            returns_df = self._prepare_returns_dataframe(strategies)
            
            # 相関計算
            correlation_matrix = returns_df.corr(method=self.config.correlation_method)  # type: ignore
            covariance_matrix = returns_df.cov() * 252  # 年率化
            
            # 統計的有意性の検定
            p_values = self._calculate_correlation_p_values(returns_df)
            
            # 信頼区間の計算
            confidence_intervals = self._calculate_confidence_intervals(
                returns_df, correlation_matrix
            )
            
            # 期間情報
            period_info = {  # type: ignore
                "start_date": str(returns_df.index.min()),  # type: ignore
                "end_date": str(returns_df.index.max()),  # type: ignore
                "total_periods": len(returns_df),
                "strategies_count": len(strategies)
            }
            
            correlation_data = CorrelationMatrix(
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                calculation_timestamp=datetime.now(),
                period_info=period_info  # type: ignore
            )
            
            # 履歴に追加
            self.correlation_history.append(correlation_data)
            
            self.logger.info(f"相関行列計算完了: {len(strategies)}戦略")
            return correlation_data
            
        except Exception as e:
            self.logger.error(f"相関行列計算エラー: {e}")
            raise
    
    def calculate_rolling_correlation(self, strategy1: str, strategy2: str, 
                                    window: Optional[int] = None) -> pd.Series:
        """ローリング相関を計算"""
        try:
            window = window or self.config.rolling_window
            
            if strategy1 not in self.strategy_data or strategy2 not in self.strategy_data:
                raise ValueError("指定された戦略が見つかりません")
            
            returns1 = self.strategy_data[strategy1].returns
            returns2 = self.strategy_data[strategy2].returns
            
            # 共通の期間で計算
            common_index = returns1.index.intersection(returns2.index)
            returns1_aligned = returns1.reindex(common_index)
            returns2_aligned = returns2.reindex(common_index)
            
            rolling_corr = returns1_aligned.rolling(window=window).corr(returns2_aligned)
            
            self.logger.info(f"ローリング相関計算完了: {strategy1} vs {strategy2}")
            return rolling_corr
            
        except Exception as e:
            self.logger.error(f"ローリング相関計算エラー: {e}")
            raise
    
    def get_correlation_summary(self, correlation_matrix: CorrelationMatrix) -> Dict:
        """相関分析サマリーを生成"""
        try:
            corr_mat = correlation_matrix.correlation_matrix
            
            # 上三角行列から相関係数を取得（対角線を除く）
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            correlations = corr_mat.where(mask).stack()
            
            # 統計サマリー
            summary = {
                "total_pairs": len(correlations),
                "mean_correlation": correlations.mean(),
                "median_correlation": correlations.median(),
                "std_correlation": correlations.std(),
                "min_correlation": correlations.min(),
                "max_correlation": correlations.max(),
                "high_correlation_pairs": len(correlations[correlations.abs() > 0.7]),
                "moderate_correlation_pairs": len(correlations[(correlations.abs() > 0.3) & (correlations.abs() <= 0.7)]),
                "low_correlation_pairs": len(correlations[correlations.abs() <= 0.3]),
                "negative_correlations": len(correlations[correlations < 0]),
                "positive_correlations": len(correlations[correlations > 0])
            }
            
            # 最も相関の高いペア
            max_corr_idx = correlations.abs().idxmax()
            summary["highest_correlation_pair"] = {
                "strategies": max_corr_idx,
                "correlation": correlations[max_corr_idx]
            }
            
            # 最も相関の低いペア
            min_corr_idx = correlations.abs().idxmin()
            summary["lowest_correlation_pair"] = {
                "strategies": min_corr_idx,
                "correlation": correlations[min_corr_idx]
            }
            
            self.logger.info("相関サマリー生成完了")
            return summary
            
        except Exception as e:
            self.logger.error(f"相関サマリー生成エラー: {e}")
            raise
    
    def detect_correlation_clusters(self, correlation_matrix: CorrelationMatrix, 
                                  threshold: float = 0.7) -> Dict[int, List[str]]:
        """相関クラスターを検出"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # 距離行列として使用するため、相関から距離に変換
            distance_matrix = 1 - correlation_matrix.correlation_matrix.abs()
            
            # 階層クラスタリング実行
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1-threshold,
                linkage='average',
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # クラスター結果を辞書形式で整理
            clusters = {}
            strategies = correlation_matrix.correlation_matrix.index.tolist()
            
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(strategies[i])
            
            self.logger.info(f"クラスター検出完了: {len(clusters)}個のクラスター")
            return clusters
            
        except ImportError:
            self.logger.warning("scikit-learnが利用できません。クラスタリングをスキップします")
            return {}
        except Exception as e:
            self.logger.error(f"クラスター検出エラー: {e}")
            raise
    
    def _calculate_strategy_returns(self, price_data: pd.DataFrame, 
                                  signals: pd.Series) -> pd.Series:  # type: ignore
        """戦略リターンを計算"""
        try:
            # 価格変化率を計算
            if 'close' in price_data.columns:
                price_returns = price_data['close'].pct_change()
            else:
                # 最初の列を価格として使用
                price_returns = price_data.iloc[:, 0].pct_change()
            
            # シグナルに基づく戦略リターン
            # シグナルを前日にシフト（当日の終値で実行されると仮定）
            strategy_returns = price_returns * signals.shift(1)
            
            return strategy_returns.dropna()
            
        except Exception as e:
            self.logger.error(f"戦略リターン計算エラー: {e}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:  # type: ignore
        """シャープレシオを計算"""
        try:
            annual_return = returns.mean() * 252  # type: ignore
            annual_vol = returns.std() * np.sqrt(252)  # type: ignore
            
            if annual_vol == 0:
                return 0.0
            
            return (annual_return - risk_free_rate) / annual_vol
            
        except Exception as e:
            self.logger.error(f"シャープレシオ計算エラー: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:  # type: ignore
        """最大ドローダウンを計算"""
        try:
            peak = cumulative_returns.expanding().max()  # type: ignore
            drawdown = (cumulative_returns - peak) / peak  # type: ignore
            return drawdown.min()  # type: ignore
            
        except Exception as e:
            self.logger.error(f"最大ドローダウン計算エラー: {e}")
            return 0.0
    
    def _prepare_returns_dataframe(self, strategies: List[str]) -> pd.DataFrame:
        """リターンデータフレームを準備"""
        try:
            returns_dict = {}
            
            for strategy in strategies:
                if strategy in self.strategy_data:
                    returns_dict[strategy] = self.strategy_data[strategy].returns
            
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < self.config.min_periods:
                raise ValueError(f"データが不足しています。最低{self.config.min_periods}期間必要です")
            
            # lookback_periodで制限
            if len(returns_df) > self.config.lookback_period:
                returns_df = returns_df.tail(self.config.lookback_period)
            
            return returns_df
            
        except Exception as e:
            self.logger.error(f"リターンデータフレーム準備エラー: {e}")
            raise
    
    def _calculate_correlation_p_values(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """相関の統計的有意性を検定"""
        try:
            try:
                from scipy.stats import pearsonr
                
                strategies = returns_df.columns
                n_strategies = len(strategies)
                p_values = np.ones((n_strategies, n_strategies))
                
                for i in range(n_strategies):
                    for j in range(i+1, n_strategies):
                        _, p_value = pearsonr(returns_df.iloc[:, i], returns_df.iloc[:, j])
                        p_values[i, j] = p_value
                        p_values[j, i] = p_value
                
                return pd.DataFrame(p_values, index=strategies, columns=strategies)
                
            except ImportError:
                self.logger.warning("scipy.statsが利用できません。p値計算をスキップします")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"p値計算エラー: {e}")
            return pd.DataFrame()
    
    def _calculate_confidence_intervals(self, returns_df: pd.DataFrame, 
                                      correlation_matrix: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """相関係数の信頼区間を計算"""
        try:
            n = len(returns_df)
            z_score = 1.96 if self.config.confidence_level == 0.95 else 2.576  # 95%または99%
            
            # Fisher変換を使用
            fisher_z = np.arctanh(correlation_matrix.values)
            se = 1 / np.sqrt(n - 3)
            
            # 信頼区間をFisher変換で計算後、逆変換
            lower_bound = np.tanh(fisher_z - z_score * se)
            upper_bound = np.tanh(fisher_z + z_score * se)
            
            strategies = correlation_matrix.index
            
            return {
                "lower_bound": pd.DataFrame(lower_bound, index=strategies, columns=strategies),
                "upper_bound": pd.DataFrame(upper_bound, index=strategies, columns=strategies)
            }
            
        except Exception as e:
            self.logger.error(f"信頼区間計算エラー: {e}")
            return {}
    
    def save_correlation_data(self, correlation_matrix: CorrelationMatrix, 
                            filepath: Union[str, Path]) -> None:
        """相関データを保存"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # データを辞書形式で準備
            save_data = {
                "correlation_matrix": correlation_matrix.correlation_matrix.to_dict(),
                "covariance_matrix": correlation_matrix.covariance_matrix.to_dict(),
                "p_values": correlation_matrix.p_values.to_dict() if not correlation_matrix.p_values.empty else {},
                "confidence_intervals": {
                    k: v.to_dict() for k, v in correlation_matrix.confidence_intervals.items()
                },
                "calculation_timestamp": correlation_matrix.calculation_timestamp.isoformat(),
                "period_info": correlation_matrix.period_info
            }
            
            # JSON形式で保存
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"相関データ保存完了: {filepath}")
            
        except Exception as e:
            self.logger.error(f"相関データ保存エラー: {e}")
            raise
    
    def load_correlation_data(self, filepath: Union[str, Path]) -> CorrelationMatrix:
        """相関データを読み込み"""
        try:
            import json
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # DataFrameに復元
            correlation_matrix = pd.DataFrame.from_dict(data["correlation_matrix"])
            covariance_matrix = pd.DataFrame.from_dict(data["covariance_matrix"])
            p_values = pd.DataFrame.from_dict(data["p_values"]) if data["p_values"] else pd.DataFrame()
            confidence_intervals = {
                k: pd.DataFrame.from_dict(v) for k, v in data["confidence_intervals"].items()
            }
            
            return CorrelationMatrix(
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                calculation_timestamp=datetime.fromisoformat(data["calculation_timestamp"]),
                period_info=data["period_info"]
            )
            
        except Exception as e:
            self.logger.error(f"相関データ読み込みエラー: {e}")
            raise

if __name__ == "__main__":
    # 基本的なテスト
    logging.basicConfig(level=logging.INFO)
    
    print("戦略相関分析システム - テスト実行")
    
    # テスト用のダミーデータ生成
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 価格データ（ランダムウォーク）
    price_data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))
    }, index=dates)
    
    # テスト戦略のシグナル
    signals1 = pd.Series(np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3]), index=dates)
    signals2 = pd.Series(np.random.choice([-1, 0, 1], len(dates), p=[0.2, 0.6, 0.2]), index=dates)
    
    # 分析器初期化
    analyzer = StrategyCorrelationAnalyzer()
    
    # 戦略データ追加
    analyzer.add_strategy_data("Strategy_A", price_data, signals1)
    analyzer.add_strategy_data("Strategy_B", price_data, signals2)
    
    # 相関分析実行
    correlation_result = analyzer.calculate_correlation_matrix()
    
    print("\n=== 相関行列 ===")
    print(correlation_result.correlation_matrix)
    
    print("\n=== 共分散行列 ===")
    print(correlation_result.covariance_matrix)
    
    # サマリー表示
    summary = analyzer.get_correlation_summary(correlation_result)
    print("\n=== 相関サマリー ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nテスト完了")
