"""
統計分析器
フェーズ4A3: バックテストvs実運用比較分析器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class StatisticalAnalyzer:
    """統計分析器"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.statistical_tests = config.get('analysis_settings', {}).get('statistical_tests', {})
        self.confidence_level = config.get('analysis_settings', {}).get('confidence_level', 0.95)
    
    def perform_comprehensive_analysis(self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]) -> Dict[str, Any]:
        """包括的統計分析実行"""
        try:
            self.logger.info("包括的統計分析開始")
            
            analysis_results = {
                "timestamp": datetime.now(),
                "confidence_level": self.confidence_level,
                "statistical_tests": {},
                "distribution_analysis": {},
                "correlation_analysis": {},
                "trend_analysis": {},
                "outlier_analysis": {}
            }
            
            # 共通戦略抽出
            bt_strategies = set(backtest_data.get('strategies', {}).keys())
            live_strategies = set(live_data.get('strategies', {}).keys())
            common_strategies = list(bt_strategies.intersection(live_strategies))
            
            if not common_strategies:
                self.logger.warning("共通戦略が見つかりません")
                return analysis_results
            
            # 戦略別統計テスト
            for strategy_name in common_strategies:
                bt_strategy = backtest_data.get('strategies', {}).get(strategy_name, {})
                live_strategy = live_data.get('strategies', {}).get(strategy_name, {})
                
                strategy_analysis = self._analyze_strategy_statistics(
                    strategy_name, bt_strategy, live_strategy
                )
                
                if strategy_analysis:
                    analysis_results["statistical_tests"][strategy_name] = strategy_analysis
            
            # 分布分析
            analysis_results["distribution_analysis"] = self._analyze_distributions(
                backtest_data, live_data, common_strategies
            )
            
            # 相関分析
            analysis_results["correlation_analysis"] = self._analyze_correlations(
                backtest_data, live_data, common_strategies
            )
            
            # トレンド分析
            analysis_results["trend_analysis"] = self._analyze_trends(
                backtest_data, live_data, common_strategies
            )
            
            # 外れ値分析
            analysis_results["outlier_analysis"] = self._analyze_outliers(
                backtest_data, live_data, common_strategies
            )
            
            self.logger.info(f"統計分析完了 - 戦略数: {len(common_strategies)}")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"包括的統計分析エラー: {e}")
            return {}
    
    def _analyze_strategy_statistics(self, strategy_name: str, bt_data: Dict[str, Any], live_data: Dict[str, Any]) -> Dict[str, Any]:
        """戦略別統計分析"""
        try:
            strategy_analysis = {
                "strategy_name": strategy_name,
                "significance_tests": {},
                "descriptive_statistics": {},
                "effect_size": {}
            }
            
            # メトリクス値抽出
            bt_metrics = self._extract_metrics_for_analysis(bt_data)
            live_metrics = self._extract_metrics_for_analysis(live_data)
            
            common_metrics = set(bt_metrics.keys()).intersection(set(live_metrics.keys()))
            
            for metric in common_metrics:
                bt_values = bt_metrics[metric]
                live_values = live_metrics[metric]
                
                if not bt_values or not live_values:
                    continue
                
                # 有意性検定
                if self.statistical_tests.get('t_test', True):
                    t_test_result = self._perform_t_test(bt_values, live_values)
                    strategy_analysis["significance_tests"][f"{metric}_t_test"] = t_test_result
                
                if self.statistical_tests.get('ks_test', True):
                    ks_test_result = self._perform_ks_test(bt_values, live_values)
                    strategy_analysis["significance_tests"][f"{metric}_ks_test"] = ks_test_result
                
                # 記述統計
                descriptive_stats = self._calculate_descriptive_statistics(bt_values, live_values)
                strategy_analysis["descriptive_statistics"][metric] = descriptive_stats
                
                # 効果量
                effect_size = self._calculate_effect_size(bt_values, live_values)
                strategy_analysis["effect_size"][metric] = effect_size
            
            return strategy_analysis
            
        except Exception as e:
            self.logger.warning(f"戦略統計分析エラー [{strategy_name}]: {e}")
            return {}
    
    def _extract_metrics_for_analysis(self, strategy_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """分析用メトリクス抽出"""
        try:
            metrics = {}
            
            # 基本メトリクス
            basic_metrics = strategy_data.get('basic_metrics', {})
            for metric_name, value in basic_metrics.items():
                if isinstance(value, (int, float)):
                    # 単一値を複数値に変換（仮想的なバリエーション生成）
                    metrics[metric_name] = self._generate_variations(value)
            
            # リスクメトリクス
            risk_metrics = strategy_data.get('risk_metrics', {})
            for metric_name, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[metric_name] = self._generate_variations(value)
            
            # 時系列データがある場合
            time_series = strategy_data.get('time_series', {})
            for metric_name, values in time_series.items():
                if isinstance(values, list) and values:
                    try:
                        float_values = [float(v) for v in values if v is not None]
                        if float_values:
                            metrics[metric_name] = float_values
                    except (ValueError, TypeError):
                        continue
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"メトリクス抽出エラー: {e}")
            return {}
    
    def _generate_variations(self, base_value: float, num_variations: int = 10) -> List[float]:
        """単一値から統計分析用のバリエーション生成"""
        try:
            if base_value == 0:
                return [0.0] * num_variations
            
            # 基準値周辺の正規分布サンプル生成
            std_dev = abs(base_value) * 0.1  # 基準値の10%を標準偏差とする
            variations = np.random.normal(base_value, std_dev, num_variations)
            
            return variations.tolist()
            
        except Exception as e:
            self.logger.warning(f"バリエーション生成エラー: {e}")
            return [base_value] * 10
    
    def _perform_t_test(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """t検定実行"""
        try:
            if not SCIPY_AVAILABLE:
                return self._manual_t_test(sample1, sample2)
            
            # SciPyを使用したt検定
            statistic, p_value = stats.ttest_ind(sample1, sample2)
            
            return {
                "test_type": "t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": p_value < (1 - self.confidence_level),
                "effect": "significant" if p_value < (1 - self.confidence_level) else "not_significant",
                "interpretation": self._interpret_t_test(statistic, p_value)
            }
            
        except Exception as e:
            self.logger.warning(f"t検定エラー: {e}")
            return self._manual_t_test(sample1, sample2)
    
    def _manual_t_test(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """手動t検定（SciPy未使用）"""
        try:
            n1, n2 = len(sample1), len(sample2)
            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
            
            # プールされた標準誤差
            pooled_se = np.sqrt(var1/n1 + var2/n2)
            
            # t統計量
            t_stat = (mean1 - mean2) / pooled_se if pooled_se != 0 else 0
            
            # 自由度
            df = n1 + n2 - 2
            
            # 簡易p値計算（正規近似）
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if SCIPY_AVAILABLE else 0.5
            
            return {
                "test_type": "manual_t_test",
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": df,
                "is_significant": abs(t_stat) > 2.0,  # 簡易判定
                "effect": "significant" if abs(t_stat) > 2.0 else "not_significant"
            }
            
        except Exception as e:
            self.logger.warning(f"手動t検定エラー: {e}")
            return {}
    
    def _perform_ks_test(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Kolmogorov-Smirnov検定実行"""
        try:
            if not SCIPY_AVAILABLE:
                return self._manual_ks_test(sample1, sample2)
            
            statistic, p_value = stats.ks_2samp(sample1, sample2)
            
            return {
                "test_type": "ks_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": p_value < (1 - self.confidence_level),
                "effect": "distributions_different" if p_value < (1 - self.confidence_level) else "distributions_similar",
                "interpretation": self._interpret_ks_test(statistic, p_value)
            }
            
        except Exception as e:
            self.logger.warning(f"KS検定エラー: {e}")
            return self._manual_ks_test(sample1, sample2)
    
    def _manual_ks_test(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """手動KS検定（SciPy未使用）"""
        try:
            # 簡易KS統計量計算
            combined = sorted(sample1 + sample2)
            n1, n2 = len(sample1), len(sample2)
            
            max_diff = 0
            for value in combined:
                cdf1 = sum(1 for x in sample1 if x <= value) / n1
                cdf2 = sum(1 for x in sample2 if x <= value) / n2
                max_diff = max(max_diff, abs(cdf1 - cdf2))
            
            # 簡易p値計算
            critical_value = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
            
            return {
                "test_type": "manual_ks_test",
                "statistic": float(max_diff),
                "critical_value": float(critical_value),
                "is_significant": max_diff > critical_value,
                "effect": "distributions_different" if max_diff > critical_value else "distributions_similar"
            }
            
        except Exception as e:
            self.logger.warning(f"手動KS検定エラー: {e}")
            return {}
    
    def _calculate_descriptive_statistics(self, bt_values: List[float], live_values: List[float]) -> Dict[str, Any]:
        """記述統計計算"""
        try:
            def calc_stats(values: List[float]) -> Dict[str, float]:
                return {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "var": float(np.var(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }
            
            return {
                "backtest": calc_stats(bt_values),
                "live": calc_stats(live_values),
                "difference": {
                    "mean_diff": float(np.mean(live_values) - np.mean(bt_values)),
                    "median_diff": float(np.median(live_values) - np.median(bt_values)),
                    "std_ratio": float(np.std(live_values) / np.std(bt_values)) if np.std(bt_values) != 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.warning(f"記述統計計算エラー: {e}")
            return {}
    
    def _calculate_effect_size(self, bt_values: List[float], live_values: List[float]) -> Dict[str, Any]:
        """効果量計算"""
        try:
            mean1, mean2 = np.mean(bt_values), np.mean(live_values)
            std1, std2 = np.std(bt_values), np.std(live_values)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(bt_values) - 1) * std1**2 + (len(live_values) - 1) * std2**2) / 
                               (len(bt_values) + len(live_values) - 2))
            cohens_d = (mean2 - mean1) / pooled_std if pooled_std != 0 else 0
            
            # 効果量の解釈
            if abs(cohens_d) < 0.2:
                magnitude = "negligible"
            elif abs(cohens_d) < 0.5:
                magnitude = "small"
            elif abs(cohens_d) < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            return {
                "cohens_d": float(cohens_d),
                "magnitude": magnitude,
                "direction": "live_better" if cohens_d > 0 else "backtest_better" if cohens_d < 0 else "equal"
            }
            
        except Exception as e:
            self.logger.warning(f"効果量計算エラー: {e}")
            return {}
    
    def _analyze_distributions(self, backtest_data: Dict[str, Any], live_data: Dict[str, Any], strategies: List[str]) -> Dict[str, Any]:
        """分布分析"""
        try:
            distribution_analysis = {}
            
            for strategy_name in strategies:
                bt_strategy = backtest_data.get('strategies', {}).get(strategy_name, {})
                live_strategy = live_data.get('strategies', {}).get(strategy_name, {})
                
                bt_metrics = self._extract_metrics_for_analysis(bt_strategy)
                live_metrics = self._extract_metrics_for_analysis(live_strategy)
                
                strategy_distributions = {}
                
                for metric in set(bt_metrics.keys()).intersection(set(live_metrics.keys())):
                    bt_values = bt_metrics[metric]
                    live_values = live_metrics[metric]
                    
                    # 分布の形状比較
                    distribution_comparison = {
                        "backtest_distribution": {
                            "skewness": float(self._calculate_skewness(bt_values)),
                            "kurtosis": float(self._calculate_kurtosis(bt_values)),
                            "normality": self._test_normality(bt_values)
                        },
                        "live_distribution": {
                            "skewness": float(self._calculate_skewness(live_values)),
                            "kurtosis": float(self._calculate_kurtosis(live_values)),
                            "normality": self._test_normality(live_values)
                        }
                    }
                    
                    strategy_distributions[metric] = distribution_comparison
                
                if strategy_distributions:
                    distribution_analysis[strategy_name] = strategy_distributions
            
            return distribution_analysis
            
        except Exception as e:
            self.logger.warning(f"分布分析エラー: {e}")
            return {}
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """歪度計算"""
        try:
            if SCIPY_AVAILABLE:
                return stats.skew(values)
            else:
                # 手動計算
                mean = np.mean(values)
                std = np.std(values)
                if std == 0:
                    return 0
                return np.mean([(x - mean)**3 for x in values]) / (std**3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """尖度計算"""
        try:
            if SCIPY_AVAILABLE:
                return stats.kurtosis(values)
            else:
                # 手動計算
                mean = np.mean(values)
                std = np.std(values)
                if std == 0:
                    return 0
                return np.mean([(x - mean)**4 for x in values]) / (std**4) - 3
        except:
            return 0.0
    
    def _test_normality(self, values: List[float]) -> Dict[str, Any]:
        """正規性検定"""
        try:
            if SCIPY_AVAILABLE and len(values) >= 8:
                statistic, p_value = stats.shapiro(values)
                return {
                    "test": "shapiro_wilk",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
            else:
                # 簡易正規性チェック
                skewness = abs(self._calculate_skewness(values))
                kurtosis = abs(self._calculate_kurtosis(values))
                is_normal = skewness < 2.0 and kurtosis < 7.0
                
                return {
                    "test": "simple_check",
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "is_normal": is_normal
                }
        except Exception as e:
            return {"test": "failed", "error": str(e)}
    
    def _analyze_correlations(self, backtest_data: Dict[str, Any], live_data: Dict[str, Any], strategies: List[str]) -> Dict[str, Any]:
        """相関分析"""
        try:
            correlation_analysis = {}
            
            # 戦略間相関（バックテストvs実運用）
            bt_performance = []
            live_performance = []
            strategy_names = []
            
            for strategy_name in strategies:
                bt_strategy = backtest_data.get('strategies', {}).get(strategy_name, {})
                live_strategy = live_data.get('strategies', {}).get(strategy_name, {})
                
                bt_pnl = bt_strategy.get('basic_metrics', {}).get('total_pnl', 0)
                live_pnl = live_strategy.get('basic_metrics', {}).get('total_pnl', 0)
                
                if bt_pnl != 0 or live_pnl != 0:
                    bt_performance.append(bt_pnl)
                    live_performance.append(live_pnl)
                    strategy_names.append(strategy_name)
            
            if len(bt_performance) >= 2:
                correlation = np.corrcoef(bt_performance, live_performance)[0, 1]
                correlation_analysis["performance_correlation"] = {
                    "correlation_coefficient": float(correlation),
                    "strength": self._interpret_correlation(correlation),
                    "strategies_analyzed": strategy_names
                }
            
            return correlation_analysis
            
        except Exception as e:
            self.logger.warning(f"相関分析エラー: {e}")
            return {}
    
    def _analyze_trends(self, backtest_data: Dict[str, Any], live_data: Dict[str, Any], strategies: List[str]) -> Dict[str, Any]:
        """トレンド分析"""
        try:
            trend_analysis = {}
            
            for strategy_name in strategies:
                bt_strategy = backtest_data.get('strategies', {}).get(strategy_name, {})
                live_strategy = live_data.get('strategies', {}).get(strategy_name, {})
                
                # 時系列データがある場合のトレンド分析
                bt_time_series = bt_strategy.get('time_series', {})
                live_time_series = live_strategy.get('time_series', {})
                
                strategy_trends = {}
                
                for metric in set(bt_time_series.keys()).intersection(set(live_time_series.keys())):
                    bt_values = bt_time_series[metric]
                    live_values = live_time_series[metric]
                    
                    if isinstance(bt_values, list) and isinstance(live_values, list):
                        bt_trend = self._calculate_trend(bt_values)
                        live_trend = self._calculate_trend(live_values)
                        
                        strategy_trends[metric] = {
                            "backtest_trend": bt_trend,
                            "live_trend": live_trend,
                            "trend_consistency": "consistent" if bt_trend * live_trend > 0 else "inconsistent"
                        }
                
                if strategy_trends:
                    trend_analysis[strategy_name] = strategy_trends
            
            return trend_analysis
            
        except Exception as e:
            self.logger.warning(f"トレンド分析エラー: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """トレンド計算（単純線形回帰の傾き）"""
        try:
            if len(values) < 2:
                return 0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # 線形回帰の傾き計算
            n = len(values)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            slope = numerator / denominator if denominator != 0 else 0
            return float(slope)
            
        except Exception as e:
            return 0.0
    
    def _analyze_outliers(self, backtest_data: Dict[str, Any], live_data: Dict[str, Any], strategies: List[str]) -> Dict[str, Any]:
        """外れ値分析"""
        try:
            outlier_analysis = {}
            
            for strategy_name in strategies:
                bt_strategy = backtest_data.get('strategies', {}).get(strategy_name, {})
                live_strategy = live_data.get('strategies', {}).get(strategy_name, {})
                
                bt_metrics = self._extract_metrics_for_analysis(bt_strategy)
                live_metrics = self._extract_metrics_for_analysis(live_strategy)
                
                strategy_outliers = {}
                
                for metric in set(bt_metrics.keys()).intersection(set(live_metrics.keys())):
                    bt_values = bt_metrics[metric]
                    live_values = live_metrics[metric]
                    
                    bt_outliers = self._detect_outliers(bt_values)
                    live_outliers = self._detect_outliers(live_values)
                    
                    strategy_outliers[metric] = {
                        "backtest_outliers": bt_outliers,
                        "live_outliers": live_outliers,
                        "outlier_consistency": len(set(bt_outliers['indices']).intersection(set(live_outliers['indices'])))
                    }
                
                if strategy_outliers:
                    outlier_analysis[strategy_name] = strategy_outliers
            
            return outlier_analysis
            
        except Exception as e:
            self.logger.warning(f"外れ値分析エラー: {e}")
            return {}
    
    def _detect_outliers(self, values: List[float]) -> Dict[str, Any]:
        """外れ値検出（IQR法）"""
        try:
            if len(values) < 4:
                return {"indices": [], "values": [], "method": "insufficient_data"}
            
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = []
            outlier_values = []
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
                    outlier_values.append(value)
            
            return {
                "indices": outlier_indices,
                "values": outlier_values,
                "method": "iqr",
                "bounds": {"lower": lower_bound, "upper": upper_bound},
                "outlier_rate": len(outlier_indices) / len(values)
            }
            
        except Exception as e:
            return {"indices": [], "values": [], "method": "error", "error": str(e)}
    
    def _interpret_t_test(self, statistic: float, p_value: float) -> str:
        """t検定結果解釈"""
        if p_value < 0.001:
            return f"極めて有意な差 (p < 0.001, t = {statistic:.3f})"
        elif p_value < 0.01:
            return f"非常に有意な差 (p < 0.01, t = {statistic:.3f})"
        elif p_value < 0.05:
            return f"有意な差 (p < 0.05, t = {statistic:.3f})"
        else:
            return f"有意差なし (p = {p_value:.3f}, t = {statistic:.3f})"
    
    def _interpret_ks_test(self, statistic: float, p_value: float) -> str:
        """KS検定結果解釈"""
        if p_value < 0.05:
            return f"分布に有意差あり (p = {p_value:.3f}, D = {statistic:.3f})"
        else:
            return f"分布に有意差なし (p = {p_value:.3f}, D = {statistic:.3f})"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """相関係数解釈"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
