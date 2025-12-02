"""
Module: Metric Importance Analyzer
File: metric_importance_analyzer.py
Description: 
  重要指標分析エンジン - 統計的手法による指標重要度分析
  相関分析と回帰分析のハイブリッド手法で戦略パフォーマンスに最も影響する指標を選定
  2-1-2「重要指標選定システム」の中核実装

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - pandas
  - numpy
  - scipy
  - sklearn
  - json
  - config.strategy_characteristics_manager
  - config.optimized_parameters
  - config.metric_selection_config
"""

import json
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import threading

# 統計・機械学習ライブラリ
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False
    warnings.warn("Advanced statistics libraries not available. Some features will be limited.")

# 内部モジュール
try:
    from .metric_selection_config import MetricSelectionConfig
    from .strategy_characteristics_manager import StrategyCharacteristicsManager
    from .optimized_parameters import OptimizedParameterManager
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.metric_selection_config import MetricSelectionConfig
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    from config.optimized_parameters import OptimizedParameterManager

# ロガーの設定
logger = logging.getLogger(__name__)

class MetricImportanceAnalyzer:
    """
    重要指標分析エンジン
    
    戦略パフォーマンスに対する各指標の重要度を統計的手法で分析し、
    最適な指標セットを選定するシステム
    """
    
    def __init__(self, config: Optional[MetricSelectionConfig] = None, base_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            config: 設定インスタンス
            base_dir: 基底ディレクトリ（デフォルト: logs/metric_importance）
        """
        # 設定の初期化
        self.config = config if config is not None else MetricSelectionConfig()
        
        # パス設定
        if base_dir is None:
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "logs" / "metric_importance"
        
        self.base_dir = Path(base_dir)
        self.analysis_dir = self.base_dir / "analysis"
        self.results_dir = self.base_dir / "results"
        
        # ディレクトリ作成
        for dir_path in [self.analysis_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 依存モジュール初期化
        self.characteristics_manager: Optional[StrategyCharacteristicsManager] = None
        self.parameter_manager: Optional[OptimizedParameterManager] = None
        self._initialize_managers()
        
        # 分析結果キャッシュ
        self.analysis_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        
        logger.info(f"MetricImportanceAnalyzer initialized: {self.base_dir}")
    
    def _initialize_managers(self):
        """依存モジュールの初期化"""
        try:
            self.characteristics_manager = StrategyCharacteristicsManager()
            self.parameter_manager = OptimizedParameterManager()
            logger.info("Dependency managers initialized")
        except Exception as e:
            logger.warning(f"Manager initialization warning: {e}")
    
    def collect_strategy_data(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        """
        戦略データの収集
        
        Args:
            strategies: 分析対象戦略リスト（Noneの場合は全戦略）
            
        Returns:
            pd.DataFrame: 分析用データフレーム
        """
        logger.info("Collecting strategy data for analysis")
        
        # 戦略リストの取得
        if strategies is None:
            strategies = self._get_available_strategies()
        
        if not strategies:
            logger.warning("No strategies found for analysis")
            return pd.DataFrame()
        
        # データ収集
        data_rows = []
        target_metrics = self.config.get_target_metrics()
        
        for strategy in strategies:
            try:
                # 戦略特性データの取得
                characteristics = self._get_strategy_characteristics(strategy)
                
                # 最適化パラメータの取得
                optimized_params = self._get_optimized_parameters(strategy)
                
                # トレンド別データの処理
                for trend_type in ["uptrend", "downtrend", "range-bound"]:
                    trend_data = characteristics.get("trend_adaptability", {}).get(trend_type, {})
                    
                    if trend_data and trend_data.get("sample_size", 0) > 0:
                        row = self._create_data_row(strategy, trend_type, trend_data, optimized_params, target_metrics)
                        if row:
                            data_rows.append(row)
                
            except Exception as e:
                logger.error(f"Error collecting data for {strategy}: {e}")
                continue
        
        if not data_rows:
            logger.warning("No valid data collected")
            return pd.DataFrame()
        
        # データフレーム作成
        df = pd.DataFrame(data_rows)
        
        # 追加指標の計算
        df = self._calculate_additional_metrics(df)
        
        logger.info(f"Data collection complete: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _get_available_strategies(self) -> List[str]:
        """利用可能な戦略リストの取得"""
        strategies = set()
        
        # 戦略特性マネージャーから取得
        if self.characteristics_manager:
            try:
                char_strategies = self.characteristics_manager.list_strategies()
                strategies.update(char_strategies)
            except Exception as e:
                logger.warning(f"Error getting strategies from characteristics manager: {e}")
        
        # 最適化パラメータから取得
        if self.parameter_manager:
            try:
                param_strategies = self._get_strategies_from_params()
                strategies.update(param_strategies)
            except Exception as e:
                logger.warning(f"Error getting strategies from parameters: {e}")
        
        return list(strategies)
    
    def _get_strategies_from_params(self) -> List[str]:
        """最適化パラメータから戦略リストを取得"""
        try:
            params_dir = Path(__file__).parent / "optimized_params"
            if not params_dir.exists():
                return []
            
            strategies = set()
            for file_path in params_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "strategy" in data:
                            strategies.add(data["strategy"])
                except Exception:
                    continue
            
            return list(strategies)
        except Exception as e:
            logger.error(f"Error getting strategies from params: {e}")
            return []
    
    def _get_strategy_characteristics(self, strategy: str) -> Dict[str, Any]:
        """戦略特性データの取得"""
        if self.characteristics_manager:
            try:
                return self.characteristics_manager.load_metadata(strategy) or {}
            except Exception as e:
                logger.warning(f"Error loading characteristics for {strategy}: {e}")
        return {}
    
    def _get_optimized_parameters(self, strategy: str) -> Dict[str, Any]:
        """最適化パラメータの取得"""
        if self.parameter_manager:
            try:
                return self.parameter_manager.load_approved_params(strategy) or {}
            except Exception as e:
                logger.warning(f"Error loading parameters for {strategy}: {e}")
        return {}
    
    def _create_data_row(self, 
                        strategy: str, 
                        trend_type: str, 
                        trend_data: Dict[str, Any],
                        optimized_params: Dict[str, Any],
                        target_metrics: List[str]) -> Optional[Dict[str, Any]]:
        """データ行の作成"""
        try:
            performance_metrics = trend_data.get("performance_metrics", {})
            
            row = {
                "strategy": strategy,
                "trend_type": trend_type,
                "sample_size": trend_data.get("sample_size", 0),
                "data_period": trend_data.get("data_period", "unknown"),
                "confidence_level": trend_data.get("confidence_level", "low"),
                "suitability_score": trend_data.get("suitability_score", 0.0)
            }
            
            # 基本指標の追加
            for metric in target_metrics:
                row[metric] = performance_metrics.get(metric, 0.0)
            
            # リスク特性の追加
            risk_chars = trend_data.get("risk_characteristics", {})
            row.update({
                "avg_holding_period": risk_chars.get("avg_holding_period", 5.0),
                "max_consecutive_losses": risk_chars.get("max_consecutive_losses", 0),
                "profit_factor": risk_chars.get("profit_factor", 1.0)
            })
            
            # 戦略固有情報
            row["strategy_type"] = self._categorize_strategy(strategy)
            
            return row
            
        except Exception as e:
            logger.error(f"Error creating data row: {e}")
            return None
    
    def _categorize_strategy(self, strategy: str) -> str:
        """戦略のカテゴリ分類"""
        strategy_lower = strategy.lower()
        if "vwap" in strategy_lower:
            return "VWAP_based"
        elif "momentum" in strategy_lower:
            return "momentum"
        elif "contrarian" in strategy_lower:
            return "contrarian"
        elif "gc" in strategy_lower or "golden" in strategy_lower:
            return "trend_following"
        elif "breakout" in strategy_lower:
            return "breakout"
        elif "gap" in strategy_lower:
            return "gap"
        else:
            return "other"
    
    def _calculate_additional_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """追加指標の計算"""
        if df.empty:
            return df
        
        try:
            # Recovery Factor = Total Return / |Max Drawdown|
            df["recovery_factor"] = np.where(
                df["max_drawdown"] != 0,
                df["total_return"] / np.abs(df["max_drawdown"]),
                0
            )
            
            # Risk Adjusted Return = Total Return / Volatility
            df["risk_adjusted_return"] = np.where(
                df["volatility"] != 0,
                df["total_return"] / df["volatility"],
                0
            )
            
            # Consistency Ratio (簡易版: 勝率の安定性指標)
            df["consistency_ratio"] = np.where(
                df["win_rate"] > 0.5,
                df["win_rate"] * (1 - df["volatility"]),
                df["win_rate"] * 0.5
            )
            
            # Downside Deviation (簡易版: ボラティリティ * 負リターン係数)
            df["downside_deviation"] = np.where(
                df["total_return"] < 0,
                df["volatility"] * 1.5,
                df["volatility"] * 0.5
            )
            
            # VaR 95% (簡易版: 正規分布仮定)
            df["var_95"] = df["total_return"] - 1.645 * df["volatility"]
            
            # Tail Ratio (簡易版: 極端リターンの比率)
            df["tail_ratio"] = np.where(
                df["max_drawdown"] != 0,
                df["total_return"] / np.abs(df["max_drawdown"]),
                1.0
            )
            
            logger.info("Additional metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
        
        return df
    
    def analyze_metric_importance(self, 
                                data: Optional[pd.DataFrame] = None,
                                target_metric: Optional[str] = None,
                                methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        指標重要度分析の実行
        
        Args:
            data: 分析用データ（Noneの場合は自動収集）
            target_metric: 目標指標（Noneの場合は設定から取得）
            methods: 分析手法リスト（Noneの場合は設定から取得）
            
        Returns:
            Dict: 分析結果
        """
        logger.info("Starting metric importance analysis")
        
        # データの準備
        if data is None:
            data = self.collect_strategy_data()
        
        if data.empty:
            logger.error("No data available for analysis")
            return {"error": "No data available"}
        
        # 目標指標の設定
        if target_metric is None:
            target_metric = self.config.get_target_variable()
        
        # 分析手法の設定
        if methods is None:
            methods = self.config.get_analysis_methods()
        
        # 分析結果の初期化
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "target_metric": target_metric,
            "data_summary": {
                "total_samples": len(data),
                "strategies_count": data["strategy"].nunique(),
                "trend_types_count": data["trend_type"].nunique(),
                "metrics_analyzed": len(self.config.get_target_metrics())
            },
            "analysis_methods": methods,
            "results": {}
        }
        
        # 分析対象指標の準備
        all_metrics = self.config.get_target_metrics()
        feature_columns = [col for col in all_metrics if col in data.columns and col != target_metric]
        
        if not feature_columns:
            logger.error("No valid feature columns found")
            return {"error": "No valid features"}
        
        # データの前処理
        analysis_data = self._preprocess_data(data, feature_columns + [target_metric])
        
        if analysis_data.empty:
            logger.error("Data preprocessing failed")
            return {"error": "Preprocessing failed"}
        
        # 各手法での分析実行
        if "correlation" in methods:
            try:
                results["results"]["correlation_analysis"] = self._perform_correlation_analysis(
                    analysis_data, feature_columns, target_metric
                )
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
                results["results"]["correlation_analysis"] = {"error": str(e)}
        
        if "regression" in methods and ADVANCED_STATS_AVAILABLE:
            try:
                results["results"]["regression_analysis"] = self._perform_regression_analysis(
                    analysis_data, feature_columns, target_metric
                )
            except Exception as e:
                logger.error(f"Regression analysis failed: {e}")
                results["results"]["regression_analysis"] = {"error": str(e)}
        
        if "feature_selection" in methods and ADVANCED_STATS_AVAILABLE:
            try:
                results["results"]["feature_selection"] = self._perform_feature_selection(
                    analysis_data, feature_columns, target_metric
                )
            except Exception as e:
                logger.error(f"Feature selection failed: {e}")
                results["results"]["feature_selection"] = {"error": str(e)}
        
        # 統合重要度スコアの計算
        try:
            results["integrated_importance"] = self._calculate_integrated_importance(results["results"])
        except Exception as e:
            logger.error(f"Integrated importance calculation failed: {e}")
            results["integrated_importance"] = {"error": str(e)}
        
        # 推奨指標の選定
        try:
            results["recommended_metrics"] = self._select_recommended_metrics(results.get("integrated_importance", {}))
        except Exception as e:
            logger.error(f"Metric recommendation failed: {e}")
            results["recommended_metrics"] = []
        
        # 結果の保存
        try:
            self._save_analysis_results(results)
        except Exception as e:
            logger.error(f"Results saving failed: {e}")
        
        logger.info("Metric importance analysis completed")
        return results
    
    def _preprocess_data(self, 
                        data: pd.DataFrame, 
                        columns: List[str]) -> pd.DataFrame:
        """
        データの前処理
        
        Args:
            data: 元データ
            columns: 処理対象列
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        try:
            # 指定列のみ抽出
            processed_data = data[columns].copy()
            
            # 欠損値の処理
            processed_data = processed_data.dropna()
            
            # 無限値の処理
            processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
            processed_data = processed_data.dropna()
            
            # 異常値の処理（簡易版）
            for column in processed_data.columns:
                if processed_data[column].dtype in ['int64', 'float64']:
                    Q1 = processed_data[column].quantile(0.01)
                    Q3 = processed_data[column].quantile(0.99)
                    processed_data[column] = processed_data[column].clip(Q1, Q3)
            
            logger.info(f"Data preprocessing completed. Shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            return pd.DataFrame()
    
    def _perform_correlation_analysis(self, 
                                    data: pd.DataFrame,
                                    feature_columns: List[str],
                                    target_metric: str) -> Dict[str, Any]:
        """相関分析による指標重要度分析（簡易版）"""
        try:
            correlations = {}
            target_data = data[target_metric]
            
            for feature in feature_columns:
                if feature in data.columns:
                    feature_data = data[feature]
                    
                    # ピアソン相関係数の計算
                    corr_coef = feature_data.corr(target_data)
                    
                    correlations[feature] = {
                        "correlation": float(corr_coef) if not np.isnan(corr_coef) else 0.0,
                        "abs_correlation": float(abs(corr_coef)) if not np.isnan(corr_coef) else 0.0,
                        "correlation_strength": self._classify_correlation_strength(abs(corr_coef))
                    }
            
            return {
                "method": "pearson",
                "importance_scores": correlations,
                "feature_count": len(correlations)
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {"error": str(e)}
    
    def _perform_regression_analysis(self, 
                                   data: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_metric: str) -> Dict[str, Any]:
        """回帰分析による指標重要度分析（簡易版）"""
        try:
            X = data[feature_columns].fillna(data[feature_columns].mean())
            y = data[target_metric].fillna(data[target_metric].mean())
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Ridge回帰
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_scaled, y)
            
            # 予測とスコア
            y_pred = model.predict(X_scaled)
            r2_score_value = r2_score(y, y_pred)
            
            # 特徴量重要度
            feature_importance = np.abs(model.coef_)
            total_importance = np.sum(feature_importance)
            
            importance_dict = {}
            for i, feature in enumerate(feature_columns):
                importance_dict[feature] = {
                    "importance": float(feature_importance[i]),
                    "normalized_importance": float(feature_importance[i] / total_importance) if total_importance > 0 else 0.0,
                    "coefficient": float(model.coef_[i])
                }
            
            return {
                "method": "ridge",
                "model_performance": {
                    "r2_score": float(r2_score_value)
                },
                "feature_importance": importance_dict
            }
            
        except Exception as e:
            logger.error(f"Regression analysis error: {e}")
            return {"error": str(e)}
    
    def _perform_feature_selection(self, 
                                 data: pd.DataFrame,
                                 feature_columns: List[str],
                                 target_metric: str) -> Dict[str, Any]:
        """特徴量選択による指標重要度分析（簡易版）"""
        try:
            X = data[feature_columns].fillna(data[feature_columns].mean())
            y = data[target_metric].fillna(data[target_metric].mean())
            
            # F統計量による特徴量選択
            k_best = min(8, len(feature_columns))
            selector = SelectKBest(score_func=f_regression, k=k_best)
            X_selected = selector.fit_transform(X, y)
            
            scores = selector.scores_
            selected_mask = selector.get_support()
            
            feature_scores = {}
            for i, feature in enumerate(feature_columns):
                feature_scores[feature] = {
                    "score": float(scores[i]) if not np.isnan(scores[i]) else 0.0,
                    "selected": bool(selected_mask[i]),
                    "rank": int(np.argsort(scores)[::-1].tolist().index(i) + 1)
                }
            
            selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selected_mask[i]]
            
            return {
                "method": "f_regression",
                "feature_scores": feature_scores,
                "selected_features": selected_features,
                "n_features_selected": len(selected_features)
            }
            
        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return {"error": str(e)}
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """相関の強さを分類"""
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _calculate_integrated_importance(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """統合重要度スコアの計算（簡易版）"""
        try:
            integrated_importance = {}
            
            # 各手法から特徴量を収集
            all_features = set()
            for method_results in analysis_results.values():
                if "feature_importance" in method_results:
                    all_features.update(method_results["feature_importance"].keys())
                elif "importance_scores" in method_results:
                    all_features.update(method_results["importance_scores"].keys())
                elif "feature_scores" in method_results:
                    all_features.update(method_results["feature_scores"].keys())
            
            # 各特徴量の統合スコア計算
            for feature in all_features:
                scores = []
                
                # 相関分析スコア
                if "correlation_analysis" in analysis_results:
                    corr_data = analysis_results["correlation_analysis"].get("importance_scores", {})
                    if feature in corr_data:
                        scores.append(corr_data[feature]["abs_correlation"])
                
                # 回帰分析スコア
                if "regression_analysis" in analysis_results:
                    reg_data = analysis_results["regression_analysis"].get("feature_importance", {})
                    if feature in reg_data:
                        scores.append(reg_data[feature]["normalized_importance"])
                
                # 特徴量選択スコア（正規化）
                if "feature_selection" in analysis_results:
                    fs_data = analysis_results["feature_selection"].get("feature_scores", {})
                    if feature in fs_data:
                        score = fs_data[feature]["score"]
                        normalized_score = min(score / 100.0, 1.0) if score > 0 else 0.0
                        scores.append(normalized_score)
                
                # 統合スコア
                final_score = np.mean(scores) if scores else 0.0
                
                integrated_importance[feature] = {
                    "final_importance_score": float(final_score),
                    "method_count": len(scores),
                    "confidence_level": "high" if len(scores) >= 2 and final_score >= 0.5 else "medium" if final_score >= 0.3 else "low"
                }
            
            # ランキング
            sorted_features = sorted(
                integrated_importance.items(),
                key=lambda x: x[1]["final_importance_score"],
                reverse=True
            )
            
            for rank, (feature, data) in enumerate(sorted_features, 1):
                data["rank"] = rank
            
            return integrated_importance
            
        except Exception as e:
            logger.error(f"Integrated importance calculation error: {e}")
            return {}
    
    def _select_recommended_metrics(self, integrated_importance: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """推奨指標の選定"""
        try:
            k_best = self.config.get("output_settings.top_k_metrics", 8)
            min_confidence = self.config.get("output_settings.min_confidence_level", "medium")
            
            # 信頼度フィルタリング
            confidence_order = {"high": 3, "medium": 2, "low": 1}
            min_confidence_score = confidence_order.get(min_confidence, 2)
            
            recommended = []
            for feature, data in integrated_importance.items():
                feature_confidence = confidence_order.get(data.get("confidence_level", "low"), 1)
                
                if feature_confidence >= min_confidence_score:
                    recommended.append({
                        "feature": feature,
                        "importance_score": data["final_importance_score"],
                        "confidence": data["confidence_level"],
                        "rank": data["rank"],
                        "method_count": data["method_count"]
                    })
            
            # スコア順でソートして上位k個を返す
            recommended.sort(key=lambda x: x["importance_score"], reverse=True)
            return recommended[:k_best]
            
        except Exception as e:
            logger.error(f"Metric recommendation error: {e}")
            return []
    
    def _save_analysis_results(self, results: Dict[str, Any]) -> str:
        """分析結果を保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_metric = results.get("target_metric", "unknown")
            filename = f"metric_importance_analysis_{target_metric}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # 結果を保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Results saving error: {e}")
            return ""

# 使用例とテスト
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== 重要指標分析エンジンテスト ===")
    
    try:
        # 分析エンジンの初期化
        analyzer = MetricImportanceAnalyzer()
        print("✓ 分析エンジン初期化完了")
        
        # データ収集テスト
        data = analyzer.collect_strategy_data()
        if not data.empty:
            print(f"✓ データ収集完了: {len(data)}行")
            
            # 分析実行
            results = analyzer.analyze_metric_importance(data)
            
            if "error" not in results:
                print("✓ 分析実行完了")
                
                # 推奨指標の表示
                recommended = results.get("recommended_metrics", [])
                print(f"\n推奨指標 (上位{len(recommended)}指標):")
                for i, metric in enumerate(recommended, 1):
                    print(f"  {i}. {metric['feature']} (スコア: {metric['importance_score']:.3f})")
            else:
                print(f"✗ 分析エラー: {results['error']}")
        else:
            print("✗ データが見つかりません")
            
    except Exception as e:
        print(f"✗ テストエラー: {e}")
    
    print("\nテスト完了")
