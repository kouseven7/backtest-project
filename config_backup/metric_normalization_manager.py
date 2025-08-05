"""
Module: Metric Normalization Manager
File: metric_normalization_manager.py
Description: 
  指標正規化システムの統合管理クラス
  スコアリングシステムとの連携、指標選定システムとの統合、データ永続化を管理
  2-1-3「指標の正規化手法の設計」の統合管理コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - json
  - pandas
  - pathlib
  - config.metric_normalization_config
  - config.metric_normalization_engine
  - config.strategy_scoring_model
  - config.metric_selection_manager
"""

import json
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict

# 内部モジュール
try:
    from .metric_normalization_config import MetricNormalizationConfig
    from .metric_normalization_engine import MetricNormalizationEngine, NormalizationResult
    from .strategy_scoring_model import StrategyScoreManager, StrategyScore
    from .metric_selection_manager import MetricSelectionManager, MetricSelectionSummary
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent))
    from metric_normalization_config import MetricNormalizationConfig
    from metric_normalization_engine import MetricNormalizationEngine, NormalizationResult
    from strategy_scoring_model import StrategyScoreManager, StrategyScore
    from metric_selection_manager import MetricSelectionManager, MetricSelectionSummary

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class NormalizationSession:
    """正規化セッションの記録"""
    session_id: str
    timestamp: str
    strategy_name: Optional[str]
    metrics_processed: List[str]
    results: Dict[str, Dict[str, Any]]
    integration_mode: str
    success_rate: float
    total_processing_time: float
    notes: str = ""

@dataclass
class NormalizationSummary:
    """正規化処理の要約"""
    session_info: NormalizationSession
    scoring_integration: Optional[Dict[str, Any]]
    selection_integration: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    data_quality_report: Dict[str, Any]
    recommendations: List[str]
    success: bool
    error_messages: List[str]

class MetricNormalizationManager:
    """
    指標正規化システムの統合管理クラス
    
    正規化処理、スコアリングシステム連携、指標選定システム統合、
    データ永続化を一括で管理するメインクラス
    """
    
    def __init__(self, 
                 config: Optional[MetricNormalizationConfig] = None,
                 base_dir: Optional[str] = None,
                 integration_mode: str = "metric_selection"):
        """
        初期化
        
        Args:
            config: 設定インスタンス
            base_dir: 基底ディレクトリ
            integration_mode: 統合モード ("scoring", "metric_selection", "standalone")
        """
        self.config = config if config is not None else MetricNormalizationConfig()
        self.integration_mode = integration_mode
        
        # パス設定
        if base_dir is None:
            project_root = Path(__file__).parent.parent
            self.base_dir = project_root / "logs" / "metric_normalization"
        else:
            self.base_dir = Path(base_dir)
        
        # ディレクトリ構造の作成
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports"
        self.sessions_dir = self.base_dir / "sessions"
        
        for dir_path in [self.data_dir, self.reports_dir, self.sessions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントの初期化
        self.engine = MetricNormalizationEngine(self.config)
        self.scoring_manager: Optional[StrategyScoreManager] = None
        self.selection_manager: Optional[MetricSelectionManager] = None
        
        # 統合システムの初期化
        self._initialize_integration_systems()
        
        # セッション管理
        self.sessions_history: List[NormalizationSession] = []
        self.current_session: Optional[NormalizationSession] = None
        
        logger.info(f"MetricNormalizationManager initialized: mode={integration_mode}, dir={self.base_dir}")
    
    def _initialize_integration_systems(self):
        """統合システムの初期化"""
        try:
            if self.integration_mode in ["scoring", "metric_selection"]:
                # スコアリングシステムの初期化
                try:
                    self.scoring_manager = StrategyScoreManager()
                    logger.info("✓ Scoring system integration enabled")
                except Exception as e:
                    logger.warning(f"Scoring system unavailable: {e}")
            
            if self.integration_mode == "metric_selection":
                # 指標選定システムの初期化
                try:
                    self.selection_manager = MetricSelectionManager()
                    logger.info("✓ Metric selection system integration enabled")
                except Exception as e:
                    logger.warning(f"Metric selection system unavailable: {e}")
                    
        except Exception as e:
            logger.error(f"Integration systems initialization failed: {e}")
    
    def normalize_strategy_metrics(self, 
                                 strategy_name: str,
                                 metrics_data: Dict[str, Union[pd.Series, np.ndarray, List]],
                                 save_session: bool = True,
                                 apply_to_scoring: bool = False) -> NormalizationSummary:
        """
        戦略指標の正規化処理
        
        Args:
            strategy_name: 戦略名
            metrics_data: 指標データの辞書
            save_session: セッション保存フラグ
            apply_to_scoring: スコアリングシステムへの適用フラグ
            
        Returns:
            NormalizationSummary: 正規化処理の要約
        """
        logger.info(f"Starting normalization for strategy: {strategy_name}")
        start_time = datetime.now()
        
        session_id = f"{strategy_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        error_messages = []
        
        try:
            # セッション開始
            self.current_session = self._create_session(session_id, strategy_name, list(metrics_data.keys()))
            
            # 正規化処理の実行
            normalization_results = self.engine.batch_normalize(metrics_data, strategy_name)
            
            # 結果の処理
            processed_results = {}
            success_count = 0
            
            for metric_name, result in normalization_results.items():
                processed_results[metric_name] = result.get_summary()
                if result.success:
                    success_count += 1
                else:
                    error_messages.append(f"{metric_name}: {result.error_message}")
            
            success_rate = success_count / len(metrics_data) if metrics_data else 0.0
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # セッション更新
            self.current_session.results = processed_results
            self.current_session.success_rate = success_rate
            self.current_session.total_processing_time = processing_time
            
            # スコアリングシステム統合
            scoring_integration = None
            if apply_to_scoring and self.scoring_manager:
                scoring_integration = self._integrate_with_scoring(
                    strategy_name, normalization_results
                )
            
            # 指標選定システム統合
            selection_integration = None
            if self.selection_manager:
                selection_integration = self._integrate_with_selection(
                    strategy_name, normalization_results
                )
            
            # データ品質レポートの生成
            quality_report = self._generate_data_quality_report(normalization_results)
            
            # パフォーマンス指標の計算
            performance_metrics = self._calculate_performance_metrics(normalization_results)
            
            # 推奨事項の生成
            recommendations = self._generate_recommendations(normalization_results, quality_report)
            
            # セッション保存
            if save_session:
                self._save_session(self.current_session)
            
            # 要約作成
            summary = NormalizationSummary(
                session_info=self.current_session,
                scoring_integration=scoring_integration,
                selection_integration=selection_integration,
                performance_metrics=performance_metrics,
                data_quality_report=quality_report,
                recommendations=recommendations,
                success=success_rate > 0.5,
                error_messages=error_messages
            )
            
            logger.info(f"✓ Normalization completed: {strategy_name}, success_rate: {success_rate:.3f}")
            return summary
            
        except Exception as e:
            logger.error(f"Normalization failed for {strategy_name}: {e}")
            error_messages.append(f"Critical error: {str(e)}")
            
            # エラー時の要約
            return NormalizationSummary(
                session_info=self.current_session or self._create_session(session_id, strategy_name, []),
                scoring_integration=None,
                selection_integration=None,
                performance_metrics={},
                data_quality_report={},
                recommendations=[],
                success=False,
                error_messages=error_messages
            )
    
    def batch_normalize_strategies(self, 
                                 strategies_data: Dict[str, Dict[str, Union[pd.Series, np.ndarray, List]]],
                                 save_sessions: bool = True) -> Dict[str, NormalizationSummary]:
        """
        複数戦略の一括正規化処理
        
        Args:
            strategies_data: {戦略名: {指標名: データ}} の辞書
            save_sessions: セッション保存フラグ
            
        Returns:
            Dict[str, NormalizationSummary]: 戦略別正規化要約
        """
        logger.info(f"Starting batch normalization for {len(strategies_data)} strategies")
        
        summaries = {}
        for strategy_name, metrics_data in strategies_data.items():
            try:
                summary = self.normalize_strategy_metrics(
                    strategy_name=strategy_name,
                    metrics_data=metrics_data,
                    save_session=save_sessions,
                    apply_to_scoring=False  # 一括処理時は個別適用を避ける
                )
                summaries[strategy_name] = summary
                
                if summary.success:
                    logger.debug(f"✓ {strategy_name}: success")
                else:
                    logger.warning(f"✗ {strategy_name}: failed")
                    
            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
                summaries[strategy_name] = NormalizationSummary(
                    session_info=self._create_session(f"{strategy_name}_error", strategy_name, []),
                    scoring_integration=None,
                    selection_integration=None,
                    performance_metrics={},
                    data_quality_report={},
                    recommendations=[],
                    success=False,
                    error_messages=[str(e)]
                )
        
        success_count = sum(1 for s in summaries.values() if s.success)
        logger.info(f"Batch normalization completed: {success_count}/{len(strategies_data)} successful")
        
        return summaries
    
    def normalize_for_metric_selection(self, 
                                     metrics_data: Dict[str, Union[pd.Series, np.ndarray, List]],
                                     target_metric: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        指標選定システム用の正規化処理
        
        Args:
            metrics_data: 指標データの辞書
            target_metric: 目標指標
            
        Returns:
            Dict[str, np.ndarray]: 正規化済み指標データ
        """
        logger.info("Normalizing metrics for metric selection system")
        
        try:
            # 正規化処理
            normalization_results = self.engine.batch_normalize(metrics_data)
            
            # 成功した結果のみを抽出
            normalized_data = {}
            for metric_name, result in normalization_results.items():
                if result.success:
                    normalized_data[metric_name] = result.normalized_data
                else:
                    logger.warning(f"Normalization failed for {metric_name}, using original data")
                    # 元データをNumPy配列として提供
                    if isinstance(metrics_data[metric_name], (list, pd.Series)):
                        normalized_data[metric_name] = np.array(metrics_data[metric_name])
                    else:
                        normalized_data[metric_name] = metrics_data[metric_name]
            
            logger.info(f"✓ Normalized {len(normalized_data)} metrics for selection system")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Normalization for metric selection failed: {e}")
            # エラー時は元データをそのまま返す
            fallback_data = {}
            for metric_name, data in metrics_data.items():
                if isinstance(data, (list, pd.Series)):
                    fallback_data[metric_name] = np.array(data)
                else:
                    fallback_data[metric_name] = data
            return fallback_data
    
    def _integrate_with_scoring(self, 
                              strategy_name: str, 
                              normalization_results: Dict[str, NormalizationResult]) -> Dict[str, Any]:
        """スコアリングシステムとの統合"""
        if not self.scoring_manager:
            return {"error": "Scoring system not available"}
        
        try:
            # 正規化済みデータの準備
            normalized_metrics = {}
            for metric_name, result in normalization_results.items():
                if result.success:
                    # スコアリングシステム用にメトリック値（平均）を計算
                    if hasattr(result.normalized_data, '__iter__'):
                        normalized_metrics[metric_name] = float(np.mean(result.normalized_data))
                    else:
                        normalized_metrics[metric_name] = float(result.normalized_data)
            
            # スコアリングシステムでの更新（仮想的な実装）
            integration_info = {
                "updated_metrics": list(normalized_metrics.keys()),
                "strategy_name": strategy_name,
                "integration_timestamp": datetime.now().isoformat(),
                "normalization_applied": True,
                "normalized_values": normalized_metrics
            }
            
            logger.info(f"✓ Scoring integration completed for {strategy_name}")
            return integration_info
            
        except Exception as e:
            logger.error(f"Scoring integration failed: {e}")
            return {"error": str(e)}
    
    def _integrate_with_selection(self, 
                                strategy_name: str, 
                                normalization_results: Dict[str, NormalizationResult]) -> Dict[str, Any]:
        """指標選定システムとの統合"""
        if not self.selection_manager:
            return {"error": "Selection system not available"}
        
        try:
            # 正規化済みデータの統計情報を抽出
            selection_data = {}
            for metric_name, result in normalization_results.items():
                if result.success:
                    selection_data[metric_name] = {
                        "normalized_values": result.normalized_data.tolist() if hasattr(result.normalized_data, 'tolist') else [result.normalized_data],
                        "confidence_score": result.confidence_score,
                        "method_used": result.method_used,
                        "statistics": result.statistics
                    }
            
            integration_info = {
                "processed_metrics": list(selection_data.keys()),
                "strategy_name": strategy_name,
                "integration_timestamp": datetime.now().isoformat(),
                "selection_ready": True,
                "data_summary": selection_data
            }
            
            logger.info(f"✓ Selection integration completed for {strategy_name}")
            return integration_info
            
        except Exception as e:
            logger.error(f"Selection integration failed: {e}")
            return {"error": str(e)}
    
    def _generate_data_quality_report(self, 
                                    normalization_results: Dict[str, NormalizationResult]) -> Dict[str, Any]:
        """データ品質レポートの生成"""
        try:
            report = {
                "total_metrics": len(normalization_results),
                "successful_normalizations": sum(1 for r in normalization_results.values() if r.success),
                "failed_normalizations": sum(1 for r in normalization_results.values() if not r.success),
                "average_confidence": np.mean([r.confidence_score for r in normalization_results.values() if r.success]),
                "outliers_detected": sum(r.outliers_detected for r in normalization_results.values()),
                "missing_values_handled": sum(r.missing_values_handled for r in normalization_results.values()),
                "method_distribution": {},
                "quality_issues": []
            }
            
            # 手法分布の計算
            methods = [r.method_used for r in normalization_results.values() if r.success]
            for method in set(methods):
                report["method_distribution"][method] = methods.count(method)
            
            # 品質問題の検出
            for metric_name, result in normalization_results.items():
                if not result.success:
                    report["quality_issues"].append(f"{metric_name}: {result.error_message}")
                elif result.confidence_score < 0.5:
                    report["quality_issues"].append(f"{metric_name}: Low confidence score ({result.confidence_score:.3f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_metrics(self, 
                                     normalization_results: Dict[str, NormalizationResult]) -> Dict[str, float]:
        """パフォーマンス指標の計算"""
        try:
            successful_results = [r for r in normalization_results.values() if r.success]
            
            if not successful_results:
                return {"error": "No successful normalizations"}
            
            metrics = {
                "success_rate": len(successful_results) / len(normalization_results),
                "average_confidence": np.mean([r.confidence_score for r in successful_results]),
                "min_confidence": np.min([r.confidence_score for r in successful_results]),
                "max_confidence": np.max([r.confidence_score for r in successful_results]),
                "total_outliers": sum(r.outliers_detected for r in successful_results),
                "total_missing_handled": sum(r.missing_values_handled for r in successful_results)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, 
                                normalization_results: Dict[str, NormalizationResult],
                                quality_report: Dict[str, Any]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        try:
            # 成功率に基づく推奨
            success_rate = quality_report.get("successful_normalizations", 0) / quality_report.get("total_metrics", 1)
            if success_rate < 0.7:
                recommendations.append("Success rate is low. Consider reviewing data quality and normalization parameters.")
            
            # 信頼度に基づく推奨
            avg_confidence = quality_report.get("average_confidence", 0)
            if avg_confidence < 0.7:
                recommendations.append("Low average confidence. Consider using robust normalization methods.")
            
            # 外れ値に基づく推奨
            total_outliers = quality_report.get("outliers_detected", 0)
            if total_outliers > len(normalization_results) * 5:  # 指標あたり5個以上の外れ値
                recommendations.append("High number of outliers detected. Consider robust normalization or outlier preprocessing.")
            
            # 手法分布に基づく推奨
            method_dist = quality_report.get("method_distribution", {})
            if len(method_dist) == 1 and "min_max" in method_dist:
                recommendations.append("Consider using diverse normalization methods for different metric types.")
            
            # 品質問題に基づく推奨
            quality_issues = quality_report.get("quality_issues", [])
            if len(quality_issues) > 0:
                recommendations.append("Address quality issues by reviewing failed normalizations and low confidence scores.")
            
            if not recommendations:
                recommendations.append("Normalization quality is good. Current configuration is working well.")
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error.")
        
        return recommendations
    
    def _create_session(self, session_id: str, strategy_name: Optional[str], metrics: List[str]) -> NormalizationSession:
        """セッションの作成"""
        return NormalizationSession(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            metrics_processed=metrics,
            results={},
            integration_mode=self.integration_mode,
            success_rate=0.0,
            total_processing_time=0.0
        )
    
    def _save_session(self, session: NormalizationSession) -> bool:
        """セッションの保存"""
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            
            # セッションデータをJSONシリアライズ可能な形式に変換
            session_dict = asdict(session)
            session_dict = self._convert_to_json_serializable(session_dict)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)
            
            # 履歴に追加
            self.sessions_history.append(session)
            
            logger.debug(f"Session saved: {session_file}")
            return True
            
        except Exception as e:
            logger.error(f"Session save failed: {e}")
            return False
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """オブジェクトをJSONシリアライズ可能な形式に変換"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # numpy数値型
            return obj.item() if hasattr(obj, 'item') else float(obj)
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            try:
                return str(obj)
            except Exception:
                return None
    
    def load_session(self, session_id: str) -> Optional[NormalizationSession]:
        """セッションの読み込み"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                logger.warning(f"Session file not found: {session_id}")
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = NormalizationSession(**session_data)
            logger.info(f"Session loaded: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Session load failed: {e}")
            return None
    
    def get_normalization_history(self, 
                                strategy_name: Optional[str] = None,
                                days_back: int = 30) -> List[NormalizationSession]:
        """正規化履歴の取得"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # セッションファイルから履歴を読み込み
            history = []
            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    session = NormalizationSession(**session_data)
                    session_time = datetime.fromisoformat(session.timestamp)
                    
                    if session_time >= cutoff_date:
                        if strategy_name is None or session.strategy_name == strategy_name:
                            history.append(session)
                            
                except Exception as e:
                    logger.warning(f"Failed to load session file {session_file}: {e}")
            
            # 時間順にソート
            history.sort(key=lambda x: x.timestamp, reverse=True)
            
            logger.info(f"Retrieved {len(history)} sessions from history")
            return history
            
        except Exception as e:
            logger.error(f"History retrieval failed: {e}")
            return []
    
    def generate_comprehensive_report(self, 
                                    summaries: Dict[str, NormalizationSummary],
                                    save_report: bool = True) -> Dict[str, Any]:
        """包括的レポートの生成"""
        try:
            report = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "strategies_analyzed": len(summaries),
                    "integration_mode": self.integration_mode
                },
                "overall_performance": {
                    "total_strategies": len(summaries),
                    "successful_strategies": sum(1 for s in summaries.values() if s.success),
                    "average_success_rate": np.mean([s.session_info.success_rate for s in summaries.values()]),
                    "total_metrics_processed": sum(len(s.session_info.metrics_processed) for s in summaries.values())
                },
                "quality_analysis": {
                    "average_confidence": np.mean([
                        s.performance_metrics.get("average_confidence", 0) 
                        for s in summaries.values() if s.success
                    ]),
                    "total_outliers": sum([
                        s.data_quality_report.get("outliers_detected", 0)
                        for s in summaries.values() if s.success
                    ]),
                    "common_issues": self._analyze_common_issues(summaries)
                },
                "integration_summary": self._analyze_integration_results(summaries),
                "recommendations": self._generate_comprehensive_recommendations(summaries),
                "strategy_details": {
                    name: {
                        "success": summary.success,
                        "metrics_count": len(summary.session_info.metrics_processed),
                        "success_rate": summary.session_info.success_rate,
                        "processing_time": summary.session_info.total_processing_time
                    }
                    for name, summary in summaries.items()
                }
            }
            
            if save_report:
                report_file = self.reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Comprehensive report saved: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_common_issues(self, summaries: Dict[str, NormalizationSummary]) -> List[str]:
        """共通問題の分析"""
        issue_counts = {}
        for summary in summaries.values():
            for issue in summary.data_quality_report.get("quality_issues", []):
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # 頻度順にソート
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{issue} (occurs {count} times)" for issue, count in common_issues[:5]]
    
    def _analyze_integration_results(self, summaries: Dict[str, NormalizationSummary]) -> Dict[str, Any]:
        """統合結果の分析"""
        scoring_successes = sum(1 for s in summaries.values() if s.scoring_integration and "error" not in s.scoring_integration)
        selection_successes = sum(1 for s in summaries.values() if s.selection_integration and "error" not in s.selection_integration)
        
        return {
            "scoring_integration": {
                "successful": scoring_successes,
                "total": len(summaries),
                "success_rate": scoring_successes / len(summaries) if summaries else 0
            },
            "selection_integration": {
                "successful": selection_successes,
                "total": len(summaries),
                "success_rate": selection_successes / len(summaries) if summaries else 0
            }
        }
    
    def _generate_comprehensive_recommendations(self, summaries: Dict[str, NormalizationSummary]) -> List[str]:
        """包括的推奨事項の生成"""
        recommendations = []
        
        success_rate = sum(1 for s in summaries.values() if s.success) / len(summaries) if summaries else 0
        
        if success_rate < 0.8:
            recommendations.append("Overall success rate is below 80%. Review normalization configurations.")
        
        if self.integration_mode == "metric_selection" and not self.selection_manager:
            recommendations.append("Metric selection integration is enabled but system is unavailable.")
        
        # 戦略固有の問題パターンを検出
        failed_strategies = [name for name, summary in summaries.items() if not summary.success]
        if len(failed_strategies) > len(summaries) * 0.3:
            recommendations.append("Multiple strategies failing. Consider global configuration review.")
        
        return recommendations

# 使用例とテスト用の関数
def create_sample_manager() -> MetricNormalizationManager:
    """サンプルマネージャーの作成"""
    config = MetricNormalizationConfig()
    manager = MetricNormalizationManager(config, integration_mode="metric_selection")
    return manager

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータの作成
    np.random.seed(42)
    sample_strategies = {
        "trend_following": {
            "sharpe_ratio": np.random.normal(1.0, 0.5, 50),
            "profit_factor": np.random.exponential(1.5, 50),
            "win_rate": np.random.beta(2, 2, 50)
        },
        "mean_reversion": {
            "sharpe_ratio": np.random.normal(0.8, 0.3, 50),
            "profit_factor": np.random.exponential(1.2, 50),
            "win_rate": np.random.beta(1.5, 2, 50)
        }
    }
    
    # マネージャーのテスト
    manager = create_sample_manager()
    summaries = manager.batch_normalize_strategies(sample_strategies)
    
    print("Normalization Summaries:")
    for strategy, summary in summaries.items():
        print(f"{strategy}: success={summary.success}, metrics={len(summary.session_info.metrics_processed)}")
    
    # 包括的レポートの生成
    comprehensive_report = manager.generate_comprehensive_report(summaries, save_report=False)
    print("Comprehensive Report Generated:", comprehensive_report.get("report_info", {}))
