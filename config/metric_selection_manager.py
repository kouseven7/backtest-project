"""
Module: Metric Selection Manager
File: metric_selection_manager.py
Description: 
  重要指標選定システム統合管理クラス
  分析・最適化・レポート生成を一括管理し、エラー耐性とユーザビリティを提供
  2-1-2「重要指標選定システム」の統合コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - json
  - pandas
  - config.metric_importance_analyzer
  - config.metric_weight_optimizer
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
from dataclasses import dataclass

# 内部モジュール
try:
    from .metric_importance_analyzer import MetricImportanceAnalyzer
    from .metric_weight_optimizer import MetricWeightOptimizer, WeightOptimizationResult
    from .metric_selection_config import MetricSelectionConfig
    from .strategy_scoring_model import StrategyScoreManager
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.metric_importance_analyzer import MetricImportanceAnalyzer
    from config.metric_weight_optimizer import MetricWeightOptimizer, WeightOptimizationResult
    from config.metric_selection_config import MetricSelectionConfig
    from config.strategy_scoring_model import StrategyScoreManager

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class MetricSelectionSummary:
    """重要指標選定の要約結果"""
    recommended_metrics: List[Dict[str, Any]]
    weight_optimization_result: Optional[WeightOptimizationResult]
    analysis_summary: Dict[str, Any]
    performance_impact: Dict[str, float]
    confidence_level: str
    timestamp: str
    success: bool
    error_messages: List[str]

class MetricSelectionManager:
    """
    重要指標選定システム統合管理クラス
    
    重要指標の分析、重みの最適化、結果の統合を一括で管理し、
    戦略スコアリングシステムとの連携を提供するメインクラス
    """
    
    def __init__(self, 
                 config: Optional[MetricSelectionConfig] = None,
                 base_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            config: 設定インスタンス
            base_dir: 基底ディレクトリ
        """
        self.config = config if config is not None else MetricSelectionConfig()
        
        # パス設定
        if base_dir is None:
            project_root = Path(__file__).parent.parent
            self.base_dir = project_root / "logs" / "metric_selection_system"
        else:
            self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.summaries_dir = self.base_dir / "summaries"
        
        # ディレクトリ作成
        for dir_path in [self.reports_dir, self.summaries_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントの初期化
        self.analyzer = MetricImportanceAnalyzer(self.config)
        self.optimizer = MetricWeightOptimizer(self.config)
        self.scoring_manager: Optional[StrategyScoreManager] = None
        
        # 実行履歴
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"MetricSelectionManager initialized: {self.base_dir}")
    
    def run_complete_analysis(self, 
                            strategies: Optional[List[str]] = None,
                            target_metric: Optional[str] = None,
                            optimization_method: str = "balanced_approach",
                            apply_weights: bool = False) -> MetricSelectionSummary:
        """
        完全な重要指標選定分析の実行
        
        Args:
            strategies: 分析対象戦略
            target_metric: 目標指標
            optimization_method: 重み最適化手法
            apply_weights: 最適化された重みを適用するか
            
        Returns:
            MetricSelectionSummary: 分析結果の要約
        """
        logger.info("Starting complete metric selection analysis")
        
        error_messages = []
        analysis_results = None
        weight_optimization_result = None
        
        try:
            # 1. 重要指標分析の実行
            logger.info("Step 1: Running metric importance analysis")
            analysis_results = self.analyzer.analyze_metric_importance(
                target_metric=target_metric
            )
            
            if "error" in analysis_results:
                error_messages.append(f"Analysis failed: {analysis_results['error']}")
                logger.error(f"Metric analysis failed: {analysis_results['error']}")
            else:
                logger.info("✓ Metric importance analysis completed")
            
            # 2. 重み最適化の実行
            if "error" not in analysis_results:
                logger.info("Step 2: Running weight optimization")
                weight_optimization_result = self.optimizer.optimize_weights(
                    importance_results=analysis_results,
                    optimization_method=optimization_method
                )
                
                if weight_optimization_result.success:
                    logger.info(f"✓ Weight optimization completed. Improvement: {weight_optimization_result.improvement_score:.3f}")
                    
                    # 3. 重みの適用（オプション）
                    if apply_weights:
                        logger.info("Step 3: Applying optimized weights")
                        if self.optimizer.apply_optimized_weights(weight_optimization_result):
                            logger.info("✓ Optimized weights applied")
                        else:
                            error_messages.append("Failed to apply optimized weights")
                else:
                    error_messages.append(f"Weight optimization failed: {weight_optimization_result.error_message}")
                    logger.error(f"Weight optimization failed: {weight_optimization_result.error_message}")
            
            # 4. 結果の統合と要約作成
            summary = self._create_analysis_summary(
                analysis_results, weight_optimization_result, error_messages
            )
            
            # 5. レポート生成
            report_path = self._generate_comprehensive_report(summary)
            logger.info(f"Comprehensive report generated: {report_path}")
            
            # 6. 実行履歴の更新
            self._update_execution_history(summary)
            
            logger.info("Complete metric selection analysis finished")
            return summary
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            error_messages.append(f"System error: {str(e)}")
            
            # エラー時の要約作成
            return MetricSelectionSummary(
                recommended_metrics=[],
                weight_optimization_result=None,
                analysis_summary={"error": str(e)},
                performance_impact={},
                confidence_level="low",
                timestamp=datetime.now().isoformat(),
                success=False,
                error_messages=error_messages
            )
    
    def _create_analysis_summary(self, 
                               analysis_results: Optional[Dict[str, Any]],
                               weight_optimization_result: Optional[WeightOptimizationResult],
                               error_messages: List[str]) -> MetricSelectionSummary:
        """分析結果の要約作成"""
        try:
            # 推奨指標の取得
            recommended_metrics = []
            if analysis_results and "recommended_metrics" in analysis_results:
                recommended_metrics = analysis_results["recommended_metrics"]
            
            # 分析要約の作成
            analysis_summary = {}
            if analysis_results:
                analysis_summary = {
                    "total_metrics_analyzed": len(self.config.get_target_metrics()),
                    "data_samples": analysis_results.get("data_summary", {}).get("total_samples", 0),
                    "strategies_analyzed": analysis_results.get("data_summary", {}).get("strategies_count", 0),
                    "analysis_methods": analysis_results.get("analysis_methods", []),
                    "recommended_count": len(recommended_metrics)
                }
            
            # パフォーマンス影響の評価
            performance_impact = self._evaluate_performance_impact(
                analysis_results, weight_optimization_result
            )
            
            # 信頼度レベルの決定
            confidence_level = self._determine_confidence_level(
                analysis_results, weight_optimization_result, error_messages
            )
            
            # 成功判定
            success = (
                len(error_messages) == 0 and
                analysis_results is not None and
                "error" not in analysis_results and
                len(recommended_metrics) > 0
            )
            
            return MetricSelectionSummary(
                recommended_metrics=recommended_metrics,
                weight_optimization_result=weight_optimization_result,
                analysis_summary=analysis_summary,
                performance_impact=performance_impact,
                confidence_level=confidence_level,
                timestamp=datetime.now().isoformat(),
                success=success,
                error_messages=error_messages
            )
            
        except Exception as e:
            logger.error(f"Summary creation error: {e}")
            return MetricSelectionSummary(
                recommended_metrics=[],
                weight_optimization_result=None,
                analysis_summary={"error": str(e)},
                performance_impact={},
                confidence_level="low",
                timestamp=datetime.now().isoformat(),
                success=False,
                error_messages=error_messages + [str(e)]
            )
    
    def _evaluate_performance_impact(self, 
                                   analysis_results: Optional[Dict[str, Any]],
                                   weight_optimization_result: Optional[WeightOptimizationResult]) -> Dict[str, float]:
        """パフォーマンス影響の評価"""
        try:
            impact = {}
            
            # 重み最適化による改善
            if weight_optimization_result and weight_optimization_result.success:
                impact["weight_improvement_score"] = weight_optimization_result.improvement_score
                impact["avg_weight_change"] = weight_optimization_result.validation_metrics.get("avg_weight_change", 0.0)
                impact["max_weight_change"] = weight_optimization_result.validation_metrics.get("max_weight_change", 0.0)
            
            # 分析の信頼性指標
            if analysis_results and "integrated_importance" in analysis_results:
                integrated_importance = analysis_results["integrated_importance"]
                
                # 高信頼度指標の比率
                high_confidence_count = sum(
                    1 for data in integrated_importance.values()
                    if data.get("confidence_level") == "high"
                )
                total_metrics = len(integrated_importance)
                impact["high_confidence_ratio"] = high_confidence_count / total_metrics if total_metrics > 0 else 0.0
                
                # 平均重要度スコア
                avg_importance = np.mean([
                    data.get("final_importance_score", 0.0)
                    for data in integrated_importance.values()
                ])
                impact["avg_importance_score"] = float(avg_importance)
            
            # データ品質指標
            if analysis_results and "data_summary" in analysis_results:
                data_summary = analysis_results["data_summary"]
                impact["data_completeness"] = min(1.0, data_summary.get("total_samples", 0) / 50.0)  # 50サンプルを目標
                impact["strategy_diversity"] = min(1.0, data_summary.get("strategies_count", 0) / 5.0)  # 5戦略を目標
            
            return impact
            
        except Exception as e:
            logger.error(f"Performance impact evaluation error: {e}")
            return {}
    
    def _determine_confidence_level(self, 
                                  analysis_results: Optional[Dict[str, Any]],
                                  weight_optimization_result: Optional[WeightOptimizationResult],
                                  error_messages: List[str]) -> str:
        """信頼度レベルの決定"""
        try:
            if error_messages:
                return "low"
            
            confidence_factors = []
            
            # データ品質
            if analysis_results and "data_summary" in analysis_results:
                data_summary = analysis_results["data_summary"]
                sample_count = data_summary.get("total_samples", 0)
                strategy_count = data_summary.get("strategies_count", 0)
                
                if sample_count >= 50 and strategy_count >= 5:
                    confidence_factors.append("high")
                elif sample_count >= 20 and strategy_count >= 3:
                    confidence_factors.append("medium")
                else:
                    confidence_factors.append("low")
            
            # 重み最適化の成功
            if weight_optimization_result and weight_optimization_result.success:
                improvement = weight_optimization_result.improvement_score
                if improvement > 0.1:
                    confidence_factors.append("high")
                elif improvement > 0.05:
                    confidence_factors.append("medium")
                else:
                    confidence_factors.append("low")
            
            # 推奨指標の数
            if analysis_results and "recommended_metrics" in analysis_results:
                recommended_count = len(analysis_results["recommended_metrics"])
                if recommended_count >= 5:
                    confidence_factors.append("high")
                elif recommended_count >= 3:
                    confidence_factors.append("medium")
                else:
                    confidence_factors.append("low")
            
            # 総合判定
            if not confidence_factors:
                return "low"
            
            high_count = confidence_factors.count("high")
            medium_count = confidence_factors.count("medium")
            low_count = confidence_factors.count("low")
            
            if high_count >= 2:
                return "high"
            elif high_count + medium_count >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "low"
    
    def _generate_comprehensive_report(self, summary: MetricSelectionSummary) -> str:
        """包括的レポートの生成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"metric_selection_report_{timestamp}.md"
            report_path = self.reports_dir / report_filename
            
            # レポート内容の生成
            report_content = self._create_report_content(summary)
            
            # ファイルに保存
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # JSON形式でも保存
            json_filename = f"metric_selection_report_{timestamp}.json"
            json_path = self.reports_dir / json_filename
            
            json_data = {
                "summary": {
                    "recommended_metrics": summary.recommended_metrics,
                    "analysis_summary": summary.analysis_summary,
                    "performance_impact": summary.performance_impact,
                    "confidence_level": summary.confidence_level,
                    "timestamp": summary.timestamp,
                    "success": summary.success,
                    "error_messages": summary.error_messages
                },
                "weight_optimization": summary.weight_optimization_result.__dict__ if summary.weight_optimization_result else None
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return ""
    
    def _create_report_content(self, summary: MetricSelectionSummary) -> str:
        """レポート内容の作成"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 重要指標選定システム実行レポート

**実行日時**: {timestamp}  
**分析成功**: {'✅ 成功' if summary.success else '❌ 失敗'}  
**信頼度レベル**: {summary.confidence_level.upper()}  

## 実行概要

### 分析結果サマリー
"""
        
        if summary.analysis_summary:
            report += f"""
- **分析対象指標数**: {summary.analysis_summary.get('total_metrics_analyzed', 'N/A')}
- **データサンプル数**: {summary.analysis_summary.get('data_samples', 'N/A')}
- **分析戦略数**: {summary.analysis_summary.get('strategies_analyzed', 'N/A')}
- **分析手法**: {', '.join(summary.analysis_summary.get('analysis_methods', []))}
- **推奨指標数**: {summary.analysis_summary.get('recommended_count', 'N/A')}
"""
        
        # 推奨指標
        if summary.recommended_metrics:
            report += "\n## 推奨指標ランキング\n\n"
            report += "| 順位 | 指標名 | 重要度スコア | 信頼度 | 手法数 |\n"
            report += "|------|--------|-------------|--------|--------|\n"
            
            for i, metric in enumerate(summary.recommended_metrics, 1):
                report += f"| {i} | {metric['feature']} | {metric['importance_score']:.3f} | {metric['confidence']} | {metric['method_count']} |\n"
        
        # 重み最適化結果
        if summary.weight_optimization_result and summary.weight_optimization_result.success:
            result = summary.weight_optimization_result
            report += f"\n## 重み最適化結果\n\n"
            report += f"- **最適化手法**: {result.optimization_method}\n"
            report += f"- **改善スコア**: {result.improvement_score:.3f}\n"
            
            report += "\n### 重みの変化\n\n"
            report += "| カテゴリ | 元の重み | 最適化後 | 変化量 |\n"
            report += "|----------|----------|----------|--------|\n"
            
            for category in result.original_weights.keys():
                original = result.original_weights[category]
                optimized = result.optimized_weights.get(category, original)
                change = optimized - original
                report += f"| {category} | {original:.3f} | {optimized:.3f} | {change:+.3f} |\n"
        
        # パフォーマンス影響
        if summary.performance_impact:
            report += "\n## パフォーマンス影響評価\n\n"
            for key, value in summary.performance_impact.items():
                report += f"- **{key}**: {value:.3f}\n"
        
        # エラーメッセージ
        if summary.error_messages:
            report += "\n## 警告・エラー\n\n"
            for i, error in enumerate(summary.error_messages, 1):
                report += f"{i}. {error}\n"
        
        # 推奨事項
        report += "\n## 推奨事項\n\n"
        
        if summary.success:
            if summary.confidence_level == "high":
                report += "- ✅ 分析結果は高い信頼性を持っています\n"
                report += "- ✅ 推奨指標の実装を進めることを推奨します\n"
            elif summary.confidence_level == "medium":
                report += "- ⚠️ 分析結果は中程度の信頼性です\n"
                report += "- ⚠️ 追加データでの検証を推奨します\n"
            else:
                report += "- ❌ 分析結果の信頼性が低いです\n"
                report += "- ❌ データ品質の改善が必要です\n"
        else:
            report += "- ❌ 分析が失敗しました\n"
            report += "- ❌ エラーを修正して再実行してください\n"
        
        report += f"\n---\n*レポート生成日時: {timestamp}*\n"
        
        return report
    
    def _update_execution_history(self, summary: MetricSelectionSummary):
        """実行履歴の更新"""
        try:
            history_entry = {
                "timestamp": summary.timestamp,
                "success": summary.success,
                "confidence_level": summary.confidence_level,
                "recommended_count": len(summary.recommended_metrics),
                "error_count": len(summary.error_messages),
                "performance_impact": summary.performance_impact
            }
            
            self.execution_history.append(history_entry)
            
            # 履歴ファイルに保存
            history_file = self.summaries_dir / "execution_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.execution_history, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"History update error: {e}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """実行履歴の取得"""
        return self.execution_history.copy()
    
    def get_latest_recommendations(self) -> List[Dict[str, Any]]:
        """最新の推奨指標を取得"""
        try:
            if self.execution_history:
                # 最新の成功した実行を探す
                for entry in reversed(self.execution_history):
                    if entry.get("success", False):
                        # 対応するレポートファイルから推奨指標を読み取り
                        timestamp = entry["timestamp"]
                        # 実装は簡略化
                        break
            
            return []
        except Exception as e:
            logger.error(f"Latest recommendations retrieval error: {e}")
            return []

# 使用例とテスト
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== 重要指標選定システム統合テスト ===")
    
    try:
        # 管理システムの初期化
        manager = MetricSelectionManager()
        print("✓ 統合管理システム初期化完了")
        
        # 完全分析の実行
        summary = manager.run_complete_analysis(
            optimization_method="balanced_approach",
            apply_weights=False  # テストでは適用しない
        )
        
        if summary.success:
            print("✓ 完全分析実行完了")
            print(f"信頼度レベル: {summary.confidence_level}")
            print(f"推奨指標数: {len(summary.recommended_metrics)}")
            
            # 推奨指標の表示
            if summary.recommended_metrics:
                print("\n推奨指標 (上位5位):")
                for i, metric in enumerate(summary.recommended_metrics[:5], 1):
                    print(f"  {i}. {metric['feature']} (スコア: {metric['importance_score']:.3f})")
            
            # 重み最適化結果
            if summary.weight_optimization_result and summary.weight_optimization_result.success:
                print(f"\n重み最適化改善スコア: {summary.weight_optimization_result.improvement_score:.3f}")
            
        else:
            print("✗ 分析失敗")
            for error in summary.error_messages:
                print(f"  エラー: {error}")
            
    except Exception as e:
        print(f"✗ テストエラー: {e}")
    
    print("\nテスト完了")
