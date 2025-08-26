"""
DSSMS Task 1.4: 切替診断システム
銘柄切替の詳細診断・分析・レポート機能

主要機能:
1. 切替決定の記録・分析
2. 成功率詳細分析
3. 日次パフォーマンス追跡
4. 診断レポート生成
5. データエクスポート機能

Author: GitHub Copilot Agent
Created: 2025-08-26
Task: 1.4 銘柄切替メカニズム復旧
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import json
import sqlite3
from collections import defaultdict, Counter

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

@dataclass
class SwitchDecisionRecord:
    """切替決定記録"""
    id: str
    timestamp: datetime
    engine_used: str
    decision_factors: Dict[str, Any]
    input_conditions: Dict[str, Any]
    output_result: Dict[str, Any]
    success: bool
    execution_time_ms: float
    performance_impact: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class DiagnosticMetrics:
    """診断メトリクス"""
    period_start: datetime
    period_end: datetime
    total_decisions: int
    successful_decisions: int
    success_rate: float
    avg_execution_time: float
    engine_performance: Dict[str, Dict[str, float]]
    decision_patterns: Dict[str, int]
    performance_impact_summary: Dict[str, float]

class SwitchDiagnostics:
    """
    DSSMS Task 1.4: 切替診断システム
    切替決定の詳細分析・診断機能
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.logger.info("=== Switch Diagnostics システム初期化開始 ===")
        
        # データベース設定
        if db_path is None:
            db_path = str(project_root / "data" / "switch_diagnostics.db")
        self.db_path = db_path
        
        # 診断記録保存
        self.decision_records: List[SwitchDecisionRecord] = []
        self.metrics_cache: Dict[str, DiagnosticMetrics] = {}
        
        # 分析設定
        self.analysis_window_days = 7
        self.success_rate_threshold = 0.30
        self.performance_threshold = 0.05
        
        self._initialize_database()
        self.logger.info("=== Switch Diagnostics システム初期化完了 ===")
    
    def _initialize_database(self):
        """データベース初期化"""
        try:
            # ディレクトリ作成
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # データベース接続・テーブル作成
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS switch_decisions (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        engine_used TEXT NOT NULL,
                        decision_factors TEXT,
                        input_conditions TEXT,
                        output_result TEXT,
                        success INTEGER NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        performance_impact REAL,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS diagnostic_metrics (
                        id TEXT PRIMARY KEY,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        total_decisions INTEGER NOT NULL,
                        successful_decisions INTEGER NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_execution_time REAL NOT NULL,
                        engine_performance TEXT,
                        decision_patterns TEXT,
                        performance_impact_summary TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info(f"データベース初期化完了: {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"データベース初期化失敗: {e}")
            raise
    
    def record_switch_decision(self, 
                             engine_used: str,
                             decision_factors: Dict[str, Any],
                             input_conditions: Dict[str, Any],
                             output_result: Dict[str, Any],
                             success: bool,
                             execution_time_ms: float,
                             performance_impact: Optional[float] = None,
                             notes: Optional[str] = None) -> str:
        """
        切替決定の記録
        
        Args:
            engine_used: 使用されたエンジン
            decision_factors: 決定要因
            input_conditions: 入力条件
            output_result: 出力結果
            success: 成功フラグ
            execution_time_ms: 実行時間
            performance_impact: パフォーマンス影響
            notes: 追加メモ
        
        Returns:
            str: 記録ID
        """
        record_id = f"SW_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now()
        
        # 記録作成
        record = SwitchDecisionRecord(
            id=record_id,
            timestamp=timestamp,
            engine_used=engine_used,
            decision_factors=decision_factors,
            input_conditions=input_conditions,
            output_result=output_result,
            success=success,
            execution_time_ms=execution_time_ms,
            performance_impact=performance_impact,
            notes=notes
        )
        
        # メモリに追加
        self.decision_records.append(record)
        
        # データベースに保存
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO switch_decisions 
                    (id, timestamp, engine_used, decision_factors, input_conditions, 
                     output_result, success, execution_time_ms, performance_impact, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.timestamp.isoformat(),
                    record.engine_used,
                    json.dumps(record.decision_factors, ensure_ascii=False),
                    json.dumps(record.input_conditions, ensure_ascii=False),
                    json.dumps(record.output_result, ensure_ascii=False),
                    1 if record.success else 0,
                    record.execution_time_ms,
                    record.performance_impact,
                    record.notes
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"データベース保存失敗: {e}")
        
        self.logger.info(f"切替決定記録完了: {record_id} (成功={success}, エンジン={engine_used})")
        return record_id
    
    def analyze_success_rate(self, 
                           period_days: Optional[int] = None,
                           engine_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        成功率分析
        
        Args:
            period_days: 分析期間（日）
            engine_filter: エンジンフィルター
        
        Returns:
            Dict[str, Any]: 成功率分析結果
        """
        if period_days is None:
            period_days = self.analysis_window_days
        
        cutoff_time = datetime.now() - timedelta(days=period_days)
        
        # 対象記録のフィルタリング
        filtered_records = [
            r for r in self.decision_records 
            if r.timestamp >= cutoff_time
        ]
        
        if engine_filter:
            filtered_records = [
                r for r in filtered_records 
                if r.engine_used == engine_filter
            ]
        
        if not filtered_records:
            return {
                "period_days": period_days,
                "total_records": 0,
                "success_rate": 0.0,
                "message": "分析対象の記録なし"
            }
        
        # 基本統計
        total_records = len(filtered_records)
        successful_records = sum(1 for r in filtered_records if r.success)
        success_rate = successful_records / total_records
        
        # エンジン別分析
        engine_analysis: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "successes": 0})
        for record in filtered_records:
            engine_analysis[record.engine_used]["total"] += 1
            if record.success:
                engine_analysis[record.engine_used]["successes"] += 1
        
        # エンジン別成功率計算
        for engine in engine_analysis:
            total = engine_analysis[engine]["total"]
            successes = engine_analysis[engine]["successes"]
            engine_analysis[engine]["success_rate"] = successes / total if total > 0 else 0.0
        
        # 時間別分析（時間帯別成功率）
        hourly_analysis: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "successes": 0})
        for record in filtered_records:
            hour = record.timestamp.hour
            hourly_analysis[hour]["total"] += 1
            if record.success:
                hourly_analysis[hour]["successes"] += 1
        
        # 時間別成功率計算
        for hour in hourly_analysis:
            total = hourly_analysis[hour]["total"]
            successes = hourly_analysis[hour]["successes"]
            hourly_analysis[hour]["success_rate"] = successes / total if total > 0 else 0.0
        
        # パフォーマンス影響分析
        perf_impacts = [r.performance_impact for r in filtered_records if r.performance_impact is not None]
        
        return {
            "analysis_period": {
                "days": period_days,
                "start_date": cutoff_time.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "overall_metrics": {
                "total_records": total_records,
                "successful_records": successful_records,
                "success_rate": success_rate,
                "target_success_rate": self.success_rate_threshold,
                "target_achieved": success_rate >= self.success_rate_threshold
            },
            "engine_performance": dict(engine_analysis),
            "hourly_patterns": dict(hourly_analysis),
            "performance_impact": {
                "samples_count": len(perf_impacts),
                "avg_impact": np.mean(perf_impacts) if perf_impacts else 0.0,
                "positive_impacts": sum(1 for p in perf_impacts if p > 0),
                "negative_impacts": sum(1 for p in perf_impacts if p < 0)
            } if perf_impacts else None,
            "execution_time_stats": {
                "avg_time_ms": np.mean([r.execution_time_ms for r in filtered_records]),
                "median_time_ms": np.median([r.execution_time_ms for r in filtered_records]),
                "max_time_ms": max([r.execution_time_ms for r in filtered_records]),
                "min_time_ms": min([r.execution_time_ms for r in filtered_records])
            }
        }
    
    def analyze_daily_performance(self, days_back: int = 30) -> Dict[str, Any]:
        """
        日次パフォーマンス分析
        
        Args:
            days_back: 分析日数
        
        Returns:
            Dict[str, Any]: 日次分析結果
        """
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        # 日別グループ化
        daily_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "decisions": 0,
            "successes": 0,
            "total_switches": 0,
            "engines_used": Counter(),
            "avg_execution_time": 0.0,
            "execution_times": []
        })
        
        for record in self.decision_records:
            if record.timestamp >= cutoff_time:
                date_key = record.timestamp.strftime('%Y-%m-%d')
                daily_data[date_key]["decisions"] += 1
                if record.success:
                    daily_data[date_key]["successes"] += 1
                
                # 切替数をoutput_resultから取得
                if "switches_count" in record.output_result:
                    daily_data[date_key]["total_switches"] += record.output_result["switches_count"]
                
                daily_data[date_key]["engines_used"][record.engine_used] += 1
                daily_data[date_key]["execution_times"].append(record.execution_time_ms)
        
        # 日別統計計算
        for date_key in daily_data:
            decisions = daily_data[date_key]["decisions"]
            successes = daily_data[date_key]["successes"]
            exec_times = daily_data[date_key]["execution_times"]
            
            daily_data[date_key]["success_rate"] = successes / decisions if decisions > 0 else 0.0
            daily_data[date_key]["avg_execution_time"] = np.mean(exec_times) if exec_times else 0.0
            daily_data[date_key]["engines_used"] = dict(daily_data[date_key]["engines_used"])
        
        # 目標達成日数計算
        target_achieved_days = sum(
            1 for data in daily_data.values() 
            if (data["success_rate"] >= self.success_rate_threshold and 
                data["total_switches"] >= 1)
        )
        
        # トレンド分析
        sorted_dates = sorted(daily_data.keys())
        success_rates = [daily_data[date]["success_rate"] for date in sorted_dates]
        switch_counts = [daily_data[date]["total_switches"] for date in sorted_dates]
        
        return {
            "analysis_period": {
                "days_analyzed": days_back,
                "actual_days_with_data": len(daily_data),
                "start_date": cutoff_time.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "daily_details": dict(daily_data),
            "summary_metrics": {
                "total_target_achieved_days": target_achieved_days,
                "target_achievement_rate": target_achieved_days / len(daily_data) if daily_data else 0.0,
                "avg_daily_success_rate": np.mean(success_rates) if success_rates else 0.0,
                "avg_daily_switches": np.mean(switch_counts) if switch_counts else 0.0,
                "total_decisions": sum(data["decisions"] for data in daily_data.values()),
                "total_successes": sum(data["successes"] for data in daily_data.values())
            },
            "trends": {
                "success_rate_trend": success_rates[-7:] if len(success_rates) >= 7 else success_rates,
                "switch_count_trend": switch_counts[-7:] if len(switch_counts) >= 7 else switch_counts,
                "dates": sorted_dates[-7:] if len(sorted_dates) >= 7 else sorted_dates
            }
        }
    
    def generate_diagnostic_report(self, 
                                 analysis_days: int = 7,
                                 include_details: bool = True) -> Dict[str, Any]:
        """
        診断レポート生成
        
        Args:
            analysis_days: 分析日数
            include_details: 詳細情報含む
        
        Returns:
            Dict[str, Any]: 診断レポート
        """
        report_timestamp = datetime.now()
        
        # 成功率分析
        success_analysis = self.analyze_success_rate(analysis_days)
        
        # 日次パフォーマンス分析
        daily_analysis = self.analyze_daily_performance(analysis_days)
        
        # エンジン比較分析
        engine_comparison = self._compare_engine_performance(analysis_days)
        
        # 問題検出
        issues = self._detect_issues(success_analysis, daily_analysis)
        
        # 改善提案
        recommendations = self._generate_recommendations(issues, success_analysis)
        
        report = {
            "report_metadata": {
                "generated_at": report_timestamp.isoformat(),
                "analysis_period_days": analysis_days,
                "report_type": "diagnostic_comprehensive"
            },
            "executive_summary": {
                "overall_success_rate": success_analysis["overall_metrics"]["success_rate"],
                "target_achievement": success_analysis["overall_metrics"]["target_achieved"],
                "daily_target_achievement_rate": daily_analysis["summary_metrics"]["target_achievement_rate"],
                "total_decisions_analyzed": success_analysis["overall_metrics"]["total_records"],
                "critical_issues_count": len([i for i in issues if i["severity"] == "critical"]),
                "recommendations_count": len(recommendations)
            },
            "success_rate_analysis": success_analysis,
            "daily_performance_analysis": daily_analysis,
            "engine_comparison": engine_comparison,
            "issues_detected": issues,
            "recommendations": recommendations
        }
        
        if include_details:
            report["detailed_records"] = self._get_recent_records_summary(analysis_days)
            report["performance_metrics"] = self._calculate_advanced_metrics(analysis_days)
        
        # レポート保存
        self._save_report(report)
        
        return report
    
    def _compare_engine_performance(self, days: int) -> Dict[str, Any]:
        """エンジンパフォーマンス比較"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        engine_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "decisions": 0,
            "successes": 0,
            "total_execution_time": 0.0,
            "switches_total": 0,
            "performance_impacts": []
        })
        
        for record in self.decision_records:
            if record.timestamp >= cutoff_time:
                engine = record.engine_used
                engine_stats[engine]["decisions"] += 1
                if record.success:
                    engine_stats[engine]["successes"] += 1
                
                engine_stats[engine]["total_execution_time"] += record.execution_time_ms
                
                if "switches_count" in record.output_result:
                    engine_stats[engine]["switches_total"] += record.output_result["switches_count"]
                
                if record.performance_impact is not None:
                    engine_stats[engine]["performance_impacts"].append(record.performance_impact)
        
        # 統計計算
        comparison_result = {}
        for engine, stats in engine_stats.items():
            decisions = stats["decisions"]
            successes = stats["successes"]
            
            comparison_result[engine] = {
                "total_decisions": decisions,
                "successful_decisions": successes,
                "success_rate": successes / decisions if decisions > 0 else 0.0,
                "avg_execution_time_ms": stats["total_execution_time"] / decisions if decisions > 0 else 0.0,
                "total_switches": stats["switches_total"],
                "avg_switches_per_decision": stats["switches_total"] / decisions if decisions > 0 else 0.0,
                "performance_impact": {
                    "sample_size": len(stats["performance_impacts"]),
                    "avg_impact": np.mean(stats["performance_impacts"]) if stats["performance_impacts"] else 0.0,
                    "positive_rate": sum(1 for p in stats["performance_impacts"] if p > 0) / len(stats["performance_impacts"]) if stats["performance_impacts"] else 0.0
                }
            }
        
        # ランキング
        engines_by_success = sorted(
            comparison_result.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        
        engines_by_speed = sorted(
            comparison_result.items(),
            key=lambda x: x[1]["avg_execution_time_ms"]
        )
        
        return {
            "engine_statistics": comparison_result,
            "rankings": {
                "by_success_rate": [(engine, stats["success_rate"]) for engine, stats in engines_by_success],
                "by_execution_speed": [(engine, stats["avg_execution_time_ms"]) for engine, stats in engines_by_speed]
            },
            "recommendations": {
                "best_overall": engines_by_success[0][0] if engines_by_success else None,
                "fastest": engines_by_speed[0][0] if engines_by_speed else None
            }
        }
    
    def _detect_issues(self, success_analysis: Dict, daily_analysis: Dict) -> List[Dict[str, Any]]:
        """問題検出"""
        issues = []
        
        # 成功率低下
        overall_success_rate = success_analysis["overall_metrics"]["success_rate"]
        if overall_success_rate < self.success_rate_threshold:
            issues.append({
                "type": "low_success_rate",
                "severity": "critical",
                "description": f"成功率が目標({self.success_rate_threshold:.1%})を下回っています",
                "current_value": overall_success_rate,
                "target_value": self.success_rate_threshold
            })
        
        # 日次目標未達成
        daily_achievement_rate = daily_analysis["summary_metrics"]["target_achievement_rate"]
        if daily_achievement_rate < 0.7:  # 70%未満
            issues.append({
                "type": "daily_target_underachievement",
                "severity": "warning",
                "description": f"日次目標達成率が低すぎます({daily_achievement_rate:.1%})",
                "current_value": daily_achievement_rate,
                "target_value": 0.7
            })
        
        # エンジン間パフォーマンス格差
        engine_perf = success_analysis.get("engine_performance", {})
        if len(engine_perf) > 1:
            success_rates = [stats["success_rate"] for stats in engine_perf.values()]
            if max(success_rates) - min(success_rates) > 0.3:  # 30%以上の差
                issues.append({
                    "type": "engine_performance_gap",
                    "severity": "warning",
                    "description": "エンジン間のパフォーマンス格差が大きすぎます",
                    "details": engine_perf
                })
        
        # 実行時間異常
        exec_stats = success_analysis.get("execution_time_stats", {})
        if exec_stats and exec_stats.get("avg_time_ms", 0) > 5000:  # 5秒以上
            issues.append({
                "type": "slow_execution",
                "severity": "warning",
                "description": f"平均実行時間が長すぎます({exec_stats['avg_time_ms']:.0f}ms)",
                "current_value": exec_stats["avg_time_ms"],
                "threshold": 5000
            })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict], analysis: Dict) -> List[Dict[str, Any]]:
        """改善提案生成"""
        recommendations = []
        
        for issue in issues:
            if issue["type"] == "low_success_rate":
                recommendations.append({
                    "priority": "high",
                    "category": "success_rate_improvement",
                    "title": "成功率向上対策",
                    "description": "V2エンジンの優先使用、パラメータ調整、緊急モード閾値見直しを推奨",
                    "specific_actions": [
                        "V2エンジンの決定ロジック調整",
                        "ハイブリッドモードの統合ルール見直し", 
                        "緊急モード発動条件の最適化"
                    ]
                })
            
            elif issue["type"] == "daily_target_underachievement":
                recommendations.append({
                    "priority": "medium",
                    "category": "daily_execution",
                    "title": "日次実行頻度向上",
                    "description": "最小実行保証機能の強化と強制実行ルールの追加を推奨",
                    "specific_actions": [
                        "日次最小切替保証の実装",
                        "時間ベース強制実行の設定",
                        "目標未達成時の自動調整"
                    ]
                })
            
            elif issue["type"] == "engine_performance_gap":
                recommendations.append({
                    "priority": "medium",
                    "category": "engine_optimization",
                    "title": "エンジン最適化",
                    "description": "高性能エンジンの優先使用と低性能エンジンの改善を推奨",
                    "specific_actions": [
                        "成功率の高いエンジンの優先使用",
                        "低性能エンジンのパラメータ調整",
                        "エンジン選択ロジックの最適化"
                    ]
                })
            
            elif issue["type"] == "slow_execution":
                recommendations.append({
                    "priority": "low",
                    "category": "performance",
                    "title": "実行速度改善",
                    "description": "処理の並列化とタイムアウト設定の最適化を推奨",
                    "specific_actions": [
                        "データ処理の並列化",
                        "キャッシュ機能の導入",
                        "タイムアウト値の調整"
                    ]
                })
        
        # 一般的な改善提案
        if success_analysis["overall_metrics"]["success_rate"] > 0.5:
            recommendations.append({
                "priority": "low",
                "category": "optimization",
                "title": "さらなる最適化",
                "description": "機械学習による予測精度向上とリアルタイム調整機能の検討",
                "specific_actions": [
                    "決定要因の重要度分析",
                    "過去データによる学習機能",
                    "リアルタイムパラメータ調整"
                ]
            })
        
        return recommendations
    
    def _get_recent_records_summary(self, days: int) -> List[Dict[str, Any]]:
        """最近の記録サマリー"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_records = [
            r for r in self.decision_records 
            if r.timestamp >= cutoff_time
        ]
        
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "engine_used": r.engine_used,
                "success": r.success,
                "execution_time_ms": r.execution_time_ms,
                "switches_count": r.output_result.get("switches_count", 0),
                "performance_impact": r.performance_impact
            }
            for r in recent_records[-50:]  # 最新50件
        ]
    
    def _calculate_advanced_metrics(self, days: int) -> Dict[str, Any]:
        """高度なメトリクス計算"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_records = [
            r for r in self.decision_records 
            if r.timestamp >= cutoff_time
        ]
        
        if not recent_records:
            return {}
        
        # 成功パターン分析
        success_patterns: Dict[str, int] = defaultdict(int)
        failure_patterns: Dict[str, int] = defaultdict(int)
        
        for record in recent_records:
            pattern_key = f"{record.engine_used}"
            if record.success:
                success_patterns[pattern_key] += 1
            else:
                failure_patterns[pattern_key] += 1
        
        # 時系列安定性
        success_by_time = [(r.timestamp, r.success) for r in recent_records]
        success_by_time.sort()
        
        # 移動平均（7日）
        window_size = min(7, len(success_by_time))
        moving_averages = []
        for i in range(window_size - 1, len(success_by_time)):
            window_successes = sum(
                1 for _, success in success_by_time[i-window_size+1:i+1] 
                if success
            )
            moving_averages.append(window_successes / window_size)
        
        return {
            "success_patterns": dict(success_patterns),
            "failure_patterns": dict(failure_patterns),
            "stability_metrics": {
                "success_rate_variance": np.var([r.success for r in recent_records]),
                "moving_average_trend": moving_averages[-5:] if len(moving_averages) >= 5 else moving_averages,
                "consistency_score": 1.0 - np.var(moving_averages) if moving_averages else 0.0
            },
            "efficiency_metrics": {
                "decisions_per_day": len(recent_records) / days if days > 0 else 0,
                "successful_decisions_per_day": sum(1 for r in recent_records if r.success) / days if days > 0 else 0,
                "avg_switches_per_success": np.mean([
                    r.output_result.get("switches_count", 0) 
                    for r in recent_records if r.success
                ]) if any(r.success for r in recent_records) else 0.0
            }
        }
    
    def _save_report(self, report: Dict[str, Any]):
        """レポート保存"""
        try:
            # ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = project_root / "output" / "switch_diagnostics"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"diagnostic_report_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"診断レポート保存完了: {report_file}")
            
        except Exception as e:
            self.logger.error(f"レポート保存失敗: {e}")
    
    def export_data(self, 
                   format_type: str = "json",
                   period_days: Optional[int] = None) -> str:
        """
        データエクスポート
        
        Args:
            format_type: エクスポート形式（json/csv/xlsx）
            period_days: エクスポート期間
        
        Returns:
            str: エクスポートファイルパス
        """
        if period_days:
            cutoff_time = datetime.now() - timedelta(days=period_days)
            export_records = [
                r for r in self.decision_records 
                if r.timestamp >= cutoff_time
            ]
        else:
            export_records = self.decision_records
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = project_root / "output" / "switch_diagnostics"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == "json":
            export_file = export_dir / f"switch_data_export_{timestamp}.json"
            export_data = [asdict(r) for r in export_records]
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        elif format_type.lower() == "csv":
            export_file = export_dir / f"switch_data_export_{timestamp}.csv"
            
            # DataFrameに変換
            df_data = []
            for record in export_records:
                row = {
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "engine_used": record.engine_used,
                    "success": record.success,
                    "execution_time_ms": record.execution_time_ms,
                    "performance_impact": record.performance_impact,
                    "notes": record.notes
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(export_file, index=False, encoding='utf-8-sig')
        
        elif format_type.lower() == "xlsx":
            export_file = export_dir / f"switch_data_export_{timestamp}.xlsx"
            
            # DataFrameに変換
            df_data = []
            for record in export_records:
                row = {
                    "ID": record.id,
                    "タイムスタンプ": record.timestamp.isoformat(),
                    "使用エンジン": record.engine_used,
                    "成功": "○" if record.success else "×",
                    "実行時間(ms)": record.execution_time_ms,
                    "パフォーマンス影響": record.performance_impact,
                    "備考": record.notes
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_excel(export_file, index=False, engine='openpyxl')
        
        else:
            raise ValueError(f"未対応のエクスポート形式: {format_type}")
        
        self.logger.info(f"データエクスポート完了: {export_file}")
        return str(export_file)
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        return {
            "database_path": self.db_path,
            "records_in_memory": len(self.decision_records),
            "analysis_window_days": self.analysis_window_days,
            "success_rate_threshold": self.success_rate_threshold,
            "last_record_time": self.decision_records[-1].timestamp.isoformat() if self.decision_records else None,
            "system_health": "正常" if len(self.decision_records) > 0 else "記録なし"
        }
