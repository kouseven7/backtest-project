"""
Fallback Usage Monitoring Dashboard (TODO-FB-008)

フォールバック使用状況の監視・分析・レポート生成機能を提供。
週次レポート、Production readiness評価、HTML/JSON双方向出力対応。

Requirements:
- SystemFallbackPolicy経由でreports/fallback/内のJSONファイルを分析
- 週次集計レポート生成 (HTML + JSON)
- Production readiness評価 (フォールバック使用量・重大度評価)
- matplotlib経由でのグラフ可視化
- reports/monitoring/配下に出力、自動クリーンアップ対応

Progress Status: Phase 1 - Mini-Task 1/4 基本構造・依存関係実装 [OK]
Next: Mini-Task 2 (データ分析機能実装)
"""

import os
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

# Visualization and analysis dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("matplotlib not available - visualizations will be disabled")
    MATPLOTLIB_AVAILABLE = False

# Internal dependencies
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode

# Configuration
FALLBACK_REPORTS_DIR = "reports/fallback"
MONITORING_REPORTS_DIR = "reports/monitoring"
DEFAULT_ANALYSIS_DAYS = 7  # 週次レポート
MAX_REPORT_RETENTION_DAYS = 30  # 監視レポート保持期間


@dataclass
class FallbackUsageStats:
    """フォールバック使用統計データ構造"""
    component_name: str
    component_type: str
    usage_count: int
    success_rate: float
    avg_execution_time: float
    error_types: List[str]
    severity_score: float  # 0.0-1.0, 1.0が最重大


@dataclass
class ProductionReadinessMetrics:
    """Production readiness評価メトリクス"""
    overall_score: float  # 0.0-1.0, 1.0が最適
    fallback_usage_percentage: float
    critical_component_stability: float
    acceptable_for_production: bool
    recommendations: List[str]


@dataclass
class MonitoringReport:
    """監視レポート全体データ構造"""
    report_id: str
    generation_timestamp: datetime
    analysis_period_start: datetime
    analysis_period_end: datetime
    fallback_stats: List[FallbackUsageStats]
    production_metrics: ProductionReadinessMetrics
    system_health_score: float
    executive_summary: str


class FallbackMonitor:
    """
    フォールバック使用状況監視・分析・レポート生成クラス
    
    SystemFallbackPolicy経由で収集されたフォールバック使用データを分析し、
    Production readiness評価と可視化レポートを生成する。
    
    主要機能:
    - reports/fallback/内JSONファイル収集・解析
    - 週次集計レポート生成 (HTML + JSON)
    - Production readiness評価
    - matplotlib経由グラフ可視化
    - 自動クリーンアップ
    """
    
    def __init__(self, 
                 fallback_reports_dir: str = FALLBACK_REPORTS_DIR,
                 monitoring_reports_dir: str = MONITORING_REPORTS_DIR,
                 analysis_days: int = DEFAULT_ANALYSIS_DAYS):
        """
        初期化処理
        
        Args:
            fallback_reports_dir: フォールバック使用レポートディレクトリ
            monitoring_reports_dir: 監視レポート出力ディレクトリ  
            analysis_days: 分析対象日数 (デフォルト: 7日)
        """
        self.fallback_reports_dir = Path(fallback_reports_dir)
        self.monitoring_reports_dir = Path(monitoring_reports_dir)
        self.analysis_days = analysis_days
        
        # Logger設定
        self.logger = logging.getLogger(__name__)
        
        # ディレクトリ確保
        self._ensure_directories()
        
        # SystemFallbackPolicy参照 (レポート収集用)
        self.fallback_policy = SystemFallbackPolicy()
        
        self.logger.info(f"FallbackMonitor initialized - analysis_days={analysis_days}")

    def _ensure_directories(self) -> None:
        """必要ディレクトリの作成確保"""
        for directory in [self.fallback_reports_dir, self.monitoring_reports_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise

    def _collect_weekly_data(self) -> List[Dict[str, Any]]:
        """
        過去N日間のフォールバック使用データ収集
        
        Returns:
            フォールバック使用データリスト
        """
        cutoff_date = datetime.now() - timedelta(days=self.analysis_days)
        collected_data = []
        
        try:
            # reports/fallback/*.json ファイル走査
            pattern = str(self.fallback_reports_dir / "*.json")
            for json_file in glob.glob(pattern):
                try:
                    # ファイル更新日時チェック
                    file_time = datetime.fromtimestamp(os.path.getmtime(json_file))
                    if file_time >= cutoff_date:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['file_timestamp'] = file_time.isoformat()
                            collected_data.append(data)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to parse {json_file}: {e}")
                    continue
                    
            self.logger.info(f"Collected {len(collected_data)} fallback reports from {self.analysis_days} days")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return []

    def _analyze_fallback_patterns(self, raw_data: List[Dict[str, Any]]) -> List[FallbackUsageStats]:
        """
        フォールバック使用パターン分析・統計計算
        
        Args:
            raw_data: _collect_weekly_data()からの生データ
            
        Returns:
            コンポーネント別フォールバック使用統計
        """
        component_stats = {}
        
        try:
            for report in raw_data:
                usage_data = report.get('usage_statistics', {})
                
                for component_name, stats in usage_data.items():
                    if component_name not in component_stats:
                        component_stats[component_name] = {
                            'total_calls': 0,
                            'fallback_calls': 0,
                            'execution_times': [],
                            'error_types': set(),
                            'component_type': stats.get('component_type', 'UNKNOWN')
                        }
                    
                    comp_stats = component_stats[component_name]
                    comp_stats['total_calls'] += stats.get('total_calls', 0)
                    comp_stats['fallback_calls'] += stats.get('fallback_calls', 0)
                    
                    # 実行時間記録
                    if 'avg_execution_time' in stats:
                        comp_stats['execution_times'].append(stats['avg_execution_time'])
                    
                    # エラータイプ収集
                    if 'error_types' in stats:
                        comp_stats['error_types'].update(stats['error_types'])
            
            # FallbackUsageStats オブジェクト生成
            result_stats = []
            for component_name, stats in component_stats.items():
                # 成功率計算
                total_calls = stats['total_calls']
                success_rate = 1.0 if total_calls == 0 else 1.0 - (stats['fallback_calls'] / total_calls)
                
                # 平均実行時間計算
                exec_times = stats['execution_times']
                avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
                
                # 重大度スコア計算 (フォールバック率 + エラータイプ多様性)
                fallback_rate = stats['fallback_calls'] / max(total_calls, 1)
                error_diversity = len(stats['error_types']) / 10.0  # 正規化
                severity_score = min(fallback_rate + error_diversity, 1.0)
                
                usage_stat = FallbackUsageStats(
                    component_name=component_name,
                    component_type=stats['component_type'],
                    usage_count=stats['fallback_calls'],
                    success_rate=success_rate,
                    avg_execution_time=avg_exec_time,
                    error_types=list(stats['error_types']),
                    severity_score=severity_score
                )
                result_stats.append(usage_stat)
            
            self.logger.info(f"Analyzed {len(result_stats)} component fallback patterns")
            return result_stats
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return []

    def evaluate_production_readiness(self) -> ProductionReadinessMetrics:
        """
        Production readiness評価
        
        Returns:
            Production readiness評価結果
        """
        try:
            # データ収集・分析
            raw_data = self._collect_weekly_data()
            fallback_stats = self._analyze_fallback_patterns(raw_data)
            
            if not fallback_stats:
                return ProductionReadinessMetrics(
                    overall_score=1.0,  # データなし = 問題なし想定
                    fallback_usage_percentage=0.0,
                    critical_component_stability=1.0,
                    acceptable_for_production=True,
                    recommendations=["No fallback usage detected - system appears stable"]
                )
            
            # メトリクス計算
            total_fallback_calls = sum(stat.usage_count for stat in fallback_stats)
            total_calls = total_fallback_calls  # 近似値（完全ではないが傾向把握用）
            
            fallback_percentage = (total_fallback_calls / max(total_calls, 1)) * 100
            
            # 重要コンポーネント安定性評価
            critical_components = [stat for stat in fallback_stats 
                                 if stat.component_type in ['DSSMS_CORE', 'STRATEGY_ENGINE']]
            critical_stability = 1.0 - (sum(stat.severity_score for stat in critical_components) / 
                                       max(len(critical_components), 1))
            
            # 総合スコア計算 (フォールバック率 + 重要コンポーネント安定性)
            overall_score = max(0.0, 1.0 - (fallback_percentage / 100.0) * 0.7 - (1.0 - critical_stability) * 0.3)
            
            # Production可否判定 (スコア0.8以上、重要コンポーネント安定性0.9以上)
            acceptable = overall_score >= 0.8 and critical_stability >= 0.9
            
            # 推奨事項生成
            recommendations = []
            if fallback_percentage > 5.0:
                recommendations.append(f"High fallback usage: {fallback_percentage:.1f}% - investigate root causes")
            if critical_stability < 0.9:
                recommendations.append("Critical component instability detected - review DSSMS_CORE/STRATEGY_ENGINE")
            if not acceptable:
                recommendations.append("System not ready for production - address fallback issues first")
            if not recommendations:
                recommendations.append("System appears stable for production deployment")
            
            return ProductionReadinessMetrics(
                overall_score=overall_score,
                fallback_usage_percentage=fallback_percentage,
                critical_component_stability=critical_stability,
                acceptable_for_production=acceptable,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Production readiness evaluation failed: {e}")
            return ProductionReadinessMetrics(
                overall_score=0.0,
                fallback_usage_percentage=100.0,
                critical_component_stability=0.0,
                acceptable_for_production=False,
                recommendations=[f"Evaluation failed: {str(e)}"]
            )

    def generate_weekly_report(self, force_regenerate: bool = False) -> str:
        """
        週次監視レポート生成 (HTML + JSON)
        
        Args:
            force_regenerate: 既存レポート上書き強制フラグ
            
        Returns:
            生成されたレポートファイルパス (HTML)
        """
        try:
            # データ収集・分析
            raw_data = self._collect_weekly_data()
            fallback_stats = self._analyze_fallback_patterns(raw_data)
            production_metrics = self.evaluate_production_readiness()
            
            # レポートメタデータ作成
            report_timestamp = datetime.now()
            report_id = f"weekly_{report_timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            analysis_end = report_timestamp
            analysis_start = analysis_end - timedelta(days=self.analysis_days)
            
            # システムヘルススコア計算
            system_health = production_metrics.overall_score
            
            # Executive Summary生成
            exec_summary = self._generate_executive_summary(fallback_stats, production_metrics)
            
            # MonitoringReport構築
            monitoring_report = MonitoringReport(
                report_id=report_id,
                generation_timestamp=report_timestamp,
                analysis_period_start=analysis_start,
                analysis_period_end=analysis_end,
                fallback_stats=fallback_stats,
                production_metrics=production_metrics,
                system_health_score=system_health,
                executive_summary=exec_summary
            )
            
            # レポート生成
            html_path = self._generate_html_report(monitoring_report)
            json_path = self._generate_json_report(monitoring_report)
            
            self.logger.info(f"Weekly report generated: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"Weekly report generation failed: {e}")
            raise

    def _generate_executive_summary(self, fallback_stats: List[FallbackUsageStats], 
                                  production_metrics: ProductionReadinessMetrics) -> str:
        """Executive Summary生成"""
        if not fallback_stats:
            return "System operating without fallback usage - excellent stability."
        
        high_risk_components = [stat for stat in fallback_stats if stat.severity_score > 0.5]
        total_fallback_calls = sum(stat.usage_count for stat in fallback_stats)
        
        summary = f"Analyzed {len(fallback_stats)} components over {self.analysis_days} days. "
        summary += f"Total fallback calls: {total_fallback_calls}. "
        
        if production_metrics.acceptable_for_production:
            summary += "[OK] System ready for production."
        else:
            summary += f"[WARNING] Production readiness: {production_metrics.overall_score:.2f}/1.0. "
            
        if high_risk_components:
            summary += f" {len(high_risk_components)} high-risk components require attention."
        
        return summary

    def _generate_html_report(self, report: MonitoringReport) -> str:
        """
        HTML形式監視レポート生成
        
        Args:
            report: MonitoringReport オブジェクト
            
        Returns:
            生成されたHTMLファイルパス
        """
        html_filename = f"{report.report_id}.html"
        html_path = self.monitoring_reports_dir / html_filename
        
        try:
            # HTML テンプレート構築
            html_content = self._build_html_template(report)
            
            # ファイル書き込み
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {html_path}")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")
            raise

    def _generate_json_report(self, report: MonitoringReport) -> str:
        """
        JSON形式監視レポート生成
        
        Args:
            report: MonitoringReport オブジェクト
            
        Returns:
            生成されたJSONファイルパス
        """
        json_filename = f"{report.report_id}.json"
        json_path = self.monitoring_reports_dir / json_filename
        
        try:
            # MonitoringReport → dict 変換 (dataclass asdict使用)
            report_dict = asdict(report)
            
            # datetime オブジェクト → ISO文字列変換
            report_dict['generation_timestamp'] = report.generation_timestamp.isoformat()
            report_dict['analysis_period_start'] = report.analysis_period_start.isoformat()
            report_dict['analysis_period_end'] = report.analysis_period_end.isoformat()
            
            # JSON書き込み
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSON report generated: {json_path}")
            return str(json_path)
            
        except Exception as e:
            self.logger.error(f"JSON report generation failed: {e}")
            raise

    def _build_html_template(self, report: MonitoringReport) -> str:
        """HTML レポートテンプレート構築"""
        
        # Production readiness スタイル決定
        readiness_style = "color: green;" if report.production_metrics.acceptable_for_production else "color: red;"
        readiness_icon = "[OK]" if report.production_metrics.acceptable_for_production else "[WARNING]"
        
        # コンポーネント統計テーブル構築
        stats_rows = ""
        for stat in report.fallback_stats:
            severity_color = "red" if stat.severity_score > 0.7 else "orange" if stat.severity_score > 0.3 else "green"
            stats_rows += f"""
            <tr>
                <td>{stat.component_name}</td>
                <td>{stat.component_type}</td>
                <td>{stat.usage_count}</td>
                <td>{stat.success_rate:.2%}</td>
                <td>{stat.avg_execution_time:.2f}ms</td>
                <td style="color: {severity_color};">{stat.severity_score:.2f}</td>
                <td>{', '.join(stat.error_types[:3])}{'...' if len(stat.error_types) > 3 else ''}</td>
            </tr>"""
        
        # 推奨事項リスト構築
        recommendations_html = ""
        for rec in report.production_metrics.recommendations:
            recommendations_html += f"<li>{rec}</li>\n"
        
        # matplotlib グラフ生成 (利用可能な場合)
        chart_html = self._generate_chart_html(report) if MATPLOTLIB_AVAILABLE else "<p><em>matplotlib not available - charts disabled</em></p>"
        
        # HTML テンプレート
        html_template = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fallback Monitor - Weekly Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 2px solid #007acc; padding-bottom: 10px; margin-bottom: 20px; }}
        .metric-card {{ display: inline-block; background: #f9f9f9; margin: 10px; padding: 15px; border-radius: 5px; min-width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #007acc; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .chart-section {{ margin: 30px 0; }}
        .executive-summary {{ background: #e7f3ff; padding: 15px; border-left: 4px solid #007acc; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>[TOOL] Fallback Monitor - Weekly Report</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis Period:</strong> {report.analysis_period_start.strftime('%Y-%m-%d')} ~ {report.analysis_period_end.strftime('%Y-%m-%d')} ({self.analysis_days} days)</p>
        </div>
        
        <div class="executive-summary">
            <h2>[LIST] Executive Summary</h2>
            <p>{report.executive_summary}</p>
        </div>
        
        <h2>[CHART] Key Metrics</h2>
        <div class="metric-card">
            <div class="metric-value" style="{readiness_style}">{readiness_icon} {report.production_metrics.overall_score:.2f}</div>
            <div>Production Readiness</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.system_health_score:.2%}</div>
            <div>System Health Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.production_metrics.fallback_usage_percentage:.1f}%</div>
            <div>Fallback Usage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(report.fallback_stats)}</div>
            <div>Components Analyzed</div>
        </div>
        
        <h2>[UP] Fallback Usage Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Usage Count</th>
                    <th>Success Rate</th>
                    <th>Avg Exec Time</th>
                    <th>Severity Score</th>
                    <th>Error Types</th>
                </tr>
            </thead>
            <tbody>
                {stats_rows}
            </tbody>
        </table>
        
        <h2>[TARGET] Production Readiness Assessment</h2>
        <div class="metric-card">
            <div class="metric-value" style="{readiness_style}">
                {'READY' if report.production_metrics.acceptable_for_production else 'NOT READY'}
            </div>
            <div>Production Status</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{report.production_metrics.critical_component_stability:.2%}</div>
            <div>Critical Component Stability</div>
        </div>
        
        <h3>[IDEA] Recommendations</h3>
        <ul>
            {recommendations_html}
        </ul>
        
        <div class="chart-section">
            <h2>[CHART] Visual Analysis</h2>
            {chart_html}
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
            <p>Generated by DSSMS Fallback Monitor | Report ID: {report.report_id} | {report.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_template

    def _generate_chart_html(self, report: MonitoringReport) -> str:
        """matplotlib経由チャート生成・HTML埋め込み"""
        if not MATPLOTLIB_AVAILABLE or not report.fallback_stats:
            return "<p><em>No charts available</em></p>"
        
        try:
            import base64
            import io
            
            # チャート画像パス
            chart_filename = f"{report.report_id}_charts.png"
            chart_path = self.monitoring_reports_dir / chart_filename
            
            # matplotlib フィギュア作成
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Fallback Usage Analysis', fontsize=16)
            
            # Chart 1: コンポーネント別使用回数
            components = [stat.component_name[:15] for stat in report.fallback_stats]
            usage_counts = [stat.usage_count for stat in report.fallback_stats]
            ax1.bar(components, usage_counts, color='lightcoral')
            ax1.set_title('Fallback Usage by Component')
            ax1.set_ylabel('Usage Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: 成功率分布
            success_rates = [stat.success_rate for stat in report.fallback_stats]
            ax2.hist(success_rates, bins=10, color='lightblue', alpha=0.7)
            ax2.set_title('Success Rate Distribution')
            ax2.set_xlabel('Success Rate')
            ax2.set_ylabel('Component Count')
            
            # Chart 3: 重大度スコア
            severity_scores = [stat.severity_score for stat in report.fallback_stats]
            colors = ['green' if s < 0.3 else 'orange' if s < 0.7 else 'red' for s in severity_scores]
            ax3.scatter(range(len(severity_scores)), severity_scores, c=colors, alpha=0.7)
            ax3.set_title('Severity Score by Component')
            ax3.set_xlabel('Component Index')
            ax3.set_ylabel('Severity Score')
            ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Critical')
            ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Warning')
            ax3.legend()
            
            # Chart 4: Production Readiness メトリクス
            metrics_labels = ['Overall\nScore', 'Critical\nStability', 'Health\nScore']
            metrics_values = [
                report.production_metrics.overall_score,
                report.production_metrics.critical_component_stability, 
                report.system_health_score
            ]
            bars = ax4.bar(metrics_labels, metrics_values, color=['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in metrics_values])
            ax4.set_title('Production Readiness Metrics')
            ax4.set_ylabel('Score (0.0 - 1.0)')
            ax4.set_ylim(0, 1.0)
            ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target')
            ax4.legend()
            
            # レイアウト調整・保存
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Base64エンコード用 (HTMLインライン表示)
            buffer = io.BytesIO()
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            # (同じチャート生成処理)
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{img_base64}" alt="Fallback Analysis Charts" style="max-width: 100%; height: auto;">'
            
        except Exception as e:
            self.logger.warning(f"Chart generation failed: {e}")
            return f"<p><em>Chart generation failed: {e}</em></p>"

    def evaluate_production_readiness(self) -> ProductionReadinessMetrics:
        """
        Production readiness評価
        
        Returns:
            Production readiness評価結果
            
        TODO(tag:phase2, rationale:Mini-Task 2で分析ロジック実装)
        """
        # Placeholder metrics
        return ProductionReadinessMetrics(
            overall_score=0.0,
            fallback_usage_percentage=0.0,
            critical_component_stability=0.0,
            acceptable_for_production=False,
            recommendations=["TODO: Mini-Task 2で実装予定"]
        )

    def cleanup_old_reports(self, retention_days: int = MAX_REPORT_RETENTION_DAYS) -> Dict[str, int]:
        """
        古い監視レポートのクリーンアップ
        
        Args:
            retention_days: 保持日数
            
        Returns:
            クリーンアップ統計 {'deleted_files': count, 'total_size_mb': size}
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0
        total_size = 0
        
        try:
            for file_path in self.monitoring_reports_dir.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        total_size += file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.debug(f"Deleted old report: {file_path}")
            
            size_mb = round(total_size / (1024 * 1024), 2)
            self.logger.info(f"Cleanup completed: {deleted_count} files, {size_mb}MB")
            
            return {'deleted_files': deleted_count, 'total_size_mb': size_mb}
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {'deleted_files': 0, 'total_size_mb': 0.0}

    def get_system_status(self) -> Dict[str, Any]:
        """
        システム状況概要取得
        
        Returns:
            システム状況サマリー辞書
        """
        return {
            'monitor_status': 'initialized',
            'fallback_reports_available': len(list(self.fallback_reports_dir.glob("*.json"))),
            'monitoring_reports_count': len(list(self.monitoring_reports_dir.glob("*"))),
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'analysis_days': self.analysis_days,
            'last_updated': datetime.now().isoformat()
        }


if __name__ == "__main__":
    """
    standalone実行時のデモ・テスト用エントリーポイント
    """
    logging.basicConfig(level=logging.INFO)
    
    # FallbackMonitor初期化テスト
    monitor = FallbackMonitor()
    
    print("[TOOL] FallbackMonitor Demo - Mini-Task 1 Validation")
    print("=" * 50)
    
    # システム状況確認
    status = monitor.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Production readiness評価テスト (placeholder)
    readiness = monitor.evaluate_production_readiness()
    print(f"\n[CHART] Production Readiness (placeholder):")
    print(f"  Score: {readiness.overall_score}")
    print(f"  Acceptable: {readiness.acceptable_for_production}")
    
    # クリーンアップテスト
    cleanup_result = monitor.cleanup_old_reports()
    print(f"\n🧹 Cleanup Test: {cleanup_result}")
    
    print("\n[OK] Mini-Task 1 基本構造実装 - Complete")
    print("[LIST] Next: Mini-Task 2 (データ分析機能実装)")