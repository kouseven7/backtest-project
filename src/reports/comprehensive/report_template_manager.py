"""
レポートテンプレート管理モジュール

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムのテンプレート管理コンポーネント

機能:
- Bootstrap5ベースのレスポンシブテンプレート
- HTML/CSS/JavaScriptテンプレート管理
- カスタマイズ可能なテーマシステム
- レベル別テンプレート生成
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from string import Template

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class ReportTemplateManager:
    """レポートテンプレート管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.logger.info("=== ReportTemplateManager 初期化開始 ===")
        
        # テンプレートディレクトリ
        self.template_dir = project_root / "templates" / "reports"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # テンプレート設定
        self.template_config = {
            'theme': 'bootstrap5',
            'responsive': True,
            'dark_mode': False,
            'animation_enabled': True,
            'chart_library': 'chartjs',
            'font_family': 'Arial, sans-serif',
            'primary_color': '#3498db',
            'secondary_color': '#2c3e50',
            'success_color': '#2ecc71',
            'warning_color': '#f39c12',
            'danger_color': '#e74c3c'
        }
        
        # HTMLテンプレート初期化
        self.html_templates = self._initialize_html_templates()
        
        self.logger.info("ReportTemplateManager 初期化完了")
    
    def _initialize_html_templates(self) -> Dict[str, Template]:
        """HTMLテンプレート初期化"""
        templates = {}
        
        # ベーステンプレート
        templates['base'] = Template("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- Custom CSS -->
    <style>
        $custom_css
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- ヘッダー -->
        <header class="bg-primary text-white py-4 mb-4">
            <div class="container">
                <h1 class="display-4">$header_title</h1>
                <p class="lead">$header_subtitle</p>
            </div>
        </header>
        
        <!-- メインコンテンツ -->
        <main>
            $main_content
        </main>
        
        <!-- フッター -->
        <footer class="bg-light text-center py-4 mt-5">
            <div class="container">
                <p class="text-muted">Generated on $generation_timestamp by DSSMS Comprehensive Report System</p>
            </div>
        </footer>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Chart.js初期化 -->
    <script>
        $chart_initialization_js
    </script>
    
    <!-- カスタムJavaScript -->
    <script>
        $custom_js
    </script>
</body>
</html>
        """)
        
        # 包括的レポートテンプレート
        templates['comprehensive'] = Template("""
        <!-- レポートメタデータ -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h2><i class="fas fa-chart-line"></i> レポート概要</h2>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>レポート情報</h5>
                                <ul class="list-unstyled">
                                    <li><strong>レポートID:</strong> $report_id</li>
                                    <li><strong>生成日時:</strong> $generation_timestamp</li>
                                    <li><strong>レポートタイプ:</strong> $report_type</li>
                                    <li><strong>詳細レベル:</strong> $detail_level</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5>データサマリー</h5>
                                <ul class="list-unstyled">
                                    $data_summary_list
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 可視化セクション -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h2><i class="fas fa-chart-bar"></i> データ可視化</h2>
                    </div>
                    <div class="card-body">
                        $visualization_content
                    </div>
                </div>
            </div>
        </div>
        
        <!-- データテーブル -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h2><i class="fas fa-table"></i> データテーブル</h2>
                    </div>
                    <div class="card-body">
                        $data_tables_content
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 詳細分析 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h2><i class="fas fa-analytics"></i> 詳細分析</h2>
                    </div>
                    <div class="card-body">
                        $detailed_analysis_content
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # 比較レポートテンプレート
        templates['comparison'] = Template("""
        <!-- 比較概要 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2><i class="fas fa-balance-scale"></i> 比較分析</h2>
                    </div>
                    <div class="card-body">
                        <h5>比較タイプ: $comparison_type</h5>
                        <p>比較対象: $comparison_items_count 項目</p>
                        $comparison_summary
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 比較可視化 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h2><i class="fas fa-chart-pie"></i> 比較チャート</h2>
                    </div>
                    <div class="card-body">
                        $comparison_visualization_content
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 比較詳細 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h2><i class="fas fa-list"></i> 比較詳細</h2>
                    </div>
                    <div class="card-body">
                        $comparison_details_content
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # パフォーマンスレポートテンプレート
        templates['performance'] = Template("""
        <!-- パフォーマンス概要 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h2><i class="fas fa-tachometer-alt"></i> パフォーマンス分析</h2>
                    </div>
                    <div class="card-body">
                        $performance_summary
                    </div>
                </div>
            </div>
        </div>
        
        <!-- パフォーマンスメトリクス -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h2><i class="fas fa-chart-area"></i> パフォーマンスメトリクス</h2>
                    </div>
                    <div class="card-body">
                        $performance_metrics_content
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # リスクレポートテンプレート
        templates['risk'] = Template("""
        <!-- リスク概要 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-danger text-white">
                        <h2><i class="fas fa-shield-alt"></i> リスク分析</h2>
                    </div>
                    <div class="card-body">
                        $risk_summary
                    </div>
                </div>
            </div>
        </div>
        
        <!-- リスクメトリクス -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h2><i class="fas fa-exclamation-triangle"></i> リスクメトリクス</h2>
                    </div>
                    <div class="card-body">
                        $risk_metrics_content
                    </div>
                </div>
            </div>
        </div>
        """)
        
        return templates
    
    def generate_html_report(
        self,
        report_data: Dict[str, Any],
        template_type: str = "comprehensive",
        level: str = "comprehensive"
    ) -> str:
        """
        HTMLレポート生成
        
        Args:
            report_data: レポートデータ
            template_type: テンプレートタイプ
            level: 詳細レベル
            
        Returns:
            生成されたHTML
        """
        try:
            self.logger.info(f"HTMLレポート生成開始: {template_type}, レベル: {level}")
            
            # メタデータ取得
            metadata = report_data.get('metadata', {})
            
            # メインコンテンツ生成
            main_content = self._generate_main_content(report_data, template_type, level)
            
            # カスタムCSS生成
            custom_css = self._generate_custom_css(level)
            
            # カスタムJavaScript生成
            custom_js = self._generate_custom_javascript(report_data)
            
            # Chart.js初期化JavaScript
            chart_js = self._generate_chart_javascript(report_data)
            
            # ベーステンプレートの変数置換
            html_content = self.html_templates['base'].substitute(
                title=f"包括的レポート - {metadata.get('report_type', 'Unknown')}",
                header_title="DSSMS包括的レポートシステム",
                header_subtitle=f"レポートタイプ: {metadata.get('report_type', 'Unknown')} | 生成日時: {metadata.get('generation_timestamp', 'Unknown')}",
                main_content=main_content,
                generation_timestamp=metadata.get('generation_timestamp', datetime.now()),
                custom_css=custom_css,
                custom_js=custom_js,
                chart_initialization_js=chart_js
            )
            
            self.logger.info("HTMLレポート生成完了")
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")
            return self._generate_error_page(str(e))
    
    def _generate_main_content(
        self,
        report_data: Dict[str, Any],
        template_type: str,
        level: str
    ) -> str:
        """メインコンテンツ生成"""
        try:
            if template_type not in self.html_templates:
                template_type = "comprehensive"
            
            template = self.html_templates[template_type]
            metadata = report_data.get('metadata', {})
            
            if template_type == "comprehensive":
                return self._generate_comprehensive_content(report_data, template, level)
            elif template_type == "comparison":
                return self._generate_comparison_content(report_data, template, level)
            elif template_type == "performance":
                return self._generate_performance_content(report_data, template, level)
            elif template_type == "risk":
                return self._generate_risk_content(report_data, template, level)
            else:
                return self._generate_comprehensive_content(report_data, template, level)
                
        except Exception as e:
            self.logger.error(f"メインコンテンツ生成エラー: {e}")
            return f"<div class='alert alert-danger'>コンテンツ生成エラー: {e}</div>"
    
    def _generate_comprehensive_content(
        self,
        report_data: Dict[str, Any],
        template: Template,
        level: str
    ) -> str:
        """包括的レポートコンテンツ生成"""
        try:
            metadata = report_data.get('metadata', {})
            
            # データサマリー生成
            data_summary_list = self._generate_data_summary_list(report_data)
            
            # 可視化コンテンツ生成
            visualization_content = self._generate_visualization_content(report_data)
            
            # データテーブル生成
            data_tables_content = self._generate_data_tables_content(report_data, level)
            
            # 詳細分析コンテンツ生成
            detailed_analysis_content = self._generate_detailed_analysis_content(report_data, level)
            
            return template.substitute(
                report_id=metadata.get('report_id', 'Unknown'),
                generation_timestamp=metadata.get('generation_timestamp', 'Unknown'),
                report_type=metadata.get('report_type', 'Unknown'),
                detail_level=metadata.get('level', 'Unknown'),
                data_summary_list=data_summary_list,
                visualization_content=visualization_content,
                data_tables_content=data_tables_content,
                detailed_analysis_content=detailed_analysis_content
            )
            
        except Exception as e:
            self.logger.error(f"包括的コンテンツ生成エラー: {e}")
            return f"<div class='alert alert-danger'>包括的コンテンツ生成エラー: {e}</div>"
    
    def _generate_comparison_content(
        self,
        report_data: Dict[str, Any],
        template: Template,
        level: str
    ) -> str:
        """比較レポートコンテンツ生成"""
        try:
            custom_params = report_data.get('custom_params', {})
            comparison_items = custom_params.get('comparison_items', [])
            comparison_type = custom_params.get('comparison_type', 'Unknown')
            
            # 比較サマリー生成
            comparison_summary = self._generate_comparison_summary(custom_params)
            
            # 比較可視化コンテンツ生成
            comparison_visualization_content = self._generate_comparison_visualization_content(report_data)
            
            # 比較詳細コンテンツ生成
            comparison_details_content = self._generate_comparison_details_content(custom_params, level)
            
            return template.substitute(
                comparison_type=comparison_type,
                comparison_items_count=len(comparison_items),
                comparison_summary=comparison_summary,
                comparison_visualization_content=comparison_visualization_content,
                comparison_details_content=comparison_details_content
            )
            
        except Exception as e:
            self.logger.error(f"比較コンテンツ生成エラー: {e}")
            return f"<div class='alert alert-danger'>比較コンテンツ生成エラー: {e}</div>"
    
    def _generate_performance_content(
        self,
        report_data: Dict[str, Any],
        template: Template,
        level: str
    ) -> str:
        """パフォーマンスレポートコンテンツ生成"""
        try:
            # パフォーマンスサマリー生成
            performance_summary = self._generate_performance_summary(report_data)
            
            # パフォーマンスメトリクス生成
            performance_metrics_content = self._generate_performance_metrics_content(report_data, level)
            
            return template.substitute(
                performance_summary=performance_summary,
                performance_metrics_content=performance_metrics_content
            )
            
        except Exception as e:
            self.logger.error(f"パフォーマンスコンテンツ生成エラー: {e}")
            return f"<div class='alert alert-danger'>パフォーマンスコンテンツ生成エラー: {e}</div>"
    
    def _generate_risk_content(
        self,
        report_data: Dict[str, Any],
        template: Template,
        level: str
    ) -> str:
        """リスクレポートコンテンツ生成"""
        try:
            # リスクサマリー生成
            risk_summary = self._generate_risk_summary(report_data)
            
            # リスクメトリクス生成
            risk_metrics_content = self._generate_risk_metrics_content(report_data, level)
            
            return template.substitute(
                risk_summary=risk_summary,
                risk_metrics_content=risk_metrics_content
            )
            
        except Exception as e:
            self.logger.error(f"リスクコンテンツ生成エラー: {e}")
            return f"<div class='alert alert-danger'>リスクコンテンツ生成エラー: {e}</div>"
    
    def _generate_data_summary_list(self, report_data: Dict[str, Any]) -> str:
        """データサマリーリスト生成"""
        try:
            summary_stats = report_data.get('data', {}).get('summary_statistics', {})
            data_overview = summary_stats.get('data_overview', {})
            
            summary_html = ""
            for key, value in data_overview.items():
                if isinstance(value, (int, float)):
                    display_key = key.replace('total_', '').replace('_', ' ').title()
                    summary_html += f"<li><strong>{display_key}:</strong> {value}</li>"
            
            return summary_html if summary_html else "<li>データサマリーなし</li>"
            
        except Exception as e:
            self.logger.error(f"データサマリーリスト生成エラー: {e}")
            return "<li>データサマリー生成エラー</li>"
    
    def _generate_visualization_content(self, report_data: Dict[str, Any]) -> str:
        """可視化コンテンツ生成"""
        try:
            visualizations = report_data.get('visualizations', {})
            html_snippets = visualizations.get('html_snippets', {})
            
            if 'chart_grid' in html_snippets:
                return html_snippets['chart_grid']
            else:
                # 個別チャートを統合
                content = '<div class="chart-grid">'
                for chart_id, chart_html in html_snippets.items():
                    if chart_id != 'init_script':
                        content += chart_html
                content += '</div>'
                return content
                
        except Exception as e:
            self.logger.error(f"可視化コンテンツ生成エラー: {e}")
            return "<div class='alert alert-warning'>可視化データなし</div>"
    
    def _generate_data_tables_content(self, report_data: Dict[str, Any], level: str) -> str:
        """データテーブルコンテンツ生成"""
        try:
            data = report_data.get('data', {})
            tables_html = ""
            
            # DSSMSデータテーブル
            if 'dssms_data' in data and data['dssms_data']:
                tables_html += self._create_data_table('DSSMS データ', data['dssms_data'], level)
            
            # 戦略データテーブル
            if 'strategy_data' in data and data['strategy_data']:
                tables_html += self._create_data_table('戦略データ', data['strategy_data'], level)
            
            # パフォーマンスデータテーブル
            if 'performance_data' in data and data['performance_data']:
                tables_html += self._create_data_table('パフォーマンスデータ', data['performance_data'], level)
            
            return tables_html if tables_html else "<div class='alert alert-info'>表示可能なテーブルデータがありません</div>"
            
        except Exception as e:
            self.logger.error(f"データテーブル生成エラー: {e}")
            return "<div class='alert alert-danger'>データテーブル生成エラー</div>"
    
    def _create_data_table(self, title: str, data: Dict[str, Any], level: str) -> str:
        """データテーブル作成"""
        try:
            table_html = f"""
            <div class="table-section mb-4">
                <h4>{title}</h4>
                <div class="table-responsive">
                    <table class="table table-striped table-bordered">
                        <thead class="table-dark">
                            <tr>
                                <th>項目</th>
                                <th>値/詳細</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for key, value in data.items():
                if level == "summary":
                    # サマリーレベル：基本情報のみ
                    if isinstance(value, dict):
                        summary_info = f"辞書項目数: {len(value)}"
                    else:
                        summary_info = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    table_html += f"<tr><td>{key}</td><td>{summary_info}</td></tr>"
                    
                elif level == "detailed":
                    # 詳細レベル：構造化情報
                    if isinstance(value, dict):
                        detail_info = "<ul>"
                        for sub_key, sub_value in list(value.items())[:5]:  # 最初の5項目
                            detail_info += f"<li><strong>{sub_key}:</strong> {str(sub_value)[:50]}{'...' if len(str(sub_value)) > 50 else ''}</li>"
                        if len(value) > 5:
                            detail_info += f"<li>... および {len(value) - 5} 項目</li>"
                        detail_info += "</ul>"
                        table_html += f"<tr><td>{key}</td><td>{detail_info}</td></tr>"
                    else:
                        table_html += f"<tr><td>{key}</td><td>{str(value)[:200]}{'...' if len(str(value)) > 200 else ''}</td></tr>"
                        
                else:  # comprehensive
                    # 包括的レベル：全情報（制限付き）
                    if isinstance(value, dict):
                        detail_info = "<details><summary>展開</summary><pre>" + json.dumps(value, indent=2, ensure_ascii=False, default=str)[:1000] + "</pre></details>"
                        table_html += f"<tr><td>{key}</td><td>{detail_info}</td></tr>"
                    else:
                        table_html += f"<tr><td>{key}</td><td><pre>{str(value)[:500]}{'...' if len(str(value)) > 500 else ''}</pre></td></tr>"
            
            table_html += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
            
            return table_html
            
        except Exception as e:
            self.logger.error(f"データテーブル作成エラー: {e}")
            return f"<div class='alert alert-danger'>テーブル作成エラー: {title}</div>"
    
    def _generate_detailed_analysis_content(self, report_data: Dict[str, Any], level: str) -> str:
        """詳細分析コンテンツ生成"""
        try:
            analysis_html = ""
            
            # レベル別分析コンテンツ
            if level == "summary":
                analysis_html = "<p>サマリーレベルでは詳細分析は省略されます。</p>"
                
            elif level == "detailed":
                # 統計サマリー
                summary_stats = report_data.get('data', {}).get('summary_statistics', {})
                if summary_stats:
                    analysis_html += "<h5>統計サマリー</h5>"
                    analysis_html += "<pre>" + json.dumps(summary_stats, indent=2, ensure_ascii=False, default=str) + "</pre>"
                
            else:  # comprehensive
                # 包括的分析
                data = report_data.get('data', {})
                
                # メタデータ分析
                metadata = report_data.get('metadata', {})
                if metadata:
                    analysis_html += "<h5>メタデータ分析</h5>"
                    analysis_html += "<pre>" + json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "</pre>"
                
                # データ品質分析
                analysis_html += "<h5>データ品質分析</h5>"
                quality_info = self._analyze_data_quality(data)
                analysis_html += f"<div class='alert alert-info'>{quality_info}</div>"
            
            return analysis_html if analysis_html else "<div class='alert alert-warning'>詳細分析データなし</div>"
            
        except Exception as e:
            self.logger.error(f"詳細分析コンテンツ生成エラー: {e}")
            return "<div class='alert alert-danger'>詳細分析生成エラー</div>"
    
    def _analyze_data_quality(self, data: Dict[str, Any]) -> str:
        """データ品質分析"""
        try:
            quality_info = []
            
            # データカテゴリ別品質チェック
            for category, category_data in data.items():
                if isinstance(category_data, dict):
                    item_count = len(category_data)
                    quality_info.append(f"{category}: {item_count} 項目")
                else:
                    quality_info.append(f"{category}: データ形式 {type(category_data).__name__}")
            
            return "<br>".join(quality_info)
            
        except Exception as e:
            return f"品質分析エラー: {e}"
    
    def _generate_comparison_summary(self, custom_params: Dict[str, Any]) -> str:
        """比較サマリー生成"""
        try:
            comparison_data = custom_params.get('comparison_data', {})
            comparison_type = custom_params.get('comparison_type', 'Unknown')
            
            summary = f"<p>比較タイプ: {comparison_type}</p>"
            
            if comparison_data:
                summary += "<ul>"
                for item_name, item_data in comparison_data.get('data', {}).items():
                    summary += f"<li>{item_name}: {type(item_data).__name__}</li>"
                summary += "</ul>"
            
            return summary
            
        except Exception as e:
            return f"比較サマリー生成エラー: {e}"
    
    def _generate_comparison_visualization_content(self, report_data: Dict[str, Any]) -> str:
        """比較可視化コンテンツ生成"""
        try:
            custom_params = report_data.get('custom_params', {})
            comparison_visualizations = custom_params.get('comparison_visualizations', {})
            
            if 'html_snippets' in comparison_visualizations:
                return comparison_visualizations['html_snippets'].get('chart_grid', '')
            
            return "<div class='alert alert-warning'>比較可視化データなし</div>"
            
        except Exception as e:
            return f"<div class='alert alert-danger'>比較可視化生成エラー: {e}</div>"
    
    def _generate_comparison_details_content(self, custom_params: Dict[str, Any], level: str) -> str:
        """比較詳細コンテンツ生成"""
        try:
            comparison_items = custom_params.get('comparison_items', [])
            details_html = ""
            
            for i, item in enumerate(comparison_items):
                details_html += f"<h6>比較項目 {i+1}</h6>"
                details_html += "<pre>" + json.dumps(item, indent=2, ensure_ascii=False, default=str) + "</pre>"
            
            return details_html if details_html else "<div class='alert alert-info'>比較詳細なし</div>"
            
        except Exception as e:
            return f"<div class='alert alert-danger'>比較詳細生成エラー: {e}</div>"
    
    def _generate_performance_summary(self, report_data: Dict[str, Any]) -> str:
        """パフォーマンスサマリー生成"""
        try:
            performance_data = report_data.get('data', {}).get('performance_data', {})
            
            if performance_data:
                summary = f"<p>パフォーマンスファイル数: {len(performance_data)}</p>"
                return summary
            
            return "<div class='alert alert-warning'>パフォーマンスデータなし</div>"
            
        except Exception as e:
            return f"<div class='alert alert-danger'>パフォーマンスサマリー生成エラー: {e}</div>"
    
    def _generate_performance_metrics_content(self, report_data: Dict[str, Any], level: str) -> str:
        """パフォーマンスメトリクスコンテンツ生成"""
        try:
            performance_data = report_data.get('data', {}).get('performance_data', {})
            
            if not performance_data:
                return "<div class='alert alert-warning'>パフォーマンスメトリクスなし</div>"
            
            return self._create_data_table('パフォーマンスメトリクス', performance_data, level)
            
        except Exception as e:
            return f"<div class='alert alert-danger'>パフォーマンスメトリクス生成エラー: {e}</div>"
    
    def _generate_risk_summary(self, report_data: Dict[str, Any]) -> str:
        """リスクサマリー生成"""
        try:
            risk_data = report_data.get('data', {}).get('risk_data', {})
            
            if risk_data:
                summary = f"<p>リスク設定項目数: {len(risk_data)}</p>"
                return summary
            
            return "<div class='alert alert-warning'>リスクデータなし</div>"
            
        except Exception as e:
            return f"<div class='alert alert-danger'>リスクサマリー生成エラー: {e}</div>"
    
    def _generate_risk_metrics_content(self, report_data: Dict[str, Any], level: str) -> str:
        """リスクメトリクスコンテンツ生成"""
        try:
            risk_data = report_data.get('data', {}).get('risk_data', {})
            
            if not risk_data:
                return "<div class='alert alert-warning'>リスクメトリクスなし</div>"
            
            return self._create_data_table('リスクメトリクス', risk_data, level)
            
        except Exception as e:
            return f"<div class='alert alert-danger'>リスクメトリクス生成エラー: {e}</div>"
    
    def _generate_custom_css(self, level: str) -> str:
        """カスタムCSS生成"""
        css = """
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
        }
        
        .card {
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border: 1px solid rgba(0, 0, 0, 0.125);
            margin-bottom: 1rem;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .table-section {
            margin-bottom: 2rem;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 0.75rem;
            font-size: 0.875rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        details {
            margin: 0.5rem 0;
        }
        
        summary {
            cursor: pointer;
            padding: 0.5rem;
            background-color: #e9ecef;
            border-radius: 0.25rem;
        }
        
        @media (max-width: 768px) {
            .chart-grid {
                grid-template-columns: 1fr;
            }
        }
        """
        
        return css
    
    def _generate_custom_javascript(self, report_data: Dict[str, Any]) -> str:
        """カスタムJavaScript生成"""
        js = """
        // レポート操作機能
        document.addEventListener('DOMContentLoaded', function() {
            // テーブルソート機能
            const tables = document.querySelectorAll('.table');
            tables.forEach(table => {
                // 基本的なテーブル機能を追加
                table.style.cursor = 'default';
            });
            
            // チャートレスポンシブ対応
            window.addEventListener('resize', function() {
                Chart.helpers.each(Chart.instances, function(chart) {
                    chart.resize();
                });
            });
        });
        """
        
        return js
    
    def _generate_chart_javascript(self, report_data: Dict[str, Any]) -> str:
        """Chart.js初期化JavaScript生成"""
        try:
            visualizations = report_data.get('visualizations', {})
            javascript_code = visualizations.get('javascript_code', [])
            
            if javascript_code:
                init_js = "document.addEventListener('DOMContentLoaded', function() {\n"
                init_js += "\n".join(javascript_code)
                init_js += "\n});"
                return init_js
            
            return ""
            
        except Exception as e:
            self.logger.error(f"チャートJavaScript生成エラー: {e}")
            return ""
    
    def _generate_error_page(self, error_message: str) -> str:
        """エラーページ生成"""
        return f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>レポート生成エラー</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h1>レポート生成エラー</h1>
                    <p>レポートの生成中にエラーが発生しました:</p>
                    <pre>{error_message}</pre>
                </div>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    # デモ実行
    manager = ReportTemplateManager()
    
    # サンプルレポートデータ
    sample_data = {
        'metadata': {
            'report_id': 'test_report_001',
            'generation_timestamp': datetime.now(),
            'report_type': 'comprehensive',
            'level': 'detailed'
        },
        'data': {
            'summary_statistics': {
                'data_overview': {
                    'total_dssms_files': 3,
                    'total_strategies': 2,
                    'total_performance_files': 1
                }
            }
        },
        'visualizations': {
            'html_snippets': {
                'chart_grid': '<div>サンプルチャート</div>'
            },
            'javascript_code': ['console.log("Chart initialized");']
        }
    }
    
    # HTMLレポート生成
    html_report = manager.generate_html_report(sample_data)
    print(f"HTMLレポート生成完了: {len(html_report)} 文字")
    print("サンプル出力（最初の200文字）:")
    print(html_report[:200] + "...")
