"""
可視化生成モジュール

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムの可視化コンポーネント

機能:
- インタラクティブ可視化（Chart.js）
- Bootstrap5ベースのレスポンシブデザイン
- 複数チャートタイプ対応
- レベル別可視化生成
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import base64
from io import BytesIO

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class VisualizationGenerator:
    """可視化生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.logger.info("=== VisualizationGenerator 初期化開始 ===")
        
        # チャート設定
        self.chart_config = {
            'default_colors': [
                '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#e91e63'
            ],
            'chart_types': {
                'line': 'Line Chart',
                'bar': 'Bar Chart',
                'pie': 'Pie Chart',
                'doughnut': 'Doughnut Chart',
                'scatter': 'Scatter Plot',
                'area': 'Area Chart'
            },
            'responsive': True,
            'maintain_aspect_ratio': False,
            'animation_duration': 1000
        }
        
        # Chart.jsテンプレート
        self.chartjs_templates = self._initialize_chartjs_templates()
        
        self.logger.info("VisualizationGenerator 初期化完了")
    
    def _initialize_chartjs_templates(self) -> Dict[str, str]:
        """Chart.jsテンプレート初期化"""
        templates = {}
        
        # 基本線グラフテンプレート
        templates['line'] = """
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: {datasets}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '{y_label}'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: '{x_label}'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                animation: {{
                    duration: {animation_duration}
                }}
            }}
        }});
        """
        
        # 棒グラフテンプレート
        templates['bar'] = """
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: {datasets}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '{y_label}'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: '{x_label}'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                animation: {{
                    duration: {animation_duration}
                }}
            }}
        }});
        """
        
        # 円グラフテンプレート
        templates['pie'] = """
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'pie',
            data: {{
                labels: {labels},
                datasets: [{{
                    data: {data},
                    backgroundColor: {colors},
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }},
                    legend: {{
                        display: true,
                        position: 'right'
                    }}
                }},
                animation: {{
                    duration: {animation_duration}
                }}
            }}
        }});
        """
        
        return templates
    
    def generate_visualizations(
        self,
        data: Dict[str, Any],
        report_type: str = "comprehensive",
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        可視化生成メイン処理
        
        Args:
            data: 集約されたデータ
            report_type: レポートタイプ
            level: 詳細レベル
            
        Returns:
            生成された可視化データ
        """
        try:
            self.logger.info(f"可視化生成開始 - タイプ: {report_type}, レベル: {level}")
            
            visualizations = {
                'metadata': {
                    'generation_timestamp': datetime.now(),
                    'report_type': report_type,
                    'level': level,
                    'total_charts': 0
                },
                'charts': {},
                'html_snippets': {},
                'javascript_code': [],
                'css_code': []
            }
            
            # DSSMSデータ可視化
            if data.get('dssms_data'):
                self.logger.info("DSSMSデータ可視化生成")
                dssms_charts = self._generate_dssms_visualizations(
                    data['dssms_data'], level
                )
                visualizations['charts'].update(dssms_charts)
            
            # 戦略データ可視化
            if data.get('strategy_data'):
                self.logger.info("戦略データ可視化生成")
                strategy_charts = self._generate_strategy_visualizations(
                    data['strategy_data'], level
                )
                visualizations['charts'].update(strategy_charts)
            
            # パフォーマンスデータ可視化
            if data.get('performance_data'):
                self.logger.info("パフォーマンスデータ可視化生成")
                performance_charts = self._generate_performance_visualizations(
                    data['performance_data'], level
                )
                visualizations['charts'].update(performance_charts)
            
            # サマリー統計可視化
            if data.get('summary_statistics'):
                self.logger.info("サマリー統計可視化生成")
                summary_charts = self._generate_summary_visualizations(
                    data['summary_statistics'], level
                )
                visualizations['charts'].update(summary_charts)
            
            # JavaScript/CSS生成
            visualizations['javascript_code'] = self._generate_javascript_code(
                visualizations['charts']
            )
            visualizations['css_code'] = self._generate_css_code()
            
            # HTMLスニペット生成
            visualizations['html_snippets'] = self._generate_html_snippets(
                visualizations['charts']
            )
            
            # メタデータ更新
            visualizations['metadata']['total_charts'] = len(visualizations['charts'])
            
            self.logger.info(f"可視化生成完了: {visualizations['metadata']['total_charts']} チャート")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"可視化生成エラー: {e}")
            return {}
    
    def _generate_dssms_visualizations(
        self,
        dssms_data: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """DSSMSデータ可視化生成"""
        try:
            charts = {}
            
            # ファイル数とレコード数の概要
            file_counts = []
            record_counts = []
            file_names = []
            
            for file_name, file_data in dssms_data.items():
                file_names.append(file_name)
                
                if isinstance(file_data, dict):
                    if 'total_records' in file_data:
                        record_counts.append(file_data['total_records'])
                    elif 'full_data' in file_data and hasattr(file_data['full_data'], '__len__'):
                        record_counts.append(len(file_data['full_data']))
                    else:
                        record_counts.append(0)
                else:
                    record_counts.append(0)
            
            # DSSMSファイルレコード数チャート
            if file_names and record_counts:
                charts['dssms_records_chart'] = {
                    'type': 'bar',
                    'title': 'DSSMSファイル別レコード数',
                    'data': {
                        'labels': file_names,
                        'datasets': [{
                            'label': 'レコード数',
                            'data': record_counts,
                            'backgroundColor': self.chart_config['default_colors'][0],
                            'borderColor': self.chart_config['default_colors'][0],
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'x_label': 'ファイル名',
                        'y_label': 'レコード数'
                    }
                }
            
            # 詳細レベル以上の場合、数値列分析チャート生成
            if level in ["detailed", "comprehensive"]:
                for file_name, file_data in dssms_data.items():
                    if isinstance(file_data, dict) and 'full_data' in file_data:
                        df = file_data['full_data']
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # 数値列の時系列チャート
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0 and 'Date' in df.columns:
                                charts[f'dssms_{file_name}_timeseries'] = self._create_timeseries_chart(
                                    df, numeric_cols[:3], f'{file_name} 時系列データ'
                                )
            
            return charts
            
        except Exception as e:
            self.logger.error(f"DSSMSデータ可視化エラー: {e}")
            return {}
    
    def _generate_strategy_visualizations(
        self,
        strategy_data: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """戦略データ可視化生成"""
        try:
            charts = {}
            
            # 戦略数と行数の概要
            strategy_names = []
            lines_counts = []
            file_sizes = []
            
            for strategy_name, strategy_info in strategy_data.items():
                strategy_names.append(strategy_name)
                
                if isinstance(strategy_info, dict):
                    lines_counts.append(strategy_info.get('lines_count', 0))
                    file_sizes.append(strategy_info.get('file_size', 0))
                else:
                    lines_counts.append(0)
                    file_sizes.append(0)
            
            # 戦略コード行数チャート
            if strategy_names and lines_counts:
                charts['strategy_lines_chart'] = {
                    'type': 'bar',
                    'title': '戦略別コード行数',
                    'data': {
                        'labels': strategy_names,
                        'datasets': [{
                            'label': 'コード行数',
                            'data': lines_counts,
                            'backgroundColor': self.chart_config['default_colors'][1],
                            'borderColor': self.chart_config['default_colors'][1],
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'x_label': '戦略名',
                        'y_label': '行数'
                    }
                }
            
            # 戦略ファイルサイズ比較（円グラフ）
            if strategy_names and file_sizes and sum(file_sizes) > 0:
                charts['strategy_size_pie'] = {
                    'type': 'pie',
                    'title': '戦略ファイルサイズ分布',
                    'data': {
                        'labels': strategy_names,
                        'data': file_sizes,
                        'colors': self.chart_config['default_colors'][:len(strategy_names)]
                    }
                }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"戦略データ可視化エラー: {e}")
            return {}
    
    def _generate_performance_visualizations(
        self,
        performance_data: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """パフォーマンスデータ可視化生成"""
        try:
            charts = {}
            
            # パフォーマンスファイル概要
            file_names = []
            record_counts = []
            
            for file_name, file_data in performance_data.items():
                file_names.append(file_name)
                
                if isinstance(file_data, dict):
                    if 'record_count' in file_data:
                        record_counts.append(file_data['record_count'])
                    elif 'data' in file_data and hasattr(file_data['data'], '__len__'):
                        record_counts.append(len(file_data['data']))
                    else:
                        record_counts.append(0)
                else:
                    record_counts.append(0)
            
            # パフォーマンスファイルレコード数チャート
            if file_names and record_counts:
                charts['performance_records_chart'] = {
                    'type': 'bar',
                    'title': 'パフォーマンスファイル別レコード数',
                    'data': {
                        'labels': file_names,
                        'datasets': [{
                            'label': 'レコード数',
                            'data': record_counts,
                            'backgroundColor': self.chart_config['default_colors'][2],
                            'borderColor': self.chart_config['default_colors'][2],
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'x_label': 'ファイル名',
                        'y_label': 'レコード数'
                    }
                }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"パフォーマンスデータ可視化エラー: {e}")
            return {}
    
    def _generate_summary_visualizations(
        self,
        summary_statistics: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """サマリー統計可視化生成"""
        try:
            charts = {}
            
            # データ概要
            if 'data_overview' in summary_statistics:
                overview = summary_statistics['data_overview']
                
                # データソース分布（円グラフ）
                labels = []
                values = []
                
                for key, value in overview.items():
                    if isinstance(value, (int, float)) and value > 0:
                        labels.append(key.replace('total_', '').replace('_', ' ').title())
                        values.append(value)
                
                if labels and values:
                    charts['data_overview_pie'] = {
                        'type': 'pie',
                        'title': 'データソース分布',
                        'data': {
                            'labels': labels,
                            'data': values,
                            'colors': self.chart_config['default_colors'][:len(labels)]
                        }
                    }
            
            # 戦略統計
            if 'strategy_statistics' in summary_statistics:
                strategy_stats = summary_statistics['strategy_statistics']
                
                if 'total_lines_of_code' in strategy_stats and strategy_stats['total_lines_of_code'] > 0:
                    charts['strategy_metrics_bar'] = {
                        'type': 'bar',
                        'title': '戦略統計',
                        'data': {
                            'labels': ['戦略数', 'コード行数'],
                            'datasets': [{
                                'label': '値',
                                'data': [
                                    strategy_stats.get('total_strategies', 0),
                                    strategy_stats.get('total_lines_of_code', 0) / 100  # スケール調整
                                ],
                                'backgroundColor': [
                                    self.chart_config['default_colors'][3],
                                    self.chart_config['default_colors'][4]
                                ],
                                'borderWidth': 1
                            }]
                        },
                        'options': {
                            'x_label': 'メトリクス',
                            'y_label': '値'
                        }
                    }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"サマリー統計可視化エラー: {e}")
            return {}
    
    def _create_timeseries_chart(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str
    ) -> Dict[str, Any]:
        """時系列チャート作成"""
        try:
            # 日付列をインデックスに設定
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # データセット構築
            datasets = []
            for i, col in enumerate(columns):
                if col in df.columns:
                    datasets.append({
                        'label': col,
                        'data': df[col].dropna().tolist(),
                        'borderColor': self.chart_config['default_colors'][i % len(self.chart_config['default_colors'])],
                        'backgroundColor': self.chart_config['default_colors'][i % len(self.chart_config['default_colors'])] + '20',
                        'fill': False,
                        'tension': 0.1
                    })
            
            # ラベル（日付）
            labels = df.index.strftime('%Y-%m-%d').tolist() if hasattr(df.index, 'strftime') else list(range(len(df)))
            
            return {
                'type': 'line',
                'title': title,
                'data': {
                    'labels': labels,
                    'datasets': datasets
                },
                'options': {
                    'x_label': '日付',
                    'y_label': '値'
                }
            }
            
        except Exception as e:
            self.logger.error(f"時系列チャート作成エラー: {e}")
            return {}
    
    def _generate_javascript_code(self, charts: Dict[str, Any]) -> List[str]:
        """JavaScript コード生成"""
        try:
            js_code = []
            
            for chart_id, chart_config in charts.items():
                try:
                    chart_type = chart_config.get('type', 'line')
                    template = self.chartjs_templates.get(chart_type, self.chartjs_templates['line'])
                    
                    # テンプレート変数置換
                    js_chart = template.format(
                        chart_id=chart_id,
                        title=chart_config.get('title', ''),
                        labels=json.dumps(chart_config['data'].get('labels', [])),
                        datasets=json.dumps(chart_config['data'].get('datasets', [])),
                        data=json.dumps(chart_config['data'].get('data', [])) if chart_type == 'pie' else '[]',
                        colors=json.dumps(chart_config['data'].get('colors', [])) if chart_type == 'pie' else '[]',
                        x_label=chart_config.get('options', {}).get('x_label', 'X軸'),
                        y_label=chart_config.get('options', {}).get('y_label', 'Y軸'),
                        animation_duration=self.chart_config['animation_duration']
                    )
                    
                    js_code.append(js_chart)
                    
                except Exception as e:
                    self.logger.warning(f"チャートJavaScript生成エラー {chart_id}: {e}")
                    continue
            
            return js_code
            
        except Exception as e:
            self.logger.error(f"JavaScript生成エラー: {e}")
            return []
    
    def _generate_css_code(self) -> List[str]:
        """CSS コード生成"""
        css_code = [
            """
            .chart-container {
                position: relative;
                height: 400px;
                width: 100%;
                margin-bottom: 30px;
                padding: 20px;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .chart-title {
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #2c3e50;
            }
            
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .small-chart {
                height: 300px;
            }
            
            .large-chart {
                height: 500px;
            }
            
            @media (max-width: 768px) {
                .chart-grid {
                    grid-template-columns: 1fr;
                }
                
                .chart-container {
                    height: 300px;
                    padding: 15px;
                }
            }
            """
        ]
        
        return css_code
    
    def _generate_html_snippets(self, charts: Dict[str, Any]) -> Dict[str, str]:
        """HTML スニペット生成"""
        try:
            html_snippets = {}
            
            # 個別チャートHTML
            for chart_id, chart_config in charts.items():
                html_snippets[chart_id] = f"""
                <div class="chart-container">
                    <div class="chart-title">{chart_config.get('title', '')}</div>
                    <canvas id="{chart_id}"></canvas>
                </div>
                """
            
            # グリッドレイアウトHTML
            chart_grid_html = '<div class="chart-grid">'
            for chart_id in charts.keys():
                chart_grid_html += html_snippets[chart_id]
            chart_grid_html += '</div>'
            
            html_snippets['chart_grid'] = chart_grid_html
            
            # チャート初期化スクリプト
            init_script = """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Chart.js初期化
                Chart.defaults.font.family = 'Arial, sans-serif';
                Chart.defaults.font.size = 12;
                Chart.defaults.color = '#2c3e50';
            });
            </script>
            """
            
            html_snippets['init_script'] = init_script
            
            return html_snippets
            
        except Exception as e:
            self.logger.error(f"HTMLスニペット生成エラー: {e}")
            return {}
    
    def generate_comparison_visualizations(
        self,
        comparison_data: Dict[str, Any],
        comparison_type: str = "strategies",
        level: str = "detailed"
    ) -> Dict[str, Any]:
        """比較可視化生成"""
        try:
            self.logger.info(f"比較可視化生成開始: {comparison_type}")
            
            charts = {}
            
            if comparison_type == "strategies":
                # 戦略比較チャート
                charts.update(self._generate_strategy_comparison_charts(comparison_data))
                
            elif comparison_type == "periods":
                # 期間比較チャート
                charts.update(self._generate_period_comparison_charts(comparison_data))
                
            elif comparison_type == "configurations":
                # 設定比較チャート
                charts.update(self._generate_config_comparison_charts(comparison_data))
            
            return {
                'charts': charts,
                'javascript_code': self._generate_javascript_code(charts),
                'css_code': self._generate_css_code(),
                'html_snippets': self._generate_html_snippets(charts)
            }
            
        except Exception as e:
            self.logger.error(f"比較可視化生成エラー: {e}")
            return {}
    
    def _generate_strategy_comparison_charts(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """戦略比較チャート生成"""
        try:
            charts = {}
            
            # 戦略データから比較メトリクス抽出
            strategy_names = []
            lines_counts = []
            
            for strategy_name, strategy_data in comparison_data.get('data', {}).items():
                strategy_names.append(strategy_name)
                lines_counts.append(strategy_data.get('lines_count', 0))
            
            # 戦略比較棒グラフ
            if strategy_names and lines_counts:
                charts['strategy_comparison_bar'] = {
                    'type': 'bar',
                    'title': '戦略比較 - コード行数',
                    'data': {
                        'labels': strategy_names,
                        'datasets': [{
                            'label': 'コード行数',
                            'data': lines_counts,
                            'backgroundColor': self.chart_config['default_colors'][:len(strategy_names)],
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'x_label': '戦略名',
                        'y_label': 'コード行数'
                    }
                }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"戦略比較チャート生成エラー: {e}")
            return {}
    
    def _generate_period_comparison_charts(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """期間比較チャート生成"""
        try:
            charts = {}
            
            # 期間データから比較メトリクス抽出
            period_names = []
            data_counts = []
            
            for period_name, period_data in comparison_data.get('data', {}).items():
                period_names.append(period_name)
                # データ数をカウント
                total_data = 0
                if isinstance(period_data, dict):
                    for category_data in period_data.values():
                        if isinstance(category_data, dict):
                            total_data += len(category_data)
                data_counts.append(total_data)
            
            # 期間比較チャート
            if period_names and data_counts:
                charts['period_comparison_line'] = {
                    'type': 'line',
                    'title': '期間比較 - データ量',
                    'data': {
                        'labels': period_names,
                        'datasets': [{
                            'label': 'データ量',
                            'data': data_counts,
                            'borderColor': self.chart_config['default_colors'][0],
                            'backgroundColor': self.chart_config['default_colors'][0] + '20',
                            'tension': 0.1
                        }]
                    },
                    'options': {
                        'x_label': '期間',
                        'y_label': 'データ量'
                    }
                }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"期間比較チャート生成エラー: {e}")
            return {}
    
    def _generate_config_comparison_charts(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """設定比較チャート生成"""
        try:
            charts = {}
            
            # 設定データから比較情報抽出
            config_names = list(comparison_data.get('data', {}).keys())
            
            if config_names:
                # 設定項目数比較
                config_counts = []
                for config_name, config_data in comparison_data.get('data', {}).items():
                    config_count = 0
                    if isinstance(config_data, dict) and 'config' in config_data:
                        config_count = len(config_data['config']) if isinstance(config_data['config'], dict) else 1
                    config_counts.append(config_count)
                
                charts['config_comparison_bar'] = {
                    'type': 'bar',
                    'title': '設定比較 - 設定項目数',
                    'data': {
                        'labels': config_names,
                        'datasets': [{
                            'label': '設定項目数',
                            'data': config_counts,
                            'backgroundColor': self.chart_config['default_colors'][2],
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'x_label': '設定名',
                        'y_label': '項目数'
                    }
                }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"設定比較チャート生成エラー: {e}")
            return {}


if __name__ == "__main__":
    # デモ実行
    generator = VisualizationGenerator()
    
    # サンプルデータ
    sample_data = {
        'dssms_data': {
            'sample1': {'total_records': 1000},
            'sample2': {'total_records': 1500}
        },
        'strategy_data': {
            'strategy1': {'lines_count': 200, 'file_size': 5000},
            'strategy2': {'lines_count': 300, 'file_size': 7500}
        }
    }
    
    # 可視化生成
    result = generator.generate_visualizations(sample_data)
    print(f"可視化生成結果: {result['metadata']['total_charts']} チャート")
    print(f"JavaScript行数: {len(result['javascript_code'])}")
    print(f"HTMLスニペット数: {len(result['html_snippets'])}")
