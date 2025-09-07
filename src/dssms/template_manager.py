"""
DSSMS Template Manager
Phase 2.3 Task 2.3.2: 多形式出力エンジン構築

Purpose:
  - テンプレートベース出力管理
  - 動的コンテンツ生成
  - フォーマット特化テンプレート
  - カスタマイゼーション対応

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - unified_output_engine.py連携
  - 各出力形式のテンプレート管理
  - 動的データ挿入
"""

import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.output_data_model import UnifiedOutputModel


class TemplateManager:
    """テンプレート管理システム"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            template_dir: テンプレートディレクトリパス
        """
        self.logger = setup_logger(__name__)
        
        # テンプレートディレクトリの設定
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent / "templates"
        
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # テンプレートキャッシュ
        self.template_cache: Dict[str, str] = {}
        
        # デフォルトテンプレートの初期化
        self._init_default_templates()
        
        self.logger.info(f"テンプレートマネージャーが初期化されました: {self.template_dir}")
    
    def _init_default_templates(self):
        """デフォルトテンプレートの初期化"""
        try:
            # Excel用テンプレート設定の作成
            self._create_excel_template_config()
            
            # HTML用テンプレートの作成
            self._create_html_template()
            
            # テキスト用テンプレートの作成
            self._create_text_template()
            
            # JSON用スキーマテンプレートの作成
            self._create_json_schema_template()
            
            self.logger.info("デフォルトテンプレートが初期化されました")
            
        except Exception as e:
            self.logger.error(f"デフォルトテンプレート初期化中にエラー: {e}")
    
    def _create_excel_template_config(self):
        """Excel用テンプレート設定の作成"""
        excel_config = {
            "name": "DSSMS統一Excelテンプレート",
            "version": "1.0",
            "sheets": {
                "Summary": {
                    "layout": "vertical_metrics",
                    "sections": [
                        {
                            "title": "基本情報",
                            "metrics": [
                                {"label": "銘柄", "field": "metadata.ticker"},
                                {"label": "分析期間", "field": "metadata.period"},
                                {"label": "生成日時", "field": "metadata.generation_timestamp"},
                                {"label": "データソース", "field": "metadata.data_source"}
                            ]
                        },
                        {
                            "title": "パフォーマンス指標",
                            "metrics": [
                                {"label": "総リターン", "field": "performance.total_return", "format": "percentage"},
                                {"label": "総損益", "field": "performance.total_pnl", "format": "currency"},
                                {"label": "勝率", "field": "performance.win_rate", "format": "percentage"},
                                {"label": "総取引数", "field": "performance.total_trades", "format": "integer"},
                                {"label": "勝ち取引数", "field": "performance.winning_trades", "format": "integer"},
                                {"label": "負け取引数", "field": "performance.losing_trades", "format": "integer"},
                                {"label": "平均勝ち", "field": "performance.average_win", "format": "currency"},
                                {"label": "平均負け", "field": "performance.average_loss", "format": "currency"},
                                {"label": "プロフィットファクター", "field": "performance.profit_factor", "format": "decimal"},
                                {"label": "シャープレシオ", "field": "performance.sharpe_ratio", "format": "decimal"},
                                {"label": "最大ドローダウン", "field": "performance.max_drawdown", "format": "percentage"},
                                {"label": "ポートフォリオ価値", "field": "performance.portfolio_value", "format": "currency"}
                            ]
                        }
                    ]
                },
                "Trades": {
                    "layout": "table",
                    "columns": [
                        {"header": "取引ID", "field": "trade_id", "width": 12},
                        {"header": "戦略", "field": "strategy", "width": 15},
                        {"header": "エントリー日", "field": "entry_date", "width": 12, "format": "date"},
                        {"header": "エグジット日", "field": "exit_date", "width": 12, "format": "date"},
                        {"header": "エントリー価格", "field": "entry_price", "width": 12, "format": "currency"},
                        {"header": "エグジット価格", "field": "exit_price", "width": 12, "format": "currency"},
                        {"header": "株数", "field": "shares", "width": 10, "format": "integer"},
                        {"header": "損益", "field": "profit_loss", "width": 12, "format": "currency"},
                        {"header": "損益率", "field": "profit_loss_pct", "width": 10, "format": "percentage"},
                        {"header": "期間（日）", "field": "duration_days", "width": 10, "format": "integer"},
                        {"header": "勝敗", "field": "is_winner", "width": 8, "format": "boolean"}
                    ]
                },
                "DSSMS_Analysis": {
                    "layout": "conditional",
                    "condition": "dssms_metrics_exists",
                    "sections": [
                        {
                            "title": "DSSMS指標",
                            "metrics": [
                                {"label": "戦略切り替え成功率", "field": "dssms_metrics.switch_success_rate", "format": "percentage"},
                                {"label": "戦略切り替え頻度", "field": "dssms_metrics.switch_frequency", "format": "decimal"}
                            ]
                        },
                        {
                            "title": "戦略スコア",
                            "layout": "key_value_table",
                            "data_field": "dssms_metrics.strategy_scores"
                        }
                    ]
                },
                "Quality_Assurance": {
                    "layout": "conditional",
                    "condition": "quality_assurance_exists",
                    "sections": [
                        {
                            "title": "品質保証指標",
                            "metrics": [
                                {"label": "データ品質スコア", "field": "quality_assurance.data_quality_score", "format": "decimal"},
                                {"label": "検証スコア", "field": "quality_assurance.validation_score", "format": "decimal"},
                                {"label": "信頼性スコア", "field": "quality_assurance.reliability_score", "format": "decimal"},
                                {"label": "品質向上適用", "field": "quality_assurance.enhancement_applied", "format": "boolean"}
                            ]
                        }
                    ]
                }
            },
            "styling": {
                "header_font": {"bold": True, "color": "FFFFFF"},
                "header_fill": {"color": "366092"},
                "positive_color": "27AE60",
                "negative_color": "E74C3C",
                "neutral_color": "34495E"
            }
        }
        
        config_path = self.template_dir / "excel_template_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(excel_config, f, indent=2, ensure_ascii=False)
    
    def _create_html_template(self):
        """HTML用テンプレートの作成"""
        html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} - {{ticker}}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }
        .section h2 {
            margin: 0 0 20px 0;
            color: #2c3e50;
            font-size: 1.5em;
            display: flex;
            align-items: center;
        }
        .section h2::before {
            content: attr(data-icon);
            margin-right: 10px;
            font-size: 1.2em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
            font-weight: 500;
        }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #34495e; }
        .table-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th {
            background: #34495e;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .quality-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }
        .badge-excellent { background: #27ae60; }
        .badge-good { background: #f39c12; }
        .badge-poor { background: #e74c3c; }
        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{title}}</h1>
            <div class="subtitle">
                {{ticker}} | {{period}} | {{generation_date}}
            </div>
        </div>
        
        <div class="content">
            <!-- パフォーマンス概要セクション -->
            <div class="section">
                <h2 data-icon="📊">パフォーマンス概要</h2>
                <div class="metrics-grid">
                    {{performance_metrics}}
                </div>
            </div>
            
            <!-- DSSMS分析セクション（条件付き） -->
            {{dssms_section}}
            
            <!-- 品質保証セクション（条件付き） -->
            {{quality_section}}
            
            <!-- 取引履歴セクション -->
            <div class="section">
                <h2 data-icon="📋">取引履歴</h2>
                <div class="table-container">
                    {{trades_table}}
                </div>
            </div>
            
            <!-- システム情報セクション -->
            <div class="section">
                <h2 data-icon="ℹ️">システム情報</h2>
                {{system_info}}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by DSSMS Unified Output Engine v1.0</p>
        </div>
    </div>
</body>
</html>"""
        
        template_path = self.template_dir / "html_report_template.html"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
    
    def _create_text_template(self):
        """テキスト用テンプレートの作成"""
        text_template = """{{border_line}}
{{title}}
{{border_line}}

銘柄: {{ticker}}
分析期間: {{period}}
生成日時: {{generation_date}}
データソース: {{data_source}}
分析タイプ: {{analysis_type}}

{{section_separator}}
パフォーマンス概要
{{section_separator}}
{{performance_summary}}

{{dssms_section}}

{{quality_section}}

{{section_separator}}
取引履歴サマリー
{{section_separator}}
総取引数: {{total_trades}}
勝ち取引数: {{winning_trades}}
負け取引数: {{losing_trades}}
勝率: {{win_rate}}

{{trades_detail_section}}

{{footer_section}}
"""
        
        template_path = self.template_dir / "text_report_template.txt"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(text_template)
    
    def _create_json_schema_template(self):
        """JSON用スキーマテンプレートの作成"""
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "DSSMS統一出力データスキーマ",
            "version": "1.0",
            "type": "object",
            "required": ["metadata", "performance", "trades"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "start_date": {"type": "string", "format": "date-time"},
                        "end_date": {"type": "string", "format": "date-time"},
                        "generation_timestamp": {"type": "string", "format": "date-time"},
                        "data_source": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "version": {"type": "string"}
                    },
                    "required": ["ticker", "start_date", "end_date", "generation_timestamp"]
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "total_return": {"type": "number"},
                        "total_pnl": {"type": "number"},
                        "win_rate": {"type": "number", "minimum": 0, "maximum": 1},
                        "total_trades": {"type": "integer", "minimum": 0},
                        "winning_trades": {"type": "integer", "minimum": 0},
                        "losing_trades": {"type": "integer", "minimum": 0},
                        "average_win": {"type": "number"},
                        "average_loss": {"type": "number"},
                        "profit_factor": {"type": "number"},
                        "sharpe_ratio": {"type": "number"},
                        "max_drawdown": {"type": "number"},
                        "portfolio_value": {"type": "number"},
                        "initial_capital": {"type": "number"}
                    },
                    "required": ["total_return", "total_pnl", "win_rate", "total_trades"]
                },
                "trades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trade_id": {"type": "string"},
                            "strategy": {"type": "string"},
                            "entry_date": {"type": "string", "format": "date-time"},
                            "exit_date": {"type": "string", "format": "date-time"},
                            "entry_price": {"type": "number"},
                            "exit_price": {"type": "number"},
                            "shares": {"type": "integer"},
                            "profit_loss": {"type": "number"},
                            "profit_loss_pct": {"type": "number"},
                            "duration_days": {"type": "integer"},
                            "is_winner": {"type": "boolean"}
                        },
                        "required": ["trade_id", "strategy", "entry_date", "exit_date"]
                    }
                },
                "dssms_metrics": {
                    "type": ["object", "null"],
                    "properties": {
                        "strategy_scores": {"type": "object"},
                        "switch_decisions": {"type": "array"},
                        "ranking_data": {"type": "object"},
                        "market_conditions": {"type": "object"},
                        "switch_success_rate": {"type": "number"},
                        "switch_frequency": {"type": "number"}
                    }
                },
                "quality_assurance": {
                    "type": ["object", "null"],
                    "properties": {
                        "data_quality_score": {"type": "number"},
                        "validation_score": {"type": "number"},
                        "reliability_score": {"type": "number"},
                        "enhancement_applied": {"type": "boolean"},
                        "validation_errors": {"type": "array"},
                        "quality_recommendations": {"type": "array"}
                    }
                }
            }
        }
        
        schema_path = self.template_dir / "json_schema_template.json"
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(json_schema, f, indent=2, ensure_ascii=False)
    
    def render_html_template(self, unified_model: UnifiedOutputModel) -> str:
        """HTMLテンプレートのレンダリング"""
        try:
            template_path = self.template_dir / "html_report_template.html"
            
            if not template_path.exists():
                self.logger.warning("HTMLテンプレートが見つかりません、デフォルトを作成します")
                self._create_html_template()
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # テンプレート変数の置換
            replacements = {
                'title': 'DSSMS統一バックテストレポート',
                'ticker': unified_model.metadata.ticker,
                'period': f"{unified_model.metadata.start_date.strftime('%Y-%m-%d')} ～ {unified_model.metadata.end_date.strftime('%Y-%m-%d')}",
                'generation_date': unified_model.metadata.generation_timestamp.strftime('%Y年%m月%d日 %H時%M分'),
                'performance_metrics': self._render_performance_metrics_html(unified_model),
                'dssms_section': self._render_dssms_section_html(unified_model),
                'quality_section': self._render_quality_section_html(unified_model),
                'trades_table': self._render_trades_table_html(unified_model),
                'system_info': self._render_system_info_html(unified_model)
            }
            
            # 変数置換
            for key, value in replacements.items():
                template = template.replace('{{' + key + '}}', str(value))
            
            return template
            
        except Exception as e:
            self.logger.error(f"HTMLテンプレートレンダリング中にエラー: {e}")
            return self._generate_fallback_html(unified_model)
    
    def render_text_template(self, unified_model: UnifiedOutputModel) -> str:
        """テキストテンプレートのレンダリング"""
        try:
            template_path = self.template_dir / "text_report_template.txt"
            
            if not template_path.exists():
                self.logger.warning("テキストテンプレートが見つかりません、デフォルトを作成します")
                self._create_text_template()
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # テンプレート変数の置換
            replacements = {
                'border_line': '=' * 80,
                'section_separator': '=' * 40,
                'title': 'DSSMS統一バックテストレポート',
                'ticker': unified_model.metadata.ticker,
                'period': f"{unified_model.metadata.start_date.strftime('%Y-%m-%d')} ～ {unified_model.metadata.end_date.strftime('%Y-%m-%d')}",
                'generation_date': unified_model.metadata.generation_timestamp.strftime('%Y年%m月%d日 %H時%M分'),
                'data_source': unified_model.metadata.data_source,
                'analysis_type': unified_model.metadata.analysis_type,
                'performance_summary': self._render_performance_summary_text(unified_model),
                'dssms_section': self._render_dssms_section_text(unified_model),
                'quality_section': self._render_quality_section_text(unified_model),
                'total_trades': unified_model.performance.total_trades,
                'winning_trades': unified_model.performance.winning_trades,
                'losing_trades': unified_model.performance.losing_trades,
                'win_rate': f"{unified_model.performance.win_rate:.2%}",
                'trades_detail_section': self._render_trades_detail_text(unified_model),
                'footer_section': '=' * 80
            }
            
            # 変数置換
            for key, value in replacements.items():
                template = template.replace('{{' + key + '}}', str(value))
            
            return template
            
        except Exception as e:
            self.logger.error(f"テキストテンプレートレンダリング中にエラー: {e}")
            return self._generate_fallback_text(unified_model)
    
    def get_excel_template_config(self) -> Dict[str, Any]:
        """Excel用テンプレート設定の取得"""
        try:
            config_path = self.template_dir / "excel_template_config.json"
            
            if not config_path.exists():
                self.logger.warning("Excel設定テンプレートが見つかりません、デフォルトを作成します")
                self._create_excel_template_config()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Excel設定テンプレート取得中にエラー: {e}")
            return {}
    
    def validate_json_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """JSON スキーマによるデータ検証"""
        try:
            import jsonschema
            
            schema_path = self.template_dir / "json_schema_template.json"
            
            if not schema_path.exists():
                self.logger.warning("JSONスキーマテンプレートが見つかりません、デフォルトを作成します")
                self._create_json_schema_template()
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # スキーマ検証の実行
            jsonschema.validate(data, schema)
            
            return {
                'is_valid': True,
                'errors': [],
                'schema_version': schema.get('version', '1.0')
            }
            
        except ImportError:
            self.logger.warning("jsonschemaライブラリが見つかりません、検証をスキップします")
            return {'is_valid': True, 'errors': ['jsonschema library not found'], 'schema_version': '1.0'}
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [str(e)],
                'schema_version': '1.0'
            }
    
    def _render_performance_metrics_html(self, unified_model: UnifiedOutputModel) -> str:
        """パフォーマンス指標のHTML レンダリング"""
        metrics = [
            ("総リターン", f"{unified_model.performance.total_return:.2%}", 
             "positive" if unified_model.performance.total_return > 0 else "negative"),
            ("総損益", f"¥{unified_model.performance.total_pnl:,.0f}", 
             "positive" if unified_model.performance.total_pnl > 0 else "negative"),
            ("勝率", f"{unified_model.performance.win_rate:.1%}", "neutral"),
            ("総取引数", str(unified_model.performance.total_trades), "neutral"),
            ("シャープレシオ", f"{unified_model.performance.sharpe_ratio:.3f}", "neutral"),
            ("最大ドローダウン", f"{unified_model.performance.max_drawdown:.2%}", "negative")
        ]
        
        html_parts = []
        for label, value, css_class in metrics:
            html_parts.append(f"""
                <div class="metric-card">
                    <div class="metric-value {css_class}">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """)
        
        return "\n".join(html_parts)
    
    def _render_dssms_section_html(self, unified_model: UnifiedOutputModel) -> str:
        """DSSMS セクションのHTML レンダリング"""
        if not unified_model.dssms_metrics:
            return ""
        
        dssms = unified_model.dssms_metrics
        
        metrics_html = f"""
            <div class="metric-card">
                <div class="metric-value neutral">{dssms.switch_success_rate:.1%}</div>
                <div class="metric-label">戦略切り替え成功率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{dssms.switch_frequency:.2f}</div>
                <div class="metric-label">戦略切り替え頻度</div>
            </div>
        """
        
        strategy_scores_html = ""
        if dssms.strategy_scores:
            strategy_scores_html = "<h3>戦略スコア</h3><div class='table-container'><table><thead><tr><th>戦略</th><th>スコア</th></tr></thead><tbody>"
            for strategy, score in dssms.strategy_scores.items():
                strategy_scores_html += f"<tr><td>{strategy}</td><td>{score:.3f}</td></tr>"
            strategy_scores_html += "</tbody></table></div>"
        
        return f"""
            <div class="section">
                <h2 data-icon="🎯">DSSMS分析</h2>
                <div class="metrics-grid">
                    {metrics_html}
                </div>
                {strategy_scores_html}
            </div>
        """
    
    def _render_quality_section_html(self, unified_model: UnifiedOutputModel) -> str:
        """品質保証セクションのHTML レンダリング"""
        if not unified_model.quality_assurance:
            return ""
        
        qa = unified_model.quality_assurance
        
        # 信頼性スコアによるバッジの決定
        if qa.reliability_score >= 0.8:
            badge_class = "badge-excellent"
            badge_text = "優秀"
        elif qa.reliability_score >= 0.6:
            badge_class = "badge-good"
            badge_text = "良好"
        else:
            badge_class = "badge-poor"
            badge_text = "要改善"
        
        metrics_html = f"""
            <div class="metric-card">
                <div class="metric-value neutral">{qa.data_quality_score:.3f}</div>
                <div class="metric-label">データ品質スコア</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{qa.validation_score:.3f}</div>
                <div class="metric-label">検証スコア</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">
                    <span class="quality-badge {badge_class}">{qa.reliability_score:.1%}</span>
                </div>
                <div class="metric-label">信頼性スコア</div>
            </div>
        """
        
        return f"""
            <div class="section">
                <h2 data-icon="✅">品質保証情報</h2>
                <p>品質向上処理: {'適用済み' if qa.enhancement_applied else '未適用'}</p>
                <div class="metrics-grid">
                    {metrics_html}
                </div>
            </div>
        """
    
    def _render_trades_table_html(self, unified_model: UnifiedOutputModel) -> str:
        """取引テーブルのHTML レンダリング"""
        if not unified_model.trades:
            return "<p>取引データがありません。</p>"
        
        html = """
            <table>
                <thead>
                    <tr>
                        <th>戦略</th>
                        <th>エントリー</th>
                        <th>エグジット</th>
                        <th>損益</th>
                        <th>損益率</th>
                        <th>期間</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for trade in unified_model.trades[:20]:  # 最初の20取引
            profit_class = "positive" if trade.is_winner else "negative"
            html += f"""
                <tr>
                    <td>{trade.strategy}</td>
                    <td>{trade.entry_date.strftime('%Y-%m-%d')}<br>¥{trade.entry_price:.2f}</td>
                    <td>{trade.exit_date.strftime('%Y-%m-%d')}<br>¥{trade.exit_price:.2f}</td>
                    <td class="{profit_class}">¥{trade.profit_loss:.0f}</td>
                    <td class="{profit_class}">{trade.profit_loss_pct:.2%}</td>
                    <td>{trade.duration_days}日</td>
                </tr>
            """
        
        if len(unified_model.trades) > 20:
            html += f"<tr><td colspan='6' style='text-align: center; font-style: italic;'>... 他 {len(unified_model.trades) - 20} 取引</td></tr>"
        
        html += "</tbody></table>"
        return html
    
    def _render_system_info_html(self, unified_model: UnifiedOutputModel) -> str:
        """システム情報のHTML レンダリング"""
        return f"""
            <p><strong>データソース:</strong> {unified_model.metadata.data_source}</p>
            <p><strong>分析タイプ:</strong> {unified_model.metadata.analysis_type}</p>
            <p><strong>バージョン:</strong> {unified_model.metadata.version}</p>
        """
    
    def _render_performance_summary_text(self, unified_model: UnifiedOutputModel) -> str:
        """パフォーマンスサマリーのテキストレンダリング"""
        return f"""総リターン: {unified_model.performance.total_return:.2%}
総損益: ¥{unified_model.performance.total_pnl:,.0f}
勝率: {unified_model.performance.win_rate:.2%}
総取引数: {unified_model.performance.total_trades}
勝ち取引数: {unified_model.performance.winning_trades}
負け取引数: {unified_model.performance.losing_trades}
平均勝ち: ¥{unified_model.performance.average_win:,.0f}
平均負け: ¥{unified_model.performance.average_loss:,.0f}
プロフィットファクター: {unified_model.performance.profit_factor:.3f}
シャープレシオ: {unified_model.performance.sharpe_ratio:.3f}
最大ドローダウン: {unified_model.performance.max_drawdown:.2%}
ポートフォリオ価値: ¥{unified_model.performance.portfolio_value:,.0f}"""
    
    def _render_dssms_section_text(self, unified_model: UnifiedOutputModel) -> str:
        """DSSMS セクションのテキストレンダリング"""
        if not unified_model.dssms_metrics:
            return ""
        
        dssms = unified_model.dssms_metrics
        
        text = f"""
{'-' * 40}
DSSMS分析
{'-' * 40}
戦略切り替え成功率: {dssms.switch_success_rate:.2%}
戦略切り替え頻度: {dssms.switch_frequency:.2f}

戦略スコア:"""
        
        for strategy, score in dssms.strategy_scores.items():
            text += f"\n  {strategy}: {score:.3f}"
        
        return text
    
    def _render_quality_section_text(self, unified_model: UnifiedOutputModel) -> str:
        """品質保証セクションのテキストレンダリング"""
        if not unified_model.quality_assurance:
            return ""
        
        qa = unified_model.quality_assurance
        
        return f"""
{'-' * 40}
品質保証情報
{'-' * 40}
データ品質スコア: {qa.data_quality_score:.3f}
検証スコア: {qa.validation_score:.3f}
信頼性スコア: {qa.reliability_score:.3f}
品質向上適用: {'はい' if qa.enhancement_applied else 'いいえ'}"""
    
    def _render_trades_detail_text(self, unified_model: UnifiedOutputModel) -> str:
        """取引詳細のテキストレンダリング"""
        if not unified_model.trades:
            return "\n取引データがありません。"
        
        text = f"""
{'-' * 40}
取引履歴（最初の10取引）
{'-' * 40}"""
        
        for i, trade in enumerate(unified_model.trades[:10], 1):
            text += f"""

取引 #{i}:
  戦略: {trade.strategy}
  エントリー: {trade.entry_date.strftime('%Y-%m-%d')} @ ¥{trade.entry_price:.2f}
  エグジット: {trade.exit_date.strftime('%Y-%m-%d')} @ ¥{trade.exit_price:.2f}
  損益: ¥{trade.profit_loss:.0f} ({trade.profit_loss_pct:.2%})
  期間: {trade.duration_days}日
  結果: {'勝ち' if trade.is_winner else '負け'}"""
        
        if len(unified_model.trades) > 10:
            text += f"\n\n... 他 {len(unified_model.trades) - 10} 取引"
        
        return text
    
    def _generate_fallback_html(self, unified_model: UnifiedOutputModel) -> str:
        """フォールバック HTML の生成"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>レポート - {unified_model.metadata.ticker}</title></head>
        <body>
        <h1>統一バックテストレポート</h1>
        <p>銘柄: {unified_model.metadata.ticker}</p>
        <p>総リターン: {unified_model.performance.total_return:.2%}</p>
        <p>総取引数: {unified_model.performance.total_trades}</p>
        </body>
        </html>
        """
    
    def _generate_fallback_text(self, unified_model: UnifiedOutputModel) -> str:
        """フォールバック テキストの生成"""
        return f"""
統一バックテストレポート
========================

銘柄: {unified_model.metadata.ticker}
総リターン: {unified_model.performance.total_return:.2%}
総取引数: {unified_model.performance.total_trades}
        """


if __name__ == "__main__":
    # テスト実行
    from src.dssms.output_data_model import MetaData, PerformanceMetrics, TradeRecord
    
    # テスト用統一モデルの作成
    test_model = UnifiedOutputModel(
        metadata=MetaData(
            ticker='TEST',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            generation_timestamp=datetime.now(),
            data_source='test',
            analysis_type='test_backtest'
        ),
        performance=PerformanceMetrics(
            total_return=0.15, total_pnl=150000, win_rate=0.6, total_trades=10,
            winning_trades=6, losing_trades=4, average_win=35000, average_loss=-12500,
            profit_factor=1.68, sharpe_ratio=1.2, max_drawdown=-0.08,
            portfolio_value=1150000, initial_capital=1000000
        ),
        trades=[
            TradeRecord(
                trade_id='T001', strategy='TestStrategy',
                entry_date=datetime(2024, 1, 15), exit_date=datetime(2024, 1, 20),
                entry_price=100.0, exit_price=105.0, shares=100,
                profit_loss=500, profit_loss_pct=0.05, duration_days=5, is_winner=True
            )
        ]
    )
    
    # テンプレートマネージャーのテスト
    template_manager = TemplateManager("test_templates")
    
    print("HTMLテンプレートテスト:")
    html_output = template_manager.render_html_template(test_model)
    print(f"HTML出力長: {len(html_output)} 文字")
    
    print("\nテキストテンプレートテスト:")
    text_output = template_manager.render_text_template(test_model)
    print(f"テキスト出力長: {len(text_output)} 文字")
    
    print("\nExcel設定テンプレートテスト:")
    excel_config = template_manager.get_excel_template_config()
    print(f"Excel設定キー: {list(excel_config.keys())}")
    
    print("\nテンプレートマネージャーテスト完了")
