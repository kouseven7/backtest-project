"""
包括的レポートエンジン

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムのメインエンジン

機能:
- ハイブリッド型レポート生成（HTML中心＋エクスポート機能）
- 既存レポートシステムとの統合
- 階層化データアプローチ（サマリー/詳細/包括的レベル）
- インタラクティブ可視化
- マルチフォーマット出力（HTML/Excel/PDF）
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

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.reports.comprehensive.data_aggregator import DataAggregator
from src.reports.comprehensive.visualization_generator import VisualizationGenerator
from src.reports.comprehensive.report_template_manager import ReportTemplateManager
from src.reports.comprehensive.export_manager import ExportManager


class ComprehensiveReportEngine:
    """包括的レポートエンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.logger.info("=== ComprehensiveReportEngine 初期化開始 ===")
        
        # 設定ロード
        self.config = self._load_config(config_path)
        
        # 出力ディレクトリ設定
        self.output_dir = Path(self.config.get('output_directory', 'output/comprehensive_reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネント初期化
        self.data_aggregator = DataAggregator()
        self.visualization_generator = VisualizationGenerator()
        self.template_manager = ReportTemplateManager()
        self.export_manager = ExportManager()
        
        # レポート生成状態
        self.current_report_id = None
        self.report_data = {}
        self.generation_timestamp = None
        
        self.logger.info("ComprehensiveReportEngine 初期化完了")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルをロード"""
        try:
            if config_path is None:
                config_path = project_root / "config" / "comprehensive_reporting" / "report_config.json"
            
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"設定ファイルをロード: {config_path}")
                return config
            else:
                # デフォルト設定
                default_config = {
                    "output_directory": "output/comprehensive_reports",
                    "report_levels": ["summary", "detailed", "comprehensive"],
                    "default_level": "comprehensive",
                    "export_formats": ["html", "excel", "pdf"],
                    "visualization_enabled": True,
                    "interactive_features": True,
                    "integration_mode": "hybrid",
                    "template_theme": "bootstrap5",
                    "data_aggregation": {
                        "cache_enabled": True,
                        "cache_duration_hours": 24,
                        "auto_refresh": True
                    },
                    "performance_settings": {
                        "max_data_points": 10000,
                        "chart_optimization": True,
                        "lazy_loading": True
                    }
                }
                self.logger.info("デフォルト設定を使用")
                return default_config
                
        except Exception as e:
            self.logger.error(f"設定ロードエラー: {e}")
            return {}
    
    def generate_comprehensive_report(
        self,
        report_type: str = "comprehensive",
        level: str = "comprehensive",
        data_sources: Optional[List[str]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        strategies: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        包括的レポート生成
        
        Args:
            report_type: レポートタイプ（comprehensive/comparison/performance/risk）
            level: 詳細レベル（summary/detailed/comprehensive）
            data_sources: データソースリスト
            date_range: 対象期間
            strategies: 対象戦略リスト
            custom_params: カスタムパラメータ
            
        Returns:
            生成結果とメタデータ
        """
        try:
            self.logger.info(f"=== 包括的レポート生成開始 ===")
            self.logger.info(f"レポートタイプ: {report_type}, レベル: {level}")
            
            # レポートID生成
            self.current_report_id = self._generate_report_id(report_type, level)
            self.generation_timestamp = datetime.now()
            
            # データ集約
            self.logger.info("データ集約開始")
            aggregated_data = self.data_aggregator.aggregate_data(
                data_sources=data_sources,
                date_range=date_range,
                strategies=strategies,
                level=level
            )
            
            # 可視化生成
            self.logger.info("可視化生成開始")
            visualizations = self.visualization_generator.generate_visualizations(
                data=aggregated_data,
                report_type=report_type,
                level=level
            )
            
            # レポートデータ構築
            self.report_data = {
                'metadata': {
                    'report_id': self.current_report_id,
                    'report_type': report_type,
                    'level': level,
                    'generation_timestamp': self.generation_timestamp,
                    'data_sources': data_sources,
                    'date_range': date_range,
                    'strategies': strategies
                },
                'data': aggregated_data,
                'visualizations': visualizations,
                'custom_params': custom_params or {}
            }
            
            # HTMLレポート生成
            self.logger.info("HTMLレポート生成開始")
            html_report = self.template_manager.generate_html_report(
                report_data=self.report_data,
                template_type=report_type,
                level=level
            )
            
            # レポート保存
            report_path = self.output_dir / f"{self.current_report_id}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            self.logger.info(f"HTMLレポート保存: {report_path}")
            
            # 結果構築
            result = {
                'success': True,
                'report_id': self.current_report_id,
                'report_path': str(report_path),
                'metadata': self.report_data['metadata'],
                'data_summary': self._create_data_summary(aggregated_data),
                'visualizations_count': len(visualizations),
                'generation_time': (datetime.now() - self.generation_timestamp).total_seconds()
            }
            
            self.logger.info(f"包括的レポート生成完了: {result['generation_time']:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'report_id': self.current_report_id
            }
    
    def export_report(
        self,
        format_type: str,
        report_id: Optional[str] = None,
        custom_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        レポートエクスポート
        
        Args:
            format_type: エクスポート形式（excel/pdf/json）
            report_id: レポートID（None の場合は最新）
            custom_options: カスタムオプション
            
        Returns:
            エクスポート結果
        """
        try:
            target_report_id = report_id or self.current_report_id
            if not target_report_id:
                raise ValueError("エクスポート対象のレポートIDが指定されていません")
            
            self.logger.info(f"レポートエクスポート開始: {target_report_id} -> {format_type}")
            
            # エクスポート実行
            export_result = self.export_manager.export_report(
                report_data=self.report_data,
                format_type=format_type,
                report_id=target_report_id,
                output_dir=self.output_dir,
                custom_options=custom_options
            )
            
            self.logger.info(f"エクスポート完了: {export_result.get('export_path')}")
            return export_result
            
        except Exception as e:
            self.logger.error(f"エクスポートエラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_comparison_report(
        self,
        comparison_items: List[Dict[str, Any]],
        comparison_type: str = "strategies",
        level: str = "detailed"
    ) -> Dict[str, Any]:
        """
        比較レポート生成
        
        Args:
            comparison_items: 比較対象リスト
            comparison_type: 比較タイプ（strategies/periods/configurations）
            level: 詳細レベル
            
        Returns:
            比較レポート結果
        """
        try:
            self.logger.info(f"比較レポート生成開始: {comparison_type}")
            
            # 比較データ集約
            comparison_data = self.data_aggregator.aggregate_comparison_data(
                comparison_items=comparison_items,
                comparison_type=comparison_type,
                level=level
            )
            
            # 比較可視化生成
            comparison_visualizations = self.visualization_generator.generate_comparison_visualizations(
                comparison_data=comparison_data,
                comparison_type=comparison_type,
                level=level
            )
            
            # 比較レポート生成
            return self.generate_comprehensive_report(
                report_type="comparison",
                level=level,
                custom_params={
                    'comparison_items': comparison_items,
                    'comparison_type': comparison_type,
                    'comparison_data': comparison_data,
                    'comparison_visualizations': comparison_visualizations
                }
            )
            
        except Exception as e:
            self.logger.error(f"比較レポート生成エラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_report_list(self, limit: int = 20) -> List[Dict[str, Any]]:
        """生成済みレポート一覧取得"""
        try:
            reports = []
            
            for html_file in self.output_dir.glob("*.html"):
                try:
                    # ファイル名からレポート情報を推定
                    report_info = {
                        'report_id': html_file.stem,
                        'file_path': str(html_file),
                        'creation_time': datetime.fromtimestamp(html_file.stat().st_mtime),
                        'file_size': html_file.stat().st_size
                    }
                    reports.append(report_info)
                    
                except Exception as e:
                    self.logger.warning(f"レポート情報取得エラー: {html_file}: {e}")
                    continue
            
            # 作成時間でソート（新しい順）
            reports.sort(key=lambda x: x['creation_time'], reverse=True)
            
            return reports[:limit]
            
        except Exception as e:
            self.logger.error(f"レポート一覧取得エラー: {e}")
            return []
    
    def delete_report(self, report_id: str) -> bool:
        """レポート削除"""
        try:
            # HTMLファイル削除
            html_path = self.output_dir / f"{report_id}.html"
            if html_path.exists():
                html_path.unlink()
                self.logger.info(f"HTMLレポート削除: {html_path}")
            
            # 関連ファイル削除（Excel, PDF等）
            for suffix in ['.xlsx', '.pdf', '.json']:
                file_path = self.output_dir / f"{report_id}{suffix}"
                if file_path.exists():
                    file_path.unlink()
                    self.logger.info(f"関連ファイル削除: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"レポート削除エラー: {e}")
            return False
    
    def _generate_report_id(self, report_type: str, level: str) -> str:
        """レポートID生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"comprehensive_{report_type}_{level}_{timestamp}"
    
    def _create_data_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """データサマリー作成"""
        try:
            summary = {
                'total_records': 0,
                'date_range': {},
                'strategies_count': 0,
                'data_sources': []
            }
            
            # DSSMSデータサマリー
            if 'dssms_data' in aggregated_data:
                dssms_data = aggregated_data['dssms_data']
                if isinstance(dssms_data, pd.DataFrame) and not dssms_data.empty:
                    summary['total_records'] += len(dssms_data)
                    summary['date_range']['start'] = dssms_data.index.min() if hasattr(dssms_data, 'index') else None
                    summary['date_range']['end'] = dssms_data.index.max() if hasattr(dssms_data, 'index') else None
            
            # 戦略データサマリー
            if 'strategy_data' in aggregated_data:
                strategy_data = aggregated_data['strategy_data']
                if isinstance(strategy_data, dict):
                    summary['strategies_count'] = len(strategy_data)
            
            # データソース情報
            if 'data_sources' in aggregated_data:
                summary['data_sources'] = aggregated_data['data_sources']
            
            return summary
            
        except Exception as e:
            self.logger.error(f"データサマリー作成エラー: {e}")
            return {}


if __name__ == "__main__":
    # デモ実行
    engine = ComprehensiveReportEngine()
    
    # サンプルレポート生成
    result = engine.generate_comprehensive_report(
        report_type="comprehensive",
        level="summary"
    )
    
    print(f"レポート生成結果: {result}")
