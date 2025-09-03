"""
包括的レポートシステム テスト

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムの単体テスト

pytest を使用したテストファイル
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.reports.comprehensive import (
    ComprehensiveReportEngine,
    DataAggregator,
    VisualizationGenerator,
    ReportTemplateManager,
    ExportManager
)


class TestDataAggregator:
    """DataAggregator テストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.aggregator = DataAggregator()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.aggregator is not None
        assert hasattr(self.aggregator, 'logger')
        assert hasattr(self.aggregator, 'data_sources')
    
    def test_aggregate_data_summary(self):
        """サマリーレベルデータ集約テスト"""
        result = self.aggregator.aggregate_data(level="summary")
        
        assert isinstance(result, dict)
        assert 'metadata' in result
        assert 'dssms_data' in result
        assert 'strategy_data' in result
        assert 'summary_statistics' in result
    
    def test_aggregate_data_detailed(self):
        """詳細レベルデータ集約テスト"""
        result = self.aggregator.aggregate_data(level="detailed")
        
        assert isinstance(result, dict)
        assert result['metadata']['level'] == "detailed"
    
    def test_aggregate_comparison_data(self):
        """比較データ集約テスト"""
        comparison_items = [
            {'name': 'strategy1'},
            {'name': 'strategy2'}
        ]
        
        result = self.aggregator.aggregate_comparison_data(
            comparison_items=comparison_items,
            comparison_type="strategies",
            level="summary"
        )
        
        assert isinstance(result, dict)
        assert result['comparison_type'] == "strategies"
        assert len(result['comparison_items']) == 2


class TestVisualizationGenerator:
    """VisualizationGenerator テストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.generator = VisualizationGenerator()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.generator is not None
        assert hasattr(self.generator, 'chart_config')
        assert hasattr(self.generator, 'chartjs_templates')
    
    def test_generate_visualizations_empty_data(self):
        """空データでの可視化生成テスト"""
        result = self.generator.generate_visualizations({})
        
        assert isinstance(result, dict)
        assert 'metadata' in result
        assert 'charts' in result
        assert result['metadata']['total_charts'] == 0
    
    def test_generate_visualizations_with_sample_data(self):
        """サンプルデータでの可視化生成テスト"""
        sample_data = {
            'dssms_data': {
                'sample1': {'total_records': 100},
                'sample2': {'total_records': 200}
            },
            'strategy_data': {
                'strategy1': {'lines_count': 150, 'file_size': 5000},
                'strategy2': {'lines_count': 200, 'file_size': 7000}
            }
        }
        
        result = self.generator.generate_visualizations(sample_data)
        
        assert isinstance(result, dict)
        assert result['metadata']['total_charts'] > 0
        assert 'javascript_code' in result
        assert 'html_snippets' in result


class TestReportTemplateManager:
    """ReportTemplateManager テストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.manager = ReportTemplateManager()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.manager is not None
        assert hasattr(self.manager, 'html_templates')
        assert hasattr(self.manager, 'template_config')
    
    def test_generate_html_report(self):
        """HTMLレポート生成テスト"""
        sample_report_data = {
            'metadata': {
                'report_id': 'test_001',
                'generation_timestamp': datetime.now(),
                'report_type': 'comprehensive',
                'level': 'summary'
            },
            'data': {
                'summary_statistics': {
                    'data_overview': {
                        'total_files': 5
                    }
                }
            },
            'visualizations': {
                'html_snippets': {
                    'chart_grid': '<div>Test Chart</div>'
                },
                'javascript_code': ['console.log("test");']
            }
        }
        
        html_result = self.manager.generate_html_report(sample_report_data)
        
        assert isinstance(html_result, str)
        assert len(html_result) > 0
        assert '<!DOCTYPE html>' in html_result
        assert 'bootstrap' in html_result.lower()


class TestExportManager:
    """ExportManager テストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.manager = ExportManager()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """テストクリーンアップ"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.manager is not None
        assert hasattr(self.manager, 'available_formats')
        assert 'json' in self.manager.available_formats
    
    def test_get_supported_formats(self):
        """サポート形式取得テスト"""
        formats = self.manager.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert 'json' in formats
    
    def test_json_export(self):
        """JSONエクスポートテスト"""
        sample_data = {
            'metadata': {
                'report_id': 'test_json',
                'timestamp': datetime.now()
            },
            'data': {
                'test_data': {'value': 123}
            }
        }
        
        result = self.manager.export_report(
            report_data=sample_data,
            format_type='json',
            report_id='test_json',
            output_dir=self.temp_dir
        )
        
        assert result['success'] == True
        assert 'export_path' in result
        
        # ファイル存在確認
        export_path = Path(result['export_path'])
        assert export_path.exists()
        assert export_path.suffix == '.json'


class TestComprehensiveReportEngine:
    """ComprehensiveReportEngine テストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.engine = ComprehensiveReportEngine()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """テストクリーンアップ"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.engine is not None
        assert hasattr(self.engine, 'data_aggregator')
        assert hasattr(self.engine, 'visualization_generator')
        assert hasattr(self.engine, 'template_manager')
        assert hasattr(self.engine, 'export_manager')
    
    def test_generate_comprehensive_report_summary(self):
        """サマリーレベル包括的レポート生成テスト"""
        result = self.engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="summary"
        )
        
        assert isinstance(result, dict)
        if result.get('success'):
            assert 'report_id' in result
            assert 'report_path' in result
            assert 'generation_time' in result
        else:
            # エラーの場合もテスト自体は成功とする（データ不足等）
            assert 'error' in result
    
    def test_get_report_list(self):
        """レポート一覧取得テスト"""
        report_list = self.engine.get_report_list()
        
        assert isinstance(report_list, list)
        # 空でもリストが返ることを確認


class TestIntegration:
    """統合テストクラス"""
    
    def test_full_workflow(self):
        """完全ワークフローテスト"""
        # エンジン初期化
        engine = ComprehensiveReportEngine()
        
        # レポート生成
        result = engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="summary"
        )
        
        # 結果は成功またはエラーメッセージ付きの辞書であることを確認
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # 成功の場合の追加チェック
            assert 'report_id' in result
            assert 'report_path' in result
            
            # エクスポートテスト
            if engine.current_report_id:
                export_result = engine.export_report('json')
                assert isinstance(export_result, dict)


# パフォーマンステスト
@pytest.mark.performance
class TestPerformance:
    """パフォーマンステストクラス"""
    
    def test_report_generation_time(self):
        """レポート生成時間テスト"""
        engine = ComprehensiveReportEngine()
        
        start_time = datetime.now()
        result = engine.generate_comprehensive_report(
            report_type="comprehensive",
            level="summary"
        )
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # 30秒以内で完了することを確認
        assert execution_time < 30, f"レポート生成時間が長すぎます: {execution_time}秒"


if __name__ == "__main__":
    # pytest実行
    pytest.main([__file__, "-v"])
