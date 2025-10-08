"""
Test suite for FallbackMonitor (TODO-FB-008)

Comprehensive testing for fallback usage monitoring dashboard functionality.
Tests data collection, analysis, production readiness evaluation, report generation, and cleanup.

Coverage:
- FallbackMonitor initialization & directory creation
- Weekly data collection from reports/fallback/
- Fallback pattern analysis & statistics calculation  
- Production readiness evaluation & metrics
- HTML/JSON report generation
- Cleanup functionality
- System status monitoring

Test Data Strategy:
- Mock fallback usage reports in reports/fallback/
- Controlled test scenarios for production readiness evaluation
- Validation of HTML/JSON output format & content
"""

import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Import target classes
from tools.fallback_monitor import (
    FallbackMonitor, 
    FallbackUsageStats, 
    ProductionReadinessMetrics, 
    MonitoringReport,
    FALLBACK_REPORTS_DIR,
    MONITORING_REPORTS_DIR
)


class TestFallbackMonitor:
    """FallbackMonitor comprehensive test suite"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Temporary workspace for testing"""
        temp_dir = tempfile.mkdtemp()
        fallback_dir = Path(temp_dir) / "reports" / "fallback"
        monitoring_dir = Path(temp_dir) / "reports" / "monitoring"
        
        fallback_dir.mkdir(parents=True, exist_ok=True)
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        yield {
            'root': temp_dir,
            'fallback_dir': str(fallback_dir),
            'monitoring_dir': str(monitoring_dir)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_fallback_reports(self, temp_workspace):
        """Sample fallback usage reports for testing"""
        fallback_dir = Path(temp_workspace['fallback_dir'])
        
        # Sample report 1: Recent with mixed fallback usage
        report1_data = {
            "report_id": "test_report_001",
            "generation_timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "usage_statistics": {
                "VWAPStrategy": {
                    "component_type": "STRATEGY_ENGINE",
                    "total_calls": 100,
                    "fallback_calls": 5,
                    "avg_execution_time": 25.5,
                    "error_types": ["DataFetchError", "CalculationError"]
                },
                "DSSMSCore": {
                    "component_type": "DSSMS_CORE", 
                    "total_calls": 200,
                    "fallback_calls": 2,
                    "avg_execution_time": 15.2,
                    "error_types": ["RankingError"]
                }
            }
        }
        
        # Sample report 2: Older report (outside analysis window)
        report2_data = {
            "report_id": "test_report_002",
            "generation_timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "usage_statistics": {
                "MultiStrategyManager": {
                    "component_type": "MULTI_STRATEGY",
                    "total_calls": 50,
                    "fallback_calls": 10,
                    "avg_execution_time": 45.0,
                    "error_types": ["IntegrationError"]
                }
            }
        }
        
        # Sample report 3: Recent with high fallback usage (critical scenario)
        report3_data = {
            "report_id": "test_report_003",
            "generation_timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "usage_statistics": {
                "RiskManager": {
                    "component_type": "RISK_MANAGER",
                    "total_calls": 80,
                    "fallback_calls": 25,
                    "avg_execution_time": 35.8,
                    "error_types": ["ValidationError", "ThresholdError", "CalculationError"]
                }
            }
        }
        
        # Write test reports
        reports = [
            ("fallback_usage_report_20241001_120000.json", report1_data),
            ("fallback_usage_report_20240920_120000.json", report2_data),
            ("fallback_usage_report_20241002_140000.json", report3_data)
        ]
        
        for filename, data in reports:
            report_path = fallback_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        return reports
    
    def test_fallback_monitor_initialization(self, temp_workspace):
        """Test FallbackMonitor initialization & directory creation"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir'],
            analysis_days=7
        )
        
        # Check initialization
        assert monitor.analysis_days == 7
        assert str(monitor.fallback_reports_dir) == temp_workspace['fallback_dir']
        assert str(monitor.monitoring_reports_dir) == temp_workspace['monitoring_dir']
        
        # Check directories exist
        assert Path(temp_workspace['fallback_dir']).exists()
        assert Path(temp_workspace['monitoring_dir']).exists()
        
        # Check logger setup
        assert monitor.logger is not None
        
        # Check SystemFallbackPolicy reference
        assert monitor.fallback_policy is not None
    
    def test_collect_weekly_data(self, temp_workspace, sample_fallback_reports):
        """Test weekly data collection from fallback reports"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir'],
            analysis_days=7
        )
        
        collected_data = monitor._collect_weekly_data()
        
        # Should collect reports within 7 days (report1 & report3, not report2)
        assert len(collected_data) == 2
        
        # Verify data structure
        for report in collected_data:
            assert 'report_id' in report
            assert 'usage_statistics' in report
            assert 'file_timestamp' in report
        
        # Check report IDs
        report_ids = [report['report_id'] for report in collected_data]
        assert 'test_report_001' in report_ids
        assert 'test_report_003' in report_ids
        assert 'test_report_002' not in report_ids  # Too old
    
    def test_analyze_fallback_patterns(self, temp_workspace, sample_fallback_reports):
        """Test fallback pattern analysis & statistics calculation"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        raw_data = monitor._collect_weekly_data()
        fallback_stats = monitor._analyze_fallback_patterns(raw_data)
        
        # Should analyze 3 components (VWAPStrategy, DSSMSCore, RiskManager)
        assert len(fallback_stats) == 3
        
        # Check component analysis
        component_names = [stat.component_name for stat in fallback_stats]
        assert 'VWAPStrategy' in component_names
        assert 'DSSMSCore' in component_names
        assert 'RiskManager' in component_names
        
        # Check statistics calculation for VWAPStrategy
        vwap_stat = next(stat for stat in fallback_stats if stat.component_name == 'VWAPStrategy')
        assert vwap_stat.component_type == 'STRATEGY_ENGINE'
        assert vwap_stat.usage_count == 5
        assert vwap_stat.success_rate == 0.95  # (100-5)/100
        assert vwap_stat.avg_execution_time == 25.5
        assert len(vwap_stat.error_types) == 2
        assert 'DataFetchError' in vwap_stat.error_types
        
        # Check severity score calculation (RiskManager should have high severity)
        risk_stat = next(stat for stat in fallback_stats if stat.component_name == 'RiskManager')
        assert risk_stat.severity_score > 0.3  # High fallback rate + multiple error types
    
    def test_evaluate_production_readiness(self, temp_workspace, sample_fallback_reports):
        """Test production readiness evaluation"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        readiness = monitor.evaluate_production_readiness()
        
        # Check metrics structure
        assert isinstance(readiness, ProductionReadinessMetrics)
        assert 0.0 <= readiness.overall_score <= 1.0
        assert readiness.fallback_usage_percentage >= 0.0
        assert 0.0 <= readiness.critical_component_stability <= 1.0
        assert isinstance(readiness.acceptable_for_production, bool)
        assert isinstance(readiness.recommendations, list)
        
        # With sample data, should detect issues
        assert readiness.overall_score < 1.0  # Not perfect due to fallback usage
        assert len(readiness.recommendations) > 0
    
    def test_generate_weekly_report(self, temp_workspace, sample_fallback_reports):
        """Test weekly report generation (HTML + JSON)"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        # Generate report
        html_path = monitor.generate_weekly_report(force_regenerate=True)
        
        # Check HTML file created
        assert Path(html_path).exists()
        assert html_path.endswith('.html')
        
        # Check JSON file created (same name, different extension)
        json_path = html_path.replace('.html', '.json')
        assert Path(json_path).exists()
        
        # Validate HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            assert 'Fallback Monitor - Weekly Report' in html_content
            assert 'Production Readiness' in html_content
            assert 'Executive Summary' in html_content
            assert 'VWAPStrategy' in html_content  # Component name
            assert 'STRATEGY_ENGINE' in html_content  # Component type
        
        # Validate JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            assert 'report_id' in json_data
            assert 'fallback_stats' in json_data
            assert 'production_metrics' in json_data
            assert 'executive_summary' in json_data
            assert len(json_data['fallback_stats']) == 3  # 3 components
    
    def test_cleanup_old_reports(self, temp_workspace):
        """Test cleanup functionality for old monitoring reports"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        monitoring_dir = Path(temp_workspace['monitoring_dir'])
        
        # Create test files with different ages
        old_file = monitoring_dir / "old_report_20240901_120000.html"
        recent_file = monitoring_dir / "recent_report_20241001_120000.html"
        
        old_file.write_text("old report content")
        recent_file.write_text("recent report content")
        
        # Modify timestamps to simulate age
        old_time = (datetime.now() - timedelta(days=35)).timestamp()
        recent_time = (datetime.now() - timedelta(days=5)).timestamp()
        
        os.utime(old_file, (old_time, old_time))
        os.utime(recent_file, (recent_time, recent_time))
        
        # Run cleanup (30 days retention)
        cleanup_result = monitor.cleanup_old_reports(retention_days=30)
        
        # Check results
        assert cleanup_result['deleted_files'] == 1
        assert cleanup_result['total_size_mb'] > 0
        
        # Check files
        assert not old_file.exists()  # Should be deleted
        assert recent_file.exists()   # Should remain
    
    def test_get_system_status(self, temp_workspace, sample_fallback_reports):
        """Test system status information retrieval"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        status = monitor.get_system_status()
        
        # Check status structure
        assert isinstance(status, dict)
        assert 'monitor_status' in status
        assert 'fallback_reports_available' in status
        assert 'monitoring_reports_count' in status
        assert 'matplotlib_available' in status
        assert 'analysis_days' in status
        assert 'last_updated' in status
        
        # Check values
        assert status['monitor_status'] == 'initialized'
        assert status['fallback_reports_available'] == 3  # From sample data
        assert status['analysis_days'] == 7
        assert isinstance(status['matplotlib_available'], bool)
    
    def test_no_fallback_data_scenario(self, temp_workspace):
        """Test behavior when no fallback data is available"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        # No data files in fallback directory
        collected_data = monitor._collect_weekly_data()
        assert len(collected_data) == 0
        
        fallback_stats = monitor._analyze_fallback_patterns(collected_data)
        assert len(fallback_stats) == 0
        
        readiness = monitor.evaluate_production_readiness()
        assert readiness.overall_score == 1.0  # Perfect when no issues
        assert readiness.acceptable_for_production == True
        assert "No fallback usage detected" in readiness.recommendations[0]
    
    @patch('tools.fallback_monitor.MATPLOTLIB_AVAILABLE', False)
    def test_no_matplotlib_scenario(self, temp_workspace, sample_fallback_reports):
        """Test behavior when matplotlib is not available"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        # Generate report without matplotlib
        html_path = monitor.generate_weekly_report()
        
        # Check HTML contains fallback message
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            assert 'matplotlib not available - charts disabled' in html_content
    
    def test_error_handling(self, temp_workspace):
        """Test error handling in various scenarios"""
        monitor = FallbackMonitor(
            fallback_reports_dir=temp_workspace['fallback_dir'],
            monitoring_reports_dir=temp_workspace['monitoring_dir']
        )
        
        # Test with corrupted JSON file
        fallback_dir = Path(temp_workspace['fallback_dir'])
        corrupted_file = fallback_dir / "corrupted_report.json"
        corrupted_file.write_text("invalid json content")
        
        # Should handle gracefully
        collected_data = monitor._collect_weekly_data()
        assert isinstance(collected_data, list)  # Should not crash
        
        # Test cleanup error handling
        cleanup_result = monitor.cleanup_old_reports(retention_days=30)
        assert 'deleted_files' in cleanup_result
        assert 'total_size_mb' in cleanup_result


def test_dataclass_structures():
    """Test dataclass structures for proper serialization"""
    
    # Test FallbackUsageStats
    usage_stat = FallbackUsageStats(
        component_name="TestComponent",
        component_type="TEST_TYPE", 
        usage_count=10,
        success_rate=0.9,
        avg_execution_time=25.5,
        error_types=["Error1", "Error2"],
        severity_score=0.3
    )
    
    # Should serialize to dict
    stat_dict = usage_stat.__dict__
    assert stat_dict['component_name'] == "TestComponent"
    assert stat_dict['usage_count'] == 10
    
    # Test ProductionReadinessMetrics
    readiness = ProductionReadinessMetrics(
        overall_score=0.85,
        fallback_usage_percentage=5.2,
        critical_component_stability=0.92,
        acceptable_for_production=True,
        recommendations=["All systems operational"]
    )
    
    readiness_dict = readiness.__dict__
    assert readiness_dict['overall_score'] == 0.85
    assert readiness_dict['acceptable_for_production'] == True


if __name__ == "__main__":
    """Standalone test execution"""
    print("[TEST] Running FallbackMonitor Test Suite")
    print("=" * 50)
    
    # Run with pytest discovery
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print(f"\n[CHART] Test Suite Result: {'[OK] PASSED' if exit_code == 0 else '[ERROR] FAILED'}")
    sys.exit(exit_code)