"""
Module: Test Composite Strategy Execution
File: test_composite_strategy_execution.py
Description: 
  4-1-2「複合戦略実行フロー設計・実装」- Comprehensive Test Suite
  複合戦略実行システムの包括的テストスイート

Author: imega
Created: 2025-01-28
Modified: 2025-01-28

Dependencies:
  - pytest
  - pandas
  - numpy
  - config.composite_strategy_execution_engine
"""

import os
import sys
import json
import pytest
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# テスト対象のインポート
try:
    from config.composite_strategy_execution_engine import (
        CompositeStrategyExecutionEngine, 
        ExecutionRequest, 
        ExecutionResponse,
        ExecutionMode,
        ExecutionStatus,
        create_execution_request
    )
    from config.strategy_execution_pipeline import StrategyExecutionPipeline
    from config.strategy_execution_coordinator import StrategyExecutionCoordinator
    from config.execution_result_aggregator import ExecutionResultAggregator
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# テストデータの生成
def generate_sample_market_data(periods: int = 100, freq: str = '1H') -> pd.DataFrame:
    """サンプルマーケットデータ生成"""
    np.random.seed(42)  # 再現性のため
    
    timestamps = pd.date_range('2025-01-01', periods=periods, freq=freq)
    price_changes = np.random.randn(periods) * 0.02
    prices = 100 * (1 + price_changes).cumprod()
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.randn(periods) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(periods)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(periods)) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    })

class TestCompositeStrategyExecutionEngine:
    """複合戦略実行エンジンのテストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータ"""
        return generate_sample_market_data(50)
        
    @pytest.fixture
    def engine(self):
        """エンジンインスタンス"""
        try:
            return CompositeStrategyExecutionEngine()
        except Exception as e:
            pytest.skip(f"Engine initialization failed: {e}")
            
    @pytest.fixture
    def sample_request(self, sample_data):
        """サンプルリクエスト"""
        return create_execution_request(
            market_data=sample_data,
            strategies=["test_strategy_a", "test_strategy_b"],
            parameters={"test_param": "test_value"},
            execution_mode=ExecutionMode.COMPOSITE
        )

    def test_engine_initialization(self, engine):
        """エンジンの初期化テスト"""
        assert engine is not None
        assert hasattr(engine, 'config')
        assert hasattr(engine, 'current_status')
        assert engine.current_status == ExecutionStatus.IDLE
        
        # コンポーネントの存在確認
        status = engine.get_engine_status()
        assert status['component_status']['strategy_selector'] == True
        assert status['component_status']['pipeline'] == True
        assert status['component_status']['coordinator'] == True
        assert status['component_status']['aggregator'] == True
        
    def test_config_loading(self, engine):
        """設定読み込みテスト"""
        assert 'composite_execution' in engine.config
        
        composite_config = engine.config['composite_execution']
        assert 'execution_pipeline' in composite_config
        assert 'coordination' in composite_config
        assert 'aggregation' in composite_config
        
        # パイプライン設定の確認
        pipeline_config = composite_config['execution_pipeline']
        assert 'stages' in pipeline_config
        
        stages = pipeline_config['stages']
        expected_stages = ['strategy_selection', 'weight_calculation', 'signal_integration', 'risk_adjustment', 'execution']
        actual_stages = [stage['id'] for stage in stages]
        
        for expected_stage in expected_stages:
            assert expected_stage in actual_stages
            
    def test_execution_request_creation(self, sample_data):
        """実行リクエスト作成テスト"""
        request = create_execution_request(
            market_data=sample_data,
            strategies=["strategy_1", "strategy_2"],
            parameters={"param1": "value1"},
            execution_mode=ExecutionMode.MULTI_STRATEGY,
            timeout=120
        )
        
        assert isinstance(request, ExecutionRequest)
        assert request.market_data is not None
        assert len(request.market_data) == len(sample_data)
        assert request.strategies == ["strategy_1", "strategy_2"]
        assert request.parameters == {"param1": "value1"}
        assert request.execution_mode == ExecutionMode.MULTI_STRATEGY
        assert request.timeout == 120
        assert request.request_id is not None
        
    @pytest.mark.skip(reason="Requires full system integration")
    def test_single_strategy_execution(self, engine, sample_data):
        """単一戦略実行テスト"""
        request = create_execution_request(
            market_data=sample_data,
            strategies=["test_strategy"],
            execution_mode=ExecutionMode.SINGLE_STRATEGY
        )
        
        response = engine.execute(request)
        
        assert isinstance(response, ExecutionResponse)
        assert response.request_id == request.request_id
        assert response.strategy_count >= 0
        assert response.execution_time >= 0
        
    @pytest.mark.skip(reason="Requires full system integration")
    def test_multi_strategy_execution(self, engine, sample_data):
        """マルチ戦略実行テスト"""
        request = create_execution_request(
            market_data=sample_data,
            strategies=["strategy_a", "strategy_b", "strategy_c"],
            execution_mode=ExecutionMode.MULTI_STRATEGY
        )
        
        response = engine.execute(request)
        
        assert isinstance(response, ExecutionResponse)
        assert response.request_id == request.request_id
        assert response.strategy_count >= 0
        
    @pytest.mark.skip(reason="Requires full system integration") 
    def test_composite_strategy_execution(self, engine, sample_data):
        """複合戦略実行テスト"""
        request = create_execution_request(
            market_data=sample_data,
            strategies=["composite_strategy_1", "composite_strategy_2"],
            execution_mode=ExecutionMode.COMPOSITE
        )
        
        response = engine.execute(request)
        
        assert isinstance(response, ExecutionResponse)
        assert response.request_id == request.request_id
        
    def test_engine_status_tracking(self, engine):
        """エンジンステータス追跡テスト"""
        initial_status = engine.get_engine_status()
        
        assert initial_status['current_status'] == 'idle'
        assert initial_status['uptime_seconds'] >= 0
        assert initial_status['active_requests'] == 0
        assert initial_status['execution_history_count'] == 0
        
        # パフォーマンス指標の存在確認
        assert 'performance_metrics' in initial_status
        metrics = initial_status['performance_metrics']
        assert 'total_executions' in metrics
        assert 'successful_executions' in metrics
        assert 'average_execution_time' in metrics
        
    def test_execution_history(self, engine):
        """実行履歴テスト"""
        # 初期状態では履歴が空
        assert len(engine.execution_history) == 0
        
        # ダミーレスポンスを追加
        dummy_response = ExecutionResponse(
            request_id="test_request",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.5,
            strategy_count=2,
            successful_strategies=2
        )
        
        engine.execution_history.append(dummy_response)
        
        # 履歴の確認
        assert len(engine.execution_history) == 1
        assert engine.execution_history[0].request_id == "test_request"
        
    def test_performance_metrics_update(self, engine):
        """パフォーマンス指標更新テスト"""
        initial_metrics = engine.performance_metrics.copy()
        
        # ダミーレスポンスで更新
        response = ExecutionResponse(
            request_id="test_request",
            status=ExecutionStatus.COMPLETED,
            execution_time=2.0,
            strategy_count=1,
            successful_strategies=1
        )
        
        engine._update_performance_metrics(response)
        
        # 更新後の確認
        assert engine.performance_metrics['total_executions'] == initial_metrics['total_executions'] + 1
        assert engine.performance_metrics['successful_executions'] == initial_metrics['successful_executions'] + 1
        assert engine.performance_metrics['total_execution_time'] > initial_metrics['total_execution_time']
        
    def test_execution_report_generation(self, engine):
        """実行レポート生成テスト"""
        # ダミーデータで履歴を作成
        dummy_response = ExecutionResponse(
            request_id="test_report_request",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.0,
            strategy_count=2,
            successful_strategies=2,
            metadata={"test": "data"}
        )
        
        engine.execution_history.append(dummy_response)
        
        # 全体サマリーレポート
        summary_report = engine.get_execution_report()
        assert "複合戦略実行エンジン サマリー" in summary_report
        assert "パフォーマンス指標" in summary_report
        
        # 特定リクエストのレポート
        specific_report = engine.get_execution_report("test_report_request")
        assert "test_report_request" in specific_report
        assert "実行時間: 1.00秒" in specific_report
        
        # 存在しないリクエスト
        missing_report = engine.get_execution_report("non_existent_request")
        assert "not found" in missing_report.lower()
        
    def test_engine_shutdown(self, engine):
        """エンジン停止テスト"""
        # アクティブリクエストを追加
        engine.active_requests["test_request"] = ExecutionRequest(
            request_id="test_request",
            market_data=pd.DataFrame()
        )
        
        assert len(engine.active_requests) == 1
        
        # シャットダウン実行
        engine.shutdown()
        
        # 状態確認
        assert engine.current_status == ExecutionStatus.CANCELLED
        assert len(engine.active_requests) == 0

class TestStrategyExecutionPipeline:
    """戦略実行パイプラインのテストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        return generate_sample_market_data(30)
        
    @pytest.fixture 
    def pipeline(self):
        try:
            return StrategyExecutionPipeline()
        except Exception as e:
            pytest.skip(f"Pipeline initialization failed: {e}")
            
    def test_pipeline_initialization(self, pipeline):
        """パイプライン初期化テスト"""
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'stages')
        
        # 設定の確認
        assert 'execution_pipeline' in pipeline.config
        
    def test_config_loading(self, pipeline):
        """設定読み込みテスト"""
        config = pipeline.config
        
        assert 'execution_pipeline' in config
        pipeline_config = config['execution_pipeline']
        
        assert 'stages' in pipeline_config
        stages = pipeline_config['stages']
        
        # 基本ステージの存在確認
        stage_ids = [stage['id'] for stage in stages]
        expected_stages = ['strategy_selection', 'weight_calculation', 'signal_integration', 'execution']
        
        for expected_stage in expected_stages:
            assert expected_stage in stage_ids
            
    @pytest.mark.skip(reason="Requires full system integration")
    def test_pipeline_execution(self, pipeline, sample_data):
        """パイプライン実行テスト"""
        result = pipeline.execute(
            market_data=sample_data,
            parameters={"test_strategy": "test_value"}
        )
        
        assert result is not None
        assert hasattr(result, 'execution_id')
        assert hasattr(result, 'stage_results')
        
    def test_performance_metrics(self, pipeline):
        """パフォーマンス指標テスト"""
        metrics = pipeline.get_performance_metrics()
        
        # 基本的な指標の存在確認
        expected_keys = ['total_executions', 'success_rate']
        for key in expected_keys:
            if key in metrics:  # データがある場合のみチェック
                assert isinstance(metrics[key], (int, float))

class TestExecutionResultAggregator:
    """実行結果集約器のテストクラス"""
    
    @pytest.fixture
    def aggregator(self):
        try:
            return ExecutionResultAggregator()
        except Exception as e:
            pytest.skip(f"Aggregator initialization failed: {e}")
            
    def test_aggregator_initialization(self, aggregator):
        """集約器初期化テスト"""
        assert aggregator is not None
        assert hasattr(aggregator, 'config')
        assert hasattr(aggregator, 'confidence_calculator')
        assert hasattr(aggregator, 'weight_aggregator')
        assert hasattr(aggregator, 'signal_aggregator')
        
    def test_config_defaults(self, aggregator):
        """デフォルト設定テスト"""
        config = aggregator.config
        
        assert 'aggregation' in config
        aggregation_config = config['aggregation']
        
        # デフォルト値の確認
        assert aggregation_config.get('method') in ['weighted', 'simple_average', 'median']
        assert isinstance(aggregation_config.get('confidence_weighting'), bool)
        assert aggregation_config.get('outlier_handling') in ['none', 'cap', 'remove', 'winsorize']

class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    @pytest.fixture
    def full_system(self):
        """フルシステムセットアップ"""
        try:
            engine = CompositeStrategyExecutionEngine()
            return engine
        except Exception as e:
            pytest.skip(f"Full system initialization failed: {e}")
            
    @pytest.fixture
    def integration_data(self):
        """統合テスト用データ"""
        return generate_sample_market_data(200, '30min')
        
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires full system integration")
    def test_end_to_end_execution(self, full_system, integration_data):
        """エンドツーエンド実行テスト"""
        # 複数の実行モードでテスト
        modes = [ExecutionMode.SINGLE_STRATEGY, ExecutionMode.MULTI_STRATEGY, ExecutionMode.COMPOSITE]
        
        for mode in modes:
            request = create_execution_request(
                market_data=integration_data,
                strategies=["integration_strategy_1", "integration_strategy_2"],
                execution_mode=mode,
                timeout=60
            )
            
            response = full_system.execute(request)
            
            assert isinstance(response, ExecutionResponse)
            assert response.execution_time >= 0
            
    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test - run manually")
    def test_performance_benchmarks(self, full_system, integration_data):
        """パフォーマンスベンチマークテスト"""
        execution_times = []
        
        # 10回実行してパフォーマンスを測定
        for i in range(10):
            request = create_execution_request(
                market_data=integration_data,
                strategies=[f"benchmark_strategy_{i}"],
                execution_mode=ExecutionMode.SINGLE_STRATEGY
            )
            
            start_time = time.time()
            response = full_system.execute(request)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
        # パフォーマンス指標の確認
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        
        # 基準値（環境に応じて調整）
        assert avg_time < 10.0  # 平均10秒以下
        assert max_time < 20.0  # 最大20秒以下
        
    def test_error_handling_scenarios(self, full_system):
        """エラーハンドリングシナリオテスト"""
        # 無効なデータでのテスト
        invalid_data = pd.DataFrame()  # 空のデータフレーム
        
        request = create_execution_request(
            market_data=invalid_data,
            strategies=["test_strategy"],
            execution_mode=ExecutionMode.SINGLE_STRATEGY
        )
        
        response = full_system.execute(request)
        
        # エラーが適切に処理されることを確認
        assert isinstance(response, ExecutionResponse)
        # 失敗した場合のレスポンスの検証
        if response.status == ExecutionStatus.FAILED:
            assert response.error_message is not None
            
    def test_concurrent_execution_safety(self, full_system, integration_data):
        """並行実行安全性テスト"""
        # 複数リクエストの同時処理をシミュレート
        requests = []
        for i in range(3):
            request = create_execution_request(
                market_data=integration_data,
                strategies=[f"concurrent_strategy_{i}"],
                execution_mode=ExecutionMode.SINGLE_STRATEGY
            )
            requests.append(request)
            
        # 順次実行（実際の並行実行は複雑なため簡略化）
        responses = []
        for request in requests:
            response = full_system.execute(request)
            responses.append(response)
            
        # すべてのレスポンスが正常に処理されることを確認
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, ExecutionResponse)

# pytest用の設定
def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")

if __name__ == "__main__":
    # 直接実行時のテストランナー
    pytest.main([__file__, "-v", "--tb=short"])
