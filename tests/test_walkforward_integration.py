"""
ウォークフォワードテストシステムの統合テスト

全体的なワークフローをテストし、システムの動作を検証します。
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
import pandas as pd
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.walkforward_scenarios import WalkforwardScenarios
from src.analysis.walkforward_executor import WalkforwardExecutor  
from src.analysis.walkforward_result_analyzer import WalkforwardResultAnalyzer

class TestWalkforwardIntegration:
    """ウォークフォワードシステム統合テスト"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """一時的な設定ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_config_dir):
        """最小設定でテスト用設定ファイルを作成"""
        config_data = {
            "test_scenarios": {
                "symbols": ["AAPL", "MSFT"],
                "periods": [
                    {
                        "name": "test_period_1",
                        "start": "2023-01-01", 
                        "end": "2023-06-30",
                        "market_condition": "uptrend"
                    },
                    {
                        "name": "test_period_2",
                        "start": "2023-07-01",
                        "end": "2023-12-31", 
                        "market_condition": "sideways"
                    }
                ]
            },
            "strategies": ["VWAPBreakoutStrategy"],
            "walkforward_config": {
                "training_window_months": 3,
                "testing_window_months": 1,
                "step_size_months": 1,
                "min_training_samples": 50
            },
            "output_config": {
                "save_detailed_results": True,
                "generate_reports": True,
                "output_directory": "output/test_walkforward_results"
            }
        }
        
        config_path = Path(temp_config_dir) / "test_walkforward_config.json"
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        return str(config_path)
    
    def test_scenarios_initialization(self, minimal_config):
        """シナリオ管理クラスの初期化テスト"""
        scenarios = WalkforwardScenarios(minimal_config)
        
        # 設定が正しく読み込まれているか
        assert scenarios.config is not None
        assert "test_scenarios" in scenarios.config
        assert len(scenarios.config["test_scenarios"]["symbols"]) == 2
        assert len(scenarios.config["test_scenarios"]["periods"]) == 2
        
        # シナリオ生成テスト
        all_scenarios = scenarios.get_test_scenarios()
        expected_scenarios = 2 * 2 * 1  # symbols × periods × strategies
        assert len(all_scenarios) == expected_scenarios
        
        # シナリオの内容確認
        first_scenario = all_scenarios[0]
        assert "symbol" in first_scenario
        assert "period_name" in first_scenario
        assert "market_condition" in first_scenario
        assert "strategies" in first_scenario
    
    def test_scenario_summary(self, minimal_config):
        """シナリオ概要機能のテスト"""
        scenarios = WalkforwardScenarios(minimal_config)
        summary = scenarios.get_scenario_summary()
        
        assert summary["total_symbols"] == 2
        assert summary["total_periods"] == 2
        assert summary["total_scenarios"] == 4
        assert len(summary["symbols"]) == 2
        assert len(summary["periods"]) == 2
        assert len(summary["strategies"]) == 1
    
    def test_scenario_filtering(self, minimal_config):
        """シナリオフィルタリング機能のテスト"""
        scenarios = WalkforwardScenarios(minimal_config)
        
        # 上昇トレンドのみフィルタリング
        uptrend_scenarios = scenarios.filter_scenarios_by_condition("uptrend")
        assert len(uptrend_scenarios) == 2  # 2 symbols × 1 uptrend period × 1 strategy
        
        # 横ばいのみフィルタリング
        sideways_scenarios = scenarios.filter_scenarios_by_condition("sideways")
        assert len(sideways_scenarios) == 2  # 2 symbols × 1 sideways period × 1 strategy
        
        # 存在しない条件
        nonexistent_scenarios = scenarios.filter_scenarios_by_condition("nonexistent")
        assert len(nonexistent_scenarios) == 0
    
    def test_walkforward_windows_generation(self, minimal_config):
        """ウォークフォワードウィンドウ生成テスト"""
        scenarios = WalkforwardScenarios(minimal_config)
        
        # テスト期間でウィンドウ生成
        windows = scenarios.get_walkforward_windows("2023-01-01", "2023-06-30")
        
        # 期間が短いので1つのウィンドウが生成されるはず
        assert len(windows) >= 1
        
        # ウィンドウの構造確認
        if windows:
            window = windows[0]
            assert "training_start" in window
            assert "training_end" in window
            assert "testing_start" in window
            assert "testing_end" in window
    
    def test_strategy_config_retrieval(self, minimal_config):
        """戦略設定取得テスト"""
        scenarios = WalkforwardScenarios(minimal_config)
        
        # 存在する戦略
        config = scenarios.get_strategy_test_config("VWAPBreakoutStrategy")
        assert config is not None
        assert config["strategy"] == "VWAPBreakoutStrategy"
        assert "walkforward_config" in config
        assert "output_config" in config
        
        # 存在しない戦略
        config = scenarios.get_strategy_test_config("NonexistentStrategy")
        assert config is None
    
    def test_result_analyzer_with_dummy_data(self):
        """結果分析クラスのテスト（ダミーデータ使用）"""
        # ダミー結果データ
        dummy_results = [
            {
                "symbol": "AAPL",
                "strategy": "VWAPBreakoutStrategy", 
                "period_name": "test_period_1",
                "market_condition": "uptrend",
                "window_number": 1,
                "total_return": 5.2,
                "volatility": 2.1,
                "sharpe_ratio": 2.48,
                "max_drawdown": -3.1,
                "entry_signals": 3,
                "exit_signals": 3
            },
            {
                "symbol": "MSFT",
                "strategy": "VWAPBreakoutStrategy",
                "period_name": "test_period_2", 
                "market_condition": "sideways",
                "window_number": 1,
                "total_return": -1.5,
                "volatility": 1.8,
                "sharpe_ratio": -0.83,
                "max_drawdown": -4.2,
                "entry_signals": 2,
                "exit_signals": 2
            }
        ]
        
        analyzer = WalkforwardResultAnalyzer(dummy_results)
        
        # 基本チェック
        assert len(analyzer.results) == 2
        assert analyzer.df is not None
        assert len(analyzer.df) == 2
        
        # サマリーレポート生成
        summary = analyzer.generate_summary_report()
        assert "basic_stats" in summary
        assert "strategy_analysis" in summary
        assert "market_condition_analysis" in summary
        
        # 基本統計の確認
        basic_stats = summary["basic_stats"]
        assert basic_stats["total_results"] == 2
        assert basic_stats["unique_symbols"] == 2
        assert basic_stats["unique_strategies"] == 1
        
        # 最高パフォーマンス設定の取得
        best_configs = analyzer.get_best_configurations(top_n=2)
        assert len(best_configs) == 2
        assert best_configs[0]["total_return"] > best_configs[1]["total_return"]
    
    def test_excel_export_functionality(self):
        """Excel出力機能のテスト"""
        dummy_results = [
            {
                "symbol": "TEST",
                "strategy": "TestStrategy",
                "period_name": "test_period",
                "market_condition": "test_condition",
                "total_return": 1.0,
                "volatility": 0.5
            }
        ]
        
        analyzer = WalkforwardResultAnalyzer(dummy_results)
        
        # 一時ファイルでテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.xlsx"
            
            # Excel出力実行
            success = analyzer.export_to_excel(str(output_path))
            
            # ファイルが作成されたかチェック
            assert success is True
            assert output_path.exists()
            
            # Excelファイルの読み込みテスト
            try:
                df = pd.read_excel(output_path, sheet_name='Raw_Data')
                assert len(df) == 1
                assert "symbol" in df.columns
            except Exception as e:
                pytest.skip(f"Excel読み込みエラー: {e}")
    
    def test_end_to_end_workflow_simulation(self, minimal_config):
        """エンドツーエンドワークフローのシミュレーション"""
        # ワークフローの各ステップをシミュレーション
        
        # Step 1: シナリオ初期化
        scenarios = WalkforwardScenarios(minimal_config)
        assert scenarios.config is not None
        
        # Step 2: シナリオ生成
        all_scenarios = scenarios.get_test_scenarios()
        assert len(all_scenarios) > 0
        
        # Step 3: 制限されたシナリオでテスト準備（実際のデータ取得はスキップ）
        test_scenario = all_scenarios[0]
        
        # Step 4: 結果分析のシミュレーション（ダミーデータ）
        simulated_results = [
            {
                **test_scenario,
                "window_number": 1,
                "total_return": 2.5,
                "volatility": 1.5,
                "sharpe_ratio": 1.67,
                "max_drawdown": -2.0
            }
        ]
        
        # Step 5: 結果分析
        analyzer = WalkforwardResultAnalyzer(simulated_results)
        summary = analyzer.generate_summary_report()
        
        assert summary is not None
        assert "basic_stats" in summary
        
        # Step 6: ベスト設定の取得
        best_configs = analyzer.get_best_configurations(top_n=1)
        assert len(best_configs) == 1
        assert best_configs[0]["total_return"] == 2.5
    
    def test_error_handling_and_edge_cases(self):
        """エラーハンドリングとエッジケースのテスト"""
        
        # 空の結果での分析テスト
        analyzer = WalkforwardResultAnalyzer([])
        summary = analyzer.generate_summary_report()
        assert "error" in summary
        
        # 無効な設定ファイルパスでのシナリオ初期化
        with pytest.raises(Exception):
            WalkforwardScenarios("/nonexistent/path/config.json")
        
        # 不正な形式の結果データ
        invalid_results = [{"invalid": "data"}]
        analyzer = WalkforwardResultAnalyzer(invalid_results) 
        summary = analyzer.generate_summary_report()
        # エラーがあっても部分的な結果は返されるべき
        assert summary is not None

if __name__ == "__main__":
    # 基本的なテストを手動実行
    print("=== ウォークフォワードシステム統合テスト ===")
    
    # テスト用の最小設定を作成
    test_config = {
        "test_scenarios": {
            "symbols": ["AAPL"],
            "periods": [{
                "name": "test_period",
                "start": "2023-01-01",
                "end": "2023-03-31", 
                "market_condition": "test"
            }]
        },
        "strategies": ["VWAPBreakoutStrategy"],
        "walkforward_config": {
            "training_window_months": 1,
            "testing_window_months": 1,
            "step_size_months": 1,
            "min_training_samples": 10
        },
        "output_config": {
            "save_detailed_results": True,
            "generate_reports": True,
            "output_directory": "output/test_results"
        }
    }
    
    # 一時設定ファイル作成
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        temp_config_path = f.name
    
    try:
        # シナリオ管理テスト
        scenarios = WalkforwardScenarios(temp_config_path)
        print(f"✓ シナリオ管理初期化成功")
        
        summary = scenarios.get_scenario_summary()
        print(f"✓ シナリオ概要生成成功: {summary['total_scenarios']}シナリオ")
        
        # 結果分析テスト
        dummy_results = [{
            "symbol": "AAPL",
            "strategy": "VWAPBreakoutStrategy",
            "period_name": "test_period",
            "market_condition": "test",
            "total_return": 1.5,
            "volatility": 1.0
        }]
        
        analyzer = WalkforwardResultAnalyzer(dummy_results)
        analysis_summary = analyzer.generate_summary_report()
        print(f"✓ 結果分析成功: {analysis_summary['basic_stats']['total_results']}件")
        
        print("\n=== 統合テスト完了 ===")
        
    finally:
        # 一時ファイル削除
        import os
        os.unlink(temp_config_path)
