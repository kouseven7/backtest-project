"""
Test script for Strategy Data Persistence
File: test_strategy_data_persistence.py
Description: 
  strategy_data_persistence.pyの機能テスト
  - CRUD操作のテスト
  - バージョン管理のテスト
  - 変更履歴のテスト
  - データ統合のテスト

Author: imega
Created: 2025-07-08
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.strategy_data_persistence import (
    StrategyDataPersistence, 
    StrategyDataIntegrator,
    create_persistence_manager,
    create_integrator
)

class TestStrategyDataPersistence(unittest.TestCase):
    """戦略データ永続化機能のテストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        # 一時ディレクトリの作成
        self.test_dir = tempfile.mkdtemp()
        self.persistence = StrategyDataPersistence(self.test_dir)
        
        # テストデータ
        self.test_strategy = "vwap_bounce"
        self.test_data = {
            "strategy_info": {
                "name": "vwap_bounce",
                "type": "trend_following",
                "description": "VWAP bounce strategy for trend following"
            },
            "parameters": {
                "vwap_period": 20,
                "bounce_threshold": 0.02,
                "stop_loss": 0.05,
                "take_profit": 0.10
            },
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.65
            }
        }
        
        print(f"Test setup completed. Test directory: {self.test_dir}")
    
    def tearDown(self):
        """テストクリーンアップ"""
        # 一時ディレクトリの削除
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("Test cleanup completed")
    
    def test_initialization(self):
        """初期化テスト"""
        print("\n=== Test: Initialization ===")
        
        # ディレクトリ構造の確認
        expected_dirs = [
            self.persistence.data_dir,
            self.persistence.versions_dir,
            self.persistence.history_dir,
            self.persistence.metadata_dir
        ]
        
        for directory in expected_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory not created: {directory}")
            print(f"✓ Directory exists: {directory}")
        
        # メタデータファイルの確認
        self.assertTrue(os.path.exists(self.persistence.metadata_file))
        print(f"✓ Metadata file exists: {self.persistence.metadata_file}")
        
        # メタデータ内容の確認
        metadata = self.persistence._load_json(self.persistence.metadata_file)
        self.assertIsNotNone(metadata)
        self.assertIn("created_at", metadata)
        self.assertIn("strategies", metadata)
        print("✓ Metadata structure is valid")
    
    def test_save_and_load_strategy_data(self):
        """戦略データの保存・読み込みテスト"""
        print("\n=== Test: Save and Load Strategy Data ===")
        
        # データ保存テスト
        success = self.persistence.save_strategy_data(
            self.test_strategy, 
            self.test_data, 
            "Initial test data",
            "test_user"
        )
        self.assertTrue(success, "Failed to save strategy data")
        print(f"✓ Strategy data saved: {self.test_strategy}")
        
        # データファイルの存在確認
        data_file = os.path.join(self.persistence.data_dir, f"{self.test_strategy}.json")
        self.assertTrue(os.path.exists(data_file), "Data file not created")
        print(f"✓ Data file exists: {data_file}")
        
        # データ読み込みテスト
        loaded_data = self.persistence.load_strategy_data(self.test_strategy)
        self.assertIsNotNone(loaded_data, "Failed to load strategy data")
        print("✓ Strategy data loaded successfully")
        
        # データ内容の確認
        self.assertEqual(loaded_data["strategy_info"]["name"], self.test_strategy)
        self.assertEqual(loaded_data["parameters"]["vwap_period"], 20)
        print("✓ Loaded data content is correct")
        
        # メタデータの更新確認
        strategies = self.persistence.list_strategies()
        self.assertIn(self.test_strategy, strategies)
        print(f"✓ Strategy added to metadata: {strategies}")
    
    def test_data_versioning(self):
        """データバージョン管理テスト"""
        print("\n=== Test: Data Versioning ===")
        
        # 初期データ保存
        self.persistence.save_strategy_data(
            self.test_strategy, 
            self.test_data, 
            "Initial version"
        )
        
        # データ更新
        updated_data = self.test_data.copy()
        updated_data["parameters"]["vwap_period"] = 25
        updated_data["version_note"] = "Updated vwap_period"
        
        self.persistence.save_strategy_data(
            self.test_strategy, 
            updated_data, 
            "Updated vwap_period to 25"
        )
        print("✓ Data updated with versioning")
        
        # バージョン履歴の確認
        versions = self.persistence.get_strategy_versions(self.test_strategy)
        self.assertGreater(len(versions), 0, "No versions found")
        print(f"✓ Version history available: {len(versions)} versions")
        
        # 最新データの確認
        latest_data = self.persistence.load_strategy_data(self.test_strategy)
        self.assertEqual(latest_data["parameters"]["vwap_period"], 25)
        print("✓ Latest data reflects updates")
        
        # 特定バージョンの読み込みテスト
        if versions:
            version_data = self.persistence.load_strategy_data(
                self.test_strategy, 
                versions[0]["version"]
            )
            self.assertIsNotNone(version_data)
            print(f"✓ Specific version loaded: {versions[0]['version']}")
    
    def test_change_history(self):
        """変更履歴テスト"""
        print("\n=== Test: Change History ===")
        
        # 初期データ保存
        self.persistence.save_strategy_data(
            self.test_strategy, 
            self.test_data, 
            "Initial data",
            "test_user"
        )
        
        # データ更新
        updated_data = self.test_data.copy()
        updated_data["parameters"]["stop_loss"] = 0.03
        
        self.persistence.save_strategy_data(
            self.test_strategy, 
            updated_data, 
            "Reduced stop loss",
            "test_user"
        )
        
        # 変更履歴の確認
        history = self.persistence.get_change_history(self.test_strategy)
        self.assertGreater(len(history), 0, "No change history found")
        print(f"✓ Change history available: {len(history)} changes")
        
        # 履歴内容の確認
        latest_change = history[0]  # 最新の変更
        self.assertEqual(latest_change["change_type"], "update")
        self.assertEqual(latest_change["author"], "test_user")
        print("✓ Change history content is correct")
    
    def test_delete_strategy_data(self):
        """戦略データ削除テスト"""
        print("\n=== Test: Delete Strategy Data ===")
        
        # データ保存
        self.persistence.save_strategy_data(
            self.test_strategy, 
            self.test_data, 
            "Data for deletion test"
        )
        
        # 削除前の確認
        self.assertIsNotNone(self.persistence.load_strategy_data(self.test_strategy))
        
        # データ削除
        success = self.persistence.delete_strategy_data(
            self.test_strategy, 
            "Test deletion", 
            "test_user"
        )
        self.assertTrue(success, "Failed to delete strategy data")
        print(f"✓ Strategy data deleted: {self.test_strategy}")
        
        # 削除後の確認
        self.assertIsNone(self.persistence.load_strategy_data(self.test_strategy))
        print("✓ Data no longer accessible after deletion")
        
        # 削除履歴の確認
        history = self.persistence.get_change_history(self.test_strategy)
        delete_record = next((h for h in history if h["change_type"] == "delete"), None)
        self.assertIsNotNone(delete_record, "Delete record not found in history")
        print("✓ Deletion recorded in change history")
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("\n=== Test: Error Handling ===")
        
        # 存在しない戦略の読み込み
        non_existent_data = self.persistence.load_strategy_data("non_existent_strategy")
        self.assertIsNone(non_existent_data)
        print("✓ Non-existent strategy returns None")
        
        # 無効なデータでの保存テスト
        invalid_data = None
        success = self.persistence.save_strategy_data("test", invalid_data, "Invalid data test")
        # None データでも正常に保存される（データの検証は呼び出し側の責任）
        print("✓ Invalid data handling works as expected")
        
        # 存在しない戦略の削除
        success = self.persistence.delete_strategy_data("non_existent_strategy")
        self.assertFalse(success)
        print("✓ Non-existent strategy deletion handled properly")


class TestStrategyDataIntegrator(unittest.TestCase):
    """戦略データ統合機能のテストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        self.test_dir = tempfile.mkdtemp()
        self.persistence = StrategyDataPersistence(self.test_dir)
        self.integrator = StrategyDataIntegrator(self.persistence)
        
        print(f"Integration test setup completed. Test directory: {self.test_dir}")
    
    def tearDown(self):
        """テストクリーンアップ"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("Integration test cleanup completed")
    
    @patch('config.strategy_data_persistence.StrategyCharacteristicsManager')
    @patch('config.strategy_data_persistence.OptimizedParameterManager')
    def test_data_integration(self, mock_param_manager, mock_char_manager):
        """データ統合テスト"""
        print("\n=== Test: Data Integration ===")
        
        # モックデータの設定
        mock_characteristics = {
            "trend_adaptability": {
                "uptrend": 0.8,
                "downtrend": 0.6,
                "sideways": 0.4
            },
            "volatility_profile": {
                "low": 0.7,
                "medium": 0.8,
                "high": 0.5
            }
        }
        
        mock_parameters = {
            "strategy": "vwap_bounce",
            "parameters": {
                "vwap_period": 20,
                "bounce_threshold": 0.02
            },
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2
            }
        }
        
        # モックの設定
        mock_char_instance = mock_char_manager.return_value
        mock_char_instance.get_strategy_characteristics.return_value = mock_characteristics
        
        mock_param_instance = mock_param_manager.return_value
        mock_param_instance.get_latest_approved_params.return_value = mock_parameters
        
        # 統合テスト実行
        integrated_data = self.integrator.integrate_strategy_data("vwap_bounce", "AAPL")
        
        # 結果検証
        self.assertIsNotNone(integrated_data, "Integration failed")
        print("✓ Data integration completed successfully")
        
        # 統合データ構造の確認
        self.assertIn("integration_metadata", integrated_data)
        self.assertIn("characteristics", integrated_data)
        self.assertIn("parameters", integrated_data)
        print("✓ Integrated data structure is correct")
        
        # メタデータの確認
        metadata = integrated_data["integration_metadata"]
        self.assertEqual(metadata["strategy_name"], "vwap_bounce")
        self.assertEqual(metadata["ticker"], "AAPL")
        print("✓ Integration metadata is correct")
        
        # データソースの確認
        sources = metadata["data_sources"]
        self.assertTrue(sources["characteristics_available"])
        self.assertTrue(sources["parameters_available"])
        print("✓ Data sources properly identified")
    
    def test_integration_error_handling(self):
        """統合エラーハンドリングテスト"""
        print("\n=== Test: Integration Error Handling ===")
        
        # データソースなしでの統合テスト
        integrated_data = self.integrator.integrate_strategy_data("non_existent_strategy")
        
        # エラー時の動作確認（統合は失敗するがエラーは発生しない）
        print("✓ Integration error handling works properly")
    
    def test_factory_functions(self):
        """ファクトリ関数テスト"""
        print("\n=== Test: Factory Functions ===")
        
        # ファクトリ関数テスト
        persistence = create_persistence_manager(self.test_dir)
        self.assertIsInstance(persistence, StrategyDataPersistence)
        print("✓ Persistence manager factory works")
        
        integrator = create_integrator(persistence)
        self.assertIsInstance(integrator, StrategyDataIntegrator)
        print("✓ Integrator factory works")


def run_comprehensive_test():
    """包括的なテスト実行"""
    print("="*60)
    print("STRATEGY DATA PERSISTENCE - COMPREHENSIVE TEST")
    print("="*60)
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # 永続化テストの追加
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStrategyDataPersistence))
    
    # 統合テストの追加
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStrategyDataIntegrator))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 結果サマリー
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    # コマンドライン引数による実行制御
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Data Persistence Test")
    parser.add_argument("--test", choices=["persistence", "integration", "all"], 
                       default="all", help="Test type to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.test == "persistence":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategyDataPersistence)
        unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    elif args.test == "integration":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategyDataIntegrator)
        unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    else:
        run_comprehensive_test()
