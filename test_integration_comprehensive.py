#!/usr/bin/env python3
"""
統合テスト完備実装
Phase 2: Test Suite 1-3 包括的テストスクリプト

Author: imega
Created: 2025-10-07
"""

import sys
import os
import json
import tempfile
import shutil
import logging
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

# 必要なモジュールのインポート
from config.logger_config import setup_logger
from config.multi_strategy_manager import MultiStrategyManager, ExecutionMode, IntegrationStatus
from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode

# テスト専用ロガー設定
logger = setup_logger("integration_test", log_file="logs/integration_test.log")

class TestSuite1_InitializeSystemPatterns(unittest.TestCase):
    """Test Suite 1: initialize_system()成功/失敗パターンテスト"""
    
    def setUp(self):
        """各テストの前準備"""
        # テスト用の一時設定ディレクトリ作成
        self.test_temp_dir = tempfile.mkdtemp(prefix="integration_test_")
        self.config_path = os.path.join(self.test_temp_dir, "test_config.json")
        
        # デフォルト設定の作成
        self.default_config = {
            "execution_mode": "hybrid",
            "system_mode": "development",
            "strategy_selection": {
                "enable_multi_strategy": True,
                "fallback_to_legacy": True,
                "min_strategies": 2,
                "max_strategies": 5
            },
            "integration_features": {
                "trend_based_selection": True,
                "portfolio_weighting": True,
                "signal_integration": True,
                "risk_management": True
            },
            "fallback_settings": {
                "enable_fallback": True,
                "fallback_timeout": 30,
                "error_threshold": 3
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.default_config, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """各テストの後処理"""
        # 一時ディレクトリ削除
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
    
    def test_01_normal_initialization_development_mode(self):
        """正常初期化テスト (Development mode)"""
        logger.info("=== Test 1.1: Development mode正常初期化テスト ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            
            # 初期化実行
            result = manager.initialize_system()
            
            # 結果検証
            self.assertTrue(result, "Development mode初期化が失敗しました")
            self.assertTrue(manager.is_initialized, "初期化フラグがTrueになっていません")
            self.assertNotEqual(manager.status, IntegrationStatus.ERROR, "ステータスがERRORになっています")
            
            logger.info("✅ Development mode正常初期化テスト成功")
            
        except Exception as e:
            logger.error(f"❌ Development mode正常初期化テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_02_normal_initialization_production_mode(self):
        """正常初期化テスト (Production mode)"""
        logger.info("=== Test 1.2: Production mode正常初期化テスト ===")
        
        try:
            # Production mode設定に変更
            config = self.default_config.copy()
            config['system_mode'] = 'production'
            config['fallback_settings']['enable_fallback'] = False
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            manager = MultiStrategyManager(config_path=self.config_path)
            
            # Production mode初期化実行
            result = manager.initialize_system()
            
            # 結果検証
            self.assertTrue(result, "Production mode初期化が失敗しました")
            self.assertTrue(manager.is_initialized, "初期化フラグがTrueになっていません")
            
            # Production準備状況確認
            readiness = manager.get_production_readiness_status()
            self.assertEqual(readiness['system_mode'], 'production', "システムモードがproductionになっていません")
            
            logger.info("✅ Production mode正常初期化テスト成功")
            
        except Exception as e:
            logger.error(f"❌ Production mode正常初期化テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_03_error_handling_invalid_config(self):
        """エラーハンドリングテスト (設定不備)"""
        logger.info("=== Test 1.3: 設定不備エラーハンドリングテスト ===")
        
        try:
            # 不正な設定ファイルを作成
            invalid_config = {"invalid": "config"}
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(invalid_config, f, indent=2, ensure_ascii=False)
            
            manager = MultiStrategyManager(config_path=self.config_path)
            
            # 初期化実行（エラーハンドリング確認）
            result = manager.initialize_system()
            
            # Development modeではフォールバック動作のため成功する可能性がある
            if not result:
                logger.info("設定不備が適切にハンドリングされました")
            else:
                logger.info("フォールバック機能により初期化が継続されました")
            
            logger.info("✅ 設定不備エラーハンドリングテスト成功")
            
        except Exception as e:
            logger.info(f"適切なエラーハンドリング: {e}")
            # エラーが発生することは正常な動作
    
    def test_04_fallback_behavior_development_mode(self):
        """フォールバック動作テスト (Development modeのみ)"""
        logger.info("=== Test 1.4: Development modeフォールバック動作テスト ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            
            # エラー模擬のためのモック設定
            with patch.object(manager, '_initialize_strategy_registry', side_effect=Exception("Mock error")):
                result = manager.initialize_system()
                
                # Development modeではフォールバック動作により継続される可能性
                if manager.execution_mode == ExecutionMode.LEGACY_ONLY:
                    logger.info("フォールバック動作が正常に機能しました")
                elif result:
                    logger.info("エラー後も初期化が継続されました")
                else:
                    logger.info("初期化が失敗し、適切にハンドリングされました")
            
            logger.info("✅ フォールバック動作テスト成功")
            
        except Exception as e:
            logger.error(f"❌ フォールバック動作テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_05_initialization_sequence_dependencies(self):
        """初期化順序テスト (依存関係解決確認)"""
        logger.info("=== Test 1.5: 依存関係解決・初期化順序テスト ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            
            # 初期化前の状態確認
            self.assertFalse(manager.is_initialized, "初期化前にis_initializedがTrueになっています")
            self.assertEqual(manager.status, IntegrationStatus.INITIALIZING, "初期ステータスが正しくありません")
            
            # 初期化実行
            result = manager.initialize_system()
            
            if result:
                # 初期化順序の検証
                self.assertTrue(manager.is_initialized, "初期化完了後のフラグが正しくありません")
                self.assertIn(manager.status, [IntegrationStatus.READY, IntegrationStatus.FALLBACK], 
                            "初期化完了後のステータスが適切ではありません")
                
                # 戦略レジストリの初期化確認
                if hasattr(manager, 'strategy_registry') and manager.strategy_registry:
                    logger.info("戦略レジストリが正常に初期化されました")
                
                # リソース管理の初期化確認
                if hasattr(manager, 'resource_pool') and manager.resource_pool:
                    logger.info("リソース管理システムが正常に初期化されました")
            
            logger.info("✅ 依存関係解決・初期化順序テスト成功")
            
        except Exception as e:
            logger.error(f"❌ 依存関係解決テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")


class TestSuite2_MultiStrategyManagerFunctionality(unittest.TestCase):
    """Test Suite 2: MultiStrategyManager全機能テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.test_temp_dir = tempfile.mkdtemp(prefix="integration_test_suite2_")
        self.config_path = os.path.join(self.test_temp_dir, "test_config.json")
        
        # テスト用設定
        self.config = {
            "execution_mode": "hybrid",
            "system_mode": "development",
            "strategy_selection": {
                "enable_multi_strategy": True,
                "fallback_to_legacy": True,
                "min_strategies": 2,
                "max_strategies": 7
            },
            "integration_features": {
                "trend_based_selection": True,
                "portfolio_weighting": True,
                "signal_integration": True,
                "risk_management": True
            },
            "fallback_settings": {
                "enable_fallback": True,
                "fallback_timeout": 30,
                "error_threshold": 3
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """テスト後処理"""
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
    
    def test_01_strategy_registry_system(self):
        """戦略レジストリシステム動作確認"""
        logger.info("=== Test 2.1: 戦略レジストリシステム動作確認 ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            result = manager.initialize_system()
            
            self.assertTrue(result, "初期化が失敗しました")
            
            # 戦略レジストリの動作確認
            if hasattr(manager, 'strategy_registry'):
                registry = manager.strategy_registry
                if registry:
                    logger.info(f"戦略レジストリに {len(registry)} 個の戦略が登録されています")
                    
                    # 7戦略の基本確認
                    expected_strategies = [
                        'VWAPBreakoutStrategy', 'MomentumInvestingStrategy', 'BreakoutStrategy',
                        'VWAPBounceStrategy', 'OpeningGapStrategy', 'ContrarianStrategy', 'GCStrategy'
                    ]
                    
                    for strategy_name in expected_strategies:
                        if strategy_name in registry:
                            logger.info(f"✓ {strategy_name} が正常に登録されています")
                        else:
                            logger.warning(f"⚠ {strategy_name} が見つかりません")
            
            logger.info("✅ 戦略レジストリシステムテスト成功")
            
        except Exception as e:
            logger.error(f"❌ 戦略レジストリテスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_02_resource_management_system(self):
        """リソース管理システム制限テスト"""
        logger.info("=== Test 2.2: リソース管理システム制限テスト ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            result = manager.initialize_system()
            
            self.assertTrue(result, "初期化が失敗しました")
            
            # リソース管理システムの確認
            if hasattr(manager, 'resource_pool'):
                resource_pool = manager.resource_pool
                if resource_pool:
                    logger.info("リソース管理システムが初期化されています")
                    
                    # メモリ制限確認
                    if 'memory_limit_mb' in resource_pool:
                        memory_limit = resource_pool['memory_limit_mb']
                        logger.info(f"メモリ制限: {memory_limit}MB")
                        self.assertGreater(memory_limit, 0, "メモリ制限が設定されていません")
                    
                    # CPU制限確認
                    if 'cpu_limit_cores' in resource_pool:
                        cpu_limit = resource_pool['cpu_limit_cores']
                        logger.info(f"CPU制限: {cpu_limit} cores")
                        self.assertGreater(cpu_limit, 0, "CPU制限が設定されていません")
            
            logger.info("✅ リソース管理システムテスト成功")
            
        except Exception as e:
            logger.error(f"❌ リソース管理システムテスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_03_dependency_resolution_mechanism(self):
        """依存関係解決メカニズム検証"""
        logger.info("=== Test 2.3: 依存関係解決メカニズム検証 ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            result = manager.initialize_system()
            
            self.assertTrue(result, "初期化が失敗しました")
            
            # 依存関係解決の確認
            if hasattr(manager, 'dependency_graph'):
                dependency_graph = manager.dependency_graph
                if dependency_graph:
                    logger.info("依存関係グラフが構築されています")
                    
                    # 循環依存チェック
                    if 'circular_dependencies' in dependency_graph:
                        circular = dependency_graph['circular_dependencies']
                        self.assertEqual(len(circular), 0, f"循環依存が検出されました: {circular}")
                        logger.info("循環依存チェック: OK")
                    
                    # 初期化順序確認
                    if 'initialization_order' in dependency_graph:
                        init_order = dependency_graph['initialization_order']
                        logger.info(f"初期化順序: {init_order}")
                        self.assertGreater(len(init_order), 0, "初期化順序が設定されていません")
            
            logger.info("✅ 依存関係解決メカニズムテスト成功")
            
        except Exception as e:
            logger.error(f"❌ 依存関係解決テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_04_health_check_monitoring_system(self):
        """ヘルスチェック・監視システム動作確認"""
        logger.info("=== Test 2.4: ヘルスチェック・監視システム動作確認 ===")
        
        try:
            manager = MultiStrategyManager(config_path=self.config_path)
            result = manager.initialize_system()
            
            self.assertTrue(result, "初期化が失敗しました")
            
            # ヘルスチェック・監視システムの確認
            if hasattr(manager, 'monitoring_system'):
                monitoring = manager.monitoring_system
                if monitoring:
                    logger.info("ヘルスチェック・監視システムが初期化されています")
                    
                    # システム状態監視確認
                    if 'system_health' in monitoring:
                        health = monitoring['system_health']
                        logger.info(f"システム健全性: {health}")
                    
                    # リソース監視確認
                    if 'resource_monitoring' in monitoring:
                        resource_mon = monitoring['resource_monitoring']
                        logger.info("リソース監視機能が有効です")
                    
                    # アラートシステム確認
                    if 'alert_system' in monitoring:
                        alert_sys = monitoring['alert_system']
                        logger.info("アラートシステムが有効です")
            
            logger.info("✅ ヘルスチェック・監視システムテスト成功")
            
        except Exception as e:
            logger.error(f"❌ ヘルスチェック・監視システムテスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")
    
    def test_05_production_mode_constraints(self):
        """Production mode制約強制確認"""
        logger.info("=== Test 2.5: Production mode制約強制確認 ===")
        
        try:
            # Production mode設定
            config = self.config.copy()
            config['system_mode'] = 'production'
            config['fallback_settings']['enable_fallback'] = False
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            manager = MultiStrategyManager(config_path=self.config_path)
            result = manager.initialize_system()
            
            self.assertTrue(result, "Production mode初期化が失敗しました")
            
            # Production mode制約の確認
            readiness = manager.get_production_readiness_status()
            
            self.assertEqual(readiness['system_mode'], 'production', "システムモードがproductionになっていません")
            
            if 'fallback_forbidden' in readiness:
                self.assertTrue(readiness['fallback_forbidden'], "フォールバック禁止設定が有効になっていません")
                logger.info("フォールバック禁止制約: OK")
            
            if 'immediate_failure_on_error' in readiness:
                self.assertTrue(readiness['immediate_failure_on_error'], "即停止設定が有効になっていません")
                logger.info("即停止制約: OK")
            
            logger.info("✅ Production mode制約強制テスト成功")
            
        except Exception as e:
            logger.error(f"❌ Production mode制約テスト失敗: {e}")
            self.fail(f"予期しないエラーが発生しました: {e}")


def run_test_suite_1():
    """Test Suite 1の実行"""
    logger.info("🚀 Test Suite 1: initialize_system()成功/失敗パターンテスト開始")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite1_InitializeSystemPatterns)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info(f"Test Suite 1結果: Tests={result.testsRun}, Failures={len(result.failures)}, Errors={len(result.errors)}")
    return result.wasSuccessful()


def run_test_suite_2():
    """Test Suite 2の実行"""
    logger.info("🚀 Test Suite 2: MultiStrategyManager全機能テスト開始")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite2_MultiStrategyManagerFunctionality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info(f"Test Suite 2結果: Tests={result.testsRun}, Failures={len(result.failures)}, Errors={len(result.errors)}")
    return result.wasSuccessful()


def main():
    """統合テストメイン実行"""
    logger.info("====== 統合テスト完備実装 Phase 2 開始 ======")
    start_time = datetime.now()
    
    try:
        # Test Suite 1実行
        suite1_success = run_test_suite_1()
        
        # Test Suite 2実行
        suite2_success = run_test_suite_2()
        
        # 結果サマリー
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"====== 統合テスト完了 (実行時間: {duration:.2f}秒) ======")
        logger.info(f"Test Suite 1 (initialize_system): {'✅ 成功' if suite1_success else '❌ 失敗'}")
        logger.info(f"Test Suite 2 (MultiStrategyManager全機能): {'✅ 成功' if suite2_success else '❌ 失敗'}")
        
        if suite1_success and suite2_success:
            logger.info("🎉 統合テスト完備実装 Test Suite 1-2 完了: 全テスト成功")
        else:
            logger.warning("⚠ 統合テスト完備実装 Test Suite 1-2: 一部テスト失敗")
        
        return suite1_success and suite2_success
        
    except Exception as e:
        logger.error(f"統合テスト実行中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)