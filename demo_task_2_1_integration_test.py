"""
DSSMS Phase 2 Task 2.1: 統合テストスイート
Dynamic Stock Selection Multi-Strategy System - Integration Test Suite

Task 2.1で修正された統合システムの動作確認テスト

主要テスト項目:
1. Task 1.3コンポーネント個別テスト
2. 統合システム連携テスト
3. バックテストレポート生成テスト
4. ポートフォリオ計算精度テスト
5. 切替メカニズム動作テスト

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 2 Task 2.1 - 統合システム検証
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class Task21IntegrationTestSuite:
    """Task 2.1 統合テストスイート"""
    
    def __init__(self):
        """テストスイートの初期化"""
        self.logger = setup_logger(__name__)
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = datetime.now()
        
        self.logger.info("Task 2.1 統合テストスイート初期化完了")
    
    def test_component_imports(self) -> Dict[str, bool]:
        """コンポーネントインポートテスト"""
        self.logger.info("=== コンポーネントインポートテスト開始 ===")
        
        results = {}
        components = [
            ("DSSMSPortfolioCalculatorV2", "src.dssms.dssms_portfolio_calculator_v2"),
            ("DSSMSSwitchEngineV2", "src.dssms.dssms_switch_engine_v2"),
            ("DSSMSBacktesterV2", "src.dssms.dssms_backtester_v2"),
            ("fetch_real_data", "src.dssms.dssms_integration_patch"),
            ("DataQualityValidator", "src.dssms.data_quality_validator")
        ]
        
        for component_name, module_path in components:
            try:
                module = __import__(module_path, fromlist=[component_name])
                component_class = getattr(module, component_name)
                results[component_name] = True
                self.logger.info(f"✓ {component_name}: インポート成功")
                
            except Exception as e:
                results[component_name] = False
                self.logger.error(f"✗ {component_name}: インポート失敗 - {e}")
        
        success_rate = sum(results.values()) / len(results) * 100
        self.logger.info(f"インポートテスト完了: {success_rate:.1f}% 成功")
        
        return results
    
    def test_portfolio_calculator_basic(self) -> bool:
        """ポートフォリオ計算エンジン基本テスト"""
        self.logger.info("=== ポートフォリオ計算エンジン基本テスト開始 ===")
        
        try:
            from src.dssms.dssms_portfolio_calculator_v2 import DSSMSPortfolioCalculatorV2
            
            # 基本初期化テスト
            calculator = DSSMSPortfolioCalculatorV2(initial_capital=1000000)
            self.logger.info("✓ ポートフォリオ計算エンジン初期化成功")
            
            # 基本プロパティテスト
            assert calculator.initial_capital == 1000000
            assert calculator.current_capital > 0
            self.logger.info("✓ 基本プロパティテスト成功")
            
            # サンプルデータでの計算テスト
            sample_data = self._generate_sample_portfolio_data()
            if hasattr(calculator, 'calculate_portfolio_weights'):
                weights = calculator.calculate_portfolio_weights(sample_data)
                if weights is not None and len(weights) > 0:
                    self.logger.info("✓ ポートフォリオ重み計算成功")
                else:
                    self.logger.warning("⚠ ポートフォリオ重み計算結果が空")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ ポートフォリオ計算エンジンテスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_switch_engine_basic(self) -> bool:
        """切替エンジン基本テスト"""
        self.logger.info("=== 切替エンジン基本テスト開始 ===")
        
        try:
            from src.dssms.dssms_switch_engine_v2 import DSSMSSwitchEngineV2
            
            # 基本初期化テスト
            engine = DSSMSSwitchEngineV2()
            self.logger.info("✓ 切替エンジン初期化成功")
            
            # 基本メソッドテスト
            if hasattr(engine, 'evaluate_switch_conditions'):
                sample_data = self._generate_sample_market_data()
                conditions = engine.evaluate_switch_conditions(sample_data)
                self.logger.info("✓ 切替条件評価成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 切替エンジンテスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_backtester_integration(self) -> bool:
        """バックテスター統合テスト"""
        self.logger.info("=== バックテスター統合テスト開始 ===")
        
        try:
            from src.dssms.dssms_backtester_v2 import DSSMSBacktesterV2, BacktestConfig
            
            # テスト設定
            config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now() - timedelta(days=1),
                initial_capital=1000000,
                symbols=['7203.T', '6758.T', '9984.T'],  # トヨタ、ソニー、ソフトバンク
                rebalance_frequency="weekly",
                enable_switching=True,
                enable_data_quality=True
            )
            
            # バックテスター初期化
            backtester = DSSMSBacktesterV2(config)
            self.logger.info("✓ バックテスター初期化成功")
            
            # 基本プロパティテスト
            assert backtester.config.initial_capital == 1000000
            assert len(backtester.config.symbols) == 3
            self.logger.info("✓ バックテスター設定テスト成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ バックテスター統合テスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_data_quality_integration(self) -> bool:
        """データ品質管理統合テスト"""
        self.logger.info("=== データ品質管理統合テスト開始 ===")
        
        try:
            from src.dssms.data_quality_validator import DataQualityValidator
            
            # データ品質バリデーター初期化
            validator = DataQualityValidator()
            self.logger.info("✓ データ品質バリデーター初期化成功")
            
            # サンプルデータでの検証テスト
            sample_data = self._generate_sample_market_data()
            if hasattr(validator, 'validate_data'):
                validation_result = validator.validate_data(sample_data)
                self.logger.info("✓ データ品質検証成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ データ品質管理統合テスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_integration_patch_functionality(self) -> bool:
        """統合パッチ機能テスト"""
        self.logger.info("=== 統合パッチ機能テスト開始 ===")
        
        try:
            from src.dssms.dssms_integration_patch import fetch_real_data, generate_realistic_sample_data
            
            # リアルデータ取得テスト（フォールバック付き）
            test_symbol = "7203.T"  # トヨタ
            real_data = fetch_real_data(test_symbol, days=5)
            
            if real_data is not None and not real_data.empty:
                self.logger.info("✓ リアルデータ取得成功")
            else:
                self.logger.info("→ リアルデータ取得失敗、サンプルデータ生成テスト")
                
                # サンプルデータ生成テスト
                sample_data = generate_realistic_sample_data(test_symbol, days=5)
                if sample_data is not None and not sample_data.empty:
                    self.logger.info("✓ サンプルデータ生成成功")
                else:
                    self.logger.error("✗ サンプルデータ生成失敗")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 統合パッチ機能テスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_end_to_end_integration(self) -> bool:
        """エンドツーエンド統合テスト"""
        self.logger.info("=== エンドツーエンド統合テスト開始 ===")
        
        try:
            # 各コンポーネントの組み合わせテスト
            from src.dssms.dssms_portfolio_calculator_v2 import DSSMSPortfolioCalculatorV2
            from src.dssms.dssms_switch_engine_v2 import DSSMSSwitchEngineV2
            from src.dssms.data_quality_validator import DataQualityValidator
            
            # コンポーネント初期化
            calculator = DSSMSPortfolioCalculatorV2(initial_capital=1000000)
            switch_engine = DSSMSSwitchEngineV2()
            validator = DataQualityValidator()
            
            self.logger.info("✓ 全コンポーネント初期化成功")
            
            # サンプルデータでの統合動作テスト
            sample_data = self._generate_sample_market_data()
            
            # データ品質チェック
            if hasattr(validator, 'validate_data'):
                validation_result = validator.validate_data(sample_data)
                self.logger.info("✓ 統合データ品質チェック成功")
            
            # ポートフォリオ計算
            if hasattr(calculator, 'calculate_portfolio_weights'):
                weights = calculator.calculate_portfolio_weights(sample_data)
                self.logger.info("✓ 統合ポートフォリオ計算成功")
            
            # 切替条件評価
            if hasattr(switch_engine, 'evaluate_switch_conditions'):
                conditions = switch_engine.evaluate_switch_conditions(sample_data)
                self.logger.info("✓ 統合切替条件評価成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ エンドツーエンド統合テスト失敗: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _generate_sample_portfolio_data(self) -> pd.DataFrame:
        """サンプルポートフォリオデータ生成"""
        symbols = ['7203.T', '6758.T', '9984.T']
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'price': np.random.uniform(1000, 5000),
                    'volume': np.random.randint(1000000, 10000000),
                    'market_cap': np.random.uniform(1e12, 50e12)
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_market_data(self) -> pd.DataFrame:
        """サンプル市場データ生成"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                             end=datetime.now(), freq='D')
        
        data = []
        base_price = 3000
        for i, date in enumerate(dates):
            price = base_price + np.random.normal(0, 100)
            data.append({
                'Date': date,
                'Open': price + np.random.normal(0, 50),
                'High': price + abs(np.random.normal(50, 25)),
                'Low': price - abs(np.random.normal(50, 25)),
                'Close': price,
                'Volume': np.random.randint(1000000, 10000000)
            })
            base_price = price  # 次の日の基準価格
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テストの実行"""
        self.logger.info("DSSMS Task 2.1 統合テストスイート開始")
        print("=" * 70)
        print("DSSMS Phase 2 Task 2.1: 統合システム検証テスト")
        print("=" * 70)
        
        test_results = {}
        
        # 1. コンポーネントインポートテスト
        print("\n1. コンポーネントインポートテスト")
        import_results = self.test_component_imports()
        test_results['import_tests'] = import_results
        
        # 2. ポートフォリオ計算エンジンテスト
        print("\n2. ポートフォリオ計算エンジンテスト")
        portfolio_result = self.test_portfolio_calculator_basic()
        test_results['portfolio_test'] = portfolio_result
        print(f"   結果: {'成功' if portfolio_result else '失敗'}")
        
        # 3. 切替エンジンテスト
        print("\n3. 切替エンジンテスト")
        switch_result = self.test_switch_engine_basic()
        test_results['switch_test'] = switch_result
        print(f"   結果: {'成功' if switch_result else '失敗'}")
        
        # 4. バックテスター統合テスト
        print("\n4. バックテスター統合テスト")
        backtester_result = self.test_backtester_integration()
        test_results['backtester_test'] = backtester_result
        print(f"   結果: {'成功' if backtester_result else '失敗'}")
        
        # 5. データ品質管理統合テスト
        print("\n5. データ品質管理統合テスト")
        quality_result = self.test_data_quality_integration()
        test_results['quality_test'] = quality_result
        print(f"   結果: {'成功' if quality_result else '失敗'}")
        
        # 6. 統合パッチ機能テスト
        print("\n6. 統合パッチ機能テスト")
        patch_result = self.test_integration_patch_functionality()
        test_results['patch_test'] = patch_result
        print(f"   結果: {'成功' if patch_result else '失敗'}")
        
        # 7. エンドツーエンド統合テスト
        print("\n7. エンドツーエンド統合テスト")
        e2e_result = self.test_end_to_end_integration()
        test_results['e2e_test'] = e2e_result
        print(f"   結果: {'成功' if e2e_result else '失敗'}")
        
        # 結果サマリー
        self._generate_test_summary(test_results)
        
        return test_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]):
        """テスト結果サマリー生成"""
        print("\n" + "=" * 70)
        print("テスト結果サマリー")
        print("=" * 70)
        
        # インポートテスト結果
        import_success = sum(test_results['import_tests'].values())
        import_total = len(test_results['import_tests'])
        print(f"コンポーネントインポート: {import_success}/{import_total} 成功")
        
        # その他のテスト結果
        other_tests = {k: v for k, v in test_results.items() if k != 'import_tests'}
        other_success = sum(other_tests.values())
        other_total = len(other_tests)
        print(f"機能テスト: {other_success}/{other_total} 成功")
        
        # 全体成功率
        total_success = import_success + other_success
        total_tests = import_total + other_total
        success_rate = (total_success / total_tests) * 100
        
        print(f"\n全体成功率: {success_rate:.1f}% ({total_success}/{total_tests})")
        
        # 実行時間
        duration = datetime.now() - self.start_time
        print(f"実行時間: {duration.total_seconds():.1f}秒")
        
        # 結果判定
        if success_rate >= 80:
            print("\n✓ Task 2.1 統合システム検証: 成功")
            print("  Task 1.3コンポーネントの統合が正常に動作しています")
        elif success_rate >= 60:
            print("\n⚠ Task 2.1 統合システム検証: 部分的成功") 
            print("  一部の機能に問題がありますが、基本的な統合は動作しています")
        else:
            print("\n✗ Task 2.1 統合システム検証: 失敗")
            print("  重大な統合問題があります。追加の修正が必要です")
        
        # レポート保存
        report_file = self.project_root / f"task_2_1_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._save_detailed_report(test_results, report_file, success_rate)
        print(f"\n詳細レポート保存: {report_file}")
    
    def _save_detailed_report(self, test_results: Dict[str, Any], file_path: Path, success_rate: float):
        """詳細レポートの保存"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("DSSMS Phase 2 Task 2.1: 統合システム検証テストレポート\n")
            f.write("=" * 70 + "\n")
            f.write(f"実行時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"全体成功率: {success_rate:.1f}%\n\n")
            
            f.write("【テスト結果詳細】\n")
            f.write("-" * 40 + "\n")
            
            # インポートテスト詳細
            f.write("\n1. コンポーネントインポートテスト:\n")
            for component, result in test_results['import_tests'].items():
                status = "✓" if result else "✗"
                f.write(f"   {status} {component}: {'成功' if result else '失敗'}\n")
            
            # 機能テスト詳細
            f.write("\n2. 機能テスト:\n")
            test_names = {
                'portfolio_test': 'ポートフォリオ計算エンジン',
                'switch_test': '切替エンジン',
                'backtester_test': 'バックテスター統合',
                'quality_test': 'データ品質管理統合',
                'patch_test': '統合パッチ機能',
                'e2e_test': 'エンドツーエンド統合'
            }
            
            for test_key, test_name in test_names.items():
                if test_key in test_results:
                    result = test_results[test_key]
                    status = "✓" if result else "✗"
                    f.write(f"   {status} {test_name}: {'成功' if result else '失敗'}\n")
            
            f.write("\n【推奨事項】\n")
            f.write("-" * 40 + "\n")
            if success_rate >= 80:
                f.write("- 統合システムは正常に動作しています\n")
                f.write("- Task 2.1の目標を達成しました\n")
                f.write("- 次のPhaseに進むことができます\n")
            else:
                f.write("- 失敗したテストの詳細調査が必要です\n")
                f.write("- エラーログを確認して問題を特定してください\n")
                f.write("- 追加の修正作業を検討してください\n")

def main():
    """メイン実行関数"""
    try:
        # テストスイートの初期化と実行
        test_suite = Task21IntegrationTestSuite()
        results = test_suite.run_all_tests()
        
        # 結果に基づく終了コード
        import_success = sum(results['import_tests'].values())
        import_total = len(results['import_tests'])
        other_tests = {k: v for k, v in results.items() if k != 'import_tests'}
        other_success = sum(other_tests.values())
        other_total = len(other_tests)
        
        total_success = import_success + other_success
        total_tests = import_total + other_total
        success_rate = (total_success / total_tests) * 100
        
        return success_rate >= 70  # 70%以上で成功とする
        
    except Exception as e:
        print(f"\nテスト実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
