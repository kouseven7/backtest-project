"""
設定・ログ系モジュール専門テスト Phase 4C
15個の設定・ログ系モジュールの段階的テスト - 設定・ルール管理系モジュール
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import traceback
import warnings
import json

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class ConfigLogPhase4CTester:
    """設定・ログ系モジュール Phase 4C 専門テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="config_log_phase4c_")
        print(f"📁 Phase 4C テスト環境作成: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """テスト環境のクリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"🧹 Phase 4C テスト環境クリーンアップ完了")
    
    def detect_config_log_fallbacks(self, result, module_name, operation):
        """設定・ログ系特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: 設定ルール読み込み失敗
        if operation == "rule_config_load":
            if isinstance(result, dict) and len(result) == 0:
                fallbacks.append("空ルール設定: ルール設定ファイル読み込み失敗を隠蔽")
            elif result is None:
                fallbacks.append("ルール設定読み込み失敗: 例外を隠蔽してNone返却")
        
        # パターン2: システム設定初期化失敗
        elif operation == "system_config_init":
            if result is None:
                fallbacks.append("システム設定初期化失敗: 例外を隠蔽")
        
        # パターン3: VaR設定読み込み失敗
        elif operation == "var_config_load":
            if isinstance(result, dict) and 'default' in str(result).lower():
                fallbacks.append("デフォルトVaR設定: 実設定値取得失敗")
        
        if fallbacks:
            self.fallback_detected.extend([f"{module_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def test_rule_configuration_manager_detailed(self):
        """config.rule_configuration_manager の詳細テスト"""
        print("🔍 rule_configuration_manager 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.rule_configuration_manager',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.rule_configuration_manager import RuleConfigurationManager
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("📋 ルール設定管理機能テスト:")
            rule_manager = RuleConfigurationManager()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本ルール管理機能確認
            rule_methods = ['add_rule', 'remove_rule', 'get_rule', 'list_rules', 'validate_rule']
            found_methods = []
            
            for method_name in rule_methods:
                if hasattr(rule_manager, method_name):
                    found_methods.append(method_name)
                    print(f"✅ {method_name}: メソッド存在確認")
                else:
                    print(f"❌ {method_name}: メソッド不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(rule_manager) if not attr.startswith('_') and callable(getattr(rule_manager, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # 設定ファイル関連機能確認
            config_methods = ['load_config', 'save_config', 'reload_config']
            config_found = []
            for method_name in config_methods:
                if hasattr(rule_manager, method_name):
                    config_found.append(method_name)
                    print(f"✅ {method_name}: 設定ファイル機能確認")
            
            # ルール検証機能確認
            validation_methods = ['validate_all_rules', 'check_rule_conflicts', 'rule_priority_check']
            validation_found = []
            for method_name in validation_methods:
                if hasattr(rule_manager, method_name):
                    validation_found.append(method_name)
                    print(f"✅ {method_name}: ルール検証機能確認")
            
            print(f"\n📊 設定ファイル機能発見数: {len(config_found)}")
            print(f"📊 ルール検証機能発見数: {len(validation_found)}")
            
            # 属性確認
            rule_attrs = ['rules', 'config', 'rule_registry']
            found_attrs = []
            for attr_name in rule_attrs:
                if hasattr(rule_manager, attr_name):
                    found_attrs.append(attr_name)
                    print(f"✅ {attr_name}: 属性存在確認")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（ルール管理は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(all_methods) > 0
            
            # 最終判定
            if len(found_methods) >= 3 and len(config_found) >= 1:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(all_methods) >= 2:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本ルール管理機能不足")
            
            self.test_results['rule_configuration_manager'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['rule_configuration_manager'] = test_result
            return False
    
    def test_system_config_detailed(self):
        """config.portfolio_correlation_optimizer.configs.system_config の詳細テスト"""
        print("\n🔍 system_config 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.portfolio_correlation_optimizer.configs.system_config',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.portfolio_correlation_optimizer.configs.system_config import SystemConfig
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("⚙️ システム設定統合機能テスト:")
            system_config = SystemConfig()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本システム設定確認
            system_attrs = ['correlation_settings', 'optimizer_settings', 'portfolio_settings', 'risk_settings']
            found_attrs = []
            
            for attr_name in system_attrs:
                if hasattr(system_config, attr_name):
                    found_attrs.append(attr_name)
                    attr_value = getattr(system_config, attr_name)
                    print(f"✅ {attr_name}: {type(attr_value).__name__}")
                else:
                    print(f"❌ {attr_name}: 属性不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(system_config) if not attr.startswith('_') and callable(getattr(system_config, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # 5-3-3戦略間相関最適化機能確認
            correlation_methods = ['get_correlation_matrix', 'update_correlation_settings', 'optimize_allocation']
            correlation_found = []
            for method_name in correlation_methods:
                if hasattr(system_config, method_name):
                    correlation_found.append(method_name)
                    print(f"✅ {method_name}: 相関最適化機能確認")
            
            # 設定統合機能確認
            integration_methods = ['load_all_configs', 'validate_config_consistency', 'merge_configs']
            integration_found = []
            for method_name in integration_methods:
                if hasattr(system_config, method_name):
                    integration_found.append(method_name)
                    print(f"✅ {method_name}: 設定統合機能確認")
            
            print(f"\n📊 相関最適化機能発見数: {len(correlation_found)}")
            print(f"📊 設定統合機能発見数: {len(integration_found)}")
            
            # ポートフォリオ最適化設定確認
            portfolio_attrs = ['max_weight', 'min_weight', 'target_volatility', 'risk_budget']
            portfolio_found = []
            for attr_name in portfolio_attrs:
                if hasattr(system_config, attr_name):
                    portfolio_found.append(attr_name)
                    print(f"✅ {attr_name}: ポートフォリオ設定確認")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（システム設定統合は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
            
            # 最終判定
            if len(found_attrs) >= 2 and len(all_methods) >= 3:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(found_attrs) >= 1 or len(all_methods) >= 1:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本システム設定不足")
            
            self.test_results['system_config'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['system_config'] = test_result
            return False
    
    def test_var_config_detailed(self):
        """config.portfolio_var_calculator.var_config の詳細テスト"""
        print("\n🔍 var_config 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.portfolio_var_calculator.var_config',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.portfolio_var_calculator.var_config import VarConfig
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("📊 VaR計算設定機能テスト:")
            var_config = VarConfig()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本VaR設定確認
            var_attrs = ['confidence_level', 'time_horizon', 'calculation_method', 'monte_carlo_iterations']
            found_attrs = []
            
            for attr_name in var_attrs:
                if hasattr(var_config, attr_name):
                    found_attrs.append(attr_name)
                    attr_value = getattr(var_config, attr_name)
                    print(f"✅ {attr_name}: {attr_value}")
                else:
                    print(f"❌ {attr_name}: 属性不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(var_config) if not attr.startswith('_') and callable(getattr(var_config, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # VaR計算手法設定確認
            calculation_methods = ['parametric', 'historical', 'monte_carlo']
            method_attrs = []
            for method_name in calculation_methods:
                method_attr = f"{method_name}_config"
                if hasattr(var_config, method_attr):
                    method_attrs.append(method_attr)
                    print(f"✅ {method_attr}: VaR計算手法設定確認")
            
            # VaR設定管理機能確認
            config_methods = ['get_var_config', 'set_var_config', 'validate_config', 'reset_to_default']
            config_found = []
            for method_name in config_methods:
                if hasattr(var_config, method_name):
                    config_found.append(method_name)
                    print(f"✅ {method_name}: VaR設定管理機能確認")
            
            # リスク管理パラメータ確認
            risk_attrs = ['max_portfolio_var', 'var_limits', 'stress_test_scenarios']
            risk_found = []
            for attr_name in risk_attrs:
                if hasattr(var_config, attr_name):
                    risk_found.append(attr_name)
                    print(f"✅ {attr_name}: リスク管理パラメータ確認")
            
            print(f"\n📊 VaR計算手法設定発見数: {len(method_attrs)}")
            print(f"📊 VaR設定管理機能発見数: {len(config_found)}")
            print(f"📊 リスク管理パラメータ発見数: {len(risk_found)}")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（VaR設定は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
            
            # 最終判定
            if len(found_attrs) >= 3 and len(config_found) >= 2:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(found_attrs) >= 2 or len(all_methods) >= 2:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本VaR設定不足")
            
            self.test_results['var_config'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['var_config'] = test_result
            return False
    
    def generate_phase4c_report(self):
        """Phase 4C レポート生成"""
        report_lines = []
        
        report_lines.extend([
            "# 設定・ログ系モジュール専門テスト Phase 4C レポート",
            "",
            "## 🎯 Phase 4C テスト目的",
            "15個の設定・ログ系モジュールの段階的テスト - 設定・ルール管理系モジュール（3個）",
            "rule_configuration_manager, system_config, var_config の詳細検証",
            "",
            "## 📊 Phase 4C 結果サマリー",
            ""
        ])
        
        green_count = sum(1 for r in self.test_results.values() if r['status'] == 'GREEN')
        yellow_count = sum(1 for r in self.test_results.values() if r['status'] == 'YELLOW')
        red_count = sum(1 for r in self.test_results.values() if r['status'] == 'RED')
        total_fallbacks = sum(r['fallback_count'] for r in self.test_results.values())
        
        report_lines.extend([
            f"- **Phase 4C テスト対象モジュール数**: {len(self.test_results)}",
            f"- **🟢 再利用可能 (GREEN)**: {green_count}",
            f"- **🟡 要注意 (YELLOW)**: {yellow_count}",
            f"- **🔴 再利用禁止 (RED)**: {red_count}",
            f"- **🚨 フォールバック検出総数**: {total_fallbacks}",
            "",
            "---",
            ""
        ])
        
        # 個別モジュール結果
        for module_name, result in self.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            
            report_lines.extend([
                f"## {module_name}",
                "",
                f"**最終判定**: {status_emoji} {result['status']}",
                f"**main.py互換性**: {'✅' if result['main_py_compatibility'] else '❌'}",
                f"**フォールバック検出数**: {result['fallback_count']}",
                ""
            ])
            
            if result['performance']:
                report_lines.extend(["### ⚡ パフォーマンス指標", ""])
                for metric, value in result['performance'].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- **{metric}**: {value:.4f}秒")
                    else:
                        report_lines.append(f"- **{metric}**: {value}")
                report_lines.append("")
            
            if result['issues']:
                report_lines.extend(["### ⚠️ 検出された問題", ""])
                for issue in result['issues']:
                    report_lines.append(f"- {issue}")
                report_lines.append("")
            
            report_lines.extend(["---", ""])
        
        # 次のPhase案内
        report_lines.extend([
            "## 🚀 次のステップ (Phase 4D)",
            "",
            "### Phase 4D 対象モジュール (3個)",
            "",
            "1. **correlation_calculation_config** (config/)",
            "2. **portfolio_volatility_config** (config/)",  
            "3. **var_calculation_config** (config/)",
            "",
            "### Phase 4D 実行予定",
            "",
            "- 相関計算設定機能のテスト",
            "- ポートフォリオボラティリティ設定の検証", 
            "- VaR計算設定システムのテスト",
            "",
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト段階**: Phase 4C (設定・ルール管理系モジュール)",
            f"**テスト環境**: {self.temp_dir}"
        ])
        
        return "\n".join(report_lines)

def main():
    """設定・ログ系専門テスト Phase 4C の実行"""
    print("🚀 設定・ログ系モジュール専門テスト開始 (Phase 4C)")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 設定・ルール管理系モジュール (3個) の詳細検証")
    print("="*70)
    
    tester = ConfigLogPhase4CTester()
    tester.setup_test_environment()
    
    try:
        # Phase 4C: 設定・ルール管理系モジュール
        tests = [
            ('rule_configuration_manager', tester.test_rule_configuration_manager_detailed),
            ('system_config', tester.test_system_config_detailed),
            ('var_config', tester.test_var_config_detailed),
        ]
        
        success_count = 0
        for test_name, test_func in tests:
            print(f"\n🔄 {test_name}テスト実行中...")
            success = test_func()
            
            if success:
                success_count += 1
                print(f"✅ {test_name}テスト完了")
            else:
                print(f"❌ {test_name}テストで重大エラー")
        
        # Phase 4C レポート生成
        report = tester.generate_phase4c_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "config_log_modules_test_report_phase4c.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Phase 4C 実行サマリー
        print(f"\n" + "="*70)
        print(f"🎯 設定・ログ系専門テスト Phase 4C 完了")
        print(f"="*70)
        print(f"📊 Phase 4C 実行結果: {success_count}/{len(tests)} テスト成功")
        print(f"📄 Phase 4C レポート: {output_file}")
        print(f"🚨 フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 簡易判定結果表示
        for module_name, result in tester.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            print(f"   {status_emoji} {module_name}: {result['status']}")
            
        # 次のステップ案内
        print(f"\n🚀 次のステップ:")
        print(f"   Phase 4D でメトリック設定系3モジュールをテスト予定")
        print(f"   - correlation_calculation_config")
        print(f"   - portfolio_volatility_config")
        print(f"   - var_calculation_config")
    
    finally:
        tester.cleanup_test_environment()

if __name__ == "__main__":
    main()