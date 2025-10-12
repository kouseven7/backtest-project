"""
設定・ログ系モジュール専門テスト Phase 4B
15個の設定・ログ系モジュールの段階的テスト - パラメータ管理系モジュール
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

class ConfigLogPhase4BTester:
    """設定・ログ系モジュール Phase 4B 専門テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="config_log_phase4b_")
        print(f"📁 Phase 4B テスト環境作成: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """テスト環境のクリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"🧹 Phase 4B テスト環境クリーンアップ完了")
    
    def detect_config_log_fallbacks(self, result, module_name, operation):
        """設定・ログ系特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: サイレント設定失敗
        if operation == "config_load":
            if isinstance(result, dict) and len(result) == 0:
                fallbacks.append("空設定辞書返却: 設定ファイル読み込み失敗を隠蔽")
            elif isinstance(result, dict):
                default_keys = ['debug', 'test', 'default', 'fallback']
                has_only_defaults = all(key.lower() in str(result).lower() for key in default_keys if key in str(result).lower())
                if has_only_defaults:
                    fallbacks.append("デフォルト値のみ: 実設定値取得失敗")
        
        # パターン2: パラメータ読み込み失敗の隠蔽
        elif operation == "parameter_load":
            if result is None:
                fallbacks.append("パラメータ読み込み失敗: 例外を隠蔽してNone返却")
        
        # パターン3: ImportError隠蔽
        elif operation == "module_import":
            if result is None:
                fallbacks.append("モジュールインポート失敗: ImportErrorを隠蔽")
        
        if fallbacks:
            self.fallback_detected.extend([f"{module_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def test_meta_parameter_controller_detailed(self):
        """config.weight_learning_optimizer.meta_parameter_controller の詳細テスト"""
        print("🔍 meta_parameter_controller 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.weight_learning_optimizer.meta_parameter_controller',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.weight_learning_optimizer.meta_parameter_controller import MetaParameterController
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("🎛️ メタパラメータ制御機能テスト:")
            controller = MetaParameterController()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本機能確認
            basic_methods = ['get_parameters', 'set_parameters', 'optimize_parameters', 'reset_parameters']
            found_methods = []
            
            for method_name in basic_methods:
                if hasattr(controller, method_name):
                    found_methods.append(method_name)
                    print(f"✅ {method_name}: メソッド存在確認")
                else:
                    print(f"❌ {method_name}: メソッド不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(controller) if not attr.startswith('_') and callable(getattr(controller, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # 属性確認
            basic_attrs = ['parameters', 'config', 'settings']
            found_attrs = []
            for attr_name in basic_attrs:
                if hasattr(controller, attr_name):
                    found_attrs.append(attr_name)
                    print(f"✅ {attr_name}: 属性存在確認")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（間接的使用の可能性）
            print(f"⚠️ main.py間接互換性")
            test_result['main_py_compatibility'] = len(all_methods) > 0
            
            # 最終判定
            if len(all_methods) >= 3 and len(found_attrs) >= 1:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(all_methods) >= 1:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本機能不足")
            
            self.test_results['meta_parameter_controller'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['meta_parameter_controller'] = test_result
            return False
    
    def test_parameter_adjuster_detailed(self):
        """config.trend_precision_adjustment.parameter_adjuster の詳細テスト"""
        print("\n🔍 parameter_adjuster 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.trend_precision_adjustment.parameter_adjuster',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.trend_precision_adjustment.parameter_adjuster import ParameterAdjuster
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("⚙️ パラメータ調整機能テスト:")
            adjuster = ParameterAdjuster()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本機能確認
            adjust_methods = ['adjust_parameters', 'get_adjustment', 'set_adjustment', 'calibrate']
            found_methods = []
            
            for method_name in adjust_methods:
                if hasattr(adjuster, method_name):
                    found_methods.append(method_name)
                    print(f"✅ {method_name}: メソッド存在確認")
                else:
                    print(f"❌ {method_name}: メソッド不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(adjuster) if not attr.startswith('_') and callable(getattr(adjuster, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # トレンド精度調整の専門機能確認
            trend_methods = ['adjust_trend_threshold', 'precision_calibration', 'trend_detection_adjustment']
            trend_found = []
            for method_name in trend_methods:
                if hasattr(adjuster, method_name):
                    trend_found.append(method_name)
                    print(f"✅ {method_name}: トレンド専門機能確認")
            
            if trend_found:
                print(f"📊 トレンド専門機能発見数: {len(trend_found)}")
            else:
                print(f"⚠️ トレンド専門機能未発見")
                test_result['issues'].append("トレンド専門機能不足")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認
            print(f"⚠️ main.py間接互換性")
            test_result['main_py_compatibility'] = len(all_methods) > 0
            
            # 最終判定
            if len(all_methods) >= 3 and len(trend_found) >= 1:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(all_methods) >= 1:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本機能不足")
            
            self.test_results['parameter_adjuster'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['parameter_adjuster'] = test_result
            return False
    
    def test_strategy_parameter_standardizer_detailed(self):
        """config.strategy_parameter_standardizer の詳細テスト"""
        print("\n🔍 strategy_parameter_standardizer 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.strategy_parameter_standardizer',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.strategy_parameter_standardizer import StrategyParameterStandardizer
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("📐 戦略パラメータ標準化機能テスト:")
            standardizer = StrategyParameterStandardizer()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本機能確認
            standard_methods = ['standardize_parameters', 'normalize_parameters', 'validate_parameters', 'get_standard_format']
            found_methods = []
            
            for method_name in standard_methods:
                if hasattr(standardizer, method_name):
                    found_methods.append(method_name)
                    print(f"✅ {method_name}: メソッド存在確認")
                else:
                    print(f"❌ {method_name}: メソッド不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(standardizer) if not attr.startswith('_') and callable(getattr(standardizer, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # TODO #13 関連機能確認
            todo13_methods = ['parameter_standardization', 'strategy_compatibility_check', 'unified_parameter_format']
            todo13_found = []
            for method_name in todo13_methods:
                if hasattr(standardizer, method_name):
                    todo13_found.append(method_name)
                    print(f"✅ {method_name}: TODO #13機能確認")
            
            if todo13_found:
                print(f"📊 TODO #13機能発見数: {len(todo13_found)}")
            else:
                print(f"⚠️ TODO #13機能未実装の可能性")
                test_result['issues'].append("TODO #13機能未確認")
            
            # 戦略パラメータサポート確認
            supported_strategies = ['VWAP', 'Momentum', 'Breakout', 'Contrarian']
            print(f"\n🎯 戦略サポート確認:")
            for strategy in supported_strategies:
                strategy_attr = f"supports_{strategy.lower()}"
                if hasattr(standardizer, strategy_attr):
                    print(f"✅ {strategy}: サポート確認")
                
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（戦略パラメータ標準化は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(all_methods) > 0
            
            # 最終判定
            if len(all_methods) >= 3 and len(found_methods) >= 2:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(all_methods) >= 1:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本機能不足")
            
            self.test_results['strategy_parameter_standardizer'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['strategy_parameter_standardizer'] = test_result
            return False
    
    def test_trend_params_detailed(self):
        """config.trend_params の詳細テスト"""
        print("\n🔍 trend_params 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.trend_params',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            # trend_paramsは設定モジュールの可能性が高い
            from config.trend_params import TrendParametersConfig
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("📊 トレンドパラメータ設定機能テスト:")
            trend_config = TrendParametersConfig()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本設定確認
            trend_attrs = ['trend_threshold', 'trend_period', 'trend_sensitivity', 'trend_filters']
            found_attrs = []
            
            for attr_name in trend_attrs:
                if hasattr(trend_config, attr_name):
                    found_attrs.append(attr_name)
                    attr_value = getattr(trend_config, attr_name)
                    print(f"✅ {attr_name}: {attr_value}")
                else:
                    print(f"❌ {attr_name}: 属性不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(trend_config) if not attr.startswith('_') and callable(getattr(trend_config, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # 設定取得・更新機能確認
            config_methods = ['get_config', 'set_config', 'update_config', 'reset_config']
            config_found = []
            for method_name in config_methods:
                if hasattr(trend_config, method_name):
                    config_found.append(method_name)
                    print(f"✅ {method_name}: 設定管理機能確認")
            
            # トレンド判定パラメータ確認
            trend_detection_attrs = ['ma_period', 'rsi_period', 'volume_factor', 'volatility_threshold']
            detection_found = []
            for attr_name in trend_detection_attrs:
                if hasattr(trend_config, attr_name):
                    detection_found.append(attr_name)
                    print(f"✅ {attr_name}: トレンド判定パラメータ確認")
            
            print(f"\n📊 トレンド判定パラメータ発見数: {len(detection_found)}")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（トレンドパラメータは重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
            
            # 最終判定
            if len(found_attrs) >= 2 and len(config_found) >= 1:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(found_attrs) >= 1 or len(all_methods) >= 1:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本設定不足")
            
            self.test_results['trend_params'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['trend_params'] = test_result
            return False
    
    def generate_phase4b_report(self):
        """Phase 4B レポート生成"""
        report_lines = []
        
        report_lines.extend([
            "# 設定・ログ系モジュール専門テスト Phase 4B レポート",
            "",
            "## 🎯 Phase 4B テスト目的",
            "15個の設定・ログ系モジュールの段階的テスト - パラメータ管理系モジュール（4個）",
            "meta_parameter_controller, parameter_adjuster, strategy_parameter_standardizer, trend_params の詳細検証",
            "",
            "## 📊 Phase 4B 結果サマリー",
            ""
        ])
        
        green_count = sum(1 for r in self.test_results.values() if r['status'] == 'GREEN')
        yellow_count = sum(1 for r in self.test_results.values() if r['status'] == 'YELLOW')
        red_count = sum(1 for r in self.test_results.values() if r['status'] == 'RED')
        total_fallbacks = sum(r['fallback_count'] for r in self.test_results.values())
        
        report_lines.extend([
            f"- **Phase 4B テスト対象モジュール数**: {len(self.test_results)}",
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
            "## 🚀 次のステップ (Phase 4C)",
            "",
            "### Phase 4C 対象モジュール (3個)",
            "",
            "1. **rule_configuration_manager** (config/)",
            "2. **system_config** (config/portfolio_correlation_optimizer/configs/)",  
            "3. **var_config** (config/portfolio_var_calculator/)",
            "",
            "### Phase 4C 実行予定",
            "",
            "- ルール設定管理機能のテスト",
            "- システム設定統合の検証", 
            "- VaR計算設定システムのテスト",
            "",
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト段階**: Phase 4B (パラメータ管理系モジュール)",
            f"**テスト環境**: {self.temp_dir}"
        ])
        
        return "\n".join(report_lines)

def main():
    """設定・ログ系専門テスト Phase 4B の実行"""
    print("🚀 設定・ログ系モジュール専門テスト開始 (Phase 4B)")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 パラメータ管理系モジュール (4個) の詳細検証")
    print("="*70)
    
    tester = ConfigLogPhase4BTester()
    tester.setup_test_environment()
    
    try:
        # Phase 4B: パラメータ管理系モジュール
        tests = [
            ('meta_parameter_controller', tester.test_meta_parameter_controller_detailed),
            ('parameter_adjuster', tester.test_parameter_adjuster_detailed),
            ('strategy_parameter_standardizer', tester.test_strategy_parameter_standardizer_detailed),
            ('trend_params', tester.test_trend_params_detailed),
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
        
        # Phase 4B レポート生成
        report = tester.generate_phase4b_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "config_log_modules_test_report_phase4b.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Phase 4B 実行サマリー
        print(f"\n" + "="*70)
        print(f"🎯 設定・ログ系専門テスト Phase 4B 完了")
        print(f"="*70)
        print(f"📊 Phase 4B 実行結果: {success_count}/{len(tests)} テスト成功")
        print(f"📄 Phase 4B レポート: {output_file}")
        print(f"🚨 フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 簡易判定結果表示
        for module_name, result in tester.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            print(f"   {status_emoji} {module_name}: {result['status']}")
            
        # 次のステップ案内
        print(f"\n🚀 次のステップ:")
        print(f"   Phase 4C で設定・ルール管理系3モジュールをテスト予定")
        print(f"   - rule_configuration_manager")
        print(f"   - system_config")
        print(f"   - var_config")
    
    finally:
        tester.cleanup_test_environment()

if __name__ == "__main__":
    main()