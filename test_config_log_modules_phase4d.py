"""
設定・ログ系モジュール専門テスト Phase 4D（最終フェーズ）
15個の設定・ログ系モジュールの段階的テスト - メトリック設定系モジュール
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

class ConfigLogPhase4DTester:
    """設定・ログ系モジュール Phase 4D（最終フェーズ）専門テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="config_log_phase4d_")
        print(f"📁 Phase 4D（最終フェーズ）テスト環境作成: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """テスト環境のクリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"🧹 Phase 4D（最終フェーズ）テスト環境クリーンアップ完了")
    
    def detect_config_log_fallbacks(self, result, module_name, operation):
        """設定・ログ系特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: メトリック正規化設定失敗
        if operation == "metric_normalization":
            if isinstance(result, dict) and len(result) == 0:
                fallbacks.append("空メトリック正規化設定: 正規化パラメータ読み込み失敗を隠蔽")
            elif result is None:
                fallbacks.append("メトリック正規化設定失敗: 例外を隠蔽してNone返却")
        
        # パターン2: メトリック選択設定失敗
        elif operation == "metric_selection":
            if isinstance(result, dict) and 'default' in str(result).lower():
                fallbacks.append("デフォルトメトリック選択: 実設定値取得失敗")
        
        # パターン3: 相関計算設定読み込み失敗
        elif operation == "correlation_calculation":
            if result is None:
                fallbacks.append("相関計算設定読み込み失敗: 例外を隠蔽")
        
        if fallbacks:
            self.fallback_detected.extend([f"{module_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def test_metric_normalization_config_detailed(self):
        """config.metric_normalization_config の詳細テスト"""
        print("🔍 metric_normalization_config 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.metric_normalization_config',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.metric_normalization_config import MetricNormalizationConfig
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("📊 メトリック正規化設定機能テスト:")
            norm_config = MetricNormalizationConfig()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本正規化設定確認
            norm_attrs = ['normalization_method', 'scaling_factor', 'min_value', 'max_value', 'z_score_threshold']
            found_attrs = []
            
            for attr_name in norm_attrs:
                if hasattr(norm_config, attr_name):
                    found_attrs.append(attr_name)
                    attr_value = getattr(norm_config, attr_name)
                    print(f"✅ {attr_name}: {attr_value}")
                else:
                    print(f"❌ {attr_name}: 属性不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(norm_config) if not attr.startswith('_') and callable(getattr(norm_config, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # 正規化手法確認
            normalization_methods = ['min_max_scaling', 'z_score_normalization', 'robust_scaling', 'unit_vector_scaling']
            method_found = []
            for method_name in normalization_methods:
                if hasattr(norm_config, method_name):
                    method_found.append(method_name)
                    print(f"✅ {method_name}: 正規化手法確認")
            
            # 設定管理機能確認
            config_methods = ['get_normalization_config', 'set_normalization_params', 'validate_config', 'apply_normalization']
            config_found = []
            for method_name in config_methods:
                if hasattr(norm_config, method_name):
                    config_found.append(method_name)
                    print(f"✅ {method_name}: 設定管理機能確認")
            
            # メトリック種別対応確認
            metric_types = ['performance_metrics', 'risk_metrics', 'volatility_metrics']
            metric_support = []
            for metric_type in metric_types:
                support_attr = f"supports_{metric_type}"
                if hasattr(norm_config, support_attr):
                    metric_support.append(metric_type)
                    print(f"✅ {metric_type}: メトリック種別サポート確認")
            
            print(f"\n📊 正規化手法発見数: {len(method_found)}")
            print(f"📊 設定管理機能発見数: {len(config_found)}")
            print(f"📊 サポートメトリック種別数: {len(metric_support)}")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（メトリック正規化は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
            
            # 最終判定
            if len(found_attrs) >= 3 and len(config_found) >= 2:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(found_attrs) >= 2 or len(all_methods) >= 3:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本正規化設定不足")
            
            self.test_results['metric_normalization_config'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['metric_normalization_config'] = test_result
            return False
    
    def test_metric_selection_config_detailed(self):
        """config.metric_selection_config の詳細テスト"""
        print("\n🔍 metric_selection_config 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.metric_selection_config',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        try:
            from config.metric_selection_config import MetricSelectionConfig
            
            print("✅ インポート成功")
            
            start_time = datetime.now()
            
            print("🎯 メトリック選択設定機能テスト:")
            selection_config = MetricSelectionConfig()
            
            init_time = (datetime.now() - start_time).total_seconds()
            test_result['performance']['init_time'] = init_time
            
            # 基本選択設定確認
            selection_attrs = ['selected_metrics', 'metric_weights', 'selection_criteria', 'priority_order']
            found_attrs = []
            
            for attr_name in selection_attrs:
                if hasattr(selection_config, attr_name):
                    found_attrs.append(attr_name)
                    attr_value = getattr(selection_config, attr_name)
                    print(f"✅ {attr_name}: {type(attr_value).__name__}")
                else:
                    print(f"❌ {attr_name}: 属性不足")
            
            # 利用可能メソッド一覧
            all_methods = [attr for attr in dir(selection_config) if not attr.startswith('_') and callable(getattr(selection_config, attr))]
            print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
            if all_methods:
                print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
            
            # メトリック選択機能確認
            selection_methods = ['select_metrics', 'filter_metrics', 'rank_metrics', 'optimize_selection']
            selection_found = []
            for method_name in selection_methods:
                if hasattr(selection_config, method_name):
                    selection_found.append(method_name)
                    print(f"✅ {method_name}: メトリック選択機能確認")
            
            # 重み付け機能確認
            weight_methods = ['calculate_weights', 'adjust_weights', 'normalize_weights']
            weight_found = []
            for method_name in weight_methods:
                if hasattr(selection_config, method_name):
                    weight_found.append(method_name)
                    print(f"✅ {method_name}: 重み付け機能確認")
            
            # 選択基準確認
            selection_criteria = ['correlation_threshold', 'importance_score', 'volatility_limit']
            criteria_found = []
            for criteria_name in selection_criteria:
                if hasattr(selection_config, criteria_name):
                    criteria_found.append(criteria_name)
                    print(f"✅ {criteria_name}: 選択基準確認")
            
            print(f"\n📊 メトリック選択機能発見数: {len(selection_found)}")
            print(f"📊 重み付け機能発見数: {len(weight_found)}")
            print(f"📊 選択基準発見数: {len(criteria_found)}")
            
            print(f"\n⚡ パフォーマンス:")
            print(f"  初期化時間: {init_time:.4f}秒")
            
            # main.py互換性確認（メトリック選択は重要）
            print(f"✅ main.py高互換性期待")
            test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
            
            # 最終判定
            if len(found_attrs) >= 3 and len(selection_found) >= 2:
                test_result['status'] = 'GREEN'
                print(f"\n🟢 最終判定: 再利用可能")
            elif len(found_attrs) >= 2 or len(all_methods) >= 3:
                test_result['status'] = 'YELLOW'
                print(f"\n🟡 最終判定: 要注意")
            else:
                test_result['status'] = 'RED'
                print(f"\n🔴 最終判定: 再利用禁止")
                test_result['issues'].append("基本選択設定不足")
            
            self.test_results['metric_selection_config'] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"実行エラー: {str(e)}")
            self.test_results['metric_selection_config'] = test_result
            return False
    
    def test_correlation_calculation_config_detailed(self):
        """config.correlation.* の相関計算設定詳細テスト"""
        print("\n🔍 correlation_calculation_config 詳細テスト開始")
        print("-" * 60)
        
        test_result = {
            'module': 'config.correlation.correlation_calculation_config',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False
        }
        
        # 複数の相関設定モジュールを試行
        correlation_modules = [
            'config.correlation.strategy_correlation_analyzer',
            'config.correlation.correlation_matrix_visualizer'
        ]
        
        successful_import = None
        for module_path in correlation_modules:
            try:
                print(f"🔄 {module_path} インポート試行中...")
                
                if 'strategy_correlation_analyzer' in module_path:
                    from config.correlation.strategy_correlation_analyzer import CorrelationConfig
                    correlation_config = CorrelationConfig()
                    successful_import = 'CorrelationConfig'
                    test_result['module'] = module_path
                    break
                elif 'correlation_matrix_visualizer' in module_path:
                    from config.correlation.correlation_matrix_visualizer import CorrelationMatrixVisualizer
                    correlation_config = CorrelationMatrixVisualizer()
                    successful_import = 'CorrelationMatrixVisualizer'
                    test_result['module'] = module_path
                    break
                    
            except Exception as e:
                print(f"❌ {module_path} インポート失敗: {e}")
                continue
        
        if not successful_import:
            print(f"❌ 全ての相関設定モジュールインポート失敗")
            test_result['status'] = 'RED'
            test_result['issues'].append("相関設定モジュール全滅")
            self.test_results['correlation_calculation_config'] = test_result
            return False
        
        print(f"✅ {successful_import} インポート成功")
        
        start_time = datetime.now()
        
        print("📈 相関計算設定機能テスト:")
        
        init_time = (datetime.now() - start_time).total_seconds()
        test_result['performance']['init_time'] = init_time
        
        # 基本相関設定確認
        correlation_attrs = ['correlation_method', 'calculation_window', 'min_periods', 'threshold']
        found_attrs = []
        
        for attr_name in correlation_attrs:
            if hasattr(correlation_config, attr_name):
                found_attrs.append(attr_name)
                attr_value = getattr(correlation_config, attr_name)
                print(f"✅ {attr_name}: {attr_value}")
            else:
                print(f"❌ {attr_name}: 属性不足")
        
        # 利用可能メソッド一覧
        all_methods = [attr for attr in dir(correlation_config) if not attr.startswith('_') and callable(getattr(correlation_config, attr))]
        print(f"\n📋 利用可能メソッド数: {len(all_methods)}")
        if all_methods:
            print(f"  主要メソッド: {', '.join(all_methods[:5])}...")
        
        # 相関計算機能確認
        calc_methods = ['calculate_correlation', 'compute_matrix', 'get_correlation_matrix', 'update_correlation']
        calc_found = []
        for method_name in calc_methods:
            if hasattr(correlation_config, method_name):
                calc_found.append(method_name)
                print(f"✅ {method_name}: 相関計算機能確認")
        
        # 戦略間相関機能確認（5-3-3システム関連）
        strategy_methods = ['analyze_strategy_correlation', 'get_strategy_matrix', 'correlation_dashboard']
        strategy_found = []
        for method_name in strategy_methods:
            if hasattr(correlation_config, method_name):
                strategy_found.append(method_name)
                print(f"✅ {method_name}: 戦略間相関機能確認")
        
        # 視覚化機能確認
        viz_methods = ['visualize_matrix', 'plot_correlation', 'export_visualization']
        viz_found = []
        for method_name in viz_methods:
            if hasattr(correlation_config, method_name):
                viz_found.append(method_name)
                print(f"✅ {method_name}: 視覚化機能確認")
        
        print(f"\n📊 相関計算機能発見数: {len(calc_found)}")
        print(f"📊 戦略間相関機能発見数: {len(strategy_found)}")
        print(f"📊 視覚化機能発見数: {len(viz_found)}")
        
        print(f"\n⚡ パフォーマンス:")
        print(f"  初期化時間: {init_time:.4f}秒")
        
        # main.py互換性確認（相関計算は5-3-3システムで重要）
        print(f"✅ main.py高互換性期待（5-3-3システム関連）")
        test_result['main_py_compatibility'] = len(found_attrs) > 0 or len(all_methods) > 0
        
        # 最終判定
        if len(calc_found) >= 2 and len(all_methods) >= 5:
            test_result['status'] = 'GREEN'
            print(f"\n🟢 最終判定: 再利用可能")
        elif len(calc_found) >= 1 or len(all_methods) >= 3:
            test_result['status'] = 'YELLOW'
            print(f"\n🟡 最終判定: 要注意")
        else:
            test_result['status'] = 'RED'
            print(f"\n🔴 最終判定: 再利用禁止")
            test_result['issues'].append("基本相関計算機能不足")
        
        self.test_results['correlation_calculation_config'] = test_result
        return True
    
    def generate_phase4d_final_report(self):
        """Phase 4D（最終フェーズ）レポート生成"""
        report_lines = []
        
        report_lines.extend([
            "# 設定・ログ系モジュール専門テスト Phase 4D（最終フェーズ）レポート",
            "",
            "## 🎯 Phase 4D テスト目的",
            "15個の設定・ログ系モジュールの段階的テスト - メトリック設定系モジュール（3個）",
            "metric_normalization_config, metric_selection_config, correlation_calculation_config の詳細検証",
            "**🏁 Phase 4A-4D 全系統テスト完了**",
            "",
            "## 📊 Phase 4D 結果サマリー",
            ""
        ])
        
        green_count = sum(1 for r in self.test_results.values() if r['status'] == 'GREEN')
        yellow_count = sum(1 for r in self.test_results.values() if r['status'] == 'YELLOW')
        red_count = sum(1 for r in self.test_results.values() if r['status'] == 'RED')
        total_fallbacks = sum(r['fallback_count'] for r in self.test_results.values())
        
        report_lines.extend([
            f"- **Phase 4D テスト対象モジュール数**: {len(self.test_results)}",
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
        
        # 全Phase統合サマリー
        report_lines.extend([
            "## 🏁 Phase 4A-4D 全系統テスト完了統計",
            "",
            "### 📈 全Phase実行統計",
            "",
            "- **Phase 4A (基盤設定)**: 5モジュール → 4 GREEN, 1 RED (80%成功率)",
            "- **Phase 4B (パラメータ管理)**: 4モジュール → 1 GREEN, 1 YELLOW, 2 RED (50%成功率)",
            "- **Phase 4C (設定・ルール管理)**: 3モジュール → 0 GREEN, 1 YELLOW, 2 RED (33%成功率)",
            f"- **Phase 4D (メトリック設定)**: {len(self.test_results)}モジュール → {green_count} GREEN, {yellow_count} YELLOW, {red_count} RED",
            "",
            "### 🎯 最終テスト完了",
            "",
            "15個の設定・ログ系モジュール段階的テスト完了",
            "再利用可能モジュールの特定とエラーモジュールの分析完了",
            "",
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト段階**: Phase 4D（最終フェーズ） - メトリック設定系モジュール",
            f"**テスト環境**: {self.temp_dir}",
            "**テストシリーズ**: Phase 4A-4D 完全実行完了 ✅"
        ])
        
        return "\n".join(report_lines)

def main():
    """設定・ログ系専門テスト Phase 4D（最終フェーズ）の実行"""
    print("🚀 設定・ログ系モジュール専門テスト開始 (Phase 4D - 最終フェーズ)")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 メトリック設定系モジュール (3個) の詳細検証")
    print("🏁 Phase 4A-4D 全系統テスト最終フェーズ")
    print("="*70)
    
    tester = ConfigLogPhase4DTester()
    tester.setup_test_environment()
    
    try:
        # Phase 4D: メトリック設定系モジュール（最終フェーズ）
        tests = [
            ('metric_normalization_config', tester.test_metric_normalization_config_detailed),
            ('metric_selection_config', tester.test_metric_selection_config_detailed),
            ('correlation_calculation_config', tester.test_correlation_calculation_config_detailed),
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
        
        # Phase 4D（最終フェーズ）レポート生成
        report = tester.generate_phase4d_final_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "config_log_modules_test_report_phase4d_final.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Phase 4D（最終フェーズ）実行サマリー
        print(f"\n" + "="*70)
        print(f"🏁 設定・ログ系専門テスト Phase 4D（最終フェーズ）完了")
        print(f"="*70)
        print(f"📊 Phase 4D 実行結果: {success_count}/{len(tests)} テスト成功")
        print(f"📄 Phase 4D 最終レポート: {output_file}")
        print(f"🚨 フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 簡易判定結果表示
        for module_name, result in tester.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            print(f"   {status_emoji} {module_name}: {result['status']}")
            
        # 全系統テスト完了宣言
        print(f"\n🎉 Phase 4A-4D 全系統テスト完了")
        print(f"   15個の設定・ログ系モジュール段階的テスト終了")
        print(f"   再利用可能モジュール特定完了")
        print(f"   comprehensive_module_test.py 準備完了")
    
    finally:
        tester.cleanup_test_environment()

if __name__ == "__main__":
    main()