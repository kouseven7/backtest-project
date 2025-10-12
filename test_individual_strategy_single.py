"""
個別戦略クラス専門テスト（単一戦略版）
VWAPBreakoutStrategy専門テスト - バックテスト基本理念遵守確認
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import warnings
from pathlib import Path

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class SingleStrategyTester:
    """VWAPBreakoutStrategy専門テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        
    def detect_strategy_fallbacks(self, result, strategy_name, operation):
        """戦略特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: モックデータ使用検出
        if operation == "backtest_execution":
            if isinstance(result, pd.DataFrame):
                # データの異常パターンチェック
                if len(result) == 0:
                    fallbacks.append("Empty backtest result - potential mock data usage")
                elif 'Entry_Signal' not in result.columns or 'Exit_Signal' not in result.columns:
                    fallbacks.append("Missing signal columns - backtest principle violation")
                elif (result['Entry_Signal'] == 0).all() and (result['Exit_Signal'] == 0).all():
                    fallbacks.append("Zero signals generated - mock or test data suspected")
        
        # パターン2: 計算失敗時のデフォルト値返却
        elif operation == "signal_generation":
            if isinstance(result, dict):
                if result.get('entry_signals', 0) == 0 and result.get('exit_signals', 0) == 0:
                    fallbacks.append("No signals generated - calculation failure fallback")
        
        if fallbacks:
            self.fallback_detected.extend([f"{strategy_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def get_real_market_data(self):
        """実際の市場データを取得（yfinanceベース）"""
        try:
            from data_fetcher import get_parameters_and_data
            
            print("📊 実際の市場データを取得中...")
            ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
            
            # データ前処理
            from data_processor import preprocess_data
            from indicators.indicator_calculator import compute_indicators
            
            stock_data = preprocess_data(stock_data)
            stock_data = compute_indicators(stock_data)
            
            print(f"✅ 実データ取得完了: {ticker} ({start_date} - {end_date}), {len(stock_data)}行")
            
            return {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'stock_data': stock_data,
                'index_data': index_data
            }
        
        except Exception as e:
            print(f"❌ 実データ取得失敗: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return None
    
    def get_main_py_compatible_params(self, strategy_name):
        """main.py準拠の承認済みパラメータを取得"""
        try:
            from config.optimized_parameters import OptimizedParameterManager
            
            param_manager = OptimizedParameterManager()
            
            # main.pyと同じ方法で最適化パラメータを読み込み
            params = param_manager.load_approved_params(strategy_name, "AAPL")  # デフォルトティッカー
            
            if params:
                print(f"✅ 承認済みパラメータ取得: {strategy_name}")
                return params
            else:
                # デフォルトパラメータ使用（main.pyのget_default_parameters相当）
                return self.get_fallback_parameters(strategy_name)
                
        except Exception as e:
            print(f"⚠️ パラメータ取得エラー ({strategy_name}): {e}")
            return self.get_fallback_parameters(strategy_name)
    
    def get_fallback_parameters(self, strategy_name):
        """main.py準拠のデフォルトパラメータ"""
        defaults = {
            'VWAPBreakoutStrategy': {
                'vwap_period': 20,
                'volume_threshold_multiplier': 1.5,
                'breakout_threshold': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            }
        }
        
        return defaults.get(strategy_name, {})
    
    def execute_strategy_with_main_py_pattern(self, strategy_class, strategy_name, stock_data, index_data, params):
        """main.py完全準拠の戦略実行パターン"""
        try:
            print(f"🚀 {strategy_name} main.py準拠実行開始")
            
            # main.pyのVWAPBreakoutStrategy初期化パターン（line 68相当）
            strategy = strategy_class(
                data=stock_data.copy(),
                index_data=index_data,
                params=params,
                price_column="Adj Close"
            )
            
            print(f"✅ 戦略初期化成功: {strategy_name}")
            print(f"📋 使用パラメータ: {params}")
            
            # バックテスト基本理念遵守: 実際のbacktest()実行
            print("🎯 バックテスト基本理念遵守: 実際のstrategy.backtest()実行")
            start_time = datetime.now()
            result = strategy.backtest()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ backtest()実行完了: {execution_time:.3f}秒")
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'strategy_instance': strategy
            }
            
        except Exception as e:
            print(f"❌ {strategy_name}実行エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'result': None,
                'execution_time': 0
            }
    
    def validate_backtest_output_comprehensive(self, result, strategy_name):
        """包括的バックテスト出力検証"""
        validation_results = {
            'signal_columns_exist': False,
            'entry_signals_generated': False,
            'exit_signals_generated': False,
            'trade_count_positive': False,
            'data_integrity': False,
            'backtest_principle_compliance': False,
            'anomaly_detection': {}
        }
        
        try:
            if result is None or not isinstance(result, pd.DataFrame):
                print("❌ バックテスト結果がNullまたは無効")
                validation_results['data_integrity'] = False
                return validation_results
            
            print(f"📊 バックテスト結果データフレーム: {len(result)}行 x {len(result.columns)}列")
            print(f"📋 結果列: {list(result.columns)}")
            
            # 必須列存在確認
            required_columns = ['Entry_Signal', 'Exit_Signal']
            validation_results['signal_columns_exist'] = all(col in result.columns for col in required_columns)
            
            if validation_results['signal_columns_exist']:
                print("✅ Entry_Signal/Exit_Signal列存在確認")
                
                # シグナル生成確認
                entry_signals = (result['Entry_Signal'] == 1).sum()
                exit_signals = (result['Exit_Signal'] != 0).sum()  # Exit_Signal: 1 or -1
                
                validation_results['entry_signals_generated'] = entry_signals > 0
                validation_results['exit_signals_generated'] = exit_signals > 0
                validation_results['trade_count_positive'] = entry_signals > 0
                
                print(f"📊 {strategy_name}シグナル統計:")
                print(f"  エントリー: {entry_signals}回")
                print(f"  エグジット: {exit_signals}回")
                
                # 異常エグジット検出
                if entry_signals > 0:
                    exit_entry_ratio = exit_signals / entry_signals
                    print(f"  エグジット/エントリー比率: {exit_entry_ratio:.2f}")
                    
                    if exit_entry_ratio > 5.0:
                        validation_results['anomaly_detection'] = {
                            'type': 'excessive_exits',
                            'ratio': exit_entry_ratio,
                            'severity': 'critical'
                        }
                        print("🚨 異常検出: 過剰エグジットパターン")
                    elif exit_entry_ratio > 2.0:
                        validation_results['anomaly_detection'] = {
                            'type': 'high_exit_ratio', 
                            'ratio': exit_entry_ratio,
                            'severity': 'warning'
                        }
                        print("⚠️ 警告: 高エグジット比率")
                else:
                    print("⚠️ エントリーシグナルなし")
            else:
                print("❌ Entry_Signal/Exit_Signal列が存在しません")
                missing_cols = [col for col in required_columns if col not in result.columns]
                print(f"   不足列: {missing_cols}")
            
            # データ整合性確認
            validation_results['data_integrity'] = len(result) > 0 and not result.empty
            
            # バックテスト基本理念遵守判定
            validation_results['backtest_principle_compliance'] = (
                validation_results['signal_columns_exist'] and
                validation_results['data_integrity'] and
                (validation_results['entry_signals_generated'] or validation_results['exit_signals_generated'])
            )
            
            if validation_results['backtest_principle_compliance']:
                print("✅ バックテスト基本理念遵守確認")
            else:
                print("❌ バックテスト基本理念違反検出")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ 検証エラー ({strategy_name}): {e}")
            print(f"詳細: {traceback.format_exc()}")
            return validation_results
    
    def test_vwap_breakout_strategy_comprehensive(self, market_data):
        """VWAPBreakoutStrategy包括的テスト"""
        print("\n🔍 VWAPBreakoutStrategy包括的テスト開始")
        print("="*70)
        
        strategy_name = 'VWAPBreakoutStrategy'
        test_result = {
            'module': 'strategies.VWAP_Breakout.VWAPBreakoutStrategy',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False,
            'backtest_compliance': False,
            'signal_stats': {},
            'data_validation': {}
        }
        
        try:
            # Step 1: インポート確認
            print("📦 Step 1: インポート確認")
            from strategies.VWAP_Breakout import VWAPBreakoutStrategy
            print("✅ インポート成功")
            
            # Step 2: パラメータ取得
            print("\n📋 Step 2: main.py準拠パラメータ取得")
            params = self.get_main_py_compatible_params(strategy_name)
            print(f"📋 取得パラメータ: {params}")
            
            # Step 3: main.py完全準拠実行
            print("\n🚀 Step 3: main.py完全準拠実行")
            execution_result = self.execute_strategy_with_main_py_pattern(
                VWAPBreakoutStrategy, strategy_name,
                market_data['stock_data'], market_data['index_data'], params
            )
            
            test_result['performance']['execution_time'] = execution_result['execution_time']
            
            if execution_result['success']:
                print("✅ main.py準拠実行成功")
                test_result['main_py_compatibility'] = True
                
                # Step 4: バックテスト出力検証
                print("\n🎯 Step 4: バックテスト基本理念遵守検証")
                validation = self.validate_backtest_output_comprehensive(
                    execution_result['result'], strategy_name
                )
                
                test_result['backtest_compliance'] = validation['backtest_principle_compliance']
                test_result['data_validation'] = validation
                
                # シグナル統計記録
                if validation['signal_columns_exist']:
                    result_df = execution_result['result']
                    entry_count = (result_df['Entry_Signal'] == 1).sum()
                    exit_count = (result_df['Exit_Signal'] != 0).sum()
                    
                    test_result['signal_stats'] = {
                        'entry_signals': int(entry_count),
                        'exit_signals': int(exit_count),
                        'data_rows': len(result_df)
                    }
                
                # Step 5: フォールバック検出
                print("\n🔍 Step 5: フォールバック機能検出")
                fallbacks = self.detect_strategy_fallbacks(
                    execution_result['result'], strategy_name, 'backtest_execution'
                )
                test_result['fallback_count'] = len(fallbacks)
                
                if fallbacks:
                    print(f"🚨 フォールバック検出: {len(fallbacks)}件")
                    for fb in fallbacks:
                        print(f"   - {fb}")
                else:
                    print("✅ フォールバック機能なし")
                
                # 異常パターン検出
                if validation['anomaly_detection']:
                    anomaly = validation['anomaly_detection']
                    print(f"🚨 異常パターン検出: {anomaly['type']} (severity: {anomaly['severity']})")
                    test_result['issues'].append(f"Anomaly detected: {anomaly['type']}")
                
            else:
                print(f"❌ 戦略実行失敗: {execution_result['error']}")
                test_result['issues'].append(f"Execution failed: {execution_result['error']}")
            
            # Step 6: 最終判定
            print("\n🎯 Step 6: 最終判定")
            if (test_result['main_py_compatibility'] and 
                test_result['backtest_compliance'] and 
                test_result['fallback_count'] == 0):
                test_result['status'] = 'GREEN'
                print("🟢 最終判定: 再利用可能")
                print("   - main.py完全互換")
                print("   - バックテスト基本理念遵守")
                print("   - フォールバック機能なし")
            elif (test_result['main_py_compatibility'] and 
                  test_result['fallback_count'] == 0):
                test_result['status'] = 'YELLOW'
                print("🟡 最終判定: 要注意")
                print("   - main.py互換だが軽微な問題あり")
            else:
                test_result['status'] = 'RED'
                print("🔴 最終判定: 再利用禁止")
                print("   - 基本理念違反またはフォールバック検出")
            
            self.test_results[strategy_name] = test_result
            return True
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"Critical error: {str(e)}")
            self.test_results[strategy_name] = test_result
            return False
    
    def generate_single_strategy_report(self):
        """単一戦略テストレポート生成"""
        report_lines = []
        
        report_lines.extend([
            "# VWAPBreakoutStrategy専門テストレポート",
            "",
            "## 🎯 テスト目的", 
            "main.py実証済みVWAPBreakoutStrategy（最高優先度戦略）の詳細検証",
            "バックテスト基本理念遵守確認と実際のstrategy.backtest()実行保証",
            "フォールバック機能の検出と再利用可能性の正確な判定",
            "",
            "## 📋 テスト対象",
            "",
            "- **戦略名**: VWAPBreakoutStrategy",
            "- **モジュールパス**: strategies.VWAP_Breakout",
            "- **main.py行番号**: 68 (最高優先度)",
            "- **初期化パターン**: data, index_data, params, price_column=\"Adj Close\"",
            "",
            "## 📊 テスト結果サマリー",
            ""
        ])
        
        if 'VWAPBreakoutStrategy' in self.test_results:
            result = self.test_results['VWAPBreakoutStrategy']
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            
            # 基本統計
            report_lines.extend([
                f"- **最終判定**: {status_emoji} {result['status']}",
                f"- **main.py互換性**: {'✅' if result.get('main_py_compatibility', False) else '❌'}",
                f"- **バックテスト基本理念遵守**: {'✅' if result.get('backtest_compliance', False) else '❌'}",
                f"- **フォールバック検出数**: {result['fallback_count']}",
                "",
                "---",
                ""
            ])
            
            # 詳細結果
            report_lines.extend([
                "## 📊 詳細テスト結果",
                "",
                f"**モジュール**: {result['module']}",
                f"**テスト状況**: {result['status']}",
                ""
            ])
            
            # パフォーマンス指標
            if result['performance']:
                report_lines.extend(["### ⚡ パフォーマンス指標", ""])
                for metric, value in result['performance'].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- **{metric}**: {value:.4f}秒")
                    else:
                        report_lines.append(f"- **{metric}**: {value}")
                report_lines.append("")
            
            # シグナル統計
            if result['signal_stats']:
                report_lines.extend(["### 📊 シグナル生成統計", ""])
                stats = result['signal_stats']
                report_lines.extend([
                    f"- **エントリーシグナル数**: {stats.get('entry_signals', 0)}",
                    f"- **エグジットシグナル数**: {stats.get('exit_signals', 0)}",
                    f"- **データ行数**: {stats.get('data_rows', 0)}",
                    ""
                ])
                
                # 取引率計算
                if stats.get('data_rows', 0) > 0:
                    entry_rate = (stats.get('entry_signals', 0) / stats.get('data_rows', 1)) * 100
                    report_lines.append(f"- **エントリー率**: {entry_rate:.2f}%")
                    report_lines.append("")
            
            # データ検証結果
            if result['data_validation']:
                validation = result['data_validation']
                report_lines.extend(["### 🎯 バックテスト基本理念検証", ""])
                
                checks = [
                    ('signal_columns_exist', 'Entry_Signal/Exit_Signal列存在'),
                    ('entry_signals_generated', 'エントリーシグナル生成'),
                    ('exit_signals_generated', 'エグジットシグナル生成'),
                    ('trade_count_positive', '取引数 > 0'),
                    ('data_integrity', 'データ整合性'),
                    ('backtest_principle_compliance', '基本理念遵守')
                ]
                
                for check_key, check_desc in checks:
                    status = "✅" if validation.get(check_key, False) else "❌"
                    report_lines.append(f"- **{check_desc}**: {status}")
                
                report_lines.append("")
                
                # 異常検出
                if validation.get('anomaly_detection'):
                    anomaly = validation['anomaly_detection']
                    report_lines.extend([
                        "### 🚨 異常パターン検出", "",
                        f"- **異常タイプ**: {anomaly['type']}",
                        f"- **深刻度**: {anomaly['severity']}",
                        f"- **エグジット/エントリー比率**: {anomaly.get('ratio', 'N/A'):.2f}",
                        ""
                    ])
            
            # 検出された問題
            if result['issues']:
                report_lines.extend(["### ⚠️ 検出された問題", ""])
                for issue in result['issues']:
                    report_lines.append(f"- {issue}")
                report_lines.append("")
            
            # フォールバック検出詳細
            if self.fallback_detected:
                report_lines.extend(["### 🔍 フォールバック機能検出詳細", ""])
                for fb in self.fallback_detected:
                    report_lines.append(f"- {fb}")
                report_lines.append("")
            
        else:
            report_lines.extend([
                "❌ **テスト未実行**: VWAPBreakoutStrategyのテスト結果が存在しません",
                ""
            ])
        
        # 総合評価
        report_lines.extend([
            "## 🎯 総合評価",
            ""
        ])
        
        if 'VWAPBreakoutStrategy' in self.test_results:
            result = self.test_results['VWAPBreakoutStrategy']
            
            if result['status'] == 'GREEN':
                report_lines.extend([
                    "🟢 **優秀**: VWAPBreakoutStrategyは新main.pyへの即座統合が可能",
                    "",
                    "**推奨事項**:",
                    "- comprehensive_module_testに最優先で組み込み",
                    "- 他戦略のテンプレートとして活用",
                    "- パフォーマンス指標を基準値として設定",
                    ""
                ])
            elif result['status'] == 'YELLOW':
                report_lines.extend([
                    "🟡 **良好**: 軽微な修正後に利用可能",
                    "",
                    "**推奨事項**:",
                    "- 検出された問題の個別修正",
                    "- 段階的統合テストの実施",
                    "- 継続的な品質監視",
                    ""
                ])
            else:
                report_lines.extend([
                    "🔴 **要改善**: 重大な問題により再利用不可",
                    "",
                    "**推奨事項**:",
                    "- 根本原因の詳細分析",
                    "- バックテスト基本理念の遵守修正",
                    "- 代替戦略の検討",
                    ""
                ])
        
        # 次のステップ
        report_lines.extend([
            "## 🚀 次のステップ",
            "",
            "### 他の戦略テスト計画",
            "",
            "1. **MomentumInvestingStrategy** (main.py line 69) - 高優先度",
            "2. **BreakoutStrategy** (main.py line 70) - 高優先度",
            "3. **VWAPBounceStrategy** (main.py line 71) - 中優先度",
            "4. **OpeningGapStrategy** (main.py line 72) - 要注意（TODO #4）",
            "5. **ContrarianStrategy** (main.py line 73) - 中優先度",
            "6. **GCStrategy** (main.py line 74) - 中優先度",
            "",
            "### バッチテスト実行",
            "",
            "VWAPBreakoutStrategyの結果を基準として、残り6戦略のバッチテストを実行し、",
            "main.py実証済み戦略の総合的な再利用可能性を評価する。",
            "",
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト対象**: VWAPBreakoutStrategy (最高優先度)",
            f"**テスト環境**: main.py準拠実データ使用",
            f"**バックテスト基本理念**: 実際のstrategy.backtest()実行保証"
        ])
        
        return "\n".join(report_lines)


def main():
    """VWAPBreakoutStrategy専門テストの実行"""
    print("🚀 VWAPBreakoutStrategy専門テスト開始")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 対象: VWAPBreakoutStrategy (main.py line 68, 最高優先度)")
    print("🎯 バックテスト基本理念遵守: 実際のstrategy.backtest()実行保証")
    print("="*70)
    
    tester = SingleStrategyTester()
    
    try:
        # 実際の市場データ取得
        print("📊 実際の市場データ取得開始...")
        market_data = tester.get_real_market_data()
        
        if market_data is None:
            print("❌ 実データ取得失敗 - テスト中断")
            return
        
        print(f"✅ テストデータ準備完了:")
        print(f"   ティッカー: {market_data['ticker']}")
        print(f"   期間: {market_data['start_date']} - {market_data['end_date']}")
        print(f"   データ行数: {len(market_data['stock_data'])}")
        
        # VWAPBreakoutStrategy包括的テスト実行
        success = tester.test_vwap_breakout_strategy_comprehensive(market_data)
        
        # レポート生成
        print("\n📄 テストレポート生成中...")
        report = tester.generate_single_strategy_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "vwap_breakout_strategy_test_report.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 実行サマリー
        print(f"\n" + "="*70)
        print(f"🎯 VWAPBreakoutStrategy専門テスト完了")
        print(f"="*70)
        print(f"📊 実行結果: {'✅ 成功' if success else '❌ 失敗'}")
        print(f"📄 詳細レポート: {output_file}")
        print(f"🚨 フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 最終判定表示
        if 'VWAPBreakoutStrategy' in tester.test_results:
            result = tester.test_results['VWAPBreakoutStrategy']
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            
            print(f"\n🎯 最終判定: {status_emoji} {result['status']}")
            print(f"   main.py互換性: {'✅' if result.get('main_py_compatibility', False) else '❌'}")
            print(f"   バックテスト基本理念: {'✅' if result.get('backtest_compliance', False) else '❌'}")
            
            if result['signal_stats']:
                stats = result['signal_stats']
                print(f"   エントリーシグナル: {stats.get('entry_signals', 0)}回")
                print(f"   エグジットシグナル: {stats.get('exit_signals', 0)}回")
            
            if result['status'] == 'GREEN':
                print(f"\n🚀 次のステップ推奨:")
                print(f"   1. comprehensive_module_testに即座統合")
                print(f"   2. 他戦略のテンプレートとして活用")
                print(f"   3. 残り6戦略のバッチテスト実行")
    
    except Exception as e:
        print(f"❌ テスト実行中に致命的エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")


if __name__ == "__main__":
    main()