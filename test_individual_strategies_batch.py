"""
個別戦略クラス専門テスト（全戦略バッチ版）
main.py実証済み全7戦略の包括的テスト - バックテスト基本理念遵守確認
"""
import sys
import os
import pandas as pd
import traceback
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class BatchStrategyTester:
    """全戦略バッチテスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.market_data = None
        
        # main.py戦略優先順位（line 776-782）
        self.strategy_priority = [
            ('VWAPBreakoutStrategy', 'strategies.VWAP_Breakout', 'VWAPBreakoutStrategy'),
            ('MomentumInvestingStrategy', 'strategies.Momentum_Investing', 'MomentumInvestingStrategy'),
            ('BreakoutStrategy', 'strategies.Breakout', 'BreakoutStrategy'),
            ('VWAPBounceStrategy', 'strategies.VWAP_Bounce', 'VWAPBounceStrategy'),
            ('OpeningGapStrategy', 'strategies.Opening_Gap', 'OpeningGapStrategy'),
            ('ContrarianStrategy', 'strategies.contrarian_strategy', 'ContrarianStrategy'),
            ('GCStrategy', 'strategies.gc_strategy_signal', 'GCStrategy')
        ]
        
    def detect_strategy_fallbacks(self, result, strategy_name, operation):
        """戦略特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: モックデータ使用検出
        if operation == "backtest_execution":
            if isinstance(result, pd.DataFrame):
                # データの異常パターンチェック
                if len(result) == 0:
                    fallbacks.append("Empty result DataFrame - possible mock data")
                elif 'Entry_Signal' not in result.columns or 'Exit_Signal' not in result.columns:
                    fallbacks.append("Missing signal columns - fallback to default structure")
                elif (result['Entry_Signal'] == 0).all() and (result['Exit_Signal'] == 0).all():
                    fallbacks.append("All zero signals - possible fallback behavior")
                    
                # TODO #4異常パターン検出 - より厳密な閾値に調整
                if 'Exit_Signal' in result.columns:
                    exit_signals = (result['Exit_Signal'] != 0).sum()
                    entry_signals = (result['Entry_Signal'] == 1).sum()
                    if exit_signals > 0 and entry_signals > 0:
                        exit_ratio = exit_signals / entry_signals
                        # 閾値を調整: 20倍以上で異常パターンとして検出
                        if exit_ratio > 20.0:  # 異常な高エグジット比率（厳密化）
                            fallbacks.append(f"Excessive exit signals (ratio: {exit_ratio:.1f}) - TODO #4 pattern")
        
        # パターン2: 計算失敗時のデフォルト値返却
        elif operation == "signal_generation":
            if isinstance(result, dict):
                if result.get('entry_signals', 0) == 0 and result.get('exit_signals', 0) == 0:
                    fallbacks.append("Zero signal generation - possible calculation failure fallback")
        
        if fallbacks:
            self.fallback_detected.extend([f"{strategy_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def get_real_market_data(self):
        """実際の市場データを取得（yfinanceベース）"""
        if self.market_data is not None:
            return self.market_data
            
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
            
            self.market_data = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'stock_data': stock_data,
                'index_data': index_data
            }
            
            return self.market_data
        
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
        """main.py準拠のデフォルトパラメータ（main.py lines 129-182）"""
        defaults = {
            'VWAPBreakoutStrategy': {
                'vwap_period': 20,
                'volume_threshold_multiplier': 1.5,
                'breakout_threshold': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'MomentumInvestingStrategy': {
                'momentum_period': 14,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'BreakoutStrategy': {
                'lookback_period': 20,
                'volume_threshold': 1.5,
                'breakout_threshold': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'VWAPBounceStrategy': {
                'vwap_period': 20,
                'deviation_threshold': 0.02,
                'volume_threshold': 1.2,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'OpeningGapStrategy': {
                'gap_threshold': 0.02,
                'volume_threshold': 1.5,
                'confirmation_period': 3,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'ContrarianStrategy': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08
            },
            'GCStrategy': {
                'short_window': 5,
                'long_window': 25,
                'stop_loss': 0.05,
                'take_profit': 0.10
            }
        }
        
        return defaults.get(strategy_name, {})
    
    def execute_strategy_with_main_py_pattern(self, strategy_class, strategy_name, stock_data, index_data, params):
        """main.py完全準拠の戦略実行パターン（main.py lines 329-356）"""
        try:
            print(f"🚀 {strategy_name} main.py準拠実行開始")
            
            # main.pyの_execute_individual_strategy準拠初期化
            if strategy_name == 'VWAPBreakoutStrategy':
                strategy = strategy_class(
                    data=stock_data.copy(),
                    index_data=index_data,
                    params=params,
                    price_column="Adj Close"
                )
            elif strategy_name == 'OpeningGapStrategy':
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params,
                    price_column="Adj Close",
                    dow_data=index_data
                )
            else:
                # 標準初期化パターン
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params
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
                
                # 異常エグジット検出（TODO #4対応）
                if entry_signals > 0:
                    exit_entry_ratio = exit_signals / entry_signals
                    print(f"  エグジット/エントリー比率: {exit_entry_ratio:.2f}")
                    
                    validation_results['anomaly_detection'] = {
                        'exit_entry_ratio': exit_entry_ratio,
                        'is_abnormal': exit_entry_ratio > 3.0,
                        'anomaly_type': 'excessive_exits' if exit_entry_ratio > 5.0 else 'high_exit_ratio' if exit_entry_ratio > 3.0 else 'normal'
                    }
                    
                    if validation_results['anomaly_detection']['is_abnormal']:
                        print(f"🚨 TODO #4異常パターン検出: {validation_results['anomaly_detection']['anomaly_type']}")
                else:
                    validation_results['anomaly_detection'] = {
                        'exit_entry_ratio': 0.0,
                        'is_abnormal': True,
                        'anomaly_type': 'no_entries'
                    }
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
    
    def test_individual_strategy(self, strategy_name, module_path, class_name, market_data):
        """個別戦略テスト実行"""
        print(f"\n🔍 {strategy_name} 包括的テスト開始")
        print("="*70)
        
        test_result = {
            'module': f'{module_path}.{class_name}',
            'status': 'UNKNOWN',
            'issues': [],
            'performance': {},
            'fallback_count': 0,
            'main_py_compatibility': False,
            'backtest_compliance': False,
            'signal_stats': {},
            'data_validation': {},
            'execution_time': 0
        }
        
        try:
            # Step 1: インポート確認
            print("📦 Step 1: インポート確認")
            try:
                module = __import__(module_path, fromlist=[class_name])
                strategy_class = getattr(module, class_name)
                print("✅ インポート成功")
            except Exception as e:
                print(f"❌ インポートエラー: {e}")
                test_result['issues'].append(f"Import failed: {str(e)}")
                test_result['status'] = 'RED'
                return test_result
            
            # Step 2: パラメータ取得
            print("\n📋 Step 2: main.py準拠パラメータ取得")
            params = self.get_main_py_compatible_params(strategy_name)
            print(f"📋 取得パラメータ: {params}")
            
            # Step 3: main.py完全準拠実行
            print("\n🚀 Step 3: main.py完全準拠実行")
            execution_result = self.execute_strategy_with_main_py_pattern(
                strategy_class, strategy_name,
                market_data['stock_data'], market_data['index_data'], params
            )
            
            test_result['performance']['execution_time'] = execution_result['execution_time']
            test_result['execution_time'] = execution_result['execution_time']
            
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
                    test_result['signal_stats'] = {
                        'entry_signals': int((result_df['Entry_Signal'] == 1).sum()),
                        'exit_signals': int((result_df['Exit_Signal'] != 0).sum()),
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
                    test_result['issues'].extend(fallbacks)
                else:
                    print("✅ フォールバック機能なし")
                
                # 異常パターン検出
                if validation['anomaly_detection'] and validation['anomaly_detection'].get('is_abnormal', False):
                    anomaly_type = validation['anomaly_detection']['anomaly_type']
                    test_result['issues'].append(f"TODO #4 anomaly detected: {anomaly_type}")
                
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
            
            return test_result
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"Critical error: {str(e)}")
            return test_result
    
    def run_batch_test(self):
        """全戦略バッチテスト実行"""
        print("🚀 全戦略バッチテスト開始")
        print("="*70)
        print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 対象: main.py実証済み全7戦略")
        print("🎯 バックテスト基本理念遵守: 実際のstrategy.backtest()実行保証")
        print("="*70)
        
        # 実際の市場データ取得
        print("📊 実際の市場データ取得開始...")
        market_data = self.get_real_market_data()
        
        if market_data is None:
            print("❌ 実データ取得失敗 - バッチテスト中断")
            return False
        
        print(f"✅ テストデータ準備完了:")
        print(f"   ティッカー: {market_data['ticker']}")
        print(f"   期間: {market_data['start_date']} - {market_data['end_date']}")
        print(f"   データ行数: {len(market_data['stock_data'])}")
        
        # 全戦略テスト実行
        for strategy_name, module_path, class_name in self.strategy_priority:
            print(f"\n{'='*70}")
            print(f"📋 戦略テスト: {strategy_name}")
            print(f"📦 モジュール: {module_path}")
            print(f"🏷️ クラス: {class_name}")
            print(f"{'='*70}")
            
            test_result = self.test_individual_strategy(
                strategy_name, module_path, class_name, market_data
            )
            
            self.test_results[strategy_name] = test_result
            
            # 進捗表示
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[test_result['status']]
            print(f"\n📊 {strategy_name}: {status_emoji} {test_result['status']}")
            
            if test_result['signal_stats']:
                stats = test_result['signal_stats']
                print(f"   エントリー: {stats.get('entry_signals', 0)}回")
                print(f"   エグジット: {stats.get('exit_signals', 0)}回")
                print(f"   実行時間: {test_result['execution_time']:.3f}秒")
        
        return True
    
    def generate_batch_report(self):
        """バッチテストレポート生成"""
        report_lines = []
        
        # ヘッダー
        report_lines.extend([
            "# 全戦略個別テスト包括レポート",
            "",
            "## 🎯 テスト目的", 
            "main.py実証済み全7戦略の詳細検証",
            "バックテスト基本理念遵守確認と実際のstrategy.backtest()実行保証",
            "フォールバック機能の検出と再利用可能性の正確な判定",
            "TODO #4異常パターン（過度なエグジット）の検出",
            "",
            "## 📋 テスト対象戦略",
            ""
        ])
        
        # 戦略リスト
        for i, (strategy_name, module_path, class_name) in enumerate(self.strategy_priority, 1):
            status = self.test_results.get(strategy_name, {}).get('status', 'UNKNOWN')
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[status]
            report_lines.append(f"{i}. **{strategy_name}** ({module_path}) - {status_emoji} {status}")
        
        report_lines.append("")
        
        # 全体統計
        total_strategies = len(self.strategy_priority)
        green_count = sum(1 for result in self.test_results.values() if result.get('status') == 'GREEN')
        yellow_count = sum(1 for result in self.test_results.values() if result.get('status') == 'YELLOW')
        red_count = sum(1 for result in self.test_results.values() if result.get('status') == 'RED')
        
        report_lines.extend([
            "## 📊 テスト結果サマリー",
            "",
            f"- **総戦略数**: {total_strategies}",
            f"- **🟢 GREEN (再利用可能)**: {green_count}戦略",
            f"- **🟡 YELLOW (要注意)**: {yellow_count}戦略", 
            f"- **🔴 RED (再利用禁止)**: {red_count}戦略",
            f"- **再利用可能率**: {(green_count/total_strategies)*100:.1f}%",
            "",
            "---",
            ""
        ])
        
        # 戦略別詳細結果
        report_lines.extend([
            "## 📋 戦略別詳細結果",
            ""
        ])
        
        for strategy_name, module_path, class_name in self.strategy_priority:
            if strategy_name in self.test_results:
                result = self.test_results[strategy_name]
                status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
                
                report_lines.extend([
                    f"### {status_emoji} {strategy_name}",
                    "",
                    f"**モジュール**: {result['module']}",
                    f"**最終判定**: {status_emoji} {result['status']}",
                    f"**main.py互換性**: {'✅' if result.get('main_py_compatibility', False) else '❌'}",
                    f"**バックテスト基本理念遵守**: {'✅' if result.get('backtest_compliance', False) else '❌'}",
                    f"**フォールバック検出数**: {result['fallback_count']}",
                    ""
                ])
                
                # パフォーマンス指標
                if result.get('execution_time', 0) > 0:
                    report_lines.append(f"**実行時間**: {result['execution_time']:.3f}秒")
                
                # シグナル統計
                if result['signal_stats']:
                    stats = result['signal_stats']
                    report_lines.extend([
                        "**シグナル統計**:",
                        f"- エントリーシグナル: {stats.get('entry_signals', 0)}回",
                        f"- エグジットシグナル: {stats.get('exit_signals', 0)}回",
                        f"- データ行数: {stats.get('data_rows', 0)}",
                        ""
                    ])
                    
                    # 取引率計算
                    if stats.get('data_rows', 0) > 0:
                        entry_rate = (stats.get('entry_signals', 0) / stats.get('data_rows', 1)) * 100
                        report_lines.append(f"- エントリー率: {entry_rate:.2f}%")
                    
                    report_lines.append("")
                
                # 異常検出結果
                if result.get('data_validation', {}).get('anomaly_detection'):
                    anomaly = result['data_validation']['anomaly_detection']
                    if anomaly.get('is_abnormal', False):
                        report_lines.extend([
                            "**🚨 異常パターン検出**:",
                            f"- 異常タイプ: {anomaly.get('anomaly_type', 'unknown')}",
                            f"- エグジット/エントリー比率: {anomaly.get('exit_entry_ratio', 0):.2f}",
                            ""
                        ])
                
                # 検出された問題
                if result['issues']:
                    report_lines.extend([
                        "**検出された問題**:",
                        ""
                    ])
                    for issue in result['issues']:
                        report_lines.append(f"- {issue}")
                    report_lines.append("")
                
                report_lines.append("---")
                report_lines.append("")
            else:
                report_lines.extend([
                    f"### ❓ {strategy_name}",
                    "",
                    "❌ **テスト未実行**: 結果が存在しません",
                    "",
                    "---",
                    ""
                ])
        
        # フォールバック検出総合
        if self.fallback_detected:
            report_lines.extend([
                "## 🚨 フォールバック検出詳細",
                "",
                f"**総検出数**: {len(self.fallback_detected)}件",
                ""
            ])
            for fb in self.fallback_detected:
                report_lines.append(f"- {fb}")
            report_lines.append("")
        
        # 総合評価
        report_lines.extend([
            "## 🎯 総合評価",
            ""
        ])
        
        if green_count == total_strategies:
            report_lines.extend([
                "🟢 **優秀**: 全戦略が再利用可能",
                "",
                "全戦略がmain.py完全互換でバックテスト基本理念を遵守しています。",
                "フォールバック機能の使用もなく、高品質な戦略実装が確認されました。"
            ])
        elif green_count >= total_strategies * 0.7:
            report_lines.extend([
                "🟡 **良好**: 大部分の戦略が再利用可能",
                "",
                f"{green_count}/{total_strategies}戦略が再利用可能です。",
                "一部の戦略で軽微な問題が検出されましたが、全体的に高品質です。"
            ])
        else:
            report_lines.extend([
                "🔴 **要改善**: 多数の戦略で問題検出",
                "",
                f"再利用可能な戦略は{green_count}/{total_strategies}戦略のみです。",
                "複数の戦略でバックテスト基本理念違反やフォールバック使用が検出されました。"
            ])
        
        report_lines.extend([
            "",
            "## 🚀 推奨アクション",
            ""
        ])
        
        # GREEN戦略
        green_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'GREEN']
        if green_strategies:
            report_lines.extend([
                "### 🟢 GREEN戦略（即座に利用可能）",
                ""
            ])
            for strategy in green_strategies:
                report_lines.append(f"- **{strategy}**: そのまま本番環境で利用可能")
            report_lines.append("")
        
        # YELLOW戦略
        yellow_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'YELLOW']
        if yellow_strategies:
            report_lines.extend([
                "### 🟡 YELLOW戦略（改善推奨）",
                ""
            ])
            for strategy in yellow_strategies:
                result = self.test_results[strategy]
                issues = ", ".join(result.get('issues', [])[:2])  # 最初の2つの問題のみ
                report_lines.append(f"- **{strategy}**: {issues}")
            report_lines.append("")
        
        # RED戦略
        red_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'RED']
        if red_strategies:
            report_lines.extend([
                "### 🔴 RED戦略（使用禁止）",
                ""
            ])
            for strategy in red_strategies:
                result = self.test_results[strategy]
                main_issue = result.get('issues', ['Unknown error'])[0]
                report_lines.append(f"- **{strategy}**: {main_issue}")
            report_lines.append("")
        
        # フッター
        report_lines.extend([
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト対象**: main.py実証済み全7戦略",
            f"**テスト環境**: main.py準拠実データ使用",
            f"**バックテスト基本理念**: 実際のstrategy.backtest()実行保証",
            f"**総フォールバック検出**: {len(self.fallback_detected)}件"
        ])
        
        return "\n".join(report_lines)


def main():
    """全戦略バッチテストの実行"""
    tester = BatchStrategyTester()
    
    try:
        # バッチテスト実行
        success = tester.run_batch_test()
        
        if not success:
            print("❌ バッチテスト実行失敗")
            return
        
        # レポート生成
        print("\n📄 バッチテストレポート生成中...")
        report = tester.generate_batch_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"batch_strategy_test_report_{timestamp}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # JSON形式の詳細結果も保存
        json_file = output_dir / f"batch_strategy_test_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(tester.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 実行サマリー
        print(f"\n" + "="*70)
        print(f"🎯 全戦略バッチテスト完了")
        print(f"="*70)
        print(f"📊 実行結果: {'✅ 成功' if success else '❌ 失敗'}")
        print(f"📄 詳細レポート: {output_file}")
        print(f"📋 JSON結果: {json_file}")
        print(f"🚨 総フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 統計表示
        total_strategies = len(tester.strategy_priority)
        green_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'GREEN')
        yellow_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'YELLOW')
        red_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'RED')
        
        print(f"\n🎯 最終統計:")
        print(f"   🟢 GREEN (再利用可能): {green_count}/{total_strategies}戦略")
        print(f"   🟡 YELLOW (要注意): {yellow_count}/{total_strategies}戦略")
        print(f"   🔴 RED (再利用禁止): {red_count}/{total_strategies}戦略")
        print(f"   📊 再利用可能率: {(green_count/total_strategies)*100:.1f}%")
        
        # 詳細表示
        print(f"\n📋 戦略別結果:")
        for strategy_name, _, _ in tester.strategy_priority:
            if strategy_name in tester.test_results:
                result = tester.test_results[strategy_name]
                status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
                signal_stats = result.get('signal_stats', {})
                entry_count = signal_stats.get('entry_signals', 0)
                exit_count = signal_stats.get('exit_signals', 0)
                exec_time = result.get('execution_time', 0)
                
                print(f"   {status_emoji} {strategy_name}: E{entry_count}/X{exit_count} ({exec_time:.3f}s)")
        
        print(f"\n🚀 次のステップ:")
        if green_count == total_strategies:
            print("   ✅ 全戦略がGREEN - comprehensive_module_testに即座統合可能")
        elif green_count > 0:
            print(f"   🟢 {green_count}戦略は即座利用可能")
            print(f"   🔄 {yellow_count + red_count}戦略の改善作業が必要")
        else:
            print("   🚨 全戦略で問題検出 - 根本的な見直しが必要")
    
    except Exception as e:
        print(f"❌ バッチテスト実行中に致命的エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")


if __name__ == "__main__":
    main()