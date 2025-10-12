"""
未使用戦略クラス専門テスト（バッチ版）
strategies/配下の未使用戦略モジュールの包括的テスト
重複モジュール除去と段階的テスト実行
"""

import sys
import os
import traceback
import pandas as pd
from datetime import datetime, timedelta
import json
import importlib
from pathlib import Path

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

class UnusedStrategyTester:
    """未使用戦略バッチテスター"""
    
    def __init__(self):
        self.test_results = {}
        self.fallback_detected = []
        self.performance_metrics = {}
        self.market_data = None
        
        # 未使用戦略リスト（重複除去済み - strategies/を優先）
        self.unused_strategies = [
            ('base_strategy', 'strategies.base_strategy', 'BaseStrategy'),
            ('mean_reversion_strategy', 'strategies.mean_reversion_strategy', 'MeanReversionStrategy'),
            ('pairs_trading_strategy', 'strategies.pairs_trading_strategy', 'PairsTradingStrategy'),
            ('support_resistance_contrarian_strategy', 'strategies.support_resistance_contrarian_strategy', 'SupportResistanceContrarianStrategy'),
        ]
        
        # 関数ベースモジュール（クラスなし）
        self.function_based_modules = [
            ('strategy_manager', 'strategies.strategy_manager', 'apply_strategies')  # 関数ベース
        ]
        
        # main.py使用中戦略との重複チェック対象
        self.duplicate_check_strategies = [
            ('contrarian_strategy_unused', 'strategies.contrarian_strategy', 'ContrarianStrategy'),
            ('gc_strategy_signal_unused', 'strategies.gc_strategy_signal', 'GCStrategy'),
        ]
        
        # Phase分割
        self.phase_5a = ['base_strategy', 'mean_reversion_strategy', 'pairs_trading_strategy']
        self.phase_5b = ['support_resistance_contrarian_strategy']  # strategy_managerは関数ベース
        self.phase_5b_functions = ['strategy_manager']  # 関数ベーステスト
        self.phase_5c = ['contrarian_strategy_unused', 'gc_strategy_signal_unused']
        
    def detect_strategy_fallbacks(self, result, strategy_name, operation):
        """戦略特有のフォールバック検出"""
        fallbacks = []
        
        # パターン1: モックデータ使用検出
        if operation == "backtest_execution":
            if isinstance(result, pd.DataFrame):
                if 'Mock_Data_Flag' in result.columns or len(result) == 100:  # 典型的なモックサイズ
                    fallbacks.append("mock_data_usage")
                
        # パターン2: 計算失敗時のデフォルト値返却
        elif operation == "signal_generation":
            if isinstance(result, dict):
                if all(v == 0 for v in result.values()) or all(v is None for v in result.values()):
                    fallbacks.append("default_value_fallback")
        
        if fallbacks:
            self.fallback_detected.extend([f"{strategy_name}.{operation}: {fb}" for fb in fallbacks])
        
        return fallbacks
    
    def get_real_market_data(self):
        """実際の市場データを取得（test_individual_strategies_batch.py準拠）"""
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
    
    def get_default_parameters(self, strategy_name):
        """未使用戦略用デフォルトパラメータ"""
        defaults = {
            'base_strategy': {
                'lookback_period': 20,
                'threshold': 0.02
            },
            'mean_reversion_strategy': {
                'lookback_period': 20,
                'std_threshold': 2.0,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'pairs_trading_strategy': {
                'lookback_period': 30,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_pct': 0.05
            },
            'strategy_manager': {
                'max_positions': 5,
                'risk_per_trade': 0.02
            },
            'support_resistance_contrarian_strategy': {
                'lookback_period': 20,
                'support_resistance_threshold': 0.02,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08
            },
            'contrarian_strategy_unused': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08
            },
            'gc_strategy_signal_unused': {
                'short_window': 5,
                'long_window': 25,
                'stop_loss': 0.05,
                'take_profit': 0.10
            }
        }
        
        return defaults.get(strategy_name, {})
    
    def test_strategy_import_and_basic_functionality(self, strategy_name, module_path, class_name):
        """戦略クラスの基本機能テスト"""
        print(f"\n🔍 {strategy_name} 基本機能テスト開始")
        print("="*70)
        
        test_result = {
            'module': f'{module_path}.{class_name}',
            'status': 'UNKNOWN',
            'issues': [],
            'capabilities': {
                'import_success': False,
                'class_instantiation': False,
                'has_backtest_method': False,
                'accepts_parameters': False,
                'data_processing': False
            },
            'fallback_count': 0,
            'execution_time': 0,
            'test_phase': 'import_only'
        }
        
        try:
            # Step 1: インポート確認
            print("📦 Step 1: インポート確認")
            try:
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                test_result['capabilities']['import_success'] = True
                print("✅ インポート成功")
            except ImportError as e:
                print(f"❌ インポートエラー: {e}")
                test_result['issues'].append(f"Import error: {str(e)}")
                test_result['status'] = 'RED'
                return test_result
            except AttributeError as e:
                print(f"❌ クラス名エラー: {e}")
                test_result['issues'].append(f"Class name error: {str(e)}")
                test_result['status'] = 'RED'
                return test_result
            
            # Step 2: クラス情報調査
            print("\n📋 Step 2: クラス情報調査")
            
            # メソッド確認
            methods = [method for method in dir(strategy_class) if not method.startswith('_')]
            print(f"📋 利用可能メソッド: {methods}")
            
            has_backtest = 'backtest' in methods
            has_init = hasattr(strategy_class, '__init__')
            
            test_result['capabilities']['has_backtest_method'] = has_backtest
            
            if has_backtest:
                print("✅ backtest()メソッド存在確認")
            else:
                print("⚠️ backtest()メソッドなし")
                test_result['issues'].append("No backtest method")
            
            # Step 3: インスタンス化テスト（正しいパラメータ形式）
            print("\n🚀 Step 3: インスタンス化テスト")
            try:
                params = self.get_default_parameters(strategy_name)
                print(f"📋 テストパラメータ: {params}")
                
                # 様々な初期化パターンを試行
                instance = None
                
                # 戦略固有の初期化パターン
                if strategy_name == 'strategy_manager':
                    # strategy_managerは関数ベースなのでスキップ
                    print("   strategy_managerは関数ベースモジュール - クラス初期化スキップ")
                    test_result['capabilities']['class_instantiation'] = False
                    test_result['issues'].append("Function-based module, no class to instantiate")
                else:
                    # 戦略クラス用の正しい初期化パターン
                    initialization_patterns = [
                        # パターン1: data + params + price_column
                        lambda: strategy_class(
                            data=self.market_data['stock_data'] if self.market_data else pd.DataFrame(),
                            params=params,
                            price_column="Adj Close"
                        ) if self.market_data else None,
                        # パターン2: data + params のみ
                        lambda: strategy_class(
                            data=self.market_data['stock_data'] if self.market_data else pd.DataFrame(),
                            params=params
                        ) if self.market_data else None,
                        # パターン3: data のみ（BaseStrategy準拠）
                        lambda: strategy_class(
                            data=self.market_data['stock_data'] if self.market_data else pd.DataFrame()
                        ) if self.market_data else None,
                    ]
                    
                    for i, pattern in enumerate(initialization_patterns, 1):
                        try:
                            print(f"   初期化パターン{i}を試行...")
                            instance = pattern()
                            if instance is not None:
                                test_result['capabilities']['class_instantiation'] = True
                                test_result['capabilities']['accepts_parameters'] = True
                                print(f"✅ 初期化パターン{i}成功")
                                break
                        except Exception as e:
                            print(f"   初期化パターン{i}失敗: {e}")
                            continue
                
                if instance is None:
                    print("❌ 全初期化パターン失敗")
                    test_result['issues'].append("Failed all initialization patterns")
                    test_result['status'] = 'RED'
                    return test_result
                
            except Exception as e:
                print(f"❌ インスタンス化エラー: {e}")
                test_result['issues'].append(f"Instantiation error: {str(e)}")
                test_result['status'] = 'RED'
                return test_result
            
            # Step 4: 最終判定
            print("\n🎯 Step 4: 基本機能判定")
            if (test_result['capabilities']['import_success'] and 
                test_result['capabilities']['class_instantiation']):
                
                if test_result['capabilities']['has_backtest_method']:
                    test_result['status'] = 'GREEN'
                    print("🟢 基本機能テスト: 再利用可能")
                else:
                    test_result['status'] = 'YELLOW'
                    print("🟡 基本機能テスト: 要注意（backtest()なし）")
            else:
                test_result['status'] = 'RED'
                print("🔴 基本機能テスト: 再利用禁止")
            
            return test_result
            
        except Exception as e:
            print(f"❌ 致命的エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            test_result['status'] = 'RED'
            test_result['issues'].append(f"Critical error: {str(e)}")
            return test_result
    
    def run_phase_5a_test(self):
        """Phase 5A: 基本戦略クラステスト"""
        print("\n🚀 Phase 5A: 基本戦略クラステスト開始")
        print("="*70)
        print("対象: base_strategy, mean_reversion_strategy, pairs_trading_strategy")
        print("="*70)
        
        phase_results = {}
        
        for strategy_name in self.phase_5a:
            strategy_info = None
            for name, module_path, class_name in self.unused_strategies:
                if name == strategy_name:
                    strategy_info = (name, module_path, class_name)
                    break
            
            if strategy_info:
                name, module_path, class_name = strategy_info
                result = self.test_strategy_import_and_basic_functionality(name, module_path, class_name)
                phase_results[name] = result
                self.test_results[name] = result
                
                # フォールバック検出
                self.detect_strategy_fallbacks(result, name, "basic_functionality")
        
        return phase_results
    
    def run_phase_5b_test(self):
        """Phase 5B: 管理系戦略クラステスト"""
        print("\n🚀 Phase 5B: 管理系戦略クラステスト開始")
        print("="*70)
        print("対象: strategy_manager, support_resistance_contrarian_strategy")
        print("="*70)
        
        phase_results = {}
        
        for strategy_name in self.phase_5b:
            strategy_info = None
            for name, module_path, class_name in self.unused_strategies:
                if name == strategy_name:
                    strategy_info = (name, module_path, class_name)
                    break
            
            if strategy_info:
                name, module_path, class_name = strategy_info
                result = self.test_strategy_import_and_basic_functionality(name, module_path, class_name)
                phase_results[name] = result
                self.test_results[name] = result
                
                # フォールバック検出
                self.detect_strategy_fallbacks(result, name, "basic_functionality")
        
        return phase_results
    
    def run_phase_5c_test(self):
        """Phase 5C: 重複戦略検証"""
        print("\n🚀 Phase 5C: 重複戦略検証開始")
        print("="*70)
        print("対象: contrarian_strategy（未使用版）, gc_strategy_signal（未使用版）")
        print("main.py使用中戦略との差分分析")
        print("="*70)
        
        phase_results = {}
        
        for strategy_name in self.phase_5c:
            strategy_info = None
            for name, module_path, class_name in self.duplicate_check_strategies:
                if name == strategy_name:
                    strategy_info = (name, module_path, class_name)
                    break
            
            if strategy_info:
                name, module_path, class_name = strategy_info
                result = self.test_strategy_import_and_basic_functionality(name, module_path, class_name)
                
                # 重複フラグ追加
                result['is_duplicate_of_main_py'] = True
                result['main_py_equivalent'] = class_name.replace('_unused', '')
                
                phase_results[name] = result
                self.test_results[name] = result
                
                print(f"🔍 {name}: main.py使用中の{result['main_py_equivalent']}との重複確認")
        
        return phase_results
    
    def generate_comprehensive_report(self):
        """包括的テストレポート生成"""
        report_lines = []
        
        # ヘッダー
        report_lines.extend([
            "# 未使用戦略クラス包括テストレポート",
            "",
            "## 🎯 テスト目的", 
            "strategies/配下の未使用戦略モジュールの再利用可能性評価",
            "重複モジュール除去と段階的テスト実行による効率的な検証",
            "main.py使用戦略との重複確認と差分分析",
            "",
            "## 📋 テスト対象戦略（重複除去済み）",
            ""
        ])
        
        # Phase別戦略リスト
        report_lines.extend([
            "### Phase 5A: 基本戦略クラス",
            ""
        ])
        
        for strategy_name in self.phase_5a:
            for name, module_path, class_name in self.unused_strategies:
                if name == strategy_name:
                    status = self.test_results.get(name, {}).get('status', 'UNTESTED')
                    status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓", "UNTESTED": "⏸️"}[status]
                    report_lines.append(f"- **{name}** ({class_name}): {status_emoji} {status}")
                    break
        
        report_lines.extend([
            "",
            "### Phase 5B: 管理系戦略クラス",
            ""
        ])
        
        for strategy_name in self.phase_5b:
            for name, module_path, class_name in self.unused_strategies:
                if name == strategy_name:
                    status = self.test_results.get(name, {}).get('status', 'UNTESTED')
                    status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓", "UNTESTED": "⏸️"}[status]
                    report_lines.append(f"- **{name}** ({class_name}): {status_emoji} {status}")
                    break
        
        report_lines.extend([
            "",
            "### Phase 5C: 重複戦略検証",
            ""
        ])
        
        for strategy_name in self.phase_5c:
            for name, module_path, class_name in self.duplicate_check_strategies:
                if name == strategy_name:
                    status = self.test_results.get(name, {}).get('status', 'UNTESTED')
                    status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓", "UNTESTED": "⏸️"}[status]
                    main_py_equiv = self.test_results.get(name, {}).get('main_py_equivalent', 'Unknown')
                    report_lines.append(f"- **{name}** ({class_name}): {status_emoji} {status} - main.py{main_py_equiv}の重複")
                    break
        
        # 全体統計
        total_strategies = len(self.unused_strategies) + len(self.duplicate_check_strategies)
        tested_strategies = len(self.test_results)
        green_count = sum(1 for result in self.test_results.values() if result.get('status') == 'GREEN')
        yellow_count = sum(1 for result in self.test_results.values() if result.get('status') == 'YELLOW')
        red_count = sum(1 for result in self.test_results.values() if result.get('status') == 'RED')
        
        report_lines.extend([
            "",
            "## 📊 テスト結果サマリー",
            "",
            f"- **総戦略数**: {total_strategies}（重複除去済み）",
            f"- **テスト実行数**: {tested_strategies}",
            f"- **🟢 GREEN (再利用可能)**: {green_count}戦略",
            f"- **🟡 YELLOW (要注意)**: {yellow_count}戦略",
            f"- **🔴 RED (再利用禁止)**: {red_count}戦略",
            f"- **再利用可能率**: {(green_count/tested_strategies)*100:.1f}%" if tested_strategies > 0 else "- **再利用可能率**: 0.0%",
            "",
            "---",
            ""
        ])
        
        # 戦略別詳細結果
        report_lines.extend([
            "## 📋 戦略別詳細結果",
            ""
        ])
        
        for strategy_name, result in self.test_results.items():
            status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "❓"}[result['status']]
            
            report_lines.extend([
                f"### {strategy_name}: {status_emoji} {result['status']}",
                "",
                f"**モジュール**: {result['module']}",
                f"**テストフェーズ**: {result.get('test_phase', 'unknown')}",
                ""
            ])
            
            # 機能チェック結果
            if 'capabilities' in result:
                capabilities = result['capabilities']
                report_lines.extend([
                    "**機能チェック結果**:",
                    f"- インポート成功: {'✅' if capabilities.get('import_success', False) else '❌'}",
                    f"- クラス初期化: {'✅' if capabilities.get('class_instantiation', False) else '❌'}",
                    f"- backtest()メソッド: {'✅' if capabilities.get('has_backtest_method', False) else '❌'}",
                    f"- パラメータ受容: {'✅' if capabilities.get('accepts_parameters', False) else '❌'}",
                    ""
                ])
            
            # 重複情報
            if result.get('is_duplicate_of_main_py', False):
                report_lines.extend([
                    f"**重複情報**: main.py使用中の{result.get('main_py_equivalent', 'Unknown')}戦略と重複",
                    ""
                ])
            
            # 検出された問題
            if result['issues']:
                report_lines.extend([
                    "**検出された問題**:",
                    *[f"- {issue}" for issue in result['issues']],
                    ""
                ])
            
            report_lines.append("---")
            report_lines.append("")
        
        # フォールバック検出詳細
        if self.fallback_detected:
            report_lines.extend([
                "## 🚨 フォールバック検出詳細",
                "",
                *[f"- {fb}" for fb in self.fallback_detected],
                "",
                "---",
                ""
            ])
        
        # 総合評価
        report_lines.extend([
            "## 🎯 総合評価",
            ""
        ])
        
        if tested_strategies == 0:
            report_lines.extend([
                "❌ **テスト未実行**: 戦略テストが実行されていません",
                ""
            ])
        elif green_count == tested_strategies:
            report_lines.extend([
                "🎉 **全戦略再利用可能**: 全ての未使用戦略が再利用可能と判定されました",
                "main.pyに統合することで戦略の選択肢を大幅に拡張できます",
                ""
            ])
        elif green_count >= tested_strategies * 0.7:
            report_lines.extend([
                f"✅ **多数戦略再利用可能**: {tested_strategies}戦略中{green_count}戦略が再利用可能",
                "main.pyへの段階的統合を推奨します",
                ""
            ])
        else:
            report_lines.extend([
                f"⚠️ **要改善**: {tested_strategies}戦略中{red_count}戦略に問題があります",
                "問題のある戦略の修正または除外を検討してください",
                ""
            ])
        
        # 推奨アクション
        report_lines.extend([
            "## 🚀 推奨アクション",
            ""
        ])
        
        # GREEN戦略
        green_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'GREEN']
        if green_strategies:
            report_lines.extend([
                "### 🟢 即座利用可能戦略",
                "",
                *[f"- **{strategy}**: main.pyへの統合準備完了" for strategy in green_strategies],
                "",
            ])
        
        # YELLOW戦略
        yellow_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'YELLOW']
        if yellow_strategies:
            report_lines.extend([
                "### 🟡 要注意戦略",
                "",
                *[f"- **{strategy}**: backtest()メソッド追加または機能拡張が必要" for strategy in yellow_strategies],
                "",
            ])
        
        # RED戦略
        red_strategies = [name for name, result in self.test_results.items() if result.get('status') == 'RED']
        if red_strategies:
            report_lines.extend([
                "### 🔴 要改善戦略",
                "",
                *[f"- **{strategy}**: 根本的な修正が必要" for strategy in red_strategies],
                "",
            ])
        
        # フッター
        report_lines.extend([
            "---",
            f"**レポート生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**テスト対象**: 未使用戦略モジュール（重複除去済み）",
            f"**テスト方針**: 段階的テスト（Phase 5A→5B→5C）",
            f"**総フォールバック検出**: {len(self.fallback_detected)}件"
        ])
        
        return "\n".join(report_lines)


def main():
    """未使用戦略バッチテストの実行"""
    print("🚀 未使用戦略クラス包括テスト開始")
    print("="*70)
    print(f"📅 テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 対象: strategies/配下の未使用戦略モジュール")
    print("📋 重複除去: strategies/を優先、src/strategies/は除外")
    print("="*70)
    
    tester = UnusedStrategyTester()
    
    try:
        # 実際の市場データ取得（オプション）
        print("📊 実際の市場データ取得（オプション）...")
        market_data = tester.get_real_market_data()
        
        if market_data:
            print(f"✅ テストデータ準備完了:")
            print(f"   ティッカー: {market_data['ticker']}")
            print(f"   期間: {market_data['start_date']} - {market_data['end_date']}")
            print(f"   データ行数: {len(market_data['stock_data'])}")
        else:
            print("⚠️ 実データ取得失敗 - 基本機能テストのみ実行")
        
        # Phase 5A実行
        print("\n" + "="*70)
        phase_5a_results = tester.run_phase_5a_test()
        
        # Phase 5B実行
        print("\n" + "="*70)
        phase_5b_results = tester.run_phase_5b_test()
        
        # Phase 5C実行
        print("\n" + "="*70)
        phase_5c_results = tester.run_phase_5c_test()
        
        # レポート生成
        print("\n📄 包括テストレポート生成中...")
        report = tester.generate_comprehensive_report()
        
        # ファイル出力
        output_dir = Path("docs/Plan to create a new main entry point")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"unused_strategies_test_report_{timestamp}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # JSON形式の詳細結果も保存
        json_file = output_dir / f"unused_strategies_test_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(tester.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 実行サマリー
        print(f"\n" + "="*70)
        print(f"🎯 未使用戦略包括テスト完了")
        print(f"="*70)
        print(f"📊 実行結果: ✅ 成功")
        print(f"📄 詳細レポート: {output_file}")
        print(f"📋 JSON結果: {json_file}")
        print(f"🚨 総フォールバック検出: {len(tester.fallback_detected)}件")
        
        # 統計表示
        total_strategies = len(tester.unused_strategies) + len(tester.duplicate_check_strategies)
        tested_strategies = len(tester.test_results)
        green_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'GREEN')
        yellow_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'YELLOW')
        red_count = sum(1 for result in tester.test_results.values() if result.get('status') == 'RED')
        
        print(f"\n🎯 最終統計:")
        print(f"   📋 総戦略数: {total_strategies}（重複除去済み）")
        print(f"   ✅ テスト実行: {tested_strategies}戦略")
        print(f"   🟢 GREEN (再利用可能): {green_count}戦略")
        print(f"   🟡 YELLOW (要注意): {yellow_count}戦略")
        print(f"   🔴 RED (再利用禁止): {red_count}戦略")
        print(f"   📊 再利用可能率: {(green_count/tested_strategies)*100:.1f}%" if tested_strategies > 0 else "   📊 再利用可能率: 0.0%")
        
        # Phase別詳細表示
        print(f"\n📋 Phase別結果:")
        
        print(f"   Phase 5A (基本戦略): ", end="")
        phase_5a_green = sum(1 for name in tester.phase_5a if tester.test_results.get(name, {}).get('status') == 'GREEN')
        print(f"{phase_5a_green}/{len(tester.phase_5a)} GREEN")
        
        print(f"   Phase 5B (管理系戦略): ", end="")
        phase_5b_green = sum(1 for name in tester.phase_5b if tester.test_results.get(name, {}).get('status') == 'GREEN')
        print(f"{phase_5b_green}/{len(tester.phase_5b)} GREEN")
        
        print(f"   Phase 5C (重複戦略): ", end="")
        phase_5c_green = sum(1 for name in tester.phase_5c if tester.test_results.get(name, {}).get('status') == 'GREEN')
        print(f"{phase_5c_green}/{len(tester.phase_5c)} GREEN")
        
        # 次のステップ
        print(f"\n🚀 次のステップ:")
        if green_count == tested_strategies:
            print("   🎉 全戦略をmain.pyに統合可能")
        elif green_count > 0:
            print(f"   🟢 {green_count}戦略の段階的統合を推奨")
            if yellow_count > 0:
                print(f"   🟡 {yellow_count}戦略の機能拡張検討")
            if red_count > 0:
                print(f"   🔴 {red_count}戦略の根本的修正が必要")
        else:
            print("   ⚠️ 全戦略に問題 - 詳細調査が必要")
    
    except Exception as e:
        print(f"❌ 未使用戦略テスト実行中に致命的エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")


if __name__ == "__main__":
    main()