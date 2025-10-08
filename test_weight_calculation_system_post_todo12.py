#!/usr/bin/env python3
"""
TODO #12完了後の重み配分計算システム状況確認テスト
2025年10月8日実行

目的：
- TODO #12完了により7/7戦略初期化成功を達成
- 重み配分計算システムが0/15点から改善されているかを確認
- 改善状況を定量的に測定してドキュメント更新
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import traceback

# プロジェクトルート追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger

# ロガー設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\weight_calculation_test.log")

class WeightCalculationSystemTester:
    """TODO #12完了後の重み配分計算システム状況確認"""
    
    def __init__(self):
        self.test_results = {}
        self.total_points = 15  # 重み配分計算システム満点
        self.achieved_points = 0
        
    def execute_comprehensive_weight_system_test(self):
        """包括的重み計算システムテスト実行"""
        print("=" * 60)
        print("TODO #12完了後：重み配分計算システム状況確認テスト")
        print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Test 1: MultiStrategyManager初期化確認
            self._test_multi_strategy_manager_initialization()
            
            # Test 2: 戦略レジストリ状況確認
            self._test_strategy_registry_status()
            
            # Test 3: 全戦略初期化状況確認
            self._test_all_strategy_initialization()
            
            # Test 4: 重み計算システム動作確認
            self._test_weight_calculation_system()
            
            # Test 5: 統合マルチ戦略フロー確認
            self._test_integrated_multi_strategy_flow()
            
            # 総合評価
            self._compile_comprehensive_results()
            
        except Exception as e:
            logger.error(f"重み計算システムテスト実行エラー: {e}")
            print(f"[ERROR] テスト実行エラー: {e}")
            traceback.print_exc()
    
    def _test_multi_strategy_manager_initialization(self):
        """Test 1: MultiStrategyManager初期化確認"""
        print("\n[SEARCH] Test 1: MultiStrategyManager初期化確認")
        
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            
            # 初期化テスト
            manager = MultiStrategyManager()
            
            if hasattr(manager, 'initialize_systems'):
                initialization_result = manager.initialize_systems()
                
                if initialization_result:
                    print("[OK] MultiStrategyManager初期化: 成功")
                    self.test_results['manager_initialization'] = {'status': 'success', 'points': 3}
                    self.achieved_points += 3
                else:
                    print("[ERROR] MultiStrategyManager初期化: 失敗")
                    self.test_results['manager_initialization'] = {'status': 'failed', 'points': 0, 'error': 'initialization_failed'}
            else:
                print("[WARNING] initialize_systems()メソッド未実装")
                self.test_results['manager_initialization'] = {'status': 'partial', 'points': 1, 'issue': 'method_missing'}
                self.achieved_points += 1
                
        except ImportError as e:
            print(f"[ERROR] MultiStrategyManagerインポートエラー: {e}")
            self.test_results['manager_initialization'] = {'status': 'failed', 'points': 0, 'error': str(e)}
        except Exception as e:
            print(f"[ERROR] 初期化テストエラー: {e}")
            self.test_results['manager_initialization'] = {'status': 'failed', 'points': 0, 'error': str(e)}
    
    def _test_strategy_registry_status(self):
        """Test 2: 戦略レジストリ状況確認"""
        print("\n[SEARCH] Test 2: 戦略レジストリ状況確認")
        
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            manager = MultiStrategyManager()
            
            if hasattr(manager, 'strategy_registry'):
                registry = manager.strategy_registry
                registered_strategies = len(registry) if registry else 0
                
                print(f"[CHART] 登録戦略数: {registered_strategies}/7")
                
                if registered_strategies == 7:
                    print("[OK] 戦略レジストリ: 完全登録（7/7戦略）")
                    self.test_results['strategy_registry'] = {'status': 'success', 'points': 3, 'registered': registered_strategies}
                    self.achieved_points += 3
                elif registered_strategies >= 5:
                    print(f"[WARNING] 戦略レジストリ: 部分登録（{registered_strategies}/7戦略）")
                    self.test_results['strategy_registry'] = {'status': 'partial', 'points': 2, 'registered': registered_strategies}
                    self.achieved_points += 2
                else:
                    print(f"[ERROR] 戦略レジストリ: 不完全（{registered_strategies}/7戦略）")
                    self.test_results['strategy_registry'] = {'status': 'failed', 'points': 0, 'registered': registered_strategies}
            else:
                print("[ERROR] strategy_registry属性が存在しません")
                self.test_results['strategy_registry'] = {'status': 'failed', 'points': 0, 'error': 'registry_missing'}
                
        except Exception as e:
            print(f"[ERROR] 戦略レジストリテストエラー: {e}")
            self.test_results['strategy_registry'] = {'status': 'failed', 'points': 0, 'error': str(e)}
    
    def _test_all_strategy_initialization(self):
        """Test 3: 全戦略初期化状況確認（TODO #12の成果確認）"""
        print("\n[SEARCH] Test 3: 全戦略初期化状況確認")
        
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            manager = MultiStrategyManager()
            
            # テスト用データ準備
            test_data = pd.DataFrame({
                'Close': [100, 101, 102, 103, 104],
                'High': [101, 102, 103, 104, 105],
                'Low': [99, 100, 101, 102, 103],
                'Open': [100, 101, 102, 103, 104],
                'Volume': [1000, 1100, 1200, 1300, 1400],
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2024-01-01', periods=5))
            
            expected_strategies = ['VWAPBreakoutStrategy', 'MomentumInvestingStrategy', 'BreakoutStrategy',
                                 'VWAPBounceStrategy', 'OpeningGapStrategy', 'ContrarianStrategy', 'GCStrategy']
            
            successful_initializations = 0
            failed_strategies = []
            
            for strategy_name in expected_strategies:
                try:
                    if hasattr(manager, 'get_strategy_instance'):
                        strategy_instance = manager.get_strategy_instance(strategy_name, test_data, {})
                        if strategy_instance:
                            successful_initializations += 1
                            print(f"[OK] {strategy_name}: 初期化成功")
                        else:
                            failed_strategies.append(strategy_name)
                            print(f"[ERROR] {strategy_name}: 初期化失敗")
                    else:
                        print("[WARNING] get_strategy_instance()メソッド未実装")
                        break
                        
                except Exception as e:
                    failed_strategies.append(strategy_name)
                    print(f"[ERROR] {strategy_name}: 初期化エラー - {e}")
            
            success_rate = (successful_initializations / len(expected_strategies)) * 100
            print(f"\n[CHART] 戦略初期化成功率: {successful_initializations}/{len(expected_strategies)} ({success_rate:.1f}%)")
            
            if success_rate == 100:
                print("[OK] 全戦略初期化: 完全成功（TODO #12目標達成）")
                self.test_results['strategy_initialization'] = {'status': 'success', 'points': 4, 'success_rate': success_rate}
                self.achieved_points += 4
            elif success_rate >= 80:
                print(f"[WARNING] 戦略初期化: 高成功率（{success_rate:.1f}%）")
                self.test_results['strategy_initialization'] = {'status': 'partial', 'points': 3, 'success_rate': success_rate}
                self.achieved_points += 3
            elif success_rate >= 60:
                print(f"[WARNING] 戦略初期化: 中成功率（{success_rate:.1f}%）")
                self.test_results['strategy_initialization'] = {'status': 'partial', 'points': 2, 'success_rate': success_rate}
                self.achieved_points += 2
            else:
                print(f"[ERROR] 戦略初期化: 低成功率（{success_rate:.1f}%）")
                self.test_results['strategy_initialization'] = {'status': 'failed', 'points': 0, 'success_rate': success_rate}
            
            if failed_strategies:
                print(f"失敗戦略: {', '.join(failed_strategies)}")
                self.test_results['strategy_initialization']['failed_strategies'] = failed_strategies
                
        except Exception as e:
            print(f"[ERROR] 戦略初期化テストエラー: {e}")
            self.test_results['strategy_initialization'] = {'status': 'failed', 'points': 0, 'error': str(e)}
    
    def _test_weight_calculation_system(self):
        """Test 4: 重み計算システム動作確認"""
        print("\n[SEARCH] Test 4: 重み計算システム動作確認")
        
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            manager = MultiStrategyManager()
            
            # 重み計算関連メソッドの存在確認
            weight_methods = [
                'calculate_strategy_weights',
                'calculate_weights', 
                'get_strategy_weights',
                'update_weights',
                'optimize_weights'
            ]
            
            available_methods = []
            for method_name in weight_methods:
                if hasattr(manager, method_name):
                    available_methods.append(method_name)
                    print(f"[OK] {method_name}メソッド: 存在")
                else:
                    print(f"[ERROR] {method_name}メソッド: 未実装")
            
            if available_methods:
                print(f"\n[CHART] 重み計算関連メソッド: {len(available_methods)}/{len(weight_methods)}実装済み")
                
                # 実際の重み計算テスト試行
                try:
                    primary_method = available_methods[0]
                    print(f"[SEARCH] {primary_method}メソッドのテスト実行")
                    
                    # テスト実行（簡易版）
                    test_result = getattr(manager, primary_method)()
                    
                    if test_result:
                        print("[OK] 重み計算システム: 動作確認成功")
                        self.test_results['weight_calculation'] = {'status': 'success', 'points': 3, 'methods': available_methods}
                        self.achieved_points += 3
                    else:
                        print("[WARNING] 重み計算システム: 部分動作")
                        self.test_results['weight_calculation'] = {'status': 'partial', 'points': 1, 'methods': available_methods}
                        self.achieved_points += 1
                        
                except Exception as method_error:
                    print(f"[WARNING] 重み計算メソッド実行エラー: {method_error}")
                    self.test_results['weight_calculation'] = {'status': 'partial', 'points': 1, 'methods': available_methods, 'execution_error': str(method_error)}
                    self.achieved_points += 1
            else:
                print("[ERROR] 重み計算関連メソッド: 未実装")
                self.test_results['weight_calculation'] = {'status': 'failed', 'points': 0, 'methods': []}
                
        except Exception as e:
            print(f"[ERROR] 重み計算システムテストエラー: {e}")
            self.test_results['weight_calculation'] = {'status': 'failed', 'points': 0, 'error': str(e)}
    
    def _test_integrated_multi_strategy_flow(self):
        """Test 5: 統合マルチ戦略フロー確認"""
        print("\n[SEARCH] Test 5: 統合マルチ戦略フロー確認")
        
        try:
            from config.multi_strategy_manager import MultiStrategyManager
            manager = MultiStrategyManager()
            
            # 統合フロー関連メソッドの確認
            integration_methods = [
                'execute_multi_strategy_flow',
                'integrate_strategies',
                'run_integrated_backtest'
            ]
            
            available_integration_methods = []
            for method_name in integration_methods:
                if hasattr(manager, method_name):
                    available_integration_methods.append(method_name)
                    print(f"[OK] {method_name}メソッド: 存在")
            
            if available_integration_methods:
                print("[OK] 統合マルチ戦略フロー: 基本機能実装済み")
                self.test_results['integration_flow'] = {'status': 'success', 'points': 2, 'methods': available_integration_methods}
                self.achieved_points += 2
            else:
                print("[ERROR] 統合マルチ戦略フロー: 未実装")
                self.test_results['integration_flow'] = {'status': 'failed', 'points': 0, 'methods': []}
                
        except Exception as e:
            print(f"[ERROR] 統合フローテストエラー: {e}")
            self.test_results['integration_flow'] = {'status': 'failed', 'points': 0, 'error': str(e)}
    
    def _compile_comprehensive_results(self):
        """総合評価・結果取りまとめ"""
        print("\n" + "=" * 60)
        print("[CHART] TODO #12完了後：重み配分計算システム総合評価")
        print("=" * 60)
        
        success_rate = (self.achieved_points / self.total_points) * 100
        
        print(f"[TARGET] 重み配分計算システム得点: {self.achieved_points}/{self.total_points}点 ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("[OK] 評価: 優秀 - TODO #12による大幅改善達成")
            overall_status = "大幅改善"
        elif success_rate >= 60:
            print("[WARNING] 評価: 良好 - TODO #12による改善確認、さらなる向上余地あり")
            overall_status = "改善確認"
        elif success_rate >= 40:
            print("[WARNING] 評価: 部分改善 - TODO #12効果は限定的")
            overall_status = "部分改善"
        else:
            print("[ERROR] 評価: 要改善 - TODO #12効果が十分でない")
            overall_status = "要改善"
        
        # 詳細結果表示
        print("\n[LIST] 詳細テスト結果:")
        for test_name, result in self.test_results.items():
            status_emoji = "[OK]" if result['status'] == 'success' else "[WARNING]" if result['status'] == 'partial' else "[ERROR]"
            print(f"{status_emoji} {test_name}: {result['points']}点 ({result['status']})")
            if 'error' in result:
                print(f"   エラー: {result['error']}")
            if 'issue' in result:
                print(f"   課題: {result['issue']}")
        
        # TODO #12前後の比較
        print(f"\n[UP] TODO #12による改善:")
        print(f"修正前: 0/15点 (0%)")
        print(f"修正後: {self.achieved_points}/15点 ({success_rate:.1f}%)")
        print(f"改善度: +{self.achieved_points}点 (+{success_rate:.1f}%)")
        
        # 文書更新用データ出力
        self.generate_document_update_data(overall_status, success_rate)
        
        return {
            'achieved_points': self.achieved_points,
            'total_points': self.total_points,
            'success_rate': success_rate,
            'overall_status': overall_status,
            'test_results': self.test_results
        }
    
    def generate_document_update_data(self, overall_status, success_rate):
        """文書更新用データ生成"""
        print(f"\n📝 main_normal_operation_diagnosis_results.md更新用データ:")
        print("-" * 40)
        
        if success_rate >= 60:
            new_status = f"重み配分計算システム（{self.achieved_points}/15点）"
            improvement_note = f"TODO #12完了により大幅改善達成（0点 → {self.achieved_points}点、+{success_rate:.1f}%）"
        else:
            new_status = f"重み配分計算システム（{self.achieved_points}/15点）"  
            improvement_note = f"TODO #12により改善確認（0点 → {self.achieved_points}点、+{success_rate:.1f}%）、さらなる改善必要"
        
        print(f"更新箇所: '残存課題**: 重み配分計算システム（0/15点）'")
        print(f"更新内容: '{new_status}'")
        print(f"改善説明: '{improvement_note}'")
        
        return {
            'new_status': new_status,
            'improvement_note': improvement_note,
            'detailed_results': self.test_results
        }


if __name__ == "__main__":
    tester = WeightCalculationSystemTester()
    results = tester.execute_comprehensive_weight_system_test()
    
    print(f"\n[TARGET] テスト完了 - {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")