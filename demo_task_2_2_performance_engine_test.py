"""
DSSMS Phase 2 Task 2.2: パフォーマンス計算エンジン修正 - 統合テストスイート
Comprehensive Test Suite for Performance Calculation Engine Fix

テスト目標:
1. 総リターン-100%問題の解決確認
2. ポートフォリオ価値0.01円問題の修正確認
3. 新旧システムの統合動作確認
4. 異常値検出・修正機能の確認
5. フォールバック機能の確認

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 2 Task 2.2 - パフォーマンス計算エンジン修正
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# Task 2.2 新システムのインポート
try:
    from src.dssms.dssms_performance_calculator_v2 import DSSMSPerformanceCalculatorV2, PerformanceStatus
    from src.dssms.portfolio_value_tracker import PortfolioValueTracker, TrackingConfiguration, ValueStatus
    from src.dssms.trade_result_analyzer import TradeResultAnalyzer, AnalysisLevel
    from src.dssms.performance_calculation_bridge import PerformanceCalculationBridge, IntegrationMode, IntegrationConfig
    TASK_2_2_AVAILABLE = True
except ImportError as e:
    TASK_2_2_AVAILABLE = False
    IMPORT_ERROR = str(e)

class TestScenario:
    """テストシナリオクラス"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.error_message = ""
        self.execution_time = 0.0
        self.details = {}

class PerformanceCalculationTestSuite:
    """パフォーマンス計算エンジン修正のテストスイート"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.test_results = []
        self.config_path = str(project_root / "config" / "performance_calculation_config.json")
        
    def run_all_tests(self) -> bool:
        """全テストの実行"""
        print("DSSMS Task 2.2: パフォーマンス計算エンジン修正 - 統合テスト")
        print("=" * 65)
        
        if not TASK_2_2_AVAILABLE:
            print(f"❌ Task 2.2 システムが利用できません: {IMPORT_ERROR}")
            return False
        
        # テストシナリオの定義
        test_scenarios = [
            ("test_v2_calculator_basic", "V2計算エンジン基本動作テスト"),
            ("test_anomaly_detection", "異常値検出機能テスト"),
            ("test_value_correction", "価値修正機能テスト"),
            ("test_portfolio_tracker", "ポートフォリオ価値追跡テスト"),
            ("test_trade_analyzer", "取引結果分析テスト"),
            ("test_integration_bridge", "統合ブリッジテスト"),
            ("test_critical_bug_fixes", "重要バグ修正確認テスト"),
            ("test_performance_scenarios", "パフォーマンスシナリオテスト"),
            ("test_fallback_mechanisms", "フォールバック機構テスト"),
            ("test_data_quality_validation", "データ品質検証テスト")
        ]
        
        print(f"\n🧪 実行予定テスト: {len(test_scenarios)}件")
        print("-" * 50)
        
        # 各テストの実行
        for test_method, description in test_scenarios:
            scenario = TestScenario(test_method, description)
            
            try:
                start_time = datetime.now()
                method = getattr(self, test_method)
                result = method()
                
                scenario.execution_time = (datetime.now() - start_time).total_seconds()
                scenario.passed = result
                
                if result:
                    print(f"✅ {description}: 成功 ({scenario.execution_time:.2f}s)")
                else:
                    print(f"❌ {description}: 失敗 ({scenario.execution_time:.2f}s)")
                    
            except Exception as e:
                scenario.execution_time = (datetime.now() - start_time).total_seconds()
                scenario.error_message = str(e)
                scenario.passed = False
                print(f"💥 {description}: エラー - {str(e)} ({scenario.execution_time:.2f}s)")
                self.logger.error(f"テスト{test_method}でエラー: {e}")
                self.logger.error(traceback.format_exc())
            
            self.test_results.append(scenario)
        
        # 結果サマリー
        self._print_test_summary()
        
        # すべてのテストが成功したかチェック
        all_passed = all(test.passed for test in self.test_results)
        return all_passed
    
    def test_v2_calculator_basic(self) -> bool:
        """V2計算エンジン基本動作テスト"""
        try:
            calculator = DSSMSPerformanceCalculatorV2(self.config_path)
            
            # 正常なデータでのテスト
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                'value': [1000000 + i * 5000 for i in range(30)]  # 順調な成長
            })
            
            result = calculator.calculate_comprehensive_performance(
                portfolio_data=sample_data,
                initial_capital=1000000
            )
            
            # 基本的な結果検証
            assert result.metrics.total_return > 0, "正のリターンが期待されます"
            assert result.metrics.portfolio_value > 1000000, "初期資本より高い価値が期待されます"
            assert result.metrics.calculation_status == PerformanceStatus.SUCCESS, "計算成功ステータスが期待されます"
            assert result.metrics.data_quality_score > 0.8, "高いデータ品質スコアが期待されます"
            
            self.logger.info(f"V2基本テスト結果: リターン {result.metrics.total_return:.2%}, 価値 ¥{result.metrics.portfolio_value:,.0f}")
            return True
            
        except Exception as e:
            self.logger.error(f"V2基本テストエラー: {e}")
            return False
    
    def test_anomaly_detection(self) -> bool:
        """異常値検出機能テスト"""
        try:
            calculator = DSSMSPerformanceCalculatorV2(self.config_path)
            
            # 異常値を含むデータ
            anomalous_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='D'),
                'value': [1000000, 1005000, 1010000, 0.01, 1015000, 1020000, 
                         1000000000, 1025000, 1030000, 1035000,  # 異常値を混入
                         1040000, 1045000, 1050000, 1055000, 1060000,
                         1065000, 1070000, 1075000, 1080000, 1085000]
            })
            
            result = calculator.calculate_comprehensive_performance(
                portfolio_data=anomalous_data,
                initial_capital=1000000
            )
            
            # 異常値が検出されることを確認
            assert len(result.metrics.anomalies_detected) > 0, "異常値が検出されるべきです"
            assert result.metrics.calculation_status in [PerformanceStatus.ANOMALY_DETECTED, PerformanceStatus.SUCCESS], "異常値検出ステータスが期待されます"
            assert result.metrics.data_quality_score < 1.0, "データ品質スコアが低下すべきです"
            
            self.logger.info(f"異常値検出テスト: {len(result.metrics.anomalies_detected)}件の異常値を検出")
            return True
            
        except Exception as e:
            self.logger.error(f"異常値検出テストエラー: {e}")
            return False
    
    def test_value_correction(self) -> bool:
        """価値修正機能テスト"""
        try:
            config = TrackingConfiguration(auto_correction_enabled=True, value_change_threshold=0.15)
            tracker = PortfolioValueTracker(config)
            
            # 修正が必要な異常値データ
            test_values = [
                {'portfolio_value': 1000000, 'timestamp': '2024-01-01'},
                {'portfolio_value': 1005000, 'timestamp': '2024-01-02'},
                {'portfolio_value': 0.01, 'timestamp': '2024-01-03'},  # 異常値
                {'portfolio_value': 1010000, 'timestamp': '2024-01-04'},
            ]
            
            corrected_count = 0
            for value_data in test_values:
                snapshot = tracker.update_value(value_data)
                if snapshot.status == ValueStatus.CORRECTION_APPLIED:
                    corrected_count += 1
            
            assert corrected_count > 0, "価値修正が実行されるべきです"
            
            # 最終的な価値が合理的範囲内にあることを確認
            final_snapshot = tracker.get_current_value()
            assert final_snapshot.portfolio_value > 100000, "修正後の価値が合理的範囲内にあるべきです"
            
            self.logger.info(f"価値修正テスト: {corrected_count}件の修正を実行")
            return True
            
        except Exception as e:
            self.logger.error(f"価値修正テストエラー: {e}")
            return False
    
    def test_portfolio_tracker(self) -> bool:
        """ポートフォリオ価値追跡テスト"""
        try:
            tracker = PortfolioValueTracker()
            
            # 時系列データの追跡
            test_data = [
                {'portfolio_value': 1000000, 'cash_value': 200000, 'position_value': 800000, 'timestamp': '2024-01-01'},
                {'portfolio_value': 1050000, 'cash_value': 200000, 'position_value': 850000, 'timestamp': '2024-01-02'},
                {'portfolio_value': 1100000, 'cash_value': 200000, 'position_value': 900000, 'timestamp': '2024-01-03'},
            ]
            
            for data in test_data:
                tracker.update_value(data)
            
            # 履歴データの確認
            history = tracker.get_value_history()
            assert len(history) == len(test_data), "すべてのデータが記録されるべきです"
            
            # 統計データの確認
            stats = tracker.get_value_statistics()
            assert stats['mean'] > 1000000, "平均価値が初期資本を上回るべきです"
            
            # DataFrame出力の確認
            df = tracker.export_to_dataframe()
            assert len(df) == len(test_data), "DataFrame出力が正しく動作すべきです"
            
            self.logger.info(f"ポートフォリオ追跡テスト: {len(history)}件のデータを追跡")
            return True
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ追跡テストエラー: {e}")
            return False
    
    def test_trade_analyzer(self) -> bool:
        """取引結果分析テスト"""
        try:
            analyzer = TradeResultAnalyzer(AnalysisLevel.COMPREHENSIVE)
            
            # サンプル取引データ
            trades_data = pd.DataFrame({
                'trade_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
                'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                'entry_price': [150, 2500, 300, 800, 3200],
                'exit_price': [155, 2400, 310, 850, 3100],
                'quantity': [100, 10, 50, 25, 20],
                'strategy': ['VWAP', 'RSI', 'MACD', 'VWAP', 'RSI'],
                'entry_time': pd.date_range(start='2024-01-01', periods=5, freq='D'),
                'exit_time': pd.date_range(start='2024-01-02', periods=5, freq='D'),
                'status': 'closed'
            })
            
            # P&Lの計算
            trades_data['pnl'] = (trades_data['exit_price'] - trades_data['entry_price']) * trades_data['quantity']
            
            # 取引データの追加
            added_count = analyzer.add_trades_from_dataframe(trades_data)
            assert added_count == len(trades_data), "すべての取引が追加されるべきです"
            
            # 統計の計算
            statistics = analyzer.calculate_comprehensive_statistics()
            assert statistics.total_trades == len(trades_data), "取引数が一致すべきです"
            assert 0 <= statistics.win_rate <= 1, "勝率が0-1の範囲内にあるべきです"
            
            # パフォーマンス内訳の生成
            breakdown = analyzer.generate_performance_breakdown()
            assert len(breakdown.by_strategy) > 0, "戦略別分析が生成されるべきです"
            
            self.logger.info(f"取引分析テスト: {statistics.total_trades}取引, 勝率 {statistics.win_rate:.2%}")
            return True
            
        except Exception as e:
            self.logger.error(f"取引分析テストエラー: {e}")
            return False
    
    def test_integration_bridge(self) -> bool:
        """統合ブリッジテスト"""
        try:
            config = IntegrationConfig(
                integration_mode=IntegrationMode.HYBRID,
                enable_cross_validation=True,
                enable_fallback=True
            )
            
            bridge = PerformanceCalculationBridge(config)
            
            # システムステータスの確認
            status = bridge.get_system_status()
            assert isinstance(status, dict), "システムステータスが正しく取得されるべきです"
            
            # サンプルデータでの統合計算
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='D'),
                'value': [1000000 + i * 2000 for i in range(20)]
            })
            
            result = bridge.calculate_performance(
                portfolio_data=sample_data,
                initial_capital=1000000
            )
            
            # 統合結果の確認
            assert result.primary_result is not None, "主要結果が生成されるべきです"
            assert result.integration_status != "error", "エラーステータスではないべきです"
            assert result.calculation_time_ms > 0, "実行時間が記録されるべきです"
            
            self.logger.info(f"統合ブリッジテスト: ステータス {result.integration_status}, 時間 {result.calculation_time_ms:.1f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"統合ブリッジテストエラー: {e}")
            return False
    
    def test_critical_bug_fixes(self) -> bool:
        """重要バグ修正確認テスト"""
        try:
            calculator = DSSMSPerformanceCalculatorV2(self.config_path)
            
            # Task 2.2の主要問題をテスト
            
            # 1. 総リターン-100%問題のテスト
            growth_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                'value': [1000000 + i * 10000 for i in range(10)]  # 10万円ずつ成長
            })
            
            result = calculator.calculate_comprehensive_performance(
                portfolio_data=growth_data,
                initial_capital=1000000
            )
            
            # -100%リターンではないことを確認
            assert result.metrics.total_return > -0.99, "総リターンが-100%ではないべきです"
            assert result.metrics.total_return > 0, "成長データに対して正のリターンが期待されます"
            
            # 2. ポートフォリオ価値0.01円問題のテスト
            assert result.metrics.portfolio_value > 1000000, "ポートフォリオ価値が初期資本より大きいべきです"
            assert result.metrics.portfolio_value > 1, "ポートフォリオ価値が0.01円のような異常値ではないべきです"
            
            # 3. 異常な最大ドローダウン100%問題のテスト
            assert 0 <= result.metrics.max_drawdown <= 1, "最大ドローダウンが合理的範囲内にあるべきです"
            assert result.metrics.max_drawdown < 0.5, "成長データに対して大きなドローダウンは期待されません"
            
            self.logger.info(f"バグ修正確認: リターン {result.metrics.total_return:.2%}, 価値 ¥{result.metrics.portfolio_value:,.0f}, DD {result.metrics.max_drawdown:.2%}")
            return True
            
        except Exception as e:
            self.logger.error(f"バグ修正確認テストエラー: {e}")
            return False
    
    def test_performance_scenarios(self) -> bool:
        """パフォーマンスシナリオテスト"""
        try:
            calculator = DSSMSPerformanceCalculatorV2(self.config_path)
            
            scenarios = [
                ("成長シナリオ", [1000000 + i * 5000 for i in range(20)]),
                ("下降シナリオ", [1000000 - i * 2000 for i in range(20)]),
                ("ボラティルシナリオ", [1000000 + ((-1)**i) * i * 3000 for i in range(20)]),
                ("フラットシナリオ", [1000000 + np.random.normal(0, 1000) for _ in range(20)])
            ]
            
            for scenario_name, values in scenarios:
                scenario_data = pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=len(values), freq='D'),
                    'value': values
                })
                
                result = calculator.calculate_comprehensive_performance(
                    portfolio_data=scenario_data,
                    initial_capital=1000000
                )
                
                # 各シナリオで合理的な結果が得られることを確認
                assert result.metrics.calculation_status in [PerformanceStatus.SUCCESS, PerformanceStatus.ANOMALY_DETECTED], f"{scenario_name}で計算が成功すべきです"
                assert not np.isnan(result.metrics.total_return), f"{scenario_name}でNaNリターンは期待されません"
                assert not np.isinf(result.metrics.total_return), f"{scenario_name}で無限リターンは期待されません"
                
                self.logger.info(f"{scenario_name}: リターン {result.metrics.total_return:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"パフォーマンスシナリオテストエラー: {e}")
            return False
    
    def test_fallback_mechanisms(self) -> bool:
        """フォールバック機構テスト"""
        try:
            config = IntegrationConfig(
                integration_mode=IntegrationMode.FALLBACK,
                enable_fallback=True
            )
            
            bridge = PerformanceCalculationBridge(config)
            
            # 通常データでのフォールバックテスト
            normal_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='D'),
                'value': [1000000 + i * 3000 for i in range(15)]
            })
            
            result = bridge.calculate_performance(
                portfolio_data=normal_data,
                initial_capital=1000000,
                force_mode=IntegrationMode.FALLBACK
            )
            
            # フォールバック機構が動作することを確認
            assert result.primary_result is not None, "フォールバック結果が生成されるべきです"
            assert "fallback" in result.integration_status, "フォールバックステータスが記録されるべきです"
            
            # 空データでのフォールバックテスト
            empty_data = pd.DataFrame()
            result_empty = bridge.calculate_performance(
                portfolio_data=empty_data,
                initial_capital=1000000,
                force_mode=IntegrationMode.FALLBACK
            )
            
            # 空データでもクラッシュしないことを確認
            assert result_empty is not None, "空データでも結果が返されるべきです"
            
            self.logger.info(f"フォールバックテスト: ステータス {result.integration_status}")
            return True
            
        except Exception as e:
            self.logger.error(f"フォールバック機構テストエラー: {e}")
            return False
    
    def test_data_quality_validation(self) -> bool:
        """データ品質検証テスト"""
        try:
            calculator = DSSMSPerformanceCalculatorV2(self.config_path)
            
            # 品質の異なるデータセット
            quality_tests = [
                ("高品質データ", pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                    'value': [1000000 + i * 1000 for i in range(30)]
                })),
                ("NaN含有データ", pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                    'value': [1000000 + i * 1000 if i % 5 != 0 else np.nan for i in range(30)]
                })),
                ("異常値含有データ", pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                    'value': [1000000 + i * 1000 if i != 15 else 0.01 for i in range(30)]
                }))
            ]
            
            for test_name, test_data in quality_tests:
                result = calculator.calculate_comprehensive_performance(
                    portfolio_data=test_data,
                    initial_capital=1000000
                )
                
                # データ品質スコアが適切に計算されることを確認
                assert 0 <= result.metrics.data_quality_score <= 1, f"{test_name}でデータ品質スコアが範囲内にあるべきです"
                
                # 品質レポートが生成されることを確認
                assert 'data_quality_score' in result.quality_report, f"{test_name}で品質レポートが生成されるべきです"
                
                self.logger.info(f"{test_name}: 品質スコア {result.metrics.data_quality_score:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"データ品質検証テストエラー: {e}")
            return False
    
    def _print_test_summary(self):
        """テストサマリーの表示"""
        print(f"\n" + "=" * 65)
        print("📋 テスト結果サマリー")
        print("=" * 65)
        
        passed_count = sum(1 for test in self.test_results if test.passed)
        total_count = len(self.test_results)
        
        print(f"総テスト数: {total_count}")
        print(f"成功: {passed_count}")
        print(f"失敗: {total_count - passed_count}")
        print(f"成功率: {passed_count / total_count * 100:.1f}%")
        
        # 失敗したテストの詳細
        failed_tests = [test for test in self.test_results if not test.passed]
        if failed_tests:
            print(f"\n❌ 失敗したテスト:")
            for test in failed_tests:
                print(f"  • {test.description}")
                if test.error_message:
                    print(f"    エラー: {test.error_message}")
        
        # 実行時間統計
        total_time = sum(test.execution_time for test in self.test_results)
        avg_time = total_time / len(self.test_results) if self.test_results else 0
        
        print(f"\n⏱️  実行時間統計:")
        print(f"  総実行時間: {total_time:.2f}秒")
        print(f"  平均実行時間: {avg_time:.2f}秒")
        
        # 最終判定
        if passed_count == total_count:
            print(f"\n🎉 すべてのテストが成功しました！")
            print(f"✅ Task 2.2 パフォーマンス計算エンジン修正: 完了")
        else:
            print(f"\n⚠️  {total_count - passed_count}件のテストが失敗しました")
            print(f"🔧 Task 2.2 パフォーマンス計算エンジン修正: 要調整")

def main():
    """メイン実行関数"""
    try:
        test_suite = PerformanceCalculationTestSuite()
        success = test_suite.run_all_tests()
        
        print(f"\n" + "=" * 65)
        if success:
            print("🏆 Task 2.2 パフォーマンス計算エンジン修正: 全テスト成功")
            print("📊 重要問題の修正が確認されました:")
            print("  ✅ 総リターン-100%問題: 修正済み")
            print("  ✅ ポートフォリオ価値0.01円問題: 修正済み")
            print("  ✅ 異常値検出・修正機能: 動作確認済み")
            print("  ✅ 統合システム: 正常動作確認済み")
        else:
            print("⚠️  Task 2.2 パフォーマンス計算エンジン修正: 一部テスト失敗")
            print("🔧 追加の調整が必要です")
        
        return success
        
    except Exception as e:
        print(f"\n💥 テストスイート実行エラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
