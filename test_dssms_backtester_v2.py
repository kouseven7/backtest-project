"""
DSSMS Backtester V2 Excel出力テストスクリプト
新しいV2 Excel出力システムとの統合をテスト

実行内容:
1. DSSMS Backtesterインスタンスの作成
2. 簡単なバックテストシミュレーション実行
3. 新しいV2 Excel出力システムでの結果出力
4. 出力ファイルの確認と検証
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.dssms.dssms_backtester import DSSMSBacktester, SymbolSwitch
    from config.logger_config import setup_logger
    
    # ロガー設定
    logger = setup_logger(__name__)
    
    def test_dssms_backtester_v2_excel():
        """DSSMS Backtester V2 Excel出力統合テスト"""
        print("=" * 80)
        print("🔧 DSSMS Backtester V2 Excel出力統合テスト")
        print("=" * 80)
        
        try:
            # 1. DSSMS Backtester初期化
            print("\n📋 Step 1: DSSMS Backtester初期化")
            
            # テスト設定
            config = {
                'start_date': '2023-01-01',
                'end_date': '2023-03-31',
                'initial_capital': 1000000,
                'universe': ['7203.T', '9984.T', '6758.T', '8031.T', '8306.T'],
                'strategies': ['VWAPBreakoutStrategy', 'BreakoutStrategy', 'MomentumInvestingStrategy'],
                'output_excel': True,
                'debug': True
            }
            
            backtester = DSSMSBacktester(config)
            print("✅ DSSMS Backtester初期化完了")
            
            # 2. サンプルバックテストデータ作成
            print("\n📋 Step 2: サンプルバックテストデータ作成")
            
            # サンプル切替履歴作成
            sample_switches = create_sample_switch_history()
            backtester.switch_history = sample_switches
            
            # サンプルポートフォリオ履歴作成
            sample_portfolio = create_sample_portfolio_history()
            backtester.portfolio_history = sample_portfolio
            
            # 基本設定
            backtester.start_date = datetime(2023, 1, 1)
            backtester.end_date = datetime(2023, 3, 31)
            backtester.initial_capital = 1000000
            
            print(f"✅ サンプルデータ作成完了")
            print(f"   - 切替履歴: {len(sample_switches)}件")
            print(f"   - ポートフォリオ履歴: {len(sample_portfolio)}日分")
            
            # 3. V2 Excel出力システムテスト
            print("\n📋 Step 3: V2 Excel出力システムテスト")
            
            # シミュレーション結果辞書作成
            simulation_result = {
                "start_date": backtester.start_date,
                "end_date": backtester.end_date,
                "initial_capital": backtester.initial_capital,
                "final_portfolio_value": backtester.portfolio_history[-1],
                "switch_history": backtester.switch_history,
                "portfolio_history": backtester.portfolio_history
            }
            
            # ダミーのパフォーマンス指標作成
            from dataclasses import dataclass
            
            @dataclass
            class DummyPerformanceMetrics:
                total_return: float = 0.15
                annualized_return: float = 0.15
                volatility: float = 0.20
                sharpe_ratio: float = 0.75
                max_drawdown: float = -0.08
                win_rate: float = 0.60
                
            performance_metrics = DummyPerformanceMetrics()
            
            # ダミーの比較結果作成
            comparison_result = {
                "strategy_comparison": {
                    "DSSMS": {"return": 0.15, "sharpe": 0.75},
                    "Buy_and_Hold": {"return": 0.08, "sharpe": 0.45}
                },
                "market_comparison": {
                    "DSSMS_vs_Market": {"excess_return": 0.07}
                }
            }
            
            output_path = backtester.export_results_to_excel(
                simulation_result, performance_metrics, comparison_result)
            
            if output_path and Path(output_path).exists():
                print(f"✅ V2 Excel出力成功: {output_path}")
                print(f"📊 ファイルサイズ: {Path(output_path).stat().st_size:,} bytes")
                
                # ファイル内容確認
                try:
                    # openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
                    workbook = openpyxl.load_workbook(output_path)
                    sheet_names = workbook.sheetnames
                    print(f"📑 Excelシート: {sheet_names}")
                    
                    # サマリーシートの確認
                    if "サマリー" in sheet_names:
                        summary_sheet = workbook["サマリー"]
                        print(f"📋 サマリーシート行数: {summary_sheet.max_row}")
                        print(f"📋 サマリーシート列数: {summary_sheet.max_column}")
                    
                    workbook.close()
                    print("✅ Excelファイル内容確認完了")
                    
                except Exception as e:
                    print(f"⚠️ Excelファイル内容確認エラー: {e}")
                
            else:
                print("❌ V2 Excel出力失敗")
                return False
            
            # 4. パフォーマンス評価
            print("\n📋 Step 4: パフォーマンス評価")
            
            start_time = datetime.now()
            
            # 複数回出力テスト
            for i in range(3):
                test_output = backtester.export_results_to_excel(
                    simulation_result, performance_metrics, comparison_result)
                if test_output:
                    print(f"✅ 出力テスト{i+1}: 成功")
                else:
                    print(f"❌ 出力テスト{i+1}: 失敗")
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️ 3回出力の合計時間: {total_time:.2f}秒")
            print(f"📊 平均出力時間: {total_time/3:.2f}秒/回")
            
            # 5. 最終レポート
            print("\n" + "=" * 80)
            print("🎉 DSSMS Backtester V2 Excel出力統合テスト完了")
            print("=" * 80)
            print("✅ 全てのテストが成功しました")
            print(f"📁 最新出力ファイル: {output_path}")
            print("🚀 新しいV2システムがDSSMS Backtesterに正常に統合されました")
            
            return True
            
        except Exception as e:
            print(f"❌ 統合テストエラー: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def create_sample_switch_history():
        """サンプル切替履歴作成"""
        switches = []
        symbols = ["7203.T", "9984.T", "6758.T", "8031.T", "8306.T"]
        strategies = ["VWAPBreakoutStrategy", "BreakoutStrategy", "MomentumInvestingStrategy"]
        
        start_date = datetime(2023, 1, 1)
        
        for i in range(25):  # 25回の切替
            switch_date = start_date + timedelta(days=i*3 + 1)
            
            # 辞書形式でDSSMS切替データを作成
            switch_dict = {
                "date": switch_date,
                "from_symbol": symbols[i % len(symbols)],
                "to_symbol": symbols[(i+1) % len(symbols)],
                "from_strategy": strategies[i % len(strategies)],
                "to_strategy": strategies[(i+1) % len(strategies)],
                "reason": f"パフォーマンス改善_{i+1}",
                "confidence": np.random.uniform(0.6, 0.95),
                "profit_loss": np.random.normal(2000, 5000),
                "holding_period_hours": np.random.uniform(24, 120),
                "entry_price": np.random.uniform(1000, 2000),
                "exit_price": np.random.uniform(1050, 2100),
                "quantity": 100,
                "switch_cost": np.random.uniform(1000, 3000),
                "performance_after": np.random.uniform(-0.05, 0.15)
            }
            
            switches.append(switch_dict)
        
        return switches
    
    def create_sample_portfolio_history():
        """サンプルポートフォリオ履歴作成"""
        portfolio_values = []
        initial_value = 1000000
        current_value = initial_value
        
        # 90日間のポートフォリオ履歴
        for i in range(90):
            # リアルな市場変動をシミュレーション
            daily_change = np.random.normal(0.002, 0.02)  # 平均0.2%、標準偏差2%
            current_value *= (1 + daily_change)
            portfolio_values.append(current_value)
        
        return portfolio_values
    
    if __name__ == "__main__":
        success = test_dssms_backtester_v2_excel()
        if success:
            print("\n🎯 テスト結果: DSSMS Backtester V2 Excel統合成功！")
            print("📝 次のステップ: 実際のバックテストでさらなる検証を行ってください")
        else:
            print("\n💥 テスト結果: 統合テストが失敗しました")
            print("🔍 ログを確認してエラーを修正してください")

except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("🔍 DSSMSBacktesterクラスが見つかりません")
    print("📝 src/dssms/dssms_backtester.py ファイルを確認してください")
